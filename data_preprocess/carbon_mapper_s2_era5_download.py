import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import cdsapi
import xarray as xr
from tqdm import tqdm
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.utils import parse_args, load_config

CSV_INPUT_PATH = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2.csv'
ERA5_OUTPUT_DIR = '/data2/yuyao/methane_emission/carbon_mapper_data/era5_downloads'
ERA5_RECORD_DIR = '/data2/yuyao/methane_emission/carbonmapper_data_era5_records'
CSV_OUTPUT_PATH = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2_era5.csv'
ERA5_STATE_PATH = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2_era5_state.csv'

ERA5_DATASET = 'reanalysis-era5-pressure-levels'
ERA5_VARIABLES = ['u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']
ERA5_PRESSURE_LEVELS = ['1000', '925', '850']

download_lock_map = {}
download_lock_map_lock = threading.Lock()
state_file_lock = threading.Lock()

def parse_iso_datetime(value):
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def round_to_hour(dt):
    """Round datetimes to the closest UTC hour supported by ERA5."""
    if dt is None:
        return None
    dt = dt.astimezone(timezone.utc)
    minute = dt.minute
    if minute >= 30:
        dt += timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def build_request(lat, lon, dt):
    area = [
        clamp(lat + 0.01, -90, 90),     # North
        clamp(lon - 0.01, -180, 180),   # West
        clamp(lat - 0.01, -90, 90),     # South
        clamp(lon + 0.01, -180, 180)    # East
    ]

    return {
        'product_type': ['reanalysis'],
        'variable': ERA5_VARIABLES,
        'year': [f'{dt.year:04d}'],
        'month': [f'{dt.month:02d}'],
        'day': [f'{dt.day:02d}'],
        'time': [f'{dt.hour:02d}:00'],
        'pressure_level': ERA5_PRESSURE_LEVELS,
        'area': area,
        'format': 'netcdf',
    }


def ensure_output_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def iso_to_filename(iso_str):
    dt = parse_iso_datetime(iso_str)
    if dt is None:
        safe = iso_str.replace(':', '').replace('-', '').replace('+', '').replace('Z', 'Z')
        return safe
    return dt.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def write_record_text(record):
    ensure_output_dir(ERA5_RECORD_DIR)
    plume_dir = os.path.join(ERA5_RECORD_DIR, str(record['plume_id']))
    ensure_output_dir(plume_dir)
    timestamp_str = iso_to_filename(record['era5_reference_time'])
    filename = f'era5_s2_{record["s2_slot"]}_{timestamp_str}.txt'
    txt_path = os.path.join(plume_dir, filename)
    try:
        with open(txt_path, 'w') as f:
            json.dump(record, f, indent=2)
    except Exception as exc:
        print(f'Failed to write ERA5 record to {txt_path}: {exc}')
        return None
    return txt_path


def get_download_lock(key):
    with download_lock_map_lock:
        lock = download_lock_map.get(key)
        if lock is None:
            lock = threading.Lock()
            download_lock_map[key] = lock
        return lock


def build_record_key(plume_id, slot, era5_time):
    if pd.isna(plume_id) or pd.isna(slot):
        return None
    if not isinstance(era5_time, str) or len(era5_time) == 0:
        return None
    try:
        slot = int(slot)
    except Exception:
        return None
    return (plume_id, slot, era5_time)


def load_records_from_txt():
    if not os.path.exists(ERA5_RECORD_DIR):
        return [], set()
    records = []
    keys = set()

    for plume_entry in os.scandir(ERA5_RECORD_DIR):
        if not plume_entry.is_dir():
            continue
        for record_entry in os.scandir(plume_entry.path):
            if not record_entry.name.endswith('.txt'):
                continue
            try:
                with open(record_entry.path, 'r') as f:
                    record = json.load(f)
            except Exception as exc:
                print(f'Failed to load ERA5 record {record_entry.path}: {exc}')
                continue
            key = build_record_key(
                record.get('plume_id'),
                record.get('s2_slot'),
                record.get('era5_reference_time')
            )
            if key is None:
                continue
            keys.add(key)
            records.append(record)
    return records, keys


def load_state_keys():
    if not os.path.exists(ERA5_STATE_PATH) or os.path.getsize(ERA5_STATE_PATH) == 0:
        return set()
    try:
        df = pd.read_csv(ERA5_STATE_PATH)
    except pd.errors.EmptyDataError:
        return set()
    keys = set()
    for _, row in df.iterrows():
        key = build_record_key(row.get('plume_id'), row.get('s2_slot'), row.get('era5_reference_time'))
        if key is None:
            continue
        keys.add(key)
    return keys


def append_state_key(key):
    plume_id, slot, era5_time = key
    os.makedirs(os.path.dirname(ERA5_STATE_PATH), exist_ok=True)
    with state_file_lock:
        file_exists = os.path.exists(ERA5_STATE_PATH)
        with open(ERA5_STATE_PATH, 'a') as f:
            if not file_exists:
                f.write('plume_id,s2_slot,era5_reference_time\n')
            f.write(f'{plume_id},{slot},{era5_time}\n')


def load_existing_records():
    if not os.path.exists(CSV_OUTPUT_PATH) or os.path.getsize(CSV_OUTPUT_PATH) == 0:
        return [], set()
    try:
        existing_df = pd.read_csv(CSV_OUTPUT_PATH)
    except pd.errors.EmptyDataError:
        return [], set()
    processed_keys = set()
    for _, row in existing_df.iterrows():
        key = build_record_key(row.get('plume_id'), row.get('s2_slot'), row.get('era5_reference_time'))
        if key is None:
            continue
        processed_keys.add(key)
    return existing_df.to_dict('records'), processed_keys


def prepare_tasks(df, processed_keys):
    aggregates = {}
    seen_keys = set(processed_keys)
    for _, row in df.iterrows():
        lat = row.get('plume_latitude')
        lon = row.get('plume_longitude')
        plume_id = row.get('plume_id', 'unknown')
        if pd.isna(lat) or pd.isna(lon):
            continue
        for idx in range(1, 4):
            dt_str = row.get(f's2_{idx}_datetime')
            if not isinstance(dt_str, str) or len(dt_str) == 0:
                continue
            parsed_dt = parse_iso_datetime(dt_str)
            rounded_dt = round_to_hour(parsed_dt)
            if rounded_dt is None:
                continue
            rounded_iso = rounded_dt.isoformat().replace('+00:00', 'Z')
            slot_key = (plume_id, idx, rounded_iso)
            if slot_key in seen_keys:
                continue
            seen_keys.add(slot_key)
            aggregate_key = (plume_id, rounded_iso)
            if aggregate_key not in aggregates:
                request = build_request(lat, lon, rounded_dt)
                timestamp_str = rounded_dt.strftime('%Y%m%dT%H%M%SZ')
                filename = f'{plume_id}_{timestamp_str}.nc'
                target_path = os.path.join(ERA5_OUTPUT_DIR, filename)
                aggregates[aggregate_key] = {
                    'plume_id': plume_id,
                    'rounded_dt': rounded_dt,
                    'rounded_iso': rounded_iso,
                    'request': request,
                    'target_path': target_path,
                    'slots': []
                }
            aggregates[aggregate_key]['slots'].append({
                'slot': idx,
                's2_datetime': dt_str
            })
    return list(aggregates.values())


def process_task(task):
    plume_id = task['plume_id']
    rounded_iso = task['rounded_iso']
    target_path = task['target_path']
    request = task['request']
    lock = get_download_lock((plume_id, rounded_iso))
    client = cdsapi.Client()
    summary = None
    with lock:
        try:
            if not os.path.exists(target_path):
                download_era5(client, request, target_path)
            else:
                print(f'Using existing ERA5 file {target_path}')
            summary = summarize_dataset(target_path)
        except Exception as exc:
            print(f'Failed to process ERA5 for {plume_id} {rounded_iso}: {exc}')
            summary = None
        finally:
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
            except Exception as cleanup_exc:
                print(f'Failed to remove ERA5 file {target_path}: {cleanup_exc}')
    if summary is None:
        return None
    records = []
    for slot_data in task['slots']:
        slot = slot_data['slot']
        record = {
            'plume_id': plume_id,
            's2_slot': slot,
            's2_datetime': slot_data['s2_datetime'],
            'era5_reference_time': rounded_iso,
            'era5_file': target_path,
        }
        record.update(summary)
        records.append(record)
    return records


def download_era5(client, request, target_path):
    for attempt in range(3):
        try:
            client.retrieve(ERA5_DATASET, request, target_path)
            return
        except Exception as exc:
            print(f'Failed to download ERA5 data (attempt {attempt + 1}): {exc}')
            if attempt == 2:
                raise
            time.sleep(5)


def summarize_dataset(dataset_path):
    var_config = [
        ('u', 'era5_u_mean'),
        ('v', 'era5_v_mean'),
        ('t', 'era5_temperature_mean'),
        ('q', 'era5_specific_humidity_mean')
    ]
    summary = {}
    with xr.open_dataset(dataset_path) as ds:
        for var_name, prefix in var_config:
            if var_name not in ds.variables:
                continue
            data = ds[var_name]
            if 'level' in data.dims:
                for level in ds['level'].values:
                    try:
                        level_val = int(level)
                    except Exception:
                        level_val = level
                    level_data = data.sel(level=level)
                    summary[f'{prefix}_{level_val}'] = float(level_data.mean().values)
            else:
                summary[prefix] = float(data.mean().values)
    return summary


def main():
    args = parse_args()
    config = load_config(args.config)
    del config

    ensure_output_dir(ERA5_OUTPUT_DIR)
    ensure_output_dir(ERA5_RECORD_DIR)

    df = pd.read_csv(CSV_INPUT_PATH, low_memory=False)
    existing_records, processed_keys = load_existing_records()
    txt_records, txt_keys = load_records_from_txt()
    record_map = {}
    for record in existing_records:
        key = build_record_key(record.get('plume_id'), record.get('s2_slot'), record.get('era5_reference_time'))
        if key is None:
            continue
        record_map[key] = record
    for record in txt_records:
        key = build_record_key(record.get('plume_id'), record.get('s2_slot'), record.get('era5_reference_time'))
        if key is None or key in record_map:
            continue
        existing_records.append(record)
        record_map[key] = record
    processed_keys.update(txt_keys)
    processed_keys.update(load_state_keys())
    tasks = prepare_tasks(df, processed_keys)

    if len(tasks) == 0:
        print('No new ERA5 downloads required.')
        if len(existing_records) > 0:
            pd.DataFrame(existing_records).to_csv(CSV_OUTPUT_PATH, index=False)
        return

    new_records = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc='ERA5 downloads'):
            result = future.result()
            if not result:
                continue
            for record in result:
                txt_path = write_record_text(record)
                if txt_path:
                    record['era5_txt_path'] = txt_path
                new_records.append(record)
                key = (record['plume_id'], record['s2_slot'], record['era5_reference_time'])
                append_state_key(key)
                processed_keys.add(key)

    if len(new_records) == 0 and len(existing_records) == 0:
        print('No ERA5 records were generated.')
        return

    combined_records = existing_records + new_records
    out_df = pd.DataFrame(combined_records)
    out_df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f'Saved {len(new_records)} new ERA5 summaries (total {len(combined_records)}) to {CSV_OUTPUT_PATH}')


if __name__ == '__main__':
    main()
