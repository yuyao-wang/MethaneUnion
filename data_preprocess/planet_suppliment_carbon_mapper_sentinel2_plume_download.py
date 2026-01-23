import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import tifffile
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    bbox_to_dimensions,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utils import load_config, parse_args

DEFAULT_PROCESSED_BASE_DIR = '/data2/yuyao/methane_emission/carbonmapper_data_s2_l2a'
DEFAULT_INPUT_MERGED_CSV = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2.csv'
DEFAULT_SUPPL_OUTPUT_CSV = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2_suppliment.csv'
DEFAULT_RESOLUTION_METERS = 10
CATALOG_MAX_RESULTS = 200
TIME_WINDOW_DAYS = 7
TIME_INTERVAL_PADDING = timedelta(hours=1)

EVALSCRIPT_TRUE_COLOR = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: {
      bands: 3,
      sampleType: "AUTO"
    }
  };
}

function evaluatePixel(sample) {
  return [
    2.5 * sample.B04,
    2.5 * sample.B03,
    2.5 * sample.B02
  ];
}
"""

SENTINEL_CONFIG: Optional[SHConfig] = None
SENTINEL_CATALOG: Optional[SentinelHubCatalog] = None
base_dir = DEFAULT_PROCESSED_BASE_DIR


def parse_iso_datetime(value):
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def datetime_to_iso_z(dt):
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def datetime_to_filename(dt):
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def extract_s2_datetime_columns(df: pd.DataFrame) -> List[Tuple[int, str]]:
    cols: List[Tuple[int, str]] = []
    for col in df.columns:
        match = re.match(r's2_(\d+)_datetime$', col)
        if match:
            cols.append((int(match.group(1)), col))
    cols.sort(key=lambda item: item[0])
    return cols


def gather_existing_s2_datetimes(row_data: pd.Series, datetime_cols: Sequence[Tuple[int, str]]) -> Set[str]:
    existing: Set[str] = set()
    for _, col in datetime_cols:
        value = row_data.get(col)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                existing.add(trimmed)
    return existing


def update_global_progress(tracker):
    if tracker is None:
        return
    with tracker['lock']:
        tracker['completed'] += 1
        completed = tracker['completed']
        total = tracker['total']
        elapsed = time.time() - tracker['start_time']
        avg_time = elapsed / completed if completed > 0 else 0
        remaining = max(0, total - completed)
        eta = remaining * avg_time
        print(f'Completed {completed} / {total} | Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min')


def build_sentinelhub_config(config: Dict) -> SHConfig:
    sh_config = SHConfig()
    sh_config.sh_client_id = config.get('sentinelhub_client_id') or os.environ.get('SENTINELHUB_CLIENT_ID') or sh_config.sh_client_id
    sh_config.sh_client_secret = config.get('sentinelhub_client_secret') or os.environ.get('SENTINELHUB_CLIENT_SECRET') or sh_config.sh_client_secret
    if not sh_config.sh_client_id or not sh_config.sh_client_secret:
        raise RuntimeError(
            "Sentinel Hub credentials are missing. Please provide sentinelhub_client_id and sentinelhub_client_secret "
            "in the config or set SENTINELHUB_CLIENT_ID / SENTINELHUB_CLIENT_SECRET environment variables."
        )
    return sh_config


def fetch_products(plume_bounds: List[float], start_dt: datetime, end_dt: datetime) -> List[Dict]:
    if SENTINEL_CATALOG is None:
        raise RuntimeError("Sentinel Hub catalog has not been initialized.")
    bbox = BBox(bbox=plume_bounds, crs=CRS.WGS84)
    time_interval = (datetime_to_iso_z(start_dt), datetime_to_iso_z(end_dt))
    search_iterator = SENTINEL_CATALOG.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=time_interval,
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": ["properties.eo:bands"]},
    )
    products: List[Dict] = []
    for item in search_iterator:
        properties = item.get('properties', {})
        acq_time = parse_iso_datetime(properties.get('datetime'))
        if acq_time is None:
            continue
        cloud_cover = properties.get('eo:cloud_cover')
        if cloud_cover is not None and cloud_cover > 20:
            continue
        products.append({
            'Id': item.get('id'),
            'Name': item.get('id'),
            'acq_time': acq_time
        })
        if len(products) >= CATALOG_MAX_RESULTS:
            break
    print(f'Fetched {len(products)} candidate Sentinel-2 acquisitions.')
    return products


def select_products(products, event_dt):
    if len(products) == 0:
        return []
    same_day = []
    before = []
    after = []
    for product in products:
        if product['acq_time'].date() == event_dt.date():
            same_day.append(product)
        elif product['acq_time'] < event_dt:
            before.append(product)
        else:
            after.append(product)
    selected = []
    if len(same_day) > 0:
        closest_same_day = min(same_day, key=lambda p: abs((p['acq_time'] - event_dt).total_seconds()))
        selected.append(closest_same_day)
        if len(before) > 0:
            closest_before = max(before, key=lambda p: p['acq_time'])
            selected.append(closest_before)
        if len(after) > 0:
            closest_after = min(after, key=lambda p: p['acq_time'])
            selected.append(closest_after)
    else:
        sorted_products = sorted(products, key=lambda p: abs((p['acq_time'] - event_dt).total_seconds()))
        selected = sorted_products[:3]
    return sorted(selected, key=lambda p: p['acq_time'])


def download_product(plume_id: str, acquisition_time: datetime, plume_bounds: List[float], tif_output_path: str):
    if SENTINEL_CONFIG is None:
        raise RuntimeError("Sentinel Hub configuration is missing.")
    bbox = BBox(bbox=plume_bounds, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=DEFAULT_RESOLUTION_METERS)
    time_interval = (
        datetime_to_iso_z(acquisition_time - TIME_INTERVAL_PADDING),
        datetime_to_iso_z(acquisition_time + TIME_INTERVAL_PADDING),
    )
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_TRUE_COLOR,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=SENTINEL_CONFIG,
    )
    data = request.get_data(save_data=False)
    if not data:
        raise RuntimeError("Sentinel Hub request returned no data.")
    image = data[0]
    os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
    tifffile.imwrite(tif_output_path, image)
    return image.shape[0], image.shape[1]


def download_task(row_index, row_data, event_dt, plume_bounds, progress_tracker, existing_datetimes: Set[str]):
    plume_id = str(row_data.get('plume_id', 'unknown'))
    plume_dir = os.path.join(base_dir, plume_id)
    os.makedirs(plume_dir, exist_ok=True)
    try:
        window_start = event_dt - timedelta(days=TIME_WINDOW_DAYS)
        window_end = event_dt + timedelta(days=TIME_WINDOW_DAYS)
        products = fetch_products(plume_bounds, window_start, window_end)
        selected_products = select_products(products, event_dt)
        candidate_iso_times = [datetime_to_iso_z(product['acq_time']) for product in selected_products]
        if candidate_iso_times:
            print(f'plume {plume_id} candidate cloud<20% acquisitions: {candidate_iso_times}')
        has_same_day = 1 if any(product['acq_time'].date() == event_dt.date() for product in selected_products) else 0
        recorded_products = []
        for product, acquisition_str in zip(selected_products, candidate_iso_times):
            if acquisition_str in existing_datetimes:
                print(f'plume {plume_id} already recorded acquisition {acquisition_str}; skipping download')
                continue
            tif_stamp = datetime_to_filename(product['acq_time'])
            tif_output_path = os.path.join(plume_dir, f's2_{tif_stamp}.tif')
            try:
                dims = download_product(plume_id, product['acq_time'], plume_bounds, tif_output_path)
            except Exception as exc:
                print(f"An error occurred while downloading {product['Name']}: {exc}")
                continue
            recorded_products.append({
                'datetime': acquisition_str,
                'path': tif_output_path,
                'height': int(dims[0]),
                'width': int(dims[1])
            })
            existing_datetimes.add(acquisition_str)
            time.sleep(0.1)
        return {'index': row_index, 'selected_products': recorded_products, 'has_same_day': has_same_day}
    except Exception as exc:
        print(f"An unknown error occurred while processing {row_data.get('plume_id')}: {exc}")
        return {'index': row_index, 'selected_products': [], 'has_same_day': 0}
    finally:
        update_global_progress(progress_tracker)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    sentinel_config = build_sentinelhub_config(config)
    sentinel_catalog = SentinelHubCatalog(config=sentinel_config)
    SENTINEL_CONFIG = sentinel_config
    SENTINEL_CATALOG = sentinel_catalog

    base_dir = config.get('local_base_dir_s2', DEFAULT_PROCESSED_BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)

    merged_csv_path = config.get('input_csv_path', DEFAULT_INPUT_MERGED_CSV)
    output_csv_path = config.get('output_csv_path', DEFAULT_SUPPL_OUTPUT_CSV)
    df = pd.read_csv(merged_csv_path)
    if 'has_same_day_s2' not in df.columns:
        df['has_same_day_s2'] = 0
    existing_datetime_cols = extract_s2_datetime_columns(df)
    max_existing_index = existing_datetime_cols[-1][0] if existing_datetime_cols else 0
    existing_datetimes_per_row: Dict[int, Set[str]] = {}
    for idx in df.index:
        row_series = df.loc[idx]
        existing_datetimes_per_row[idx] = gather_existing_s2_datetimes(row_series, existing_datetime_cols)
    parsed_times = df['datetime'].apply(parse_iso_datetime)
    plume_tif_mask = df['plume_tif'].apply(lambda v: isinstance(v, str) and len(v) > 0)
    valid_time_mask = parsed_times.notna()
    processable_mask = plume_tif_mask & valid_time_mask
    total_processable = int(processable_mask.sum())
    overall_start_time = time.time()
    progress_tracker = None
    if total_processable > 0:
        progress_tracker = {
            'lock': threading.Lock(),
            'completed': 0,
            'total': total_processable,
            'start_time': overall_start_time
        }
    futures = []
    with ThreadPoolExecutor(max_workers=config.get('max_workers', 8)) as executor:
        for index, row in df.iterrows():
            if not processable_mask.iloc[index]:
                continue
            parsed_time = parsed_times.iloc[index]
            parsed_time = parsed_time.astimezone(timezone.utc)
            lat = row['plume_latitude']
            lon = row['plume_longitude']
            plume_bounds = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
            futures.append(executor.submit(
                download_task,
                index,
                row.to_dict(),
                parsed_time,
                plume_bounds,
                progress_tracker,
                existing_datetimes_per_row.get(index, set()).copy()
            ))
    results = [future.result() for future in futures]
    max_new_records = 0
    for res in results:
        max_new_records = max(max_new_records, len(res.get('selected_products', [])))
    if max_new_records > 0:
        for offset in range(1, max_new_records + 1):
            col_idx = max_existing_index + offset
            for suffix in ('datetime', 'path', 'height', 'width'):
                col_name = f's2_{col_idx}_{suffix}'
                if col_name not in df.columns:
                    df[col_name] = ""
    for res in results:
        idx = res.get('index')
        selected_products = res.get('selected_products', [])
        has_same_day = res.get('has_same_day', 0)
        df.at[idx, 'has_same_day_s2'] = has_same_day
        for offset, product in enumerate(selected_products, start=1):
            col_idx = max_existing_index + offset
            df.at[idx, f's2_{col_idx}_datetime'] = product.get('datetime', '')
            df.at[idx, f's2_{col_idx}_path'] = product.get('path', '')
            df.at[idx, f's2_{col_idx}_height'] = product.get('height', '')
            df.at[idx, f's2_{col_idx}_width'] = product.get('width', '')
    df.to_csv(output_csv_path, index=False)
    total_elapsed = time.time() - overall_start_time
    print(f"All tasks completed in {total_elapsed/60:.2f} minutes.")
    print("All tasks completed.")
