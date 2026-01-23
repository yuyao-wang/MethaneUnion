import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import tifffile
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utils import load_config, parse_args

DEFAULT_LOCAL_BASE_DIR = '/data2/yuyao/methane_emission/carbonmapper_data_l2a_90360'
DEFAULT_DRIVE_ROOT = os.environ.get('GOOGLE_DRIVE_ROOT')
OFFSETS_DAYS = [90, 360]
SEARCH_WINDOW_DAYS = 50
PLUME_COMPLETION_MARKER = 'download_stub_pre.json'
DEFAULT_RESOLUTION_METERS = 10
CATALOG_MAX_RESULTS = 200
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
base_dir = DEFAULT_LOCAL_BASE_DIR


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def datetime_to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def datetime_to_filename(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def update_global_progress(tracker: Optional[Dict]):
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
        print(f'{completed}/{total} done; elapsed {elapsed/60:.2f}m; ETA {eta/60:.2f}m')


def build_sentinelhub_config(config: Dict) -> SHConfig:
    sh_config = SHConfig()
    sh_config.sh_client_id = config.get('sentinelhub_client_id') or os.environ.get('SENTINELHUB_CLIENT_ID') or sh_config.sh_client_id
    sh_config.sh_client_secret = config.get('sentinelhub_client_secret') or os.environ.get('SENTINELHUB_CLIENT_SECRET') or sh_config.sh_client_secret
    if not sh_config.sh_client_id or not sh_config.sh_client_secret:
        raise RuntimeError(
            "Sentinel Hub credentials are missing. Provide sentinelhub_client_id / sentinelhub_client_secret in the config "
            "or export SENTINELHUB_CLIENT_ID / SENTINELHUB_CLIENT_SECRET environment variables."
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


def select_closest_product(products: List[Dict], target_dt: datetime) -> Optional[Dict]:
    if len(products) == 0:
        return None
    same_day = [p for p in products if p['acq_time'].date() == target_dt.date()]
    if len(same_day) > 0:
        return min(same_day, key=lambda p: abs((p['acq_time'] - target_dt).total_seconds()))
    return min(products, key=lambda p: abs((p['acq_time'] - target_dt).total_seconds()))


def download_product(acquisition_time: datetime, plume_bounds: List[float], tif_output_path: str) -> Tuple[int, int]:
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


def load_completed_offsets(marker_file: str) -> Set[int]:
    if not os.path.exists(marker_file):
        return set()
    try:
        with open(marker_file, 'r') as f:
            payload = json.load(f)
        offsets = payload.get('completed_offsets', [])
        return set(int(v) for v in offsets)
    except Exception:
        return set()


def persist_completed_offsets(marker_file: str, offsets: Set[int]):
    os.makedirs(os.path.dirname(marker_file), exist_ok=True)
    with open(marker_file, 'w') as f:
        json.dump({
            'completed_offsets': sorted(offsets),
            'updated_at': datetime_to_iso_z(datetime.now(timezone.utc))
        }, f)


def download_task(row_index: int, row_data: Dict, base_event_dt: datetime,
                  plume_bounds: List[float], offsets: List[int], search_window_days: int,
                  progress_tracker: Optional[Dict]):
    plume_id = str(row_data.get('plume_id', 'unknown'))
    plume_dir = os.path.join(base_dir, plume_id)
    os.makedirs(plume_dir, exist_ok=True)
    marker_file = os.path.join(plume_dir, PLUME_COMPLETION_MARKER)
    completed_offsets = load_completed_offsets(marker_file)
    pending_offsets = [offset for offset in offsets if offset not in completed_offsets]
    if len(pending_offsets) == 0:
        update_global_progress(progress_tracker)
        return {'index': row_index, 'records': {}}

    new_records: Dict[int, Dict] = {}
    updated_offsets = set(completed_offsets)
    try:
        for offset in pending_offsets:
            target_dt = base_event_dt - timedelta(days=offset)
            window_start = target_dt - timedelta(days=search_window_days)
            window_end = target_dt
            products = fetch_products(plume_bounds, window_start, window_end)
            selected_product = select_closest_product(products, target_dt)
            if selected_product is None:
                print(f'No S2 for plume {plume_id} offset {offset}')
                continue

            acquisition_str = datetime_to_iso_z(selected_product['acq_time'])
            tif_stamp = datetime_to_filename(selected_product['acq_time'])
            tif_output_path = os.path.join(plume_dir, f's2_minus{offset}_{tif_stamp}.tif')
            try:
                dims = download_product(selected_product['acq_time'], plume_bounds, tif_output_path)
            except Exception as exc:
                print(f"Download error for plume {plume_id} offset {offset}: {exc}")
                continue
            new_records[offset] = {
                'datetime': acquisition_str,
                'path': tif_output_path,
                'height': int(dims[0]),
                'width': int(dims[1])
            }
            updated_offsets.add(offset)
            time.sleep(0.1)
        if len(updated_offsets) > len(completed_offsets):
            persist_completed_offsets(marker_file, updated_offsets)
        return {'index': row_index, 'records': new_records}
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"Plume {row_data.get('plume_id')} failed: {exc}")
        return {'index': row_index, 'records': new_records}
    finally:
        update_global_progress(progress_tracker)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    sentinel_config = build_sentinelhub_config(config)
    sentinel_catalog = SentinelHubCatalog(config=sentinel_config)
    SENTINEL_CONFIG = sentinel_config
    SENTINEL_CATALOG = sentinel_catalog

    drive_root = config.get('google_drive_dir', DEFAULT_DRIVE_ROOT)
    if drive_root:
        base_dir = os.path.join(drive_root, 'carbonmapper_data_l2a_90360')
    else:
        base_dir = config.get('local_base_dir', DEFAULT_LOCAL_BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)

    merged_csv_path = config.get('carbon_mapper_merged_csv', '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv')
    output_csv_path = config.get('carbon_mapper_90360_output_csv', '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2_90360.csv')
    df = pd.read_csv(merged_csv_path)

    for offset in OFFSETS_DAYS:
        prefix = f's2_minus{offset}'
        for suffix in ('datetime', 'path', 'height', 'width'):
            col_name = f'{prefix}_{suffix}'
            if col_name not in df.columns:
                df[col_name] = ""

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
    max_workers = config.get('max_workers', 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            if not processable_mask.iloc[index]:
                continue
            base_event_dt = parsed_times.iloc[index]
            if base_event_dt is None:
                continue
            base_event_dt = base_event_dt.astimezone(timezone.utc)
            lat = row['plume_latitude']
            lon = row['plume_longitude']
            plume_bounds = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
            futures.append(executor.submit(
                download_task,
                index,
                row.to_dict(),
                base_event_dt,
                plume_bounds,
                OFFSETS_DAYS,
                SEARCH_WINDOW_DAYS,
                progress_tracker
            ))

    results = [future.result() for future in futures]
    for result in results:
        idx = result.get('index')
        if idx is None:
            continue
        records = result.get('records', {})
        for offset, record in records.items():
            prefix = f's2_minus{offset}'
            df.at[idx, f'{prefix}_datetime'] = record.get('datetime', '')
            df.at[idx, f'{prefix}_path'] = record.get('path', '')
            df.at[idx, f'{prefix}_height'] = record.get('height', '')
            df.at[idx, f'{prefix}_width'] = record.get('width', '')

    df.to_csv(output_csv_path, index=False)
    total_elapsed = time.time() - overall_start_time
    print(f"All tasks completed in {total_elapsed/60:.2f} minutes.")
    print("All tasks completed.")
