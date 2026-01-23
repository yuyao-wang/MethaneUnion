import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
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
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utils import load_config, parse_args

WINDOW_SIZE = 512
DEFAULT_BASE_DIR = '/data2/yuyao/methane_emission/carbonmapper_data_l89_l2sp'
MAX_L8_PER_PLUME = 3
PLUME_COMPLETION_MARKER = "landsat_l2sp_complete.txt"
MAX_CLOUD_COVER_PERCENT = 20.0
SEARCH_WINDOW_DAYS = 7
DEFAULT_RESOLUTION_METERS = 30
TIME_INTERVAL_PADDING = timedelta(hours=1)
CATALOG_MAX_RESULTS = 200

EVALSCRIPT_L8_STACK = """
//VERSION=3
function setup() {
  return {
    input: ["B01","B02","B03","B04","B05","B06","B07","B10"],
    output: {
      bands: 8,
      sampleType: "FLOAT32"
    }
  };
}
function evaluatePixel(sample) {
  return [
    sample.B01,
    sample.B02,
    sample.B03,
    sample.B04,
    sample.B05,
    sample.B06,
    sample.B07,
    sample.B10
  ];
}
"""

SENTINEL_CONFIG: Optional[SHConfig] = None
SENTINEL_CATALOG: Optional[SentinelHubCatalog] = None
LANDSAT_COLLECTION: DataCollection = DataCollection.LANDSAT_OT_L2
base_dir = DEFAULT_BASE_DIR


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def datetime_to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_product_id(value: str) -> str:
    if not value:
        return "unknown"
    return re.sub(r'[^0-9A-Za-z_\-]+', '_', value)


def update_global_progress(tracker):
    if tracker is None:
        return
    with tracker["lock"]:
        tracker["completed"] += 1
        completed = tracker["completed"]
        total = tracker["total"]
        elapsed = time.time() - tracker["start_time"]
        avg_time = elapsed / completed if completed > 0 else 0
        remaining = max(0, total - completed)
        eta = remaining * avg_time
        progress_bar = tracker.get("tqdm")
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix({"ETA(min)": f"{eta/60:.1f}"}, refresh=False)
        else:
            print(f"Completed {completed}/{total} | Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min")


def build_sentinelhub_config(config: Dict) -> SHConfig:
    sh_config = SHConfig()
    sh_config.sh_client_id = config.get('sentinelhub_client_id') or os.environ.get('SENTINELHUB_CLIENT_ID') or sh_config.sh_client_id
    sh_config.sh_client_secret = config.get('sentinelhub_client_secret') or os.environ.get('SENTINELHUB_CLIENT_SECRET') or sh_config.sh_client_secret
    sh_base_url = config.get('sentinelhub_base_url') or os.environ.get('SENTINELHUB_BASE_URL')
    if sh_base_url:
        sh_config.sh_base_url = sh_base_url
    if not sh_config.sh_client_id or not sh_config.sh_client_secret:
        raise RuntimeError(
            "Sentinel Hub credentials are missing. Set sentinelhub_client_id / sentinelhub_client_secret "
            "in the config or export SENTINELHUB_CLIENT_ID / SENTINELHUB_CLIENT_SECRET."
        )
    return sh_config


def resolve_landsat_collection(config: Dict) -> DataCollection:
    name = config.get('landsat_data_collection', 'LANDSAT_OT_L2')
    if isinstance(name, DataCollection):
        return name
    if not isinstance(name, str):
        return DataCollection.LANDSAT_OT_L2
    attr_name = name.upper()
    collection = getattr(DataCollection, attr_name, None)
    if collection is None:
        raise ValueError(f"Unsupported Sentinel Hub data collection: {name}")
    return collection


def fetch_products(plume_bounds: List[float], start_dt: datetime, end_dt: datetime) -> List[Dict]:
    if SENTINEL_CATALOG is None:
        raise RuntimeError("Sentinel Hub catalog has not been initialized.")
    bbox = BBox(bbox=plume_bounds, crs=CRS.WGS84)
    time_interval = (datetime_to_iso_z(start_dt), datetime_to_iso_z(end_dt))
    try:
        search_iterator = SENTINEL_CATALOG.search(
            LANDSAT_COLLECTION,
            bbox=bbox,
            time=time_interval,
            fields={
                "include": [
                    "id",
                    "properties.datetime",
                    "properties.eo:cloud_cover",
                    "properties.view:sun_azimuth",
                    "properties.view:sun_elevation",
                ],
                "exclude": ["properties.eo:bands"],
            },
        )
        products: List[Dict] = []
        for item in search_iterator:
            properties = item.get('properties', {})
            acq_time = parse_iso_datetime(properties.get('datetime'))
            if acq_time is None:
                continue
            cloud_cover = properties.get('eo:cloud_cover')
            if cloud_cover is not None and cloud_cover > MAX_CLOUD_COVER_PERCENT:
                continue
            products.append({
                'Id': item.get('id'),
                'acq_time': acq_time,
                'sun_azimuth': properties.get('view:sun_azimuth'),
                'sun_elevation': properties.get('view:sun_elevation'),
            })
            if len(products) >= CATALOG_MAX_RESULTS:
                break
        return products
    except Exception as exc:
        print(f"[error] Catalog search failed: {exc}")
        return []


def select_landsat_items(items, event_dt, max_scenes=3):
    if not items:
        return []
    same_day = []
    before = []
    after = []
    for item in items:
        t = item["acq_time"]
        if t.date() == event_dt.date():
            same_day.append(item)
        elif t < event_dt:
            before.append(item)
        else:
            after.append(item)
    selected = []
    if same_day:
        closest_same_day = min(same_day, key=lambda p: abs((p["acq_time"] - event_dt).total_seconds()))
        selected.append(closest_same_day)
        if before:
            closest_before = max(before, key=lambda p: p["acq_time"])
            selected.append(closest_before)
        if after:
            closest_after = min(after, key=lambda p: p["acq_time"])
            selected.append(closest_after)
    else:
        sorted_items = sorted(items, key=lambda p: abs((p["acq_time"] - event_dt).total_seconds()))
        selected = sorted_items[:max_scenes]
    return sorted(selected, key=lambda p: p["acq_time"])


def download_product(acquisition_time: datetime, plume_bounds: List[float], tif_output_path: str) -> Optional[Dict[str, int]]:
    if SENTINEL_CONFIG is None:
        raise RuntimeError("Sentinel Hub configuration is missing.")
    bbox = BBox(bbox=plume_bounds, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=DEFAULT_RESOLUTION_METERS)
    time_interval = (
        datetime_to_iso_z(acquisition_time - TIME_INTERVAL_PADDING),
        datetime_to_iso_z(acquisition_time + TIME_INTERVAL_PADDING),
    )
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_L8_STACK,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=LANDSAT_COLLECTION,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=SENTINEL_CONFIG,
    )
    try:
        data = request.get_data(save_data=False)
    except Exception as exc:
        print(f"[error] Sentinel Hub download failed: {exc}")
        return None
    if not data:
        return None
    image = data[0]  # H x W x bands
    stacked = np.transpose(image, (2, 0, 1))  # bands x H x W
    os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
    tifffile.imwrite(tif_output_path, stacked)
    return {"height": int(stacked.shape[1]), "width": int(stacked.shape[2])}


def download_task_l8(
    row_index,
    row_data,
    plume_bounds,
    progress_tracker,
    max_scenes=MAX_L8_PER_PLUME,
):
    plume_id = str(row_data.get('plume_id', 'unknown'))
    plume_dir = os.path.join(base_dir, plume_id)
    plume_marker_file = os.path.join(plume_dir, PLUME_COMPLETION_MARKER)

    try:
        if os.path.exists(plume_marker_file):
            print(f"[skip] plume {plume_id}; completion marker found at {plume_marker_file}")
            return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}

        event_dt = parse_iso_datetime(row_data.get('datetime'))
        if event_dt is None:
            print(f"[warn] plume {plume_id}: invalid datetime")
            return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}

        event_dt = event_dt.astimezone(timezone.utc)
        window_start = event_dt - timedelta(days=SEARCH_WINDOW_DAYS)
        window_end = event_dt + timedelta(days=SEARCH_WINDOW_DAYS)

        products = fetch_products(plume_bounds, window_start, window_end)
        if not products:
            print(f"[info] plume {plume_id}: no Landsat scenes in Sentinel Hub search window")
            return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}

        selected_items = select_landsat_items(products, event_dt, max_scenes=max_scenes)
        has_same_day = 1 if any(it["acq_time"].date() == event_dt.date() for it in selected_items) else 0

        os.makedirs(plume_dir, exist_ok=True)
        recorded_scenes = []

        for it in selected_items:
            acquisition_time = it.get("acq_time")
            product_id = safe_product_id(it.get("Id", ""))
            if acquisition_time is None or not product_id:
                continue
            tif_output_path = os.path.join(plume_dir, f'l8_{product_id}.tif')
            try:
                dims = download_product(acquisition_time, plume_bounds, tif_output_path)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[error] plume {plume_id}: download scene {product_id} failed: {exc}")
                continue
            if dims is None:
                continue
            recorded_scenes.append({
                "scene_id": it.get("Id", ""),
                "datetime": datetime_to_iso_z(acquisition_time),
                "tif_path": tif_output_path,
                "height": dims["height"],
                "width": dims["width"],
                "sun_azimuth": it.get("sun_azimuth", ""),
                "sun_elevation": it.get("sun_elevation", ""),
                "image_quality_oli": "",
                "image_quality_tirs": "",
            })
            time.sleep(0.1)

        if recorded_scenes:
            with open(plume_marker_file, 'w') as f:
                f.write(datetime_to_iso_z(datetime.now(timezone.utc)))

        return {
            'index': row_index,
            'selected_scenes': recorded_scenes,
            'has_same_day_l8': has_same_day,
        }

    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[error] Unknown error while processing plume {row_data.get('plume_id')}: {exc}")
        return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}
    finally:
        update_global_progress(progress_tracker)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    sentinel_config = build_sentinelhub_config(config)
    sentinel_catalog = SentinelHubCatalog(config=sentinel_config)
    SENTINEL_CONFIG = sentinel_config
    SENTINEL_CATALOG = sentinel_catalog
    LANDSAT_COLLECTION = resolve_landsat_collection(config)

    base_dir = config.get('local_base_dir_l89_l2sp', DEFAULT_BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)

    merged_csv_path = config.get('carbon_mapper_merged_csv', '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv')
    output_csv_path = config.get('carbon_mapper_l89_output_csv', '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_l8.csv')
    df = pd.read_csv(merged_csv_path)

    new_cols = []
    for i in range(1, MAX_L8_PER_PLUME + 1):
        new_cols.extend([
            f"l8_{i}_scene_id",
            f"l8_{i}_datetime",
            f"l8_{i}_tif",
            f"l8_{i}_height",
            f"l8_{i}_width",
            f"l8_{i}_sun_azimuth",
            f"l8_{i}_sun_elevation",
            f"l8_{i}_image_quality_oli",
            f"l8_{i}_image_quality_tirs",
        ])
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""

    plume_tif_mask = df["plume_tif"].apply(lambda v: isinstance(v, str) and len(v) > 0)
    processable_mask = plume_tif_mask

    total_processable = int(processable_mask.sum())
    overall_start_time = time.time()
    progress_tracker = None
    progress_bar = None
    if total_processable > 0:
        progress_bar = tqdm(total=total_processable, desc="L8/L9 plumes", dynamic_ncols=True)
        progress_tracker = {
            "lock": threading.Lock(),
            "completed": 0,
            "total": total_processable,
            "start_time": overall_start_time,
            "tqdm": progress_bar,
        }

    futures = []
    max_workers = config.get('max_workers', 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            if not processable_mask.iloc[index]:
                continue
            lat = row['plume_latitude']
            lon = row['plume_longitude']
            plume_bounds = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
            futures.append(executor.submit(
                download_task_l8,
                index,
                row.to_dict(),
                plume_bounds,
                progress_tracker
            ))

    results = [f.result() for f in futures]
    if progress_bar is not None:
        progress_bar.close()

    for res in results:
        if res is None:
            continue
        idx = res["index"]
        scenes = res.get("selected_scenes", [])
        for i in range(MAX_L8_PER_PLUME):
            prefix = f"l8_{i+1}_"
            if i < len(scenes):
                info = scenes[i]
                df.at[idx, prefix + "scene_id"] = info.get("scene_id", "")
                df.at[idx, prefix + "datetime"] = info.get("datetime", "")
                df.at[idx, prefix + "tif"] = info.get("tif_path", "")
                df.at[idx, prefix + "height"] = info.get("height", "")
                df.at[idx, prefix + "width"] = info.get("width", "")
                df.at[idx, prefix + "sun_azimuth"] = info.get("sun_azimuth", "")
                df.at[idx, prefix + "sun_elevation"] = info.get("sun_elevation", "")
                df.at[idx, prefix + "image_quality_oli"] = info.get("image_quality_oli", "")
                df.at[idx, prefix + "image_quality_tirs"] = info.get("image_quality_tirs", "")

    df.to_csv(output_csv_path, index=False)
    total_elapsed = time.time() - overall_start_time
    print(f"All tasks completed in {total_elapsed/60:.2f} minutes.")
    print("Output saved to:", output_csv_path)
