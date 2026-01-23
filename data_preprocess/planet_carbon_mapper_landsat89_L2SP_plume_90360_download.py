import ast
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.utils import load_config, parse_args

OFFSETS_DAYS = [90, 360]
SEARCH_TOLERANCE_DAYS = 30
MAX_CLOUD_COVER = 20.0
PLUME_COMPLETION_MARKER = "landsat_pc_offsets.json"
MAX_DOWNLOAD_WORKERS = 4
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = REPO_ROOT / "carbonmapper_data_l89_l2sp_90360"
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "carbon_mapper_data"
    / "csvs"
    / "merged_file_with_s2_l8_filtered_with_flags_low_cloud_only.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "carbon_mapper_data"
    / "csvs"
    / "merged_file_with_s2_l8_filtered_with_flags_low_cloud_only_with_l8_90360.csv"
)
PLUME_TIF_TIMESTAMP_PATTERN = re.compile(r"(\\d{4})(\\d{2})(\\d{2})[tT](\\d{2})(\\d{2})(\\d{2})")
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
scene_lock_map: Dict[str, threading.Lock] = {}
scene_lock_map_lock = threading.Lock()


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


def extract_datetime_from_plume_tif(plume_tif: Optional[str]) -> Optional[datetime]:
    if not isinstance(plume_tif, str) or "GAO" not in plume_tif:
        return None
    match = PLUME_TIF_TIMESTAMP_PATTERN.search(plume_tif)
    if not match:
        return None
    year, month, day, hour, minute, second = map(int, match.groups())
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_plume_bounds(row_data: Dict) -> List[float]:
    raw_bounds = row_data.get("plume_bounds")
    if isinstance(raw_bounds, str):
        try:
            parsed = ast.literal_eval(raw_bounds)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
                lon_min, lat_min, lon_max, lat_max = map(float, parsed)
                lon_min, lon_max = sorted([lon_min, lon_max])
                lat_min, lat_max = sorted([lat_min, lat_max])
                return [lon_min, lat_min, lon_max, lat_max]
        except Exception:
            pass
    lat = float(row_data.get("plume_latitude", 0.0))
    lon = float(row_data.get("plume_longitude", 0.0))
    delta = 0.01
    return [lon - delta, lat - delta, lon + delta, lat + delta]


def build_sentinelhub_config(config: Dict) -> SHConfig:
    sh_config = SHConfig()
    sh_config.sh_client_id = config.get('sentinelhub_client_id') or os.environ.get('SENTINELHUB_CLIENT_ID') or sh_config.sh_client_id
    sh_config.sh_client_secret = config.get('sentinelhub_client_secret') or os.environ.get('SENTINELHUB_CLIENT_SECRET') or sh_config.sh_client_secret
    sh_base_url = config.get('sentinelhub_base_url') or os.environ.get('SENTINELHUB_BASE_URL')
    if sh_base_url:
        sh_config.sh_base_url = sh_base_url
    if not sh_config.sh_client_id or not sh_config.sh_client_secret:
        raise RuntimeError(
            "Sentinel Hub credentials are missing. Provide sentinelhub_client_id / sentinelhub_client_secret "
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
            if cloud_cover is not None and cloud_cover > MAX_CLOUD_COVER:
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


def select_best_product(products: List[Dict], target_dt: datetime) -> Optional[Dict]:
    if not products:
        return None
    same_day = [prod for prod in products if prod['acq_time'].date() == target_dt.date()]
    candidates = same_day if same_day else products
    return min(candidates, key=lambda prod: abs((prod['acq_time'] - target_dt).total_seconds()))


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
    image = data[0]
    stacked = np.transpose(image, (2, 0, 1))
    os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
    tifffile.imwrite(tif_output_path, stacked)
    return {"height": int(stacked.shape[1]), "width": int(stacked.shape[2])}


def build_output_record(product: Dict, tif_path: str, dims: Dict[str, int]) -> Dict:
    return {
        "scene_id": product.get("Id", ""),
        "datetime": datetime_to_iso_z(product['acq_time']),
        "tif": tif_path,
        "sun_azimuth": product.get("sun_azimuth"),
        "sun_elevation": product.get("sun_elevation"),
        "image_quality_oli": "",
        "image_quality_tirs": "",
        "cloud_cover": "",
        "height": dims.get("height"),
        "width": dims.get("width"),
    }


def update_global_progress(tracker: Optional[Dict]):
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
        print(f"Progress {completed}/{total} | Elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min")


def download_task(
    row_index: int,
    row_data: Dict,
    base_event_dt: datetime,
    plume_bounds: List[float],
    processed_root: str,
    offsets: List[int],
    tolerance_days: int,
    progress_tracker: Optional[Dict],
) -> Dict:
    plume_id = str(row_data.get("plume_id", "unknown"))
    plume_dir = os.path.join(processed_root, plume_id)
    os.makedirs(plume_dir, exist_ok=True)
    marker_file = os.path.join(plume_dir, PLUME_COMPLETION_MARKER)
    new_records: Dict[int, Dict] = {}

    completed_offsets: Dict[int, Dict] = {}
    if os.path.exists(marker_file):
        try:
            payload = json.load(open(marker_file))
            completed_offsets = {int(entry["offset"]): entry for entry in payload.get("completed_offsets", [])}
        except Exception:
            completed_offsets = {}

    pending_offsets = [offset for offset in offsets if offset not in completed_offsets]
    if not pending_offsets:
        update_global_progress(progress_tracker)
        return {"index": row_index, "records": new_records}

    def process_offset(offset: int):
        target_dt = base_event_dt - timedelta(days=offset)
        window_start = target_dt - timedelta(days=tolerance_days)
        window_end = target_dt + timedelta(days=tolerance_days)
        products = fetch_products(plume_bounds, window_start, window_end)
        if not products:
            print(f"[info] plume {plume_id}: no Sentinel Hub Landsat scenes for offset {offset}")
            return offset, None
        selected = select_best_product(products, target_dt)
        if selected is None:
            return offset, None
        tif_path = os.path.join(plume_dir, f"l8_minus{offset}_{selected.get('Id', 'unknown')}.tif")
        dims = download_product(selected['acq_time'], plume_bounds, tif_path)
        if dims is None:
            return offset, None
        record = build_output_record(selected, tif_path, dims)
        return offset, record

    try:
        with ThreadPoolExecutor(max_workers=min(len(pending_offsets), MAX_DOWNLOAD_WORKERS)) as executor:
            futures = [executor.submit(process_offset, offset) for offset in pending_offsets]
            for future in futures:
                offset, record = future.result()
                if record is not None:
                    new_records[offset] = record
                    completed_offsets[offset] = {"offset": offset, "scene_id": record.get("scene_id"), "datetime": record.get("datetime")}

        if completed_offsets:
            payload = {
                "updated_at": datetime_to_iso_z(datetime.now(timezone.utc)),
                "completed_offsets": sorted(completed_offsets.values(), key=lambda entry: entry["offset"]),
            }
            with open(marker_file, "w") as handle:
                json.dump(payload, handle)

        return {"index": row_index, "records": new_records}
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[error] plume {plume_id}: unexpected failure {exc}")
        return {"index": row_index, "records": new_records}
    finally:
        update_global_progress(progress_tracker)


def main():
    args = parse_args()
    config = load_config(args.config)

    sentinel_config = build_sentinelhub_config(config)
    sentinel_catalog = SentinelHubCatalog(config=sentinel_config)
    global SENTINEL_CONFIG, SENTINEL_CATALOG, LANDSAT_COLLECTION
    SENTINEL_CONFIG = sentinel_config
    SENTINEL_CATALOG = sentinel_catalog
    LANDSAT_COLLECTION = resolve_landsat_collection(config)

    processed_root = str(config.get("l8_90360_processed_dir", DEFAULT_PROCESSED_DIR))
    input_csv = str(config.get("l8_90360_input_csv", DEFAULT_INPUT_CSV))
    output_csv = str(config.get("l8_90360_output_csv", DEFAULT_OUTPUT_CSV))

    os.makedirs(processed_root, exist_ok=True)

    df = pd.read_csv(input_csv)
    per_offset_fields = [
        "scene_id",
        "datetime",
        "tif",
        "sun_azimuth",
        "sun_elevation",
        "image_quality_oli",
        "image_quality_tirs",
        "cloud_cover",
        "height",
        "width",
    ]
    for offset in OFFSETS_DAYS:
        prefix = f"l8_minus{offset}"
        for field in per_offset_fields:
            col_name = f"{prefix}_{field}"
            if col_name not in df.columns:
                df[col_name] = ""

    plume_times = df["plume_tif"].apply(extract_datetime_from_plume_tif)
    fallback_times = df["datetime"].apply(parse_iso_datetime)
    base_times: Dict[int, Optional[datetime]] = {}
    for idx in df.index:
        base_time = plume_times.loc[idx]
        if base_time is None:
            base_time = fallback_times.loc[idx]
        if base_time is not None:
            base_time = base_time.astimezone(timezone.utc)
        base_times[idx] = base_time

    processable = [idx for idx, value in base_times.items() if value is not None]
    progress_tracker = None
    if processable:
        progress_tracker = {
            "lock": threading.Lock(),
            "completed": 0,
            "total": len(processable),
            "start_time": time.time(),
        }

    futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, row in df.iterrows():
            base_dt = base_times.get(index)
            if base_dt is None:
                continue
            row_dict = row.to_dict()
            plume_bounds = parse_plume_bounds(row_dict)
            futures.append(
                executor.submit(
                    download_task,
                    index,
                    row_dict,
                    base_dt,
                    plume_bounds,
                    processed_root,
                    OFFSETS_DAYS,
                    SEARCH_TOLERANCE_DAYS,
                    progress_tracker,
                )
            )

    results = [future.result() for future in futures]
    for result in results:
        idx = result.get("index")
        records = result.get("records", {})
        for offset, record in records.items():
            prefix = f"l8_minus{offset}"
            for field in per_offset_fields:
                col_name = f"{prefix}_{field}"
                df.at[idx, col_name] = record.get(field, "")

    df.to_csv(output_csv, index=False)
    print(f"All tasks completed. Output saved to {output_csv}")


if __name__ == "__main__":
    main()
