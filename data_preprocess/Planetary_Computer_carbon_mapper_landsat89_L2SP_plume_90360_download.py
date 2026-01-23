import ast
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
import shutil

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from planetary_computer import sign
from pystac import Item
from pystac_client import Client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util.utils import load_config, parse_args  # noqa: E402
from landsat_c2_downloader import normalize_scene_id  # noqa: E402
import data_preprocess.carbon_mapper_landsat89_L2SP_plume_download as l8_processing  # noqa: E402


PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"
L8_PLATFORMS = ["landsat-8", "landsat-9"]
OFFSETS_DAYS = [90, 360]
SEARCH_TOLERANCE_DAYS = 30
MAX_CLOUD_COVER = 20.0
MAX_SEARCH_ITEMS = 200
SCENE_DOWNLOAD_MARKER = ".planetary_pc_download_complete"
PLUME_COMPLETION_MARKER = "landsat_pc_offsets.json"
MAX_ASSET_DOWNLOAD_WORKERS = 6
MAX_SCENE_DOWNLOAD_WORKERS = 4
CHUNK_SIZE_BYTES = 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = REPO_ROOT / "carbonmapper_data_l89_l2sp_90360"
DEFAULT_RAW_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_L89_L2SP_90360"
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

ASSET_KEY_SUFFIXES = [
    ("coastal", "SR_B1.TIF"),
    ("blue", "SR_B2.TIF"),
    ("green", "SR_B3.TIF"),
    ("red", "SR_B4.TIF"),
    ("nir08", "SR_B5.TIF"),
    ("swir16", "SR_B6.TIF"),
    ("swir22", "SR_B7.TIF"),
    ("lwir11", "ST_B10.TIF"),
    ("mtl.txt", "MTL.txt"),
]

PLUME_TIF_TIMESTAMP_PATTERN = re.compile(
    r"(\\d{4})(\\d{2})(\\d{2})[tT](\\d{2})(\\d{2})(\\d{2})"
)

scene_lock_map: Dict[str, threading.Lock] = {}
scene_lock_map_lock = threading.Lock()
existing_records_by_offset: Dict[int, List[Dict]] = {offset: [] for offset in OFFSETS_DAYS}
existing_record_keys: Set[Tuple[int, str]] = set()
existing_records_lock = threading.Lock()


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


def sanitize_row_value(value):
    if isinstance(value, str):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def add_record_to_cache(offset: int, record: Dict):
    if offset not in existing_records_by_offset:
        existing_records_by_offset[offset] = []
    tif_path = record.get("tif")
    if not isinstance(tif_path, str) or len(tif_path) == 0:
        return
    dt = l8_processing.parse_iso_datetime(record.get("datetime"))
    if dt is None:
        return
    key = (offset, os.path.abspath(tif_path))
    with existing_records_lock:
        if key in existing_record_keys:
            return
        entry = dict(record)
        entry["_datetime"] = dt
        existing_records_by_offset.setdefault(offset, []).append(entry)
        existing_record_keys.add(key)


def find_cached_record(offset: int, target_dt: datetime, tolerance_days: int) -> Optional[Dict]:
    tolerance = timedelta(days=tolerance_days)
    with existing_records_lock:
        candidates = list(existing_records_by_offset.get(offset, []))
    best = None
    best_delta = None
    for candidate in candidates:
        dt = candidate.get("_datetime")
        if not isinstance(dt, datetime):
            continue
        delta = abs((dt - target_dt))
        if delta <= tolerance:
            if best is None or delta < best_delta:
                best = candidate
                best_delta = delta
    return best


def clone_cached_record(candidate: Dict, plume_dir: str, offset: int) -> Optional[Dict]:
    src_path = candidate.get("tif")
    if not isinstance(src_path, str) or not os.path.exists(src_path):
        return None
    scene_id = candidate.get("scene_id", "unknown")
    dest_name = f"l8_minus{offset}_{scene_id}.tif"
    dest_path = os.path.join(plume_dir, dest_name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.abspath(src_path) != os.path.abspath(dest_path):
        shutil.copy2(src_path, dest_path)
    new_record = {k: v for k, v in candidate.items() if k != "_datetime"}
    new_record["tif"] = dest_path
    return new_record


def seed_existing_records_from_dataframe(df: pd.DataFrame, per_offset_fields: List[str]):
    for offset in OFFSETS_DAYS:
        prefix = f"l8_minus{offset}"
        tif_col = f"{prefix}_tif"
        datetime_col = f"{prefix}_datetime"
        if tif_col not in df.columns or datetime_col not in df.columns:
            continue
        for _, row in df.iterrows():
            tif_path = sanitize_row_value(row.get(tif_col))
            if not isinstance(tif_path, str) or len(tif_path) == 0:
                continue
            if not os.path.exists(tif_path):
                continue
            record = {}
            missing_required = False
            for field in per_offset_fields:
                col_name = f"{prefix}_{field}"
                if col_name not in df.columns:
                    continue
                value = sanitize_row_value(row.get(col_name))
                if field == "datetime" and value is None:
                    missing_required = True
                    break
                record[field] = value
            if missing_required:
                continue
            add_record_to_cache(offset, record)


def apply_previous_results(df: pd.DataFrame, prev_df: Optional[pd.DataFrame], per_offset_fields: List[str]) -> pd.DataFrame:
    if prev_df is None or "plume_id" not in prev_df.columns:
        return df
    usable_prev = prev_df.drop_duplicates(subset=["plume_id"], keep="last").set_index("plume_id")
    for idx in df.index:
        plume_id = df.at[idx, "plume_id"]
        if plume_id not in usable_prev.index:
            continue
        prev_row = usable_prev.loc[plume_id]
        for offset in OFFSETS_DAYS:
            prefix = f"l8_minus{offset}"
            for field in per_offset_fields:
                col_name = f"{prefix}_{field}"
                if col_name not in df.columns or col_name not in prev_row.index:
                    continue
                value = prev_row[col_name]
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                df.at[idx, col_name] = value
    return df
def load_completed_offsets(marker_file: str) -> Dict[int, Dict]:
    if not os.path.exists(marker_file):
        return {}
    try:
        with open(marker_file, "r") as handle:
            payload = json.load(handle)
        return {
            int(entry["offset"]): entry
            for entry in payload.get("completed_offsets", [])
            if "offset" in entry
        }
    except Exception:
        return {}


def persist_completed_offsets(marker_file: str, records: Dict[int, Dict]):
    os.makedirs(os.path.dirname(marker_file), exist_ok=True)
    payload = {
        "updated_at": l8_processing.datetime_to_iso_z(datetime.now(timezone.utc)),
        "completed_offsets": list(
            sorted(
                (
                    {"offset": offset, **{k: v for k, v in info.items() if k != "record"}}
                    for offset, info in records.items()
                ),
                key=lambda x: x["offset"],
            )
        ),
    }
    with open(marker_file, "w") as handle:
        json.dump(payload, handle)


def update_global_progress(tracker: Optional[Dict]):
    if tracker is None:
        return
    with tracker["lock"]:
        tracker["completed"] += 1
        completed = tracker["completed"]
        total = tracker["total"]
        elapsed = time.time() - tracker["start_time"]
        avg_time = elapsed / completed if completed else 0
        remaining = max(0, total - completed)
        eta = remaining * avg_time
        print(
            f"Progress {completed}/{total} "
            f"| Elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min"
        )


def get_scene_lock(scene_id: str) -> threading.Lock:
    with scene_lock_map_lock:
        lock = scene_lock_map.get(scene_id)
        if lock is None:
            lock = threading.Lock()
            scene_lock_map[scene_id] = lock
        return lock


def create_http_session(max_pool: int = MAX_ASSET_DOWNLOAD_WORKERS) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=max_pool, pool_maxsize=max_pool)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_asset(item: Item, asset_key: str, dest_path: str, session: requests.Session):
    asset = item.assets.get(asset_key)
    if asset is None:
        raise RuntimeError(f"Asset '{asset_key}' not found for item {item.id}")
    signed_href = sign(asset.href)
    tmp_path = dest_path + ".partial"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        with session.get(signed_href, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(CHUNK_SIZE_BYTES):
                    if chunk:
                        handle.write(chunk)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        raise
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def get_normalized_scene_id(item: Item) -> str:
    raw_id = item.properties.get("landsat:scene_id") or item.id or ""
    if not raw_id:
        return ""
    return normalize_scene_id(raw_id)


def is_l89_scene_id(scene_id: str) -> bool:
    if not isinstance(scene_id, str) or len(scene_id) < 3:
        return False
    prefixes = ("LC08", "LC09", "LC8", "LC9")
    return any(scene_id.startswith(prefix) for prefix in prefixes)


def is_l89_item(item: Item) -> bool:
    scene_id = get_normalized_scene_id(item)
    if not scene_id:
        return False
    platform = (item.properties or {}).get("platform")
    platform_ok = platform in L8_PLATFORMS if platform else True
    return platform_ok and is_l89_scene_id(scene_id)


def ensure_scene_assets(item: Item, raw_root: str) -> str:
    scene_id = get_normalized_scene_id(item) or "unknown"
    scene_dir = os.path.join(raw_root, scene_id)
    lock = get_scene_lock(scene_id)
    with lock:
        os.makedirs(scene_dir, exist_ok=True)
        marker = os.path.join(scene_dir, SCENE_DOWNLOAD_MARKER)
        required = []
        for asset_key, suffix in ASSET_KEY_SUFFIXES:
            target_name = f"{scene_id}_{suffix}"
            required.append((asset_key, os.path.join(scene_dir, target_name)))

        missing = [asset for asset in required if not os.path.exists(asset[1])]
        if not missing and os.path.exists(marker):
            return scene_dir

        if missing:
            session = create_http_session()
            try:
                max_workers = min(len(missing), MAX_ASSET_DOWNLOAD_WORKERS)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(download_asset, item, asset_key, dest, session)
                        for asset_key, dest in missing
                    ]
                    for future in futures:
                        try:
                            future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f"Failed to download asset for {scene_id}: {exc}"
                            ) from exc
            finally:
                session.close()

        with open(marker, "w") as handle:
            handle.write(l8_processing.datetime_to_iso_z(datetime.now(timezone.utc)))
    return scene_dir


def parse_cloud_cover(item: Item) -> Optional[float]:
    props = item.properties or {}
    value = props.get("eo:cloud_cover")
    if value is None:
        value = props.get("landsat:cloud_cover_land")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def search_landsat_items(
    client: Client,
    bbox: List[float],
    start_dt: datetime,
    end_dt: datetime,
) -> List[Item]:
    datetime_window = (
        f"{l8_processing.datetime_to_iso_z(start_dt)}/"
        f"{l8_processing.datetime_to_iso_z(end_dt)}"
    )
    search = client.search(
        collections=[LANDSAT_COLLECTION],
        bbox=bbox,
        datetime=datetime_window,
        max_items=MAX_SEARCH_ITEMS,
        limit=MAX_SEARCH_ITEMS,
        query={"platform": {"in": L8_PLATFORMS}},
    )
    return list(search.items())


def select_best_item(items: List[Item], target_dt: datetime) -> Optional[Item]:
    if not items:
        return None

    def valid(item: Item) -> bool:
        if not is_l89_item(item):
            return False
        cc = parse_cloud_cover(item)
        return cc is not None and cc <= MAX_CLOUD_COVER and item.datetime is not None

    filtered = [item for item in items if valid(item)]
    if not filtered:
        return None

    same_day = [
        item
        for item in filtered
        if item.datetime and item.datetime.date() == target_dt.date()
    ]

    candidate_pool = same_day if same_day else filtered
    return min(
        candidate_pool,
        key=lambda it: abs((it.datetime - target_dt).total_seconds()),
    )


def build_output_record(
    item: Item,
    scene_dir: str,
    plume_dir: str,
    plume_bounds: List[float],
    offset: int,
) -> Optional[Dict]:
    scene_id = get_normalized_scene_id(item) or "unknown"
    out_tif_name = f"l8_minus{offset}_{scene_id}.tif"
    out_tif_path = os.path.join(plume_dir, out_tif_name)

    dims = l8_processing.build_landsat_stack_for_plume(
        scene_dir, scene_id, plume_bounds, out_tif_path
    )
    if dims is None:
        return None

    mtl_path = os.path.join(scene_dir, f"{scene_id}_MTL.txt")
    meta = l8_processing.parse_landsat_mtl(mtl_path)
    acq_dt_iso = meta.get("acq_datetime_iso")
    if not acq_dt_iso and item.datetime:
        acq_dt_iso = l8_processing.datetime_to_iso_z(item.datetime)

    cloud_cover = parse_cloud_cover(item)
    return {
        "scene_id": scene_id,
        "datetime": acq_dt_iso or "",
        "tif": out_tif_path,
        "sun_azimuth": meta.get("sun_azimuth"),
        "sun_elevation": meta.get("sun_elevation"),
        "image_quality_oli": meta.get("image_quality_oli"),
        "image_quality_tirs": meta.get("image_quality_tirs"),
        "cloud_cover": cloud_cover,
        "height": dims.get("height"),
        "width": dims.get("width"),
    }


def download_task(
    row_index: int,
    row_data: Dict,
    base_event_dt: datetime,
    plume_bounds: List[float],
    client: Client,
    processed_root: str,
    raw_root: str,
    offsets: List[int],
    tolerance_days: int,
    progress_tracker: Optional[Dict],
) -> Dict:
    plume_id = str(row_data.get("plume_id", "unknown"))
    plume_dir = os.path.join(processed_root, plume_id)
    os.makedirs(plume_dir, exist_ok=True)
    marker_file = os.path.join(plume_dir, PLUME_COMPLETION_MARKER)
    completed_offsets = load_completed_offsets(marker_file)
    new_records: Dict[int, Dict] = {}

    pending_offsets = [offset for offset in offsets if offset not in completed_offsets]
    if not pending_offsets:
        return {"index": row_index, "records": new_records}

    def process_single_offset(offset: int):
        target_dt = base_event_dt - timedelta(days=offset)
        window_start = target_dt - timedelta(days=tolerance_days)
        window_end = target_dt + timedelta(days=tolerance_days)

        cached_entry = find_cached_record(offset, target_dt, tolerance_days)
        if cached_entry is not None:
            cloned_record = clone_cached_record(cached_entry, plume_dir, offset)
            if cloned_record is not None:
                add_record_to_cache(offset, cloned_record)
                return offset, cloned_record, {
                    "offset": offset,
                    "scene_id": cloned_record.get("scene_id"),
                    "datetime": cloned_record.get("datetime"),
                }
            return offset, None, None

        items = search_landsat_items(client, plume_bounds, window_start, window_end)
        if not items:
            print(f"[info] plume {plume_id}: no Landsat items for offset {offset}")
            return offset, None, None

        selected = select_best_item(items, target_dt)
        if selected is None:
            print(
                f"[info] plume {plume_id}: no low-cloud Landsat scenes "
                f"within ±{tolerance_days} days for offset {offset}"
            )
            return offset, None, None

        try:
            scene_dir = ensure_scene_assets(selected, raw_root)
        except Exception as exc:
            print(f"[error] plume {plume_id}: failed to fetch raw scene: {exc}")
            return offset, None, None

        record = build_output_record(
            selected, scene_dir, plume_dir, plume_bounds, offset
        )
        if record is None:
            return offset, None, None

        add_record_to_cache(offset, record)
        return offset, record, {
            "offset": offset,
            "scene_id": record.get("scene_id"),
            "datetime": record.get("datetime"),
        }

    try:
        with ThreadPoolExecutor(
            max_workers=min(len(pending_offsets), MAX_SCENE_DOWNLOAD_WORKERS)
        ) as executor:
            futures = [executor.submit(process_single_offset, offset) for offset in pending_offsets]
            for future in futures:
                offset, record, completed_info = future.result()
                if record is not None:
                    new_records[offset] = record
                if completed_info is not None:
                    completed_offsets[offset] = completed_info

        if new_records:
            persist_completed_offsets(marker_file, completed_offsets)

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

    processed_root = str(
        config.get("l8_90360_processed_dir", DEFAULT_PROCESSED_DIR)
    )
    raw_root = str(config.get("l8_90360_raw_dir", DEFAULT_RAW_DIR))
    input_csv = str(
        config.get(
            "l8_90360_input_csv",
            DEFAULT_INPUT_CSV,
        )
    )
    output_csv = str(
        config.get(
            "l8_90360_output_csv",
            DEFAULT_OUTPUT_CSV,
        )
    )

    os.makedirs(processed_root, exist_ok=True)
    os.makedirs(raw_root, exist_ok=True)

    prev_output_df = None
    if os.path.exists(output_csv):
        try:
            prev_output_df = pd.read_csv(output_csv)
        except Exception as exc:
            print(f"[warn] Failed to load existing output CSV '{output_csv}': {exc}")

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

    if prev_output_df is not None:
        df = apply_previous_results(df, prev_output_df, per_offset_fields)
        seed_existing_records_from_dataframe(prev_output_df, per_offset_fields)
    seed_existing_records_from_dataframe(df, per_offset_fields)

    plume_times = df["plume_tif"].apply(extract_datetime_from_plume_tif)
    fallback_times = df["datetime"].apply(l8_processing.parse_iso_datetime)
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

    client = Client.open(PLANETARY_COMPUTER_STAC_URL)
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
                    client,
                    processed_root,
                    raw_root,
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
