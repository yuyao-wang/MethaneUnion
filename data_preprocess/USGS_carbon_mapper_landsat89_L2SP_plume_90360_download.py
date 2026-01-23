import ast
import json
import os
import re
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse
import shutil

import pandas as pd
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util.utils import load_config, parse_args  # noqa: E402
from landsat_c2_downloader import normalize_scene_id  # noqa: E402
import data_preprocess.carbon_mapper_landsat89_L2SP_plume_download as l8_processing  # noqa: E402


OFFSETS_DAYS = [90, 360]
SEARCH_TOLERANCE_DAYS = 30
MAX_CLOUD_COVER = 20.0
SCENE_DOWNLOAD_MARKER = ".planetary_pc_download_complete"
PLUME_COMPLETION_MARKER = "landsat_pc_offsets.json"
MAX_SCENE_DOWNLOAD_WORKERS = 4
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
USGS_SERVICE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"
USGS_DATASET_NAME = "landsat_ot_c2_l2"
USGS_DOWNLOAD_POLL_INTERVAL = 30
USGS_MAX_RESULTS = 200
DEFAULT_USGS_USERNAME = os.getenv("USGS_USERNAME", "")
DEFAULT_USGS_TOKEN = os.getenv("USGS_TOKEN", "")

PLUME_TIF_TIMESTAMP_PATTERN = re.compile(
    r"(\\d{4})(\\d{2})(\\d{2})[tT](\\d{2})(\\d{2})(\\d{2})"
)

scene_lock_map: Dict[str, threading.Lock] = {}
scene_lock_map_lock = threading.Lock()
existing_records_by_offset: Dict[int, List[Dict]] = {offset: [] for offset in OFFSETS_DAYS}
existing_record_keys: Set[Tuple[int, str]] = set()
existing_records_lock = threading.Lock()


class USGSError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


def send_usgs_request(endpoint: str, data: Dict[str, object], api_key: Optional[str] = None, exit_if_error: bool = True):
    url = f"{USGS_SERVICE_URL}{endpoint}"
    payload = json.dumps(data)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Auth-Token"] = api_key
    try:
        response = requests.post(url, payload, headers=headers, timeout=600)
    except requests.RequestException as exc:
        raise USGSError(f"Request to {endpoint} failed: {exc}") from exc

    http_status = response.status_code
    try:
        out = response.json()
    except ValueError as exc:
        raise USGSError(f"Invalid response from {endpoint}: {response.text}", status_code=http_status) from exc
    finally:
        response.close()

    error_code = out.get("errorCode")
    if error_code is not None:
        message = f"{error_code} - {out.get('errorMessage')}"
        print(message)
        if exit_if_error:
            raise USGSError(message, status_code=http_status, error_code=error_code)
        return None

    if http_status in (400, 401, 404):
        message = f"HTTP {http_status} for {endpoint}"
        print(message)
        if exit_if_error:
            raise USGSError(message, status_code=http_status)
        return None

    return out.get("data")


class USGSM2MClient:
    def __init__(self, username: str, token: str) -> None:
        if not username or not token:
            raise ValueError("USGS username/token must be provided")
        self.username = username
        self.token = token
        self.api_key: Optional[str] = None
        self.lock = threading.Lock()
        self.request_lock = threading.RLock()
        self.login()

    def login(self) -> None:
        payload = {"username": self.username, "token": self.token}
        with self.request_lock:
            data = send_usgs_request("login-token", payload, api_key=None)
        if not data:
            raise USGSError("Failed to retrieve apiKey from login-token response")
        with self.lock:
            self.api_key = data
        print("[info] Logged into USGS M2M service")

    def logout(self) -> None:
        key = self._get_api_key()
        if not key:
            return
        try:
            send_usgs_request("logout", {}, api_key=key, exit_if_error=False)
        finally:
            with self.lock:
                self.api_key = None

    def _get_api_key(self) -> Optional[str]:
        with self.lock:
            return self.api_key

    def request(self, endpoint: str, payload: Dict[str, object], exit_if_error: bool = True):
        attempts = 0
        while attempts < 2:
            key = self._get_api_key()
            if key is None:
                self.login()
                key = self._get_api_key()
            try:
                with self.request_lock:
                    return send_usgs_request(endpoint, payload, api_key=key, exit_if_error=exit_if_error)
            except USGSError as exc:
                if exc.status_code == 401 and attempts == 0:
                    print("[info] USGS apiKey expired, refreshing...")
                    self.login()
                    attempts += 1
                    continue
                raise
        raise USGSError("Failed to complete request after refreshing apiKey")


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


def download_usgs_scene(
    client: USGSM2MClient,
    entity_id: str,
    scene_id: str,
    raw_root: str,
) -> str:
    product_id = normalize_scene_id(scene_id)
    scene_dir = os.path.join(raw_root, product_id)
    marker = os.path.join(scene_dir, SCENE_DOWNLOAD_MARKER)

    if os.path.exists(scene_dir) and os.path.exists(marker):
        return scene_dir

    if os.path.exists(scene_dir):
        shutil.rmtree(scene_dir, ignore_errors=True)

    os.makedirs(raw_root, exist_ok=True)

    options_payload = {"datasetName": USGS_DATASET_NAME, "entityIds": [entity_id]}
    options = client.request("download-options", options_payload)
    available = [opt for opt in options if opt.get("available")]
    if not available:
        raise RuntimeError(f"No available USGS download options for {scene_id}")

    selected_option = available[0]
    downloads = [{"entityId": entity_id, "productId": selected_option["id"]}]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_label = f"l8_{product_id}"
    max_base_len = max(1, 50 - len(timestamp) - 1)
    if len(base_label) > max_base_len:
        base_label = base_label[:max_base_len]
    label = f"{base_label}_{timestamp}"

    request_payload = {"downloads": downloads, "label": label, "returnAvailable": True}
    request_data = client.request("download-request", request_payload)
    download_urls = [item["url"] for item in request_data.get("availableDownloads", [])]
    preparing_ids = [item["downloadId"] for item in request_data.get("preparingDownloads", [])]

    if preparing_ids:
        print(f"[info] waiting for {len(preparing_ids)} download(s) to become available...")
        time.sleep(USGS_DOWNLOAD_POLL_INTERVAL)
        retrieve_payload = {"label": label}
        retrieve_result = client.request("download-retrieve", retrieve_payload, exit_if_error=False)
        if retrieve_result:
            for item in retrieve_result.get("available", []):
                if item.get("downloadId") in preparing_ids:
                    download_urls.append(item["url"])

    if not download_urls:
        raise RuntimeError(f"No download URLs returned for scene {scene_id}")

    download_url = download_urls[0]
    parsed_path = urlparse(download_url).path
    archive_name = os.path.basename(parsed_path) or f"{product_id}.zip"
    _, ext = os.path.splitext(archive_name)
    if not ext:
        ext = ".zip"
    archive_path = os.path.join(raw_root, f"{product_id}{ext}")

    print(f"[download] {scene_id} ({entity_id}) -> {archive_path}")
    with requests.get(download_url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(archive_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    os.makedirs(scene_dir, exist_ok=True)
    try:
        shutil.unpack_archive(archive_path, scene_dir)
    except shutil.ReadError:
        with tarfile.open(archive_path) as tar:
            tar.extractall(scene_dir)
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)

    with open(marker, "w") as handle:
        handle.write(l8_processing.datetime_to_iso_z(datetime.now(timezone.utc)))
    return scene_dir


def ensure_usgs_scene(product: Dict, raw_root: str, client: USGSM2MClient) -> str:
    scene_id = product.get("scene_id")
    entity_id = product.get("entity_id")
    if not scene_id or not entity_id:
        raise RuntimeError("Invalid USGS product information")
    normalized = normalize_scene_id(scene_id)
    lock = get_scene_lock(normalized)
    with lock:
        scene_dir = os.path.join(raw_root, normalized)
        marker = os.path.join(scene_dir, SCENE_DOWNLOAD_MARKER)
        if os.path.exists(scene_dir) and os.path.exists(marker):
            return scene_dir
        return download_usgs_scene(client, entity_id, scene_id, raw_root)


def search_landsat_products(
    client: USGSM2MClient,
    bbox: List[float],
    start_dt: datetime,
    end_dt: datetime,
) -> List[Dict]:
    lon_min, lat_min, lon_max, lat_max = bbox
    spatial_filter = {
        "filterType": "mbr",
        "lowerLeft": {"latitude": lat_min, "longitude": lon_min},
        "upperRight": {"latitude": lat_max, "longitude": lon_max},
    }
    temporal_filter = {
        "start": start_dt.strftime("%Y-%m-%d"),
        "end": end_dt.strftime("%Y-%m-%d"),
    }
    cloud_filter = {"min": 0, "max": MAX_CLOUD_COVER}
    payload = {
        "datasetName": USGS_DATASET_NAME,
        "maxResults": USGS_MAX_RESULTS,
        "sceneFilter": {
            "spatialFilter": spatial_filter,
            "acquisitionFilter": temporal_filter,
            "cloudCoverFilter": cloud_filter,
        },
        "sortField": "cloudCover",
        "sortDirection": "ASC",
    }
    try:
        data = client.request("scene-search", payload)
    except USGSError as exc:
        print(f"[error] Failed USGS search: {exc}")
        return []

    items: List[Dict] = []
    for scene in data.get("results", []):
        entity_id = scene.get("entityId")
        display_id = scene.get("displayId") or entity_id
        if not entity_id or not display_id:
            continue
        acq_time = l8_processing.parse_iso_datetime(scene.get("acquisitionDate"))
        if acq_time is None:
            coverage = scene.get("temporalCoverage") or {}
            acq_time = l8_processing.parse_iso_datetime(coverage.get("startDate"))
        if acq_time is None:
            continue
        cloud = scene.get("cloudCover")
        try:
            cloud_cover = float(cloud) if cloud is not None else None
        except (TypeError, ValueError):
            cloud_cover = None
        items.append(
            {
                "entity_id": entity_id,
                "scene_id": display_id,
                "acq_time": acq_time.astimezone(timezone.utc),
                "cloud_cover": cloud_cover,
            }
        )
    return items


def select_best_product(products: List[Dict], target_dt: datetime) -> Optional[Dict]:
    if not products:
        return None
    filtered = []
    for prod in products:
        acq_time = prod.get("acq_time")
        cloud_cover = prod.get("cloud_cover")
        if acq_time is None:
            continue
        if cloud_cover is not None and cloud_cover > MAX_CLOUD_COVER:
            continue
        filtered.append(prod)
    if not filtered:
        return None
    same_day = [prod for prod in filtered if prod["acq_time"].date() == target_dt.date()]
    candidates = same_day if same_day else filtered
    return min(candidates, key=lambda prod: abs((prod["acq_time"] - target_dt).total_seconds()))


def build_output_record(
    product: Dict,
    scene_dir: str,
    plume_dir: str,
    plume_bounds: List[float],
    offset: int,
) -> Optional[Dict]:
    scene_id = normalize_scene_id(product.get("scene_id", "unknown")) or "unknown"
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
    if not acq_dt_iso:
        acq_time = product.get("acq_time")
        if isinstance(acq_time, datetime):
            acq_dt_iso = l8_processing.datetime_to_iso_z(acq_time)

    return {
        "scene_id": scene_id,
        "datetime": acq_dt_iso or "",
        "tif": out_tif_path,
        "sun_azimuth": meta.get("sun_azimuth"),
        "sun_elevation": meta.get("sun_elevation"),
        "image_quality_oli": meta.get("image_quality_oli"),
        "image_quality_tirs": meta.get("image_quality_tirs"),
        "cloud_cover": product.get("cloud_cover"),
        "height": dims.get("height"),
        "width": dims.get("width"),
    }


def download_task(
    row_index: int,
    row_data: Dict,
    base_event_dt: datetime,
    plume_bounds: List[float],
    usgs_client: USGSM2MClient,
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

        products = search_landsat_products(usgs_client, plume_bounds, window_start, window_end)
        if not products:
            print(f"[info] plume {plume_id}: no USGS Landsat scenes for offset {offset}")
            return offset, None, None

        selected = select_best_product(products, target_dt)
        if selected is None:
            print(
                f"[info] plume {plume_id}: no low-cloud Landsat scenes "
                f"within ±{tolerance_days} days for offset {offset}"
            )
            return offset, None, None

        try:
            scene_dir = ensure_usgs_scene(selected, raw_root, usgs_client)
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

    usgs_username = config.get("usgs_username", DEFAULT_USGS_USERNAME)
    usgs_token = config.get("usgs_token", DEFAULT_USGS_TOKEN)
    if not usgs_username or not usgs_token:
        raise ValueError(
            "USGS credentials are missing. Provide usgs_username/usgs_token in the config or "
            "set USGS_USERNAME/USGS_TOKEN environment variables."
        )

    usgs_client = USGSM2MClient(usgs_username, usgs_token)
    try:
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
                        usgs_client,
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
    finally:
        usgs_client.logout()


if __name__ == "__main__":
    main()
