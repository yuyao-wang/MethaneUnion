# download_landsat_for_cm.py

import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from pathlib import Path
from urllib.parse import urlparse
import shutil
import tarfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import tifffile
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utils import parse_args, load_config

from tqdm import tqdm

# Translated comment
WINDOW_SIZE = 512
# L8_PLUME_BASE_DIR = "/data2/yuyao/methane_emission/landsat_l2sp_plume_stacks"
base_dir = '/data2/yuyao/methane_emission/carbonmapper_data_l89_l2sp'
MAX_L8_PER_PLUME = 3
PLUME_COMPLETION_MARKER = "landsat_l2sp_complete.txt"
MAX_CLOUD_COVER_PERCENT = 20.0
USGS_SERVICE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"
USGS_DATASET_NAME = "landsat_ot_c2_l2"
USGS_DOWNLOAD_POLL_INTERVAL = 30  # seconds
USGS_MAX_RESULTS = 200
DEFAULT_USGS_USERNAME = os.getenv("USGS_USERNAME", "")
DEFAULT_USGS_TOKEN = os.getenv("USGS_TOKEN", "")
SEARCH_WINDOW_DAYS = 7
DOWNLOAD_MARKER = ".download_complete"

product_lock_map: Dict[str, threading.Lock] = {}
product_lock_map_lock = threading.Lock()


def get_product_lock(product_name: str) -> threading.Lock:
    with product_lock_map_lock:
        lock = product_lock_map.get(product_name)
        if lock is None:
            lock = threading.Lock()
            product_lock_map[product_name] = lock
        return lock


class USGSError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


def send_usgs_request(endpoint: str, data: Dict[str, object], api_key: Optional[str] = None, exit_if_error: bool = True):
    """
    Helper wrapper for POST requests to the USGS M2M API.
    """
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
    """
    Light wrapper over the USGS M2M API that keeps the apiKey fresh.
    """

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
        """
        Issue a request using the cached apiKey, retrying once if the token is expired (401).
        """
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


def select_landsat_items(items, event_dt, max_scenes=3):
    """
 STAC items distancetime max_scenes .
 , .
    """
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
        # Translated comment
        closest_same_day = min(same_day, key=lambda p: abs((p["acq_time"] - event_dt).total_seconds()))
        selected.append(closest_same_day)

        if before:
            closest_before = max(before, key=lambda p: p["acq_time"])
            selected.append(closest_before)
        if after:
            closest_after = min(after, key=lambda p: p["acq_time"])
            selected.append(closest_after)
    else:
        # Translated comment
        sorted_items = sorted(items, key=lambda p: abs((p["acq_time"] - event_dt).total_seconds()))
        selected = sorted_items[:max_scenes]

    # Translated comment
    return sorted(selected, key=lambda p: p["acq_time"])

def normalize_scene_id(scene_id: str) -> str:
    """
    Landsat products sometimes append suffixes like "_SR" or "_ST".
    Normalize to the base product ID so local folders stay consistent.
    """
    suffixes = ("_SR", "_ST")
    for suffix in suffixes:
        if scene_id.endswith(suffix):
            return scene_id[: -len(suffix)]
    return scene_id


def search_landsat_products(
    client: USGSM2MClient,
    plume_bounds: List[float],
    window_start: datetime,
    window_end: datetime,
    max_results: int = USGS_MAX_RESULTS,
) -> List[Dict[str, object]]:
    """
    Query USGS M2M for Landsat 8/9 Collection 2 Level-2 scenes intersecting the plume bounding box.
    """
    lon_min, lat_min, lon_max, lat_max = plume_bounds
    spatial_filter = {
        "filterType": "mbr",
        "lowerLeft": {"latitude": lat_min, "longitude": lon_min},
        "upperRight": {"latitude": lat_max, "longitude": lon_max},
    }
    temporal_filter = {
        "start": window_start.strftime("%Y-%m-%d"),
        "end": window_end.strftime("%Y-%m-%d"),
    }
    cloud_filter = {"min": 0, "max": MAX_CLOUD_COVER_PERCENT}
    payload = {
        "datasetName": USGS_DATASET_NAME,
        "maxResults": max_results,
        "startingNumber": 1,
        "sceneFilter": {
            "spatialFilter": spatial_filter,
            "acquisitionFilter": temporal_filter,
            "cloudCoverFilter": cloud_filter,
        },
        "sortField": "cloudCover",
        "sortDirection": "ASC",
    }

    try:
        response = client.request("scene-search", payload)
    except USGSError as exc:
        print(f"[error] Failed to search USGS scenes: {exc}")
        return []

    results = []
    for scene in response.get("results", []):
        entity_id = scene.get("entityId")
        display_id = scene.get("displayId") or entity_id
        if entity_id is None or display_id is None:
            continue
        acq_time = parse_iso_datetime(scene.get("acquisitionDate"))
        if acq_time is None:
            temporal = scene.get("temporalCoverage") or {}
            acq_time = parse_iso_datetime(temporal.get("startDate"))
        if acq_time is None:
            date_str = scene.get("acquisitionDate")
            if isinstance(date_str, str):
                try:
                    acq_time = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    acq_time = None
        if acq_time is None:
            continue
        results.append(
            {
                "entity_id": entity_id,
                "scene_id": display_id,
                "acq_time": acq_time,
            }
        )
    return results


def download_usgs_scene(
    client: USGSM2MClient,
    entity_id: str,
    scene_id: str,
    output_dir: str,
) -> str:
    """
    Request download links from USGS and extract the archive into output_dir/scene_id.
    """
    product_id = normalize_scene_id(scene_id)
    scene_path = os.path.join(output_dir, product_id)
    marker_file = os.path.join(scene_path, DOWNLOAD_MARKER)

    if os.path.exists(scene_path) and os.path.exists(marker_file):
        return scene_path

    if os.path.exists(scene_path):
        print(f"[info] removing incomplete download for {scene_id}")
        shutil.rmtree(scene_path, ignore_errors=True)

    os.makedirs(output_dir, exist_ok=True)

    options_payload = {"datasetName": USGS_DATASET_NAME, "entityIds": [entity_id]}
    options = client.request("download-options", options_payload)
    available = [opt for opt in options if opt.get("available")]
    if not available:
        raise RuntimeError(f"No available download options for scene {scene_id}")

    selected_option = available[0]
    downloads = [
        {"entityId": entity_id, "productId": selected_option["id"]},
    ]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    label_base = f"l8_{product_id}"
    max_base_len = max(1, 50 - len(timestamp) - 1)  # keep full label <= 50 chars
    if len(label_base) > max_base_len:
        label_base = label_base[:max_base_len]
    label = f"{label_base}_{timestamp}"
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
    archive_path = os.path.join(output_dir, f"{product_id}{ext}")

    print(f"[download] {scene_id} ({entity_id}) -> {archive_path}")
    with requests.get(download_url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(archive_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

    os.makedirs(scene_path, exist_ok=True)
    try:
        shutil.unpack_archive(archive_path, scene_path)
    except shutil.ReadError:
        # Some downloads are TAR archives renamed as .zip; try tarfile explicitly.
        with tarfile.open(archive_path) as tar:
            tar.extractall(scene_path)
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)

    with open(marker_file, "w") as marker:
        marker.write(datetime_to_iso_z(datetime.now(timezone.utc)))

    return scene_path



def find_scene_file(scene_dir: str, filename: str) -> Optional[str]:
    scene_path = Path(scene_dir)
    if not scene_path.exists():
        return None
    for path in scene_path.rglob(filename):
        if path.is_file():
            return str(path)
    return None


def latlon_to_pixel(lat: float, lon: float, dataset: rasterio.io.DatasetReader):
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return ~dataset.transform * (x, y)


def crop_band_to_window(tif_path: str, plume_bounds: List[float]) -> Optional[np.ndarray]:
    if not os.path.exists(tif_path):
        print(f"[warn] band not found: {tif_path}")
        return None

    with rasterio.open(tif_path) as dataset:
        lon_min, lat_min, lon_max, lat_max = plume_bounds

        top_left = latlon_to_pixel(lat_max, lon_min, dataset)
        bottom_right = latlon_to_pixel(lat_min, lon_max, dataset)

        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2

        half = WINDOW_SIZE // 2
        col_start = int(np.floor(center_x - half))
        row_start = int(np.floor(center_y - half))
        col_end = col_start + WINDOW_SIZE
        row_end = row_start + WINDOW_SIZE

        if col_start < 0:
            col_end += -col_start
            col_start = 0
        if row_start < 0:
            row_end += -row_start
            row_start = 0
        if col_end > dataset.width:
            shift = col_end - dataset.width
            col_start -= shift
            col_end = dataset.width
        if row_end > dataset.height:
            shift = row_end - dataset.height
            row_start -= shift
            row_end = dataset.height

        col_start = max(0, col_start)
        row_start = max(0, row_start)
        window_width = max(0, col_end - col_start)
        window_height = max(0, row_end - row_start)

        if window_width == 0 or window_height == 0:
            print(f"[warn] empty window for {tif_path}")
            return None

        window = Window(col_start, row_start, window_width, window_height)
        clipped = dataset.read(1, window=window)
        return clipped


def parse_landsat_mtl(mtl_path: Optional[str]) -> Dict[str, Optional[float]]:
    """
 MTL.txt :       - DATE_ACQUIRED + SCENE_CENTER_TIME -> acq_datetime_iso
      - SUN_AZIMUTH, SUN_ELEVATION
      - IMAGE_QUALITY_OLI, IMAGE_QUALITY_TIRS
    """
    result = {
        "acq_datetime_iso": None,
        "sun_azimuth": None,
        "sun_elevation": None,
        "image_quality_oli": None,
        "image_quality_tirs": None,
    }

    if not mtl_path or not os.path.exists(mtl_path):
        print(f"[warn] MTL not found: {mtl_path}")
        return result

    meta: Dict[str, str] = {}
    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("GROUP") or line.startswith("END_") or line == "END":
                continue
            if "=" not in line:
                continue
            k, v = [x.strip() for x in line.split("=", 1)]
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            meta[k] = v

    date_str = meta.get("DATE_ACQUIRED")
    time_str = meta.get("SCENE_CENTER_TIME")

    if date_str and time_str:
        t = time_str
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt_str = f"{date_str}T{t}"
        try:
            dt = datetime.fromisoformat(dt_str)
            result["acq_datetime_iso"] = datetime_to_iso_z(dt)
        except Exception as e:
            print(f"[warn] failed to parse datetime from MTL: {dt_str} ({e})")

    def _get_float(k: str):
        v = meta.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _get_int(k: str):
        v = meta.get(k)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    result["sun_azimuth"] = _get_float("SUN_AZIMUTH")
    result["sun_elevation"] = _get_float("SUN_ELEVATION")
    result["image_quality_oli"] = _get_int("IMAGE_QUALITY_OLI")
    result["image_quality_tirs"] = _get_int("IMAGE_QUALITY_TIRS")

    return result


def build_landsat_stack_for_plume(
    scene_dir: str,
    scene_id: str,
    plume_bounds: List[float],
    out_tif_path: str,
) -> Optional[Dict[str, int]]:
    """
 scene_dir load SR_B1-7 + ST_B10, WINDOW_SIZE x WINDOW_SIZE, stack [8,H,W].
    """
    band_suffixes = [f"SR_B{b}" for b in range(1, 8)] + ["ST_B10"]

    bands = []
    current_shape = None

    for suffix in band_suffixes:
        tif_name = f"{scene_id}_{suffix}.TIF"
        tif_path = find_scene_file(scene_dir, tif_name)
        if tif_path is None:
            print(f"[warn] missing band {tif_name}")
            continue
        clipped = crop_band_to_window(tif_path, plume_bounds)
        if clipped is None:
            print(f"[warn] skip band {tif_name}")
            continue

        if current_shape is None:
            current_shape = clipped.shape
        else:
            if clipped.shape != current_shape:
                print(f"[warn] shape mismatch for {tif_name}: {clipped.shape} != {current_shape}")
                continue

        bands.append(clipped)

    if not bands:
        print(f"[warn] no valid bands for {scene_id}")
        return None

    stacked = np.stack(bands, axis=0)  # [B,H,W]
    os.makedirs(os.path.dirname(out_tif_path), exist_ok=True)
    print(f"[write] L8 stack -> {out_tif_path}")
    tifffile.imwrite(out_tif_path, stacked)

    h, w = current_shape
    return {"height": int(h), "width": int(w)}


def process_single_landsat_scene(
    scene_id: str,
    landsat_raw_root: str,
    plume_dir: str,
    plume_bounds: List[float],
) -> Optional[Dict[str, object]]:
    """
 :  1. landsat_raw_root/scene_id
 2. parse MTL time / /  3. stack plume_dir
    """
    product_id = normalize_scene_id(scene_id)
    scene_dir = os.path.join(landsat_raw_root, product_id)
    if not os.path.exists(scene_dir):
        print(f"[warn] scene directory missing: {scene_dir}")
        return None

    mtl_path = find_scene_file(scene_dir, f"{product_id}_MTL.txt")
    meta = parse_landsat_mtl(mtl_path)
    acq_dt_iso = meta.get("acq_datetime_iso")

    if acq_dt_iso is None:
        parts = scene_id.split("_")
        if len(parts) >= 4 and len(parts[3]) >= 8:
            date_str = parts[3][:8]
            try:
                dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                acq_dt_iso = datetime_to_iso_z(dt)
            except Exception:
                acq_dt_iso = None

    out_tif_name = f"l8_{product_id}.tif"
    out_tif_path = os.path.join(plume_dir, out_tif_name)

    dims = build_landsat_stack_for_plume(scene_dir, product_id, plume_bounds, out_tif_path)
    if dims is None:
        return None

    return {
        "scene_id": scene_id,
        "product_id": product_id,
        "datetime": acq_dt_iso or "",
        "tif_path": out_tif_path,
        "height": dims["height"],
        "width": dims["width"],
        "sun_azimuth": meta.get("sun_azimuth"),
        "sun_elevation": meta.get("sun_elevation"),
        "image_quality_oli": meta.get("image_quality_oli"),
        "image_quality_tirs": meta.get("image_quality_tirs"),
        "root_dir": scene_dir,
    }


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


def download_task_l8(
    row_index,
    row_data,
    plume_bounds,
    usgs_client: USGSM2MClient,
    progress_tracker,
    max_scenes=MAX_L8_PER_PLUME,
):
    """
 plume:  - USGS M2M Landsat 8/9 Collection 2 Level-2
 - max_scenes time
 - downloadgenerate plume stack
    """
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

        products = search_landsat_products(usgs_client, plume_bounds, window_start, window_end)
        if not products:
            print(f"[info] plume {plume_id}: no L8/L9 scenes found in USGS search window")
            return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}

        selected_items = select_landsat_items(products, event_dt, max_scenes=max_scenes)
        has_same_day = 1 if any(it["acq_time"].date() == event_dt.date() for it in selected_items) else 0

        out_root = config['raw_data_dir_l89_l2sp']
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(plume_dir, exist_ok=True)

        recorded_scenes = []

        for it in selected_items:
            scene_id = it.get("scene_id")
            entity_id = it.get("entity_id")
            acq_time = it.get("acq_time")

            if scene_id is None or entity_id is None:
                continue

            product_id = normalize_scene_id(scene_id)
            lock = get_product_lock(product_id)
            try:
                with lock:
                    download_usgs_scene(
                        usgs_client,
                        entity_id,
                        scene_id,
                        out_root,
                    )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[error] plume {plume_id}: download scene {scene_id} failed: {exc}")
                continue

            try:
                scene_info = process_single_landsat_scene(
                    scene_id,
                    out_root,
                    plume_dir,
                    plume_bounds,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[error] plume {plume_id}: process scene {scene_id} failed: {exc}")
                continue

            if scene_info is None:
                continue

            if not scene_info.get("datetime") and acq_time is not None:
                scene_info["datetime"] = datetime_to_iso_z(acq_time)

            recorded_scenes.append(scene_info)
            time.sleep(0.2)

        if recorded_scenes:
            # Translated comment
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

    usgs_username = config.get("usgs_username", DEFAULT_USGS_USERNAME)
    usgs_token = config.get("usgs_token", DEFAULT_USGS_TOKEN)
    if not usgs_username or not usgs_token:
        raise ValueError("USGS credentials are missing. Set usgs_username/usgs_token in the config or USGS_USERNAME/USGS_TOKEN env vars.")

    usgs_client = USGSM2MClient(usgs_username, usgs_token)

    merged_csv_path = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv"
    output_csv_path = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_l8.csv"

    try:
        df = pd.read_csv(merged_csv_path)

        # Translated comment
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

        # Translated comment
        plume_tif_mask = df["plume_tif"].apply(lambda v: isinstance(v, str) and len(v) > 0)
        processable_mask = plume_tif_mask

        total_processable = int(processable_mask.sum())
        overall_start_time = time.time()
        progress_tracker = None
        progress_bar = None
        if total_processable > 0:
            if tqdm is not None:
                progress_bar = tqdm(total=total_processable, desc="L8/L9 plumes", dynamic_ncols=True)
            progress_tracker = {
                "lock": threading.Lock(),
                "completed": 0,
                "total": total_processable,
                "start_time": overall_start_time,
                "tqdm": progress_bar,
            }

        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
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
                    usgs_client,
                    progress_tracker
                ))

        results = [f.result() for f in futures]

        if progress_bar is not None:
            progress_bar.close()

        # Translated comment
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
    finally:
        usgs_client.logout()
