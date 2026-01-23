import json
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import rasterio
import requests
import tifffile
from concurrent.futures import ThreadPoolExecutor
from pyproj import Transformer
from rasterio.windows import Window

# CDSE_USERNAME0 = 'yuyao16@ualberta.ca'
# CDSE_PASSWORD0 = 'finhah-3zihty-seHmuf'

CDSE_USERNAME0 = 'yuyaow42@gmail.com'
CDSE_PASSWORD0 = 'finhah-3zihty-seHmuf'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utils import load_config, parse_args  # noqa: E402

pattern = re.compile(r".*B[0-9A-Za-z]+_20m\.jp2$")
type_pattern = r".*B([0-9A-Za-z]+)_20m\.jp2$"

DEFAULT_LOCAL_BASE_DIR = '/data2/yuyao/methane_emission/carbonmapper_data_l2a_90360'
DEFAULT_DRIVE_ROOT = os.environ.get('GOOGLE_DRIVE_ROOT')
RAW_SUBDIR_NAME = 'raw_data_dir_s2_90360'
OFFSETS_DAYS = [90, 360]
SEARCH_WINDOW_DAYS = 50
PLUME_COMPLETION_MARKER = 'download_stub_pre.json'
DOWNLOAD_COMPLETION_MARKER = '.download_complete'
BACKOFF_STATUS_CODE = 429
BACKOFF_BASE_SECONDS = 30
BACKOFF_MAX_SECONDS = 120
BACKOFF_MAX_RETRIES = 9
base_dir = DEFAULT_LOCAL_BASE_DIR
raw_data_dir = os.path.join('/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao', RAW_SUBDIR_NAME)
# raw_data_dir = os.path.join('./', RAW_SUBDIR_NAME)
product_lock_map: Dict[str, threading.Lock] = {}
product_lock_map_lock = threading.Lock()
proxy_manager_lock = threading.Lock()
proxy_manager: Optional['ProxyManager'] = None


class ProxyManager:
    def __init__(self, proxies: List[str], cooldown_seconds: int = 180) -> None:
        self._proxies = [p.strip() for p in proxies if isinstance(p, str) and p.strip()]
        self._cooldowns: Dict[str, float] = {}
        self._index = 0
        self._lock = threading.Lock()
        self.cooldown_seconds = cooldown_seconds

    def has_proxies(self) -> bool:
        return len(self._proxies) > 0

    def acquire(self) -> Optional[str]:
        if not self._proxies:
            return None
        with self._lock:
            now = time.time()
            n = len(self._proxies)
            for _ in range(n):
                proxy = self._proxies[self._index]
                self._index = (self._index + 1) % n
                if self._cooldowns.get(proxy, 0) <= now:
                    return proxy
        return None

    def report_failure(self, proxy: Optional[str]):
        if not proxy:
            return
        with self._lock:
            self._cooldowns[proxy] = time.time() + self.cooldown_seconds

    def report_success(self, proxy: Optional[str]):
        if not proxy:
            return
        with self._lock:
            self._cooldowns.pop(proxy, None)

    def size(self) -> int:
        return len(self._proxies)


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def datetime_to_query_string(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def datetime_to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def datetime_to_filename(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_proxy_dict(proxy_url: Optional[str]) -> Optional[Dict[str, str]]:
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def compute_backoff_delay(attempt: int, headers: Optional[Dict[str, str]] = None) -> int:
    retry_after = None
    if headers:
        raw_retry = headers.get('Retry-After')
        if raw_retry:
            try:
                retry_after = int(raw_retry)
            except ValueError:
                try:
                    retry_dt = parse_iso_datetime(raw_retry)
                    if retry_dt:
                        retry_after = max(0, int((retry_dt - datetime.now(timezone.utc)).total_seconds()))
                except Exception:
                    retry_after = None
    if retry_after is None:
        retry_after = BACKOFF_BASE_SECONDS * (attempt + 1)
    return min(BACKOFF_MAX_SECONDS, max(BACKOFF_BASE_SECONDS, retry_after))


def request_with_backoff(request_fn, description: str = 'request'):
    for attempt in range(BACKOFF_MAX_RETRIES):
        proxy = None
        with proxy_manager_lock:
            if proxy_manager is not None:
                proxy = proxy_manager.acquire()
        try:
            response = request_fn(proxy)
        except requests.RequestException:
            if proxy_manager and proxy:
                proxy_manager.report_failure(proxy)
            raise
        if response.status_code == BACKOFF_STATUS_CODE:
            wait_seconds = compute_backoff_delay(attempt, response.headers)
            print(f'HTTP {BACKOFF_STATUS_CODE} on {description}; retry in {wait_seconds}s')
            response.close()
            time.sleep(wait_seconds)
            if proxy_manager and proxy:
                proxy_manager.report_failure(proxy)
            continue
        if response.status_code == 402:
            print(f"HTTP 402 on {description}; proxy {proxy or 'direct'} will be rotated")
            response.close()
            if proxy_manager and proxy:
                proxy_manager.report_failure(proxy)
            time.sleep(1)
            continue
        if proxy_manager and proxy:
            proxy_manager.report_success(proxy)
        return response
    raise RuntimeError(f'Exceeded maximum retries for {description} due to repeated HTTP {BACKOFF_STATUS_CODE}')


def load_credential_pool(config) -> List[Dict[str, str]]:
    pool: List[Dict[str, str]] = []
    idx = 0
    while True:
        username = config.get(f'cdse_username{idx}')
        password = config.get(f'cdse_password{idx}')
        env_username = os.environ.get(f'CDSE_USERNAME{idx}')
        env_password = os.environ.get(f'CDSE_PASSWORD{idx}')
        if not username:
            username = env_username
        if not password:
            password = env_password
        if username and password:
            pool.append({'username': username, 'password': password})
            idx += 1
            continue
        if username or password:
            print(f'Warning: incomplete credential pair for index {idx}; skipping')
            idx += 1
            continue
        break
    if len(pool) == 0:
        default_username = config.get('cdse_username', CDSE_USERNAME0)
        default_password = config.get('cdse_password', CDSE_PASSWORD0)
        if default_username and default_password:
            pool.append({'username': default_username, 'password': default_password})
    return pool


def build_proxy_manager(config: Dict) -> Optional[ProxyManager]:
    pool_cfg = config.get('proxy_pool')
    if not isinstance(pool_cfg, dict):
        return None
    entries = pool_cfg.get('entries') or []
    enabled = pool_cfg.get('enabled', True)
    if not enabled:
        return None
    proxies: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            value = entry.strip()
            if not value:
                continue
            if not value.startswith('http://') and not value.startswith('https://'):
                value = f"http://{value}"
            proxies.append(value)
    if not proxies:
        return None
    cooldown = pool_cfg.get('cooldown_seconds', 180)
    manager = ProxyManager(proxies, cooldown_seconds=cooldown)
    if manager.has_proxies():
        print(f"[info] proxy pool enabled with {manager.size()} entries")
        return manager
    return None


def get_product_lock(product_name: str) -> threading.Lock:
    with product_lock_map_lock:
        lock = product_lock_map.get(product_name)
        if lock is None:
            lock = threading.Lock()
            product_lock_map[product_name] = lock
        return lock


# get authentication
def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as exc:  # pragma: no cover - network errors only at runtime
        raise Exception(
            f"Access token creation failed. Response from the server was: {r.text}"
        ) from exc
    return r.json()["access_token"]


def latlon_to_pixel(lat: float, lon: float, dataset) -> Tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return ~dataset.transform * (x, y)


def parse_a_file(file_path: str, plume_bounds: List[float]) -> Optional[np.ndarray]:
    with rasterio.open(file_path) as dataset:
        top_left = latlon_to_pixel(plume_bounds[3], plume_bounds[0], dataset)
        bottom_right = latlon_to_pixel(plume_bounds[1], plume_bounds[2], dataset)
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2

        window_size = 512
        half_window = window_size // 2
        col_start = int(np.floor(center_x - half_window))
        row_start = int(np.floor(center_y - half_window))
        col_end = col_start + window_size
        row_end = row_start + window_size
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
            return None
        window = Window(col_start, row_start, window_width, window_height)
        clipped = dataset.read(1, window=window)
        return clipped


def download(access_token: str, output_dir: str, plume_id: str, product_id: str, name: str,
             plume_bounds: List[float], tif_output_path: str) -> Optional[Tuple[int, int]]:
    output_path = os.path.join(output_dir, name)
    marker_file = os.path.join(output_path, DOWNLOAD_COMPLETION_MARKER)
    if os.path.exists(output_path) and not os.path.exists(marker_file):
        print(f'found incomplete product {name}, removing and redownloading')
        shutil.rmtree(output_path, ignore_errors=True)
    if (not os.path.exists(output_path)) or (not os.path.exists(marker_file)):
        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        headers = {"Authorization": f"Bearer {access_token}"}
        session = requests.Session()
        session.headers.update(headers)
        response = request_with_backoff(
            lambda proxy: session.get(
                url,
                headers=headers,
                stream=True,
                proxies=build_proxy_dict(proxy),
            ),
            description=f'download {name}',
        )

        folder_pattern = 'GRANULE/L2A_T'
        subfolder_pattern = '/IMG_DATA/R20m'
        QI_data_pattern = '/QI_DATA/'
        important_file = 'MTD_MSIL2A.xml'

        try:
            if response.status_code == 200:
                zip_output_path = os.path.join(output_dir, name + '.zip')
                with open(zip_output_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)

                os.makedirs(output_path, exist_ok=True)
                with zipfile.ZipFile(zip_output_path, 'r') as zip_ref:
                    all_files = zip_ref.namelist()
                    directories_to_extract = [
                        file for file in all_files
                        if folder_pattern in file and (subfolder_pattern in file or QI_data_pattern in file)
                    ]
                    important_files = [file for file in all_files if important_file in file]

                    for file in directories_to_extract:
                        if file.endswith('/'):
                            continue
                        filename = file[file.rfind('/') + 1:]
                        if len(filename) == 0:
                            continue
                        target_path = os.path.join(output_dir, name, filename)
                        with zip_ref.open(file) as f:
                            content = f.read()
                            with open(target_path, 'wb') as target_file:
                                target_file.write(content)

                    for file in important_files:
                        if file.endswith('/'):
                            continue
                        filename = file[file.rfind('/') + 1:]
                        if len(filename) == 0:
                            continue
                        target_path = os.path.join(output_dir, name, filename)
                        with zip_ref.open(file) as f:
                            content = f.read()
                            with open(target_path, 'wb') as target_file:
                                target_file.write(content)

                if os.path.exists(zip_output_path):
                    os.remove(zip_output_path)
                with open(marker_file, 'w') as marker:
                    marker.write('ok')
            else:
                print(f'request failed {response.status_code}')
                return None
        finally:
            response.close()
            session.close()
    else:
        print(f'output path {output_path} already exists with completion marker')

    img_output = None
    current_shape = None
    jp2_dir = Path(output_path)
    for file_path in jp2_dir.rglob('*.jp2'):
        if os.path.isfile(file_path) and pattern.match(str(file_path)):
            spectrum_type_str = re.search(type_pattern, file_path.name).group(1)
            spectrum_type = 8 if spectrum_type_str == '8A' else int(spectrum_type_str)
            clipped = parse_a_file(file_path, plume_bounds)
            if clipped is None:
                continue
            if img_output is None:
                current_shape = clipped.shape
                img_output = np.zeros((12, current_shape[0], current_shape[1]), dtype=clipped.dtype)
            if clipped.shape != current_shape:
                print(f'skipping band {file_path} due to mismatched shape {clipped.shape} != {current_shape}')
                continue
            img_output[spectrum_type - 1] = clipped
    if img_output is None:
        print(f'no valid JP2 data found for product {name}')
        return None
    os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
    print(f'final tif file output path {tif_output_path}')
    tifffile.imwrite(tif_output_path, img_output)
    return current_shape


def fetch_products(poly: str, start_ts: str, end_ts: str) -> List[Dict]:
    products: List[Dict] = []
    next_link = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq 'SENTINEL-2' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le 20.00) "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON({poly})') "
        f"and ContentDate/Start gt {start_ts} "
        f"and ContentDate/Start lt {end_ts}"
        "&$top=1000"
    )
    
    while next_link:
        print(f'current link {next_link}')
        try:
            resp = request_with_backoff(
                lambda proxy: requests.get(next_link, proxies=build_proxy_dict(proxy)),
                description='catalogue query'
            )
            resp.raise_for_status()
            payload = resp.json()
            print(f"payload received {len(payload.get('value', []))} products")
        except Exception as exc:
            print(f'Failed to query catalogue: {exc}')
            break

        values = payload.get('value', [])
        for product in values:
            content_date = product.get('ContentDate', {})
            start_time_str = content_date.get('Start')
            if not start_time_str:
                print("No ContentDate.Start for product:", product.get('Name'))
                continue

            acq_time = parse_iso_datetime(start_time_str)
            if acq_time is None:
                print("Failed to parse datetime:", start_time_str)
                continue

            products.append({
                'Id': product.get('Id'),
                'Name': product.get('Name'),
                'acq_time': acq_time
            })

        next_link = payload.get('@odata.nextLink', "")
        print("Collected products:", len(products))
        print(f'next link {next_link}')
    return products


def select_closest_product(products: List[Dict], target_dt: datetime) -> Optional[Dict]:
    if len(products) == 0:
        return None
    same_day = [p for p in products if p['acq_time'].date() == target_dt.date()]
    if len(same_day) > 0:
        return min(same_day, key=lambda p: abs((p['acq_time'] - target_dt).total_seconds()))
    return min(products, key=lambda p: abs((p['acq_time'] - target_dt).total_seconds()))


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


class RefreshableAccessToken:
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.value = get_access_token(username, password)
        self.lock = threading.Lock()

    def update(self):
        with self.lock:
            self.value = get_access_token(self.username, self.password)

    def get(self) -> str:
        with self.lock:
            return self.value


def refresh_variable(variable: RefreshableAccessToken):
    while True:
        variable.update()
        time.sleep(300)


def download_task(row_index: int, row_data: Dict, poly: str, base_event_dt: datetime,
                  plume_bounds: List[float], offsets: List[int], search_window_days: int,
                  raw_root: str, access_token: RefreshableAccessToken,
                  progress_tracker: Optional[Dict]):
    plume_id = str(row_data.get('plume_id', 'unknown'))
    plume_dir = os.path.join(base_dir, plume_id)
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
            products = fetch_products(poly, datetime_to_query_string(window_start), datetime_to_query_string(window_end))
            selected_product = select_closest_product(products, target_dt)
            if selected_product is None:
                print(f'No S2 for plume {plume_id} offset {offset}')
                continue

            local_path = os.path.join(raw_root, selected_product['Name'])
            acquisition_str = datetime_to_iso_z(selected_product['acq_time'])
            tif_stamp = datetime_to_filename(selected_product['acq_time'])
            tif_output_path = os.path.join(plume_dir, f's2_minus{offset}_{tif_stamp}.tif')
            product_lock = get_product_lock(selected_product['Name'])
            dims = None
            try:
                with product_lock:
                    dims = download(access_token.get(), raw_root, plume_id, selected_product['Id'], selected_product['Name'], plume_bounds, tif_output_path)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"Download error for {selected_product['Name']}: {exc}")
                continue
            if dims is None:
                continue
            new_records[offset] = {
                'datetime': acquisition_str,
                'path': local_path,
                'height': int(dims[0]),
                'width': int(dims[1])
            }
            updated_offsets.add(offset)
            time.sleep(0.2)
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

    manager = build_proxy_manager(config)
    with proxy_manager_lock:
        proxy_manager = manager

    drive_root = config.get('google_drive_dir', DEFAULT_DRIVE_ROOT)
    if drive_root:
        base_dir = os.path.join(drive_root, 'carbonmapper_data_l2a_90360')
    else:
        base_dir = config.get('local_base_dir', DEFAULT_LOCAL_BASE_DIR)
    raw_data_dir = config.get('raw_data_dir', os.path.join(base_dir, RAW_SUBDIR_NAME))
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

    credential_pool = load_credential_pool(config)
    if len(credential_pool) == 0:
        raise RuntimeError('CDSE credentials not provided. Define cdse_username/cdse_password or indexed pairs in the config or CDSE_USERNAME/ CDSE_PASSWORD env vars.')

    token_pool: List[RefreshableAccessToken] = []
    for cred in credential_pool:
        token = RefreshableAccessToken(cred['username'], cred['password'])
        token_pool.append(token)
        thread = threading.Thread(target=refresh_variable, args=(token, ))
        thread.daemon = True
        thread.start()

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
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, row in df.iterrows():
            if not processable_mask.iloc[index]:
                continue
            base_event_dt = parsed_times.iloc[index]
            if base_event_dt is None:
                continue
            base_event_dt = base_event_dt.astimezone(timezone.utc)
            lat = row['plume_latitude']
            lon = row['plume_longitude']
            down_left = (lon - 0.01, lat - 0.01)
            up_right = (lon + 0.01, lat + 0.01)
            dot_poly = '(' + str(down_left[0]) + ' ' + str(down_left[1]) + ',' + str(down_left[0]) + ' ' + str(up_right[1]) + ',' + str(up_right[0]) + ' ' + str(up_right[1]) + ',' + str(up_right[0]) + ' ' + str(down_left[1]) + ',' + str(down_left[0]) + ' ' + str(down_left[1]) + ')'
            plume_bounds = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
            futures.append(executor.submit(
                download_task,
                index,
                row.to_dict(),
                dot_poly,
                base_event_dt,
                plume_bounds,
                OFFSETS_DAYS,
                SEARCH_WINDOW_DAYS,
                raw_data_dir,
                token_pool[len(futures) % len(token_pool)],
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
