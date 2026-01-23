import json
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import rasterio
import requests
import tifffile
from pyproj import Transformer
from rasterio.windows import Window

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utils import parse_args, load_config

pattern = re.compile(r".*B[0-9A-Za-z]+_20m\.jp2$")
type_pattern = r".*B([0-9A-Za-z]+)_20m\.jp2$"
DEFAULT_PROCESSED_BASE_DIR = '/data2/yuyao/methane_emission/carbonmapper_data_s2_l2a'
DEFAULT_SUPPL_RAW_DIR = '/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_S2_suppliment'
DEFAULT_INPUT_MERGED_CSV = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2.csv'
DEFAULT_SUPPL_OUTPUT_CSV = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2_suppliment.csv'
base_dir = DEFAULT_PROCESSED_BASE_DIR
raw_data_dir = DEFAULT_SUPPL_RAW_DIR
product_lock_map = {}
product_lock_map_lock = threading.Lock()
BACKOFF_STATUS_CODE = 429
BACKOFF_BASE_SECONDS = 30
BACKOFF_MAX_SECONDS = 120
BACKOFF_MAX_RETRIES = 5
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
                url = self._proxies[self._index]
                self._index = (self._index + 1) % n
                if self._cooldowns.get(url, 0) <= now:
                    return url
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

def parse_iso_datetime(value):
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None

def datetime_to_query_string(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

def datetime_to_iso_z(dt):
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

def datetime_to_filename(dt):
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

def get_product_lock(product_name):
    with product_lock_map_lock:
        lock = product_lock_map.get(product_name)
        if lock is None:
            lock = threading.Lock()
            product_lock_map[product_name] = lock
        return lock


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
    return pool


def build_proxy_manager(config: Dict) -> Optional[ProxyManager]:
    pool_cfg = config.get('proxy_pool')
    if not isinstance(pool_cfg, dict):
        return None
    if not pool_cfg.get('enabled', True):
        return None
    entries = pool_cfg.get('entries') or []
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

# get authentication
def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
        }
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Access token creation failed. Reponse from the server was: {r.json()}"
            )
    return r.json()["access_token"]

def refresh_access_token(token) -> str:
    data = {
        "client_id": "cdse-public",
        "refresh_token": token,
        "grant_type": "refresh_token",
        }
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
        )
        r.raise_for_status()
        print(r.json())
    except Exception as e:
        raise Exception(
            f"Access token refresh failed. Reponse from the server was: {r.json()}"
            )
    return r.json()["access_token"]

def latlon_to_pixel(lat, lon, dataset):
    """Convert latitude and longitude to pixel coordinates."""
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return ~dataset.transform * (x, y)

def parse_a_file(file_path, plume_bounds):
    # 读取 Sentinel-2 JP2 文件
    with rasterio.open(file_path) as dataset:
        # 定义感兴趣区域的经纬度边界
        # plume_bounds = [-103.54983976823348, 32.06765206888606, -103.54028239690392, 32.07575083506975]
        
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

def download(access_token, output_dir, plume_id, product_id, name, plume_bounds, tif_output_path):
    output_path = os.path.join(output_dir, name)
    marker_file = os.path.join(output_path, '.download_complete')
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
                    directories_to_extract = [file for file in all_files if folder_pattern in file and (subfolder_pattern in file or QI_data_pattern in file)]
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
                        
                    # Extract the important file
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
    # np.save(np_file_path, img_output)
    if img_output is None:
        print(f'no valid JP2 data found for product {name}')
        return None
    os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
    print(f'final tif file output path {tif_output_path}')
    tifffile.imwrite(tif_output_path, img_output)
    return current_shape

def fetch_products(poly, start_ts, end_ts):
    products = []
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
            print(f'payload received {len(payload.get("value", []))} products')
        except Exception as exc:
            print(f'Failed to query catalogue: {exc}')
            break

        values = payload.get('value', [])
        for product in values:
            # ✅ 正确读取 ContentDate.Start
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

# (-10 42,-10 12,70 12,70 42,-10 42)
def get_geo_filter(config):
    up_left = config['region_up_left']
    down_right = config['region_down_right']
    poly = '(' + str(up_left[0]) + ' ' + str(up_left[1]) + ',' + str(up_left[0]) + ' ' + str(down_right[1]) + ',' + str(down_right[0]) + ' ' + str(down_right[1]) + ',' + str(down_right[0]) + ' ' + str(up_left[1]) + ',' + str(up_left[0]) + ' ' + str(up_left[1]) + ')'
    return poly

class RefreshableAccessToken:
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.value = get_access_token(username, password)
        self.lock = threading.Lock()

    def update(self):
        with self.lock:
            self.value = get_access_token(self.username, self.password)

    def get(self):
        with self.lock:
            return self.value


def refresh_variable(variable):
    while True:
        print(f'current time {datetime.now()} refresh token')
        variable.update()
        time.sleep(300)

def download_task(row_index, row_data, poly, event_dt, plume_bounds, access_token, progress_tracker, existing_datetimes: Set[str]):
    plume_id = str(row_data.get('plume_id', 'unknown'))
    plume_dir = os.path.join(base_dir, plume_id)
    os.makedirs(plume_dir, exist_ok=True)
    try:
        window_start = event_dt - timedelta(days=7)
        window_end = event_dt + timedelta(days=7)
        products = fetch_products(poly, datetime_to_query_string(window_start), datetime_to_query_string(window_end))
        selected_products = select_products(products, event_dt)
        candidate_iso_times = [datetime_to_iso_z(product['acq_time']) for product in selected_products]
        if candidate_iso_times:
            print(f'plume {plume_id} candidate cloud<20% acquisitions: {candidate_iso_times}')
        has_same_day = 1 if any(product['acq_time'].date() == event_dt.date() for product in selected_products) else 0
        recorded_products = []
        os.makedirs(raw_data_dir, exist_ok=True)
        for product, acquisition_str in zip(selected_products, candidate_iso_times):
            if acquisition_str in existing_datetimes:
                print(f'plume {plume_id} already recorded acquisition {acquisition_str}; skipping download')
                continue
            local_path = os.path.join(raw_data_dir, product['Name'])
            tif_stamp = datetime_to_filename(product['acq_time'])
            tif_output_path = os.path.join(plume_dir, f's2_{tif_stamp}.tif')
            product_lock = get_product_lock(product['Name'])
            try:
                with product_lock:
                    dims = download(access_token.get(), raw_data_dir, plume_id, product['Id'], product['Name'], plume_bounds, tif_output_path)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"An error occurred while downloading {product['Name']}: {exc}")
                continue
            if dims is None:
                continue
            recorded_products.append({
                'datetime': acquisition_str,
                'path': local_path,
                'height': int(dims[0]),
                'width': int(dims[1])
            })
            existing_datetimes.add(acquisition_str)
            time.sleep(0.2)
        return {'index': row_index, 'selected_products': recorded_products, 'has_same_day': has_same_day}
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"An unknown error occurred while processing {row_data.get('plume_id')}: {exc}")
        return {'index': row_index, 'selected_products': [], 'has_same_day': 0}
    finally:
        update_global_progress(progress_tracker)

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    manager = build_proxy_manager(config)
    with proxy_manager_lock:
        proxy_manager = manager

    credential_pool = load_credential_pool(config)
    if not credential_pool:
        raise RuntimeError("No CDSE credentials configured for supplementary downloader.")
    token_pool = []
    for cred in credential_pool:
        token = RefreshableAccessToken(cred['username'], cred['password'])
        token_pool.append(token)
        thread = threading.Thread(target=refresh_variable, args=(token,))
        thread.daemon = True
        thread.start()

    base_dir = config.get('local_base_dir_s2', DEFAULT_PROCESSED_BASE_DIR)
    raw_data_dir = config.get('raw_data_dir_suppliment', DEFAULT_SUPPL_RAW_DIR)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

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
    with ThreadPoolExecutor(max_workers=12) as executor:
        for index, row in df.iterrows():
            if not processable_mask.iloc[index]:
                continue
            parsed_time = parsed_times.iloc[index]
            parsed_time = parsed_time.astimezone(timezone.utc)
            lat = row['plume_latitude']
            lon = row['plume_longitude']

            down_left = (lon - 0.01, lat - 0.01)
            up_right = (lon + 0.01, lat + 0.01)

            dot_poly = '(' + str(down_left[0]) + ' ' + str(down_left[1]) + ',' + str(down_left[0]) + ' ' + str(up_right[1]) + ',' + str(up_right[0]) + ' ' + str(up_right[1]) + ',' + str(up_right[0]) + ' ' + str(down_left[1]) + ',' + str(down_left[0]) + ' ' + str(down_left[1]) + ')'
            plume_bounds = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
            token = token_pool[len(futures) % len(token_pool)]
            futures.append(executor.submit(
                download_task,
                index,
                row.to_dict(),
                dot_poly,
                parsed_time,
                plume_bounds,
                token,
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
