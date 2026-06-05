import json
import requests
import pandas as pd
import os
import zipfile
import shutil
import sys
from datetime import datetime, timedelta, timezone
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.utils import parse_args, load_config
from concurrent.futures import ThreadPoolExecutor
import threading

import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import re
import pandas as pd
import tifffile


pattern = re.compile(r".*B[0-9A-Za-z]+_20m\.jp2$")
type_pattern = r".*B([0-9A-Za-z]+)_20m\.jp2$"
base_dir = '/data2/yuyao/methane_emission/carbonmapper_data_l2a'
PLUME_COMPLETION_MARKER = 'download_stub_pre.txt'
product_lock_map = {}
product_lock_map_lock = threading.Lock()

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

def get_product_lock(product_name):
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

USER = "yuyao16@ualberta.ca"
PASSWORD = "finhah-3zihty-seHmuf"
# access_token = get_access_token(USER, PASSWORD)

def latlon_to_pixel(lat, lon, dataset):
    """Convert latitude and longitude to pixel coordinates."""
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return ~dataset.transform * (x, y)

def parse_a_file(file_path, plume_bounds):
    # Read the Sentinel-2 JP2 file
    with rasterio.open(file_path) as dataset:
        # define the latitude/longitude bounds of the region of interest
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
        response = session.get(url, headers=headers, stream=True)

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
            resp = requests.get(next_link)
            resp.raise_for_status()
            payload = resp.json()
            print(f'payload received {len(payload.get("value", []))} products')
        except Exception as exc:
            print(f'Failed to query catalogue: {exc}')
            break

        values = payload.get('value', [])
        for product in values:
            # Translated comment
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
    def __init__(self) -> None:
        self.value = get_access_token(USER, PASSWORD)
        self.lock = threading.Lock()
    def update(self):
        with self.lock:
            self.value = get_access_token(USER, PASSWORD)
    def get(self):
        with self.lock:
            return self.value


def refresh_variable(variable):
    while True:
        print(f'current time {datetime.now()} refresh token')
        variable.update()
        time.sleep(300)

def download_task(row_index, row_data, poly, event_dt, plume_bounds, access_token, progress_tracker):
    plume_id = str(row_data.get('plume_id', 'unknown'))
    plume_dir = os.path.join(base_dir, plume_id)
    plume_marker_file = os.path.join(plume_dir, PLUME_COMPLETION_MARKER)
    try:
        if os.path.exists(plume_marker_file):
            print(f'skipping plume {plume_id}; completion marker found at {plume_marker_file}')
            return {'index': row_index, 'selected_products': [], 'has_same_day': 0}
        window_start = event_dt - timedelta(days=7)
        window_end = event_dt + timedelta(days=7)
        products = fetch_products(poly, datetime_to_query_string(window_start), datetime_to_query_string(window_end))
        selected_products = select_products(products, event_dt)
        has_same_day = 1 if any(product['acq_time'].date() == event_dt.date() for product in selected_products) else 0
        recorded_products = []
        out_dir = config['raw_data_dir']
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(plume_dir, exist_ok=True)
        for product in selected_products:
            local_path = os.path.join(out_dir, product['Name'])
            acquisition_str = datetime_to_iso_z(product['acq_time'])
            tif_stamp = datetime_to_filename(product['acq_time'])
            tif_output_path = os.path.join(plume_dir, f's2_{tif_stamp}.tif')
            product_lock = get_product_lock(product['Name'])
            try:
                with product_lock:
                    dims = download(access_token.get(), out_dir, plume_id, product['Id'], product['Name'], plume_bounds, tif_output_path)
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
            time.sleep(0.2)
        if len(selected_products) > 0:
            os.makedirs(plume_dir, exist_ok=True)
            with open(plume_marker_file, 'w') as f:
                f.write(datetime_to_iso_z(datetime.now(timezone.utc)))
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
    access_token = RefreshableAccessToken()
    thread = threading.Thread(target=refresh_variable, args = (access_token, ))
    thread.start()

    merged_csv_path = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv'
    output_csv_path = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2.csv'
    df = pd.read_csv(merged_csv_path)
    new_columns = []
    for i in range(1, 4):
        new_columns.extend([
            f's2_{i}_datetime', f's2_{i}_path',
            f's2_{i}_height', f's2_{i}_width'
        ])
    for col in new_columns:
        df[col] = ""
    df['has_same_day_s2'] = 0
    total_len = len(df)
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
            futures.append(executor.submit(
                download_task,
                index,
                row.to_dict(),
                dot_poly,
                parsed_time,
                plume_bounds,
                access_token,
                progress_tracker
            ))
    results = [future.result() for future in futures]
    for res in results:
        idx = res.get('index')
        selected_products = res.get('selected_products', [])
        has_same_day = res.get('has_same_day', 0)
        df.at[idx, 'has_same_day_s2'] = has_same_day
        for i in range(3):
            col_dt = f's2_{i+1}_datetime'
            col_path = f's2_{i+1}_path'
            col_height = f's2_{i+1}_height'
            col_width = f's2_{i+1}_width'
            if i < len(selected_products):
                df.at[idx, col_dt] = selected_products[i].get('datetime', '')
                df.at[idx, col_path] = selected_products[i].get('path', '')
                df.at[idx, col_height] = selected_products[i].get('height', '')
                df.at[idx, col_width] = selected_products[i].get('width', '')
    df.to_csv(output_csv_path, index=False)
    total_elapsed = time.time() - overall_start_time
    print(f"All tasks completed in {total_elapsed/60:.2f} minutes.")
    print("All tasks completed.")
