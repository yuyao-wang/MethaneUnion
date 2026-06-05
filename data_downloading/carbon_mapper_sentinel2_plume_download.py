import json
import requests
import pandas as pd
import os
import zipfile
import shutil
import sys
from datetime import datetime, timedelta
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.utils import parse_args, load_config
from concurrent.futures import ThreadPoolExecutor
import threading

import rasterio
from rasterio.windows import Window
import pyproj
from pyproj import Proj, transform
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import re
import pandas as pd
import tifffile

USER = 'yuyao16@ualberta.ca'
PASSWORD = 'finhah-3zihty-seHmuf'

def debug(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][pid:{os.getpid()}][tid:{threading.get_ident()}] {msg}", flush=True)

pattern = re.compile(r".*B[0-9A-Za-z]+_20m\.jp2$")
type_pattern = r".*B([0-9A-Za-z]+)_20m\.jp2$"
base_dir = '/data2/yuyao/methane_emission/carbon_mapper_data/CM_S2_L2A'

# get authentication
def get_access_token(username: str, password: str) -> str:
    debug("requesting access token")
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
    debug("refreshing access token")
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

# access_token = get_access_token(USER, PASSWORD)

def latlon_to_pixel(lat, lon, dataset):
    """Convert latitude and longitude to pixel coordinates."""
    p1 = Proj(dataset.crs)
    p2 = Proj(proj='latlong', datum='WGS84')
    x, y = transform(p2, p1, lon, lat)
    return ~dataset.transform * (x, y)

def parse_a_file(file_path, plume_bounds):
    debug(f"parsing jp2 file {file_path}")
    # Read the Sentinel-2 JP2 file
    with rasterio.open(file_path) as dataset:
        # define the latitude/longitude bounds of the region of interest
        # plume_bounds = [-103.54983976823348, 32.06765206888606, -103.54028239690392, 32.07575083506975]
        
        top_left = latlon_to_pixel(plume_bounds[3], plume_bounds[0], dataset)
        bottom_right = latlon_to_pixel(plume_bounds[1], plume_bounds[2], dataset)
        
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2
        
        window_size = 512
        window = Window(center_x - window_size//2, center_y - window_size//2, window_size, window_size)
        
        clipped = dataset.read(window=window)
        return clipped

def download(access_token, output_dir, plume_id, product_id, name, plume_bounds):
    output_path = os.path.join(output_dir, name)
    if not os.path.exists(output_path):
        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        
        headers = {"Authorization": f"Bearer {access_token}"}

        session = requests.Session()
        session.headers.update(headers)
        debug(f"downloading product {product_id} name={name}")
        response = session.get(url, headers=headers, stream=True)
        debug(f"download response {response.status_code} for {product_id}")

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
                debug(f"zip saved {zip_output_path} size={os.path.getsize(zip_output_path)}")
                
                os.makedirs(output_path)
                with zipfile.ZipFile(zip_output_path, 'r') as zip_ref:
                    all_files = zip_ref.namelist()
                    directories_to_extract = [file for file in all_files if folder_pattern in file and (subfolder_pattern in file or QI_data_pattern in file)]
                    important_files = [file for file in all_files if important_file in file]
                    debug(f"zip entries total={len(all_files)} to_extract={len(directories_to_extract)} important={len(important_files)}")

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
                    debug("zip downloaded; exiting early due to exit(0) in code path")
                    exit(0)
                    # os.remove(zip_output_path)
            else:
                print(f'request failed {response.status_code}')
                return
        finally:
            response.close()
            session.close()
    else:
        print(f'output path {output_path} already exists')
    np_file_path = os.path.join(output_dir, name + '_' +  '_'.join([str(x) for x in plume_bounds]) + '.npy')
    if os.path.exists(np_file_path):
        print(f'npy file {np_file_path} already exists')
        return
    img_output = np.zeros((12, 512, 512))
    jp2_dir = Path(output_path)
    for file_path in jp2_dir.rglob('*.jp2'):
        if os.path.isfile(file_path) and pattern.match(str(file_path)):
            spectrum_type_str = re.search(type_pattern, file_path.name).group(1)
            spectrum_type = 8 if spectrum_type_str == '8A' else int(spectrum_type_str)
            debug(f"reading spectrum {spectrum_type_str} -> band {spectrum_type}")
            img_output[spectrum_type - 1] = parse_a_file(file_path, plume_bounds)
    # np.save(np_file_path, img_output)
    plume_dir = os.path.join(base_dir, plume_id)
    os.makedirs(plume_dir, exist_ok=True)
    tif_file = os.path.join(plume_dir, 's2.tif')
    print(f'final tif file output path {tif_file}')
    tifffile.imwrite(tif_file, img_output)

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
        debug("refresh thread: updating access token")
        variable.update()
        time.sleep(300)

def download_task(plume_id, poly, date_param, plume_bounds, access_token):
    
    try:
        token_clock = time.time()
        debug(f"download_task start plume_id={plume_id} date_param={date_param} poly={poly}")
        next_link = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 20.00) and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON({poly})') and ContentDate/Start gt {date_param[0]}T00:00:00.000Z and ContentDate/Start lt {date_param[1]}T00:00:00.000Z&$top=1000"
        while len(next_link) > 0:
            debug(f'current link {next_link}')
            
            resp_raw = requests.get(next_link)
            debug(f"catalog response {resp_raw.status_code} url={resp_raw.url}")
            resp = resp_raw.json()
            # print(json)
            if resp is None or 'value' not in resp:
                print(f'bad response {resp}')
                break
            df = pd.DataFrame.from_dict(resp['value'])
            if '@odata.nextLink' in resp:
                next_link = resp['@odata.nextLink']
            else:
                next_link = ""
            debug(f'next link {next_link}')

            print(f'In total {len(df)} files need to download')
            suc_cnt = 0
            for index, row in df.iterrows():
                current_time = datetime.now()
                debug(f'processing index={index} product id={row["Id"]} name={row["Name"]}')
                try:
                    out_dir = config['raw_data_dir']
                    debug(f'out dir is {out_dir}')
                    os.makedirs(out_dir, exist_ok=True)
                    download(access_token.get(), out_dir, plume_id, row['Id'], row['Name'], plume_bounds)
                    # if (time.time() - token_clock) > 500:
                    #     access_token = get_access_token(USER, PASSWORD)
                    #     token_clock = time.time()
                    #     print(f'current time {datetime.now()} refresh token')
                    suc_cnt += 1
                    time.sleep(0.2)
                except KeyboardInterrupt as ki:
                    print(f"User terminated")
                    break
                except Exception as e:
                    debug(f"An error occurred: {e}")
                except:
                    debug("An unknown error occurred")
            print(f'In total {len(df)} files need to download, {suc_cnt} files succeed')
            plume_dir = os.path.join(base_dir, plume_id)
            os.makedirs(plume_dir, exist_ok=True)
            stub_file = os.path.join(plume_dir, 'download_stub_pre.txt')
            with open(stub_file, 'w') as f:
                pass
    except KeyboardInterrupt:
        print("User terminated")
        raise
    except Exception as e:
        print(f"An unknown error occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    debug("script start")
    args = parse_args()
    config = load_config(args.config)
    debug(f"config loaded from {args.config}")
    access_token = RefreshableAccessToken()
    thread = threading.Thread(target=refresh_variable, args = (access_token, ))
    thread.start()
    debug("refresh thread started")

    df = pd.read_csv('/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv')
    total_len = len(df)
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, row in df.iterrows():
            debug(f'currently processing index {index}/{total_len}')
            # if index < 10132:
            #     continue
            if isinstance(row['plume_tif'], str) and len(row['plume_tif']) > 0:
                t = row['datetime'] # 2019-10-19T14:52:09+00
                parsed_time = datetime.fromisoformat(t)
                date_part = parsed_time.date()
                tomorrow_date = date_part + timedelta(days=1)
                d1_str = date_part.strftime("%Y-%m-%d")
                d2_str = tomorrow_date.strftime("%Y-%m-%d")
                print(f'org time {parsed_time} query time {d1_str} ')
                lat = row['plume_latitude']
                lon = row['plume_longitude']

                down_left = (lon - 0.01, lat - 0.01)
                up_right = (lon + 0.01, lat + 0.01)

                dot_poly = '(' + str(down_left[0]) + ' ' + str(down_left[1]) + ',' + str(down_left[0]) + ' ' + str(up_right[1]) + ',' + str(up_right[0]) + ' ' + str(up_right[1]) + ',' + str(up_right[0]) + ' ' + str(down_left[1]) + ',' + str(down_left[0]) + ' ' + str(down_left[1]) + ')'
            
                date_param = (d1_str, d2_str)
                plume_dir = os.path.join(base_dir, row['plume_id'])
                tif_file = os.path.join(plume_dir, 's2.tif')
                if os.path.exists(tif_file):
                    continue
                # stub_file = os.path.join(plume_dir, 'download_stub.txt')
                # if os.path.exists(stub_file):
                #     continue
                # print(f'date param {date_param}')
                futures.append(executor.submit(download_task, row['plume_id'], dot_poly, date_param, [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01], access_token))
            results = [future.result() for future in futures]
    print("All tasks completed.")
