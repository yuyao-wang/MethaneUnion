import json
import requests
import pandas as pd
import os
import zipfile
import sys
from datetime import datetime, timedelta
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.utils import parse_args, load_config
from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd

USER = 'yuyao16@ualberta.ca'
PASSWORD = 'finhah-3zihty-seHmuf'

S5P_COLLECTION_NAME = 'SENTINEL-5P'
# Update this to the exact Sentinel-5P productType you need (e.g., CH4 product).
S5P_PRODUCT_TYPE = 'L2__CH4___'
base_dir = '/data2/yuyao/methane_emission/carbon_mapper_data/CM_S5P_L2'

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

# access_token = get_access_token(USER, PASSWORD)

def download(access_token, output_dir, plume_id, product_id, name):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    if os.path.exists(output_path):
        print(f'output path {output_path} already exists')
        return

    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {access_token}"}

    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url, headers=headers, stream=True)
    try:
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            if zipfile.is_zipfile(output_path):
                extract_dir = os.path.join(output_dir, name + '_extracted')
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.endswith('.nc'):
                            zip_ref.extract(member, extract_dir)
        else:
            print(f'request failed {response.status_code}')
            return
    finally:
        response.close()
        session.close()

    plume_dir = os.path.join(base_dir, plume_id)
    os.makedirs(plume_dir, exist_ok=True)
    stub_file = os.path.join(plume_dir, 'download_stub_s5p.txt')
    with open(stub_file, 'w') as f:
        pass

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

def parse_iso_datetime(value):
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def select_closest_product(products, target_dt):
    if len(products) == 0:
        return None
    return min(products, key=lambda p: abs((p['acq_time'] - target_dt).total_seconds()))


def download_task(plume_id, poly, date_param, plume_bounds, access_token, anchor_dt):
    
    try:
        next_link = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{S5P_COLLECTION_NAME}' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{S5P_PRODUCT_TYPE}') and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON({poly})') and ContentDate/Start gt {date_param[0]}T00:00:00.000Z and ContentDate/Start lt {date_param[1]}T00:00:00.000Z&$top=1000"
        products = []
        while len(next_link) > 0:
            print(f'current link {next_link}')
            
            resp = requests.get(next_link).json()
            # print(json)
            if resp is None or 'value' not in resp:
                print(f'bad response {resp}')
                break
            df = pd.DataFrame.from_dict(resp['value'])
            if '@odata.nextLink' in resp:
                next_link = resp['@odata.nextLink']
            else:
                next_link = ""
            print(f'next link {next_link}')

            print(f'In total {len(df)} files need to download')
            for index, row in df.iterrows():
                start_time_str = row.get('ContentDate', {}).get('Start')
                acq_time = parse_iso_datetime(start_time_str)
                if acq_time is None:
                    continue
                products.append({
                    'Id': row['Id'],
                    'Name': row['Name'],
                    'acq_time': acq_time,
                })
        selected_product = select_closest_product(products, anchor_dt)
        if selected_product is None:
            print(f'No S5P for plume {plume_id}')
            return

        current_time = datetime.now()
        print(
            f'pid: {os.getpid()} current time: {current_time} '
            f'closest product id: {selected_product["Id"]}, file name: {selected_product["Name"]}'
        )
        out_dir = config['raw_data_dir']
        print(f'out dir is {out_dir}')
        os.makedirs(out_dir, exist_ok=True)
        download(
            access_token.get(),
            out_dir,
            plume_id,
            selected_product['Id'],
            selected_product['Name']
        )
        time.sleep(0.2)
        plume_dir = os.path.join(base_dir, plume_id)
        os.makedirs(plume_dir, exist_ok=True)
        stub_file = os.path.join(plume_dir, 'download_stub_pre.txt')
        with open(stub_file, 'w') as f:
            pass
    except Exception as e:
        print(f"An unknown error occurred: {e}")

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    access_token = RefreshableAccessToken()
    thread = threading.Thread(target=refresh_variable, args = (access_token, ))
    thread.start()

    df = pd.read_csv('/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv')
    total_len = len(df)
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, row in df.iterrows():
            print(f'currently processing index {index}/{total_len}')
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
                stub_file = os.path.join(plume_dir, 'download_stub_s5p.txt')
                if os.path.exists(stub_file):
                    continue
                # stub_file = os.path.join(plume_dir, 'download_stub.txt')
                # if os.path.exists(stub_file):
                #     continue
                # print(f'date param {date_param}')
                futures.append(
                    executor.submit(
                        download_task,
                        row['plume_id'],
                        dot_poly,
                        date_param,
                        [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01],
                        access_token,
                        parsed_time
                    )
                )
            results = [future.result() for future in futures]
    print("All tasks completed.")
