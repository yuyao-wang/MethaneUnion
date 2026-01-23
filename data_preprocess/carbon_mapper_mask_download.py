import pandas as pd
from datetime import datetime, timedelta
import os
import requests
import time
import random

base_dir = '/data2/yuyao/methane_emission/carbon_mapper_data_masks'
df = pd.read_csv('/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv')

def download_tif(url, path):
    print(f'now downloading file {url}')
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            file.write(response.content)
        print(f'文件下载并保存成功')
    else:
        print(f'无法下载{url}，状态码:', response.status_code)

for index, row in df.iterrows():
    if isinstance(row['plume_tif'], str) and len(row['plume_tif']) > 0:
        t = row['datetime'] # 2019-10-19T14:52:09+00
        parsed_time = datetime.fromisoformat(t)
        date_part = parsed_time.date()
        tomorrow_date = date_part + timedelta(days=1)

        # d_url = get_download_url(row['plume_latitude'], row['plume_longitude'], date_part.strftime("%Y-%m-%d"), tomorrow_date.strftime("%Y-%m-%d"))
        plume_dir = os.path.join(base_dir, row['plume_id'])
        os.makedirs(plume_dir, exist_ok=True)
        plume_file = os.path.join(plume_dir, 'plume.tif')
        if os.path.exists(plume_file):
            print(f'file {plume_file} already exists, skip...')
            continue
        download_tif(row['plume_tif'], plume_file)
        sleep_time = random.uniform(0.2, 0.8)
        time.sleep(sleep_time)

    if isinstance(row['con_tif'], str) and len(row['con_tif']) > 0:
        t = row['datetime'] # 2019-10-19T14:52:09+00
        parsed_time = datetime.fromisoformat(t)
        date_part = parsed_time.date()
        tomorrow_date = date_part + timedelta(days=1)

        # d_url = get_download_url(row['plume_latitude'], row['plume_longitude'], date_part.strftime("%Y-%m-%d"), tomorrow_date.strftime("%Y-%m-%d"))
        plume_dir = os.path.join(base_dir, row['plume_id'])
        os.makedirs(plume_dir, exist_ok=True)
        plume_file = os.path.join(plume_dir, 'con.tif')
        if os.path.exists(plume_file):
            print(f'file {plume_file} already exists, skip...')
            continue
        download_tif(row['con_tif'], plume_file)
        sleep_time = random.uniform(0.2, 0.8)
        time.sleep(sleep_time)