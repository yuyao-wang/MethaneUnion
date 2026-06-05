import pandas as pd
from datetime import datetime, timedelta
import os
import requests
import time
import random

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from affine import Affine
from pyproj import Proj, transform
import numpy as np
from osgeo import gdal

def reproject_to_sentinel2(src_path, dst_path, target_res=20, target_size=(512, 512)):
    with rasterio.open(src_path) as src:
        # Translated comment
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, resolution=(target_res, target_res),
            left=src.bounds.left, bottom=src.bounds.bottom, right=src.bounds.right, top=src.bounds.top
        )
        
        # Translated comment
        profile = src.profile
        profile.update(
            transform=transform,
            crs=src.crs,
            width=int((src.bounds.right - src.bounds.left) / target_res),
            height=int((src.bounds.top - src.bounds.bottom) / target_res)
        )

        # Translated comment
        with rasterio.open(dst_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )

def resize_and_expand_to_512(src_path, dst_path, center_lat, center_lon, target_res=20):
    try:
        with rasterio.open(src_path) as src:
            if src.height == 0 or src.width == 0:
                return
            target_width, target_height = 512, 512
            
            # Translated comment
            new_data = np.zeros((1, target_height, target_width), dtype='uint8')
            
            # Translated comment
            data = np.zeros((src.count, src.height, src.width), dtype='uint8')
            for i in range(1, src.count + 1):
                data[i-1] = src.read(i, resampling=Resampling.bilinear)
            
            # Translated comment
            gray_data = np.max(data > 0, axis=0).astype('uint8')

            # Translated comment
            proj_wgs84 = Proj(init='epsg:4326')
            proj_utm = Proj(src.crs)
            center_x, center_y = transform(proj_wgs84, proj_utm, center_lon, center_lat)
            
            # Translated comment
            center_pixel_x = int((center_x - src.bounds.left) / target_res)
            center_pixel_y = int((src.bounds.top - center_y) / target_res)

            # Translated comment
            start_x = target_width // 2 - center_pixel_x
            start_y = target_height // 2 - center_pixel_y
            
            # Translated comment
            new_data[0, start_y:start_y + src.height, start_x:start_x + src.width] = gray_data
            
            # Translated comment
            new_transform = Affine(target_res, 0, center_x - (target_width // 2) * target_res,
                                0, -target_res, center_y + (target_height // 2) * target_res)
            profile = src.profile
            profile.update(
                transform=new_transform,
                width=target_width,
                height=target_height,
                count=1,
                dtype='uint8'
            )
            
            # Translated comment
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(new_data)
    except Exception as e:
        print(f'{e}')

base_dir = '/data2/yuyao/methane_emission/carbon_mapper_data_masks'
df = pd.read_csv('/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv')
for index, row in df.iterrows():
    if isinstance(row['plume_tif'], str) and len(row['plume_tif']) > 0:
        # if row['plume_id'] in ['GAO20210427t173055p0000-2', 'GAO20210429t170220p0000-1']:
        if row['plume_id'] in ['GAO20210515t145818p0000-1', 'GAO20230821t163453p0000-B', 'GAO20230821t163453p0000-C', 'GAO20240514t185646p0000-C']:
            continue
        print(f"current processing {row['plume_id']}")
        plume_dir = os.path.join(base_dir, row['plume_id'])
        input_tif = os.path.join(plume_dir, 'plume.tif')
        reprojected_tif = os.path.join(plume_dir, 'reprojected.tif')
        resized_tif = os.path.join(plume_dir, 'resized_512x512.tif')
        if not os.path.exists(input_tif):
            print(f"plume.tif not found, skipping {row['plume_id']}: {input_tif}")
            continue
        lat = row['plume_latitude']
        lon = row['plume_longitude']
        # Reproject the image to match Sentinel-2 resolution
        reproject_to_sentinel2(input_tif, reprojected_tif)

        # Resize the reprojected image to 512x512
        resize_and_expand_to_512(reprojected_tif, resized_tif, center_lat = lat, center_lon = lon, target_res=20)
