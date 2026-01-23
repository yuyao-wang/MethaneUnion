import ee
import geemap
from datetime import datetime, timedelta
import pandas as pd
import time
# 进行认证和初始化
ee.Authenticate()
ee.Initialize()

def export_sentinel2_image(lon, lat, start_date, end_date, export_path, filename_prefix):
    # 定义中心点
    point = ee.Geometry.Point(lon, lat)
    
    # 计算边界范围（512*512像素, 分辨率20米）
    half_size = 512 * 20 / 2  # 512像素，每像素20米的一半大小（10公里）
    buffer = point.buffer(half_size)
    bounds = buffer.bounds().getInfo()['coordinates'][0]

    # 将日期转换为datetime对象
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    def get_images_within_days(start, days):
        end = start + timedelta(days=days)
        return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(buffer) \
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE', True)

    current_start_date = start_date_dt
    found_images = []

    while current_start_date <= end_date_dt:
        images = get_images_within_days(current_start_date, 15)
        if images.size().getInfo() >= 3:
            found_images = images.toList(3)
            break
        current_start_date += timedelta(days=1)

    if found_images.size().getInfo() == 3:
        # 选择第一个图像作为示例
        image = ee.Image(found_images.get(0)).clip(buffer)
        
        # 选择波段 1 - 12
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        image = image.select(bands)
        
        # 定义导出任务
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename_prefix,
            folder=export_path,
            fileNamePrefix=filename_prefix,
            region=bounds,
            scale=20,
            crs='EPSG:4326'
        )
        
        # 启动导出任务
        task.start()
        print(f"Exported image for point ({lon}, {lat}) within date range {start_date} to {end_date}.")
    else:
        print(f"Not enough images for point ({lon}, {lat}) within date range {start_date} to {end_date}.")

def export_single_sentinel2_image(lon, lat, start_date, end_date, export_path, filename_prefix):
    # 定义中心点
    point = ee.Geometry.Point(lon, lat)
    
    # 计算边界范围（512*512像素, 分辨率20米）
    half_size = 512 * 20 / 2  # 512像素，每像素20米的一半大小
    bounds = point.buffer(half_size).bounds().getInfo()['coordinates'][0]
    rect_bounds = [bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]]
    square_bounds = ee.Geometry.Rectangle(rect_bounds)

    # 获取日期范围内的Sentinel-2图像集合
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(square_bounds) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .sort('NODATA_PIXEL_PERCENTAGE', True)

    # 获取符合条件的第一张图像
    image = collection.first()
    
    if image.getInfo():
        image = image.clip(square_bounds)
        
        # 选择波段 1 - 12
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        image = image.select(bands)

        image = image.reproject(crs='EPSG:4326', scale=20).clip(square_bounds)
        
        # 定义导出任务
        task = ee.batch.Export.image.toDrive(
            image=image,
            description='s2',
            folder=export_path,
            fileNamePrefix=filename_prefix,
            region=square_bounds,
            scale=20,  # 保持20米分辨率
            crs='EPSG:4326'  # 使用EPSG:4326投影
        )
        
        # 启动导出任务
        task.start()
        print(f"Exported image for point ({lon}, {lat}) within date range {start_date} to {end_date}.")
        return 1
    else:
        print(f"No suitable image found for point ({lon}, {lat}) within date range {start_date} to {end_date}.")
        return 0

def export_all_sentinel2_image(lon, lat, start_date, end_date, export_path, filename_prefix):
    # 定义中心点
    point = ee.Geometry.Point(lon, lat)
    
    # 计算边界范围（512*512像素, 分辨率20米）
    half_size = 512 * 20 / 2  # 512像素，每像素20米的一半大小
    bounds = point.buffer(half_size).bounds().getInfo()['coordinates'][0]
    rect_bounds = [bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]]
    square_bounds = ee.Geometry.Rectangle(rect_bounds)

    # 获取日期范围内的Sentinel-2图像集合
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(square_bounds) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE', True)

    images = collection.toList(collection.size())
    total_cnt = collection.size().getInfo()
    for i in range(total_cnt):
        image = ee.Image(images.get(i))
        image = image.clip(square_bounds)
            
        # 选择波段 1 - 12
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        image = image.select(bands)

        image = image.reproject(crs='EPSG:4326', scale=20).clip(square_bounds)
        
        file_name = filename_prefix + '_' + str(i)
        # 定义导出任务
        task = ee.batch.Export.image.toDrive(
            image=image,
            description='s2',
            folder=export_path,
            fileNamePrefix=file_name,
            region=square_bounds,
            scale=20,  # 保持20米分辨率
            crs='EPSG:4326'  # 使用EPSG:4326投影
        )
        
        # 启动导出任务
        task.start()
    if total_cnt > 0:
        print(f"total count {total_cnt} Exported image for point ({lon}, {lat}) within date range {start_date} to {end_date}.")
    else:
        print(f"No suitable image found for point ({lon}, {lat}) within date range {start_date} to {end_date}.")
    return total_cnt

def check_all_exist(lon, lat, date_list):
    # 定义中心点
    point = ee.Geometry.Point(lon, lat)
    
    # 计算边界范围（512*512像素, 分辨率20米）
    half_size = 512 * 20 / 2  # 512像素，每像素20米的一半大小
    bounds = point.buffer(half_size).bounds().getInfo()['coordinates'][0]
    rect_bounds = [bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]]
    square_bounds = ee.Geometry.Rectangle(rect_bounds)

    for single_date in date_list:
        start_date, end_date = single_date
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(square_bounds) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE', True)
        if collection.size().getInfo() == 0:
            return False
    return True

df = pd.read_csv('/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv')
# df['date'] = pd.to_datetime(df['datetime'])
# train_df = df[(df['date'] < '2020-07-08') | (df['date'] > '2021-08-10')] 
# test_df = df[(df['date'] >= '2020-07-08') & (df['date'] <= '2021-08-10')]
# train_df = train_df.reset_index(drop=True)
print(f'total test plume count: {len(df)}')
total_reference_count = 0
total_leak_count = 0
total_pair_count = 0

# tasks = ee.batch.Task.list()

# 遍历最近的5个任务并输出状态
# for task in tasks[:5]:
#     status = task.status()
#     print(f"Task ID: {task.id}, State: {status['state']}")
#     if status['state'] == 'FAILED':
#         print(f"Task ID: {task.id} failed.")
#         print(f"Error message: {status['error_message']}")
for index, row in df.iterrows():
    print(f'currently processing index {index}/{len(df)}')
    tasks = ee.batch.Task.list()
    in_queue_tasks = [task for task in tasks if (task.state == 'RUNNING' or task.state == 'READY')]
    print(f'current in queue task count {len(in_queue_tasks)}')
    if index % 2 == 0:
        while len(in_queue_tasks) > 2500:
            print(f'too many tasks in the queue, current task count {len(in_queue_tasks)}')
            time.sleep(100)
            tasks = ee.batch.Task.list()
            in_queue_tasks = [task for task in tasks if (task.state == 'RUNNING' or task.state == 'READY')]

    if isinstance(row['plume_tif'], str) and len(row['plume_tif']) > 0:
        t = row['datetime'] # 2019-10-19T14:52:09+00
        parsed_time = datetime.fromisoformat(t)

        lat = row['plume_latitude']
        lon = row['plume_longitude']
        # export_path = row['plume_id'] + '_3_90_365_L2A'
        export_path = "CM_S2_L2A_3_90_365"

        date_list = []
        date_list_1 = []
        date_part = parsed_time.date() + timedelta(days=-425)
        tomorrow_date = date_part + timedelta(days=90)
        d1_str = date_part.strftime("%Y-%m-%d")
        d2_str = tomorrow_date.strftime("%Y-%m-%d")
        date_list.append(((d1_str, d2_str)))
        date_list_1.append(((d1_str, d2_str)))

        date_part = parsed_time.date() + timedelta(days=-90)
        tomorrow_date = date_part + timedelta(days=60)
        d1_str = date_part.strftime("%Y-%m-%d")
        d2_str = tomorrow_date.strftime("%Y-%m-%d")
        date_list.append(((d1_str, d2_str)))
        date_list_1.append(((d1_str, d2_str)))

        date_part = parsed_time.date() + timedelta(days=-1)
        tomorrow_date = date_part + timedelta(days=3)
        d1_str = date_part.strftime("%Y-%m-%d")
        d2_str = tomorrow_date.strftime("%Y-%m-%d")
        date_list.append(((d1_str, d2_str)))

        # date_part = parsed_time.date() + timedelta(days=4)
        # tomorrow_date = date_part + timedelta(days=2)
        # d1_str = date_part.strftime("%Y-%m-%d")
        # d2_str = tomorrow_date.strftime("%Y-%m-%d")
        # date_list.append(((d1_str, d2_str)))

        all_exist = check_all_exist(lon=lon, lat=lat, date_list = date_list_1)

        if all_exist:
            print(f'org time {parsed_time} query time {date_list[0][0]}')
            year_reference_count = export_single_sentinel2_image(lon=lon, lat=lat, start_date=date_list[0][0], end_date=date_list[0][1], export_path=export_path, filename_prefix = f"{row['plume_id']}_reference_year")

            print(f'org time {parsed_time} query time {date_list[1][0]}')
            month_reference_count = export_single_sentinel2_image(lon=lon, lat=lat, start_date=date_list[1][0], end_date=date_list[1][1], export_path=export_path, filename_prefix = f"{row['plume_id']}_reference_month")

            print(f'org time {parsed_time} query time {date_list[2][0]}')
            leak_count = export_all_sentinel2_image(lon=lon, lat=lat, start_date=date_list[2][0], end_date=date_list[2][1], export_path=export_path, filename_prefix = f"{row['plume_id']}_leak")

            # print(f'org time {parsed_time} query time {date_list[2][0]}')
            # export_single_sentinel2_image(lon=lon, lat=lat, start_date=date_list[2][0], end_date=date_list[2][1], export_path=export_path, filename_prefix = 's2_+5')
            print(f'currently downloaded year_reference_count {year_reference_count} month_reference_count {month_reference_count} leak_count {leak_count} pair_count {year_reference_count * month_reference_count * leak_count}')
            total_reference_count += year_reference_count * month_reference_count
            total_leak_count += leak_count
            total_pair_count += year_reference_count * month_reference_count * leak_count
            print(f'export path {export_path} downloaded total_reference_count {total_reference_count} total_leak_count {total_leak_count} total_pair_count {total_pair_count} in total')
            # if total_reference_count > 0 or leak_count > 0:
            #     exit(-1)
        else:
            print(f'location lon {lon} lat {lat} does not have all data')