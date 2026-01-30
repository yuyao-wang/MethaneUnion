import os
import gc
import time
import shutil
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import torch
import torch.nn.functional as F
from pathlib import Path
from pyproj import Transformer
import multiprocessing as mp
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor

# ==================== 用户配置区 ====================
CSV_PATH = "./merged_with_emit_tag.csv"
UPDATED_CSV_PATH = "./merged_with_simulated_path.csv"
SRF_CSV = "./landsat9_oli_srf.csv"

# 远程磁盘路径 - 确保这里路径最后有斜杠，或者使用 Path
REMOTE_EMIT_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT")
LOCAL_BUFFER_DIR = Path("./local_emit_buffer") 
OUTPUT_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_l89_L2SR/EMIT_simulated_landsat9_60resolution")

CHIP_SIZE_PX = 512
SCALE_M = 60
NUM_GPUS = 2
BUFFER_SIZE = 6 
# ===================================================

def get_utm_crs(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"

def get_spectral_matrix(emit_waves, srf_df):
    matrix = np.zeros((len(emit_waves), 7))
    for i in range(1, 8):
        w = np.interp(emit_waves, srf_df["wavelength"], srf_df[f"b{i}"], left=0, right=0)
        matrix[:, i-1] = w / (w.sum() + 1e-12)
    return torch.tensor(matrix, dtype=torch.float32)

# ==================== 生产者：多线程数据拉取 ====================
def download_task(granule_id, buffer_dict):
    """单个文件的下载任务"""
    try:
        # 搜索远程文件
        remote_files = list(REMOTE_EMIT_DIR.glob(f"*{granule_id}*.nc"))
        if not remote_files:
            short_id = granule_id.split('.')[0]
            remote_files = list(REMOTE_EMIT_DIR.glob(f"*{short_id}*.nc"))

        if remote_files:
            remote_path = remote_files[0]
            local_path = LOCAL_BUFFER_DIR / remote_path.name
            
            # 如果本地没写完，执行拷贝
            if not local_path.exists():
                # 使用临时文件名，防止 GPU 进程读取到一个只写了一半的文件
                tmp_path = local_path.with_suffix('.tmp')
                shutil.copy2(remote_path, tmp_path)
                tmp_path.rename(local_path)
            
            buffer_dict[granule_id] = str(local_path)
            return True
        else:
            buffer_dict[granule_id] = "NOT_FOUND"
            return False
    except Exception as e:
        print(f"\n[Thread Error] {granule_id}: {e}", flush=True)
        return False

def data_prefetcher(granule_queue, buffer_dict):
    """多线程分发器"""
    # 这里的 max_workers 建议设为 3-5，太多可能会拖慢远程磁盘响应
    print(f"[Prefetcher] Multi-threaded loader started", flush=True)
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        while True:
            # 只有当 Buffer 有空位时才分发新下载任务
            if len(buffer_dict) < BUFFER_SIZE:
                try:
                    granule_id = granule_queue.get(timeout=5)
                    if granule_id is None: break
                    
                    # 提交异步下载任务
                    executor.submit(download_task, granule_id, buffer_dict)
                except:
                    continue # 队列暂时空了
            else:
                time.sleep(1) # Buffer 满了，歇一会

# ==================== 消费者：GPU 计算函数 ====================
def process_worker(gpu_id, granule_list, target_df, srf_df, return_dict, buffer_dict):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    local_results = {}
    spec_mat = None 
    pbar = tqdm(granule_list, desc=f"GPU {gpu_id}", position=gpu_id, leave=True)

    for granule_id in pbar:
        # 等待 Prefetcher 准备好数据
        wait_count = 0
        while granule_id not in buffer_dict:
            time.sleep(1)
            wait_count += 1
            if wait_count > 60: break # 等太久了

        local_path_str = buffer_dict.get(granule_id)
        if not local_path_str or local_path_str == "NOT_FOUND":
            continue
        
        rfl_path = Path(local_path_str)
        group = target_df[target_df['emit_granule_id'] == granule_id]

        try:
            with xr.open_dataset(rfl_path, group='location') as ds_loc:
                full_lats, full_lons = ds_loc['lat'].values, ds_loc['lon'].values
            
            if spec_mat is None:
                with xr.open_dataset(rfl_path, group='sensor_band_parameters') as dsb:
                    spec_mat = get_spectral_matrix(dsb['wavelengths'].values, srf_df).to(device)

            with xr.open_dataset(rfl_path) as ds_rfl:
                reflectance = ds_rfl['reflectance']

                for _, row in group.iterrows():
                    plume_id = row['plume_id']
                    out_tif = OUTPUT_DIR / f"{plume_id}_sim_L9.tif"
                    if out_tif.exists():
                        local_results[plume_id] = str(out_tif.absolute())
                        continue

                    # 1. 裁剪
                    p_lat, p_lon = row['plume_latitude'], row['plume_longitude']
                    mask = (full_lats > p_lat - 0.2) & (full_lats < p_lat + 0.2) & \
                           (full_lons > p_lon - 0.2) & (full_lons < p_lon + 0.2)
                    y_idx, x_idx = np.where(mask)
                    if len(y_idx) == 0: continue
                    y_m, y_M, x_m, x_M = y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max()

                    # 2. 计算 (仅读取切片)
                    rfl_slice = reflectance[y_m:y_M, x_m:x_M, :].values
                    crop_rfl = torch.from_numpy(np.nan_to_num(rfl_slice, 0)).to(device)
                    simulated = torch.matmul(crop_rfl, spec_mat).permute(2, 0, 1).unsqueeze(0)

                    # 3. 坐标映射
                    utm_epsg = get_utm_crs(p_lat, p_lon)
                    to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
                    to_wgs = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
                    cx, cy = to_utm.transform(p_lon, p_lat)
                    offset = (CHIP_SIZE_PX / 2 - 0.5) * SCALE_M
                    tx = (cx - offset) + np.arange(CHIP_SIZE_PX) * SCALE_M
                    ty = (cy + offset) - np.arange(CHIP_SIZE_PX) * SCALE_M
                    mx, my = np.meshgrid(tx, ty)
                    t_lon, t_lat = to_wgs.transform(mx, my)

                    # 4. Grid Sample
                    lat_c, lon_c = full_lats[y_m:y_M, x_m:x_M], full_lons[y_m:y_M, x_m:x_M]
                    grid_x = 2.0 * (torch.from_numpy(t_lon).to(device) - lon_c.min()) / (lon_c.max() - lon_c.min()) - 1.0
                    grid_y = 2.0 * (torch.from_numpy(t_lat).to(device) - lat_c.min()) / (lat_c.max() - lat_c.min()) - 1.0
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to(torch.float32)
                    
                    final_out = F.grid_sample(simulated, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                    
                    # 5. 保存
                    sim_da = xr.DataArray(final_out.squeeze().cpu().numpy(), dims=("band", "y", "x"), 
                                         coords={"band": np.arange(1,8), "y": ty, "x": tx})
                    sim_da.rio.write_crs(utm_epsg, inplace=True)
                    sim_da.rio.to_raster(out_tif)
                    local_results[plume_id] = str(out_tif.absolute())

            # 清理 Buffer
            if rfl_path.exists():
                rfl_path.unlink() # 处理完删除本地缓存
            buffer_dict.pop(granule_id, None)
            torch.cuda.empty_cache()

        except Exception as e:
            pbar.write(f"Error on {plume_id}: {e}")

    return_dict[gpu_id] = local_results

def main():
    LOCAL_BUFFER_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print("--- EMIT Multi-GPU Buffer Pipeline ---", flush=True)
    df = pd.read_csv(CSV_PATH)
    srf_df = pd.read_csv(SRF_CSV)
    
    mask = (df['plume_latitude'] >= 30) & (df['plume_latitude'] <= 35) & (df['has_emit'] == 1)
    target_df = df[mask].copy()
    granule_ids = target_df['emit_granule_id'].unique().tolist()
    
    print(f"Total Granules: {len(granule_ids)}")

    manager = mp.Manager()
    buffer_dict = manager.dict()
    granule_queue = manager.Queue()
    for gid in granule_ids: granule_queue.put(gid)
    for _ in range(NUM_GPUS): granule_queue.put(None)
    
    return_dict = manager.dict()
    split_granules = np.array_split(granule_ids, NUM_GPUS)

    mp.set_start_method('spawn', force=True)
    
    # 启动预取进程
    p_fetch = mp.Process(target=data_prefetcher, args=(granule_queue, buffer_dict))
    p_fetch.start()

    # 启动 GPU 进程
    workers = []
    for i in range(NUM_GPUS):
        p = mp.Process(target=process_worker, args=(i, split_granules[i].tolist(), target_df, srf_df, return_dict, buffer_dict))
        p.start()
        workers.append(p)

    try:
        for p in workers: p.join()
    except KeyboardInterrupt:
        print("\nTerminating...")
        for p in workers: p.terminate()
        p_fetch.terminate()
    finally:
        p_fetch.terminate()

    # 更新 CSV
    all_paths = {}
    for res in return_dict.values(): all_paths.update(res)
    df['simulated_512_path'] = df['plume_id'].map(all_paths)
    df.to_csv(UPDATED_CSV_PATH, index=False)
    print(f"Updated CSV saved to {UPDATED_CSV_PATH}")

if __name__ == "__main__":
    main()