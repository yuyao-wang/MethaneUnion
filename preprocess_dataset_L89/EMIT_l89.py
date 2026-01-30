import os
import gc
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import torch
import torch.nn.functional as F
from pathlib import Path
from pyproj import Transformer
import multiprocessing as mp

# ==================== 用户配置区 ====================
CSV_PATH = "./merged_with_emit_tag.csv"
SRF_CSV = "./landsat9_oli_srf.csv"
EMIT_RAW_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT")
OUTPUT_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_l89_L2SR/EMIT_simulated_landsat9_60resolution")
CHIP_SIZE_PX = 512
SCALE_M = 60
NUM_GPUS = 2
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

def process_worker(gpu_id, granule_list, target_df, srf_df):
    """
    每个 GPU 进程执行的任务
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # 预加载光谱矩阵（假设所有文件波段一致）
    # 实际生产中建议在循环内部根据第一个文件初始化
    spec_mat = None 

    for granule_id in granule_list:
        group = target_df[target_df['emit_granule_id'] == granule_id]
        rfl_files = list(EMIT_RAW_DIR.glob(f"*{granule_id}*RFL*.nc"))
        if not rfl_files: continue
        rfl_path = rfl_files[0]

        print(f"GPU {gpu_id} | Processing {granule_id} ({len(group)} plumes)")

        try:
            with xr.open_dataset(rfl_path, group='location') as ds_loc:
                full_lats = ds_loc['lat'].values
                full_lons = ds_loc['lon'].values
            
            if spec_mat is None:
                with xr.open_dataset(rfl_path, group='sensor_band_parameters') as dsb:
                    spec_mat = get_spectral_matrix(dsb['wavelengths'].values, srf_df).to(device)

            ds_rfl = xr.open_dataset(rfl_path)
            reflectance = ds_rfl['reflectance']

            for _, row in group.iterrows():
                plume_id = row['plume_id']
                out_tif = OUTPUT_DIR / f"{plume_id}_sim_L9.tif"
                if out_tif.exists(): continue

                # 1. 局部裁剪以节省显存
                p_lat, p_lon = row['plume_latitude'], row['plume_longitude']
                # 裁剪半径约 20km
                mask_spatial = (full_lats > p_lat - 0.2) & (full_lats < p_lat + 0.2) & \
                               (full_lons > p_lon - 0.2) & (full_lons < p_lon + 0.2)
                y_idx, x_idx = np.where(mask_spatial)
                if len(y_idx) == 0: continue
                y_m, y_M, x_m, x_M = y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max()

                # 提取裁剪块并转为 Tensor
                crop_lat = torch.from_numpy(full_lats[y_m:y_M, x_m:x_m]).to(device)
                crop_lon = torch.from_numpy(full_lons[y_m:y_M, x_m:x_m]).to(device)
                crop_rfl = torch.from_numpy(np.nan_to_num(reflectance[y_m:y_M, x_m:x_m, :].values, 0)).to(device)

                # 2. 光谱卷积 (Pixels, Bands) @ (Bands, 7)
                # (H, W, 285) @ (285, 7) -> (H, W, 7) -> (1, 7, H, W) 适配 grid_sample
                simulated = torch.matmul(crop_rfl, spec_mat).permute(2, 0, 1).unsqueeze(0)

                # 3. 准备采样网格 (Target Grid)
                utm_epsg = get_utm_crs(p_lat, p_lon)
                to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
                to_wgs = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
                
                cx, cy = to_utm.transform(p_lon, p_lat)
                off = (CHIP_SIZE_PX / 2 - 0.5) * SCALE_M
                tx = (cx - off) + np.arange(CHIP_SIZE_PX) * SCALE_M
                ty = (cy + offset) - np.arange(CHIP_SIZE_PX) * SCALE_M # 注意这里 offset 的逻辑同前
                mx, my = np.meshgrid(tx, ty)
                t_lon, t_lat = to_wgs.transform(mx, my)

                # 将目标经纬度映射到裁剪块的相对坐标 [-1, 1]
                # grid_sample 要求 grid 的维度是 (1, H_out, W_out, 2)，最后一维是 (x, y) 且映射到 [-1, 1]
                lat_min, lat_max = crop_lat.min(), crop_lat.max()
                lon_min, lon_max = crop_lon.min(), crop_lon.max()
                
                grid_x = 2.0 * (torch.from_numpy(t_lon).to(device) - lon_min) / (lon_max - lon_min) - 1.0
                grid_y = 2.0 * (torch.from_numpy(t_lat).to(device) - lat_min) / (lat_max - lat_min) - 1.0
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).float()

                # 4. 执行重采样
                # mode='bilinear' 比最近邻更平滑，padding_mode='zeros' 处理越界
                final_output = F.grid_sample(simulated, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                res_np = final_output.squeeze().cpu().numpy()

                # 5. 保存
                sim_da = xr.DataArray(res_np, dims=("band", "y", "x"), coords={"band": np.arange(1, 8), "y": ty, "x": tx})
                sim_da.rio.write_crs(utm_epsg, inplace=True)
                sim_da.rio.to_raster(out_tif)

                del crop_rfl, simulated, final_output, grid
                torch.cuda.empty_cache()

            ds_rfl.close()
        except Exception as e:
            print(f"Error on GPU {gpu_id} for {granule_id}: {e}")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(CSV_PATH)
    srf_df = pd.read_csv(SRF_CSV)
    
    mask = (df['plume_latitude'] >= 30) & (df['plume_latitude'] <= 35) & (df['has_emit'] == 1)
    target_df = df[mask].copy()
    
    granule_ids = target_df['emit_granule_id'].unique()
    # 任务平分给两个 GPU
    split_granules = np.array_split(granule_ids, NUM_GPUS)

    mp.set_start_method('spawn', force=True)
    processes = []
    for i in range(NUM_GPUS):
        p = mp.Process(target=process_worker, args=(i, split_granules[i].tolist(), target_df, srf_df))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()