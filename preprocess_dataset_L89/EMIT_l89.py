import os
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import earthaccess
import time
from pathlib import Path
from pyproj import Transformer, CRS
from scipy.interpolate import interp1d
import cupy as cp
from scipy.spatial import KDTree as CPU_KDTree
from datetime import datetime

# ==================== 用户配置区 ====================
CSV_PATH = "./merged_with_emit_tag.csv"
SRF_CSV = "./landsat9_oli_srf.csv"
EMIT_RAW_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT")
OUTPUT_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_l89_L2SR/EMIT_simulated_landsat9")

# 空间参数
CHIP_SIZE_PX = 512    # 输出 512x512 像素
SCALE_M = 30          # 模拟 Landsat 的 30米分辨率
# ===================================================

EMIT_RAW_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 登录 NASA Earthdata
auth = earthaccess.login()

def get_utm_crs(lat, lon):
    """根据经纬度自动计算合适的 UTM 投影 EPSG 代码"""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = f"EPSG:326{zone:02d}"
    else:
        epsg = f"EPSG:327{zone:02d}"
    return epsg

def download_emit_granule(granule_id):
    """下载 EMIT 数据"""
    # 检查本地是否已有 RFL 文件
    existing = list(EMIT_RAW_DIR.glob(f"*{granule_id}*RFL*.nc"))
    if existing:
        return existing[0]
    
    results = earthaccess.search_data(short_name='EMITL2ARFL', granule_name=granule_id)
    if not results:
        return None
    
    # 过滤掉不必要的文件，只留 RFL
    links = [link for link in results[0].data_links() if "UNCERT" not in link and "RFL" in link]
    files = earthaccess.download(links, str(EMIT_RAW_DIR))
    return Path(files[0]) if files else None

def process_emit_to_simulated_landsat(row, srf_df):
    plume_id = row['plume_id']
    granule_id = row['emit_granule_id']
    lat, lon = row['plume_latitude'], row['plume_longitude']
    
    out_tif = OUTPUT_DIR / f"{plume_id}_sim_L9.tif"
    if out_tif.exists():
        print(f"   [Skip] {plume_id} exists.")
        return

    print(f"\n[Processing] {plume_id} | Granule: {granule_id}")

    # 1. 下载/加载 EMIT 数据
    rfl_path = download_emit_granule(granule_id)
    if not rfl_path:
        print(f"   [Error] Could not find/download EMIT for {plume_id}")
        return

    # 2. 读取 EMIT 光谱与坐标
    ds = xr.open_dataset(rfl_path, engine='netcdf4')
    ds_band = xr.open_dataset(rfl_path, group='sensor_band_parameters', engine='netcdf4')
    ds_loc = xr.open_dataset(rfl_path, group='location', engine='netcdf4')
    
    waves = ds_band['wavelengths'].values
    e_lon, e_lat = ds_loc['lon'].values, ds_loc['lat'].values
    rfl_val = ds['reflectance'].values # (reflectance_y, reflectance_x, bands)
    
    # 3. 创建目标网格 (Reference Grid)
    # 计算 UTM 坐标以生成以米为单位的正方形
    utm_epsg = get_utm_crs(lat, lon)
    transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    center_x, center_y = transformer_to_utm.transform(lon, lat)
    
    half_size = (CHIP_SIZE_PX * SCALE_M) / 2
    # 生成标准的网格坐标轴
    target_x = np.linspace(center_x - half_size, center_x + half_size, CHIP_SIZE_PX)
    target_y = np.linspace(center_y + half_size, center_y - half_size, CHIP_SIZE_PX) # Y通常递减
    
    # 将目标网格点转回经纬度，用于在 EMIT 原始数据中索引
    transformer_to_wgs84 = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
    mesh_x, mesh_y = np.meshgrid(target_x, target_y)
    t_lon, t_lat = transformer_to_wgs84.transform(mesh_x, mesh_y)

    # 4. GPU 光谱卷积 (EMIT -> Landsat Bands)
    # EMIT shape: (Y, X, 285)
    cp_rfl = cp.array(np.nan_to_num(rfl_val, 0))
    sim_7band_list = []
    for i in range(1, 8):
        f = interp1d(srf_df["wavelength"], srf_df[f"b{i}"], fill_value=0, bounds_error=False)
        w_weights = cp.array(f(waves))
        w_weights /= (w_weights.sum() + 1e-12)
        # 对最后一个维度进行卷积
        sim_7band_list.append(cp.tensordot(cp_rfl, w_weights, axes=(2, 0)))
    
    cp_sim_stack = cp.stack(sim_7band_list) # (7, Y_emit, X_emit)

    # 5. 空间重采样 (使用 CPU 计算索引，GPU 提取数据)
    # 将 EMIT 的经纬度展平
    e_lat_flat = e_lat.ravel()
    e_lon_flat = e_lon.ravel()
    e_coords = np.stack([e_lat_flat, e_lon_flat], axis=1)

    # 将目标网格经纬度展平
    t_coords = np.stack([t_lat.ravel(), t_lon.ravel()], axis=1)

    # 在 CPU 上构建树并查询（对 512x512 的量级，通常只需 < 1秒）
    tree = CPU_KDTree(e_coords)
    max_dist = 0.001  # 约 100 米的经纬度距离，根据实际情况调整
    dist, indices = tree.query(t_coords, distance_upper_bound=max_dist)

    # 处理无效索引 (KDTree 找不到时会返回 len(e_coords))
    invalid_mask = dist == float('inf')
    # 将索引转为 GPU 数组，以便在显存中提取像素
    cp_indices = cp.array(indices)
    cp_invalid_mask = cp.array(invalid_mask).reshape((CHIP_SIZE_PX, CHIP_SIZE_PX))

    final_output = cp.zeros((7, CHIP_SIZE_PX, CHIP_SIZE_PX), dtype=cp.float32)
    # 在循环中应用掩膜
    for b in range(7):
        # 在 GPU 上进行大规模数据重组
        band_flat = cp_sim_stack[b].ravel()
        extracted = band_flat[cp_indices].reshape((CHIP_SIZE_PX, CHIP_SIZE_PX))
        # 将超出范围的点设为 0 或 NaN
        extracted[cp_invalid_mask] = 0 
        final_output[b] = extracted

    # 6. 保存为 GeoTIFF
    sim_da = xr.DataArray(
        final_output.get(),
        dims=("band", "y", "x"),
        coords={"band": np.arange(1, 8), "y": target_y, "x": target_x}
    )
    sim_da.rio.write_crs(utm_epsg, inplace=True)
    sim_da.rio.to_raster(out_tif)
    print(f"   [Success] Saved to {out_tif}")

def main():
    df = pd.read_csv(CSV_PATH)
    srf_df = pd.read_csv(SRF_CSV)
    
    # 筛选条件（可根据需要调整）
    # 目标区域：Permian Basin 示例
    mask = (df['plume_latitude'] >= 30) & (df['plume_latitude'] <= 35) & \
           (df['has_emit'] == 1)
    target_df = df[mask]
    
    print(f"Total tasks: {len(target_df)}")

    for _, row in target_df.iterrows():
        try:
            process_emit_to_simulated_landsat(row, srf_df)
        except Exception as e:
            print(f"   [Error] Task {row['plume_id']} failed: {e}")

if __name__ == "__main__":
    main()