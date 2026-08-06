import os
import gc
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import cupy as cp
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import KDTree as CPU_KDTree
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# ==================== user configuration section ====================
CSV_PATH = "./merged_with_emit_tag.csv"
SRF_CSV = "./WV3_VNIR_SWIR_response.csv"
EMIT_RAW_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT")
OUTPUT_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/EMIT_simulated_WV3_L2A_60resolution_NOnorm")

CHIP_SIZE_PX = 512    
SCALE_M = 60          
MAX_WORKERS = 12  # Translated comment
# ===================================================

os.umask(0)
EMIT_RAW_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def get_utm_crs(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"

def find_local_granule(granule_id):
    clean_id = str(granule_id).strip().replace('\r', '').replace('\n', '')
    search_pattern = f"*{clean_id}*.nc"
    existing = list(EMIT_RAW_DIR.glob(search_pattern))
    if existing:
        rfl_files = [f for f in existing if "RFL" in f.name and "UNCERT" not in f.name and "MASK" not in f.name]
        return rfl_files[0] if rfl_files else None
    return None

def process_single_task(row_tuple, srf_data):
    """Translated to English."""
    _, row = row_tuple
    plume_id = row['plume_id']
    out_tif = OUTPUT_DIR / f"{plume_id}_sim_WV3.tif"
    
    if out_tif.exists():
        return f"[Skip] {plume_id}"

    rfl_path = find_local_granule(row['emit_granule_id'])
    if not rfl_path:
        return f"[Error] Not found: {plume_id}"

    try:
        # Translated comment
        with xr.open_dataset(rfl_path, group='location', engine='netcdf4') as ds_loc:
            lats, lons = ds_loc['lat'].values, ds_loc['lon'].values
            mask_spatial = (lats > row['plume_latitude'] - 0.15) & (lats < row['plume_latitude'] + 0.15) & \
                           (lons > row['plume_longitude'] - 0.15) & (lons < row['plume_longitude'] + 0.15)
            y_idxs, x_idxs = np.where(mask_spatial)
            if len(y_idxs) == 0: return f"[Empty] {plume_id}"
            
            y_min, y_max, x_min, x_max = y_idxs.min(), y_idxs.max(), x_idxs.min(), x_idxs.max()
            lat_crop, lon_crop = lats[y_min:y_max, x_min:x_max], lons[y_min:y_max, x_min:x_max]

        # Translated comment
        with xr.open_dataset(rfl_path, engine='netcdf4') as ds:
            rfl_crop = ds['reflectance'][y_min:y_max, x_min:x_max, :].values
        
        with xr.open_dataset(rfl_path, group='sensor_band_parameters') as dsb:
            emit_waves = dsb['wavelengths'].values

        # Translated comment
        wv3_bands = ['Coastal (MS7)', 'Blue (MS4)', 'Green (MS3)', 'Yellow (MS6)', 'Red (MS2)', 
                     'Red Edge (MS5)', 'NIR1 (MS1)', 'NIR2 (MS8)', 'SWIR1', 'SWIR2', 'SWIR3', 
                     'SWIR4', 'SWIR5', 'SWIR6', 'SWIR7', 'SWIR8']
        srf_matrix = np.zeros((len(emit_waves), 16))
        for i, b in enumerate(wv3_bands):
            w = np.interp(emit_waves, srf_data['waves'], srf_data[b], left=0, right=0)
            srf_matrix[:, i] = w / (w.sum() + 1e-12)

        # Translated comment
        cp_rfl = cp.array(np.nan_to_num(rfl_crop, 0))
        cp_srf = cp.array(srf_matrix)
        sim_conv = cp.matmul(cp_rfl, cp_srf)
        
        target_scales = [10000] * 16
        cp_simulated_list = []
        for b in range(16):
            band_data = sim_conv[:, :, b]
            valid = (band_data > 0)
            if valid.any():
                p_low, p_high = cp.percentile(band_data[valid], 1), cp.percentile(band_data[valid], 99)
                stretched = (band_data - p_low) / (p_high - p_low + 1e-6)
                final_band = (stretched * (target_scales[b] * 0.6)) + (target_scales[b] * 0.8)
                final_band = cp.where(valid, final_band, 0)
            else:
                final_band = band_data
            cp_simulated_list.append(cp.clip(final_band, 0, 65535))
        
        cp_simulated = cp.stack(cp_simulated_list, axis=0)

        # Translated comment
        utm_epsg = get_utm_crs(row['plume_latitude'], row['plume_longitude'])
        to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
        to_wgs = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
        
        cx, cy = to_utm.transform(row['plume_longitude'], row['plume_latitude'])
        hs = (CHIP_SIZE_PX * SCALE_M) / 2
        tx, ty = np.linspace(cx-hs, cx+hs, CHIP_SIZE_PX), np.linspace(cy+hs, cy-hs, CHIP_SIZE_PX)
        mx, my = np.meshgrid(tx, ty)
        t_lon, t_lat = to_wgs.transform(mx, my)
        
        tree = CPU_KDTree(np.stack([lat_crop.ravel(), lon_crop.ravel()], axis=1))
        dist, idxs = tree.query(np.stack([t_lat.ravel(), t_lon.ravel()], axis=1), distance_upper_bound=0.001)
        
        # Translated comment
        res = cp_simulated.reshape(16, -1)[:, cp.array(idxs)].reshape(16, CHIP_SIZE_PX, CHIP_SIZE_PX)
        res[:, cp.array(dist == float('inf')).reshape(CHIP_SIZE_PX, CHIP_SIZE_PX)] = 0
        
        res_np = res.get().astype(np.uint16)
        sim_da = xr.DataArray(res_np, dims=("band", "y", "x"), coords={"band": np.arange(1, 17), "y": ty, "x": tx})
        sim_da.rio.write_crs(utm_epsg, inplace=True).rio.to_raster(out_tif)

        # Translated comment
        del cp_rfl, cp_srf, sim_conv, cp_simulated, res, res_np
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        return f"[Success] {plume_id}"
    except Exception as e:
        return f"[Error] {plume_id}: {str(e)}"

def main():
    df = pd.read_csv(CSV_PATH)
    srf_df = pd.read_csv(SRF_CSV)
    
    # Translated comment
    wv3_bands = ['Coastal (MS7)', 'Blue (MS4)', 'Green (MS3)', 'Yellow (MS6)', 'Red (MS2)', 
                 'Red Edge (MS5)', 'NIR1 (MS1)', 'NIR2 (MS8)', 'SWIR1', 'SWIR2', 'SWIR3', 
                 'SWIR4', 'SWIR5', 'SWIR6', 'SWIR7', 'SWIR8']
    srf_data = {b: srf_df[b].values for b in wv3_bands}
    srf_data['waves'] = srf_df['nm/Band'].values

    # mask = (df['plume_latitude'] >= 30) & (df['plume_latitude'] <= 35) & (df['has_emit'] == 1)
    tasks = list(df.iterrows())
    
    print(f"Starting parallel processing for {len(tasks)} tasks with {MAX_WORKERS} workers...")

    # Translated comment
    worker_func = partial(process_single_task, srf_data=srf_data)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(worker_func, tasks):
            print(result)

if __name__ == "__main__":
    main()