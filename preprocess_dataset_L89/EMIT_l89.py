# convert EMIT to L9, 60m with optimized band alignment
import os
import gc
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import earthaccess
import cupy as cp
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import KDTree as CPU_KDTree

os.umask(0)  # Translated comment
# ==================== user configuration section ====================
CSV_PATH = "./merged_with_emit_tag.csv"
SRF_CSV = "./landsat9_oli_srf.csv"
EMIT_RAW_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT")
OUTPUT_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_l89_L2SR/EMIT_simulated_landsat9_60resolution_NOnorm")

CHIP_SIZE_PX = 512    
SCALE_M = 60          
# ===================================================

EMIT_RAW_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Login to NASA Earthdata
auth = earthaccess.login()

def get_utm_crs(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"

def download_emit_granule(granule_id):
    existing = list(EMIT_RAW_DIR.glob(f"*{granule_id}*RFL*.nc"))
    if existing: return existing[0]
    
    results = earthaccess.search_data(short_name='EMITL2ARFL', granule_name=granule_id)
    if not results: return None
    
    links = [link for link in results[0].data_links() if "UNCERT" not in link and "RFL" in link]
    files = earthaccess.download(links, str(EMIT_RAW_DIR))
    return Path(files[0]) if files else None

def get_spectral_matrix(emit_waves, srf_df):
    matrix = np.zeros((len(emit_waves), 7))
    for i in range(1, 8):
        w = np.interp(emit_waves, srf_df["wavelength"], srf_df[f"b{i}"], left=0, right=0)
        matrix[:, i-1] = w / (w.sum() + 1e-12)
    return cp.array(matrix)

def main():
    df = pd.read_csv(CSV_PATH)
    srf_df = pd.read_csv(SRF_CSV)
    
    # Translated comment
    mask = (df['plume_latitude'] >= 30) & (df['plume_latitude'] <= 35) & (df['has_emit'] == 1)
    target_df = df[mask]
    
    print(f"Total tasks to process: {len(target_df)}")
    
    spectral_matrix = None 
    target_scales = [10100, 11100, 13100, 16100, 20100, 23100, 21100]

    for _, row in target_df.iterrows():
        plume_id = row['plume_id']
        out_tif = OUTPUT_DIR / f"{plume_id}_sim_L9.tif"
        
        if out_tif.exists():
            print(f"   [Skip] {plume_id} already exists.")
            continue

        rfl_path = download_emit_granule(row['emit_granule_id'])
        if not rfl_path: 
            print(f"   [Error] Could not download/find granule for {plume_id}")
            continue

        print(f"\n[Processing] {plume_id}")

        try:
            # Translated comment
            with xr.open_dataset(rfl_path, group='location', engine='netcdf4') as ds_loc:
                lats, lons = ds_loc['lat'].values, ds_loc['lon'].values
                mask_spatial = (lats > row['plume_latitude'] - 0.15) & (lats < row['plume_latitude'] + 0.15) & \
                               (lons > row['plume_longitude'] - 0.15) & (lons < row['plume_longitude'] + 0.15)
                y_idxs, x_idxs = np.where(mask_spatial)
                
                if len(y_idxs) == 0:
                    print(f"   [Error] Plume {plume_id} not found in spatial extent.")
                    continue
                
                y_min, y_max, x_min, x_max = y_idxs.min(), y_idxs.max(), x_idxs.min(), x_idxs.max()
                lat_crop, lon_crop = lats[y_min:y_max, x_min:x_max], lons[y_min:y_max, x_min:x_max]

            # Translated comment
            with xr.open_dataset(rfl_path, engine='netcdf4') as ds:
                rfl_crop = ds['reflectance'][y_min:y_max, x_min:x_max, :].values
                
            if spectral_matrix is None:
                with xr.open_dataset(rfl_path, group='sensor_band_parameters') as dsb:
                    spectral_matrix = get_spectral_matrix(dsb['wavelengths'].values, srf_df)

            # Translated comment
            cp_rfl = cp.array(np.nan_to_num(rfl_crop, 0))
            sim_conv = cp.matmul(cp_rfl, spectral_matrix) # (Y, X, 7)
            
            cp_simulated_list = []
            for b in range(7):
                band_data = sim_conv[:, :, b]
                valid_mask = (band_data > 0)
                
                if valid_mask.any():
                    p_low = cp.percentile(band_data[valid_mask], 1)
                    p_high = cp.percentile(band_data[valid_mask], 99)
                    # Translated comment
                    stretched = (band_data - p_low) / (p_high - p_low + 1e-6)
                else:
                    stretched = band_data

                # Translated comment
                offset = target_scales[b] * 0.8
                final_band = (stretched * (target_scales[b] * 0.6)) + offset
                # Translated comment
                final_band = cp.where(band_data > 0, final_band, 0)
                cp_simulated_list.append(cp.clip(final_band, 0, 65535))
            
            cp_simulated = cp.stack(cp_simulated_list, axis=0) # (7, Y, X)

            # Translated comment
            utm_epsg = get_utm_crs(row['plume_latitude'], row['plume_longitude'])
            to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
            to_wgs = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
            
            center_x, center_y = to_utm.transform(row['plume_longitude'], row['plume_latitude'])
            half_size = (CHIP_SIZE_PX * SCALE_M) / 2
            target_x = np.linspace(center_x - half_size, center_x + half_size, CHIP_SIZE_PX)
            target_y = np.linspace(center_y + half_size, center_y - half_size, CHIP_SIZE_PX)
            
            mesh_x, mesh_y = np.meshgrid(target_x, target_y)
            t_lon, t_lat = to_wgs.transform(mesh_x, mesh_y)
            
            tree = CPU_KDTree(np.stack([lat_crop.ravel(), lon_crop.ravel()], axis=1))
            dist, indices = tree.query(np.stack([t_lat.ravel(), t_lon.ravel()], axis=1), distance_upper_bound=0.001)
            
            # Translated comment
            cp_indices = cp.array(indices)
            final_output = cp_simulated.reshape(7, -1)[:, cp_indices].reshape(7, CHIP_SIZE_PX, CHIP_SIZE_PX)
            invalid_mask = cp.array(dist == float('inf')).reshape(CHIP_SIZE_PX, CHIP_SIZE_PX)
            final_output[:, invalid_mask] = 0

            # Translated comment
            res_np = final_output.get().astype(np.uint16)
            sim_da = xr.DataArray(
                res_np,
                dims=("band", "y", "x"),
                coords={"band": np.arange(1, 8), "y": target_y, "x": target_x}
            )
            sim_da.rio.write_crs(utm_epsg, inplace=True)
            sim_da.rio.to_raster(out_tif)
            
            print(f"   [Success] Saved: {out_tif.name} | Mean: {res_np.mean():.0f}")

            # Translated comment
            del cp_rfl, sim_conv, cp_simulated, cp_simulated_list, final_output, rfl_crop, tree, res_np, sim_da
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        except Exception as e:
            print(f"   [Error] Task {plume_id} failed: {str(e)}")

if __name__ == "__main__":
    main()