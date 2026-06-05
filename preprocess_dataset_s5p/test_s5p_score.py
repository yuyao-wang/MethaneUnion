import pandas as pd
import numpy as np
import xarray as xr
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

csv_path = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/s5p_all_OFFL.csv"
out_csv  = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/s5p_all_OFFL_detectability_score.csv"

# Translated comment
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_WORKERS = 4  # Translated comment

df = pd.read_csv(csv_path, low_memory=False)

def pick_ch4_var(ds):
    # Translated comment
    for k in ["methane_mixing_ratio_bias_corrected", "methane_mixing_ratio", "xch4"]:
        if k in ds.variables:
            return k
    return None

def nearest_ch4_qa(nc_path, lat0, lon0):
    """
 : ch4, qa, dist_km, iy, ix, var_name
 np.nan
    """
    try:
        ds = xr.open_dataset(nc_path, group="PRODUCT", engine="netcdf4", decode_timedelta=True)
    except Exception:
        return (np.nan, np.nan, np.nan, -1, -1, None)

    try:
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        qa = ds["qa_value"].values if "qa_value" in ds.variables else None
        var = pick_ch4_var(ds)
        if var is None:
            ds.close()
            return (np.nan, np.nan, np.nan, -1, -1, None)
        ch4 = ds[var].values

        # Translated comment
        if lat.ndim == 3:
            lat2, lon2 = lat[0], lon[0]
            ch42 = ch4[0] if ch4.ndim == 3 else ch4
            qa2  = qa[0]  if (qa is not None and qa.ndim == 3) else qa
        else:
            lat2, lon2 = lat, lon
            ch42 = ch4
            qa2  = qa

        # Translated comment
        lat_rad = np.deg2rad(lat2.astype(np.float64))
        lon_rad = np.deg2rad(lon2.astype(np.float64))
        lat0r = math.radians(float(lat0))
        lon0r = math.radians(float(lon0))

        dlon = (lon_rad - lon0r + np.pi) % (2*np.pi) - np.pi
        x = dlon * np.cos(0.5 * (lat_rad + lat0r))
        y = (lat_rad - lat0r)
        dist2 = x*x + y*y

        flat_idx = np.nanargmin(dist2)
        iy, ix = np.unravel_index(flat_idx, dist2.shape)
        dist_km = math.sqrt(float(dist2[iy, ix])) * 6371.0

        ch4_val = float(ch42[iy, ix]) if np.isfinite(ch42[iy, ix]) else np.nan
        qa_val  = float(qa2[iy, ix])  if (qa2 is not None and np.isfinite(qa2[iy, ix])) else np.nan

        ds.close()
        return (ch4_val, qa_val, dist_km, int(iy), int(ix), var)

    except Exception:
        try:
            ds.close()
        except Exception:
            pass
        return (np.nan, np.nan, np.nan, -1, -1, None)

def process_row(i_row):
    i, row = i_row
    lat0, lon0 = float(row["lat"]), float(row["lon"])

    p0   = row["S5p_path"]
    p90  = row["s5p_minus90_path"]
    p360 = row["s5p_minus360_path"]

    ch4_0, qa_0, d0, iy0, ix0, v0 = nearest_ch4_qa(p0, lat0, lon0)
    ch4_90, qa_90, d90, iy90, ix90, v90 = nearest_ch4_qa(p90, lat0, lon0)
    ch4_360, qa_360, d360, iy360, ix360, v360 = nearest_ch4_qa(p360, lat0, lon0)

    hist = np.array([ch4_90, ch4_360], dtype=np.float64)
    valid_hist = hist[np.isfinite(hist)]
    baseline = np.median(valid_hist) if valid_hist.size > 0 else np.nan
    score = ch4_0 - baseline if np.isfinite(ch4_0) and np.isfinite(baseline) else np.nan

    return {
        "plume_id": row["plume_id"],
        "plume_time": row["plume_time"],
        "lat": lat0,
        "lon": lon0,

        "ch4_t0": ch4_0,
        "ch4_t-90": ch4_90,
        "ch4_t-360": ch4_360,
        "baseline_med": baseline,
        "score": score,

        "qa_t0": qa_0,
        "qa_t-90": qa_90,
        "qa_t-360": qa_360,

        "dist_km_t0": d0,
        "dist_km_t-90": d90,
        "dist_km_t-360": d360,

        "var_t0": v0,
        "var_t-90": v90,
        "var_t-360": v360,
    }

# Translated comment
rows = []
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futures = [ex.submit(process_row, (i, row)) for i, row in df.iterrows()]
    for k, fut in enumerate(as_completed(futures), 1):
        rows.append(fut.result())
        if k % 200 == 0:
            print(f"done {k}/{len(df)}")

res = pd.DataFrame(rows)
res.to_csv(out_csv, index=False)
print("Saved:", out_csv, "rows:", len(res))

# Translated comment
# Translated comment
valid = res[np.isfinite(res["score"])].copy()
print("Valid score rows:", len(valid), "/", len(res))
print("score quantiles:", valid["score"].quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]).to_dict())

# Translated comment
valid_qa = valid[(valid["qa_t0"].isna()) | (valid["qa_t0"] >= 0.5)]
print("Valid+QA rows:", len(valid_qa))

for thr in [5, 10, 20, 30, 50]:
    cnt = (valid_qa["score"] > thr).sum()
    print(f"score > {thr}: {cnt} ({cnt/len(valid_qa):.3f})")
