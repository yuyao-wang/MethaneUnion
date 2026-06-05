import os
import re
import math
import time
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

# ====== paths provided by you (hard-coded)======
S5_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s5p")
PLUME_CSV = Path("/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv")

OUT_INDEX_CSV = S5_DIR / "_s5p_index.csv"           # already exists; do not rebuild
OUT_MATCH_CSV = S5_DIR / "_plume_to_s5p_match.csv"

# Translated comment
FNAME_RE = re.compile(
    r"(?P<prefix>S5P)_(?P<proc>OFFL|RPRO|NRTI)?_?L2__CH4____"
    r"(?P<t0>\d{8}T\d{6})_(?P<t1>\d{8}T\d{6})_(?P<orbit>\d+)"
    r".*\.nc$"
)

def parse_time(s):
    return datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

def safe_minmax(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan)
    return (float(a.min()), float(a.max()))

def wrap_lon_bounds(lon_vals):
    lon = np.asarray(lon_vals)
    raw_min, raw_max = safe_minmax(lon)
    lon360 = (lon % 360 + 360) % 360
    alt_min, alt_max = safe_minmax(lon360)
    return raw_min, raw_max, alt_min, alt_max

def build_or_load_index():
    if not OUT_INDEX_CSV.exists():
        raise RuntimeError(
            f"Index file not found：{OUT_INDEX_CSV}\n"
            "You said the index already exists, so this script will not rebuild it. Please verify the path."
        )
    idx = pd.read_csv(OUT_INDEX_CSV, parse_dates=["t_start", "t_end"])
    idx["t_start"] = pd.to_datetime(idx["t_start"], utc=True)
    idx["t_end"] = pd.to_datetime(idx["t_end"], utc=True)

    # Translated comment
    float_cols = [
        "lat_min", "lat_max", "lon_min", "lon_max",
        "lon360_min", "lon360_max"
    ]
    for c in float_cols:
        if c in idx.columns:
            idx[c] = pd.to_numeric(idx[c], errors="coerce")
    return idx

# ====== automatic plume CSV column detection ======
def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

def parse_plume_time_series(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def load_plumes():
    df = pd.read_csv(PLUME_CSV)

    id_col = pick_col(df.columns, ["plume_id", "id", "plumeid", "plumeid_str", "plume"])
    lat_col = pick_col(df.columns, ["lat", "latitude", "plume_lat", "plume_latitude"])
    lon_col = pick_col(df.columns, ["lon", "longitude", "plume_lon", "plume_longitude"])
    time_col = pick_col(df.columns, ["time", "timestamp", "datetime", "acq_time", "scene_time", "start_time", "t0"])

    if id_col is None or lat_col is None or lon_col is None or time_col is None:
        raise RuntimeError(
            "Required columns were not found in merged_file.csv. At minimum it needs：plume_id / lat / lon / time。\n"
            f"Detected columns：id={id_col}, lat={lat_col}, lon={lon_col}, time={time_col}\n"
            "Print df.columns to inspect the real column names, then extend the candidate list."
        )

    out = df[[id_col, lat_col, lon_col, time_col]].copy()
    out.columns = ["plume_id", "lat", "lon", "time"]

    out["time"] = parse_plume_time_series(out["time"])
    out = out.dropna(subset=["time", "lat", "lon", "plume_id"])
    out["lat"] = out["lat"].astype(float)
    out["lon"] = out["lon"].astype(float)

    # Optional latitude/longitude range filter to avoid slowdowns or errors from bad values
    out = out[(out["lat"] >= -90) & (out["lat"] <= 90)]
    out = out[(out["lon"] >= -180) & (out["lon"] <= 180)]

    return out.reset_index(drop=True)

# Translated comment
def bbox_filter_vec(cand: pd.DataFrame, lat: float, lon: float, pad: float = 0.2) -> pd.DataFrame:
    # lat
    lat_ok = (cand["lat_min"] - pad <= lat) & (lat <= cand["lat_max"] + pad)

    # raw lon
    raw_ok = (cand["lon_min"] - pad <= lon) & (lon <= cand["lon_max"] + pad)
    raw_cross = cand["lon_min"] > cand["lon_max"]
    raw_ok = raw_ok | (raw_cross & ((lon >= cand["lon_min"] - pad) | (lon <= cand["lon_max"] + pad)))

    # lon360
    lon360 = (lon % 360 + 360) % 360
    alt_ok = (cand["lon360_min"] - pad <= lon360) & (lon360 <= cand["lon360_max"] + pad)
    alt_cross = cand["lon360_min"] > cand["lon360_max"]
    alt_ok = alt_ok | (alt_cross & ((lon360 >= cand["lon360_min"] - pad) | (lon360 <= cand["lon360_max"] + pad)))

    return cand[lat_ok & (raw_ok | alt_ok)]

# ====== S5P file-level cache to avoid repeatedly opening datasets ======
@lru_cache(maxsize=16)
def load_s5p_arrays(nc_path: str):
    # decode_timedelta=True suppresses the FutureWarning you saw
    ds = xr.open_dataset(nc_path, group="PRODUCT", engine="netcdf4", decode_timedelta=True)

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    qa = ds["qa_value"].values if "qa_value" in ds.variables else None

    ch4 = None
    for k in ["methane_mixing_ratio", "methane_mixing_ratio_bias_corrected", "xch4"]:
        if k in ds.variables:
            ch4 = ds[k].values
            break

    ds.close()
    return lat, lon, qa, ch4

def nearest_pixel_distance_km(nc_path: str, lat0: float, lon0: float):
    lat, lon, qa, ch4 = load_s5p_arrays(nc_path)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0r = math.radians(lat0)
    lon0r = math.radians(lon0)

    # Wrap longitude differences to [-pi, pi]
    dlon = (lon_rad - lon0r + np.pi) % (2 * np.pi) - np.pi
    x = dlon * np.cos(0.5 * (lat_rad + lat0r))
    y = (lat_rad - lat0r)
    dist2 = x * x + y * y

    # Find the nearest pixel
    flat_idx = np.nanargmin(dist2)
    unr = np.unravel_index(flat_idx, dist2.shape)
    if len(unr) == 3:
        it, iy, ix = unr
    elif len(unr) == 2:
        it, iy, ix = 0, unr[0], unr[1]
    else:
        raise RuntimeError(f"Unexpected latitude dims/shape: {lat.shape}")

    min_dist_km = math.sqrt(float(dist2[it, iy, ix])) * 6371.0

    qa_val = np.nan
    if qa is not None:
        v = qa[it, iy, ix]
        if np.isfinite(v):
            qa_val = float(v)

    ch4_val = np.nan
    if ch4 is not None:
        v = ch4[it, iy, ix]
        if np.isfinite(v):
            ch4_val = float(v)

    return min_dist_km, int(iy), int(ix), qa_val, ch4_val

# Translated comment
def match_all(plumes: pd.DataFrame,
              index_df: pd.DataFrame,
              time_window_hours: int = 24,
              bbox_pad_deg: float = 0.2,
              max_dist_km: float = 25.0,
              qa_thresh: float = 0.5,
              log_every: int = 200):

    results = []
    index_df = index_df.copy()

    # Translated comment
    index_df = index_df.sort_values("t_start").reset_index(drop=True)

    t_start_all = time.time()
    n = len(plumes)

    # Translated comment
    t_start_arr = index_df["t_start"].values
    t_end_arr = index_df["t_end"].values

    for i in range(n):
        row = plumes.iloc[i]
        t = row["time"]
        lat = float(row["lat"])
        lon = float(row["lon"])

        t0 = t - pd.Timedelta(hours=time_window_hours)
        t1 = t + pd.Timedelta(hours=time_window_hours)

        # Translated comment
        time_mask = (t_end_arr >= np.datetime64(t0)) & (t_start_arr <= np.datetime64(t1))
        cand = index_df.loc[time_mask]
        if len(cand) == 0:
            results.append({
                "plume_id": row["plume_id"], "plume_time": t.isoformat(),
                "lat": lat, "lon": lon,
                "matched": False, "reason": "no_time_candidate"
            })
            continue

        # Translated comment
        cand2 = bbox_filter_vec(cand, lat, lon, pad=bbox_pad_deg)
        if len(cand2) == 0:
            results.append({
                "plume_id": row["plume_id"], "plume_time": t.isoformat(),
                "lat": lat, "lon": lon,
                "matched": False, "reason": "no_bbox_candidate"
            })
            continue

        # Fine matching: compute the nearest-pixel distance for each candidate and keep the minimum
        best = None
        for _, r in cand2.iterrows():
            try:
                d_km, iy, ix, qa, ch4 = nearest_pixel_distance_km(r["file"], lat, lon)
            except Exception:
                continue

            if best is None or d_km < best["dist_km"]:
                best = {
                    "file": r["file"],
                    "proc": r.get("proc", ""),
                    "orbit": int(r["orbit"]) if "orbit" in r else -1,
                    "t_start": pd.to_datetime(r["t_start"], utc=True).isoformat() if "t_start" in r else "",
                    "t_end": pd.to_datetime(r["t_end"], utc=True).isoformat() if "t_end" in r else "",
                    "dist_km": float(d_km),
                    "iy": iy, "ix": ix,
                    "qa": qa, "ch4": ch4,
                }

        if best is None:
            results.append({
                "plume_id": row["plume_id"], "plume_time": t.isoformat(),
                "lat": lat, "lon": lon,
                "matched": False, "reason": "pixel_match_failed"
            })
            continue

        # match decision: distance + QA
        reason = ""
        if best["dist_km"] > max_dist_km:
            reason = f"too_far>{max_dist_km}km"
        elif (not np.isnan(best["qa"])) and (best["qa"] < qa_thresh):
            reason = f"low_qa<{qa_thresh}"
        matched = (reason == "")

        results.append({
            "plume_id": row["plume_id"],
            "plume_time": t.isoformat(),
            "lat": lat, "lon": lon,
            "matched": bool(matched),
            "reason": reason,
            **best
        })

        if log_every and (i % log_every == 0) and i > 0:
            elapsed = time.time() - t_start_all
            avg = elapsed / i
            # Translated comment
            print(f"[{i}/{n}] elapsed={elapsed/60:.1f}min avg={avg:.3f}s/row cache_size={load_s5p_arrays.cache_info().currsize}")

    return pd.DataFrame(results)

# ====== entry point ======
print("Loading S5P index (no rebuild) ...")
idx = build_or_load_index()
print("Index size:", len(idx))

print("Loading plumes ...")
plumes = load_plumes()
print("Plumes size:", len(plumes), "time range:", plumes["time"].min(), "->", plumes["time"].max())

print("Matching (should be much faster with vectorized bbox + LRU cache) ...")
matched_df = match_all(
    plumes,
    idx,
    time_window_hours=24,
    bbox_pad_deg=0.2,
    max_dist_km=25.0,
    qa_thresh=0.5,
    log_every=200
)

matched_df.to_csv(OUT_MATCH_CSV, index=False)
print("Saved match table:", OUT_MATCH_CSV)

if "matched" in matched_df.columns and len(matched_df) > 0:
    print("Matched rate:", float(matched_df["matched"].mean()), "(", int(matched_df["matched"].sum()), "/", len(matched_df), ")")

print(matched_df.head(10))
