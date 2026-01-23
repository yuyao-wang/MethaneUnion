import os
import shutil
import random
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import tifffile
from tqdm import tqdm

try:
    import imagecodecs  # noqa: F401
except Exception:
    imagecodecs = None

# =========================
# Config
# =========================
COMPLEMENT_DIR = "/data2/yuyao/methane_emission/carbonmapper_data_s2_l2a_complement_by_gee"
GEE_REF_DIR = "/data2/yuyao/methane_emission/carbonmapper_data_s2_l2a_gee_download"
MASK_DIR = "/data2/yuyao/methane_emission/carbon_mapper_data_masks"

MERGED_CSV = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv"
RAW_CSV = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/raw_s2_90360_cleaned.csv"

RAW_BASE_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_raw_s2_90360"
FIXED_BASE_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_raw_s2_90360_fixed_512"
FIXED_CSV = os.path.join(FIXED_BASE_DIR, "raw_s2_90360_cleaned_fixed.csv")

TEMPORAL_TRAIN = "/data2/yuyao/methane_emission/data_csv/s2_90360_temporal_split_cleaned/train.csv"
TEMPORAL_TEST = "/data2/yuyao/methane_emission/data_csv/s2_90360_temporal_split_cleaned/test.csv"

CHIPS_BASE_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_s2_90360_32_fixed_512_12samples"
CHIPS_TRAIN = os.path.join(CHIPS_BASE_DIR, "train.csv")
CHIPS_TEST = os.path.join(CHIPS_BASE_DIR, "test.csv")

PATCH_SIZE = 512
MIN_KEEP_RATIO = 0.8
PAD_IF_SMALLER = True

ZERO_THRESH = 0.2
BAND_INDEX_0BASED = 11

CHIP_SIZE = 32
N_POS = 16
N_NEG = 16
ZERO_RATIO_THRESH = 0.20
TEMPORAL_SPLIT = pd.Timestamp("2025-04-01", tz="UTC")

OUT_S2_RAW = "s2_raw.tif"
OUT_S2_STD = "s2_std.tif"
OUT_90_STD = "s2_90_std_512.tif"
OUT_360_STD = "s2_360_std_512.tif"


# =========================
# Helpers
# =========================
def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def crop_patch_from_latlon(src, center_lat, center_lon, patch_size=512):
    """Return patch(BHW), start_col_row, read_width, read_height."""
    try:
        center_row, center_col = src.index(center_lon, center_lat)
    except Exception:
        return None, None, 0, 0

    half = patch_size // 2
    start_col = center_col - half
    start_row = center_row - half

    read_start_col = max(start_col, 0)
    read_start_row = max(start_row, 0)
    read_end_col = min(start_col + patch_size, src.width)
    read_end_row = min(start_row + patch_size, src.height)

    read_width = read_end_col - read_start_col
    read_height = read_end_row - read_start_row
    if read_width <= 0 or read_height <= 0:
        return None, None, 0, 0

    window = Window(read_start_col, read_start_row, read_width, read_height)
    patch = src.read(window=window)

    if read_width != patch_size or read_height != patch_size:
        if not PAD_IF_SMALLER:
            return None, None, read_width, read_height
        padded = np.zeros((src.count, patch_size, patch_size), dtype=patch.dtype)
        dst_x0 = max(0, -start_col)
        dst_y0 = max(0, -start_row)
        padded[:, dst_y0:dst_y0 + read_height, dst_x0:dst_x0 + read_width] = patch
        patch = padded

    return patch, (start_col, start_row), read_width, read_height


def save_patch_geotiff(patch_bhw, out_path, src, start_col_row, patch_size=512):
    """Write patch as GeoTIFF bands-first, keeping CRS/transform if available."""
    start_col, start_row = start_col_row
    win = Window(start_col, start_row, patch_size, patch_size)
    transform = src.window_transform(win)

    meta = src.meta.copy()
    meta.update({
        "height": patch_size,
        "width": patch_size,
        "count": patch_bhw.shape[0],
        "transform": transform,
    })
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(patch_bhw)


def standardize_s2_to_rasterio(src_s2_path, dst_s2_std_path):
    """Make a rasterio-readable standard multi-band GeoTIFF from s2_path."""
    arr = tifffile.imread(src_s2_path)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected ndim for s2: {arr.ndim}, shape={arr.shape}")

    if arr.shape[0] in (12, 13):
        bhw = arr
    elif arr.shape[-1] in (12, 13):
        bhw = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Can't infer bands for s2: shape={arr.shape}")

    bands, height, width = bhw.shape
    meta = None
    try:
        with rasterio.open(src_s2_path) as src:
            meta = src.meta.copy()
            meta.update({"height": height, "width": width, "count": bands})
    except Exception:
        meta = {
            "driver": "GTiff",
            "dtype": bhw.dtype,
            "count": bands,
            "height": height,
            "width": width,
        }

    with rasterio.open(dst_s2_std_path, "w", **meta) as dst:
        dst.write(bhw)


def to_chw(data):
    if data.ndim == 2:
        return data[None, :, :]
    if data.ndim != 3:
        return None
    if data.shape[0] in (12, 13):
        return data
    if data.shape[-1] in (12, 13):
        return data.transpose(2, 0, 1)
    if data.shape[0] <= 20 and data.shape[0] < data.shape[-1]:
        return data
    return data.transpose(2, 0, 1)


def get_crop(width=512, height=512, center_size=20, crop_width=128):
    center_x = width // 2
    center_y = height // 2
    center_left = center_x - center_size // 2
    center_right = center_x + center_size // 2
    center_top = center_y - center_size // 2
    center_bottom = center_y + center_size // 2

    left_min = max(0, center_right - crop_width)
    left_max = min(center_left, width - crop_width)
    top_min = max(0, center_bottom - crop_width)
    top_max = min(center_top, height - crop_width)

    x = random.randint(left_min, left_max)
    y = random.randint(top_min, top_max)
    return x, y


def is_center_contained(crop_xy, image_size=32, width=512, height=512, center_size=10):
    center_x = width // 2
    center_y = height // 2

    cx1 = center_x - center_size // 2
    cy1 = center_y - center_size // 2
    cx2 = center_x + center_size // 2
    cy2 = center_y + center_size // 2

    x, y = crop_xy
    x1, y1 = x, y
    x2, y2 = x + image_size, y + image_size

    return (x1 <= cx1 and y1 <= cy1 and x2 >= cx2 and y2 >= cy2)


class ThreadSafeCounter:
    def __init__(self, start=0):
        self.lock = threading.Lock()
        self.counter = start

    def increment(self):
        with self.lock:
            self.counter += 1
            return self.counter

    def get(self):
        with self.lock:
            return self.counter


def band12_zero_too_much(arr_chw, band_index=BAND_INDEX_0BASED, zero_ratio_thresh=ZERO_RATIO_THRESH):
    total = arr_chw[band_index].size
    if total == 0:
        return True
    zero_count = np.sum(arr_chw[band_index] == 0)
    return (zero_count / total) >= zero_ratio_thresh


def crop_chw(arr_chw, crop_xy, size):
    x, y = crop_xy
    return arr_chw[:, y:y + size, x:x + size]


def crop_hw(arr_hw, crop_xy, size):
    x, y = crop_xy
    return arr_hw[y:y + size, x:x + size]


# =========================
# Step 1-3: Build raw dataset + CSV
# =========================
def build_new_raw_rows():
    raw_df = pd.read_csv(RAW_CSV)
    raw_ids = set(raw_df["plume_id"].astype(str))
    raw_cols = raw_df.columns.tolist()

    merged_df = pd.read_csv(MERGED_CSV).set_index("plume_id")

    complement_ids = []
    for fname in os.listdir(COMPLEMENT_DIR):
        if fname.endswith("_s2.tif"):
            complement_ids.append(fname[:-7])
    complement_ids = sorted(set(complement_ids))

    new_ids = [pid for pid in complement_ids if pid not in raw_ids]
    print(f"Found {len(new_ids)} new plume_ids from complement_by_gee.")

    new_rows = []
    skipped = 0
    for plume_id in tqdm(new_ids, desc="Build raw rows"):
        if plume_id not in merged_df.index:
            skipped += 1
            continue

        meta = merged_df.loc[plume_id]
        lat = meta.get("plume_latitude")
        lon = meta.get("plume_longitude")
        dt = meta.get("datetime")
        if pd.isna(lat) or pd.isna(lon) or not isinstance(dt, str):
            skipped += 1
            continue

        s2_src = os.path.join(COMPLEMENT_DIR, f"{plume_id}_s2.tif")
        if not os.path.exists(s2_src):
            skipped += 1
            continue

        ref_month = os.path.join(GEE_REF_DIR, f"{plume_id}_reference_month.tif")
        ref_year = os.path.join(GEE_REF_DIR, f"{plume_id}_reference_year.tif")
        if not os.path.exists(ref_month) or not os.path.exists(ref_year):
            skipped += 1
            continue

        mask_dir = os.path.join(MASK_DIR, plume_id)
        plume_mask = os.path.join(mask_dir, "plume.tif")
        reprojected = os.path.join(mask_dir, "reprojected.tif")
        resized = os.path.join(mask_dir, "resized_512x512.tif")
        if not (os.path.exists(plume_mask) and os.path.exists(reprojected) and os.path.exists(resized)):
            skipped += 1
            continue

        out_dir = os.path.join(RAW_BASE_DIR, plume_id)
        safe_mkdir(out_dir)

        out_s2 = os.path.join(out_dir, "s2.tif")
        if not os.path.exists(out_s2):
            try:
                with rasterio.open(s2_src) as src:
                    patch, start, rw, rh = crop_patch_from_latlon(src, lat, lon, PATCH_SIZE)
                    if patch is None or (rw < PATCH_SIZE * MIN_KEEP_RATIO or rh < PATCH_SIZE * MIN_KEEP_RATIO):
                        skipped += 1
                        continue
                    save_patch_geotiff(patch, out_s2, src, start, PATCH_SIZE)
            except Exception:
                skipped += 1
                continue

        out_ref_month = os.path.join(out_dir, os.path.basename(ref_month))
        out_ref_year = os.path.join(out_dir, os.path.basename(ref_year))
        if not os.path.exists(out_ref_month):
            shutil.copy(ref_month, out_ref_month)
        if not os.path.exists(out_ref_year):
            shutil.copy(ref_year, out_ref_year)

        out_plume = os.path.join(out_dir, "plume.tif")
        out_reprojected = os.path.join(out_dir, "reprojected.tif")
        out_resized = os.path.join(out_dir, "resized_512x512.tif")
        if not os.path.exists(out_plume):
            shutil.copy(plume_mask, out_plume)
        if not os.path.exists(out_reprojected):
            shutil.copy(reprojected, out_reprojected)
        if not os.path.exists(out_resized):
            shutil.copy(resized, out_resized)

        row = {col: None for col in raw_cols}
        row["plume_id"] = plume_id
        row["datetime"] = dt
        row["plume_latitude"] = float(lat)
        row["plume_longitude"] = float(lon)
        row["s2_path"] = out_s2
        row["s2_90_path"] = out_ref_month
        row["s2_360_path"] = out_ref_year
        row["plume_path"] = out_plume
        row["reprojected_path"] = out_reprojected
        row["resized_512x512_path"] = out_resized
        new_rows.append(row)

    print(f"Raw rows ready: {len(new_rows)}, skipped: {skipped}")
    return raw_df, new_rows


def append_raw_csv(raw_df, new_rows):
    if not new_rows:
        print("No new raw rows to append.")
        return pd.DataFrame(columns=raw_df.columns), raw_df

    new_df = pd.DataFrame(new_rows, columns=raw_df.columns)
    out_df = pd.concat([raw_df, new_df], ignore_index=True)
    out_df.to_csv(RAW_CSV, index=False)
    print(f"Appended {len(new_df)} rows to {RAW_CSV}.")
    return new_df, out_df


# =========================
# Step 4: Build fixed dataset + CSV
# =========================
def build_fixed_rows(new_raw_df):
    if new_raw_df.empty:
        print("No new rows for fixed dataset.")
        return pd.DataFrame()

    if os.path.exists(FIXED_CSV):
        fixed_df = pd.read_csv(FIXED_CSV)
        fixed_cols = fixed_df.columns.tolist()
        fixed_ids = set(fixed_df["plume_id"].astype(str))
    else:
        fixed_cols = [
            "plume_id", "datetime", "plume_latitude", "plume_longitude",
            "s2_path", "s2_90_path", "s2_360_path",
            "plume_path", "reprojected_path", "resized_512x512_path",
            "s2_path_raw_copy", "s2_path_std", "s2_90_path_std", "s2_360_path_std",
        ]
        fixed_ids = set()

    new_fixed_rows = []
    for _, row in tqdm(new_raw_df.iterrows(), total=len(new_raw_df), desc="Build fixed rows"):
        plume_id = str(row["plume_id"])
        if plume_id in fixed_ids:
            continue

        center_lat = row["plume_latitude"]
        center_lon = row["plume_longitude"]

        out_dir = os.path.join(FIXED_BASE_DIR, plume_id)
        safe_mkdir(out_dir)

        # 0) copy s2 raw
        ps2 = row.get("s2_path")
        if not (isinstance(ps2, str) and os.path.exists(ps2)):
            continue

        out_s2_raw = os.path.join(out_dir, OUT_S2_RAW)
        if not os.path.exists(out_s2_raw):
            shutil.copy(ps2, out_s2_raw)

        # 0b) standardize s2
        out_s2_std = os.path.join(out_dir, OUT_S2_STD)
        if not os.path.exists(out_s2_std):
            try:
                standardize_s2_to_rasterio(ps2, out_s2_std)
            except Exception:
                out_s2_std = None

        # 1) crop 90d
        out90 = os.path.join(out_dir, OUT_90_STD)
        p90 = row.get("s2_90_path")
        if isinstance(p90, str) and os.path.exists(p90):
            try:
                with rasterio.open(p90) as src:
                    patch90, start90, rw, rh = crop_patch_from_latlon(src, center_lat, center_lon, PATCH_SIZE)
                    if patch90 is not None and (rw >= PATCH_SIZE * MIN_KEEP_RATIO and rh >= PATCH_SIZE * MIN_KEEP_RATIO):
                        save_patch_geotiff(patch90, out90, src, start90, PATCH_SIZE)
                    else:
                        out90 = None
            except Exception:
                out90 = None
        else:
            out90 = None

        # 2) crop 360d
        out360 = os.path.join(out_dir, OUT_360_STD)
        p360 = row.get("s2_360_path")
        if isinstance(p360, str) and os.path.exists(p360):
            try:
                with rasterio.open(p360) as src:
                    patch360, start360, rw, rh = crop_patch_from_latlon(src, center_lat, center_lon, PATCH_SIZE)
                    if patch360 is not None and (rw >= PATCH_SIZE * MIN_KEEP_RATIO and rh >= PATCH_SIZE * MIN_KEEP_RATIO):
                        save_patch_geotiff(patch360, out360, src, start360, PATCH_SIZE)
                    else:
                        out360 = None
            except Exception:
                out360 = None
        else:
            out360 = None

        fixed_row = {col: None for col in fixed_cols}
        for col in [
            "plume_id", "datetime", "plume_latitude", "plume_longitude",
            "s2_path", "s2_90_path", "s2_360_path",
            "plume_path", "reprojected_path", "resized_512x512_path",
        ]:
            fixed_row[col] = row.get(col)

        fixed_row["s2_path_raw_copy"] = out_s2_raw
        fixed_row["s2_path_std"] = out_s2_std
        fixed_row["s2_90_path_std"] = out90
        fixed_row["s2_360_path_std"] = out360
        new_fixed_rows.append(fixed_row)

    fixed_new_df = pd.DataFrame(new_fixed_rows, columns=fixed_cols)
    if os.path.exists(FIXED_CSV):
        fixed_df = pd.read_csv(FIXED_CSV)
        fixed_out = pd.concat([fixed_df, fixed_new_df], ignore_index=True)
    else:
        fixed_out = fixed_new_df
    fixed_out.to_csv(FIXED_CSV, index=False)
    print(f"Appended {len(fixed_new_df)} rows to {FIXED_CSV}.")
    return fixed_new_df


# =========================
# Step 5: Update temporal split
# =========================
def update_temporal_split(new_raw_df):
    if new_raw_df.empty:
        print("No new rows for temporal split.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(TEMPORAL_TRAIN)
    test_df = pd.read_csv(TEMPORAL_TEST)
    cols = train_df.columns.tolist()

    train_rows = []
    test_rows = []

    for _, row in new_raw_df.iterrows():
        dt = pd.to_datetime(row["datetime"], utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        date_str = dt.isoformat(sep=" ")

        out_row = {col: None for col in cols}
        for col in cols:
            if col == "date":
                out_row["date"] = date_str
            else:
                out_row[col] = row.get(col)

        if dt < TEMPORAL_SPLIT:
            train_rows.append(out_row)
        else:
            test_rows.append(out_row)

    new_train_df = pd.DataFrame(train_rows, columns=cols)
    new_test_df = pd.DataFrame(test_rows, columns=cols)

    if not new_train_df.empty:
        train_out = pd.concat([train_df, new_train_df], ignore_index=True)
        train_out.to_csv(TEMPORAL_TRAIN, index=False)
        print(f"Appended {len(new_train_df)} rows to {TEMPORAL_TRAIN}.")
    else:
        print("No new rows for train split.")

    if not new_test_df.empty:
        test_out = pd.concat([test_df, new_test_df], ignore_index=True)
        test_out.to_csv(TEMPORAL_TEST, index=False)
        print(f"Appended {len(new_test_df)} rows to {TEMPORAL_TEST}.")
    else:
        print("No new rows for test split.")

    return new_train_df, new_test_df


# =========================
# Step 6: Crop chips and update chips CSVs
# =========================
def process_row(row, index, total_count, output_cnt, plume_cnt):
    t_data = to_chw(tifffile.imread(row["s2_path"]))
    t1_data = to_chw(tifffile.imread(row["s2_90_path"]))
    t2_data = to_chw(tifffile.imread(row["s2_360_path"]))
    mask = tifffile.imread(row["resized_512x512_path"])

    if t_data is None or t1_data is None or t2_data is None or mask is None:
        return []

    data = []

    crop_list = [get_crop(crop_width=CHIP_SIZE) for _ in range(N_POS)]
    for crop_xy in crop_list:
        nt_data = crop_chw(t_data, crop_xy, CHIP_SIZE)
        nt1_data = crop_chw(t1_data, crop_xy, CHIP_SIZE)
        nt2_data = crop_chw(t2_data, crop_xy, CHIP_SIZE)
        n_mask = crop_hw(mask, crop_xy, CHIP_SIZE)

        if nt_data.shape[-2:] != (CHIP_SIZE, CHIP_SIZE) or n_mask.shape != (CHIP_SIZE, CHIP_SIZE):
            continue

        if band12_zero_too_much(nt_data) or band12_zero_too_much(nt1_data) or band12_zero_too_much(nt2_data):
            continue

        cnt = output_cnt.increment()
        dir_path = os.path.join(CHIPS_BASE_DIR, str(cnt))
        os.makedirs(dir_path, exist_ok=True)

        nt_path = os.path.join(dir_path, "s2.tif")
        nt1_path = os.path.join(dir_path, "s2_90.tif")
        nt2_path = os.path.join(dir_path, "s2_360.tif")
        n_mask_path = os.path.join(dir_path, "plume.tif")

        tifffile.imwrite(nt_path, nt_data)
        tifffile.imwrite(nt1_path, nt1_data)
        tifffile.imwrite(nt2_path, nt2_data)
        tifffile.imwrite(n_mask_path, n_mask)

        data.append({
            "id": cnt,
            "s2_path": nt_path,
            "s2_90_path": nt1_path,
            "s2_360_path": nt2_path,
            "plume_mask_path": n_mask_path,
            "label": 1,
            "latitude": row["plume_latitude"],
            "longitude": row["plume_longitude"],
            "datetime": row["datetime"],
        })
        plume_cnt.increment()

    crop_list = [(random.randint(0, 512 - CHIP_SIZE), random.randint(0, 512 - CHIP_SIZE)) for _ in range(N_NEG)]
    for crop_xy in crop_list:
        nt_data = crop_chw(t_data, crop_xy, CHIP_SIZE)
        nt1_data = crop_chw(t1_data, crop_xy, CHIP_SIZE)
        nt2_data = crop_chw(t2_data, crop_xy, CHIP_SIZE)
        n_mask = crop_hw(mask, crop_xy, CHIP_SIZE)

        if nt_data.shape[-2:] != (CHIP_SIZE, CHIP_SIZE) or n_mask.shape != (CHIP_SIZE, CHIP_SIZE):
            continue

        if band12_zero_too_much(nt_data) or band12_zero_too_much(nt1_data) or band12_zero_too_much(nt2_data):
            continue

        cnt = output_cnt.increment()
        dir_path = os.path.join(CHIPS_BASE_DIR, str(cnt))
        os.makedirs(dir_path, exist_ok=True)

        nt_path = os.path.join(dir_path, "s2.tif")
        nt1_path = os.path.join(dir_path, "s2_90.tif")
        nt2_path = os.path.join(dir_path, "s2_360.tif")

        label = is_center_contained(crop_xy, image_size=CHIP_SIZE)
        if not label:
            n_mask = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=mask.dtype)
        else:
            plume_cnt.increment()

        n_mask_path = os.path.join(dir_path, "plume.tif")

        tifffile.imwrite(nt_path, nt_data)
        tifffile.imwrite(nt1_path, nt1_data)
        tifffile.imwrite(nt2_path, nt2_data)
        tifffile.imwrite(n_mask_path, n_mask)

        data.append({
            "id": cnt,
            "s2_path": nt_path,
            "s2_90_path": nt1_path,
            "s2_360_path": nt2_path,
            "plume_mask_path": n_mask_path,
            "label": 0 if not label else 1,
            "latitude": row["plume_latitude"],
            "longitude": row["plume_longitude"],
            "datetime": row["datetime"],
        })

    print(f"processed {index} / {total_count} plume_cnt {plume_cnt.get()} / {output_cnt.get()}")
    return data


def run_split_df(df, out_csv_path, output_cnt, plume_cnt):
    if df.empty:
        return pd.DataFrame()

    data_all = []
    batch_size = 32
    total_rows = len(df)
    num_batches = math.ceil(total_rows / batch_size)

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]

        futures = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            for index, row in batch_df.iterrows():
                futures.append(executor.submit(process_row, row, index, total_rows, output_cnt, plume_cnt))
            for future in futures:
                data_all.extend(future.result())

    out_df = pd.DataFrame(data_all)

    if os.path.exists(out_csv_path):
        org_df = pd.read_csv(out_csv_path)
        out_df = pd.concat([org_df, out_df], ignore_index=True)

    out_df.to_csv(out_csv_path, index=False)
    print(f"wrote {out_csv_path}, size={len(out_df)}")
    return out_df


def update_chips(new_train_df, new_test_df):
    if new_train_df.empty and new_test_df.empty:
        print("No new rows for chips.")
        return

    safe_mkdir(CHIPS_BASE_DIR)

    max_id = 0
    for path in [CHIPS_TRAIN, CHIPS_TEST]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "id" in df.columns and not df.empty:
                max_id = max(max_id, int(df["id"].max()))

    output_cnt = ThreadSafeCounter(start=max_id)
    plume_cnt = ThreadSafeCounter()

    run_split_df(new_train_df, CHIPS_TRAIN, output_cnt, plume_cnt)
    run_split_df(new_test_df, CHIPS_TEST, output_cnt, plume_cnt)


# =========================
# Main
# =========================
if __name__ == "__main__":
    raw_df, new_rows = build_new_raw_rows()
    new_raw_df, _ = append_raw_csv(raw_df, new_rows)

    fixed_new_df = build_fixed_rows(new_raw_df)
    new_train_df, new_test_df = update_temporal_split(new_raw_df)

    update_chips(new_train_df, new_test_df)

    print("All tasks completed.")
