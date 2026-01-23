import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import ee
import pandas as pd
from google.cloud import storage

# ====== Config ======
# 只提供“需要下载 prev 的 plume_id + 现有 l89 文件名”
INDEX_CSV = "/data2/yuyao/methane_emission/preprocess_dataset_L89/merged_plumes_with_l89_sr.csv"
# 用来拿 lat/lon（按 plume_id join）
MERGED_FILE_CSV = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv"

COMPLEMENT_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_L89_L2SR_by_gee"

GCP_PROJECT = "project-eca602a8-5837-4ae6-b4c"
GCS_BUCKET = "l89_bckt"
GCS_PREFIX = "cm_l89_sr_t1_prev"   # ✅ 建议换个前缀，避免跟 t0 混

MAX_PENDING_TASKS = 200
PENDING_TASK_SLEEP_SECONDS = 60

CLOUD_COVER_MAX = 20
CHIP_SIZE_PX = 512
SCALE_M = 30

# ✅ 关键：往回搜多少天来找“上一张过境”
# Landsat 重访 ~16 天，但有轨道重叠、L8+L9 等，所以建议 60 足够稳
SEARCH_BACK_DAYS = 60

EXPORT_SCALED_REFLECTANCE = False
USE_NATIVE_CRS = True
SKIP_GCS_SCAN = False

SR_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT", "SR_QA_AEROSOL"]
EXPORT_BANDS = SR_BANDS + QA_BANDS

# =========================
def init_gee(project_id=GCP_PROJECT):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)

def parse_acq_time_from_filename(fname: str):
    """
    GAO..._l89_sr_LANDSAT_8_20191022T173354Z.tif -> 2019-10-22T17:33:54Z
    """
    if not isinstance(fname, str):
        return None
    m = re.search(r"_(\d{8}T\d{6}Z)\.tif$", fname)
    if not m:
        return None
    s = m.group(1).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def build_region(lon: float, lat: float):
    point = ee.Geometry.Point([lon, lat])
    half_size_m = (CHIP_SIZE_PX * SCALE_M) / 2.0
    return point.buffer(half_size_m).bounds()

def merge_l89_sr_t1_collection(region, start_dt: datetime, end_dt: datetime):
    start_ee = ee.Date(start_dt.isoformat())
    end_ee = ee.Date(end_dt.isoformat())

    c8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(start_ee, end_ee)
        .filter(ee.Filter.lte("CLOUD_COVER", CLOUD_COVER_MAX))
    )
    c9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(region)
        .filterDate(start_ee, end_ee)
        .filter(ee.Filter.lte("CLOUD_COVER", CLOUD_COVER_MAX))
    )
    return c8.merge(c9)

def add_sort_keys_prev(img, anchor_dt: datetime):
    """
    只考虑“早于 anchor 的影像”，并给出：
    - time_diff_seconds = anchor - img (正数越小越好)
    """
    anchor = ee.Date(anchor_dt.isoformat())
    t = ee.Date(img.get("system:time_start"))
    diff = anchor.difference(t, "second")  # anchor - img
    return img.set("time_diff_seconds", diff)

def find_prev_overpass_image(region, anchor_dt: datetime):
    """
    在 (anchor-SEARCH_BACK_DAYS, anchor) 内找上一张过境
    约束：system:time_start < anchor_dt
    排序：time_diff_seconds 升序（最近的过去），再 cloud cover 升序
    """
    start_dt = anchor_dt - timedelta(days=SEARCH_BACK_DAYS)
    end_dt = anchor_dt  # 只搜到 anchor 之前

    col = merge_l89_sr_t1_collection(region, start_dt, end_dt)

    # 强制 < anchor
    anchor_ms = int(anchor_dt.timestamp() * 1000)
    col = col.filter(ee.Filter.lt("system:time_start", anchor_ms))

    col = col.map(lambda im: add_sort_keys_prev(im, anchor_dt))

    if col.size().getInfo() == 0:
        return None

    # ✅ 先按“离 anchor 最近”（time_diff_seconds 最小），再按云量
    col = col.sort("time_diff_seconds", True).sort("CLOUD_COVER", True)
    return ee.Image(col.first())

def maybe_scale_reflectance(img):
    if not EXPORT_SCALED_REFLECTANCE:
        return img
    sr = img.select(SR_BANDS).multiply(2.75e-05).add(-0.2)
    qa = img.select(QA_BANDS)
    out = sr.addBands(qa, overwrite=True)
    return out.copyProperties(img, img.propertyNames())

# ======= 去重：bucket 扫描（按 plume_id_prev） =======
def load_gcs_ids(bucket_name: str, prefix: str):
    ids = set()
    if SKIP_GCS_SCAN or os.getenv("SKIP_GCS_SCAN") == "1":
        print("[GCS] SKIP_GCS_SCAN enabled; skip bucket scan.")
        return ids

    client = storage.Client(project=GCP_PROJECT)
    p = prefix.strip("/")
    list_prefix = (p + "/") if p else ""

    print(f"[GCS] scanning gs://{bucket_name}/{list_prefix} ...")
    for blob in client.list_blobs(bucket_name, prefix=list_prefix):
        base = blob.name.split("/")[-1]
        # {plume_id}_l89_prev_...
        if "_l89_prev_" not in base:
            continue
        pid = base.split("_l89_prev_", 1)[0].strip()
        if pid:
            ids.add(pid)
    print(f"[GCS] found {len(ids)} prev plume_ids already in bucket")
    return ids

def load_gee_task_ids(suffix: str):
    ids = set()
    pending = 0
    try:
        tasks = ee.batch.Task.list()
    except Exception as exc:
        print(f"failed to list GEE tasks: {exc}")
        return ids, pending

    for task in tasks:
        try:
            st = task.status()
        except Exception:
            continue
        state = st.get("state")
        desc = st.get("description", "") or ""
        if state in ("READY", "RUNNING"):
            pending += 1
            if desc.endswith(suffix):
                ids.add(desc[: -len(suffix)])
    return ids, pending

def wait_for_task_capacity(suffix: str):
    while True:
        _, pending = load_gee_task_ids(suffix)
        if pending < MAX_PENDING_TASKS:
            return pending
        print(f"pending task limit reached ({pending}/{MAX_PENDING_TASKS}); sleep {PENDING_TASK_SLEEP_SECONDS}s")
        time.sleep(PENDING_TASK_SLEEP_SECONDS)

def export_prev_to_gcs(img, region, plume_id: str):
    """
    文件名：{plume_id}_l89_prev_{SPACECRAFT}_{acqtime}.tif
    """
    img = img.select(EXPORT_BANDS)
    img = img.toUint16()
    img = maybe_scale_reflectance(img)
    img = img.clip(region)

    spacecraft = ee.String(img.get("SPACECRAFT_ID"))
    acq_time = ee.Date(img.get("system:time_start")).format("YYYYMMdd'T'HHmmss'Z'")

    file_base = (
        ee.String(plume_id)
        .cat("_l89_prev_")
        .cat(spacecraft)
        .cat("_")
        .cat(acq_time)
    )

    if GCS_PREFIX:
        file_name_prefix = f"{GCS_PREFIX.strip('/')}/" + file_base.getInfo()
    else:
        file_name_prefix = file_base.getInfo()

    suffix = "_l89_prev2"
    desc = f"{plume_id}{suffix}"

    export_kwargs = dict(
        image=img,
        description=desc,
        bucket=GCS_BUCKET,
        fileNamePrefix=file_name_prefix,
        region=region,
        scale=SCALE_M,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    if not USE_NATIVE_CRS:
        export_kwargs["crs"] = "EPSG:4326"

    task = ee.batch.Export.image.toCloudStorage(**export_kwargs)
    task.start()
    return desc, file_name_prefix, suffix

# =========================
def main():
    os.makedirs(COMPLEMENT_DIR, exist_ok=True)

    print("[init] init GEE ...")
    init_gee()

    print("[init] read index csv:", INDEX_CSV)
    idx_df = pd.read_csv(INDEX_CSV)
    if not {"plume_id", "l89_filename"}.issubset(idx_df.columns):
        raise ValueError("INDEX_CSV must contain plume_id, l89_filename (and l89_path optional)")

    idx_df["plume_id"] = idx_df["plume_id"].astype(str)

    print("[init] read merged_file for lat/lon:", MERGED_FILE_CSV)
    mdf = pd.read_csv(MERGED_FILE_CSV, usecols=["plume_id", "plume_latitude", "plume_longitude"])
    mdf["plume_id"] = mdf["plume_id"].astype(str)
    loc_map = dict(zip(mdf["plume_id"], zip(mdf["plume_latitude"], mdf["plume_longitude"])))

    # 去重：tasks + bucket
    suffix = "_l89_prev2"
    task_ids, pending = load_gee_task_ids(suffix)
    gcs_ids = load_gcs_ids(GCS_BUCKET, GCS_PREFIX)
    existing = set(task_ids) | set(gcs_ids)

    print(f"[dedup] existing prev ids: {len(existing)} | pending tasks: {pending}")

    total = len(idx_df)
    for i, row in idx_df.iterrows():
        plume_id = str(row["plume_id"])
        if plume_id in existing:
            continue

        anchor_dt = parse_acq_time_from_filename(str(row.get("l89_filename", "")))
        if anchor_dt is None:
            print(f"[skip] {plume_id}: cannot parse anchor acq time from l89_filename")
            continue

        latlon = loc_map.get(plume_id)
        if not latlon:
            print(f"[skip] {plume_id}: no lat/lon in merged_file.csv")
            continue
        lat, lon = latlon
        if pd.isna(lat) or pd.isna(lon):
            print(f"[skip] {plume_id}: lat/lon is NaN")
            continue

        if pending >= MAX_PENDING_TASKS:
            pending = wait_for_task_capacity(suffix)

        region = build_region(float(lon), float(lat))

        img = find_prev_overpass_image(region, anchor_dt)
        if img is None:
            print(f"[miss] {plume_id}: no prev overpass within {SEARCH_BACK_DAYS} days")
            continue

        desc, prefix, _ = export_prev_to_gcs(img, region, plume_id)
        print(f"[ok] {i+1}/{total} {plume_id} -> task={desc} gs://{GCS_BUCKET}/{prefix}.tif")

        existing.add(plume_id)
        pending += 1
        time.sleep(0.2)

    print("All prev-overpass tasks submitted.")

if __name__ == "__main__":
    main()
