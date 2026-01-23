import os
import re
import time
from datetime import datetime, timedelta, timezone

import ee
import pandas as pd
from google.cloud import storage

# ====== Config ======
RAW_CSV = "/data2/yuyao/methane_emission/preprocess_dataset_L89/merged_plumes_with_l89_sr.csv"
MERGED_FILE_CSV = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv"

COMPLEMENT_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_L89_90360_L2SR_by_gee"

GCP_PROJECT = "project-eca602a8-5837-4ae6-b4c"
GCS_BUCKET = "l89_bckt"
GCS_PREFIX = "cm_l89_90360_sr_t1_m90_m360"  # 建议换个新前缀

MAX_PENDING_TASKS = 200
PENDING_TASK_SLEEP_SECONDS = 60

CLOUD_COVER_MAX = 20
CHIP_SIZE_PX = 512
SCALE_M = 30

OFFSETS_DAYS = [90, 360]
SEARCH_WINDOW_DAYS = 50  # target_dt ± 50d

EXPORT_SCALED_REFLECTANCE = False
USE_NATIVE_CRS = True
SKIP_GCS_SCAN = False

SR_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT", "SR_QA_AEROSOL"]
EXPORT_BANDS = SR_BANDS + QA_BANDS


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


def add_sort_keys(img, target_dt: datetime):
    anchor = ee.Date(target_dt.isoformat())
    diff = img.date().difference(anchor, "second").abs()
    return img.set("abs_time_diff_seconds", diff)


def find_best_image(region, target_dt: datetime, window_days: int):
    start_dt = target_dt - timedelta(days=window_days)
    end_dt = target_dt + timedelta(days=window_days)
    col = merge_l89_sr_t1_collection(region, start_dt, end_dt)
    col = col.map(lambda im: add_sort_keys(im, target_dt))

    if col.size().getInfo() == 0:
        return None

    col = col.sort("CLOUD_COVER", True).sort("abs_time_diff_seconds", True)
    return ee.Image(col.first())


def maybe_scale_reflectance(img):
    if not EXPORT_SCALED_REFLECTANCE:
        return img
    sr = img.select(SR_BANDS).multiply(2.75e-05).add(-0.2)
    qa = img.select(QA_BANDS)
    out = sr.addBands(qa, overwrite=True)
    return out.copyProperties(img, img.propertyNames())


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
        # {record_id}_l89_sr_...
        if "_l89_sr_" not in base:
            continue
        rid = base.split("_l89_sr_", 1)[0].strip()
        if rid:
            ids.add(rid)
    print(f"[GCS] found {len(ids)} record_ids already in bucket")
    return ids


def export_image_to_gcs(img, region, record_id: str):
    img = img.select(EXPORT_BANDS)
    img = img.toUint16()
    img = maybe_scale_reflectance(img)
    img = img.clip(region)

    spacecraft = ee.String(img.get("SPACECRAFT_ID"))
    acq_time = ee.Date(img.get("system:time_start")).format("YYYYMMdd'T'HHmmss'Z'")
    file_base = (
        ee.String(record_id)
        .cat("_l89_sr_")
        .cat(spacecraft)
        .cat("_")
        .cat(acq_time)
    )

    if GCS_PREFIX:
        file_name_prefix = f"{GCS_PREFIX.strip('/')}/" + file_base.getInfo()
    else:
        file_name_prefix = file_base.getInfo()

    suffix = "_l89_sr_m903602"
    desc = f"{record_id}{suffix}"

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


def main():
    os.makedirs(COMPLEMENT_DIR, exist_ok=True)

    print("[init] init GEE ...")
    init_gee()

    print("[init] read raw csv:", RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    need_cols = {"plume_id", "l89_filename"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"RAW_CSV must contain {need_cols}, got {df.columns.tolist()}")

    df["plume_id"] = df["plume_id"].astype(str)

    print("[init] read merged_file for lat/lon:", MERGED_FILE_CSV)
    mdf = pd.read_csv(MERGED_FILE_CSV, usecols=["plume_id", "plume_latitude", "plume_longitude"])
    mdf["plume_id"] = mdf["plume_id"].astype(str)
    loc_map = dict(zip(mdf["plume_id"], zip(mdf["plume_latitude"], mdf["plume_longitude"])))

    suffix = "_l89_sr_m903602"
    task_ids, pending = load_gee_task_ids(suffix)
    gcs_ids = load_gcs_ids(GCS_BUCKET, GCS_PREFIX)
    existing = set(task_ids) | set(gcs_ids)

    print(f"[dedup] existing record_ids: {len(existing)} | pending tasks: {pending}")

    total = len(df)
    for i, row in df.iterrows():
        plume_id = str(row["plume_id"])
        l89_fname = str(row["l89_filename"])

        anchor_dt = parse_acq_time_from_filename(l89_fname)
        if anchor_dt is None:
            print(f"[skip] {plume_id}: cannot parse anchor time from l89_filename={l89_fname}")
            continue

        latlon = loc_map.get(plume_id)
        if not latlon:
            print(f"[skip] {plume_id}: no lat/lon in merged_file.csv")
            continue
        lat, lon = latlon
        if pd.isna(lat) or pd.isna(lon):
            print(f"[skip] {plume_id}: lat/lon is NaN")
            continue

        region = build_region(float(lon), float(lat))

        for offset in OFFSETS_DAYS:
            record_id = f"{plume_id}_minus{offset}"
            if record_id in existing:
                continue

            if pending >= MAX_PENDING_TASKS:
                pending = wait_for_task_capacity(suffix)

            target_dt = anchor_dt - timedelta(days=offset)
            img = find_best_image(region, target_dt, SEARCH_WINDOW_DAYS)
            if img is None:
                print(f"[miss] {record_id}: no image within target±{SEARCH_WINDOW_DAYS}d")
                continue

            desc, prefix, _ = export_image_to_gcs(img, region, record_id)
            print(f"[ok] {i+1}/{total} {record_id} -> task={desc} gs://{GCS_BUCKET}/{prefix}.tif")

            existing.add(record_id)
            pending += 1
            time.sleep(0.2)

    print("All tasks submitted.")


if __name__ == "__main__":
    main()
