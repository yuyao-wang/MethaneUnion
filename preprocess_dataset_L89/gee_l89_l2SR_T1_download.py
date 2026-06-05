import os
import sys
import time
from datetime import datetime, timedelta, timezone

import ee
import pandas as pd

# Translated comment
from google.cloud import storage


# Translated comment
RAW_CSV = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv"

# Translated comment
COMPLEMENT_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_L89_L2SR_by_gee"

# Translated comment
# Translated comment
GCP_PROJECT = "project-eca602a8-5837-4ae6-b4c"

# Translated comment
GCS_BUCKET = "l89_bckt"
# Translated comment
GCS_PREFIX = "cm_l89_sr_t1"

# Translated comment
MAX_PENDING_TASKS = 200
PENDING_TASK_SLEEP_SECONDS = 60

# Translated comment
CLOUD_COVER_MAX = 20  # Translated comment
CHIP_SIZE_PX = 512  # Translated comment
SCALE_M = 30  # Translated comment
WINDOW_HOURS = 24     # anchor + 24h

# Translated comment
EXPORT_SCALED_REFLECTANCE = False

# Translated comment
USE_NATIVE_CRS = True

# Translated comment
SKIP_GCS_SCAN = False

# Translated comment
SR_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT", "SR_QA_AEROSOL"]
EXPORT_BANDS = SR_BANDS + QA_BANDS
# ============================


def init_gee(project_id="project-eca602a8-5837-4ae6-b4c"):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)

def log_step(message: str):
    # Translated comment
    print(message, flush=True)

def parse_iso_datetime(value: str):
    """
 merged_file.csv datetime:
 : 2019-10-19T14:52:09+00 2019-10-24T17:21:32.669192Z
 timezone-aware datetime(UTC)
    """
    if not isinstance(value, str) or len(value) == 0:
        return None
    s = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def load_downloaded_plume_ids(complement_dir: str):
    """
 output directorycomplete plume_id()
 : <plume_id>_l89_sr.tif directory l89_sr.tif
    """
    if not os.path.isdir(complement_dir):
        return set()
    plume_ids = set()
    for root, _, files in os.walk(complement_dir):
        for name in files:
            if not name.lower().endswith(".tif"):
                continue
            if name.endswith("_l89_sr.tif"):
                plume_ids.add(name[:-11])  # remove "_l89_sr.tif"
            elif name == "l89_sr.tif":
                plume_ids.add(os.path.basename(root))
    return plume_ids


def load_gee_task_plume_ids(suffix: str):
    """
 GEE task READY/RUNNING ,  description suffix ( '_l89_sr')
    """
    plume_ids = set()
    pending_count = 0
    try:
        tasks = ee.batch.Task.list()
    except Exception as exc:
        print(f"failed to list GEE tasks: {exc}")
        return plume_ids, pending_count

    for task in tasks:
        try:
            status = task.status()
        except Exception:
            continue
        state = status.get("state")
        desc = status.get("description", "") or ""
        if state in ("READY", "RUNNING"):
            pending_count += 1
            if desc.endswith(suffix):
                plume_ids.add(desc[: -len(suffix)])
    return plume_ids, pending_count


def wait_for_task_capacity(suffix: str):
    while True:
        _, pending_count = load_gee_task_plume_ids(suffix)
        if pending_count < MAX_PENDING_TASKS:
            return pending_count
        print(
            f"pending task limit reached ({pending_count}/{MAX_PENDING_TASKS}); "
            f"waiting {PENDING_TASK_SLEEP_SECONDS}s"
        )
        time.sleep(PENDING_TASK_SLEEP_SECONDS)


def build_region(lon: float, lat: float):
    """
 + bbox:     half_size(m) = CHIP_SIZE_PX * SCALE_M / 2
    """
    point = ee.Geometry.Point([lon, lat])
    half_size_m = (CHIP_SIZE_PX * SCALE_M) / 2.0
    return point.buffer(half_size_m).bounds()


def merge_l89_sr_t1_collection(region, start_dt: datetime, end_dt: datetime):
    """
 Landsat-8/9 SR Tier 1(Collection 2 Level-2)
    """
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


def add_sort_keys(img, anchor_dt: datetime):
    anchor = ee.Date(anchor_dt.isoformat())
    diff = img.date().difference(anchor, "second").abs()
    return img.set("abs_time_diff_seconds", diff)


def find_best_image(region, anchor_dt: datetime):
    end_dt = anchor_dt + timedelta(hours=WINDOW_HOURS)
    col = merge_l89_sr_t1_collection(region, anchor_dt, end_dt)
    col = col.map(lambda im: add_sort_keys(im, anchor_dt))

    if col.size().getInfo() == 0:
        return None

    col = col.sort("CLOUD_COVER", True).sort("abs_time_diff_seconds", True)
    return ee.Image(col.first())


def maybe_scale_reflectance(img):
    """
    Landsat C2 L2 SR: reflectance = SR * 2.75e-05 + (-0.2)
 QA band     """
    if not EXPORT_SCALED_REFLECTANCE:
        return img

    sr = img.select(SR_BANDS).multiply(2.75e-05).add(-0.2)
    qa = img.select(QA_BANDS)
    out = sr.addBands(qa, overwrite=True)
    return out.copyProperties(img, img.propertyNames())


# Translated comment
def load_gcs_plume_ids(bucket_name: str, prefix: str):
    """
 GCS , plume_id .
 file:       {prefix}/{plume_id}_l89_sr_{SPACECRAFT}_{YYYYmmddTHHMMSSZ}.tif
 match "_l89_sr_" plume_id.
    """
    plume_ids = set()

    if SKIP_GCS_SCAN or os.getenv("SKIP_GCS_SCAN") == "1":
        print("[GCS] SKIP_GCS_SCAN enabled; skip bucket scan.")
        return plume_ids

    # Translated comment
    client = storage.Client(project=GCP_PROJECT)

    # Translated comment
    gcs_prefix = prefix.strip("/")
    if gcs_prefix:
        list_prefix = gcs_prefix + "/"
    else:
        list_prefix = ""

    # Translated comment
    print(f"[GCS] scanning existing exports in gs://{bucket_name}/{list_prefix} ...")
    start_ts = time.time()
    scanned = 0
    for blob in client.list_blobs(bucket_name, prefix=list_prefix):
        scanned += 1
        if scanned % 1000 == 0:
            print(f"[GCS] scanned {scanned} blobs ...", flush=True)
        name = blob.name  # e.g. cm_l89_sr_t1/GAO..._l89_sr_LANDSAT_8_2019....tif
        base = name.split("/")[-1]
        if "_l89_sr_" not in base:
            continue
        # plume_id = base.split("_l89_sr_", 1)[0]
        plume_id = base.split("_l89_sr_", 1)[0].strip()
        if plume_id:
            plume_ids.add(plume_id)

    elapsed = time.time() - start_ts
    print(f"[GCS] found {len(plume_ids)} plume_ids already exported in bucket (scanned {scanned} blobs in {elapsed:.1f}s).")
    return plume_ids
# ======================================


def export_image_to_gcs(img, region, plume_id: str):
    """
 Cloud Storage:  file plume_id + sensor + acquisition time
    """
    img = img.select(EXPORT_BANDS)
    img = img.toUint16()
    img = maybe_scale_reflectance(img)
    img = img.clip(region)

    spacecraft = ee.String(img.get("SPACECRAFT_ID"))
    acq_time = ee.Date(img.get("system:time_start")).format("YYYYMMdd'T'HHmmss'Z'")
    file_base = (
        ee.String(plume_id)
        .cat("_l89_sr_")
        .cat(spacecraft)
        .cat("_")
        .cat(acq_time)
    )

    # Translated comment
    if GCS_PREFIX and len(GCS_PREFIX) > 0:
        file_name_prefix = f"{GCS_PREFIX.strip('/')}/" + file_base.getInfo()
    else:
        file_name_prefix = file_base.getInfo()

    desc = f"{plume_id}_l89_sr2"

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
    return desc, file_name_prefix


def main():
    os.makedirs(COMPLEMENT_DIR, exist_ok=True)

    log_step("[init] initializing GEE ...")
    init_gee()
    log_step(f"[init] reading csv: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    log_step(f"[init] csv rows: {len(df)}")

    suffix = "_l89_sr2"

    # Translated comment
    log_step("[dedup] scanning local outputs ...")
    existing_plume_ids = load_downloaded_plume_ids(COMPLEMENT_DIR)

    # Translated comment
    log_step("[dedup] checking GEE tasks ...")
    task_plume_ids, pending_tasks = load_gee_task_plume_ids(suffix)

    # Translated comment
    log_step("[dedup] scanning GCS bucket ...")
    gcs_plume_ids = load_gcs_plume_ids(GCS_BUCKET, GCS_PREFIX)

    existing_plume_ids |= task_plume_ids
    existing_plume_ids |= gcs_plume_ids

    if existing_plume_ids:
        print(f"found {len(existing_plume_ids)} existing plume_ids (local + pending tasks + gcs)")
    if pending_tasks:
        print(f"found {pending_tasks} pending GEE export tasks")

    total_len = len(df)
    for index, row in df.iterrows():
        plume_id = str(row.get("plume_id"))
        print(f"currently processing index {index}/{total_len} plume_id {plume_id}")

        if plume_id in existing_plume_ids:
            print(f"skip plume_id {plume_id} (exists locally / pending / gcs)")
            continue

        if pending_tasks >= MAX_PENDING_TASKS:
            pending_tasks = wait_for_task_capacity(suffix)

        anchor_dt = parse_iso_datetime(row.get("datetime"))
        if anchor_dt is None:
            print(f"skip plume_id {plume_id} due to invalid datetime {row.get('datetime')}")
            continue

        lat = row.get("plume_latitude")
        lon = row.get("plume_longitude")
        if pd.isna(lat) or pd.isna(lon):
            print(f"skip plume_id {plume_id} due to missing lat/lon")
            continue

        region = build_region(float(lon), float(lat))

        img = find_best_image(region, anchor_dt)
        if img is None:
            print(f"no suitable L8/L9 SR T1 image for plume_id {plume_id} in [anchor, anchor+{WINDOW_HOURS}h)")
            continue

        desc, prefix = export_image_to_gcs(img, region, plume_id)
        print(f"export started: task={desc}, gcs_prefix=gs://{GCS_BUCKET}/{prefix}.tif")

        # Translated comment
        existing_plume_ids.add(plume_id)
        pending_tasks += 1
        time.sleep(0.2)

    print("All tasks submitted.")


if __name__ == "__main__":
    main()
