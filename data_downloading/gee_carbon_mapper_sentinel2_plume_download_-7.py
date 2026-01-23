import os
import sys
import time
from datetime import datetime, timedelta

import ee
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.utils import parse_args

RAW_CSV = '/data2/yuyao/methane_emission/carbon_mapper_data/csvs/raw_s2_90360_cleaned.csv'
COMPLEMENT_DIR = '/data2/yuyao/methane_emission/carbonmapper_data_s2_l2a_complement_by_gee_-7'
CLOUD_COVER_MAX = 20
BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
DRIVE_FOLDER = 'CM_S2_L2A-7'
DRIVE_CREDENTIALS = os.path.join(os.path.dirname(__file__), 'credentials.json')
DRIVE_TOKEN = os.path.join(os.path.dirname(__file__), 'token.json')
DRIVE_USE_SERVICE_ACCOUNT = False
MAX_PENDING_TASKS = 200
PENDING_TASK_SLEEP_SECONDS = 60

try:
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    build = None

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def init_gee():
    print("Initializing Earth Engine...")
    ee.Authenticate()
    ee.Initialize()
    print("Earth Engine initialized.")


def parse_iso_datetime(value):
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def load_downloaded_plume_ids(complement_dir):
    if not os.path.isdir(complement_dir):
        return set()
    plume_ids = set()
    for root, _, files in os.walk(complement_dir):
        for name in files:
            if not name.lower().endswith('.tif'):
                continue
            if name.endswith('_s2.tif'):
                plume_ids.add(name[:-7])
            elif name == 'plume.tif':
                plume_ids.add(os.path.basename(root))
    return plume_ids


def build_drive_service():
    if build is None:
        print("Google Drive API deps not installed; skip Drive check.")
        return None
    if DRIVE_USE_SERVICE_ACCOUNT:
        if not os.path.exists(DRIVE_CREDENTIALS):
            print(f"Drive credentials not found at {DRIVE_CREDENTIALS}; skip Drive check.")
            return None
        creds = service_account.Credentials.from_service_account_file(
            DRIVE_CREDENTIALS, scopes=SCOPES
        )
    else:
        creds = None
        if os.path.exists(DRIVE_TOKEN):
            creds = Credentials.from_authorized_user_file(DRIVE_TOKEN, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(DRIVE_CREDENTIALS):
                    print(f"Drive OAuth credentials not found at {DRIVE_CREDENTIALS}; skip Drive check.")
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(
                    DRIVE_CREDENTIALS, SCOPES
                )
                creds = flow.run_console()
            with open(DRIVE_TOKEN, "w") as fh:
                fh.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_folder_id(service, folder_id, folder_name):
    if folder_id:
        return folder_id
    safe_name = folder_name.replace("'", "\\'")
    query = f"mimeType='{FOLDER_MIME_TYPE}' and name='{safe_name}' and trashed=false"
    resp = service.files().list(q=query, fields="files(id,name)").execute()
    matches = resp.get("files", [])
    if not matches:
        raise RuntimeError(f"Drive folder not found: {folder_name}")
    if len(matches) > 1:
        msg = ", ".join([f"{f['name']}({f['id']})" for f in matches])
        raise RuntimeError(f"Multiple folders found: {msg}")
    return matches[0]["id"]


def iter_folder_files(service, folder_id):
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,mimeType)"
    while True:
        resp = service.files().list(q=query, fields=fields, pageToken=page_token).execute()
        for item in resp.get("files", []):
            yield item
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def load_drive_plume_ids(drive_folder_name):
    print(f"Checking Drive folder '{drive_folder_name}' for existing exports...")
    service = build_drive_service()
    if service is None:
        return set()
    try:
        folder_id = resolve_folder_id(service, "", drive_folder_name)
    except Exception as exc:
        print(f"skip Drive check for {drive_folder_name}: {exc}")
        return set()

    plume_ids = set()
    for item in iter_folder_files(service, folder_id):
        if item.get("mimeType") == FOLDER_MIME_TYPE:
            continue
        name = item.get("name", "")
        if not name.lower().endswith(".tif"):
            continue
        if name.endswith("_s2.tif"):
            plume_ids.add(name[:-7])
            continue
        base = name[:-4]
        if base.endswith("_s2"):
            plume_ids.add(base[:-3])
    return plume_ids


def load_gee_task_plume_ids():
    plume_ids = set()
    pending_count = 0
    try:
        print("Listing GEE tasks...")
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
        description = status.get("description", "")
        if state in ("READY", "RUNNING"):
            pending_count += 1
            if description.endswith("_s2"):
                plume_ids.add(description[:-3])
    return plume_ids, pending_count


def wait_for_task_capacity():
    while True:
        _, pending_count = load_gee_task_plume_ids()
        if pending_count < MAX_PENDING_TASKS:
            return pending_count
        print(
            f'pending task limit reached ({pending_count}/{MAX_PENDING_TASKS}); '
            f'waiting {PENDING_TASK_SLEEP_SECONDS}s'
        )
        time.sleep(PENDING_TASK_SLEEP_SECONDS)


def build_region(lon, lat):
    point = ee.Geometry.Point(lon, lat)
    half_size = 512 * 20 / 2
    return point.buffer(half_size).bounds()


def find_best_image(region, start_datetime, end_datetime):
    print(
        f"Searching images between {start_datetime.isoformat()} and "
        f"{end_datetime.isoformat()} with cloud < {CLOUD_COVER_MAX}%"
    )
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region)
        .filterDate(ee.Date(start_datetime.isoformat()), ee.Date(end_datetime.isoformat()))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER_MAX))
        .sort('system:time_start', False)
    )
    start_check = time.time()
    print("Checking collection size (getInfo)...")
    size = collection.size().getInfo()
    elapsed = time.time() - start_check
    print(f"Collection size: {size} (checked in {elapsed:.2f}s)")
    if size == 0:
        return None
    return ee.Image(collection.first())


def export_image_to_drive(image, region, plume_id):
    image = image.select(BANDS).reproject(crs='EPSG:4326', scale=20).clip(region)
    file_prefix = f'{plume_id}_s2'
    print(f"Starting export task for {file_prefix}...")
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=file_prefix,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_prefix,
        region=region,
        scale=20,
        crs='EPSG:4326',
    )
    task.start()


if __name__ == '__main__':
    parse_args()
    init_gee()

    print(f"Loading CSV from {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows.")
    existing_plume_ids = load_downloaded_plume_ids(COMPLEMENT_DIR)
    drive_plume_ids = load_drive_plume_ids(DRIVE_FOLDER)
    task_plume_ids, pending_tasks = load_gee_task_plume_ids()
    existing_plume_ids |= drive_plume_ids
    existing_plume_ids |= task_plume_ids
    if existing_plume_ids:
        print(f'found {len(existing_plume_ids)} existing plume files (local + Drive)')
    if pending_tasks:
        print(f'found {pending_tasks} pending GEE export tasks')

    total_len = len(df)
    for index, row in df.iterrows():
        plume_id = str(row.get('plume_id'))
        print(f'currently processing index {index}/{total_len} plume_id {plume_id}')

        if plume_id in existing_plume_ids:
            print(f'skip plume_id {plume_id} already in Drive or {COMPLEMENT_DIR}')
            continue
        if pending_tasks >= MAX_PENDING_TASKS:
            pending_tasks = wait_for_task_capacity()

        parsed_time = parse_iso_datetime(row.get('datetime'))
        if parsed_time is None:
            print(f'skip plume_id {plume_id} due to invalid datetime {row.get("datetime")}')
            continue

        lat = row.get('plume_latitude')
        lon = row.get('plume_longitude')
        if lat is None or lon is None:
            print(f'skip plume_id {plume_id} due to missing lat/lon')
            continue

        target_end = parsed_time - timedelta(days=7)
        target_start = parsed_time - timedelta(days=50)

        print(f"Building region for lon={lon}, lat={lat}")
        region = build_region(lon, lat)
        image = find_best_image(region, target_start, target_end)
        if image is None:
            print(f'no suitable image for plume_id {plume_id} between {target_start} and {target_end}')
            continue

        print(
            f'exporting plume_id {plume_id} for latest image between '
            f'{target_start} and {target_end} to Drive folder {DRIVE_FOLDER}'
        )
        export_image_to_drive(image, region, plume_id)
        pending_tasks += 1
        time.sleep(0.2)

    print('All tasks completed.')
