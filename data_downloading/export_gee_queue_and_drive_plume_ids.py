import argparse
import os
from datetime import datetime

import ee
import pandas as pd

DRIVE_CREDENTIALS = os.path.join(os.path.dirname(__file__), 'credentials.json')
DRIVE_TOKEN = os.path.join(os.path.dirname(__file__), 'token.json')
DRIVE_USE_SERVICE_ACCOUNT = False

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export GEE pending tasks and Drive files to a CSV for skipping."
    )
    parser.add_argument(
        "--drive-folder",
        default="CM_S2_L2A",
        help="Drive folder name to scan for *_s2.tif files.",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(os.path.dirname(__file__), "gee_queue_drive_plume_ids.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def init_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


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
    fields = "nextPageToken, files(id,name,mimeType,modifiedTime)"
    while True:
        resp = service.files().list(q=query, fields=fields, pageToken=page_token).execute()
        for item in resp.get("files", []):
            yield item
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def load_drive_plume_ids(drive_folder_name):
    service = build_drive_service()
    if service is None:
        return []
    folder_id = resolve_folder_id(service, "", drive_folder_name)
    records = []
    for item in iter_folder_files(service, folder_id):
        if item.get("mimeType") == FOLDER_MIME_TYPE:
            continue
        name = item.get("name", "")
        if not name.lower().endswith(".tif"):
            continue
        plume_id = None
        if name.endswith("_s2.tif"):
            plume_id = name[:-7]
        else:
            base = name[:-4]
            if base.endswith("_s2"):
                plume_id = base[:-3]
        if plume_id:
            records.append(
                {
                    "plume_id": plume_id,
                    "source": "drive",
                    "name": name,
                    "modifiedTime": item.get("modifiedTime", ""),
                }
            )
    return records


def load_gee_task_plume_ids():
    records = []
    tasks = ee.batch.Task.list()
    for task in tasks:
        try:
            status = task.status()
        except Exception:
            continue
        state = status.get("state")
        description = status.get("description", "")
        if state not in ("READY", "RUNNING"):
            continue
        plume_id = None
        if description.endswith("_s2"):
            plume_id = description[:-3]
        if plume_id:
            records.append(
                {
                    "plume_id": plume_id,
                    "source": "gee_task",
                    "name": description,
                    "state": state,
                }
            )
    return records


def main():
    args = parse_args()
    init_gee()
    records = []
    records.extend(load_gee_task_plume_ids())
    records.extend(load_drive_plume_ids(args.drive_folder))
    df = pd.DataFrame(records)
    if df.empty:
        print("No queue/drive plume ids found.")
        return
    df["exported_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} records to {args.output_csv}")


if __name__ == "__main__":
    main()
