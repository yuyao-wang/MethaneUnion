#!/usr/bin/env python3
import argparse
import io
import os
import sys
import time
from datetime import datetime, timezone

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
except ImportError:
    print(
        "Missing deps. Install with:\n"
        "  pip install google-api-python-client google-auth google-auth-oauthlib",
        file=sys.stderr,
    )
    sys.exit(1)

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def parse_args():
    p = argparse.ArgumentParser(
        description="Download files from Drive folder and delete after download."
    )
    p.add_argument("--drive-folder", default="CM_S2_L2A_3_90_365")
    p.add_argument("--drive-folder-id", default="")
    p.add_argument("--local-dir", required=True)
    p.add_argument("--credentials", default="credentials.json")
    p.add_argument("--token", default="token.json")
    p.add_argument("--service-account", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--once", action="store_true")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--min-age-seconds", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--delete-after", dest="delete_after", action="store_true", default=True)
    p.add_argument("--no-delete", dest="delete_after", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_drive_service(args):
    if args.service_account:
        creds = service_account.Credentials.from_service_account_file(
            args.credentials, scopes=SCOPES
        )
    else:
        creds = None
        if os.path.exists(args.token):
            creds = Credentials.from_authorized_user_file(args.token, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    args.credentials, SCOPES
                )
                creds = flow.run_console()
            with open(args.token, "w") as f:
                f.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_folder_id(service, folder_id, folder_name):
    if folder_id:
        return folder_id
    safe_name = folder_name.replace("'", "\\'")
    query = f"mimeType='{FOLDER_MIME_TYPE}' and name='{safe_name}' and trashed=false"
    resp = service.files().list(q=query, fields="files(id,name,parents)").execute()
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
    fields = "nextPageToken, files(id,name,size,mimeType,modifiedTime)"
    while True:
        resp = (
            service.files()
            .list(q=query, fields=fields, pageToken=page_token, orderBy="createdTime")
            .execute()
        )
        for item in resp.get("files", []):
            yield item
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def parse_drive_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def should_skip_by_age(item, min_age_seconds):
    if min_age_seconds <= 0:
        return False
    modified = parse_drive_time(item.get("modifiedTime"))
    if not modified:
        return False
    age = datetime.now(timezone.utc) - modified
    return age.total_seconds() < min_age_seconds


def download_file(service, item, dest_path, chunk_size):
    request = service.files().get_media(fileId=item["id"])
    tmp_path = dest_path + ".part"
    with io.FileIO(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    os.replace(tmp_path, dest_path)


def file_size_matches(item, local_path):
    if "size" not in item:
        return True
    try:
        return os.path.getsize(local_path) == int(item["size"])
    except OSError:
        return False


def process_once(service, folder_id, args):
    os.makedirs(args.local_dir, exist_ok=True)
    processed = 0
    for item in iter_folder_files(service, folder_id):
        if item.get("mimeType") == FOLDER_MIME_TYPE:
            continue
        if should_skip_by_age(item, args.min_age_seconds):
            continue

        name = item["name"]
        dest_path = os.path.join(args.local_dir, name)

        if os.path.exists(dest_path) and not args.overwrite:
            if file_size_matches(item, dest_path):
                print(f"skip existing {name}")
                if args.delete_after and not args.dry_run:
                    service.files().delete(fileId=item["id"]).execute()
                processed += 1
            else:
                print(f"local file exists with different size: {name}")
            if args.max_files and processed >= args.max_files:
                break
            continue

        if args.dry_run:
            print(f"dry-run download {name}")
            processed += 1
            if args.max_files and processed >= args.max_files:
                break
            continue

        try:
            print(f"downloading {name}")
            download_file(service, item, dest_path, chunk_size=8 * 1024 * 1024)
            if not file_size_matches(item, dest_path):
                print(f"size mismatch after download: {name}")
                continue
            if args.delete_after:
                service.files().delete(fileId=item["id"]).execute()
                print(f"deleted from drive: {name}")
            processed += 1
        except HttpError as exc:
            print(f"drive error for {name}: {exc}")
        except Exception as exc:
            print(f"download failed for {name}: {exc}")

        if args.max_files and processed >= args.max_files:
            break
    return processed


def main():
    args = parse_args()
    service = build_drive_service(args)
    folder_id = resolve_folder_id(service, args.drive_folder_id, args.drive_folder)

    while True:
        try:
            count = process_once(service, folder_id, args)
            if args.once:
                break
            if count == 0:
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"sync loop error: {exc}")
            if args.once:
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
