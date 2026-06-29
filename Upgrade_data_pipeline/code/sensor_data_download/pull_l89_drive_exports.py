#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from manifest_state import ensure_manifest_columns, row_download_done, update_master_from_records

try:
    import rasterio
except ImportError:
    rasterio = None

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
TIMEPOINTS = {"t0", "prev1", "prev2", "prev3", "seasonal", "year"}
SUCCESS_STATUSES = {
    "downloaded",
    "downloaded_deleted_drive",
    "skip_existing_valid",
    "skip_existing_valid_deleted_drive",
    "resume_skip_completed",
    "resume_skip_completed_deleted_drive",
}


def log(message: str) -> None:
    print(message, flush=True)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def parse_drive_time(value: Any) -> Optional[datetime]:
    if not has_value(value):
        return None


def parse_file_image_time(value: Any) -> str:
    if not has_value(value):
        return ""
    text = str(value).strip()
    try:
        dt = datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return ""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def should_skip_by_age(item: dict[str, Any], min_age_seconds: int) -> bool:
    if min_age_seconds <= 0:
        return False
    modified = parse_drive_time(item.get("modifiedTime"))
    if modified is None:
        return False
    age = datetime.now(timezone.utc) - modified
    return age.total_seconds() < min_age_seconds


def build_drive_service(args: argparse.Namespace):
    if args.service_account:
        creds = service_account.Credentials.from_service_account_file(
            args.credentials,
            scopes=SCOPES,
        )
    else:
        creds = None
        if os.path.exists(args.token):
            creds = Credentials.from_authorized_user_file(args.token, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(args.credentials, SCOPES)
                creds = flow.run_console()
            with open(args.token, "w") as fh:
                fh.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_folder_id(service: Any, folder_id: str, folder_name: str) -> str:
    if folder_id:
        return folder_id
    safe_name = folder_name.replace("'", "\\'")
    query = f"mimeType='{FOLDER_MIME_TYPE}' and name='{safe_name}' and trashed=false"
    resp = service.files().list(q=query, fields="files(id,name,parents)").execute()
    matches = resp.get("files", [])
    if not matches:
        raise RuntimeError(f"Drive folder not found: {folder_name}")
    if len(matches) > 1:
        names = ", ".join(f"{m['name']}({m['id']})" for m in matches)
        raise RuntimeError(f"multiple Drive folders named {folder_name}: {names}")
    return matches[0]["id"]


def iter_folder_files(service: Any, folder_id: str) -> Iterable[dict[str, Any]]:
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,size,mimeType,modifiedTime)"
    while True:
        resp = (
            service.files()
            .list(q=query, fields=fields, pageToken=page_token, orderBy="createdTime")
            .execute()
        )
        yield from resp.get("files", [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def parse_l89_drive_name(name: str) -> Optional[dict[str, str]]:
    base = name[:-4] if name.lower().endswith(".tif") else name
    parts = base.split("__")
    if len(parts) < 5 or parts[0] != "l89":
        return None
    tp = parts[1]
    plume_id = parts[2]
    spacecraft = parts[3]
    image_time = parts[4]
    if tp not in TIMEPOINTS:
        return None
    return {
        "timepoint": tp,
        "plume_id": plume_id,
        "spacecraft": spacecraft,
        "image_time": image_time,
    }


def load_manifest_targets(manifest: str, raw_root: str) -> dict[tuple[str, str], Path]:
    df = pd.read_csv(manifest, low_memory=False)
    out: dict[tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        if str(row.get("sensor", "")).upper() != "L89":
            continue
        if row_download_done(row.to_dict()):
            continue
        plume_id = str(row.get("plume_id", "")).strip()
        tp = str(row.get("timepoint", "")).strip()
        if not plume_id or tp not in TIMEPOINTS:
            continue
        target = row.get("target_raw_dir", "")
        if has_value(target):
            out[(plume_id, tp)] = Path(str(target))
        else:
            out[(plume_id, tp)] = Path(raw_root) / "L89" / tp / plume_id
    return out


def file_size_matches(item: dict[str, Any], local_path: Path) -> bool:
    if "size" not in item:
        return True
    try:
        return local_path.stat().st_size == int(item["size"])
    except OSError:
        return False


def download_file(service: Any, item: dict[str, Any], dest_path: Path, chunk_size: int) -> None:
    request = service.files().get_media(fileId=item["id"])
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    with io.FileIO(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    os.replace(tmp_path, dest_path)


def validate_tif(path: Path) -> tuple[bool, str]:
    if path.stat().st_size <= 0:
        return False, "empty_file"
    if rasterio is None:
        return True, "size_only"
    try:
        with rasterio.open(path) as ds:
            if ds.count <= 0 or ds.width <= 0 or ds.height <= 0:
                return False, "invalid_shape"
            return True, f"rasterio_ok:{ds.count}x{ds.height}x{ds.width}"
    except Exception as exc:
        return False, f"rasterio_error:{exc}"


def load_completed_records(out_csv: str) -> dict[tuple[str, str, str], Path]:
    path = Path(out_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    required = {"plume_id", "timepoint", "drive_file_name", "raw_path", "status"}
    if not required.issubset(df.columns):
        return {}

    completed: dict[tuple[str, str, str], Path] = {}
    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip()
        if status not in SUCCESS_STATUSES:
            continue
        plume_id = str(row.get("plume_id", "")).strip()
        tp = str(row.get("timepoint", "")).strip()
        drive_file_name = str(row.get("drive_file_name", "")).strip()
        raw_path = row.get("raw_path", "")
        if not plume_id or tp not in TIMEPOINTS or not drive_file_name or not has_value(raw_path):
            continue
        p = Path(str(raw_path))
        if not p.exists():
            continue
        ok, _ = validate_tif(p)
        if ok:
            completed[(plume_id, tp, drive_file_name)] = p
    return completed


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def process_once(
    service: Any,
    folder_id: str,
    targets: dict[tuple[str, str], Path],
    completed: dict[tuple[str, str, str], Path],
    args: argparse.Namespace,
) -> int:
    processed = 0
    records: list[dict[str, Any]] = []
    for item in iter_folder_files(service, folder_id):
        if args.max_files and processed >= args.max_files:
            break
        if item.get("mimeType") == FOLDER_MIME_TYPE:
            continue
        if should_skip_by_age(item, args.min_age_seconds):
            continue

        name = item.get("name", "")
        parsed = parse_l89_drive_name(name)
        if parsed is None:
            continue

        key = (parsed["plume_id"], parsed["timepoint"])
        target_dir = targets.get(key)
        if target_dir is None:
            if not args.allow_unplanned:
                log(f"skip unplanned Drive file: {name}")
                continue
            target_dir = Path(args.raw_root) / "L89" / parsed["timepoint"] / parsed["plume_id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / name

        record = {
            "plume_id": parsed["plume_id"],
            "timepoint": parsed["timepoint"],
            "status": "",
            "drive_file_id": item.get("id", ""),
            "drive_file_name": name,
            "drive_folder": args.drive_folder,
            "raw_path": str(dest_path),
            "image_time_file": parsed["image_time"],
            "spacecraft": parsed["spacecraft"],
            "size": item.get("size", ""),
            "message": "",
        }

        done_path = completed.get((parsed["plume_id"], parsed["timepoint"], name))
        if args.resume and done_path is not None and not args.overwrite:
            ok, msg = validate_tif(done_path)
            if ok:
                record["raw_path"] = str(done_path)
                record["status"] = "resume_skip_completed"
                record["message"] = msg
                if args.delete_after and not args.dry_run:
                    service.files().delete(fileId=item["id"]).execute()
                    record["status"] = "resume_skip_completed_deleted_drive"
                records.append(record)
                processed += 1
                continue

        if dest_path.exists() and not args.overwrite:
            if file_size_matches(item, dest_path):
                ok, msg = validate_tif(dest_path)
                record["status"] = "skip_existing_valid" if ok else "existing_invalid"
                record["message"] = msg
                if ok and args.delete_after and not args.dry_run:
                    service.files().delete(fileId=item["id"]).execute()
                    record["status"] = "skip_existing_valid_deleted_drive"
                processed += 1
            else:
                record["status"] = "existing_size_mismatch"
            records.append(record)
            continue

        if args.dry_run:
            record["status"] = "dry_run"
            records.append(record)
            processed += 1
            log(f"dry-run download {name} -> {dest_path}")
            continue

        try:
            log(f"downloading {name} -> {dest_path}")
            download_file(service, item, dest_path, args.chunk_size_mb * 1024 * 1024)
            if not file_size_matches(item, dest_path):
                record["status"] = "size_mismatch"
                records.append(record)
                continue
            ok, msg = validate_tif(dest_path)
            record["message"] = msg
            if not ok:
                record["status"] = "invalid_tif"
                records.append(record)
                continue
            if args.delete_after:
                service.files().delete(fileId=item["id"]).execute()
                record["status"] = "downloaded_deleted_drive"
            else:
                record["status"] = "downloaded"
            processed += 1
        except HttpError as exc:
            record["status"] = "drive_error"
            record["message"] = str(exc)
        except Exception as exc:
            record["status"] = "download_error"
            record["message"] = str(exc)
        records.append(record)

    append_rows(Path(args.out_csv), records)
    if not args.no_master_update:
        changed = update_master_from_records(
            args.manifest,
            "L89",
            records,
            record_to_master_update,
            source_log=args.out_csv,
        )
        log(f"updated master manifest rows: {changed}")
    return processed


def record_to_master_update(record: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(record.get("raw_path", "")))
    status = str(record.get("status", "")).strip()
    update = {
        "download_status": status,
        "selection_source": "l89_drive_pull",
        "status_message": record.get("message", ""),
    }
    if raw_path.exists() and raw_path.stat().st_size > 0:
        update["downloaded_path"] = str(raw_path)
        update["processed_path"] = str(raw_path)
    image_time = parse_file_image_time(record.get("image_time_file", ""))
    if image_time:
        update["image_time"] = image_time
    if has_value(record.get("drive_file_name")):
        update["product_name"] = record.get("drive_file_name", "")
    if has_value(record.get("spacecraft")) and image_time:
        update["overpass_key"] = f"{record.get('spacecraft')}|{record.get('image_time_file', '')[:8]}"
    return update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull completed L89 GEE Drive exports into the six-time raw directory."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", default="Upgrade_data_pipeline/csv/l89_drive_pull_manifest.csv")
    parser.add_argument("--raw-root", default="/mnt/engg-niulab/yuyao/sensors_raw_data")
    parser.add_argument("--drive-folder", default="L89_6time_raw_exports")
    parser.add_argument("--drive-folder-id", default="")
    parser.add_argument("--credentials", default="data_downloading/credentials.json")
    parser.add_argument("--token", default="data_downloading/token.json")
    parser.add_argument("--service-account", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--min-age-seconds", type=int, default=30)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-after", dest="delete_after", action="store_true", default=True)
    parser.add_argument("--no-delete", dest="delete_after", action="store_false")
    parser.add_argument("--allow-unplanned", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-master-update", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_manifest_columns(args.manifest)
    targets = load_manifest_targets(args.manifest, args.raw_root)
    log(f"loaded L89 manifest target dirs: {len(targets)}")
    completed = load_completed_records(args.out_csv) if args.resume else {}
    if args.resume:
        log(f"loaded completed pull records for resume: {len(completed)}")
    service = build_drive_service(args)
    folder_id = resolve_folder_id(service, args.drive_folder_id, args.drive_folder)

    while True:
        try:
            count = process_once(service, folder_id, targets, completed, args)
            if args.once:
                break
            if count == 0:
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            log(f"sync loop error: {exc}")
            if args.once:
                break
            time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
