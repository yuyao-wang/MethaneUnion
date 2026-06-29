#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import ee
import pandas as pd

from manifest_state import ensure_manifest_columns, row_download_done, select_rows_for_missing_download

try:
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
except ImportError:
    build = None


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
PREV_TIMEPOINTS = {"prev1": 1, "prev2": 2, "prev3": 3}
TIMEPOINT_ORDER = {tp: i for i, tp in enumerate(TIMEPOINTS)}
T0_SUCCESS_STATUSES = {
    "submitted",
    "selected",
    "skip_local_tif_exists",
    "skip_drive_file_exists",
    "skip_gee_task_pending",
}
SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

SR_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT", "SR_QA_AEROSOL"]
EXPORT_BANDS = SR_BANDS + QA_BANDS


@dataclass(frozen=True)
class SelectedImage:
    asset_id: str
    acq_dt: datetime
    cloud: Optional[float]
    spacecraft: str
    wrs_path: str
    wrs_row: str


def log(message: str) -> None:
    print(message, flush=True)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def parse_dt(value: Any) -> Optional[datetime]:
    if not has_value(value):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    if text.endswith("+00"):
        text += ":00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def acq_file_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_task_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)[:95]


def local_tif_exists(target_raw_dir: Any) -> bool:
    if not has_value(target_raw_dir):
        return False
    p = Path(str(target_raw_dir))
    return p.is_dir() and any(p.glob("*.tif"))


def init_gee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def build_region(lon: float, lat: float, chip_size_px: int, scale_m: float) -> ee.Geometry:
    point = ee.Geometry.Point([lon, lat])
    half_size_m = chip_size_px * scale_m / 2.0
    return point.buffer(half_size_m).bounds()


def merge_l89_sr_t1_collection(
    region: ee.Geometry,
    start_dt: datetime,
    end_dt: datetime,
    cloud_max: float,
) -> ee.ImageCollection:
    start_ee = ee.Date(start_dt.isoformat())
    end_ee = ee.Date(end_dt.isoformat())
    c8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(start_ee, end_ee)
        .filter(ee.Filter.lte("CLOUD_COVER", cloud_max))
    )
    c9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(region)
        .filterDate(start_ee, end_ee)
        .filter(ee.Filter.lte("CLOUD_COVER", cloud_max))
    )
    return c8.merge(c9)


def feature_to_selected(feature: dict[str, Any]) -> Optional[SelectedImage]:
    props = feature.get("properties") or {}
    asset_id = feature.get("id") or props.get("system:id")
    t_ms = props.get("system:time_start")
    if not asset_id or t_ms is None:
        return None
    try:
        acq_dt = datetime.fromtimestamp(float(t_ms) / 1000.0, tz=timezone.utc)
    except Exception:
        return None
    cloud = props.get("CLOUD_COVER")
    try:
        cloud_value = float(cloud) if cloud is not None else None
    except Exception:
        cloud_value = None
    return SelectedImage(
        asset_id=str(asset_id),
        acq_dt=acq_dt,
        cloud=cloud_value,
        spacecraft=str(props.get("SPACECRAFT_ID") or "LANDSAT"),
        wrs_path=str(props.get("WRS_PATH") or ""),
        wrs_row=str(props.get("WRS_ROW") or ""),
    )


def fetch_candidates(
    region: ee.Geometry,
    start_dt: datetime,
    end_dt: datetime,
    cloud_max: float,
    limit: int,
) -> list[SelectedImage]:
    col = merge_l89_sr_t1_collection(region, start_dt, end_dt, cloud_max)
    features = col.limit(limit).getInfo().get("features", [])
    out: list[SelectedImage] = []
    for feature in features:
        selected = feature_to_selected(feature)
        if selected is not None:
            out.append(selected)
    return out


def select_t0(
    region: ee.Geometry,
    event_dt: datetime,
    cloud_max: float,
    limit: int,
) -> Optional[SelectedImage]:
    candidates = fetch_candidates(
        region,
        event_dt,
        event_dt + timedelta(hours=24),
        cloud_max,
        limit,
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (
            abs((c.acq_dt - event_dt).total_seconds()),
            c.cloud if c.cloud is not None else 999.0,
        ),
    )


def select_offset(
    region: ee.Geometry,
    anchor_dt: datetime,
    offset_days: int,
    search_window_days: int,
    cloud_max: float,
    limit: int,
) -> Optional[SelectedImage]:
    target_dt = anchor_dt - timedelta(days=offset_days)
    candidates = fetch_candidates(
        region,
        target_dt - timedelta(days=search_window_days),
        target_dt + timedelta(days=search_window_days),
        cloud_max,
        limit,
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (
            abs((c.acq_dt - target_dt).total_seconds()),
            c.cloud if c.cloud is not None else 999.0,
        ),
    )


def l89_overpass_key(image: SelectedImage) -> str:
    # Adjacent WRS rows from the same path/date are one local overpass for this
    # plume-level dataset; row is intentionally excluded from the key.
    return f"{image.spacecraft}|P{image.wrs_path}|{image.acq_dt.strftime('%Y%m%d')}"


def select_prev_n(
    region: ee.Geometry,
    t0_image: SelectedImage,
    n: int,
    search_back_days: int,
    cloud_max: float,
    limit: int,
) -> Optional[SelectedImage]:
    anchor_dt = t0_image.acq_dt
    t0_key = l89_overpass_key(t0_image)
    candidates = fetch_candidates(
        region,
        anchor_dt - timedelta(days=search_back_days),
        anchor_dt,
        cloud_max,
        limit,
    )
    before = [
        c
        for c in candidates
        if c.acq_dt < anchor_dt
        and c.asset_id != t0_image.asset_id
        and l89_overpass_key(c) != t0_key
    ]
    by_pass: dict[str, SelectedImage] = {}
    for cand in before:
        pass_key = l89_overpass_key(cand)
        old = by_pass.get(pass_key)
        if old is None:
            by_pass[pass_key] = cand
            continue
        old_key = (
            old.cloud if old.cloud is not None else 999.0,
            abs((old.acq_dt - anchor_dt).total_seconds()),
        )
        new_key = (
            cand.cloud if cand.cloud is not None else 999.0,
            abs((cand.acq_dt - anchor_dt).total_seconds()),
        )
        if new_key < old_key:
            by_pass[pass_key] = cand

    ordered = sorted(by_pass.values(), key=lambda c: c.acq_dt, reverse=True)
    if len(ordered) < n:
        return None
    return ordered[n - 1]


def selected_for_row(row: pd.Series, tp: str, args: argparse.Namespace, t0_image: Optional[SelectedImage] = None) -> tuple[Optional[SelectedImage], str]:
    event_dt = parse_dt(row.get("event_time"))
    t0_dt = parse_dt(row.get("t0_available_time")) or event_dt
    lat = row.get("plume_latitude")
    lon = row.get("plume_longitude")
    if event_dt is None:
        return None, "invalid_event_time"
    if t0_dt is None:
        return None, "invalid_t0_time"
    if pd.isna(lat) or pd.isna(lon):
        return None, "missing_latlon"

    region = build_region(float(lon), float(lat), args.chip_size_px, args.scale_m)
    if tp == "t0":
        image = select_t0(region, event_dt, args.cloud_max, args.ee_candidate_limit)
    elif tp in PREV_TIMEPOINTS:
        if t0_image is None:
            return None, "missing_actual_t0_anchor"
        image = select_prev_n(
            region,
            t0_image,
            PREV_TIMEPOINTS[tp],
            args.prev_search_back_days,
            args.cloud_max,
            args.ee_candidate_limit,
        )
    elif tp == "seasonal":
        image = select_offset(
            region,
            t0_dt,
            90,
            args.offset_search_window_days,
            args.cloud_max,
            args.ee_candidate_limit,
        )
    elif tp == "year":
        offset = int(row.get("year_offset_days") or 360)
        image = select_offset(
            region,
            t0_dt,
            offset,
            args.offset_search_window_days,
            args.cloud_max,
            args.ee_candidate_limit,
        )
    else:
        return None, f"unknown_timepoint:{tp}"

    if image is None:
        return None, "no_image"
    return image, "selected"


def file_prefix_for(row: pd.Series, tp: str, selected: SelectedImage) -> str:
    plume_id = str(row["plume_id"]).strip()
    return (
        f"l89__{tp}__{plume_id}__{selected.spacecraft}"
        f"__{acq_file_time(selected.acq_dt)}"
    )


def task_description_for(row: pd.Series, tp: str) -> str:
    return safe_task_text(f"l89__{tp}__{row['plume_id']}")


def export_to_drive(
    row: pd.Series,
    tp: str,
    selected: SelectedImage,
    args: argparse.Namespace,
) -> tuple[str, str]:
    lat = float(row["plume_latitude"])
    lon = float(row["plume_longitude"])
    region = build_region(lon, lat, args.chip_size_px, args.scale_m)
    img = ee.Image(selected.asset_id).select(EXPORT_BANDS).toUint16().clip(region)
    desc = task_description_for(row, tp)
    file_prefix = file_prefix_for(row, tp, selected)
    export_kwargs = dict(
        image=img,
        description=desc,
        folder=args.drive_folder,
        fileNamePrefix=file_prefix,
        region=region,
        scale=args.scale_m,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    if args.crs:
        export_kwargs["crs"] = args.crs
    task = ee.batch.Export.image.toDrive(**export_kwargs)
    task.start()
    return desc, file_prefix


def load_pending_task_descriptions() -> tuple[set[str], int]:
    descriptions: set[str] = set()
    pending = 0
    try:
        tasks = ee.batch.Task.list()
    except Exception as exc:
        log(f"failed to list GEE tasks: {exc}")
        return descriptions, pending
    for task in tasks:
        try:
            status = task.status()
        except Exception:
            continue
        state = status.get("state")
        desc = status.get("description", "") or ""
        if state in {"READY", "RUNNING"}:
            pending += 1
            descriptions.add(desc)
    return descriptions, pending


def wait_for_capacity(max_pending: int, sleep_seconds: int) -> int:
    while True:
        _, pending = load_pending_task_descriptions()
        if pending < max_pending:
            return pending
        log(f"pending task limit reached ({pending}/{max_pending}); sleep {sleep_seconds}s")
        time.sleep(sleep_seconds)


def build_drive_service(args: argparse.Namespace):
    if build is None:
        return None
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
                if not os.path.exists(args.credentials):
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(args.credentials, SCOPES)
                creds = flow.run_console()
            with open(args.token, "w") as fh:
                fh.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_folder_id(service: Any, folder_name: str, folder_id: str = "") -> Optional[str]:
    if folder_id:
        return folder_id
    safe_name = folder_name.replace("'", "\\'")
    query = f"mimeType='{FOLDER_MIME_TYPE}' and name='{safe_name}' and trashed=false"
    resp = service.files().list(q=query, fields="files(id,name)").execute()
    matches = resp.get("files", [])
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(f"{m['name']}({m['id']})" for m in matches)
        raise RuntimeError(f"multiple Drive folders named {folder_name}: {names}")
    return matches[0]["id"]


def iter_drive_files(service: Any, folder_id: str) -> Iterable[dict[str, Any]]:
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,mimeType)"
    while True:
        resp = service.files().list(q=query, fields=fields, pageToken=page_token).execute()
        yield from resp.get("files", [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def parse_l89_drive_name(name: str) -> Optional[tuple[str, str]]:
    base = name[:-4] if name.lower().endswith(".tif") else name
    parts = base.split("__")
    if len(parts) < 5 or parts[0] != "l89":
        return None
    return parts[2], parts[1]


def load_drive_export_keys(args: argparse.Namespace) -> set[tuple[str, str]]:
    if args.skip_drive_scan:
        return set()
    service = build_drive_service(args)
    if service is None:
        log("Drive API unavailable or credentials missing; skip Drive scan")
        return set()
    folder_id = resolve_folder_id(service, args.drive_folder, args.drive_folder_id)
    if folder_id is None:
        log(f"Drive folder not found yet: {args.drive_folder}; skip Drive scan")
        return set()
    keys: set[tuple[str, str]] = set()
    for item in iter_drive_files(service, folder_id):
        if item.get("mimeType") == FOLDER_MIME_TYPE:
            continue
        parsed = parse_l89_drive_name(item.get("name", ""))
        if parsed is not None:
            keys.add(parsed)
    log(f"Drive scan found {len(keys)} L89 exported files")
    return keys


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def row_record(row: pd.Series, tp: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "plume_id": str(row["plume_id"]),
        "timepoint": tp,
        "status": "",
        "event_time": row.get("event_time", ""),
        "t0_available_time": row.get("t0_available_time", ""),
        "image_time": "",
        "cloud": "",
        "spacecraft": "",
        "wrs_path": "",
        "wrs_row": "",
        "asset_id": "",
        "drive_folder": args.drive_folder,
        "drive_file_prefix": "",
        "gee_task_description": task_description_for(row, tp),
        "target_raw_dir": row.get("target_raw_dir", ""),
        "message": "",
    }


def load_work_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.manifest, low_memory=False)
    tps = {tp.strip() for tp in args.timepoints.split(",") if tp.strip()}
    bad = sorted(tps - set(TIMEPOINTS))
    if bad:
        raise ValueError(f"unknown timepoints: {bad}")
    mask = (
        (df["sensor"].astype(str).str.upper() == "L89")
        & (df["action"].astype(str) == "download")
    )
    candidates = df[mask].copy()
    selected = select_rows_for_missing_download(candidates.to_dict("records"), tps)
    out = pd.DataFrame(selected, columns=candidates.columns)
    if args.limit:
        out = out.head(args.limit)
    out["_timepoint_order"] = out["timepoint"].astype(str).map(TIMEPOINT_ORDER).fillna(999)
    out = out.sort_values(["plume_id", "_timepoint_order"], kind="stable").drop(columns=["_timepoint_order"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit missing L89 six-time GEE Drive exports from the upgrade manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", default="Upgrade_data_pipeline/csv/l89_gee_drive_submit_manifest.csv")
    parser.add_argument("--drive-folder", default="L89_6time_raw_exports")
    parser.add_argument("--drive-folder-id", default="")
    parser.add_argument("--credentials", default="data_downloading/credentials.json")
    parser.add_argument("--token", default="data_downloading/token.json")
    parser.add_argument("--service-account", action="store_true")
    parser.add_argument("--skip-drive-scan", action="store_true")
    parser.add_argument("--timepoints", default="t0,prev1,prev2,prev3,seasonal,year")
    parser.add_argument("--gee-project", default="ringed-tractor-475719-d6")
    parser.add_argument("--cloud-max", type=float, default=20.0)
    parser.add_argument("--chip-size-px", type=int, default=512)
    parser.add_argument("--scale-m", type=float, default=30.0)
    parser.add_argument("--prev-search-back-days", type=int, default=120)
    parser.add_argument("--offset-search-window-days", type=int, default=50)
    parser.add_argument("--ee-candidate-limit", type=int, default=500)
    parser.add_argument("--max-pending-tasks", type=int, default=180)
    parser.add_argument("--pending-task-sleep-seconds", type=int, default=60)
    parser.add_argument("--submit-sleep-seconds", type=float, default=0.2)
    parser.add_argument("--crs", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_manifest_columns(args.manifest)
    rows = load_work_rows(args)
    log(f"loaded L89 download rows: {len(rows)}")
    if rows.empty:
        return 0

    init_gee(args.gee_project)
    pending_descs, pending = load_pending_task_descriptions()
    drive_keys = load_drive_export_keys(args)
    log(f"pending GEE tasks: {pending}")
    log("mode: submit" if args.submit else "mode: dry-run, add --submit to start tasks")

    out_rows: list[dict[str, Any]] = []
    submitted = 0
    t0_success: dict[str, bool] = {}
    t0_failure: dict[str, str] = {}
    t0_selected: dict[str, SelectedImage] = {}
    for n, (_, row) in enumerate(rows.iterrows(), start=1):
        plume_id = str(row["plume_id"])
        tp = str(row["timepoint"])
        record = row_record(row, tp, args)
        if row_download_done(row.to_dict()):
            record["status"] = "master_completed"
            record["image_time"] = row.get("image_time", "")
            record["cloud"] = row.get("cloud_cover", "")
            record["asset_id"] = row.get("product_id", "") or row.get("product_name", "")
            if tp == "t0":
                t0_success[plume_id] = True
                selected, reason = selected_for_row(row, tp, args)
                if selected is not None:
                    t0_selected[plume_id] = selected
                else:
                    record["message"] = reason
            out_rows.append(record)
            continue
        if tp != "t0" and not t0_success.get(plume_id, False):
            record["status"] = "skip_t0_failed"
            record["message"] = t0_failure.get(plume_id, "t0 has not succeeded for this plume")
            out_rows.append(record)
            log(f"[{n}/{len(rows)}] skip_t0_failed {plume_id} {tp}: {record['message']}")
            if len(out_rows) >= 100:
                write_rows(Path(args.out_csv), out_rows)
                out_rows.clear()
            continue
        if local_tif_exists(row.get("target_raw_dir")):
            record["status"] = "skip_local_tif_exists"
            if tp == "t0":
                t0_success[plume_id] = True
                selected, reason = selected_for_row(row, tp, args)
                if selected is not None:
                    t0_selected[plume_id] = selected
                    record["image_time"] = iso_z(selected.acq_dt)
                    record["cloud"] = selected.cloud if selected.cloud is not None else ""
                    record["spacecraft"] = selected.spacecraft
                    record["wrs_path"] = selected.wrs_path
                    record["wrs_row"] = selected.wrs_row
                    record["asset_id"] = selected.asset_id
                else:
                    record["message"] = reason
            out_rows.append(record)
            continue
        if (plume_id, tp) in drive_keys:
            record["status"] = "skip_drive_file_exists"
            if tp == "t0":
                t0_success[plume_id] = True
                selected, reason = selected_for_row(row, tp, args)
                if selected is not None:
                    t0_selected[plume_id] = selected
                    record["image_time"] = iso_z(selected.acq_dt)
                    record["cloud"] = selected.cloud if selected.cloud is not None else ""
                    record["spacecraft"] = selected.spacecraft
                    record["wrs_path"] = selected.wrs_path
                    record["wrs_row"] = selected.wrs_row
                    record["asset_id"] = selected.asset_id
                else:
                    record["message"] = reason
            out_rows.append(record)
            continue
        if record["gee_task_description"] in pending_descs:
            record["status"] = "skip_gee_task_pending"
            if tp == "t0":
                t0_success[plume_id] = True
                selected, reason = selected_for_row(row, tp, args)
                if selected is not None:
                    t0_selected[plume_id] = selected
                    record["image_time"] = iso_z(selected.acq_dt)
                    record["cloud"] = selected.cloud if selected.cloud is not None else ""
                    record["spacecraft"] = selected.spacecraft
                    record["wrs_path"] = selected.wrs_path
                    record["wrs_row"] = selected.wrs_row
                    record["asset_id"] = selected.asset_id
                else:
                    record["message"] = reason
            out_rows.append(record)
            continue

        selected, reason = selected_for_row(row, tp, args, t0_selected.get(plume_id))
        if selected is None:
            record["status"] = reason
            if tp == "t0":
                t0_success[plume_id] = False
                t0_failure[plume_id] = reason
            out_rows.append(record)
            log(f"[{n}/{len(rows)}] miss {plume_id} {tp}: {reason}")
            continue

        record.update(
            {
                "status": "selected" if not args.submit else "submitted",
                "image_time": iso_z(selected.acq_dt),
                "cloud": selected.cloud if selected.cloud is not None else "",
                "spacecraft": selected.spacecraft,
                "wrs_path": selected.wrs_path,
                "wrs_row": selected.wrs_row,
                "asset_id": selected.asset_id,
                "drive_file_prefix": file_prefix_for(row, tp, selected),
            }
        )
        if tp == "t0" and record["status"] in T0_SUCCESS_STATUSES:
            t0_success[plume_id] = True
            t0_selected[plume_id] = selected

        if args.submit:
            if pending >= args.max_pending_tasks:
                pending = wait_for_capacity(args.max_pending_tasks, args.pending_task_sleep_seconds)
            desc, prefix = export_to_drive(row, tp, selected, args)
            record["gee_task_description"] = desc
            record["drive_file_prefix"] = prefix
            pending += 1
            submitted += 1
            time.sleep(args.submit_sleep_seconds)
            if tp == "t0" and record["status"] in T0_SUCCESS_STATUSES:
                t0_success[plume_id] = True
                t0_selected[plume_id] = selected

        out_rows.append(record)
        log(
            f"[{n}/{len(rows)}] {record['status']} {plume_id} {tp} "
            f"{record['image_time']} cloud={record['cloud']}"
        )
        if len(out_rows) >= 100:
            write_rows(Path(args.out_csv), out_rows)
            out_rows.clear()

    write_rows(Path(args.out_csv), out_rows)
    log(f"done; submitted={submitted}; log={args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
