#!/usr/bin/env python3
from __future__ import annotations

import csv
import fcntl
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


CANONICAL_FIELDS = [
    "download_status",
    "downloaded_path",
    "processed_path",
    "image_time",
    "product_id",
    "product_name",
    "overpass_key",
    "cloud_cover",
    "selection_source",
    "status_message",
    "target_time",
    "time_delta_hours",
    "qc_ok",
    "qc_reason",
    "qc_center_iy",
    "qc_center_ix",
    "qc_center_distance_km",
    "qc_patch_missing_ratio",
    "qc_patch_finite_count",
    "qc_patch_total",
    "qc_ch4_var",
    "qc_candidate_rank",
    "qc_candidates_checked",
    "candidate_attempts",
    "source_log",
    "updated_at_utc",
]

SUCCESS_STATUSES = {
    "available",
    "available_existing_corrected",
    "downloaded",
    "downloaded_deleted_drive",
    "linked_existing",
    "skip_existing",
    "skip_existing_raw",
    "skip_existing_512",
    "skip_existing_valid",
    "skip_existing_valid_deleted_drive",
    "skip_local_tif_exists",
    "resume_skip_completed",
    "resume_skip_completed_deleted_drive",
    "downloaded_crop_ok",
    "skip_existing_crop_ok",
    "master_completed_crop_ok",
    "resume_skip_completed_crop_ok",
}


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def existing_file(value: Any) -> Optional[Path]:
    if not has_value(value):
        return None
    path = Path(str(value).strip())
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def row_download_done(row: dict[str, Any]) -> bool:
    return existing_file(row.get("downloaded_path")) is not None


def row_processed_done(row: dict[str, Any]) -> bool:
    return existing_file(row.get("processed_path")) is not None


def canonical_fieldnames(fieldnames: Iterable[str] | None) -> list[str]:
    out = list(fieldnames or [])
    for field in CANONICAL_FIELDS:
        if field not in out:
            out.append(field)
    return out


@contextmanager
def manifest_lock(manifest_path: str | Path):
    path = Path(manifest_path)
    lock_name = f"methane_manifest_{path.name}.lock"
    lock_path = Path("/tmp") / lock_name
    with lock_path.open("w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def rewrite_manifest(
    manifest_path: str | Path,
    update_row: Callable[[dict[str, str]], bool],
) -> int:
    path = Path(manifest_path)
    changed = 0
    with manifest_lock(path):
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            with path.open(newline="") as src:
                reader = csv.DictReader(src)
                fields = canonical_fieldnames(reader.fieldnames)
                with tmp.open("w", newline="") as dst:
                    writer = csv.DictWriter(dst, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    for row in reader:
                        before = tuple(row.get(field, "") for field in fields)
                        touched = update_row(row)
                        after = tuple(row.get(field, "") for field in fields)
                        if touched or before != after:
                            changed += 1
                        writer.writerow({field: row.get(field, "") for field in fields})
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
    return changed


def ensure_manifest_columns(manifest_path: str | Path) -> int:
    def noop(_: dict[str, str]) -> bool:
        return False

    return rewrite_manifest(manifest_path, noop)


def load_master_completed_records(
    manifest_path: str | Path,
    sensor: str,
) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    path = Path(manifest_path)
    if not path.exists():
        return out
    sensor_norm = sensor.upper()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("sensor", "")).strip().upper() != sensor_norm:
                continue
            if not row_download_done(row):
                continue
            plume_id = str(row.get("plume_id", "")).strip()
            timepoint = str(row.get("timepoint", "")).strip()
            if plume_id and timepoint:
                out[(plume_id, timepoint)] = row
    return out


def select_rows_for_missing_download(
    rows: list[dict[str, Any]],
    requested_timepoints: set[str],
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    todo_plumes: set[str] = set()
    for row in rows:
        tp = str(row.get("timepoint", "")).strip()
        if tp not in requested_timepoints:
            continue
        if overwrite or not row_download_done(row):
            selected.append(row)
            plume_id = str(row.get("plume_id", "")).strip()
            if plume_id:
                todo_plumes.add(plume_id)

    existing_keys = {
        (str(row.get("plume_id", "")).strip(), str(row.get("timepoint", "")).strip())
        for row in selected
    }
    for row in rows:
        plume_id = str(row.get("plume_id", "")).strip()
        tp = str(row.get("timepoint", "")).strip()
        if plume_id in todo_plumes and tp == "t0" and (plume_id, "t0") not in existing_keys:
            selected.append(row)
            existing_keys.add((plume_id, "t0"))
    return selected


def update_master_from_records(
    manifest_path: str | Path,
    sensor: str,
    records: Iterable[dict[str, Any]],
    mapper: Callable[[dict[str, Any]], dict[str, Any]],
    source_log: str,
) -> int:
    sensor_norm = sensor.upper()
    updates: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        plume_id = str(record.get("plume_id", "")).strip()
        tp = str(record.get("timepoint", "")).strip()
        if not plume_id or not tp:
            continue
        update = mapper(record)
        if not update:
            continue
        update["source_log"] = source_log
        update["updated_at_utc"] = utc_now_iso()
        updates[(plume_id, tp)] = update

    if not updates:
        return 0

    def apply(row: dict[str, str]) -> bool:
        if str(row.get("sensor", "")).strip().upper() != sensor_norm:
            return False
        key = (str(row.get("plume_id", "")).strip(), str(row.get("timepoint", "")).strip())
        update = updates.get(key)
        if update is None:
            return False
        has_good_path = row_download_done(row)
        status = str(update.get("download_status", "")).strip()
        update_has_path = existing_file(update.get("downloaded_path")) is not None

        if has_good_path and not update_has_path:
            row["download_status"] = status or row.get("download_status", "")
            row["status_message"] = str(update.get("status_message", row.get("status_message", "")))
            row["source_log"] = source_log
            row["updated_at_utc"] = str(update["updated_at_utc"])
            return True

        for field in CANONICAL_FIELDS:
            if field in update:
                row[field] = str(update.get(field, ""))
        return True

    return rewrite_manifest(manifest_path, apply)
