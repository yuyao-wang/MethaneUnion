#!/usr/bin/env python3
"""Recover GEE exports that reached Drive or remain active without manifest rows."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import pandas as pd

import pull_s2_6time_gee_drive_exports as puller
import s2_6time_gee_export as exporter


ACTIVE_TASK_STATES = {"READY", "RUNNING"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile unrecorded GEE tasks and Drive files into an S2 export manifest."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timepoints", default="prev1,prev2,prev3")
    parser.add_argument("--raw-root", default=puller.DEFAULT_RAW_ROOT)
    parser.add_argument("--drive-folder", default=puller.DEFAULT_DRIVE_FOLDER)
    parser.add_argument("--drive-folder-id", default="")
    parser.add_argument(
        "--credentials",
        default="/home/yuyao/methane_train/data_downloading/credentials.json",
    )
    parser.add_argument(
        "--token",
        default="/home/yuyao/methane_train/data_downloading/token.json",
    )
    parser.add_argument("--service-account", action="store_true")
    parser.add_argument("--ee-project", default="")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def expected_prefixes(
    input_csv: Path,
    timepoints: set[str],
) -> dict[str, tuple[str, str]]:
    frame = pd.read_csv(input_csv, low_memory=False)
    prefixes: dict[str, tuple[str, str]] = {}
    for plume_id in frame["plume_id"].astype(str):
        for timepoint in timepoints:
            suffix = puller.file_stem(puller.CANONICAL_FILENAME[timepoint])
            prefixes[f"{plume_id}_{suffix}"] = (plume_id, timepoint)
    return prefixes


def latest_manifest_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    if not path.exists():
        return latest
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            plume_id = str(row.get("plume_id", "")).strip()
            timepoint = str(row.get("timepoint", "")).strip()
            if plume_id and timepoint:
                latest[(plume_id, timepoint)] = row
    return latest


def manifest_has_source(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    status = str(row.get("status", "")).strip()
    prefix = str(row.get("file_prefix", "")).strip()
    if status in {"drive_exists", "local_exists"}:
        return True
    return status in {"submitted", "already_submitted"} and bool(prefix)


def drive_recovery_rows(
    service: Any,
    folder_id: str,
    expected: dict[str, tuple[str, str]],
    latest: dict[tuple[str, str], dict[str, str]],
    drive_folder: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[str]]:
    recovered: dict[tuple[str, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for item in puller.iter_folder_files(service, folder_id):
        counts["files_seen"] += 1
        name = str(item.get("name", ""))
        prefix = puller.file_stem(name)
        key = expected.get(prefix)
        if key is None:
            counts["outside_requested_input"] += 1
            continue
        counts["requested_files"] += 1
        if manifest_has_source(latest.get(key)):
            counts["already_recorded"] += 1
            continue
        if key in recovered:
            counts["duplicate_requested_files"] += 1
            continue
        plume_id, timepoint = key
        recovered[key] = {
            "plume_id": plume_id,
            "timepoint": timepoint,
            "status": "drive_exists",
            "task_state": "COMPLETED",
            "drive_folder": drive_folder,
            "file_prefix": prefix,
            "message": (
                f"reconciled existing Drive file id={item.get('id', '')} "
                f"name={name} size={item.get('size', '')}"
            ),
        }
    return recovered, counts


def active_task_recovery_rows(
    expected: dict[str, tuple[str, str]],
    latest: dict[tuple[str, str], dict[str, str]],
    drive_rows: dict[tuple[str, str], dict[str, Any]],
    drive_folder: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[str]]:
    recovered: dict[tuple[str, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for task in ee.batch.Task.list():
        state = str(getattr(task, "state", ""))
        if state not in ACTIVE_TASK_STATES:
            continue
        counts["active_tasks_seen"] += 1
        description = str(getattr(task, "config", {}).get("description", ""))
        key = expected.get(description)
        if key is None:
            counts["outside_requested_input"] += 1
            continue
        counts["requested_active_tasks"] += 1
        if key in drive_rows or manifest_has_source(latest.get(key)):
            counts["already_covered"] += 1
            continue
        if key in recovered:
            counts["duplicate_requested_tasks"] += 1
            continue
        plume_id, timepoint = key
        recovered[key] = {
            "plume_id": plume_id,
            "timepoint": timepoint,
            "status": "already_submitted",
            "task_id": str(getattr(task, "id", "")),
            "task_state": state,
            "drive_folder": drive_folder,
            "file_prefix": description,
            "message": "reconciled active GEE task missing from manifest",
        }
    return recovered, counts


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("rb+") as handle:
        handle.seek(0, 2)
        if handle.tell() > 0:
            handle.seek(-1, 2)
            if handle.read(1) not in {b"\n", b"\r"}:
                handle.seek(0, 2)
                handle.write(b"\n")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=exporter.OUT_FIELDS)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in exporter.OUT_FIELDS})


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    manifest = Path(args.manifest)
    timepoints = {
        value.strip()
        for value in str(args.timepoints).split(",")
        if value.strip()
    }
    unknown = timepoints - set(puller.TIMEPOINTS)
    if unknown:
        raise ValueError(f"unknown timepoints: {sorted(unknown)}")

    expected = expected_prefixes(input_csv, timepoints)
    latest = latest_manifest_rows(manifest)

    drive_service = puller.build_drive_service(args)
    folder_id = puller.resolve_folder_id(
        drive_service,
        args.drive_folder_id,
        args.drive_folder,
    )
    drive_rows, drive_counts = drive_recovery_rows(
        drive_service,
        folder_id,
        expected,
        latest,
        args.drive_folder,
    )

    initialize_kwargs = {}
    if args.ee_project:
        initialize_kwargs["project"] = args.ee_project
    ee.Initialize(**initialize_kwargs)
    task_rows, task_counts = active_task_recovery_rows(
        expected,
        latest,
        drive_rows,
        args.drive_folder,
    )

    rows = list(drive_rows.values()) + list(task_rows.values())
    print(
        f"expected={len(expected)} manifest_latest={len(latest)} "
        f"recover_drive={len(drive_rows)} recover_active={len(task_rows)} "
        f"total_append={len(rows)}"
    )
    print(f"drive_counts={dict(drive_counts)}")
    print(f"task_counts={dict(task_counts)}")
    if not args.apply:
        print("dry-run only; pass --apply after stopping the exporter")
        return 0
    if not rows:
        print("nothing to append")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = manifest.with_name(f"{manifest.name}.pre_reconcile_{timestamp}")
    shutil.copy2(manifest, backup)
    append_rows(manifest, rows)
    print(f"backup={backup}")
    print(f"appended={len(rows)} manifest={manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
