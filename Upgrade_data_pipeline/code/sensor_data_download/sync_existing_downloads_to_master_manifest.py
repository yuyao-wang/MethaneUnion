#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from manifest_state import existing_file, update_master_from_records


S2_512_NAME = {
    "s2.tif": "s2_0_std_512.tif",
    "s2_-7.tif": "s2_-7_std_512.tif",
    "s2_prev1.tif": "s2_prev1_std_512.tif",
    "s2_prev2.tif": "s2_prev2_std_512.tif",
    "s2_prev3.tif": "s2_prev3_std_512.tif",
    "s2_-90.tif": "s2_-90_std_512.tif",
    "s2_-360.tif": "s2_-360_std_512.tif",
}


def s5p_downloaded_path(product_name: str) -> str:
    product = Path(str(product_name).strip()).name
    if not product:
        return ""
    raw_dir = Path("/mnt/engg-niulab/yuyao/sensors_raw_data/S5P/raw_data_dir_s5p")
    direct = raw_dir / product
    if direct.exists() and direct.stat().st_size > 0:
        return str(direct)
    for candidate in raw_dir.glob(f"**/{product}"):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate)
    return ""


def s2_processed_path(raw_path: str) -> str:
    path = Path(raw_path)
    out_name = S2_512_NAME.get(path.name)
    if not out_name:
        return ""
    candidate = Path("/mnt/engg-niulab/yuyao/preprocessed_512/S2") / path.parent.name / out_name
    return str(candidate) if existing_file(candidate) is not None else ""


def mapper_from_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    sensor = str(record.get("sensor", "")).upper()
    raw_path = (
        s5p_downloaded_path(str(record.get("product_name", "")))
        if sensor == "S5P"
        else str(record.get("raw_path", "")).strip()
    )
    downloaded = existing_file(raw_path)
    if downloaded is None:
        return {
            "download_status": "missing_path_from_snapshot",
            "status_message": raw_path,
        }

    processed_path = ""
    if sensor == "S2":
        processed_path = s2_processed_path(str(downloaded))
    elif sensor in {"L89", "EMIT"}:
        processed_path = str(downloaded)

    return {
        "download_status": "available_existing_corrected",
        "downloaded_path": str(downloaded),
        "processed_path": processed_path,
        "image_time": record.get("image_time", ""),
        "product_id": record.get("product_id", ""),
        "product_name": record.get("product_name", ""),
        "overpass_key": record.get("overpass_key", ""),
        "cloud_cover": record.get("cloud_cover", ""),
        "selection_source": record.get("correction_note", "") or record.get("source_original_timepoint", ""),
        "status_message": record.get("source_original_status", ""),
    }


def load_snapshot_records(path: str) -> dict[str, list[dict[str, Any]]]:
    by_sensor: dict[str, list[dict[str, Any]]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("corrected_status") != "available":
                continue
            if str(row.get("path_exists", "")).lower() not in {"yes", "true", "1"}:
                continue
            sensor = str(row.get("sensor", "")).upper()
            if not sensor:
                continue
            by_sensor.setdefault(sensor, []).append(row)
    return by_sensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill the canonical six-time manifest with already verified existing downloads."
    )
    parser.add_argument(
        "--manifest",
        default="Upgrade_data_pipeline/csv/multisensor_6time_download_manifest.csv",
    )
    parser.add_argument(
        "--corrected-snapshot",
        default="Upgrade_data_pipeline/csv/multisensor_6time_download_manifest_corrected_prev_snapshot.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    by_sensor = load_snapshot_records(args.corrected_snapshot)
    total_changed = 0
    for sensor in sorted(by_sensor):
        changed = update_master_from_records(
            args.manifest,
            sensor,
            by_sensor[sensor],
            mapper_from_snapshot,
            source_log=args.corrected_snapshot,
        )
        total_changed += changed
        print(f"{sensor}: input_available={len(by_sensor[sensor])} master_rows_updated={changed}", flush=True)
    print(f"total_master_rows_updated={total_changed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
