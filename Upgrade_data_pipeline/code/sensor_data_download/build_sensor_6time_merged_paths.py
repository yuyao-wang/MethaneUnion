#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
BASE_COLUMNS = [
    "plume_id",
    "event_time",
    "plume_latitude",
    "plume_longitude",
    "plume_bounds",
    "sensor",
    "has_t0_any",
    "has_all6_raw",
    "has_all6_512",
    "has_all6_npz",
    "has_all6_any",
]
TP_COLUMNS = [
    "image_time",
    "raw_path",
    "512_path",
    "npz_path",
    "product_id",
    "product_name",
    "overpass_key",
    "cloud_cover",
    "selection_source",
    "path_source",
]
FIELDNAMES = BASE_COLUMNS + [f"{tp}_{col}" for tp in TIMEPOINTS for col in TP_COLUMNS]

FAIL_STATUSES = {
    "download_failed",
    "download_error",
    "no_product",
    "no_granule",
    "error",
    "failed",
    "skip_t0_failed",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def has_value(value: Any) -> bool:
    return bool(clean(value))


class ExistsCache:
    def __init__(self) -> None:
        self.cache: dict[str, bool] = {}

    def exists(self, value: Any) -> bool:
        text = clean(value)
        if not text:
            return False
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        try:
            path = Path(text)
            ok = path.exists() and path.stat().st_size > 0
        except OSError:
            ok = False
        self.cache[text] = ok
        return ok


def parse_time(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    if text.endswith("+00"):
        text = text + ":00"
    if text.endswith("+0000"):
        text = text[:-5] + "+00:00"
    if text.endswith("Z"):
        return text
    if re.search(r"[+-]\d\d:\d\d$", text):
        return text.replace("+00:00", "Z")
    if re.match(r"^20\d{6}T\d{6}Z?$", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}T{text[9:11]}:{text[11:13]}:{text[13:15]}Z"
    return text


def parse_l89_file_time(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    match = re.search(r"(20\d{6})T(\d{6})Z?", text)
    if not match:
        return parse_time(text)
    return parse_time(f"{match.group(1)}T{match.group(2)}Z")


def parse_emit_time(*values: Any) -> str:
    for value in values:
        text = clean(value)
        if not text:
            continue
        match = re.search(r"(20\d{6})T(\d{6})", text)
        if match:
            return parse_time(f"{match.group(1)}T{match.group(2)}Z")
        match = re.search(r"emi(?P<date>\d{8})t(?P<time>\d{6})", text, re.I)
        if match:
            return parse_time(f"{match.group('date')}T{match.group('time')}Z")
    return ""


def empty_tp_record() -> dict[str, Any]:
    out = {col: "" for col in TP_COLUMNS}
    out["_path_sources"] = set()
    return out


def get_record(rows: dict[str, dict[str, Any]], sensor: str, plume_id: str) -> dict[str, Any]:
    row = rows.setdefault(
        plume_id,
        {
            "plume_id": plume_id,
            "event_time": "",
            "plume_latitude": "",
            "plume_longitude": "",
            "plume_bounds": "",
            "sensor": sensor,
            "timepoints": {tp: empty_tp_record() for tp in TIMEPOINTS},
        },
    )
    return row


def fill_if_blank(row: dict[str, Any], key: str, value: Any) -> None:
    text = clean(value)
    if text and not clean(row.get(key)):
        row[key] = text


def fill_tp_if_blank(tp_rec: dict[str, Any], key: str, value: Any) -> None:
    text = clean(value)
    if text and not clean(tp_rec.get(key)):
        tp_rec[key] = text


def classify_path(sensor: str, field: str, value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    name = Path(text).name.lower()
    suffix = Path(name).suffix.lower()
    field = field.lower()

    if sensor == "S5P":
        if suffix == ".nc":
            return "raw_path"
        if suffix == ".npz":
            return "npz_path"
        return "raw_path"

    if sensor == "EMIT":
        if suffix == ".nc":
            return "raw_path"
        if suffix == ".npz":
            return "npz_path"
        if "512" in name and suffix in {".tif", ".tiff"}:
            return "512_path"
        return "raw_path"

    if sensor == "L89":
        if "_std_512" in name or ("512" in name and field in {"processed_path", "existing_512_path", "target_512_path"}):
            return "512_path"
        if field in {"existing_512_path", "target_512_path"}:
            return "512_path"
        return "raw_path"

    return "raw_path"


def add_path(
    tp_rec: dict[str, Any],
    sensor: str,
    field: str,
    value: Any,
    source: str,
    exists_cache: ExistsCache,
) -> bool:
    text = clean(value)
    if not text or not exists_cache.exists(text):
        return False
    kind = classify_path(sensor, field, text)
    if not kind:
        return False
    if not clean(tp_rec.get(kind)):
        tp_rec[kind] = text
    tp_rec["_path_sources"].add(f"{kind}:{source}")
    return True


def add_metadata(tp_rec: dict[str, Any], row: dict[str, Any], sensor: str, source: str) -> None:
    image_time = clean(row.get("image_time"))
    if not image_time and sensor == "L89":
        image_time = parse_l89_file_time(row.get("image_time_file"))
    if not image_time and sensor == "EMIT":
        image_time = parse_emit_time(row.get("granule_id"), row.get("source_nc"), row.get("raw_path"), row.get("plume_id"))
    image_time = parse_time(image_time)
    fill_tp_if_blank(tp_rec, "image_time", image_time)

    product_id = clean(row.get("product_id")) or clean(row.get("granule_id")) or clean(row.get("drive_file_id")) or clean(row.get("asset_id"))
    product_name = clean(row.get("product_name")) or clean(row.get("granule_id")) or clean(row.get("drive_file_name")) or clean(row.get("asset_id"))
    fill_tp_if_blank(tp_rec, "product_id", product_id)
    fill_tp_if_blank(tp_rec, "product_name", product_name)
    fill_tp_if_blank(tp_rec, "overpass_key", row.get("overpass_key"))
    fill_tp_if_blank(tp_rec, "cloud_cover", row.get("cloud_cover") or row.get("cloud"))
    fill_tp_if_blank(tp_rec, "selection_source", row.get("selection_source") or row.get("status") or source)


def load_master(
    rows_by_sensor: dict[str, dict[str, dict[str, Any]]],
    manifest: Path,
    sensors: set[str],
    exists_cache: ExistsCache,
) -> Counter:
    counts: Counter = Counter()
    # Do not read target_512_path here. It is a planned output path in older
    # manifests and would reintroduce the ambiguity this table is meant to remove.
    path_fields = ["processed_path", "existing_512_path", "downloaded_path", "existing_raw_path"]
    with manifest.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sensor = clean(row.get("sensor")).upper()
            if sensor not in sensors:
                continue
            plume_id = clean(row.get("plume_id"))
            tp = clean(row.get("timepoint"))
            if not plume_id or tp not in TIMEPOINTS:
                continue
            out = get_record(rows_by_sensor[sensor], sensor, plume_id)
            fill_if_blank(out, "event_time", parse_time(row.get("event_time")))
            fill_if_blank(out, "plume_latitude", row.get("plume_latitude"))
            fill_if_blank(out, "plume_longitude", row.get("plume_longitude"))
            fill_if_blank(out, "plume_bounds", row.get("plume_bounds"))
            tp_rec = out["timepoints"][tp]
            if tp == "t0" and not clean(tp_rec.get("image_time")):
                fill_tp_if_blank(tp_rec, "image_time", parse_time(row.get("t0_available_time")))
            add_metadata(tp_rec, row, sensor, "master")
            for field in path_fields:
                if add_path(tp_rec, sensor, field, row.get(field), f"master.{field}", exists_cache):
                    counts[f"{sensor}.master_path"] += 1
            counts[f"{sensor}.master_rows"] += 1
    return counts


def load_l89_logs(rows: dict[str, dict[str, Any]], csv_dir: Path, exists_cache: ExistsCache) -> Counter:
    counts: Counter = Counter()
    # Submit manifest has the most useful image/cloud/product metadata.
    submit = csv_dir / "l89_gee_drive_submit_manifest.csv"
    if submit.exists():
        with submit.open(newline="") as fh:
            for row in csv.DictReader(fh):
                plume_id = clean(row.get("plume_id"))
                tp = clean(row.get("timepoint"))
                if not plume_id or tp not in TIMEPOINTS:
                    continue
                out = get_record(rows, "L89", plume_id)
                fill_if_blank(out, "event_time", parse_time(row.get("event_time")))
                tp_rec = out["timepoints"][tp]
                add_metadata(tp_rec, row, "L89", "l89_submit")
                if not clean(tp_rec.get("overpass_key")):
                    spacecraft = clean(row.get("spacecraft"))
                    wrs_path = clean(row.get("wrs_path"))
                    wrs_row = clean(row.get("wrs_row"))
                    if spacecraft or wrs_path or wrs_row:
                        tp_rec["overpass_key"] = "|".join([spacecraft, wrs_path, wrs_row])
                counts["L89.submit_rows"] += 1

    pull = csv_dir / "l89_drive_pull_manifest.csv"
    if pull.exists():
        with pull.open(newline="") as fh:
            for row in csv.DictReader(fh):
                status = clean(row.get("status"))
                if status in FAIL_STATUSES:
                    continue
                plume_id = clean(row.get("plume_id"))
                tp = clean(row.get("timepoint"))
                if not plume_id or tp not in TIMEPOINTS:
                    continue
                out = get_record(rows, "L89", plume_id)
                tp_rec = out["timepoints"][tp]
                add_metadata(tp_rec, row, "L89", "l89_pull")
                if add_path(tp_rec, "L89", "raw_path", row.get("raw_path"), "l89_pull.raw_path", exists_cache):
                    counts["L89.pull_path"] += 1
                counts["L89.pull_rows"] += 1
    return counts


def load_emit_log(rows: dict[str, dict[str, Any]], csv_dir: Path, exists_cache: ExistsCache) -> Counter:
    counts: Counter = Counter()
    path = csv_dir / "emit_download_manifest.csv"
    if not path.exists():
        return counts
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            status = clean(row.get("status"))
            if status in FAIL_STATUSES:
                continue
            plume_id = clean(row.get("plume_id"))
            tp = clean(row.get("timepoint"))
            if not plume_id or tp not in TIMEPOINTS:
                continue
            out = get_record(rows, "EMIT", plume_id)
            tp_rec = out["timepoints"][tp]
            add_metadata(tp_rec, row, "EMIT", "emit_log")
            if add_path(tp_rec, "EMIT", "source_nc", row.get("source_nc"), "emit_log.source_nc", exists_cache):
                counts["EMIT.log_raw_path"] += 1
            if add_path(tp_rec, "EMIT", "raw_path", row.get("raw_path"), "emit_log.raw_path", exists_cache):
                counts["EMIT.log_npz_path"] += 1
            counts["EMIT.log_rows"] += 1
    return counts


def load_s5p_log(rows: dict[str, dict[str, Any]], csv_dir: Path, exists_cache: ExistsCache) -> Counter:
    counts: Counter = Counter()
    path = csv_dir / "s5p_download_manifest_fixed.csv"
    if not path.exists():
        path = csv_dir / "s5p_download_manifest.csv"
    if not path.exists():
        return counts
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            status = clean(row.get("status"))
            if status in FAIL_STATUSES:
                continue
            plume_id = clean(row.get("plume_id"))
            tp = clean(row.get("timepoint"))
            if not plume_id or tp not in TIMEPOINTS:
                continue
            out = get_record(rows, "S5P", plume_id)
            tp_rec = out["timepoints"][tp]
            add_metadata(tp_rec, row, "S5P", "s5p_log")
            if add_path(tp_rec, "S5P", "raw_path", row.get("raw_path"), "s5p_log.raw_path", exists_cache):
                counts["S5P.log_raw_path"] += 1
            counts["S5P.log_rows"] += 1
    return counts


def finalize_row(row: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {key: clean(row.get(key)) for key in BASE_COLUMNS if key not in {
        "has_t0_any", "has_all6_raw", "has_all6_512", "has_all6_npz", "has_all6_any"
    }}
    tps = row["timepoints"]
    has_raw = {tp: bool(clean(tps[tp].get("raw_path"))) for tp in TIMEPOINTS}
    has_512 = {tp: bool(clean(tps[tp].get("512_path"))) for tp in TIMEPOINTS}
    has_npz = {tp: bool(clean(tps[tp].get("npz_path"))) for tp in TIMEPOINTS}
    has_any = {tp: has_raw[tp] or has_512[tp] or has_npz[tp] for tp in TIMEPOINTS}
    out["has_t0_any"] = "1" if has_any["t0"] else "0"
    out["has_all6_raw"] = "1" if all(has_raw.values()) else "0"
    out["has_all6_512"] = "1" if all(has_512.values()) else "0"
    out["has_all6_npz"] = "1" if all(has_npz.values()) else "0"
    out["has_all6_any"] = "1" if all(has_any.values()) else "0"
    for tp in TIMEPOINTS:
        tp_rec = tps[tp]
        tp_rec["path_source"] = ";".join(sorted(tp_rec.get("_path_sources", set())))
        for col in TP_COLUMNS:
            out[f"{tp}_{col}"] = clean(tp_rec.get(col))
    return {field: out.get(field, "") for field in FIELDNAMES}


def write_sensor_table(sensor: str, rows: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, int]:
    out_path = out_dir / f"{sensor.lower()}_6time_merged_paths.csv"
    finalized = [finalize_row(row) for row in rows.values()]
    finalized = [row for row in finalized if row["has_t0_any"] == "1"]
    finalized.sort(key=lambda row: row["plume_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(finalized)
    return {
        "rows": len(finalized),
        "t0": sum(1 for row in finalized if row["has_t0_any"] == "1"),
        "all6_any": sum(1 for row in finalized if row["has_all6_any"] == "1"),
        "all6_raw": sum(1 for row in finalized if row["has_all6_raw"] == "1"),
        "all6_512": sum(1 for row in finalized if row["has_all6_512"] == "1"),
        "all6_npz": sum(1 for row in finalized if row["has_all6_npz"] == "1"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean six-time wide path tables per sensor.")
    parser.add_argument("--manifest", default="Upgrade_data_pipeline/csv/multisensor_6time_download_manifest.csv")
    parser.add_argument("--csv-dir", default="Upgrade_data_pipeline/csv")
    parser.add_argument("--out-dir", default="Upgrade_data_pipeline/csv")
    parser.add_argument("--sensors", default="L89,EMIT,S5P")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sensors = {sensor.strip().upper() for sensor in args.sensors.split(",") if sensor.strip()}
    unsupported = sensors - {"L89", "EMIT", "S5P"}
    if unsupported:
        raise ValueError(f"unsupported sensors for this run: {sorted(unsupported)}")
    manifest = Path(args.manifest)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    exists_cache = ExistsCache()
    rows_by_sensor: dict[str, dict[str, dict[str, Any]]] = {sensor: {} for sensor in sensors}
    counts = Counter()
    counts.update(load_master(rows_by_sensor, manifest, sensors, exists_cache))
    if "L89" in sensors:
        counts.update(load_l89_logs(rows_by_sensor["L89"], csv_dir, exists_cache))
    if "EMIT" in sensors:
        counts.update(load_emit_log(rows_by_sensor["EMIT"], csv_dir, exists_cache))
    if "S5P" in sensors:
        counts.update(load_s5p_log(rows_by_sensor["S5P"], csv_dir, exists_cache))

    for sensor in sorted(sensors):
        summary = write_sensor_table(sensor, rows_by_sensor[sensor], out_dir)
        print(f"{sensor}: {summary}")
    print(f"paths_checked: {len(exists_cache.cache)}")
    print(f"source_counts: {dict(counts.most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
