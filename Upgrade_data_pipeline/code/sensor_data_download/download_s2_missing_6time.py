#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import math
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from manifest_state import (
    ensure_manifest_columns,
    load_master_completed_records,
    select_rows_for_missing_download,
    update_master_from_records,
)


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
PREV_INDEX = {"prev1": 1, "prev2": 2, "prev3": 3}
DEFAULT_SCRATCH_PRODUCT_DIR = "/diniuvol/yuyao/s2_download_scratch"
RAW_TIF_NAME = {
    "t0": "s2.tif",
    "prev1": "s2_-7.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_-90.tif",
    "year": "s2_-360.tif",
}
SUCCESS_STATUSES = {
    "downloaded",
    "skip_existing_raw",
    "skip_existing_512",
    "master_completed",
    "resume_skip_completed",
}
OUT_FIELDS = [
    "plume_id",
    "timepoint",
    "status",
    "raw_path",
    "target_512_path",
    "existing_512_path",
    "product_name",
    "product_id",
    "acquisition_time",
    "query_start_utc",
    "query_end_utc",
    "selection_source",
    "message",
]


def log(message: str) -> None:
    print(message, flush=True)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def existing_file(value: Any) -> Optional[Path]:
    if not has_value(value):
        return None
    path = Path(str(value).strip())
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def parse_utc(value: Any) -> Optional[datetime]:
    if not has_value(value):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_manifest(path: str, timepoints: set[str], limit: int) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sensor = str(row.get("sensor", "")).strip().upper()
            action = str(row.get("action", "")).strip()
            if sensor != "S2" or action != "download":
                continue
            all_rows.append(row)
    rows = select_rows_for_missing_download(all_rows, timepoints)
    if limit:
        rows = rows[:limit]
    return rows


def group_rows_by_plume(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        plume_id = str(row.get("plume_id", "")).strip()
        groups.setdefault(plume_id, []).append(row)
    return list(groups.values())


def completion_ok(row: dict[str, str], check_files: bool = True) -> bool:
    status = str(row.get("status", "")).strip()
    if status not in SUCCESS_STATUSES:
        return False
    if str(row.get("timepoint", "")).strip() == "t0":
        if not (
            has_value(row.get("product_name"))
            and has_value(row.get("product_id"))
            and has_value(row.get("acquisition_time"))
        ):
            return False
    if not check_files:
        return (
            has_value(row.get("raw_path"))
            or has_value(row.get("target_512_path"))
            or has_value(row.get("existing_512_path"))
        )
    if existing_file(row.get("raw_path")) is not None:
        return True
    if existing_file(row.get("target_512_path")) is not None:
        return True
    if existing_file(row.get("existing_512_path")) is not None:
        return True
    return False


def load_completed_records(path: str, check_files: bool = True) -> dict[tuple[str, str], dict[str, str]]:
    out = Path(path)
    if not out.exists():
        return {}
    done: dict[tuple[str, str], dict[str, str]] = {}
    with out.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not completion_ok(row, check_files=check_files):
                continue
            plume_id = str(row.get("plume_id", "")).strip()
            tp = str(row.get("timepoint", "")).strip()
            if plume_id and tp:
                done[(plume_id, tp)] = row
    return done


def load_completed(path: str) -> set[tuple[str, str]]:
    return set(load_completed_records(path).keys())


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in OUT_FIELDS})


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def load_legacy_s2_module(repo_root: Path):
    path = repo_root / "data_preprocess" / "carbon_mapper_sentinel2_90360_plume_download.py"
    if not path.exists():
        raise FileNotFoundError(f"legacy S2 module not found: {path}")
    spec = importlib.util.spec_from_file_location("legacy_s2_90360", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy S2 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_config(legacy: Any, config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    return legacy.load_config(config_path)


def load_cdse_credentials(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, str]]:
    if args.cdse_username or args.cdse_password:
        if not args.cdse_username or not args.cdse_password:
            raise RuntimeError("provide both --cdse-username and --cdse-password")
        return [{"username": args.cdse_username, "password": args.cdse_password}]

    pool: list[dict[str, str]] = []
    idx = args.cdse_env_index
    while True:
        username = os.environ.get(f"CDSE_USERNAME{idx}") or config.get(f"cdse_username{idx}")
        password = os.environ.get(f"CDSE_PASSWORD{idx}") or config.get(f"cdse_password{idx}")
        if username and password:
            pool.append({"username": username, "password": password})
            idx += 1
            continue
        if username or password:
            raise RuntimeError(f"incomplete CDSE credential pair at index {idx}")
        break

    if not pool:
        raise RuntimeError(
            f"CDSE credentials not provided. Set CDSE_USERNAME{args.cdse_env_index}/"
            f"CDSE_PASSWORD{args.cdse_env_index} "
            "or pass --cdse-username/--cdse-password."
        )
    return pool


def start_token_pool(legacy: Any, credentials: list[dict[str, str]]) -> list[Any]:
    tokens: list[Any] = []
    for cred in credentials:
        token = legacy.RefreshableAccessToken(cred["username"], cred["password"])
        tokens.append(token)
        thread = threading.Thread(target=legacy.refresh_variable, args=(token,), daemon=True)
        thread.start()
    return tokens


def parse_bounds(row: dict[str, Any]) -> tuple[list[float], str]:
    raw_bounds = row.get("plume_bounds")
    bounds: Optional[list[float]] = None
    if has_value(raw_bounds):
        try:
            parsed = ast.literal_eval(str(raw_bounds))
            if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
                bounds = [float(v) for v in parsed]
        except Exception:
            bounds = None
    if bounds is None:
        lat = float(row["plume_latitude"])
        lon = float(row["plume_longitude"])
        bounds = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]

    min_lon, min_lat, max_lon, max_lat = bounds
    poly = (
        f"({min_lon} {min_lat},{min_lon} {max_lat},"
        f"{max_lon} {max_lat},{max_lon} {min_lat},{min_lon} {min_lat})"
    )
    return bounds, poly


def point_centered_crop_bounds(row: dict[str, Any]) -> list[float]:
    latitude = float(row["plume_latitude"])
    longitude = float(row["plume_longitude"])
    return [
        longitude - 0.01,
        latitude - 0.01,
        longitude + 0.01,
        latitude + 0.01,
    ]


def target_raw_path(row: dict[str, Any], raw_root: str) -> Path:
    plume_id = str(row.get("plume_id", "")).strip()
    tp = str(row.get("timepoint", "")).strip()
    if has_value(row.get("target_raw_dir")):
        target_dir = Path(str(row["target_raw_dir"]).strip())
    else:
        target_dir = Path(raw_root) / "S2" / tp / plume_id
    return target_dir / RAW_TIF_NAME[tp]


def s2_overpass_key(product: dict[str, Any]) -> str:
    acq = product.get("acq_time")
    name = str(product.get("Name", ""))
    orbit = ""
    match = re.search(r"_R(\d{3})_", name)
    if match:
        orbit = match.group(1)
    if acq is None:
        return f"S2|R{orbit}|unknown"
    acq = acq.astimezone(timezone.utc)
    minute = (acq.minute // 10) * 10
    bucket = f"{acq:%Y%m%dT%H}{minute:02d}"
    return f"S2|R{orbit}|{bucket}"


def product_from_completed(row: Optional[dict[str, str]]) -> Optional[dict[str, Any]]:
    if not row:
        return None
    acq = parse_utc(row.get("acquisition_time"))
    product_id = str(row.get("product_id", "")).strip()
    product_name = str(row.get("product_name", "")).strip()
    if acq is None or not product_id or not product_name:
        return None
    return {"Id": product_id, "Name": product_name, "acq_time": acq}


def s2_overpass_key_from_values(product_name: Any, acquisition_time: Any) -> str:
    acq = parse_utc(acquisition_time)
    name = str(product_name or "")
    orbit = ""
    match = re.search(r"_R(\d{3})_", name)
    if match:
        orbit = match.group(1)
    if acq is None:
        return f"S2|R{orbit}|unknown"
    minute = (acq.minute // 10) * 10
    return f"S2|R{orbit}|{acq:%Y%m%dT%H}{minute:02d}"


def master_record_to_s2(row: dict[str, str]) -> dict[str, str]:
    return {
        "plume_id": row.get("plume_id", ""),
        "timepoint": row.get("timepoint", ""),
        "status": "master_completed",
        "raw_path": row.get("downloaded_path", ""),
        "target_512_path": row.get("target_512_path", ""),
        "existing_512_path": row.get("processed_path", ""),
        "product_name": row.get("product_name", ""),
        "product_id": row.get("product_id", ""),
        "acquisition_time": row.get("image_time", ""),
        "query_start_utc": "",
        "query_end_utc": "",
        "selection_source": row.get("selection_source", ""),
        "message": row.get("status_message", ""),
    }


def group_by_overpass(products: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_pass: dict[str, list[dict[str, Any]]] = {}
    for product in products:
        acq = product.get("acq_time")
        if acq is None:
            continue
        by_pass.setdefault(s2_overpass_key(product), []).append(product)
    return sorted(by_pass.values(), key=lambda group: max(p["acq_time"] for p in group), reverse=True)


def select_previous_product(legacy: Any, products: list[dict[str, Any]], t0_product: dict[str, Any], n: int) -> Optional[dict[str, Any]]:
    anchor = t0_product["acq_time"].astimezone(timezone.utc)
    t0_product_id = str(t0_product.get("Id", "")).strip()
    t0_key = s2_overpass_key(t0_product)
    prior = [
        product
        for product in products
        if product.get("acq_time") is not None
        and product["acq_time"].astimezone(timezone.utc) < anchor
        and str(product.get("Id", "")).strip() != t0_product_id
        and s2_overpass_key(product) != t0_key
    ]
    groups = group_by_overpass(prior)
    if len(groups) < n:
        return None
    return legacy.select_closest_product(groups[n - 1], anchor)


def find_product(
    legacy: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    t0_product: Optional[dict[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], str, str, str, str]:
    tp = str(row["timepoint"]).strip()
    event_dt = parse_utc(row.get("event_time"))
    if event_dt is None:
        return None, "", "", "invalid_time", "missing or invalid event_time"

    t0_dt = parse_utc(row.get("t0_available_time")) or event_dt
    _, poly = parse_bounds(row)

    if tp == "t0":
        start = event_dt
        end = event_dt + timedelta(hours=24)
        target = event_dt
        source = "t0_24h_window"
        selector = lambda products: legacy.select_closest_product(products, target)
    elif tp in PREV_INDEX:
        if t0_product is None:
            return None, "", "", "missing_t0_anchor", "prev requires actual t0 product/time"
        t0_dt = t0_product["acq_time"].astimezone(timezone.utc)
        start = t0_dt - timedelta(days=args.prev_search_back_days)
        end = t0_dt
        source = f"{tp}_distinct_overpass_before_actual_t0"
        selector = lambda products: select_previous_product(legacy, products, t0_product, PREV_INDEX[tp])
    elif tp == "seasonal":
        target = event_dt - timedelta(days=90)
        start = target - timedelta(days=args.offset_search_window_days)
        end = target
        source = "legacy_minus90_window"
        selector = lambda products: legacy.select_closest_product(products, target)
    elif tp == "year":
        target = event_dt - timedelta(days=args.year_offset_days)
        start = target - timedelta(days=args.offset_search_window_days)
        end = target
        source = f"legacy_minus{args.year_offset_days}_window"
        selector = lambda products: legacy.select_closest_product(products, target)
    else:
        return None, "", "", "unknown_timepoint", tp

    start_s = legacy.datetime_to_query_string(start)
    end_s = legacy.datetime_to_query_string(end)
    products = legacy.fetch_products(poly, start_s, end_s)
    selected = selector(products)
    if selected is None:
        return None, start_s, end_s, "no_product", source
    return selected, start_s, end_s, source, ""


def base_record(row: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    return {
        "plume_id": str(row.get("plume_id", "")).strip(),
        "timepoint": str(row.get("timepoint", "")).strip(),
        "status": "",
        "raw_path": str(raw_path),
        "target_512_path": str(row.get("target_512_path", "")).strip(),
        "existing_512_path": str(row.get("existing_512_path", "")).strip(),
        "product_name": "",
        "product_id": "",
        "acquisition_time": "",
        "query_start_utc": "",
        "query_end_utc": "",
        "selection_source": "",
        "message": "",
    }


def resume_record_from_completed(
    row: dict[str, Any],
    raw_path: Path,
    completed_record: Optional[dict[str, str]],
) -> dict[str, Any]:
    record = base_record(row, raw_path)
    if completed_record:
        for field in [
            "raw_path",
            "target_512_path",
            "existing_512_path",
            "product_name",
            "product_id",
            "acquisition_time",
            "query_start_utc",
            "query_end_utc",
            "selection_source",
            "message",
        ]:
            value = completed_record.get(field, "")
            if has_value(value):
                record[field] = value
    record["status"] = "resume_skip_completed"
    return record


def completed_record_has_file(record: Optional[dict[str, str]]) -> bool:
    if not record:
        return False
    return (
        existing_file(record.get("raw_path")) is not None
        or existing_file(record.get("target_512_path")) is not None
        or existing_file(record.get("existing_512_path")) is not None
    )


def record_is_success(record: dict[str, Any]) -> bool:
    return str(record.get("status", "")).strip() in SUCCESS_STATUSES


def skip_t0_failed_record(row: dict[str, Any], raw_root: str, message: str) -> dict[str, Any]:
    record = base_record(row, target_raw_path(row, raw_root))
    record["status"] = "skip_t0_failed"
    record["message"] = message
    return record


def download_with_token_refresh(
    legacy: Any,
    token: Any,
    args: argparse.Namespace,
    plume_id: str,
    product_id: str,
    product_name: str,
    bounds: list[float],
    raw_path: Path,
) -> tuple[Optional[tuple[int, int]], int, str]:
    last_message = ""
    attempts = max(1, args.auth_retries + 1)
    for attempt in range(1, attempts + 1):
        dims = legacy.download(
            token.get(),
            args.raw_product_dir,
            plume_id,
            product_id,
            product_name,
            bounds,
            str(raw_path),
            cleanup_product=(args.scratch_cleanup == "immediate"),
        )
        if dims is not None:
            return dims, attempt, ""
        if attempt >= attempts:
            break
        last_message = f"legacy download returned None on attempt {attempt}; refreshed CDSE token and retried"
        log(f"{last_message}: {plume_id} {product_name}")
        try:
            token.update()
        except Exception as exc:
            last_message = f"token refresh failed after download failure: {exc}"
            log(f"{last_message}: {plume_id} {product_name}")
            break
        if args.auth_retry_sleep > 0:
            time.sleep(args.auth_retry_sleep)
    return None, attempts, last_message


def process_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    legacy: Any,
    token: Any,
    completed: set[tuple[str, str]],
    completed_records: dict[tuple[str, str], dict[str, str]],
    t0_product: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    plume_id = str(row.get("plume_id", "")).strip()
    tp = str(row.get("timepoint", "")).strip()
    raw_path = target_raw_path(row, args.raw_root)
    record = base_record(row, raw_path)

    if args.resume and (plume_id, tp) in completed and not args.overwrite:
        completed_record = completed_records.get((plume_id, tp))
        if completed_record_has_file(completed_record):
            return resume_record_from_completed(row, raw_path, completed_record)

    selected: Optional[dict[str, Any]] = None
    query_start = ""
    query_end = ""
    source = ""
    message = ""
    selected, query_start, query_end, source, message = find_product(legacy, row, args, t0_product)
    record["query_start_utc"] = query_start
    record["query_end_utc"] = query_end
    record["selection_source"] = source
    record["message"] = message
    if selected is not None:
        record["product_name"] = selected["Name"]
        record["product_id"] = selected["Id"]
        record["acquisition_time"] = iso_z(selected["acq_time"])
    elif tp in PREV_INDEX:
        record["status"] = "no_product"
        return record

    found_512 = existing_file(row.get("existing_512_path")) or existing_file(row.get("target_512_path"))
    if found_512 is not None and not args.overwrite:
        record["status"] = "skip_existing_512"
        record["message"] = str(found_512)
        return record

    if existing_file(raw_path) is not None and not args.overwrite:
        record["status"] = "skip_existing_raw"
        return record

    if selected is None:
        record["status"] = "no_product"
        return record

    product_name = selected["Name"]
    product_id = selected["Id"]
    acquisition_time = selected["acq_time"]
    record["product_name"] = product_name
    record["product_id"] = product_id
    record["acquisition_time"] = iso_z(acquisition_time)

    bounds = point_centered_crop_bounds(row)
    product_lock = legacy.get_product_lock(product_name)
    with product_lock:
        dims, attempts, retry_message = download_with_token_refresh(
            legacy,
            token,
            args,
            plume_id,
            product_id,
            product_name,
            bounds,
            raw_path,
        )
    if dims is None:
        record["status"] = "download_failed"
        record["message"] = retry_message or f"legacy download failed after {attempts} attempts"
        return record
    if existing_file(raw_path) is None:
        record["status"] = "download_failed"
        record["message"] = "legacy download returned but raw tif is missing"
        return record

    record["status"] = "downloaded"
    record["message"] = f"{int(dims[0])}x{int(dims[1])}; attempts={attempts}"
    return record


def process_plume_group(
    group_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    legacy: Any,
    token: Any,
    completed: set[tuple[str, str]],
    completed_records: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    if not group_rows:
        return []
    plume_id = str(group_rows[0].get("plume_id", "")).strip()
    t0_rows = [row for row in group_rows if str(row.get("timepoint", "")).strip() == "t0"]
    other_rows = [row for row in group_rows if str(row.get("timepoint", "")).strip() != "t0"]
    records: list[dict[str, Any]] = []

    t0_completed_record = completed_records.get((plume_id, "t0"))
    t0_ok = completed_record_has_file(t0_completed_record)
    t0_product = product_from_completed(t0_completed_record) if t0_ok else None
    t0_message = "t0 is not completed in the current output manifest"
    for row in t0_rows:
        rec = process_row(row, args, legacy, token, completed, completed_records)
        records.append(rec)
        if record_is_success(rec):
            t0_ok = True
            t0_message = ""
            t0_product = product_from_completed(rec) or t0_product
        else:
            t0_message = (
                f"t0 status={rec.get('status', '')}; "
                f"message={rec.get('message', '') or rec.get('selection_source', '')}"
            )

    if t0_ok:
        for row in other_rows:
            records.append(process_row(row, args, legacy, token, completed, completed_records, t0_product))
    else:
        for row in other_rows:
            records.append(skip_t0_failed_record(row, args.raw_root, t0_message))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing Sentinel-2 six-time raw crops using the legacy CDSE S2 pipeline."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", default="Upgrade_data_pipeline/csv/s2_download_manifest.csv")
    parser.add_argument("--raw-root", default="/mnt/engg-niulab/yuyao/sensors_raw_data")
    parser.add_argument("--raw-product-dir", default="")
    parser.add_argument(
        "--scratch-product-dir",
        default=DEFAULT_SCRATCH_PRODUCT_DIR,
        help="Local scratch directory for temporary Sentinel-2 SAFE zip/extracted product cache.",
    )
    parser.add_argument(
        "--scratch-cleanup",
        choices=["end", "immediate", "none"],
        default="end",
        help=(
            "Clean local scratch products. 'end' keeps extracted SAFE products during this run "
            "for reuse and removes them on normal exit; 'immediate' removes each SAFE after one "
            "crop; 'none' leaves scratch products for a later resume."
        ),
    )
    parser.add_argument("--legacy-config", default="")
    parser.add_argument("--timepoints", default="t0,prev1,prev2,prev3,seasonal,year")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prev-search-back-days", type=int, default=120)
    parser.add_argument("--offset-search-window-days", type=int, default=50)
    parser.add_argument("--year-offset-days", type=int, default=360)
    parser.add_argument("--cdse-env-index", type=int, default=1)
    parser.add_argument("--cdse-username", default="")
    parser.add_argument("--cdse-password", default="")
    parser.add_argument("--auth-retries", type=int, default=3)
    parser.add_argument("--auth-retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-master-update", action="store_true")
    parser.add_argument(
        "--master-update-interval",
        type=int,
        default=1000,
        help="Sync completed output records back to the master manifest every N log rows. "
        "Use 0 to sync only once at the end.",
    )
    return parser.parse_args()


def cleanup_scratch_products(raw_product_dir: str) -> int:
    product_dir = Path(raw_product_dir)
    if not product_dir.exists():
        return 0
    removed = 0
    for path in list(product_dir.glob("*.SAFE.zip")) + list(product_dir.glob("*.SAFE")):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except Exception as exc:
            log(f"failed to clean scratch product {path}: {exc}")
    return removed


def record_to_master_update(record: dict[str, Any]) -> dict[str, Any]:
    raw_path = existing_file(record.get("raw_path"))
    processed_path = existing_file(record.get("existing_512_path")) or existing_file(record.get("target_512_path"))
    status = str(record.get("status", "")).strip()
    update = {
        "download_status": status,
        "selection_source": record.get("selection_source", ""),
        "status_message": record.get("message", ""),
    }
    if raw_path is not None:
        update["downloaded_path"] = str(raw_path)
    if processed_path is not None:
        update["processed_path"] = str(processed_path)
    if has_value(record.get("acquisition_time")):
        update["image_time"] = record.get("acquisition_time", "")
    if has_value(record.get("product_id")):
        update["product_id"] = record.get("product_id", "")
    if has_value(record.get("product_name")):
        update["product_name"] = record.get("product_name", "")
        update["overpass_key"] = s2_overpass_key_from_values(
            record.get("product_name", ""),
            record.get("acquisition_time", ""),
        )
    return update


def sync_master_updates(args: argparse.Namespace, records: list[dict[str, Any]], context: str) -> int:
    if args.no_master_update or not records:
        return 0
    changed = update_master_from_records(
        args.manifest,
        "S2",
        records,
        record_to_master_update,
        source_log=args.out_csv,
    )
    log(f"updated master manifest rows ({context}): {changed}")
    return changed


def main() -> int:
    args = parse_args()
    requested_timepoints = {tp.strip() for tp in args.timepoints.split(",") if tp.strip()}
    unknown = sorted(requested_timepoints - set(TIMEPOINTS))
    if unknown:
        raise ValueError(f"unknown timepoints: {unknown}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    if not args.raw_product_dir:
        args.raw_product_dir = args.scratch_product_dir
    Path(args.raw_product_dir).mkdir(parents=True, exist_ok=True)
    log(f"S2 product scratch/cache dir: {args.raw_product_dir}")

    ensure_manifest_columns(args.manifest)
    rows = read_manifest(args.manifest, requested_timepoints, args.limit)
    out_csv = Path(args.out_csv)
    master_completed_records = {
        key: master_record_to_s2(row)
        for key, row in load_master_completed_records(args.manifest, "S2").items()
    }
    completed_records = dict(master_completed_records)
    if args.resume:
        out_completed_records = load_completed_records(args.out_csv, check_files=False)
        completed_records.update(out_completed_records)
        log(f"loaded completed S2 rows from output manifest: {len(out_completed_records)}")
    completed = set(completed_records.keys())
    log(f"loaded S2 download rows: {len(rows)}")
    log(f"loaded completed S2 rows from master manifest: {len(master_completed_records)}")
    log(f"loaded completed S2 rows total for resume: {len(completed)}")

    repo_root = repo_root_from_script()
    legacy = load_legacy_s2_module(repo_root)
    config = load_legacy_config(legacy, args.legacy_config)
    with legacy.proxy_manager_lock:
        legacy.proxy_manager = legacy.build_proxy_manager(config)
    credentials = load_cdse_credentials(args, config)
    token_pool = start_token_pool(legacy, credentials)

    groups = group_rows_by_plume(rows)
    log(f"loaded S2 plume groups: {len(groups)}")
    completed_rows = 0
    pending_master_records: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for idx, group in enumerate(groups):
                token = token_pool[idx % len(token_pool)]
                futures.append(executor.submit(process_plume_group, group, args, legacy, token, completed, completed_records))
            for future in as_completed(futures):
                try:
                    records = future.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    record = {field: "" for field in OUT_FIELDS}
                    record["status"] = "error"
                    record["message"] = str(exc)
                    records = [record]
                for record in records:
                    pending_master_records.append(record)
                    completed_rows += 1
                    append_row(out_csv, record)
                    log(
                        f"[{completed_rows}/{len(rows)}] {record.get('status', '')} "
                        f"{record.get('plume_id', '')} {record.get('timepoint', '')} "
                        f"{record.get('product_name', '')}"
                    )
                    if args.master_update_interval > 0 and len(pending_master_records) >= args.master_update_interval:
                        sync_master_updates(args, pending_master_records, f"periodic after {completed_rows} rows")
                        pending_master_records.clear()
                    time.sleep(0.05)
    finally:
        sync_master_updates(args, pending_master_records, "final")
        if args.scratch_cleanup == "end":
            removed = cleanup_scratch_products(args.raw_product_dir)
            log(f"cleaned S2 scratch products on exit: {removed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
