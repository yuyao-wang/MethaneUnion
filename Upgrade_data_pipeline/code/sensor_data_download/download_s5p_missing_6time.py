#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import importlib.util
import math
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from manifest_state import (
    ensure_manifest_columns,
    load_master_completed_records,
    select_rows_for_missing_download,
    update_master_from_records,
)

S5P_PREPROCESS_DIR = Path(__file__).resolve().parents[1] / "S5P_preprocess"
if str(S5P_PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(S5P_PREPROCESS_DIR))
from s5p_crop_qc import qc_s5p_product


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
PREV_INDEX = {"prev1": 1, "prev2": 2, "prev3": 3}
SUCCESS_STATUSES = {
    "downloaded",
    "skip_existing",
    "master_completed",
    "resume_skip_completed",
    "downloaded_crop_ok",
    "skip_existing_crop_ok",
    "master_completed_crop_ok",
    "resume_skip_completed_crop_ok",
}
CROP_OK_STATUSES = {
    "downloaded_crop_ok",
    "skip_existing_crop_ok",
    "master_completed_crop_ok",
    "resume_skip_completed_crop_ok",
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
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_old_poly(lon: float, lat: float, half_deg: float) -> str:
    down_left = (lon - half_deg, lat - half_deg)
    up_right = (lon + half_deg, lat + half_deg)
    return (
        f"({down_left[0]} {down_left[1]},"
        f"{down_left[0]} {up_right[1]},"
        f"{up_right[0]} {up_right[1]},"
        f"{up_right[0]} {down_left[1]},"
        f"{down_left[0]} {down_left[1]})"
    )


def validate_nc(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, "missing_or_empty"
    # Keep this consistent with the legacy S5P downloader: do not open NetCDF
    # files here. netCDF4/HDF5 is not safe under this threaded downloader on the
    # mounted niulab filesystem and can segfault after getfattr warnings.
    return True, "size_ok"


def existing_nc(raw_dir: Path, product_name: str) -> Optional[Path]:
    candidates = list(raw_dir.glob("*.nc")) + list(raw_dir.glob("*_extracted/**/*.nc"))
    stem = product_name[:-4] if product_name.endswith(".nc") else product_name
    for path in candidates:
        if path.name == product_name or stem in path.name:
            ok, _ = validate_nc(path)
            if ok:
                return path
    return None


def raw_path_matches_product(raw_path: Any, product_name: Any) -> bool:
    if not has_value(raw_path):
        return False
    if not has_value(product_name):
        return True
    name = Path(str(raw_path)).name
    product = str(product_name).strip()
    stem = product[:-3] if product.endswith(".nc") else product
    return name == product or stem in name


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


def load_completed_records(out_csv: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = Path(out_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    required = {"plume_id", "timepoint", "raw_path", "status"}
    if not required.issubset(df.columns):
        return {}
    done: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        if str(row.get("status", "")).strip() not in SUCCESS_STATUSES:
            continue
        raw_path = row.get("raw_path", "")
        if not has_value(raw_path):
            continue
        if not raw_path_matches_product(raw_path, row.get("product_name", "")):
            continue
        ok, _ = validate_nc(Path(str(raw_path)))
        if ok:
            done[(str(row["plume_id"]).strip(), str(row["timepoint"]).strip())] = row.to_dict()
    return done


def load_completed(out_csv: str) -> set[tuple[str, str]]:
    return set(load_completed_records(out_csv).keys())


def load_keep_plume_ids(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    if not path.exists() or path.stat().st_size <= 0:
        return set()
    df = pd.read_csv(path, usecols=["plume_id"])
    return {str(value).strip() for value in df["plume_id"] if has_value(value)}


RESUME_TERMINAL_STATUSES = SUCCESS_STATUSES | {
    "no_crop_valid_product",
    "no_product",
    "skip_t0_failed",
}


def load_resume_keys(path_value: str) -> set[tuple[str, str]]:
    if not path_value:
        return set()
    path = Path(path_value)
    if not path.exists() or path.stat().st_size <= 0:
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            status = str(row.get("status", "")).strip()
            if status not in RESUME_TERMINAL_STATUSES:
                continue
            plume_id = str(row.get("plume_id", "")).strip()
            timepoint = str(row.get("timepoint", "")).strip()
            if plume_id and timepoint:
                keys.add((plume_id, timepoint))
    return keys


def load_work_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.manifest, low_memory=False)
    tps = {tp.strip() for tp in args.timepoints.split(",") if tp.strip()}
    bad = sorted(tps - set(TIMEPOINTS))
    if bad:
        raise ValueError(f"unknown timepoints: {bad}")
    sensor_mask = df["sensor"].astype(str).str.upper() == "S5P"
    if args.recheck_crop_qc:
        action_mask = df["action"].astype(str) != "skip_no_sensor_t0"
    else:
        action_mask = df["action"].astype(str) == "download"
    candidates = df[sensor_mask & action_mask].copy()
    candidates = candidates[candidates["timepoint"].astype(str).isin(tps)].copy()

    keep_ids = load_keep_plume_ids(args.keep_existing_csv) if args.skip_kept_plumes else set()
    if keep_ids:
        candidates = candidates[~candidates["plume_id"].astype(str).isin(keep_ids)].copy()

    if args.resume and not args.overwrite:
        resume_keys = load_resume_keys(args.out_csv)
        if resume_keys:
            keys = list(zip(candidates["plume_id"].astype(str), candidates["timepoint"].astype(str)))
            candidates = candidates[[key not in resume_keys for key in keys]].copy()

    if args.recheck_crop_qc:
        rows = candidates.copy()
    else:
        selected = select_rows_for_missing_download(candidates.to_dict("records"), tps, overwrite=args.overwrite)
        rows = pd.DataFrame(selected, columns=candidates.columns)
    if args.limit:
        rows = rows.head(args.limit)
    return rows


def load_legacy_module(repo_root: Path):
    legacy_path = repo_root / "data_downloading" / "carbon_mapper_sentinel5p_90360_plume_download.py"
    spec = importlib.util.spec_from_file_location("legacy_s5p_download", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy S5P script: {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_config(legacy: Any, config_path: str) -> dict[str, Any]:
    if config_path and Path(config_path).exists():
        return legacy.load_config(config_path)
    return {}


def init_legacy_runtime(args: argparse.Namespace):
    repo_root = Path.cwd()
    legacy = load_legacy_module(repo_root)
    config = load_legacy_config(legacy, args.legacy_config)

    if args.cdse_username:
        config["cdse_username0"] = args.cdse_username
    if args.cdse_password:
        config["cdse_password0"] = args.cdse_password

    manager = legacy.build_proxy_manager(config)
    with legacy.proxy_manager_lock:
        legacy.proxy_manager = manager

    credentials = legacy.load_credential_pool(config)
    if not credentials:
        raise RuntimeError("CDSE credentials not found. Set CDSE_USERNAME0/CDSE_PASSWORD0 or pass --cdse-username/--cdse-password.")

    tokens = []
    for cred in credentials:
        token = legacy.RefreshableAccessToken(cred["username"], cred["password"])
        tokens.append(token)
        thread = threading.Thread(target=legacy.refresh_variable, args=(token,))
        thread.daemon = True
        thread.start()
    return legacy, tokens


def fetch_products(legacy: Any, poly: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    return legacy.fetch_products(poly, legacy.datetime_to_query_string(start), legacy.datetime_to_query_string(end))


def s5p_overpass_key(product: dict[str, Any]) -> str:
    name = str(product.get("Name", "")).strip()
    match = re.search(r"_(\d{5})_\d{2}_\d{6}_", name)
    if match:
        return match.group(1)
    acq = product.get("acq_time")
    if acq:
        return acq.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return name


def product_from_completed(row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not row:
        return None
    acq = parse_dt(row.get("image_time"))
    product_id = str(row.get("product_id", "")).strip()
    product_name = str(row.get("product_name", "")).strip()
    if acq is None or not product_id or not product_name:
        return None
    return {"Id": product_id, "Name": product_name, "acq_time": acq}


def master_record_to_s5p(row: dict[str, str]) -> dict[str, Any]:
    status = str(row.get("download_status", "")).strip()
    completed_status = "master_completed_crop_ok" if status in CROP_OK_STATUSES else "master_completed"
    return {
        "plume_id": row.get("plume_id", ""),
        "timepoint": row.get("timepoint", ""),
        "status": completed_status,
        "product_id": row.get("product_id", ""),
        "product_name": row.get("product_name", ""),
        "image_time": row.get("image_time", ""),
        "selection_source": row.get("selection_source", ""),
        "raw_path": row.get("downloaded_path", ""),
        "raw_data_dir": "",
        "target_raw_dir_from_manifest": row.get("target_raw_dir", ""),
        "target_time": row.get("target_time", ""),
        "time_delta_hours": row.get("time_delta_hours", ""),
        "qc_ok": row.get("qc_ok", ""),
        "qc_reason": row.get("qc_reason", ""),
        "qc_center_iy": row.get("qc_center_iy", ""),
        "qc_center_ix": row.get("qc_center_ix", ""),
        "qc_center_distance_km": row.get("qc_center_distance_km", ""),
        "qc_patch_missing_ratio": row.get("qc_patch_missing_ratio", ""),
        "qc_patch_finite_count": row.get("qc_patch_finite_count", ""),
        "qc_patch_total": row.get("qc_patch_total", ""),
        "qc_ch4_var": row.get("qc_ch4_var", ""),
        "qc_candidate_rank": row.get("qc_candidate_rank", ""),
        "qc_candidates_checked": row.get("qc_candidates_checked", ""),
        "candidate_attempts": row.get("candidate_attempts", ""),
        "message": row.get("status_message", ""),
    }


def select_prev(products: list[dict[str, Any]], t0_product: dict[str, Any], n: int) -> Optional[dict[str, Any]]:
    anchor = t0_product["acq_time"].astimezone(timezone.utc)
    t0_id = str(t0_product.get("Id", "")).strip()
    t0_key = s5p_overpass_key(t0_product)
    before = [
        p
        for p in products
        if p.get("acq_time")
        and p["acq_time"] < anchor
        and p.get("Id")
        and str(p.get("Id", "")).strip() != t0_id
        and s5p_overpass_key(p) != t0_key
    ]
    by_pass: dict[str, dict[str, Any]] = {}
    for product in before:
        key = s5p_overpass_key(product)
        old = by_pass.get(key)
        if old is None or product["acq_time"] > old["acq_time"]:
            by_pass[key] = product
    ordered = sorted(by_pass.values(), key=lambda p: p["acq_time"], reverse=True)
    if len(ordered) < n:
        return None
    return ordered[n - 1]


def select_product(row: pd.Series, tp: str, args: argparse.Namespace, legacy: Any, t0_product: Optional[dict[str, Any]] = None) -> tuple[Optional[dict[str, Any]], str]:
    event_dt = parse_dt(row.get("event_time"))
    t0_dt = parse_dt(row.get("t0_available_time")) or event_dt
    if event_dt is None or t0_dt is None:
        return None, "invalid_time"
    lat = row.get("plume_latitude")
    lon = row.get("plume_longitude")
    if pd.isna(lat) or pd.isna(lon):
        return None, "missing_latlon"

    poly = build_old_poly(float(lon), float(lat), args.geo_half_deg)
    if tp == "t0":
        day = event_dt.date()
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        products = fetch_products(legacy, poly, start, end)
        return legacy.select_closest_product(products, event_dt), "legacy_t0_event_utc_day"

    if tp in PREV_INDEX:
        if t0_product is None:
            return None, "missing_actual_t0_anchor"
        t0_dt = t0_product["acq_time"].astimezone(timezone.utc)
        products = fetch_products(legacy, poly, t0_dt - timedelta(days=args.prev_search_back_days), t0_dt)
        return select_prev(products, t0_product, PREV_INDEX[tp]), "prev_distinct_overpass_before_actual_t0"

    if tp == "seasonal":
        target = event_dt - timedelta(days=90)
        products = fetch_products(legacy, poly, target - timedelta(days=args.offset_search_window_days), target)
        return legacy.select_closest_product(products, target), "legacy_minus90_window_before_target"

    if tp == "year":
        target = event_dt - timedelta(days=args.year_offset_days)
        products = fetch_products(legacy, poly, target - timedelta(days=args.offset_search_window_days), target)
        return legacy.select_closest_product(products, target), "legacy_minus360_window_before_target"

    return None, f"unknown_timepoint:{tp}"


def extract_nc_from_product(path: Path, out_dir: Path) -> Optional[Path]:
    if zipfile.is_zipfile(path):
        extract_dir = out_dir / f"{path.name}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not members:
                return None
            zf.extract(members[0], extract_dir)
            return extract_dir / members[0]
    if path.suffix.lower() == ".nc":
        return path
    return None


def download_product_to_old_raw_dir(
    legacy: Any,
    token: Any,
    raw_dir: Path,
    product: dict[str, Any],
    overwrite: bool,
) -> tuple[Path, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    product_id = str(product["Id"])
    product_name = str(product["Name"])

    if not overwrite:
        found = existing_nc(raw_dir, product_name)
        if found is not None:
            return found, "existing"

    output_path = raw_dir / product_name
    tmp_path = output_path.with_name(output_path.name + ".part")
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token.get()}"}

    session = legacy.requests.Session()
    session.headers.update(headers)
    response = legacy.request_with_backoff(
        lambda proxy: session.get(
            url,
            headers=headers,
            stream=True,
            proxies=legacy.build_proxy_dict(proxy),
        ),
        description=f"download {product_name}",
    )
    try:
        if response.status_code != 200:
            raise RuntimeError(f"request failed {response.status_code}: {response.text[:300]}")
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        os.replace(tmp_path, output_path)
    finally:
        response.close()
        session.close()

    nc = extract_nc_from_product(output_path, raw_dir)
    if nc is None:
        raise RuntimeError(f"downloaded product has no .nc: {output_path}")
    ok, msg = validate_nc(nc)
    if not ok:
        raise RuntimeError(msg)
    return nc, "downloaded"


def acq_time(product: dict[str, Any]) -> Optional[datetime]:
    dt = product.get("acq_time")
    if dt is None:
        return None
    return dt.astimezone(timezone.utc)


def distinct_latest_overpasses(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pass: dict[str, dict[str, Any]] = {}
    for product in products:
        dt = acq_time(product)
        if dt is None:
            continue
        key = s5p_overpass_key(product)
        old = by_pass.get(key)
        if old is None or dt > acq_time(old):
            by_pass[key] = product
    return list(by_pass.values())


def candidate_limit(args: argparse.Namespace, tp: str) -> int:
    if tp == "t0":
        return int(args.max_candidates_t0)
    if tp in PREV_INDEX:
        return int(args.max_candidates_prev)
    return int(args.max_candidates_offset)


def candidate_products(
    row: pd.Series,
    tp: str,
    args: argparse.Namespace,
    legacy: Any,
    t0_product: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], str, Optional[datetime]]:
    event_dt = parse_dt(row.get("event_time"))
    if event_dt is None:
        return [], "invalid_time", None
    lat = row.get("plume_latitude")
    lon = row.get("plume_longitude")
    if pd.isna(lat) or pd.isna(lon):
        return [], "missing_latlon", None

    poly = build_old_poly(float(lon), float(lat), args.geo_half_deg)
    if tp == "t0":
        day = event_dt.date()
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        products = fetch_products(legacy, poly, start, end)
        ordered = sorted(
            [p for p in products if acq_time(p) is not None],
            key=lambda p: abs((acq_time(p) - event_dt).total_seconds()),
        )
        return ordered[: candidate_limit(args, tp)], "crop_qc_t0_event_utc_day", event_dt

    if tp in PREV_INDEX:
        if t0_product is None or acq_time(t0_product) is None:
            return [], "missing_actual_t0_anchor", None
        anchor = acq_time(t0_product)
        products = fetch_products(legacy, poly, anchor - timedelta(days=args.prev_search_back_days), anchor)
        t0_key = s5p_overpass_key(t0_product)
        t0_id = str(t0_product.get("Id", "")).strip()
        before = [
            p
            for p in products
            if acq_time(p) is not None
            and acq_time(p) < anchor
            and str(p.get("Id", "")).strip() != t0_id
            and s5p_overpass_key(p) != t0_key
        ]
        ordered = sorted(distinct_latest_overpasses(before), key=lambda p: acq_time(p), reverse=True)
        source = f"crop_qc_prev_distinct_overpass_{args.prev_search_back_days}d_before_actual_t0"
        return ordered[: candidate_limit(args, tp)], source, anchor

    if tp == "seasonal":
        target = event_dt - timedelta(days=90)
        products = fetch_products(legacy, poly, target - timedelta(days=args.offset_search_window_days), target)
        ordered = sorted(
            [p for p in products if acq_time(p) is not None],
            key=lambda p: abs((acq_time(p) - target).total_seconds()),
        )
        return ordered[: candidate_limit(args, tp)], "crop_qc_minus90_window_before_target", target

    if tp == "year":
        target = event_dt - timedelta(days=args.year_offset_days)
        products = fetch_products(legacy, poly, target - timedelta(days=args.offset_search_window_days), target)
        ordered = sorted(
            [p for p in products if acq_time(p) is not None],
            key=lambda p: abs((acq_time(p) - target).total_seconds()),
        )
        return ordered[: candidate_limit(args, tp)], "crop_qc_minus360_window_before_target", target

    return [], f"unknown_timepoint:{tp}", None


def blank_record(row: pd.Series, raw_dir: Path) -> dict[str, Any]:
    return {
        "plume_id": str(row["plume_id"]).strip(),
        "timepoint": str(row["timepoint"]).strip(),
        "status": "",
        "product_id": "",
        "product_name": "",
        "image_time": "",
        "selection_source": "",
        "raw_path": "",
        "raw_data_dir": str(raw_dir),
        "target_raw_dir_from_manifest": row.get("target_raw_dir", ""),
        "target_time": "",
        "time_delta_hours": "",
        "qc_ok": "",
        "qc_reason": "",
        "qc_center_iy": "",
        "qc_center_ix": "",
        "qc_center_distance_km": "",
        "qc_patch_missing_ratio": "",
        "qc_patch_finite_count": "",
        "qc_patch_total": "",
        "qc_ch4_var": "",
        "qc_candidate_rank": "",
        "qc_candidates_checked": "",
        "candidate_attempts": "",
        "message": "",
    }


def update_record_product(record: dict[str, Any], product: dict[str, Any], target_dt: Optional[datetime], rank: int) -> None:
    dt = acq_time(product)
    record["product_id"] = str(product.get("Id", ""))
    record["product_name"] = str(product.get("Name", ""))
    record["image_time"] = iso_z(dt) if dt is not None else ""
    record["target_time"] = iso_z(target_dt) if target_dt is not None else ""
    if dt is not None and target_dt is not None:
        record["time_delta_hours"] = f"{abs((dt - target_dt).total_seconds()) / 3600.0:.3f}"
    record["qc_candidate_rank"] = rank


def update_record_qc(record: dict[str, Any], qc: dict[str, Any]) -> None:
    record["qc_ok"] = bool(qc.get("ok", False))
    record["qc_reason"] = qc.get("reason", "")
    record["qc_center_iy"] = qc.get("center_iy", "")
    record["qc_center_ix"] = qc.get("center_ix", "")
    record["qc_center_distance_km"] = qc.get("center_distance_km", "")
    record["qc_patch_missing_ratio"] = qc.get("patch_missing_ratio", "")
    record["qc_patch_finite_count"] = qc.get("patch_finite_count", "")
    record["qc_patch_total"] = qc.get("patch_total", "")
    record["qc_ch4_var"] = qc.get("ch4_var", "")


def existing_product_path_from_row(row: pd.Series, product: dict[str, Any]) -> Optional[Path]:
    product_name = str(product.get("Name", ""))
    for col in ("downloaded_path", "existing_raw_path", "processed_path", "raw_path"):
        if col not in row.index or not has_value(row.get(col)):
            continue
        path = Path(str(row.get(col)).strip())
        if raw_path_matches_product(path, product_name):
            ok, _ = validate_nc(path)
            if ok:
                return path
    return None


def run_crop_qc(
    qc_executor: Optional[ProcessPoolExecutor],
    path: str,
    lat: float,
    lon: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    kwargs = {
        "crop_size": int(args.qc_crop_size),
        "max_missing_ratio": float(args.qc_max_missing_ratio),
        "max_center_distance_km": float(args.qc_max_center_distance_km),
    }
    if qc_executor is None:
        return qc_s5p_product(path, lat, lon, **kwargs)
    future = qc_executor.submit(qc_s5p_product, path, lat, lon, **kwargs)
    return future.result()


def attempt_product_crop_qc(
    row: pd.Series,
    record: dict[str, Any],
    product: dict[str, Any],
    rank: int,
    target_dt: Optional[datetime],
    args: argparse.Namespace,
    legacy: Any,
    token: Any,
    raw_dir: Path,
    qc_executor: Optional[ProcessPoolExecutor] = None,
) -> tuple[bool, Optional[dict[str, Any]], str]:
    update_record_product(record, product, target_dt, rank)
    try:
        existing_from_row = None if args.overwrite else existing_product_path_from_row(row, product)
        if existing_from_row is not None:
            nc, source = existing_from_row, "existing"
        else:
            with legacy.get_product_lock(str(product["Name"])):
                nc, source = download_product_to_old_raw_dir(legacy, token, raw_dir, product, args.overwrite)
        record["raw_path"] = str(nc)
    except Exception as exc:
        reason = f"download_fail:{type(exc).__name__}:{str(exc)[:120]}"
        record["qc_reason"] = reason
        return False, None, f"{rank}:{Path(str(product.get('Name', ''))).name}:{reason}"

    qc = run_crop_qc(
        qc_executor,
        record["raw_path"],
        float(row["plume_latitude"]),
        float(row["plume_longitude"]),
        args,
    )
    update_record_qc(record, qc)
    reason = str(qc.get("reason", ""))
    if qc.get("ok"):
        record["status"] = "skip_existing_crop_ok" if source == "existing" else "downloaded_crop_ok"
        record["message"] = "crop_ok"
        return True, product, f"{rank}:{Path(str(product.get('Name', ''))).name}:ok"
    return False, None, f"{rank}:{Path(str(product.get('Name', ''))).name}:{reason}"


def process_row_crop_aware(
    row: pd.Series,
    args: argparse.Namespace,
    legacy: Any,
    token: Any,
    raw_dir: Path,
    t0_product: Optional[dict[str, Any]] = None,
    qc_executor: Optional[ProcessPoolExecutor] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    tp = str(row["timepoint"]).strip()
    record = blank_record(row, raw_dir)
    products, source, target_dt = candidate_products(row, tp, args, legacy, t0_product)
    record["selection_source"] = source
    record["target_time"] = iso_z(target_dt) if target_dt is not None else ""
    if not products:
        record["status"] = "no_product"
        record["message"] = source
        return record, None

    attempts: list[str] = []
    last_record = record
    for rank, product in enumerate(products, start=1):
        candidate_record = dict(record)
        ok, selected, attempt = attempt_product_crop_qc(
            row, candidate_record, product, rank, target_dt, args, legacy, token, raw_dir, qc_executor
        )
        attempts.append(attempt)
        candidate_record["qc_candidates_checked"] = len(attempts)
        candidate_record["candidate_attempts"] = ";".join(attempts[-10:])
        last_record = candidate_record
        if ok:
            return candidate_record, selected

    last_record["status"] = "no_crop_valid_product"
    last_record["message"] = f"checked={len(attempts)}; last_reason={last_record.get('qc_reason', '')}"
    return last_record, None


def process_prev_rows_crop_aware(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    legacy: Any,
    token: Any,
    raw_dir: Path,
    t0_product: dict[str, Any],
    qc_executor: Optional[ProcessPoolExecutor] = None,
) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    pending = []
    for _, row in rows.iterrows():
        tp = str(row["timepoint"]).strip()
        if tp in PREV_INDEX:
            pending.append((PREV_INDEX[tp], row))
    pending.sort(key=lambda item: item[0])
    if not pending:
        return []

    first_row = pending[0][1]
    products, source, target_dt = candidate_products(first_row, "prev1", args, legacy, t0_product)
    selected_keys = {s5p_overpass_key(t0_product)}
    attempts: list[str] = []
    records: list[dict[str, Any]] = []
    pending_index = 0

    for rank, product in enumerate(products, start=1):
        if pending_index >= len(pending):
            break
        key = s5p_overpass_key(product)
        if key in selected_keys:
            continue
        _, row = pending[pending_index]
        record = blank_record(row, raw_dir)
        record["selection_source"] = source
        record["target_time"] = iso_z(target_dt) if target_dt is not None else ""
        ok, selected, attempt = attempt_product_crop_qc(
            row, record, product, rank, target_dt, args, legacy, token, raw_dir, qc_executor
        )
        attempts.append(attempt)
        record["qc_candidates_checked"] = len(attempts)
        record["candidate_attempts"] = ";".join(attempts[-10:])
        if ok and selected is not None:
            selected_keys.add(key)
            records.append(record)
            pending_index += 1

    for _, row in pending[pending_index:]:
        record = blank_record(row, raw_dir)
        record["selection_source"] = source
        record["target_time"] = iso_z(target_dt) if target_dt is not None else ""
        record["status"] = "no_crop_valid_product"
        record["qc_candidates_checked"] = len(attempts)
        record["candidate_attempts"] = ";".join(attempts[-10:])
        record["message"] = f"needed_prev_index={PREV_INDEX[str(row['timepoint']).strip()]}; checked={len(attempts)}"
        records.append(record)
    return records


def process_row_legacy(
    row: pd.Series,
    args: argparse.Namespace,
    legacy: Any,
    token: Any,
    completed: set[tuple[str, str]],
    raw_dir: Path,
    t0_product: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    plume_id = str(row["plume_id"]).strip()
    tp = str(row["timepoint"]).strip()
    record = blank_record(row, raw_dir)

    if args.resume and (plume_id, tp) in completed and not args.overwrite:
        record["status"] = "resume_skip_completed"
        return record

    product, source = select_product(row, tp, args, legacy, t0_product)
    record["selection_source"] = source
    if product is None:
        record["status"] = "no_product"
        return record

    update_record_product(record, product, None, 1)
    with legacy.get_product_lock(str(product["Name"])):
        nc, source_type = download_product_to_old_raw_dir(legacy, token, raw_dir, product, args.overwrite)
    record["raw_path"] = str(nc)
    ok, msg = validate_nc(nc)
    record["message"] = msg
    record["status"] = "skip_existing" if source_type == "existing" else ("downloaded" if ok else "invalid_nc")
    return record


def record_is_success(record: dict[str, Any]) -> bool:
    return str(record.get("status", "")).strip() in SUCCESS_STATUSES


def skip_t0_failed_record(row: pd.Series, raw_dir: Path, message: str) -> dict[str, Any]:
    record = blank_record(row, raw_dir)
    record["status"] = "skip_t0_failed"
    record["message"] = message
    return record


def process_plume_group(
    group: pd.DataFrame,
    args: argparse.Namespace,
    legacy: Any,
    token: Any,
    completed: set[tuple[str, str]],
    completed_records: dict[tuple[str, str], dict[str, Any]],
    raw_dir: Path,
    qc_executor: Optional[ProcessPoolExecutor] = None,
) -> list[dict[str, Any]]:
    if group.empty:
        return []
    plume_id = str(group.iloc[0]["plume_id"]).strip()
    timepoints = group["timepoint"].astype(str)
    t0_rows = group[timepoints == "t0"]
    other_rows = group[timepoints != "t0"]
    records: list[dict[str, Any]] = []

    if not args.crop_aware_qc:
        t0_ok = (plume_id, "t0") in completed
        t0_product = product_from_completed(completed_records.get((plume_id, "t0")))
        t0_message = "t0 is not completed in the current output manifest"
        for _, row in t0_rows.iterrows():
            rec = process_row_legacy(row, args, legacy, token, completed, raw_dir)
            records.append(rec)
            if record_is_success(rec):
                t0_ok = True
                t0_message = ""
                t0_product = product_from_completed(rec)
            else:
                t0_message = f"t0 status={rec.get('status', '')}; message={rec.get('message', '') or rec.get('selection_source', '')}"
        if t0_ok:
            for _, row in other_rows.iterrows():
                records.append(process_row_legacy(row, args, legacy, token, completed, raw_dir, t0_product))
        else:
            for _, row in other_rows.iterrows():
                records.append(skip_t0_failed_record(row, raw_dir, t0_message))
        return records

    t0_ok = False
    t0_product = None
    t0_message = "t0 is not completed in the current crop-aware selection"
    for _, row in t0_rows.iterrows():
        rec, selected = process_row_crop_aware(row, args, legacy, token, raw_dir, qc_executor=qc_executor)
        records.append(rec)
        if record_is_success(rec) and selected is not None:
            t0_ok = True
            t0_product = selected
            t0_message = ""
        else:
            t0_message = f"t0 status={rec.get('status', '')}; message={rec.get('message', '') or rec.get('selection_source', '')}"

    if not t0_ok or t0_product is None:
        for _, row in other_rows.iterrows():
            records.append(skip_t0_failed_record(row, raw_dir, t0_message))
        return records

    prev_rows = other_rows[other_rows["timepoint"].astype(str).isin(PREV_INDEX.keys())]
    records.extend(process_prev_rows_crop_aware(prev_rows, args, legacy, token, raw_dir, t0_product, qc_executor))
    offset_rows = other_rows[~other_rows["timepoint"].astype(str).isin(PREV_INDEX.keys())]
    for _, row in offset_rows.iterrows():
        rec, _ = process_row_crop_aware(row, args, legacy, token, raw_dir, t0_product, qc_executor)
        records.append(rec)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing S5P six-time products using the existing legacy S5P CDSE download code."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", default="Upgrade_data_pipeline/csv/s5p_download_manifest.csv")
    parser.add_argument("--raw-root", default="/mnt/engg-niulab/yuyao/sensors_raw_data")
    parser.add_argument("--raw-data-dir", default="")
    parser.add_argument("--legacy-config", default="data_downloading/config/cm_s5p_90360_config.yaml")
    parser.add_argument("--timepoints", default="t0,prev1,prev2,prev3,seasonal,year")
    parser.add_argument("--workers", type=int, default=4, help="Thread workers for catalogue queries/downloads.")
    parser.add_argument("--qc-workers", type=int, default=4, help="Spawned process workers for NetCDF crop-aware QC. Use 1 to run QC inline.")
    parser.add_argument("--qc-mp-context", choices=["spawn", "forkserver", "fork"], default="spawn")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--geo-half-deg", type=float, default=0.01)
    parser.add_argument("--prev-search-back-days", type=int, default=30)
    parser.add_argument("--offset-search-window-days", type=int, default=50)
    parser.add_argument("--year-offset-days", type=int, default=360)
    parser.add_argument("--max-candidates-t0", type=int, default=8)
    parser.add_argument("--max-candidates-prev", type=int, default=60)
    parser.add_argument("--max-candidates-offset", type=int, default=60)
    parser.add_argument("--qc-crop-size", type=int, default=3)
    parser.add_argument("--qc-max-missing-ratio", type=float, default=0.50)
    parser.add_argument("--qc-max-center-distance-km", type=float, default=25.0)
    parser.add_argument("--crop-aware-qc", dest="crop_aware_qc", action="store_true", default=True)
    parser.add_argument("--no-crop-aware-qc", dest="crop_aware_qc", action="store_false")
    parser.add_argument("--recheck-crop-qc", action="store_true", help="Revisit existing S5P paths and replace locally invalid crop candidates.")
    parser.add_argument("--keep-existing-csv", default="", help="Existing correct crop CSV whose plume_id values should be kept and skipped.")
    parser.add_argument("--skip-kept-plumes", dest="skip_kept_plumes", action="store_true", default=True)
    parser.add_argument("--no-skip-kept-plumes", dest="skip_kept_plumes", action="store_false")
    parser.add_argument("--cdse-username", default="")
    parser.add_argument("--cdse-password", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-master-update", action="store_true")
    args = parser.parse_args()
    if args.qc_crop_size % 2 != 1:
        raise SystemExit("--qc-crop-size must be odd")
    return args


def record_to_master_update(record: dict[str, Any]) -> dict[str, Any]:
    status = str(record.get("status", "")).strip()
    raw_path = Path(str(record.get("raw_path", "")))
    product_name = record.get("product_name", "")
    update = {
        "download_status": status,
        "selection_source": record.get("selection_source", ""),
        "status_message": record.get("message", ""),
    }
    if raw_path.exists() and raw_path.stat().st_size > 0 and raw_path_matches_product(raw_path, product_name):
        update["downloaded_path"] = str(raw_path)
    if has_value(record.get("image_time")):
        update["image_time"] = record.get("image_time", "")
    if has_value(record.get("product_id")):
        update["product_id"] = record.get("product_id", "")
    if has_value(product_name):
        update["product_name"] = product_name
        update["overpass_key"] = s5p_overpass_key(
            {
                "Name": product_name,
                "acq_time": parse_dt(record.get("image_time")),
            }
        )
    for field in [
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
    ]:
        if has_value(record.get(field)):
            update[field] = record.get(field, "")
    return update


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_data_dir) if args.raw_data_dir else Path(args.raw_root) / "S5P" / "raw_data_dir_s5p"

    legacy, tokens = init_legacy_runtime(args)
    ensure_manifest_columns(args.manifest)
    rows = load_work_rows(args)
    completed_records = {
        key: master_record_to_s5p(row)
        for key, row in load_master_completed_records(args.manifest, "S5P").items()
    }
    completed = set(completed_records.keys())
    log(f"loaded S5P download rows: {len(rows)}")
    log(f"raw_data_dir: {raw_dir}")
    log(f"loaded completed S5P rows from master manifest: {len(completed)}")

    groups = [group.copy() for _, group in rows.groupby("plume_id", sort=False)]
    log(f"loaded S5P plume groups: {len(groups)}")
    completed_rows = 0
    all_records: list[dict[str, Any]] = []
    qc_executor: Optional[ProcessPoolExecutor] = None
    if args.crop_aware_qc and int(args.qc_workers) > 1:
        ctx = mp.get_context(args.qc_mp_context)
        qc_executor = ProcessPoolExecutor(max_workers=int(args.qc_workers), mp_context=ctx)
        log(f"crop-aware QC process workers: {args.qc_workers} ({args.qc_mp_context})")
    else:
        log("crop-aware QC process workers: inline")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for i, group in enumerate(groups):
                token = tokens[i % len(tokens)]
                futures.append(
                    executor.submit(
                        process_plume_group,
                        group,
                        args,
                        legacy,
                        token,
                        completed,
                        completed_records,
                        raw_dir,
                        qc_executor,
                    )
                )
            for future in as_completed(futures):
                try:
                    records = future.result()
                except Exception as exc:
                    record = {
                        "plume_id": "",
                        "timepoint": "",
                        "status": "error",
                        "product_id": "",
                        "product_name": "",
                        "image_time": "",
                        "selection_source": "",
                        "raw_path": "",
                        "raw_data_dir": str(raw_dir),
                        "target_raw_dir_from_manifest": "",
                        "target_time": "",
                        "time_delta_hours": "",
                        "qc_ok": "",
                        "qc_reason": "",
                        "qc_center_iy": "",
                        "qc_center_ix": "",
                        "qc_center_distance_km": "",
                        "qc_patch_missing_ratio": "",
                        "qc_patch_finite_count": "",
                        "qc_patch_total": "",
                        "qc_ch4_var": "",
                        "qc_candidate_rank": "",
                        "qc_candidates_checked": "",
                        "candidate_attempts": "",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                    records = [record]
                for record in records:
                    all_records.append(record)
                    completed_rows += 1
                    append_rows(Path(args.out_csv), [record])
                    log(
                        f"[{completed_rows}/{len(rows)}] {record['status']} "
                        f"{record.get('plume_id', '')} {record.get('timepoint', '')} "
                        f"{record.get('product_name', '')}"
                    )
    finally:
        if qc_executor is not None:
            qc_executor.shutdown(wait=True)
    if not args.no_master_update:
        changed = update_master_from_records(
            args.manifest,
            "S5P",
            all_records,
            record_to_master_update,
            source_log=args.out_csv,
        )
        log(f"updated master manifest rows: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
