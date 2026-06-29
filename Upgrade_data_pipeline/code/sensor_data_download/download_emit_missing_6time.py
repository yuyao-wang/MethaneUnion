#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr

from manifest_state import (
    ensure_manifest_columns,
    load_master_completed_records,
    select_rows_for_missing_download,
    update_master_from_records,
)


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
PREV_INDEX = {"prev1": 1, "prev2": 2, "prev3": 3}
SUCCESS_STATUSES = {"downloaded", "linked_existing", "skip_existing", "master_completed", "resume_skip_completed"}
OUT_FIELDS = [
    "plume_id",
    "timepoint",
    "status",
    "granule_id",
    "raw_path",
    "source_nc",
    "selection_source",
    "message",
]
SEARCH_CACHE_FIELDS = [
    "cache_key",
    "status",
    "granule_id",
    "granule_time",
    "search_type",
    "query_start",
    "query_end",
    "target_time",
    "plume_group",
    "plume_id",
    "timepoint",
    "lat",
    "lon",
    "message",
    "updated_at_utc",
]

granule_locks: dict[str, Lock] = {}
granule_locks_lock = Lock()
local_io_semaphore: Optional[BoundedSemaphore] = None


def log(message: str) -> None:
    print(message, flush=True)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def parse_time(value: Any) -> Optional[pd.Timestamp]:
    if not has_value(value):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def iso_time(value: Optional[pd.Timestamp]) -> str:
    if value is None or pd.isna(value):
        return ""
    return value.isoformat()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_get(obj: Any, keys: list[str]) -> Any:
    cur = obj
    for key in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            try:
                cur = cur[key]
            except Exception:
                cur = getattr(cur, key, None)
    return cur


def get_granule_id(granule: Any) -> Optional[str]:
    candidates = [
        safe_get(granule, ["umm", "GranuleUR"]),
        safe_get(granule, ["meta", "native-id"]),
        safe_get(granule, ["meta", "concept-id"]),
    ]
    for value in candidates:
        if has_value(value):
            return str(value).strip()
    return None


def get_granule_time(granule: Any) -> Optional[pd.Timestamp]:
    raw = safe_get(granule, ["umm", "TemporalExtent", "RangeDateTime", "BeginningDateTime"])
    return parse_time(raw)


def emit_overpass_key(granule_id: str) -> str:
    match = re.search(r"_(\d{7})_\d{3}$", str(granule_id).strip())
    if match:
        return match.group(1)
    return str(granule_id).strip()


def plume_group_id(plume_id: str) -> str:
    text = str(plume_id).strip()
    if re.search(r"-[A-Za-z]+$", text):
        return text.rsplit("-", 1)[0]
    return text


def plume_id_to_emit_granule_pattern(plume_id: str) -> Optional[str]:
    match = re.match(r"^emi(?P<date>\d{8})t(?P<time>\d{6})p(?P<path>\d+)(?:-[A-Za-z]+)?$", str(plume_id).strip(), re.I)
    if not match:
        return None
    scene = match.group("path")[-3:]
    return f"EMIT_L2A_RFL_001_{match.group('date')}T{match.group('time')}_*_{scene}"


def select_emit_ch4_32(wavelengths_nm: np.ndarray, n_low: int = 5, n_high: int = 6) -> list[int]:
    wavelengths_nm = np.asarray(wavelengths_nm)
    full_min, full_max = 2137, 2493
    core_min, core_max = 2250, 2410

    low = np.where((wavelengths_nm > full_min) & (wavelengths_nm < core_min))[0]
    core = np.where((wavelengths_nm >= core_min) & (wavelengths_nm <= core_max))[0]
    high = np.where((wavelengths_nm > core_max) & (wavelengths_nm <= full_max))[0]
    if len(low) == 0 or len(core) == 0 or len(high) == 0:
        raise RuntimeError("EMIT wavelengths do not cover expected CH4 SWIR range")

    low_sel = np.round(np.linspace(low[0], low[-1], n_low)).astype(int)
    high_sel = np.round(np.linspace(high[0], high[-1], n_high)).astype(int)
    return np.unique(np.concatenate([low_sel, core, high_sel])).astype(int).tolist()


def validate_npz(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, "missing_or_empty"
    try:
        with np.load(path) as data:
            required = {"reflectance_ch4", "lat", "lon", "wavelengths_nm", "band_indices"}
            missing = required - set(data.files)
            if missing:
                return False, f"missing_keys:{sorted(missing)}"
            arr = data["reflectance_ch4"]
            if arr.ndim != 3 or arr.shape[-1] == 0:
                return False, f"bad_reflectance_shape:{arr.shape}"
        return True, "npz_ok"
    except Exception as exc:
        return False, f"npz_error:{exc}"


def append_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_completed_records(out_csv: str) -> dict[tuple[str, str], dict[str, Any]]:
    p = Path(out_csv)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, low_memory=False)
    except pd.errors.EmptyDataError:
        return {}
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
        ok, _ = validate_npz(Path(str(raw_path)))
        if ok:
            done[(str(row["plume_id"]).strip(), str(row["timepoint"]).strip())] = row.to_dict()
    return done


def load_completed(out_csv: str) -> set[tuple[str, str]]:
    return set(load_completed_records(out_csv).keys())


def append_unique_usage_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_cols = ["plume_id", "timepoint", "raw_path"]
    if path.exists():
        try:
            existing = pd.read_csv(path, usecols=key_cols)
            key = tuple(str(row.get(c, "")).strip() for c in key_cols)
            for _, old in existing.iterrows():
                old_key = tuple(str(old.get(c, "")).strip() for c in key_cols)
                if old_key == key:
                    return
        except Exception:
            pass
    append_rows(path, [row], list(row.keys()))


def append_granule_usage(
    granule_cache: Path,
    row: pd.Series,
    tp: str,
    granule_id: str,
    selection_source: str,
    target_npz: Path,
) -> None:
    usage = {
        "plume_id": str(row.get("plume_id", "")).strip(),
        "timepoint": tp,
        "granule_id": granule_id,
        "selection_source": selection_source,
        "event_time": row.get("event_time", ""),
        "t0_available_time": row.get("t0_available_time", ""),
        "plume_latitude": row.get("plume_latitude", ""),
        "plume_longitude": row.get("plume_longitude", ""),
        "target_raw_dir": row.get("target_raw_dir", ""),
        "raw_path": str(target_npz),
        "logged_at_utc": utc_now_iso(),
    }
    append_unique_usage_row(granule_cache / "used_by.csv", usage)


class SearchCache:
    def __init__(self, path: Path, ignore_existing: bool = False) -> None:
        self.path = path
        self.lock = Lock()
        self.key_locks: dict[str, Lock] = {}
        self.key_locks_lock = Lock()
        self.rows: dict[str, dict[str, str]] = {}
        if ignore_existing or not path.exists():
            return
        try:
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    key = str(row.get("cache_key", "")).strip()
                    if key:
                        self.rows[key] = dict(row)
        except Exception:
            self.rows = {}

    def key_lock(self, key: str) -> Lock:
        with self.key_locks_lock:
            lock = self.key_locks.get(key)
            if lock is None:
                lock = Lock()
                self.key_locks[key] = lock
            return lock

    def get(self, key: str) -> Optional[dict[str, str]]:
        with self.lock:
            row = self.rows.get(key)
            return dict(row) if row else None

    def put(self, row: dict[str, Any]) -> dict[str, str]:
        row = {field: str(row.get(field, "")) for field in SEARCH_CACHE_FIELDS}
        row["updated_at_utc"] = row.get("updated_at_utc") or utc_now_iso()
        key = row["cache_key"]
        with self.lock:
            old = self.rows.get(key)
            if old is not None:
                return dict(old)
            append_rows(self.path, [row], SEARCH_CACHE_FIELDS)
            self.rows[key] = row
            return dict(row)


class GranuleObjectCache:
    def __init__(self, short_name: str, retries: int) -> None:
        self.short_name = short_name
        self.retries = retries
        self.lock = Lock()
        self.key_locks: dict[str, Lock] = {}
        self.key_locks_lock = Lock()
        self.objects: dict[str, Any] = {}

    def key_lock(self, granule_id: str) -> Lock:
        with self.key_locks_lock:
            lock = self.key_locks.get(granule_id)
            if lock is None:
                lock = Lock()
                self.key_locks[granule_id] = lock
            return lock

    def get(self, granule_id: str) -> Optional[Any]:
        with self.lock:
            obj = self.objects.get(granule_id)
            if obj is not None:
                return obj
        with self.key_lock(granule_id):
            with self.lock:
                obj = self.objects.get(granule_id)
                if obj is not None:
                    return obj
            obj = earthaccess_search_by_name(granule_id, self.short_name, self.retries, count=1)
            if obj is None:
                return None
            with self.lock:
                self.objects[granule_id] = obj
            return obj

    def remember(self, granule: Any, granule_id: str) -> None:
        with self.lock:
            self.objects[granule_id] = granule


def load_emit_legacy(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p, low_memory=False)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        pid = str(row.get("plume_id", "")).strip()
        if pid:
            out[pid] = row.to_dict()
    return out


def load_work_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.manifest, low_memory=False)
    tps = {tp.strip() for tp in args.timepoints.split(",") if tp.strip()}
    bad = sorted(tps - set(TIMEPOINTS))
    if bad:
        raise ValueError(f"unknown timepoints: {bad}")
    mask = (
        (df["sensor"].astype(str).str.upper() == "EMIT")
        & (df["action"].astype(str) == "download")
    )
    candidates = df[mask].copy()
    rows = candidates.to_dict("records")
    selected = select_rows_for_missing_download(rows, tps, overwrite=args.overwrite)
    out = pd.DataFrame(selected, columns=candidates.columns)
    if args.limit:
        out = out.head(args.limit)
    return out


def legacy_granule_id(row: pd.Series, tp: str, legacy: dict[str, dict[str, Any]]) -> Optional[str]:
    pid = str(row["plume_id"]).strip()
    item = legacy.get(pid)
    if not item:
        return None
    col = None
    if tp == "t0":
        col = "emit_granule_id"
    elif tp == "seasonal":
        col = "emit_-90_granule_id"
    elif tp == "year":
        col = "emit_-180_granule_id"
    if col and has_value(item.get(col)):
        return str(item[col]).strip()
    return None


def earthaccess_search_by_name(granule_name: str, short_name: str, retries: int, count: int = 1) -> Optional[Any]:
    for attempt in range(1, retries + 1):
        try:
            results = earthaccess.search_data(short_name=short_name, granule_name=granule_name, count=count)
            return results[0] if results else None
        except Exception as exc:
            if attempt == retries:
                raise
            wait = min(2 ** (attempt - 1), 16)
            log(f"search_by_name failed {granule_name} attempt {attempt}/{retries}: {exc}; sleep {wait}s")
            time.sleep(wait)
    return None


def earthaccess_search_point(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    short_name: str,
    count: int,
    retries: int,
) -> list[Any]:
    for attempt in range(1, retries + 1):
        try:
            return earthaccess.search_data(
                short_name=short_name,
                point=(float(lon), float(lat)),
                temporal=(start.isoformat(), end.isoformat()),
                count=count,
            )
        except Exception as exc:
            if attempt == retries:
                raise
            wait = min(2 ** (attempt - 1), 16)
            log(f"point search failed attempt {attempt}/{retries}: {exc}; sleep {wait}s")
            time.sleep(wait)
    return []


def select_from_results(results: list[Any], target: pd.Timestamp) -> Optional[Any]:
    candidates = []
    for g in results:
        gid = get_granule_id(g)
        gt = get_granule_time(g)
        if gid and gt is not None and not pd.isna(gt):
            candidates.append((gid, gt, g))
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs((x[1] - target).total_seconds()))[2]


def select_prev_from_results(results: list[Any], t0_granule: Any, n: int) -> Optional[Any]:
    anchor = get_granule_time(t0_granule)
    t0_gid = get_granule_id(t0_granule) or ""
    if anchor is None or pd.isna(anchor):
        return None
    t0_key = emit_overpass_key(t0_gid)
    by_pass: dict[str, tuple[pd.Timestamp, Any]] = {}
    for g in results:
        gid = get_granule_id(g)
        gt = get_granule_time(g)
        if not gid or gt is None or pd.isna(gt) or gt >= anchor:
            continue
        if gid == t0_gid or emit_overpass_key(gid) == t0_key:
            continue
        key = emit_overpass_key(gid)
        old = by_pass.get(key)
        if old is None or gt > old[0]:
            by_pass[key] = (gt, g)
    ordered = sorted(by_pass.values(), key=lambda x: x[0], reverse=True)
    if len(ordered) < n:
        return None
    return ordered[n - 1][1]


def cache_direct_id_search(
    row: pd.Series,
    tp: str,
    granule_id: str,
    args: argparse.Namespace,
    cache: SearchCache,
    objects: GranuleObjectCache,
    source: str,
) -> tuple[Optional[Any], Optional[str], str]:
    key = f"id|{args.short_name}|{granule_id}"
    with cache.key_lock(key):
        cached = cache.get(key)
        if cached is not None:
            if cached.get("status") != "found" or not cached.get("granule_id"):
                return None, None, cached.get("search_type", source)
            obj = objects.get(cached["granule_id"])
            return obj, cached["granule_id"], cached.get("search_type", source)

        obj = earthaccess_search_by_name(granule_id, args.short_name, args.search_retries, count=1)
        found_id = get_granule_id(obj) if obj is not None else ""
        if obj is not None and found_id:
            objects.remember(obj, found_id)
        cache.put(
            {
                "cache_key": key,
                "status": "found" if found_id else "no_granule",
                "granule_id": found_id,
                "granule_time": iso_time(get_granule_time(obj)) if obj is not None else "",
                "search_type": source,
                "query_start": "",
                "query_end": "",
                "target_time": "",
                "plume_group": plume_group_id(str(row.get("plume_id", ""))),
                "plume_id": str(row.get("plume_id", "")).strip(),
                "timepoint": tp,
                "lat": row.get("plume_latitude", ""),
                "lon": row.get("plume_longitude", ""),
                "message": "" if found_id else "direct_granule_id_not_found",
                "updated_at_utc": utc_now_iso(),
            }
        )
        return obj, found_id or None, source


def cache_plume_id_t0_search(
    row: pd.Series,
    args: argparse.Namespace,
    cache: SearchCache,
    objects: GranuleObjectCache,
) -> tuple[Optional[Any], Optional[str], str]:
    plume_id = str(row.get("plume_id", "")).strip()
    pattern = plume_id_to_emit_granule_pattern(plume_id)
    if pattern is None:
        return None, None, "plume_id_parse_failed"
    key = f"plume_t0|{args.short_name}|{pattern}"
    event_time = parse_time(row.get("event_time"))
    with cache.key_lock(key):
        cached = cache.get(key)
        if cached is not None:
            if cached.get("status") != "found" or not cached.get("granule_id"):
                return None, None, cached.get("search_type", "plume_id_granule_pattern")
            obj = objects.get(cached["granule_id"])
            return obj, cached["granule_id"], cached.get("search_type", "plume_id_granule_pattern")

        results = []
        try:
            results = earthaccess.search_data(short_name=args.short_name, granule_name=pattern, count=args.search_count)
        except Exception as exc:
            raise RuntimeError(f"plume_id pattern search failed for {pattern}: {exc}") from exc
        obj = select_from_results(results, event_time) if event_time is not None else (results[0] if results else None)
        found_id = get_granule_id(obj) if obj is not None else ""
        if obj is not None and found_id:
            objects.remember(obj, found_id)
        cache.put(
            {
                "cache_key": key,
                "status": "found" if found_id else "no_granule",
                "granule_id": found_id,
                "granule_time": iso_time(get_granule_time(obj)) if obj is not None else "",
                "search_type": "plume_id_granule_pattern",
                "query_start": "",
                "query_end": "",
                "target_time": iso_time(event_time),
                "plume_group": plume_group_id(plume_id),
                "plume_id": plume_id,
                "timepoint": "t0",
                "lat": row.get("plume_latitude", ""),
                "lon": row.get("plume_longitude", ""),
                "message": "" if found_id else f"no_result_for_pattern:{pattern}",
                "updated_at_utc": utc_now_iso(),
            }
        )
        return obj, found_id or None, "plume_id_granule_pattern"


def point_search_window(row: pd.Series, tp: str, args: argparse.Namespace, t0_time: Optional[pd.Timestamp] = None) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp], str]:
    event_time = parse_time(row.get("event_time"))
    if t0_time is None:
        t0_time = parse_time(row.get("t0_available_time")) or event_time
    if event_time is None or t0_time is None:
        return None, None, None, "invalid_time"
    if tp in PREV_INDEX:
        start, end = t0_time - pd.Timedelta(days=args.prev_search_back_days), t0_time
        return start, end, t0_time, f"{tp}_grouped_point_search"
    if tp == "seasonal":
        target = t0_time - pd.Timedelta(days=90)
        start = target - pd.Timedelta(days=args.offset_before_days)
        end = target + pd.Timedelta(days=args.offset_after_days)
        return start, end, target, "seasonal_grouped_point_search"
    if tp == "year":
        target = t0_time - pd.Timedelta(days=args.year_offset_days)
        start = target - pd.Timedelta(days=args.offset_before_days)
        end = target + pd.Timedelta(days=args.offset_after_days)
        return start, end, target, "year_grouped_point_search"
    return None, None, None, f"unsupported_point_search:{tp}"


def cache_grouped_point_search(
    row: pd.Series,
    tp: str,
    args: argparse.Namespace,
    cache: SearchCache,
    objects: GranuleObjectCache,
    t0_granule: Optional[Any] = None,
) -> tuple[Optional[Any], Optional[str], str]:
    t0_time = get_granule_time(t0_granule) if t0_granule is not None else None
    if tp in PREV_INDEX and t0_time is None:
        return None, None, "missing_actual_t0_anchor"
    start, end, target, source = point_search_window(row, tp, args, t0_time)
    if start is None or end is None or target is None:
        return None, None, source
    group = plume_group_id(str(row.get("plume_id", "")).strip())
    key = "|".join(
        [
            "point_v2_actual_t0_overpass",
            args.short_name,
            tp,
            group,
            iso_time(start),
            iso_time(end),
            iso_time(target),
        ]
    )
    with cache.key_lock(key):
        cached = cache.get(key)
        if cached is not None:
            if cached.get("status") != "found" or not cached.get("granule_id"):
                return None, None, cached.get("search_type", source)
            obj = objects.get(cached["granule_id"])
            return obj, cached["granule_id"], cached.get("search_type", source)

        lat = float(row["plume_latitude"])
        lon = float(row["plume_longitude"])
        results = earthaccess_search_point(lat, lon, start, end, args.short_name, args.search_count, args.search_retries)
        obj = select_prev_from_results(results, t0_granule, PREV_INDEX[tp]) if tp in PREV_INDEX else select_from_results(results, target)
        found_id = get_granule_id(obj) if obj is not None else ""
        if obj is not None and found_id:
            objects.remember(obj, found_id)
        cache.put(
            {
                "cache_key": key,
                "status": "found" if found_id else "no_granule",
                "granule_id": found_id,
                "granule_time": iso_time(get_granule_time(obj)) if obj is not None else "",
                "search_type": source,
                "query_start": iso_time(start),
                "query_end": iso_time(end),
                "target_time": iso_time(target),
                "plume_group": group,
                "plume_id": str(row.get("plume_id", "")).strip(),
                "timepoint": tp,
                "lat": lat,
                "lon": lon,
                "message": "" if found_id else f"no_result_in_{len(results)}_candidates",
                "updated_at_utc": utc_now_iso(),
            }
        )
        return obj, found_id or None, source


def find_granule(
    row: pd.Series,
    tp: str,
    args: argparse.Namespace,
    legacy: dict[str, dict[str, Any]],
    cache: SearchCache,
    objects: GranuleObjectCache,
    t0_granule: Optional[Any] = None,
) -> tuple[Optional[Any], Optional[str], str]:
    legacy_id = legacy_granule_id(row, tp, legacy)
    if legacy_id:
        return cache_direct_id_search(row, tp, legacy_id, args, cache, objects, "legacy_granule_id")

    if tp == "t0":
        return cache_plume_id_t0_search(row, args, cache, objects)

    if tp in PREV_INDEX or tp in {"seasonal", "year"}:
        return cache_grouped_point_search(row, tp, args, cache, objects, t0_granule)

    return None, None, f"unknown_timepoint:{tp}"


def cache_lock(granule_id: str) -> Lock:
    with granule_locks_lock:
        lock = granule_locks.get(granule_id)
        if lock is None:
            lock = Lock()
            granule_locks[granule_id] = lock
        return lock


def find_nc(root: Path, granule_id: str) -> Optional[Path]:
    matches = list(root.glob(f"*{granule_id}*.nc"))
    if not matches:
        matches = list(root.rglob(f"*{granule_id}*.nc"))
    return matches[0] if matches else None


def safe_path_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def scratch_dir_for(args: argparse.Namespace, granule_id: str) -> Path:
    return Path(args.scratch_root) / safe_path_name(granule_id)


def copy_file_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def move_file_cross_fs(src: Path, dst: Path) -> None:
    copy_file_atomic(src, dst)
    src.unlink()


def cleanup_scratch_dir(path: Path, scratch_root: str) -> None:
    root = Path(scratch_root).resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise RuntimeError(f"refusing to clean scratch path outside scratch root: {target}")
    shutil.rmtree(target, ignore_errors=True)


def url_basename(url: str) -> str:
    return unquote(Path(urlparse(str(url)).path).name)


def rfl_data_link(granule: Any, granule_id: str) -> Optional[str]:
    try:
        links = list(granule.data_links(access="external"))
    except TypeError:
        links = list(granule.data_links())
    except Exception:
        links = []
    exact_name = f"{granule_id}.nc"
    candidates = []
    for link in links:
        name = url_basename(str(link))
        if name == exact_name:
            return str(link)
        if name.startswith("EMIT_L2A_RFL_") and name.endswith(".nc") and "RFLUNCERT" not in name:
            candidates.append(str(link))
    return candidates[0] if candidates else None


def download_granule_nc(granule: Any, granule_id: str, cache_dir: Path, args: argparse.Namespace) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = find_nc(cache_dir, granule_id)
    if existing is not None and existing.stat().st_size > 0:
        return existing
    download_target: Any = [granule]
    if not args.download_all_links:
        rfl_url = rfl_data_link(granule, granule_id)
        if rfl_url:
            download_target = [rfl_url]
        else:
            log(f"RFL-only link not found for {granule_id}; falling back to all granule links")
    for attempt in range(1, args.download_retries + 1):
        try:
            earthaccess.download(download_target, str(cache_dir), threads=args.download_threads, show_progress=False)
            found = find_nc(cache_dir, granule_id)
            if found is None:
                raise RuntimeError(f"download completed but nc not found for {granule_id}")
            return found
        except Exception as exc:
            if attempt == args.download_retries:
                raise
            wait = min(2 ** (attempt - 1), 16)
            log(f"download failed {granule_id} attempt {attempt}/{args.download_retries}: {exc}; sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download failed for {granule_id}")


def materialize_nc_in_scratch(
    granule: Any,
    granule_id: str,
    granule_cache: Path,
    args: argparse.Namespace,
) -> tuple[Path, str]:
    scratch_dir = scratch_dir_for(args, granule_id)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    existing_scratch = find_nc(scratch_dir, granule_id)
    if existing_scratch is not None and existing_scratch.stat().st_size > 0:
        return existing_scratch, "scratch_existing"

    existing_cache = find_nc(granule_cache, granule_id)
    if existing_cache is not None and existing_cache.stat().st_size > 0:
        scratch_nc = scratch_dir / existing_cache.name
        copy_file_atomic(existing_cache, scratch_nc)
        return scratch_nc, "copied_from_raw_cache"

    return download_granule_nc(granule, granule_id, scratch_dir, args), "downloaded_to_scratch"


def store_outputs_from_scratch(
    scratch_nc: Path,
    scratch_npz: Path,
    granule_cache: Path,
    cache_npz: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    move_file_cross_fs(scratch_npz, cache_npz)
    raw_cache_nc = granule_cache / scratch_nc.name
    if args.delete_nc_after_npz:
        if raw_cache_nc.exists():
            raw_cache_nc.unlink()
        if scratch_nc.exists():
            scratch_nc.unlink()
        return None

    if raw_cache_nc.exists() and raw_cache_nc.stat().st_size > 0 and not args.overwrite:
        if scratch_nc.exists():
            scratch_nc.unlink()
        return raw_cache_nc

    move_file_cross_fs(scratch_nc, raw_cache_nc)
    return raw_cache_nc


def extract_ch4_npz(
    nc_path: Path,
    out_npz: Path,
    granule_id: str,
    source_nc_label: Optional[str] = None,
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_name(out_npz.name + ".part")
    with xr.open_dataset(nc_path, group="sensor_band_parameters", engine="netcdf4") as dsb:
        wavelengths = dsb["wavelengths"].values.astype(np.float32)
    idx = select_emit_ch4_32(wavelengths)
    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        reflectance = ds["reflectance"][:, :, idx].values.astype(np.float32)
    with xr.open_dataset(nc_path, group="location", engine="netcdf4") as loc:
        lat = loc["lat"].values.astype(np.float32)
        lon = loc["lon"].values.astype(np.float32)
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            reflectance_ch4=reflectance,
            lat=lat,
            lon=lon,
            wavelengths_nm=wavelengths[idx].astype(np.float32),
            band_indices=np.asarray(idx, dtype=np.int16),
            granule_id=np.asarray(granule_id),
            source_nc=np.asarray(source_nc_label or str(nc_path)),
        )
    os.replace(tmp, out_npz)


def link_or_copy(src: Path, dst: Path, copy_from_cache: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_from_cache:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def process_row(
    row: pd.Series,
    args: argparse.Namespace,
    legacy: dict[str, dict[str, Any]],
    completed: set[tuple[str, str]],
    search_cache: SearchCache,
    object_cache: GranuleObjectCache,
    t0_granule: Optional[Any] = None,
) -> dict[str, Any]:
    plume_id = str(row["plume_id"]).strip()
    tp = str(row["timepoint"]).strip()
    target_dir = Path(str(row["target_raw_dir"]))
    target_npz = target_dir / "emit_ch4_32.npz"
    record = {
        "plume_id": plume_id,
        "timepoint": tp,
        "status": "",
        "granule_id": "",
        "raw_path": str(target_npz),
        "source_nc": "",
        "selection_source": "",
        "image_time": "",
        "overpass_key": "",
        "message": "",
    }

    if args.resume and (plume_id, tp) in completed and not args.overwrite:
        record["status"] = "resume_skip_completed"
        return record

    granule, granule_id, source = find_granule(row, tp, args, legacy, search_cache, object_cache, t0_granule)
    record["selection_source"] = source
    if granule is None or not granule_id:
        record["status"] = "no_granule"
        return record
    record["granule_id"] = granule_id
    record["image_time"] = iso_time(get_granule_time(granule))
    record["overpass_key"] = emit_overpass_key(granule_id)

    if target_npz.exists() and not args.overwrite:
        ok, msg = validate_npz(target_npz)
        record["status"] = "skip_existing" if ok else "existing_invalid"
        record["message"] = msg
        return record

    base = Path(args.cache_dir) if args.cache_dir else Path(args.raw_root) / "EMIT" / "raw_granules"
    granule_cache = base / granule_id
    cache_npz = granule_cache / "emit_ch4_32.npz"
    with cache_lock(granule_id):
        append_granule_usage(granule_cache, row, tp, granule_id, source, target_npz)
        if not cache_npz.exists() or args.overwrite:
            scratch_dir = scratch_dir_for(args, granule_id)
            if local_io_semaphore is None:
                raise RuntimeError("local_io_semaphore is not initialized")
            with local_io_semaphore:
                try:
                    nc_path, nc_source = materialize_nc_in_scratch(granule, granule_id, granule_cache, args)
                    scratch_npz = scratch_dir / "emit_ch4_32.npz"
                    expected_nc = granule_cache / nc_path.name
                    source_nc_label = (
                        f"deleted_after_npz:{nc_path.name}"
                        if args.delete_nc_after_npz
                        else str(expected_nc)
                    )
                    extract_ch4_npz(nc_path, scratch_npz, granule_id, source_nc_label)
                    final_nc = store_outputs_from_scratch(nc_path, scratch_npz, granule_cache, cache_npz, args)
                    record["source_nc"] = str(final_nc) if final_nc is not None else f"deleted_after_npz:{nc_path.name}"
                    record["message"] = nc_source
                finally:
                    if args.cleanup_scratch:
                        cleanup_scratch_dir(scratch_dir, args.scratch_root)
        else:
            nc_path = find_nc(granule_cache, granule_id)
            record["source_nc"] = str(nc_path) if nc_path else ""

    link_or_copy(cache_npz, target_npz, args.copy_from_cache)
    ok, msg = validate_npz(target_npz)
    record["message"] = ";".join(part for part in [record.get("message", ""), msg] if part)
    record["status"] = "downloaded" if ok else "invalid_npz"
    return record


def record_is_success(record: dict[str, Any]) -> bool:
    return str(record.get("status", "")).strip() in SUCCESS_STATUSES


def skip_t0_failed_record(row: pd.Series, message: str) -> dict[str, Any]:
    target_dir = Path(str(row["target_raw_dir"]))
    return {
        "plume_id": str(row["plume_id"]).strip(),
        "timepoint": str(row["timepoint"]).strip(),
        "status": "skip_t0_failed",
        "granule_id": "",
        "raw_path": str(target_dir / "emit_ch4_32.npz"),
        "source_nc": "",
        "selection_source": "",
        "image_time": "",
        "overpass_key": "",
        "message": message,
    }


def master_record_to_emit(row: dict[str, str]) -> dict[str, Any]:
    granule_id = row.get("product_id", "") or row.get("product_name", "")
    return {
        "plume_id": row.get("plume_id", ""),
        "timepoint": row.get("timepoint", ""),
        "status": "master_completed",
        "granule_id": granule_id,
        "raw_path": row.get("downloaded_path", ""),
        "source_nc": "",
        "selection_source": row.get("selection_source", ""),
        "image_time": row.get("image_time", ""),
        "overpass_key": row.get("overpass_key", ""),
        "message": row.get("status_message", ""),
    }


def process_plume_group(
    group: pd.DataFrame,
    args: argparse.Namespace,
    legacy: dict[str, dict[str, Any]],
    completed: set[tuple[str, str]],
    completed_records: dict[tuple[str, str], dict[str, Any]],
    search_cache: SearchCache,
    object_cache: GranuleObjectCache,
) -> list[dict[str, Any]]:
    if group.empty:
        return []
    plume_id = str(group.iloc[0]["plume_id"]).strip()
    timepoints = group["timepoint"].astype(str)
    t0_rows = group[timepoints == "t0"]
    other_rows = group[timepoints != "t0"]
    records: list[dict[str, Any]] = []

    t0_ok = (plume_id, "t0") in completed
    t0_granule: Optional[Any] = None
    t0_done = completed_records.get((plume_id, "t0"))
    if t0_done and has_value(t0_done.get("granule_id")):
        t0_granule = object_cache.get(str(t0_done["granule_id"]).strip())
    t0_message = "t0 is not completed in the current output manifest"
    for _, row in t0_rows.iterrows():
        rec = process_row(row, args, legacy, completed, search_cache, object_cache)
        records.append(rec)
        if record_is_success(rec):
            t0_ok = True
            t0_message = ""
            if has_value(rec.get("granule_id")):
                t0_granule = object_cache.get(str(rec["granule_id"]).strip())
        else:
            t0_message = (
                f"t0 status={rec.get('status', '')}; "
                f"message={rec.get('message', '') or rec.get('selection_source', '')}"
            )

    if t0_ok:
        for _, row in other_rows.iterrows():
            records.append(process_row(row, args, legacy, completed, search_cache, object_cache, t0_granule))
    else:
        for _, row in other_rows.iterrows():
            records.append(skip_t0_failed_record(row, t0_message))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download missing EMIT six-time SWIR CH4 subsets from the upgrade manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--legacy-emit-csv", default="preprocess_dataset_EMIT/merged_with_emit_tag.csv")
    parser.add_argument("--out-csv", default="Upgrade_data_pipeline/csv/emit_download_manifest.csv")
    parser.add_argument("--search-cache-csv", default="Upgrade_data_pipeline/csv/emit_granule_search_cache.csv")
    parser.add_argument("--raw-root", default="/mnt/engg-niulab/yuyao/sensors_raw_data")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--scratch-root", default="/diniuvol/yuyao/emit_download_scratch")
    parser.add_argument("--timepoints", default="t0,prev1,prev2,prev3,seasonal,year")
    parser.add_argument("--short-name", default="EMITL2ARFL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-threads", type=int, default=4)
    parser.add_argument("--local-io-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--search-count", type=int, default=200)
    parser.add_argument("--search-retries", type=int, default=5)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--prev-search-back-days", type=int, default=365)
    parser.add_argument("--offset-before-days", type=int, default=180)
    parser.add_argument("--offset-after-days", type=int, default=80)
    parser.add_argument("--year-offset-days", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ignore-search-cache", action="store_true")
    parser.add_argument("--copy-from-cache", action="store_true")
    parser.add_argument("--delete-nc-after-npz", action="store_true")
    parser.add_argument("--no-cleanup-scratch", dest="cleanup_scratch", action="store_false")
    parser.set_defaults(cleanup_scratch=True)
    parser.add_argument("--no-master-update", action="store_true")
    parser.add_argument(
        "--download-all-links",
        action="store_true",
        help="Download all files attached to an EMIT granule. Default downloads only EMIT_L2A_RFL_*.nc.",
    )
    return parser.parse_args()


def record_to_master_update(record: dict[str, Any]) -> dict[str, Any]:
    status = str(record.get("status", "")).strip()
    raw_path = Path(str(record.get("raw_path", "")))
    update = {
        "download_status": status,
        "selection_source": record.get("selection_source", ""),
        "status_message": record.get("message", ""),
    }
    ok, _ = validate_npz(raw_path) if raw_path.exists() else (False, "missing")
    if ok:
        update["downloaded_path"] = str(raw_path)
        update["processed_path"] = str(raw_path)
    if has_value(record.get("granule_id")):
        update["product_id"] = record.get("granule_id", "")
        update["product_name"] = record.get("granule_id", "")
        update["overpass_key"] = record.get("overpass_key", "") or emit_overpass_key(str(record.get("granule_id", "")))
    if has_value(record.get("image_time")):
        update["image_time"] = record.get("image_time", "")
    return update


def main() -> int:
    global local_io_semaphore
    args = parse_args()
    if args.local_io_workers < 1:
        raise ValueError("--local-io-workers must be >= 1")
    Path(args.scratch_root).mkdir(parents=True, exist_ok=True)
    local_io_semaphore = BoundedSemaphore(args.local_io_workers)
    auth = earthaccess.login()
    if auth is None:
        raise RuntimeError("earthaccess.login() failed; authenticate NASA Earthdata first")
    ensure_manifest_columns(args.manifest)
    rows = load_work_rows(args)
    legacy = load_emit_legacy(args.legacy_emit_csv)
    completed_records = {
        key: master_record_to_emit(row)
        for key, row in load_master_completed_records(args.manifest, "EMIT").items()
    }
    completed = set(completed_records.keys())
    search_cache = SearchCache(Path(args.search_cache_csv), ignore_existing=args.ignore_search_cache)
    object_cache = GranuleObjectCache(args.short_name, args.search_retries)
    log(f"loaded EMIT download rows: {len(rows)}")
    log(f"loaded legacy EMIT rows: {len(legacy)}")
    log(f"loaded EMIT search cache rows: {len(search_cache.rows)}")
    log(f"loaded completed EMIT rows from master manifest: {len(completed)}")

    groups = [group.copy() for _, group in rows.groupby("plume_id", sort=False)]
    log(f"loaded EMIT plume groups: {len(groups)}")
    completed_rows = 0
    all_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_plume_group, group, args, legacy, completed, completed_records, search_cache, object_cache)
            for group in groups
        ]
        for fut in as_completed(futures):
            records = fut.result()
            for rec in records:
                all_records.append(rec)
                completed_rows += 1
                append_rows(Path(args.out_csv), [rec], OUT_FIELDS)
                log(
                    f"[{completed_rows}/{len(rows)}] {rec['status']} "
                    f"{rec['plume_id']} {rec['timepoint']} {rec.get('granule_id', '')}"
                )
    if not args.no_master_update:
        changed = update_master_from_records(
            args.manifest,
            "EMIT",
            all_records,
            record_to_master_update,
            source_log=args.out_csv,
        )
        log(f"updated master manifest rows: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
