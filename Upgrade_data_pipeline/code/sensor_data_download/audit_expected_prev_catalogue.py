#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
import pandas as pd


PREV_RANKS = {"prev1": 1, "prev2": 2, "prev3": 3}
SUCCESS_STATUSES = {
    "available",
    "available_existing_corrected",
    "downloaded",
    "linked_existing",
    "skip_existing",
    "skip_existing_raw",
    "skip_existing_512",
    "skip_existing_valid",
    "resume_skip_completed",
}

OUT_FIELDS = [
    "sensor",
    "plume_id",
    "expected_timepoint",
    "event_time",
    "plume_latitude",
    "plume_longitude",
    "t0_product_id",
    "t0_product_name",
    "t0_image_time",
    "t0_overpass_key",
    "query_start_utc",
    "query_end_utc",
    "catalogue_product_count",
    "catalogue_distinct_overpass_count",
    "expected_product_id",
    "expected_product_name",
    "expected_image_time",
    "expected_overpass_key",
    "expected_status",
    "local_status",
    "local_path",
    "local_path_source",
    "local_current_timepoint",
    "current_manifest_product_id",
    "current_manifest_product_name",
    "current_manifest_image_time",
    "current_manifest_path",
    "current_manifest_status",
    "notes",
]


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def parse_utc(value: Any) -> Optional[datetime]:
    if not has_value(value):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return ""
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = parse_utc(value)
        if dt is None:
            return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def existing_path(value: Any) -> str:
    if not has_value(value):
        return ""
    path = Path(str(value).strip())
    try:
        if path.exists() and path.stat().st_size > 0:
            return str(path)
    except OSError:
        return ""
    return ""


def path_text(value: Any) -> str:
    if not has_value(value):
        return ""
    return str(value).strip()


def product_identity(product_id: Any, product_name: Any) -> list[str]:
    out: list[str] = []
    for value in [product_id, product_name]:
        if has_value(value):
            out.append(str(value).strip())
    return out


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_bounds(row: pd.Series) -> tuple[list[float], str]:
    raw = row.get("plume_bounds")
    bounds = None
    if has_value(raw):
        try:
            import ast

            parsed = ast.literal_eval(str(raw))
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


def build_old_poly(lon: float, lat: float, half_deg: float) -> str:
    return (
        f"({lon - half_deg} {lat - half_deg},"
        f"{lon - half_deg} {lat + half_deg},"
        f"{lon + half_deg} {lat + half_deg},"
        f"{lon + half_deg} {lat - half_deg},"
        f"{lon - half_deg} {lat - half_deg})"
    )


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
    return f"S2|R{orbit}|{acq:%Y%m%dT%H}{minute:02d}"


def s5p_overpass_key(product: dict[str, Any]) -> str:
    name = str(product.get("Name", "")).strip()
    match = re.search(r"_(\d{5})_\d{2}_\d{6}_", name)
    if match:
        return f"S5P|{match.group(1)}"
    acq = product.get("acq_time")
    if acq is not None:
        return f"S5P|{acq.astimezone(timezone.utc):%Y%m%d}"
    return f"S5P|{name}"


def normalize_emit_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("EMIT|"):
        return text
    match = re.search(r"_(\d{7})_\d{3}$", text)
    if match:
        return f"EMIT|{match.group(1)}"
    if re.fullmatch(r"\d{7}", text):
        return f"EMIT|{text}"
    return text


def emit_overpass_key(granule_id: str) -> str:
    return normalize_emit_key(granule_id)


def emit_time_from_granule_id(granule_id: str) -> str:
    match = re.search(r"_(\d{8}T\d{6})_", str(granule_id))
    if not match:
        return ""
    dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return iso_z(dt)


def product_from_master_t0(row: pd.Series, sensor: str) -> Optional[dict[str, Any]]:
    t0_time = parse_utc(row.get("image_time"))
    if t0_time is None:
        return None
    pid = str(row.get("product_id", "")).strip() if has_value(row.get("product_id")) else ""
    pname = str(row.get("product_name", "")).strip() if has_value(row.get("product_name")) else ""
    if not pid and not pname:
        return None
    key = str(row.get("overpass_key", "")).strip()
    if sensor == "S2" and not key:
        key = s2_overpass_key({"Name": pname, "acq_time": t0_time})
    elif sensor == "S5P" and not key:
        key = s5p_overpass_key({"Name": pname, "acq_time": t0_time})
    elif sensor == "EMIT":
        key = normalize_emit_key(key or pid or pname)
    return {"Id": pid or pname, "Name": pname or pid, "acq_time": t0_time, "overpass_key": key}


def group_products_by_overpass(
    products: list[dict[str, Any]],
    key_func: Any,
    anchor: datetime,
    t0_product: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    t0_id = str(t0_product.get("Id", "")).strip()
    t0_name = str(t0_product.get("Name", "")).strip()
    t0_key = str(t0_product.get("overpass_key", "")).strip()
    groups: dict[str, list[dict[str, Any]]] = {}
    for product in products:
        acq = product.get("acq_time")
        if acq is None:
            continue
        acq = acq.astimezone(timezone.utc)
        if acq >= anchor:
            continue
        pid = str(product.get("Id", "")).strip()
        pname = str(product.get("Name", "")).strip()
        key = key_func(product)
        if pid and (pid == t0_id or pid == t0_name):
            continue
        if pname and (pname == t0_id or pname == t0_name):
            continue
        if key and key == t0_key:
            continue
        groups.setdefault(key, []).append(product)
    return sorted(groups.values(), key=lambda group: max(p["acq_time"] for p in group), reverse=True)


def select_group_representative(products: list[dict[str, Any]], target: datetime) -> dict[str, Any]:
    same_day = [p for p in products if p["acq_time"].date() == target.date()]
    choices = same_day if same_day else products
    return min(choices, key=lambda p: abs((p["acq_time"] - target).total_seconds()))


def cdse_datetime_to_query_string(runtime_module: Any, value: datetime) -> str:
    return runtime_module.datetime_to_query_string(value)


def quiet_cdse_fetch_products(
    runtime_module: Any,
    collection_name: str,
    product_type: str,
    poly: str,
    start: datetime,
    end: datetime,
    cloud_max: Optional[float] = None,
) -> list[dict[str, Any]]:
    start_ts = cdse_datetime_to_query_string(runtime_module, start)
    end_ts = cdse_datetime_to_query_string(runtime_module, end)
    cloud_filter = ""
    if cloud_max is not None:
        cloud_filter = (
            "and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {cloud_max:.2f}) "
        )
    next_link = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq '{collection_name}' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
        f"{cloud_filter}"
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON({poly})') "
        f"and ContentDate/Start gt {start_ts} "
        f"and ContentDate/Start lt {end_ts}"
        "&$top=1000"
    )
    products: list[dict[str, Any]] = []
    while next_link:
        resp = runtime_module.request_with_backoff(
            lambda proxy: runtime_module.requests.get(next_link, proxies=runtime_module.build_proxy_dict(proxy)),
            description="catalogue query",
        )
        resp.raise_for_status()
        payload = resp.json()
        for product in payload.get("value", []):
            start_time = product.get("ContentDate", {}).get("Start")
            if not start_time:
                continue
            acq = runtime_module.parse_iso_datetime(start_time)
            if acq is None:
                continue
            products.append({"Id": product.get("Id"), "Name": product.get("Name"), "acq_time": acq})
        next_link = payload.get("@odata.nextLink", "")
    return products


class LocalIndex:
    def __init__(self, raw_root: str) -> None:
        self.raw_root = Path(raw_root)
        self.by_sensor_plume_product: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        self.by_sensor_product: dict[tuple[str, str], list[dict[str, str]]] = {}

    def add(
        self,
        sensor: str,
        plume_id: str,
        timepoint: str,
        path: str,
        product_id: Any,
        product_name: Any,
        image_time: Any,
        source: str,
    ) -> None:
        path = path_text(path)
        if not path:
            return
        record = {
            "sensor": sensor,
            "plume_id": str(plume_id or "").strip(),
            "timepoint": str(timepoint or "").strip(),
            "path": path,
            "product_id": str(product_id or "").strip(),
            "product_name": str(product_name or "").strip(),
            "image_time": iso_z(image_time),
            "source": source,
        }
        keys = product_identity(product_id, product_name)
        for key in keys:
            self.by_sensor_product.setdefault((sensor, key), []).append(record)
            if record["plume_id"]:
                self.by_sensor_plume_product.setdefault((sensor, record["plume_id"], key), []).append(record)

    def find_global_file(self, sensor: str, product_id: Any, product_name: Any) -> tuple[str, str, str, str]:
        if sensor == "EMIT":
            for key in product_identity(product_id, product_name):
                gid = emit_granule_from_path(key) or key
                if not gid:
                    continue
                base = self.raw_root / "EMIT" / "raw_granules" / gid
                for path in [base / "emit_ch4_32.npz", base / f"{gid}.nc"]:
                    if existing_path(path):
                        return "found_product_global", str(path), "emit_raw_granules:on_demand", ""
        if sensor == "S5P":
            for key in product_identity(product_id, product_name):
                pname = s5p_product_from_path(key) or str(key)
                if not pname:
                    continue
                base = self.raw_root / "S5P" / "raw_data_dir_s5p"
                direct = base / pname
                if existing_path(direct):
                    return "found_product_global", str(direct), "s5p_raw_data_dir:on_demand", ""
                stem = pname[:-3] if pname.endswith(".nc") else pname
                matches = list(base.glob(f"{stem}*.nc")) + list(base.glob(f"*{stem}*_extracted/**/*.nc"))
                for path in matches:
                    if existing_path(path):
                        return "found_product_global", str(path), "s5p_raw_data_dir:on_demand", ""
        return "missing_local_file", "", "", ""

    def find(self, sensor: str, plume_id: str, product_id: Any, product_name: Any) -> tuple[str, str, str, str]:
        keys = product_identity(product_id, product_name)
        plume_matches: list[dict[str, str]] = []
        global_matches: list[dict[str, str]] = []
        for key in keys:
            plume_matches.extend(self.by_sensor_plume_product.get((sensor, plume_id, key), []))
            if sensor != "S2":
                global_matches.extend(self.by_sensor_product.get((sensor, key), []))
        plume_matches = [rec for rec in plume_matches if existing_path(rec["path"])]
        global_matches = [rec for rec in global_matches if existing_path(rec["path"])]
        matches = plume_matches or global_matches
        if not matches and sensor == "EMIT":
            for tp in ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]:
                candidate = self.raw_root / "EMIT" / tp / plume_id / "emit_ch4_32.npz"
                path = existing_path(candidate)
                if not path:
                    continue
                gid = emit_granule_from_path(path)
                if gid and any(gid == key for key in keys):
                    return "found_for_plume", path, "emit_timepoint_npz:on_demand", tp
        if not matches:
            return self.find_global_file(sensor, product_id, product_name)
        # Keep deterministic output and prefer plume-level records over global raw caches.
        matches = sorted(matches, key=lambda r: (0 if r in plume_matches else 1, r["source"], r["path"]))
        rec = matches[0]
        status = "found_for_plume" if plume_matches else "found_product_global"
        return status, rec["path"], rec["source"], rec["timepoint"]


def read_emit_granule_from_npz(path: str) -> str:
    try:
        with np.load(path) as data:
            if "granule_id" in data.files:
                value = data["granule_id"]
                if value.shape == ():
                    return str(value.item())
                return str(value.tolist())
    except Exception:
        return ""
    return ""


def emit_granule_from_path_name(path: str) -> str:
    match = re.search(r"(EMIT_L2A_RFL_\d+_\d{8}T\d{6}_\d{7}_\d{3})", str(path))
    if match:
        return match.group(1)
    return ""


def emit_granule_from_path(path: str) -> str:
    found = emit_granule_from_path_name(path)
    if found:
        return found
    if str(path).endswith(".npz"):
        return read_emit_granule_from_npz(path)
    return ""


def s5p_product_from_path(path: str) -> str:
    name = Path(str(path)).name
    match = re.search(r"(S5P_(?:OFFL|RPRO)_L2__CH4____[^/]+?\.nc)", name)
    if match:
        return match.group(1)
    return name if name.startswith("S5P_") and name.endswith(".nc") else ""


def build_local_index(args: argparse.Namespace, master: pd.DataFrame, wanted: set[tuple[str, str]]) -> LocalIndex:
    index = LocalIndex(args.raw_root)
    wanted_plumes = {plume_id for _, plume_id in wanted}
    wanted_keys = {f"{sensor}\0{plume_id}" for sensor, plume_id in wanted}
    master_keys = master["sensor"].astype(str).str.upper() + "\0" + master["plume_id"].astype(str)
    master_subset = master[master_keys.isin(wanted_keys)].copy()

    for _, row in master_subset.iterrows():
        sensor = str(row.get("sensor", "")).strip().upper()
        if sensor not in {"S2", "EMIT", "S5P"}:
            continue
        plume_id = str(row.get("plume_id", "")).strip()
        tp = str(row.get("timepoint", "")).strip()
        product_id = row.get("product_id", "")
        product_name = row.get("product_name", "")
        image_time = row.get("image_time", "")
        for col in ["downloaded_path", "processed_path", "existing_raw_path", "existing_512_path"]:
            if col in master.columns:
                path = path_text(row.get(col, ""))
                if not path:
                    continue
                pid, pname, ptime = product_id, product_name, image_time
                if sensor == "EMIT":
                    gid = emit_granule_from_path_name(path) or str(product_id or product_name or "").strip()
                    pid = pname = gid
                    ptime = emit_time_from_granule_id(gid) or image_time
                elif sensor == "S5P":
                    pname2 = s5p_product_from_path(path)
                    if pname2:
                        pname = pname2
                index.add(sensor, plume_id, tp, path, pid, pname, ptime, f"master:{col}")

    s2_log = Path(args.s2_download_manifest)
    if s2_log.exists():
        try:
            df = pd.read_csv(s2_log, low_memory=False)
            for _, row in df.iterrows():
                if str(row.get("plume_id", "")).strip() not in wanted_plumes:
                    continue
                if str(row.get("status", "")).strip() != "downloaded":
                    continue
                index.add(
                    "S2",
                    row.get("plume_id", ""),
                    row.get("timepoint", ""),
                    row.get("raw_path", ""),
                    row.get("product_id", ""),
                    row.get("product_name", ""),
                    row.get("acquisition_time", ""),
                    "s2_download_manifest:downloaded",
                )
        except Exception as exc:
            print(f"warning: failed to read {s2_log}: {exc}", flush=True)

    emit_log = Path(args.emit_download_manifest)
    if emit_log.exists():
        try:
            df = pd.read_csv(emit_log, low_memory=False)
            for _, row in df.iterrows():
                if str(row.get("plume_id", "")).strip() not in wanted_plumes:
                    continue
                path = path_text(row.get("raw_path", ""))
                if not path:
                    continue
                gid = emit_granule_from_path_name(path) or str(row.get("granule_id", "")).strip()
                if not gid:
                    continue
                index.add(
                    "EMIT",
                    row.get("plume_id", ""),
                    row.get("timepoint", ""),
                    path,
                    gid,
                    gid,
                    emit_time_from_granule_id(gid),
                    "emit_download_manifest:path_identity",
                )
        except Exception as exc:
            print(f"warning: failed to read {emit_log}: {exc}", flush=True)

    s5p_log = Path(args.s5p_download_manifest)
    if s5p_log.exists():
        try:
            df = pd.read_csv(s5p_log, low_memory=False)
            for _, row in df.iterrows():
                if str(row.get("plume_id", "")).strip() not in wanted_plumes:
                    continue
                path = path_text(row.get("raw_path", ""))
                if not path:
                    continue
                pname = s5p_product_from_path(path) or row.get("product_name", "")
                index.add(
                    "S5P",
                    row.get("plume_id", ""),
                    row.get("timepoint", ""),
                    path,
                    row.get("product_id", ""),
                    pname,
                    row.get("image_time", ""),
                    "s5p_download_manifest:path_identity",
                )
        except Exception as exc:
            print(f"warning: failed to read {s5p_log}: {exc}", flush=True)

    return index


def current_manifest_snapshot(master_rows: pd.DataFrame, tp: str) -> tuple[str, str, str, str, str]:
    rows = master_rows[master_rows["timepoint"].astype(str) == tp]
    if rows.empty:
        return "", "", "", "", ""
    row = rows.iloc[0]
    path = ""
    for col in ["downloaded_path", "processed_path", "existing_raw_path", "existing_512_path"]:
        if col in row and existing_path(row.get(col, "")):
            path = str(row.get(col, ""))
            break
    return (
        str(row.get("product_id", "") or ""),
        str(row.get("product_name", "") or ""),
        iso_z(row.get("image_time", "")),
        path,
        str(row.get("download_status", "") or ""),
    )


class CatalogueRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        repo = Path.cwd()
        self.s2 = load_module("legacy_s2_catalogue_audit", repo / "data_preprocess" / "carbon_mapper_sentinel2_90360_plume_download.py")
        self.s5p = load_module("legacy_s5p_catalogue_audit", repo / "data_downloading" / "carbon_mapper_sentinel5p_90360_plume_download.py")
        with self.s2.proxy_manager_lock:
            self.s2.proxy_manager = self.s2.build_proxy_manager({})
        with self.s5p.proxy_manager_lock:
            self.s5p.proxy_manager = self.s5p.build_proxy_manager({})
        self._earthaccess = None

    @property
    def earthaccess(self) -> Any:
        if self._earthaccess is None:
            import earthaccess

            self._earthaccess = earthaccess
        return self._earthaccess


def query_s2_prev(runtime: CatalogueRuntime, row: pd.Series, t0_product: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str, int, int]:
    anchor = t0_product["acq_time"].astimezone(timezone.utc)
    start = anchor - timedelta(days=runtime.args.s2_prev_search_back_days)
    end = anchor
    _, poly = parse_bounds(row)
    products = quiet_cdse_fetch_products(
        runtime.s2,
        "SENTINEL-2",
        "S2MSI2A",
        poly,
        start,
        end,
        cloud_max=20.0,
    )
    groups = group_products_by_overpass(products, s2_overpass_key, anchor, t0_product)
    selected = []
    for group in groups[:3]:
        item = select_group_representative(group, anchor)
        item["overpass_key"] = s2_overpass_key(item)
        selected.append(item)
    return selected, iso_z(start), iso_z(end), len(products), len(groups)


def query_s5p_prev(runtime: CatalogueRuntime, row: pd.Series, t0_product: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str, int, int]:
    anchor = t0_product["acq_time"].astimezone(timezone.utc)
    start = anchor - timedelta(days=runtime.args.s5p_prev_search_back_days)
    end = anchor
    poly = build_old_poly(float(row["plume_longitude"]), float(row["plume_latitude"]), runtime.args.s5p_geo_half_deg)
    products = quiet_cdse_fetch_products(
        runtime.s5p,
        runtime.s5p.S5P_COLLECTION_NAME,
        runtime.s5p.S5P_PRODUCT_TYPE,
        poly,
        start,
        end,
        cloud_max=None,
    )
    groups = group_products_by_overpass(products, s5p_overpass_key, anchor, t0_product)
    selected = []
    for group in groups[:3]:
        item = max(group, key=lambda p: p["acq_time"])
        item["overpass_key"] = s5p_overpass_key(item)
        selected.append(item)
    return selected, iso_z(start), iso_z(end), len(products), len(groups)


def emit_get_id(granule: Any) -> str:
    for path in [["umm", "GranuleUR"], ["meta", "native-id"], ["meta", "concept-id"]]:
        cur = granule
        for key in path:
            if cur is None:
                break
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                try:
                    cur = cur[key]
                except Exception:
                    cur = getattr(cur, key, None)
        if has_value(cur):
            return str(cur).strip()
    return ""


def emit_get_time(granule: Any) -> Optional[datetime]:
    cur = granule
    for key in ["umm", "TemporalExtent", "RangeDateTime", "BeginningDateTime"]:
        if cur is None:
            break
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            try:
                cur = cur[key]
            except Exception:
                cur = getattr(cur, key, None)
    return parse_utc(cur)


def query_emit_prev(runtime: CatalogueRuntime, row: pd.Series, t0_product: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str, int, int]:
    anchor = t0_product["acq_time"].astimezone(timezone.utc)
    start = anchor - timedelta(days=runtime.args.emit_prev_search_back_days)
    end = anchor
    results = runtime.earthaccess.search_data(
        short_name=runtime.args.emit_short_name,
        point=(float(row["plume_longitude"]), float(row["plume_latitude"])),
        temporal=(iso_z(start), iso_z(end)),
        count=runtime.args.emit_search_count,
    )
    t0_key = normalize_emit_key(t0_product.get("overpass_key") or t0_product.get("Id") or t0_product.get("Name"))
    t0_id = str(t0_product.get("Id", "")).strip()
    by_pass: dict[str, dict[str, Any]] = {}
    for granule in results:
        gid = emit_get_id(granule)
        gtime = emit_get_time(granule)
        if not gid or gtime is None or gtime >= anchor:
            continue
        key = emit_overpass_key(gid)
        if gid == t0_id or key == t0_key:
            continue
        item = {"Id": gid, "Name": gid, "acq_time": gtime, "overpass_key": key}
        old = by_pass.get(key)
        if old is None or item["acq_time"] > old["acq_time"]:
            by_pass[key] = item
    selected = sorted(by_pass.values(), key=lambda p: p["acq_time"], reverse=True)[:3]
    return selected, iso_z(start), iso_z(end), len(results), len(by_pass)


def make_empty_expected_rows(
    sensor: str,
    row: pd.Series,
    t0_product: dict[str, Any],
    master_rows: pd.DataFrame,
    query_start: str,
    query_end: str,
    product_count: int,
    overpass_count: int,
    note: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tp in PREV_RANKS:
        cur_pid, cur_name, cur_time, cur_path, cur_status = current_manifest_snapshot(master_rows, tp)
        rows.append(
            {
                "sensor": sensor,
                "plume_id": row["plume_id"],
                "expected_timepoint": tp,
                "event_time": row.get("event_time", ""),
                "plume_latitude": row.get("plume_latitude", ""),
                "plume_longitude": row.get("plume_longitude", ""),
                "t0_product_id": t0_product.get("Id", ""),
                "t0_product_name": t0_product.get("Name", ""),
                "t0_image_time": iso_z(t0_product.get("acq_time")),
                "t0_overpass_key": t0_product.get("overpass_key", ""),
                "query_start_utc": query_start,
                "query_end_utc": query_end,
                "catalogue_product_count": product_count,
                "catalogue_distinct_overpass_count": overpass_count,
                "expected_status": "no_catalogue_overpass",
                "local_status": "",
                "current_manifest_product_id": cur_pid,
                "current_manifest_product_name": cur_name,
                "current_manifest_image_time": cur_time,
                "current_manifest_path": cur_path,
                "current_manifest_status": cur_status,
                "notes": note,
            }
        )
    return rows


def audit_one(
    sensor: str,
    row: pd.Series,
    master_rows: pd.DataFrame,
    runtime: CatalogueRuntime,
    local_index: LocalIndex,
) -> list[dict[str, Any]]:
    t0_product = product_from_master_t0(row, sensor)
    if t0_product is None:
        return []
    try:
        if sensor == "S2":
            selected, query_start, query_end, product_count, overpass_count = query_s2_prev(runtime, row, t0_product)
            key_func = lambda p: p.get("overpass_key") or s2_overpass_key(p)
        elif sensor == "S5P":
            selected, query_start, query_end, product_count, overpass_count = query_s5p_prev(runtime, row, t0_product)
            key_func = lambda p: p.get("overpass_key") or s5p_overpass_key(p)
        elif sensor == "EMIT":
            selected, query_start, query_end, product_count, overpass_count = query_emit_prev(runtime, row, t0_product)
            key_func = lambda p: p.get("overpass_key") or emit_overpass_key(str(p.get("Id", "")))
        else:
            return []
    except Exception as exc:
        return make_empty_expected_rows(sensor, row, t0_product, master_rows, "", "", 0, 0, f"catalogue_error:{type(exc).__name__}:{exc}")

    rows: list[dict[str, Any]] = []
    for tp, rank in PREV_RANKS.items():
        product = selected[rank - 1] if len(selected) >= rank else None
        cur_pid, cur_name, cur_time, cur_path, cur_status = current_manifest_snapshot(master_rows, tp)
        base = {
            "sensor": sensor,
            "plume_id": row["plume_id"],
            "expected_timepoint": tp,
            "event_time": row.get("event_time", ""),
            "plume_latitude": row.get("plume_latitude", ""),
            "plume_longitude": row.get("plume_longitude", ""),
            "t0_product_id": t0_product.get("Id", ""),
            "t0_product_name": t0_product.get("Name", ""),
            "t0_image_time": iso_z(t0_product.get("acq_time")),
            "t0_overpass_key": t0_product.get("overpass_key", ""),
            "query_start_utc": query_start,
            "query_end_utc": query_end,
            "catalogue_product_count": product_count,
            "catalogue_distinct_overpass_count": overpass_count,
            "current_manifest_product_id": cur_pid,
            "current_manifest_product_name": cur_name,
            "current_manifest_image_time": cur_time,
            "current_manifest_path": cur_path,
            "current_manifest_status": cur_status,
            "notes": "",
        }
        if product is None:
            base["expected_status"] = "no_catalogue_overpass"
            base["local_status"] = ""
            rows.append(base)
            continue
        expected_id = str(product.get("Id", "")).strip()
        expected_name = str(product.get("Name", "")).strip()
        local_status, local_path, local_source, local_tp = local_index.find(sensor, str(row["plume_id"]), expected_id, expected_name)
        base.update(
            {
                "expected_product_id": expected_id,
                "expected_product_name": expected_name,
                "expected_image_time": iso_z(product.get("acq_time")),
                "expected_overpass_key": key_func(product),
                "expected_status": "expected",
                "local_status": local_status,
                "local_path": local_path,
                "local_path_source": local_source,
                "local_current_timepoint": local_tp,
            }
        )
        rows.append(base)
    return rows


def load_existing_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["sensor", "plume_id"])
    except Exception:
        return set()
    return {(str(r.sensor), str(r.plume_id)) for r in df.itertuples(index=False)}


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            exists = path.exists()
            with path.open("a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                for record in records:
                    writer.writerow({field: record.get(field, "") for field in OUT_FIELDS})
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Catalogue-only expected prev1/prev2/prev3 audit for S2, EMIT, and S5P.")
    parser.add_argument("--manifest", default="Upgrade_data_pipeline/csv/multisensor_6time_download_manifest.csv")
    parser.add_argument("--out-csv", default="Upgrade_data_pipeline/csv/s2_emit_s5p_expected_prev_catalogue_audit.csv")
    parser.add_argument("--raw-root", default="/mnt/engg-niulab/yuyao/sensors_raw_data")
    parser.add_argument("--sensors", default="S2,EMIT,S5P")
    parser.add_argument("--limit-per-sensor", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--s2-download-manifest", default="Upgrade_data_pipeline/csv/s2_download_manifest.csv")
    parser.add_argument("--emit-download-manifest", default="Upgrade_data_pipeline/csv/emit_download_manifest.csv")
    parser.add_argument("--s5p-download-manifest", default="Upgrade_data_pipeline/csv/s5p_download_manifest.csv")
    parser.add_argument("--s2-prev-search-back-days", type=int, default=120)
    parser.add_argument("--emit-prev-search-back-days", type=int, default=365)
    parser.add_argument("--emit-short-name", default="EMITL2ARFL")
    parser.add_argument("--emit-search-count", type=int, default=200)
    parser.add_argument("--s5p-prev-search-back-days", type=int, default=30)
    parser.add_argument("--s5p-geo-half-deg", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sensors = [s.strip().upper() for s in args.sensors.split(",") if s.strip()]
    master = pd.read_csv(args.manifest, low_memory=False)
    runtime = CatalogueRuntime(args)
    done = load_existing_done(Path(args.out_csv)) if args.resume else set()
    append_lock = Lock()

    tasks: list[tuple[str, pd.Series, pd.DataFrame]] = []
    for sensor in sensors:
        sensor_rows = master[(master["sensor"].astype(str).str.upper() == sensor)].copy()
        plume_groups = sensor_rows.groupby(sensor_rows["plume_id"].astype(str), sort=False)
        t0_rows = sensor_rows[sensor_rows["timepoint"].astype(str) == "t0"].copy()
        valid = []
        for _, row in t0_rows.iterrows():
            plume_id = str(row.get("plume_id", "")).strip()
            if args.resume and (sensor, plume_id) in done:
                continue
            if product_from_master_t0(row, sensor) is None:
                continue
            try:
                plume_rows = plume_groups.get_group(plume_id).copy()
            except KeyError:
                continue
            valid.append((sensor, row, plume_rows))
            if args.limit_per_sensor and len(valid) >= args.limit_per_sensor:
                break
        tasks.extend(valid)
        print(f"{sensor}: queued {len(valid)} plume t0 anchors", flush=True)

    wanted = {(task[0], str(task[1].get("plume_id", "")).strip()) for task in tasks}
    local_index = build_local_index(args, master, wanted)
    print(f"local index built for {len(wanted)} plume/sensor anchors", flush=True)

    completed = 0
    started = time.time()
    def run_task(task: tuple[str, pd.Series, pd.DataFrame]) -> list[dict[str, Any]]:
        return audit_one(task[0], task[1], task[2], runtime, local_index)

    if args.workers == 1:
        for task in tasks:
            records = run_task(task)
            append_records(Path(args.out_csv), records)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                elapsed = max(time.time() - started, 1)
                print(f"completed {completed}/{len(tasks)} plume anchors; {completed/elapsed:.2f}/s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_task, task) for task in tasks]
            for fut in as_completed(futures):
                records = fut.result()
                with append_lock:
                    append_records(Path(args.out_csv), records)
                    completed += 1
                    if completed % 25 == 0 or completed == len(tasks):
                        elapsed = max(time.time() - started, 1)
                        print(f"completed {completed}/{len(tasks)} plume anchors; {completed/elapsed:.2f}/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
