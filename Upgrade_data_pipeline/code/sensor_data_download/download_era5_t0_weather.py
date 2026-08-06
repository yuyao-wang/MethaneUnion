#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "boundary_layer_height",
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_column_water_vapour",
    "total_cloud_cover",
    "total_precipitation",
]
VAR_TO_SHORT = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "boundary_layer_height": "blh",
    "surface_pressure": "sp",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "total_column_water_vapour": "tcwv",
    "total_cloud_cover": "tcc",
    "total_precipitation": "tp",
}
SUCCESS_AVAILABLE = {
    "downloaded",
    "downloaded_deleted_drive",
    "linked_existing",
    "resume_skip_completed",
    "resume_skip_completed_deleted_drive",
    "selected",
    "skip_drive_file_exists",
    "skip_existing",
    "skip_existing_512",
    "skip_existing_raw",
    "skip_existing_valid",
    "skip_existing_valid_deleted_drive",
    "skip_gee_task_pending",
    "skip_local_tif_exists",
    "submitted",
}
SUCCESS_LOCAL = {
    "downloaded",
    "downloaded_deleted_drive",
    "linked_existing",
    "resume_skip_completed",
    "resume_skip_completed_deleted_drive",
    "skip_existing",
    "skip_existing_512",
    "skip_existing_raw",
    "skip_existing_valid",
    "skip_existing_valid_deleted_drive",
    "skip_local_tif_exists",
}
DEFAULT_LOGS = [
    "Upgrade_data_pipeline/csv/s2_download_manifest.csv",
    "Upgrade_data_pipeline/csv/l89_gee_drive_submit_manifest.csv",
    "Upgrade_data_pipeline/csv/l89_drive_pull_manifest.csv",
    "Upgrade_data_pipeline/csv/emit_download_manifest.csv",
    "Upgrade_data_pipeline/csv/s5p_download_manifest.csv",
]
L89_COMPLETE_TIMEPOINTS = [
    ("t0", "has_t0", "t0_512_path", "t0_image_time"),
    ("prev1", "has_prev1", "prev1_512_path", "prev1_image_time"),
    ("prev2", "has_prev2", "prev2_512_path", "prev2_image_time"),
    ("prev3", "has_prev3", "prev3_512_path", "prev3_image_time"),
    ("seasonal", "has_seasonal", "seasonal_512_path", "seasonal_image_time"),
    ("year", "has_year", "year_512_path", "year_image_time"),
]
OUT_FIELDS = [
    "plume_id",
    "sensor",
    "timepoint",
    "status",
    "event_time",
    "image_time",
    "image_time_source",
    "era5_time_utc",
    "time_delta_minutes",
    "plume_latitude",
    "plume_longitude",
    "era5_latitude",
    "era5_longitude",
    "image_evidence",
    "era5_cache_path",
    "u10",
    "v10",
    "wind_speed_10m",
    "wind_dir_10m",
    "u100",
    "v100",
    "wind_speed_100m",
    "wind_dir_100m",
    "boundary_layer_height",
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_column_water_vapour",
    "total_cloud_cover",
    "total_precipitation",
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


def truthy(value: Any) -> bool:
    if not has_value(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_time(value: Any) -> Optional[pd.Timestamp]:
    if not has_value(value):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def iso_z(ts: pd.Timestamp) -> str:
    return ts.tz_convert(timezone.utc).isoformat().replace("+00:00", "Z")


def round_hour(ts: pd.Timestamp, mode: str) -> pd.Timestamp:
    if mode == "floor":
        return ts.floor("h")
    if mode == "ceil":
        return ts.ceil("h")
    return (ts + pd.Timedelta(minutes=30)).floor("h")


def parse_emit_time(value: Any) -> Optional[pd.Timestamp]:
    if not has_value(value):
        return None
    text = str(value)
    match = re.search(r"(?P<date>20\d{6})T(?P<time>\d{6})", text)
    if match:
        return parse_time(f"{match.group('date')}T{match.group('time')}Z")
    match = re.search(r"emi(?P<date>\d{8})t(?P<time>\d{6})", text, re.I)
    if match:
        return parse_time(f"{match.group('date')}T{match.group('time')}Z")
    return None


def log_image_time(sensor: str, row: dict[str, Any]) -> tuple[str, str]:
    if sensor == "S2":
        return str(row.get("acquisition_time", "")).strip(), "s2_download_manifest.acquisition_time"
    if sensor == "L89":
        if has_value(row.get("image_time")):
            return str(row.get("image_time", "")).strip(), "l89_submit_manifest.image_time"
        ts = parse_time(row.get("image_time_file"))
        if ts is not None:
            return iso_z(ts), "l89_pull_manifest.image_time_file"
        return "", ""
    if sensor == "S5P":
        return str(row.get("image_time", "")).strip(), f"{sensor.lower()}_manifest.image_time"
    if sensor == "EMIT":
        for key in ("granule_id", "source_nc", "raw_path", "plume_id"):
            ts = parse_emit_time(row.get(key, ""))
            if ts is not None:
                return iso_z(ts), f"emit_{key}"
    return "", ""


def infer_sensor_from_path(path: str) -> str:
    name = Path(path).name.lower()
    if "s2" in name:
        return "S2"
    if "l89" in name:
        return "L89"
    if "emit" in name:
        return "EMIT"
    if "s5p" in name:
        return "S5P"
    return ""


def existing_path(value: Any) -> bool:
    if not has_value(value):
        return False
    try:
        p = Path(str(value).strip())
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


def expected_target_exists(row: dict[str, Any]) -> bool:
    sensor = str(row.get("sensor", "")).strip().upper()
    target_dir = row.get("target_raw_dir", "")
    if sensor == "EMIT" and has_value(target_dir):
        return existing_path(Path(str(target_dir)) / "emit_ch4_32.npz")
    for key in ("existing_raw_path", "existing_512_path", "target_512_path"):
        if existing_path(row.get(key, "")):
            return True
    return False


def ensure_plume(plumes: dict[str, dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    plume_id = str(row.get("plume_id", "")).strip()
    item = plumes.setdefault(
        plume_id,
        {
            "plume_id": plume_id,
            "event_time": row.get("event_time", ""),
            "plume_latitude": row.get("plume_latitude", ""),
            "plume_longitude": row.get("plume_longitude", ""),
            "image_records": {},
        },
    )
    for key in ("event_time", "plume_latitude", "plume_longitude"):
        if not has_value(item.get(key)) and has_value(row.get(key)):
            item[key] = row.get(key)
    return item


def ensure_image_record(item: dict[str, Any], sensor: str, timepoint: str) -> dict[str, Any]:
    records = item.setdefault("image_records", {})
    key = (sensor, timepoint)
    return records.setdefault(
        key,
        {
            "sensor": sensor,
            "timepoint": timepoint,
            "planned": False,
            "available": False,
            "local": False,
            "image_time": "",
            "time_source": "",
            "evidence": set(),
        },
    )


def set_record_time(record: dict[str, Any], image_time: Any, source: str, prefer: bool = False) -> None:
    if not has_value(image_time):
        return
    if prefer or not has_value(record.get("image_time")):
        record["image_time"] = str(image_time).strip()
        record["time_source"] = source


def load_manifest_images(path: Path, include_manifest_records: bool = False) -> dict[str, dict[str, Any]]:
    plumes: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("action", "")).strip() != "download":
                continue
            sensor = str(row.get("sensor", "")).strip().upper()
            timepoint = str(row.get("timepoint", "")).strip()
            plume_id = str(row.get("plume_id", "")).strip()
            if not plume_id or not sensor or not timepoint:
                continue
            item = ensure_plume(plumes, row)
            if not include_manifest_records:
                continue
            record = ensure_image_record(item, sensor, timepoint)
            if timepoint == "t0":
                set_record_time(record, row.get("t0_available_time", ""), "manifest.t0_available_time")
            if truthy(row.get("sensor_has_t0", "")) or timepoint != "t0":
                record["planned"] = True
                record["evidence"].add("manifest_planned")
    return plumes


def merge_image_log(plumes: dict[str, dict[str, Any]], log_path: Path) -> None:
    if not log_path.exists():
        return
    sensor = infer_sensor_from_path(str(log_path))
    if not sensor:
        return
    try:
        fh = log_path.open(newline="")
    except OSError:
        return
    with fh:
        reader = csv.DictReader(fh)
        for row in reader:
            timepoint = str(row.get("timepoint", "")).strip()
            plume_id = str(row.get("plume_id", "")).strip()
            if not plume_id or not timepoint or plume_id not in plumes:
                continue
            status = str(row.get("status", "")).strip()
            record = ensure_image_record(plumes[plume_id], sensor, timepoint)
            image_time, source = log_image_time(sensor, row)
            set_record_time(record, image_time, source, prefer=True)
            if status in SUCCESS_AVAILABLE:
                record["available"] = True
                record["evidence"].add(status)
            if status in SUCCESS_LOCAL:
                record["local"] = True


def merge_l89_complete_paths(plumes: dict[str, dict[str, Any]], path: Path) -> None:
    if not path.exists():
        return
    try:
        fh = path.open(newline="")
    except OSError:
        return
    with fh:
        reader = csv.DictReader(fh)
        for row in reader:
            plume_id = str(row.get("plume_id", "")).strip()
            if not plume_id:
                continue
            item = ensure_plume(plumes, row)
            for timepoint, flag_col, path_col, time_col in L89_COMPLETE_TIMEPOINTS:
                if not truthy(row.get(flag_col, "")):
                    continue
                if not has_value(row.get(path_col, "")) or not has_value(row.get(time_col, "")):
                    continue
                record = ensure_image_record(item, "L89", timepoint)
                set_record_time(record, row.get(time_col, ""), f"l89_complete_paths.{time_col}", prefer=True)
                record["planned"] = True
                record["available"] = True
                record["local"] = True
                record["evidence"].add("l89_512_complete_paths")


def record_selected(record: dict[str, Any], mode: str) -> bool:
    if mode == "manifest":
        return bool(record.get("planned") or record.get("available") or record.get("local"))
    if mode == "local":
        return bool(record.get("local"))
    return bool(record.get("available") or record.get("local"))


def selected_image_items(plumes: dict[str, dict[str, Any]], mode: str, time_rounding: str) -> list[dict[str, Any]]:
    out = []
    for item in plumes.values():
        event_ts = parse_time(item.get("event_time"))
        lat = pd.to_numeric(item.get("plume_latitude"), errors="coerce")
        lon = pd.to_numeric(item.get("plume_longitude"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        for _, record in sorted(item.get("image_records", {}).items()):
            if not record_selected(record, mode):
                continue
            image_ts = parse_time(record.get("image_time"))
            if image_ts is None:
                continue
            out_item = {
                "plume_id": item["plume_id"],
                "sensor": record["sensor"],
                "timepoint": record["timepoint"],
                "event_ts": event_ts,
                "image_ts": image_ts,
                "era5_ts": round_hour(image_ts, time_rounding),
                "lat": float(lat),
                "lon": float(lon),
                "time_source": record.get("time_source", ""),
                "evidence": set(record.get("evidence", set())),
            }
            out.append(out_item)
    out.sort(key=lambda x: (x["era5_ts"], x["plume_id"], x["sensor"], x["timepoint"]))
    return out


def load_done(out_csv: Path) -> set[tuple[str, str, str]]:
    if not out_csv.exists():
        return set()
    done = set()
    try:
        with out_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if str(row.get("status", "")).strip() == "ok" and has_value(row.get("plume_id")):
                    done.add(
                        (
                            str(row["plume_id"]).strip(),
                            str(row.get("sensor", "")).strip().upper(),
                            str(row.get("timepoint", "")).strip(),
                        )
                    )
    except Exception:
        return set()
    return done


def tag_float(value: float) -> str:
    text = f"{value:.5f}"
    return text.replace("-", "m").replace(".", "p")


def normalize_lon_180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def tile_indices(lat: float, lon: float, tile_degree: float) -> tuple[int, int]:
    lon = normalize_lon_180(lon)
    return (
        math.floor((lat + 90.0) / tile_degree),
        math.floor((lon + 180.0) / tile_degree),
    )


def tile_bounds(lat_idx: int, lon_idx: int, tile_degree: float, pad_degree: float) -> tuple[float, float, float, float]:
    south = -90.0 + lat_idx * tile_degree
    north = min(90.0, south + tile_degree)
    west = -180.0 + lon_idx * tile_degree
    east = min(180.0, west + tile_degree)
    south = max(-90.0, south - pad_degree)
    north = min(90.0, north + pad_degree)
    west = max(-180.0, west - pad_degree)
    east = min(180.0, east + pad_degree)
    return north, west, south, east


def group_key(item: dict[str, Any], tile_degree: float) -> tuple[str, int, int]:
    lat_idx, lon_idx = tile_indices(item["lat"], item["lon"], tile_degree)
    return item["era5_ts"].strftime("%Y%m%d"), lat_idx, lon_idx


def cache_path_for_group(cache_dir: Path, key: tuple[str, int, int], tile_degree: float) -> Path:
    date_s, lat_idx, lon_idx = key
    date_dir = cache_dir / date_s[:4] / date_s[4:6] / date_s[6:8]
    tile_tag = f"tile{tag_float(tile_degree)}_lat{lat_idx:03d}_lon{lon_idx:03d}"
    return date_dir / f"era5_{date_s}_{tile_tag}.nc"


def era5_area_for_group(key: tuple[str, int, int], tile_degree: float, pad_degree: float) -> list[float]:
    _, lat_idx, lon_idx = key
    north, west, south, east = tile_bounds(lat_idx, lon_idx, tile_degree, pad_degree)
    return [north, west, south, east]


def request_era5(target: Path, date_s: str, hours: list[str], area: list[float], args: argparse.Namespace) -> None:
    import cdsapi

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    request = {
        "product_type": ["reanalysis"],
        "variable": ERA5_VARIABLES,
        "year": [date_s[:4]],
        "month": [date_s[4:6]],
        "day": [date_s[6:8]],
        "time": hours,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    if args.grid:
        request["grid"] = [args.grid, args.grid]
    client = cdsapi.Client()
    client.retrieve(ERA5_DATASET, request, str(tmp))
    os.replace(tmp, target)


def netcdf_paths(path: Path) -> list[Path]:
    if not zipfile.is_zipfile(path):
        return [path]
    out_dir = path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    with zipfile.ZipFile(path) as zf:
        nc_names = [name for name in zf.namelist() if name.endswith(".nc")]
        if not nc_names:
            raise RuntimeError(f"zip has no NetCDF file: {path}")
        for nc_name in nc_names:
            out = out_dir / Path(nc_name).name
            if not out.exists() or out.stat().st_size == 0:
                zf.extract(nc_name, out_dir)
                extracted = out_dir / nc_name
                if extracted != out:
                    extracted.replace(out)
            out_paths.append(out)
    return out_paths


def netcdf_path(path: Path) -> Path:
    paths = netcdf_paths(path)
    return paths[0]


def coord_name(ds: Any, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise RuntimeError(f"missing coordinate; tried {candidates}")


def normalize_lon_for_dataset(lon: float, lons: np.ndarray) -> float:
    lon = normalize_lon_180(lon)
    finite = np.asarray(lons)[np.isfinite(lons)]
    if finite.size and finite.min() >= 0 and lon < 0:
        return lon + 360.0
    return lon


def find_data_var(ds: Any, long_name: str) -> Any:
    short = VAR_TO_SHORT[long_name]
    if short in ds.data_vars:
        return ds[short]
    if long_name in ds.data_vars:
        return ds[long_name]
    for name, arr in ds.data_vars.items():
        if str(arr.attrs.get("long_name", "")).strip().lower() == long_name.replace("_", " "):
            return ds[name]
    return None


def get_data_var(ds: Any, long_name: str) -> Any:
    arr = find_data_var(ds, long_name)
    if arr is None:
        raise RuntimeError(f"variable not found in NetCDF: {long_name}")
    return arr


def scalar_value(ds: Any, var_name: str, ts: pd.Timestamp, lat: float, lon: float) -> float:
    arr = get_data_var(ds, var_name)
    lat_name = coord_name(ds, ("latitude", "lat"))
    lon_name = coord_name(ds, ("longitude", "lon"))
    time_name = None
    for candidate in ("valid_time", "time"):
        if candidate in arr.coords or candidate in arr.dims:
            time_name = candidate
            break
    sel: dict[str, Any] = {
        lat_name: lat,
        lon_name: normalize_lon_for_dataset(lon, ds[lon_name].values),
    }
    if time_name:
        sel[time_name] = np.datetime64(ts.to_datetime64())
    try:
        picked = arr.sel(sel, method="nearest")
    except Exception:
        picked = arr
        for dim, value in sel.items():
            if dim in picked.coords or dim in picked.dims:
                picked = picked.sel({dim: value}, method="nearest")
    value = np.asarray(picked.values).reshape(-1)[0]
    return float(value)


def scalar_value_from_datasets(datasets: list[Any], var_name: str, ts: pd.Timestamp, lat: float, lon: float) -> float:
    for ds in datasets:
        if find_data_var(ds, var_name) is None:
            continue
        return scalar_value(ds, var_name, ts, lat, lon)
    raise RuntimeError(f"variable not found in NetCDF: {var_name}")


def selected_grid_point(ds: Any, lat: float, lon: float) -> tuple[float, float]:
    lat_name = coord_name(ds, ("latitude", "lat"))
    lon_name = coord_name(ds, ("longitude", "lon"))
    picked_lat = float(ds[lat_name].sel({lat_name: lat}, method="nearest").values)
    lon_for_ds = normalize_lon_for_dataset(lon, ds[lon_name].values)
    picked_lon = float(ds[lon_name].sel({lon_name: lon_for_ds}, method="nearest").values)
    return picked_lat, normalize_lon_180(picked_lon)


def wind_speed(u: float, v: float) -> float:
    return float(math.sqrt(u * u + v * v))


def wind_dir_from_uv(u: float, v: float) -> float:
    return float((270.0 - math.degrees(math.atan2(v, u))) % 360.0)


def extract_weather(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    import xarray as xr

    nc_paths = netcdf_paths(path)
    ts = item["era5_ts"]
    lat = item["lat"]
    lon = item["lon"]
    datasets = []
    try:
        datasets = [xr.open_dataset(nc) for nc in nc_paths]
        values = {name: scalar_value_from_datasets(datasets, name, ts, lat, lon) for name in ERA5_VARIABLES}
        era5_lat, era5_lon = selected_grid_point(datasets[0], lat, lon)
    finally:
        for ds in datasets:
            ds.close()

    u10 = values["10m_u_component_of_wind"]
    v10 = values["10m_v_component_of_wind"]
    u100 = values["100m_u_component_of_wind"]
    v100 = values["100m_v_component_of_wind"]
    delta = (ts - item["image_ts"]).total_seconds() / 60.0
    return {
        "plume_id": item["plume_id"],
        "sensor": item["sensor"],
        "timepoint": item["timepoint"],
        "status": "ok",
        "event_time": iso_z(item["event_ts"]) if item["event_ts"] is not None else "",
        "image_time": iso_z(item["image_ts"]),
        "image_time_source": item["time_source"],
        "era5_time_utc": iso_z(ts),
        "time_delta_minutes": f"{delta:.1f}",
        "plume_latitude": f"{lat:.8f}",
        "plume_longitude": f"{lon:.8f}",
        "era5_latitude": f"{era5_lat:.5f}",
        "era5_longitude": f"{era5_lon:.5f}",
        "image_evidence": ";".join(sorted(item["evidence"])),
        "era5_cache_path": str(path),
        "u10": u10,
        "v10": v10,
        "wind_speed_10m": wind_speed(u10, v10),
        "wind_dir_10m": wind_dir_from_uv(u10, v10),
        "u100": u100,
        "v100": v100,
        "wind_speed_100m": wind_speed(u100, v100),
        "wind_dir_100m": wind_dir_from_uv(u100, v100),
        "boundary_layer_height": values["boundary_layer_height"],
        "surface_pressure": values["surface_pressure"],
        "2m_temperature": values["2m_temperature"],
        "2m_dewpoint_temperature": values["2m_dewpoint_temperature"],
        "total_column_water_vapour": values["total_column_water_vapour"],
        "total_cloud_cover": values["total_cloud_cover"],
        "total_precipitation": values["total_precipitation"],
        "message": "",
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUT_FIELDS})


def process_group(key: tuple[str, int, int], items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    path = cache_path_for_group(Path(args.cache_dir), key, args.tile_degree)
    hours = sorted({item["era5_ts"].strftime("%H:00") for item in items})
    area = era5_area_for_group(key, args.tile_degree, args.tile_pad_degree)
    try:
        if args.overwrite_cache or not path.exists() or path.stat().st_size == 0:
            request_era5(path, key[0], hours, area, args)
            if args.request_sleep_seconds > 0:
                time.sleep(args.request_sleep_seconds)
        return [extract_weather(path, item) for item in items]
    except Exception as exc:
        return [
            {
                "plume_id": item["plume_id"],
                "sensor": item["sensor"],
                "timepoint": item["timepoint"],
                "status": "error",
                "event_time": iso_z(item["event_ts"]) if item["event_ts"] is not None else "",
                "image_time": iso_z(item["image_ts"]),
                "image_time_source": item["time_source"],
                "era5_time_utc": iso_z(item["era5_ts"]),
                "plume_latitude": item["lat"],
                "plume_longitude": item["lon"],
                "image_evidence": ";".join(sorted(item["evidence"])),
                "era5_cache_path": str(path),
                "message": f"{type(exc).__name__}: {exc}",
            }
            for item in items
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ERA5 single-level weather at each available sensor image time/location."
    )
    parser.add_argument("--manifest", default="Upgrade_data_pipeline/csv/multisensor_6time_download_manifest.csv")
    parser.add_argument("--logs", default=",".join(DEFAULT_LOGS))
    parser.add_argument(
        "--l89-complete-csv",
        default="Upgrade_data_pipeline/csv/l89_6time_complete_paths.csv",
        help="Current L89 512 complete-path table. Use an empty string to disable this source.",
    )
    parser.add_argument("--out-csv", default="Upgrade_data_pipeline/csv/era5_image_time_weather.csv")
    parser.add_argument("--cache-dir", default="/mnt/engg-niulab/yuyao/sensors_raw_data/ERA5/image_time_weather_cache")
    parser.add_argument("--image-source", "--t0-source", dest="image_source", choices=["available", "local", "manifest"], default="available")
    parser.add_argument("--time-rounding", choices=["nearest", "floor", "ceil"], default="nearest")
    parser.add_argument("--tile-degree", type=float, default=5.0)
    parser.add_argument("--tile-pad-degree", type=float, default=0.25)
    parser.add_argument("--grid", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    plumes = load_manifest_images(manifest, include_manifest_records=args.image_source == "manifest")
    for path in [Path(p.strip()) for p in args.logs.split(",") if p.strip()]:
        merge_image_log(plumes, path)
    if str(args.l89_complete_csv).strip():
        merge_l89_complete_paths(plumes, Path(args.l89_complete_csv))
    items = selected_image_items(plumes, args.image_source, args.time_rounding)
    if args.resume:
        done = load_done(Path(args.out_csv))
        items = [item for item in items if (item["plume_id"], item["sensor"], item["timepoint"]) not in done]
        log(f"resume: skipped completed plume-sensor-timepoint rows: {len(done)}")
    if args.limit:
        items = items[: args.limit]

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(group_key(item, args.tile_degree), []).append(item)
    plume_count = len({item["plume_id"] for item in items})
    by_sensor: dict[str, int] = {}
    for item in items:
        by_sensor[item["sensor"]] = by_sensor.get(item["sensor"], 0) + 1
    log(f"selected plume-sensor-timepoint image rows: {len(items)}")
    log(f"selected unique plume_ids: {plume_count}")
    log(f"selected by sensor: {dict(sorted(by_sensor.items()))}")
    by_timepoint: dict[str, int] = {}
    for item in items:
        by_timepoint[item["timepoint"]] = by_timepoint.get(item["timepoint"], 0) + 1
    log(f"selected by timepoint: {dict(sorted(by_timepoint.items()))}")
    log(f"unique ERA5 date-tile requests: {len(groups)}")
    log(f"image-source: {args.image_source}; ERA5 time = rounded sensor image time; time-rounding: {args.time_rounding}")
    if args.dry_run:
        return 0
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    out_csv = Path(args.out_csv)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_group, key, group_items, args) for key, group_items in groups.items()]
        for future in as_completed(futures):
            rows = future.result()
            completed += len(rows)
            append_rows(out_csv, rows)
            ok = sum(1 for row in rows if row.get("status") == "ok")
            log(f"[{completed}/{len(items)}] group_rows={len(rows)} ok={ok}")
    log(f"done: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
