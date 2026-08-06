#!/usr/bin/env python3
"""Crop six-time S5P CH4 samples from the path table.

This is the six-time version of preprocess_dataset_s5p/crop_and_label_s5p.py.
It keeps the same npz output format, but each timepoint is cropped independently:

* all six raw S5P products are required.
* the plume latitude/longitude is the target for every timepoint.
* every timepoint finds its own nearest pixel in its own S5P lat/lon grid.
* each S5P patch is 3x3 pixels resized to 224x224 with NaN-aware interpolation.
* all six timepoints must pass distance and missing-ratio checks before output.

The output npz contains:

    ch4.shape == (6, 224, 224)
    channels == ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from netCDF4 import Dataset


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
EARTH_RADIUS_KM = 6371.0088
PATH_COLS = {
    "t0": ["t0_raw_path", "s5p_t0_path"],
    "prev1": ["prev1_raw_path", "s5p_prev1_path"],
    "prev2": ["prev2_raw_path", "s5p_prev2_path"],
    "prev3": ["prev3_raw_path", "s5p_prev3_path"],
    "seasonal": ["seasonal_raw_path", "s5p_seasonal_path"],
    "year": ["year_raw_path", "s5p_year_path"],
}
LEGACY_PATH_ALIASES = {
    "t0": "S5p_path",
    "seasonal": "s5p_minus90_path",
    "year": "s5p_minus360_path",
}

CH4_CANDIDATES = [
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
]

DEFAULT_IN_CSV = Path("Upgrade_data_pipeline/csv/s5p_6time_with_centers.csv")

warnings.filterwarnings("ignore", category=RuntimeWarning)


@contextmanager
def silence_fd2():
    """Suppress HDF5/getfattr messages that bypass Python warnings."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(devnull)
        os.close(old)


def valid_text(value) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def get_path(row: pd.Series, timepoint: str) -> str:
    for col in [*PATH_COLS[timepoint], LEGACY_PATH_ALIASES.get(timepoint, "")]:
        if not col:
            continue
        if col in row.index and valid_text(row.get(col)):
            return str(row.get(col)).strip()
    return ""


def get_2d(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected dims: {arr.shape}")


def to_nan_invalid(arr, attrs=None) -> np.ndarray:
    out = np.array(arr, dtype=np.float32, copy=False)
    attrs = attrs or {}
    fill_value = attrs.get("_FillValue", None)
    missing_value = attrs.get("missing_value", None)
    if fill_value is not None:
        out = np.where(out == np.float32(fill_value), np.nan, out)
    if missing_value is not None:
        out = np.where(out == np.float32(missing_value), np.nan, out)
    out = np.where(np.abs(out) > 1e20, np.nan, out)
    return get_2d(out)


def pick_ch4_var(product_group) -> Optional[str]:
    for name in CH4_CANDIDATES:
        if name in product_group.variables:
            return name
    return None


def read_lat_lon(path_nc: str) -> Tuple[np.ndarray, np.ndarray]:
    with silence_fd2():
        ds = Dataset(str(path_nc), "r")
    try:
        prod = ds.groups["PRODUCT"]
        lat = get_2d(prod.variables["latitude"][:]).astype(np.float64, copy=False)
        lon = get_2d(prod.variables["longitude"][:]).astype(np.float64, copy=False)
        return lat, lon
    finally:
        ds.close()


def approx_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r = np.deg2rad(float(lat1))
    lon1r = np.deg2rad(float(lon1))
    lat2r = np.deg2rad(float(lat2))
    lon2r = np.deg2rad(float(lon2))
    dlon = (lon1r - lon2r + np.pi) % (2 * np.pi) - np.pi
    x = dlon * np.cos(0.5 * (lat1r + lat2r))
    y = lat1r - lat2r
    return float(np.sqrt(x * x + y * y) * EARTH_RADIUS_KM)


def nearest_iyix_distance(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[int, int, float]:
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    lat0r = np.deg2rad(float(lat0))
    lon0r = np.deg2rad(float(lon0))
    dlon = (lonr - lon0r + np.pi) % (2 * np.pi) - np.pi
    x = dlon * np.cos(0.5 * (latr + lat0r))
    y = latr - lat0r
    d2 = x * x + y * y
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    if not np.isfinite(d2).any():
        raise ValueError("no_finite_lat_lon")
    flat = int(np.argmin(d2))
    iy, ix = np.unravel_index(flat, d2.shape)
    return int(iy), int(ix), float(np.sqrt(d2[iy, ix]) * EARTH_RADIUS_KM)


def nearest_iyix(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[int, int]:
    iy, ix, _ = nearest_iyix_distance(lat, lon, lat0, lon0)
    return iy, ix


def nearest_iyix_distance_window(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
    center: Tuple[int, int],
    radius: int,
) -> Tuple[int, int, float]:
    cy, cx = center
    radius = max(1, int(radius))
    y0 = max(0, int(cy) - radius)
    y1 = min(lat.shape[0], int(cy) + radius + 1)
    x0 = max(0, int(cx) - radius)
    x1 = min(lat.shape[1], int(cx) + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return nearest_iyix_distance(lat, lon, lat0, lon0)
    iy, ix, dist_km = nearest_iyix_distance(lat[y0:y1, x0:x1], lon[y0:y1, x0:x1], lat0, lon0)
    return int(y0 + iy), int(x0 + ix), dist_km


def candidate_pos_centers(py: int, px: int, height: int, width: int, half: int) -> List[Tuple[int, int]]:
    centers: List[Tuple[int, int]] = []
    y_min, y_max = half, height - half - 1
    x_min, x_max = half, width - half - 1
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cy = py + dy
            cx = px + dx
            if y_min <= cy <= y_max and x_min <= cx <= x_max:
                centers.append((cy, cx))
    centers.sort(key=lambda c: (c[0] - py) ** 2 + (c[1] - px) ** 2)
    return centers


def compute_centers_from_t0(
    t0_path: str,
    ch4_t0: np.ndarray,
    lat0: float,
    lon0: float,
    center_crop_size: int,
    max_missing_ratio_t0: float,
    max_pos_per_plume: int,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    lat, lon = read_lat_lon(t0_path)
    return compute_centers_from_t0_arrays(
        lat=lat,
        lon=lon,
        ch4_t0=ch4_t0,
        lat0=lat0,
        lon0=lon0,
        center_crop_size=center_crop_size,
        max_missing_ratio_t0=max_missing_ratio_t0,
        max_pos_per_plume=max_pos_per_plume,
    )


def compute_centers_from_t0_arrays(
    lat: np.ndarray,
    lon: np.ndarray,
    ch4_t0: np.ndarray,
    lat0: float,
    lon0: float,
    center_crop_size: int,
    max_missing_ratio_t0: float,
    max_pos_per_plume: int,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    center_half = center_crop_size // 2
    py, px = nearest_iyix(lat, lon, lat0, lon0)
    height, width = ch4_t0.shape
    kept: List[Tuple[int, int]] = []
    for cy, cx in candidate_pos_centers(py, px, height, width, center_half):
        patch = crop_center(ch4_t0, cy, cx, center_half)
        if patch is None:
            continue
        if missing_ratio(patch) <= max_missing_ratio_t0:
            kept.append((cy, cx))
        if len(kept) >= max_pos_per_plume:
            break
    return py, px, kept


def read_ch4_lat_lon(
    path_nc: str,
    ch4name_hint: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool, Optional[str]]:
    if not valid_text(path_nc):
        return None, None, None, False, ch4name_hint
    path = Path(str(path_nc))
    if not path.exists():
        return None, None, None, False, ch4name_hint

    try:
        with silence_fd2():
            ds = Dataset(str(path), "r")
        try:
            prod = ds.groups["PRODUCT"]
            ch4_name = ch4name_hint if (ch4name_hint and ch4name_hint in prod.variables) else pick_ch4_var(prod)
            if ch4_name is None:
                return None, None, None, False, ch4name_hint
            var = prod.variables[ch4_name]
            arr = to_nan_invalid(var[:], getattr(var, "__dict__", {}))
            lat = get_2d(prod.variables["latitude"][:]).astype(np.float64, copy=False)
            lon = get_2d(prod.variables["longitude"][:]).astype(np.float64, copy=False)
            return arr, lat, lon, True, ch4_name
        finally:
            ds.close()
    except Exception:
        return None, None, None, False, ch4name_hint

def crop_center(a2d: np.ndarray, cy: int, cx: int, half: int) -> Optional[np.ndarray]:
    height, width = a2d.shape
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
        return None
    return a2d[y0:y1, x0:x1]


def missing_ratio(patch2d: np.ndarray) -> float:
    return 1.0 - (np.isfinite(patch2d).sum() / patch2d.size)


def center_lat_lon(
    lat: Optional[np.ndarray],
    lon: Optional[np.ndarray],
    center: Tuple[int, int],
) -> Optional[Tuple[float, float]]:
    if lat is None or lon is None:
        return None
    cy, cx = center
    if cy < 0 or cx < 0 or cy >= lat.shape[0] or cx >= lat.shape[1]:
        return None
    lat0 = float(lat[cy, cx])
    lon0 = float(lon[cy, cx])
    if not (np.isfinite(lat0) and np.isfinite(lon0)):
        return None
    return lat0, lon0


def nan_out(out_size: int) -> np.ndarray:
    return np.full((out_size, out_size), np.nan, dtype=np.float32)


def resize_nan_aware(src2d: np.ndarray, out_size: int) -> np.ndarray:
    src = src2d.astype(np.float32, copy=False)
    finite = np.isfinite(src)
    values = np.where(finite, src, 0.0).astype(np.float32)
    weights = finite.astype(np.float32)

    values_r = cv2.resize(values, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    weights_r = cv2.resize(weights, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(weights_r > 1e-6, values_r / weights_r, np.nan).astype(np.float32)


def parse_centers(value) -> List[Tuple[int, int]]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return []
    centers: List[Tuple[int, int]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        cy, cx = part.split(",")
        centers.append((int(float(cy)), int(float(cx))))
    return centers


def outside_exclude_11x11(py: int, px: int, cy: int, cx: int, neg_exclude_half: int) -> bool:
    return not (abs(cy - py) <= neg_exclude_half and abs(cx - px) <= neg_exclude_half)


def find_neg_center(
    ch4_t0: np.ndarray,
    py: int,
    px: int,
    crop_half: int,
    max_missing_ratio_t0: float,
    neg_exclude_half: int,
    random_tries: int,
    seed: int,
) -> Optional[Tuple[int, int]]:
    height, width = ch4_t0.shape
    y_min, y_max = crop_half, height - crop_half - 1
    x_min, x_max = crop_half, width - crop_half - 1
    if y_min > y_max or x_min > x_max:
        return None

    corners = [(y_min, x_min), (y_min, x_max), (y_max, x_min), (y_max, x_max)]
    corners.sort(key=lambda c: (c[0] - py) ** 2 + (c[1] - px) ** 2, reverse=True)

    for cy, cx in corners:
        if not outside_exclude_11x11(py, px, cy, cx, neg_exclude_half):
            continue
        patch = crop_center(ch4_t0, cy, cx, crop_half)
        if patch is not None and missing_ratio(patch) <= max_missing_ratio_t0:
            return cy, cx

    rng = random.Random(seed)
    for _ in range(random_tries):
        cy = rng.randint(y_min, y_max)
        cx = rng.randint(x_min, x_max)
        if not outside_exclude_11x11(py, px, cy, cx, neg_exclude_half):
            continue
        patch = crop_center(ch4_t0, cy, cx, crop_half)
        if patch is not None and missing_ratio(patch) <= max_missing_ratio_t0:
            return cy, cx
    return None


def iter_neg_center_candidates(
    ch4_t0: np.ndarray,
    py: int,
    px: int,
    crop_half: int,
    max_missing_ratio_t0: float,
    neg_exclude_half: int,
    search_half: int,
    random_tries: int,
    seed: int,
):
    height, width = ch4_t0.shape
    y_min, y_max = crop_half, height - crop_half - 1
    x_min, x_max = crop_half, width - crop_half - 1
    if y_min > y_max or x_min > x_max:
        return

    seen = set()

    def maybe_yield(cy: int, cx: int):
        if (cy, cx) in seen:
            return None
        seen.add((cy, cx))
        if cy < y_min or cy > y_max or cx < x_min or cx > x_max:
            return None
        if not outside_exclude_11x11(py, px, cy, cx, neg_exclude_half):
            return None
        patch = crop_center(ch4_t0, cy, cx, crop_half)
        if patch is None or missing_ratio(patch) > max_missing_ratio_t0:
            return None
        return cy, cx

    # Keep the old pipeline's far-corner preference first.
    corners = [(y_min, x_min), (y_min, x_max), (y_max, x_min), (y_max, x_max)]
    corners.sort(key=lambda c: (c[0] - py) ** 2 + (c[1] - px) ** 2, reverse=True)
    for cy, cx in corners:
        candidate = maybe_yield(cy, cx)
        if candidate is not None:
            yield candidate

    # Six-time products only overlap reliably near the plume. Search outward
    # from the 11x11 exclusion zone, then run randomized local trials.
    search_half = max(int(search_half), int(neg_exclude_half) + 1)
    rng = random.Random(seed)
    angle_count = 16
    for radius in range(int(neg_exclude_half) + 1, search_half + 1):
        coords = [
            (
                py + int(round(radius * np.sin(2 * np.pi * k / angle_count))),
                px + int(round(radius * np.cos(2 * np.pi * k / angle_count))),
            )
            for k in range(angle_count)
        ]
        rng.shuffle(coords)
        for cy, cx in coords:
            candidate = maybe_yield(cy, cx)
            if candidate is not None:
                yield candidate

    for _ in range(random_tries):
        cy = rng.randint(max(y_min, py - search_half), min(y_max, py + search_half))
        cx = rng.randint(max(x_min, px - search_half), min(x_max, px + search_half))
        candidate = maybe_yield(cy, cx)
        if candidate is not None:
            yield candidate


def empty_crop_info() -> Dict[str, object]:
    return {
        "center_iy": np.nan,
        "center_ix": np.nan,
        "center_distance_km": np.nan,
        "missing_ratio": 1.0,
    }


def crop_time_stack_geoaligned(
    arrays: Dict[str, Optional[np.ndarray]],
    latlons: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    target_lat: float,
    target_lon: float,
    crop_half: int,
    out_size: int,
    max_missing_ratio: float,
    max_center_distance_km: float,
    forced_centers: Optional[Dict[str, Tuple[int, int]]] = None,
    search_centers: Optional[Dict[str, Tuple[int, int]]] = None,
    search_radius: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, bool], Dict[str, Dict[str, object]]]:
    forced_centers = forced_centers or {}
    search_centers = search_centers or {}
    channels: List[np.ndarray] = []
    has_data: Dict[str, bool] = {}
    crop_info: Dict[str, Dict[str, object]] = {}

    for tp in TIMEPOINTS:
        arr = arrays.get(tp)
        lat, lon = latlons.get(tp, (None, None))
        info = empty_crop_info()
        patch = None

        if arr is not None and lat is not None and lon is not None:
            try:
                if tp in forced_centers:
                    cy, cx = forced_centers[tp]
                    target_cell = center_lat_lon(lat, lon, (cy, cx))
                    dist_km = np.nan if target_cell is None else approx_distance_km(
                        target_cell[0], target_cell[1], target_lat, target_lon
                    )
                elif tp in search_centers and search_radius is not None:
                    cy, cx, dist_km = nearest_iyix_distance_window(
                        lat,
                        lon,
                        target_lat,
                        target_lon,
                        search_centers[tp],
                        int(search_radius),
                    )
                else:
                    cy, cx, dist_km = nearest_iyix_distance(lat, lon, target_lat, target_lon)

                patch = crop_center(arr, cy, cx, crop_half)
                miss = 1.0 if patch is None else missing_ratio(patch)
                info = {
                    "center_iy": int(cy),
                    "center_ix": int(cx),
                    "center_distance_km": float(dist_km),
                    "missing_ratio": float(miss),
                }
                if (not np.isfinite(dist_km)) or dist_km > max_center_distance_km or miss > max_missing_ratio:
                    patch = None
            except Exception:
                patch = None

        if patch is None:
            channels.append(nan_out(out_size))
            has_data[tp] = False
        else:
            channels.append(resize_nan_aware(patch, out_size))
            has_data[tp] = True
        crop_info[tp] = info

    return np.stack(channels, axis=0).astype(np.float32), has_data, crop_info


def add_timepoint_columns(
    row_out: Dict[str, object],
    paths: Dict[str, str],
    has_data: Dict[str, bool],
    crop_info: Dict[str, Dict[str, object]],
) -> None:
    for tp in TIMEPOINTS:
        info = crop_info.get(tp, empty_crop_info())
        row_out[f"has_{tp}"] = bool(has_data.get(tp, False))
        row_out[f"{tp}_path"] = paths[tp]
        row_out[f"{tp}_center_iy"] = info["center_iy"]
        row_out[f"{tp}_center_ix"] = info["center_ix"]
        row_out[f"{tp}_center_distance_km"] = info["center_distance_km"]
        row_out[f"{tp}_missing_ratio"] = info["missing_ratio"]


def make_sample_row(
    *,
    plume_id: str,
    plume_time: str,
    lat0: float,
    lon0: float,
    ch4_used: Optional[str],
    image_path: Path,
    label: int,
    center_iy: int,
    center_ix: int,
    paths: Dict[str, str],
    has_data: Dict[str, bool],
    crop_info: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    missing = [tp for tp in TIMEPOINTS if not has_data.get(tp, False)]
    row_out = {
        "plume_id": plume_id,
        "plume_time": plume_time,
        "lat": lat0,
        "lon": lon0,
        "ch4_var": ch4_used,
        "image_path": str(image_path),
        "center_iy": center_iy,
        "center_ix": center_ix,
        "label": int(label),
        "channels": ",".join(TIMEPOINTS),
        "missing_timepoints": ",".join(missing),
    }
    add_timepoint_columns(row_out, paths, has_data, crop_info)
    return row_out


def required_timepoint_tuple(cfg: Dict[str, object]) -> Tuple[str, ...]:
    return tuple(TIMEPOINTS) if cfg.get("require_complete", False) else tuple(cfg.get("required_timepoints", ("t0",)))


def validate_required_timepoints(has_data: Dict[str, bool], cfg: Dict[str, object], *, prefix: str) -> None:
    missing_required = [tp for tp in required_timepoint_tuple(cfg) if not has_data.get(tp, False)]
    if missing_required:
        raise RuntimeError(f"{prefix}:{','.join(missing_required)}")


def process_one(task) -> List[Dict[str, object]]:
    idx, row_dict, pos_base, neg_base, cfg = task
    row = pd.Series(row_dict)
    crop_half = cfg["crop_size"] // 2

    plume_id = str(row["plume_id"])
    plume_time = str(row.get("plume_time", row.get("event_time", "")))
    lat0 = float(row.get("lat", row.get("plume_latitude")))
    lon0 = float(row.get("lon", row.get("plume_longitude")))

    paths = {tp: get_path(row, tp) for tp in TIMEPOINTS}
    ch4_hint = row.get("ch4_var", None) if valid_text(row.get("ch4_var", None)) else None

    arrays: Dict[str, Optional[np.ndarray]] = {}
    latlons: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    ch4_used = ch4_hint
    for tp in TIMEPOINTS:
        arr, lat, lon, is_ok, ch4_used = read_ch4_lat_lon(paths[tp], ch4_used)
        arrays[tp] = arr if is_ok else None
        latlons[tp] = (lat, lon) if is_ok else (None, None)

    out_rows: List[Dict[str, object]] = []
    pos_dir = Path(pos_base) / plume_id
    neg_dir = Path(neg_base) / plume_id
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    pos_stack, pos_has_data, pos_crop_info = crop_time_stack_geoaligned(
        arrays=arrays,
        latlons=latlons,
        target_lat=lat0,
        target_lon=lon0,
        crop_half=crop_half,
        out_size=cfg["out_size"],
        max_missing_ratio=cfg["max_missing_ratio_timepoint"],
        max_center_distance_km=cfg["max_center_distance_km"],
    )
    validate_required_timepoints(pos_has_data, cfg, prefix="incomplete_required_pos_timepoints")

    t0_info = pos_crop_info.get("t0", empty_crop_info())
    center_iy = int(t0_info["center_iy"])
    center_ix = int(t0_info["center_ix"])
    pos_npz = pos_dir / "s5p_pos_00.npz"
    if not cfg.get("count_only", False) and not cfg.get("negative_only", False):
        np.savez_compressed(
            pos_npz,
            ch4=pos_stack,
            meta=np.array(
                {
                    "label": 1,
                    "plume_id": plume_id,
                    "ch4_var": ch4_used,
                    "channels": TIMEPOINTS,
                    "center_iy": center_iy,
                    "center_ix": center_ix,
                    "nearest_iy": center_iy,
                    "nearest_ix": center_ix,
                    "target_lat": float(lat0),
                    "target_lon": float(lon0),
                    "paths": paths,
                    "has_timepoint": pos_has_data,
                    "crop_info": pos_crop_info,
                    "alignment": "per_timepoint_plume_latlon_nearest",
                },
                dtype=object,
            ),
        )
    if not cfg.get("negative_only", False):
        out_rows.append(
            make_sample_row(
                plume_id=plume_id,
                plume_time=plume_time,
                lat0=lat0,
                lon0=lon0,
                ch4_used=ch4_used,
                image_path=pos_npz,
                label=1,
                center_iy=center_iy,
                center_ix=center_ix,
                paths=paths,
                has_data=pos_has_data,
                crop_info=pos_crop_info,
            )
        )

    t0_arr = arrays.get("t0")
    t0_lat, t0_lon = latlons.get("t0", (None, None))
    if t0_arr is None or t0_lat is None or t0_lon is None:
        raise RuntimeError("open_t0_fail")
    neg_search_centers = {}
    for tp in TIMEPOINTS:
        info = pos_crop_info.get(tp, empty_crop_info())
        if np.isfinite(info.get("center_iy", np.nan)) and np.isfinite(info.get("center_ix", np.nan)):
            neg_search_centers[tp] = (int(info["center_iy"]), int(info["center_ix"]))
    neg_search_radius = int(cfg["neg_search_half"]) + int(cfg["neg_search_margin"])
    last_neg_error = "no_valid_neg_center"
    neg_payload = None
    for neg_center in iter_neg_center_candidates(
        t0_arr,
        center_iy,
        center_ix,
        crop_half,
        cfg["max_missing_ratio_t0"],
        cfg["neg_exclude_half"],
        cfg["neg_search_half"],
        cfg["neg_random_tries"],
        seed=idx * 1000,
    ):
        neg_target = center_lat_lon(t0_lat, t0_lon, neg_center)
        if neg_target is None:
            last_neg_error = "neg_target_latlon_invalid"
            continue
        neg_lat, neg_lon = neg_target
        neg_stack, neg_has_data, neg_crop_info = crop_time_stack_geoaligned(
            arrays=arrays,
            latlons=latlons,
            target_lat=neg_lat,
            target_lon=neg_lon,
            crop_half=crop_half,
            out_size=cfg["out_size"],
            max_missing_ratio=cfg["max_missing_ratio_timepoint"],
            max_center_distance_km=cfg["max_center_distance_km"],
            forced_centers={"t0": neg_center},
            search_centers=neg_search_centers,
            search_radius=neg_search_radius,
        )
        missing_required = [tp for tp in required_timepoint_tuple(cfg) if not neg_has_data.get(tp, False)]
        if missing_required:
            last_neg_error = f"incomplete_required_neg_timepoints:{','.join(missing_required)}"
            continue
        neg_payload = (neg_center, neg_lat, neg_lon, neg_stack, neg_has_data, neg_crop_info)
        break

    if neg_payload is None:
        raise RuntimeError(last_neg_error)

    neg_center, neg_lat, neg_lon, neg_stack, neg_has_data, neg_crop_info = neg_payload
    neg_center_iy, neg_center_ix = neg_center
    neg_npz = neg_dir / "s5p_neg_00.npz"
    if not cfg.get("count_only", False):
        np.savez_compressed(
            neg_npz,
            ch4=neg_stack,
            meta=np.array(
                {
                    "label": 0,
                    "plume_id": plume_id,
                    "ch4_var": ch4_used,
                    "channels": TIMEPOINTS,
                    "center_iy": int(neg_center_iy),
                    "center_ix": int(neg_center_ix),
                    "nearest_iy": int(center_iy),
                    "nearest_ix": int(center_ix),
                    "target_lat": float(neg_lat),
                    "target_lon": float(neg_lon),
                    "plume_lat": float(lat0),
                    "plume_lon": float(lon0),
                    "paths": paths,
                    "has_timepoint": neg_has_data,
                    "crop_info": neg_crop_info,
                    "alignment": "negative_t0_latlon_geoaligned_per_timepoint",
                    "negative_exclude_half": int(cfg["neg_exclude_half"]),
                    "negative_search_half": int(cfg["neg_search_half"]),
                },
                dtype=object,
            ),
        )
    out_rows.append(
        make_sample_row(
            plume_id=plume_id,
            plume_time=plume_time,
            lat0=lat0,
            lon0=lon0,
            ch4_used=ch4_used,
            image_path=neg_npz,
            label=0,
            center_iy=int(neg_center_iy),
            center_ix=int(neg_center_ix),
            paths=paths,
            has_data=neg_has_data,
            crop_info=neg_crop_info,
        )
    )
    return out_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Crop six-time S5P CH4 npz samples.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_IN_CSV)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=3, help="Legacy S5P crop size used for final samples.")
    parser.add_argument("--center-crop-size", type=int, default=5, help="Window used to validate candidate positive centers when centers are not precomputed.")
    parser.add_argument("--out-size", type=int, default=224)
    parser.add_argument("--max-missing-ratio-t0", type=float, default=0.50)
    parser.add_argument("--max-missing-ratio-timepoint", type=float, default=0.50)
    parser.add_argument("--max-center-distance-km", type=float, default=25.0)
    parser.add_argument("--max-pos-per-plume", type=int, default=8)
    parser.add_argument("--require-complete", action="store_true", help="Drop rows unless all six S5P timepoints pass distance/missing checks.")
    parser.add_argument("--required-timepoints", default="t0", help="Comma-separated timepoints required to keep a sample. Use t0,seasonal,year for 3-time S5P.")
    parser.add_argument("--neg-exclude-half", type=int, default=5)
    parser.add_argument("--neg-search-half", type=int, default=50, help="Local t0 search half-window for six-time negative candidates.")
    parser.add_argument("--neg-search-margin", type=int, default=25, help="Extra per-timepoint nearest-search margin around each positive center.")
    parser.add_argument("--neg-random-tries", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mp-context", choices=["spawn", "forkserver", "fork"], default="spawn", help="Multiprocessing start method. spawn is safer for netCDF4/HDF5.")
    parser.add_argument("--inflight-multiplier", type=int, default=4, help="Submit at most workers*N pending tasks to avoid huge broken-pool fanout.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--count-only", action="store_true", help="Only count/write CSV rows; do not write npz files.")
    parser.add_argument("--negative-only", action="store_true", help="Only write negative samples; useful when positive crops already exist.")
    parser.add_argument("--keep-existing-csv", type=Path, default=None, help="Existing correct S5P crop CSV to keep and merge into output.")
    parser.add_argument("--skip-kept-plumes", dest="skip_kept_plumes", action="store_true", default=True)
    parser.add_argument("--no-skip-kept-plumes", dest="skip_kept_plumes", action="store_false")
    parser.add_argument("--append-kept-to-output", dest="append_kept_to_output", action="store_true", default=True)
    parser.add_argument("--no-append-kept-to-output", dest="append_kept_to_output", action="store_false")
    args = parser.parse_args()
    if args.crop_size % 2 != 1:
        raise SystemExit("--crop-size must be odd")
    if args.center_crop_size % 2 != 1:
        raise SystemExit("--center-crop-size must be odd")
    required_timepoints = {tp.strip() for tp in args.required_timepoints.split(",") if tp.strip()}
    bad = sorted(required_timepoints - set(TIMEPOINTS))
    if bad:
        raise SystemExit(f"--required-timepoints contains unknown timepoints: {bad}")
    return args


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv, low_memory=False)
    input_plumes = len(df)
    kept_df = pd.DataFrame()
    kept_ids: set[str] = set()
    if args.keep_existing_csv is not None and args.keep_existing_csv.exists() and args.keep_existing_csv.stat().st_size > 0:
        kept_df = pd.read_csv(args.keep_existing_csv, low_memory=False)
        if "plume_id" in kept_df.columns:
            kept_ids = {str(value).strip() for value in kept_df["plume_id"] if valid_text(value)}
    if kept_ids and args.skip_kept_plumes:
        df = df[~df["plume_id"].astype(str).isin(kept_ids)].copy()

    required = ["plume_id"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing col {col}")
    if not ({"lat", "lon"} <= set(df.columns) or {"plume_latitude", "plume_longitude"} <= set(df.columns)):
        raise RuntimeError("Missing coordinate columns: need lat/lon or plume_latitude/plume_longitude")
    if args.limit:
        df = df.head(args.limit).copy()

    pos_dir = args.out_root / "samples2" / "pos"
    neg_dir = args.out_root / "samples2" / "neg"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "crop_size": int(args.crop_size),
        "center_crop_size": int(args.center_crop_size),
        "out_size": int(args.out_size),
        "max_missing_ratio_t0": float(args.max_missing_ratio_t0),
        "max_missing_ratio_timepoint": float(args.max_missing_ratio_timepoint),
        "max_center_distance_km": float(args.max_center_distance_km),
        "max_pos_per_plume": int(args.max_pos_per_plume),
        "require_complete": bool(args.require_complete),
        "required_timepoints": tuple(tp.strip() for tp in args.required_timepoints.split(",") if tp.strip()),
        "count_only": bool(args.count_only),
        "negative_only": bool(args.negative_only),
        "neg_exclude_half": int(args.neg_exclude_half),
        "neg_search_half": int(args.neg_search_half),
        "neg_search_margin": int(args.neg_search_margin),
        "neg_random_tries": int(args.neg_random_tries),
    }
    rows = df.to_dict("records")
    tasks = [(i, rows[i], str(pos_dir), str(neg_dir), cfg) for i in range(len(rows))]

    all_samples: List[Dict[str, object]] = []
    err_cnt: Counter = Counter()
    done = 0

    def report_progress() -> None:
        if done % args.print_every == 0 or done == len(tasks):
            pct = 100.0 * done / max(len(tasks), 1)
            print(
                f"[{done}/{len(tasks)} | {pct:5.1f}%] samples={len(all_samples)} "
                f"errors={sum(err_cnt.values())}",
                flush=True,
            )

    if args.workers <= 1:
        for task in tasks:
            done += 1
            try:
                all_samples.extend(process_one(task))
            except Exception as exc:
                err_cnt[str(exc)] += 1
            report_progress()
    else:
        ctx = mp.get_context(args.mp_context)
        max_inflight = max(1, int(args.workers) * max(1, int(args.inflight_multiplier)))
        next_task = 0
        futures: Dict[object, int] = {}

        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
                while next_task < len(tasks) and len(futures) < max_inflight:
                    futures[executor.submit(process_one, tasks[next_task])] = next_task
                    next_task += 1

                while futures:
                    finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        futures.pop(fut)
                        done += 1
                        try:
                            all_samples.extend(fut.result())
                        except BrokenProcessPool as exc:
                            raise RuntimeError(
                                "S5P worker process died abruptly, likely from netCDF4/HDF5 multiprocessing "
                                "or memory pressure. Re-run with --workers 1, or lower workers, and keep "
                                "--mp-context spawn."
                            ) from exc
                        except Exception as exc:
                            err_cnt[str(exc)] += 1
                        report_progress()

                    while next_task < len(tasks) and len(futures) < max_inflight:
                        futures[executor.submit(process_one, tasks[next_task])] = next_task
                        next_task += 1
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "S5P process pool broke before a task result returned. Re-run with --workers 1, "
                "or lower workers, and keep --mp-context spawn."
            ) from exc

    out = pd.DataFrame(all_samples)
    base_cols = ["plume_id", "plume_time", "lat", "lon", "ch4_var", "image_path", "center_iy", "center_ix", "label", "channels", "missing_timepoints"]
    time_cols: List[str] = []
    for tp in TIMEPOINTS:
        time_cols.extend([
            f"has_{tp}",
            f"{tp}_path",
            f"{tp}_center_iy",
            f"{tp}_center_ix",
            f"{tp}_center_distance_km",
            f"{tp}_missing_ratio",
        ])
    cols = [c for c in base_cols + time_cols if c in out.columns]
    out = out[cols] if cols else out
    kept_count = 0
    if args.keep_existing_csv is not None and args.append_kept_to_output and not kept_df.empty:
        kept_count = len(kept_df)
        out = pd.concat([kept_df, out], ignore_index=True, sort=False)
        if "image_path" in out.columns:
            out = out.drop_duplicates("image_path", keep="first")
        elif {"plume_id", "label", "center_iy", "center_ix"} <= set(out.columns):
            out = out.drop_duplicates(["plume_id", "label", "center_iy", "center_ix"], keep="first")
        preferred = [c for c in base_cols + time_cols if c in out.columns]
        remaining = [c for c in out.columns if c not in preferred]
        out = out[preferred + remaining]
    out.to_csv(args.out_csv, index=False)

    print(f"saved: {args.out_csv}")
    print(f"input_plumes: {input_plumes}")
    print(f"kept_existing_samples: {kept_count}")
    print(f"processed_plumes: {len(rows)}")
    print(f"total_samples: {len(out)}")
    if "label" in out.columns:
        print(f"label_counts: {out['label'].value_counts(dropna=False).to_dict()}")
    print(f"top_errors: {err_cnt.most_common(10)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
