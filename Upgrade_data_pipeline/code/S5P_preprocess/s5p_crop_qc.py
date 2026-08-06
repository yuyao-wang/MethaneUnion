#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from netCDF4 import Dataset


EARTH_RADIUS_KM = 6371.0088
CH4_CANDIDATES = [
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
]


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


def valid_text(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def get_2d(arr: Any) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"unexpected_dims:{arr.shape}")


def to_nan_invalid(raw: Any, var: Any = None) -> np.ndarray:
    if np.ma.isMaskedArray(raw):
        arr = np.ma.filled(raw, np.nan).astype(np.float32, copy=False)
    else:
        arr = np.asarray(raw, dtype=np.float32)

    if var is not None:
        for key in ("_FillValue", "missing_value"):
            if hasattr(var, key):
                try:
                    arr = np.where(arr == np.float32(getattr(var, key)), np.nan, arr)
                except Exception:
                    pass
    arr = np.where(np.abs(arr) > 1e20, np.nan, arr)
    return get_2d(arr)


def pick_ch4_var(product_group: Any) -> Optional[str]:
    for name in CH4_CANDIDATES:
        if name in product_group.variables:
            return name
    return None


def nearest_iyix_distance(
    lat: np.ndarray,
    lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[int, int, float]:
    latr = np.deg2rad(lat.astype(np.float64, copy=False))
    lonr = np.deg2rad(lon.astype(np.float64, copy=False))
    lat0r = math.radians(float(target_lat))
    lon0r = math.radians(float(target_lon))
    dlon = (lonr - lon0r + np.pi) % (2 * np.pi) - np.pi
    x = dlon * np.cos(0.5 * (latr + lat0r))
    y = latr - lat0r
    d2 = x * x + y * y
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    if not np.isfinite(d2).any():
        raise ValueError("no_finite_lat_lon")
    flat = int(np.argmin(d2))
    iy, ix = np.unravel_index(flat, d2.shape)
    return int(iy), int(ix), float(math.sqrt(float(d2[iy, ix])) * EARTH_RADIUS_KM)


def read_ch4_patch(var: Any, cy: int, cx: int, half: int) -> np.ndarray:
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    if len(var.shape) == 3:
        raw = var[0, y0:y1, x0:x1]
    elif len(var.shape) == 2:
        raw = var[y0:y1, x0:x1]
    else:
        raise ValueError(f"unexpected_ch4_dims:{var.shape}")
    return to_nan_invalid(raw, var)


def qc_s5p_product(
    path_nc: str | Path,
    target_lat: float,
    target_lon: float,
    crop_size: int = 3,
    max_missing_ratio: float = 0.50,
    max_center_distance_km: float = 25.0,
) -> dict[str, Any]:
    """Return crop-aware QC for one S5P product at one plume lat/lon."""
    out: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "center_iy": np.nan,
        "center_ix": np.nan,
        "center_distance_km": np.nan,
        "patch_missing_ratio": 1.0,
        "patch_finite_count": 0,
        "patch_total": int(crop_size * crop_size),
        "ch4_var": "",
    }
    if crop_size % 2 != 1:
        out["reason"] = "crop_size_not_odd"
        return out
    if not valid_text(path_nc):
        out["reason"] = "path_empty"
        return out
    path = Path(str(path_nc))
    if not path.exists() or path.stat().st_size <= 0:
        out["reason"] = "path_missing_or_empty"
        return out

    half = crop_size // 2
    try:
        with silence_fd2():
            ds = Dataset(str(path), "r")
        try:
            prod = ds.groups.get("PRODUCT")
            if prod is None:
                out["reason"] = "no_PRODUCT_group"
                return out
            if "latitude" not in prod.variables or "longitude" not in prod.variables:
                out["reason"] = "no_lat_lon"
                return out
            ch4_name = pick_ch4_var(prod)
            if ch4_name is None:
                out["reason"] = "no_ch4_var"
                return out

            lat = get_2d(prod.variables["latitude"][:])
            lon = get_2d(prod.variables["longitude"][:])
            cy, cx, dist_km = nearest_iyix_distance(lat, lon, target_lat, target_lon)
            out.update(
                {
                    "center_iy": int(cy),
                    "center_ix": int(cx),
                    "center_distance_km": float(dist_km),
                    "ch4_var": ch4_name,
                }
            )
            if (not math.isfinite(dist_km)) or dist_km > max_center_distance_km:
                out["reason"] = "distance_gt_threshold"
                return out

            height, width = lat.shape
            y0, y1 = cy - half, cy + half + 1
            x0, x1 = cx - half, cx + half + 1
            if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
                out["reason"] = "crop_oob"
                return out

            patch = read_ch4_patch(prod.variables[ch4_name], cy, cx, half)
            finite = int(np.isfinite(patch).sum())
            missing = 1.0 - finite / float(patch.size)
            out.update(
                {
                    "patch_missing_ratio": float(missing),
                    "patch_finite_count": finite,
                    "patch_total": int(patch.size),
                }
            )
            if missing > max_missing_ratio:
                out["reason"] = "patch_missing_gt_threshold"
                return out
            out["ok"] = True
            out["reason"] = "ok"
            return out
        finally:
            ds.close()
    except Exception as exc:
        out["reason"] = f"open_or_read_fail:{type(exc).__name__}:{str(exc)[:120]}"
        return out
