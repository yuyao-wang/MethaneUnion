#!/usr/bin/env python3
"""Build the six-time S5P center table from the upgraded download manifest.

This is the script version of the center-finding logic that used to live in
preprocess_dataset_s5p/check.ipynb. It reads the upgraded multisensor manifest,
pivots S5P rows to one row per plume, and computes the t0 nearest pixel plus
candidate positive centers from the t0 S5P product.

No imagery is downloaded or cropped here.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
PATH_COLS = {tp: f"s5p_{tp}_path" for tp in TIMEPOINTS}
CROP_OK_STATUSES = {
    "downloaded_crop_ok",
    "skip_existing_crop_ok",
    "master_completed_crop_ok",
    "resume_skip_completed_crop_ok",
}

CH4_CANDIDATES = [
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
]

DEFAULT_MANIFEST = Path("Upgrade_data_pipeline/csv/multisensor_6time_download_manifest.csv")
DEFAULT_OUT_CSV = Path("Upgrade_data_pipeline/csv/s5p_6time_with_centers.csv")

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


def existing_path(*values) -> str:
    """Return the first existing path among the given manifest values."""
    for value in values:
        if not valid_text(value):
            continue
        path = Path(str(value).strip())
        if path.exists():
            return str(path)
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


def missing_ratio(patch2d: np.ndarray) -> float:
    return 1.0 - (np.isfinite(patch2d).sum() / patch2d.size)


def crop_center(a2d: np.ndarray, cy: int, cx: int, half: int) -> Optional[np.ndarray]:
    height, width = a2d.shape
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
        return None
    return a2d[y0:y1, x0:x1]


def pick_ch4_var(product_group) -> Optional[str]:
    for name in CH4_CANDIDATES:
        if name in product_group.variables:
            return name
    return None


def nearest_iyix(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[int, int]:
    lat = lat.astype(np.float64, copy=False)
    lon = lon.astype(np.float64, copy=False)
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    lat0r = math.radians(float(lat0))
    lon0r = math.radians(float(lon0))

    dlon = (lonr - lon0r + np.pi) % (2 * np.pi) - np.pi
    x = dlon * np.cos(0.5 * (latr + lat0r))
    y = latr - lat0r
    d2 = x * x + y * y

    flat = np.nanargmin(d2)
    iy, ix = np.unravel_index(flat, d2.shape)
    return int(iy), int(ix)


def candidate_pos_centers(py: int, px: int, height: int, width: int, crop_half: int) -> List[Tuple[int, int]]:
    centers: List[Tuple[int, int]] = []
    y_min, y_max = crop_half, height - crop_half - 1
    x_min, x_max = crop_half, width - crop_half - 1
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cy = py + dy
            cx = px + dx
            if y_min <= cy <= y_max and x_min <= cx <= x_max:
                centers.append((cy, cx))
    centers.sort(key=lambda c: (c[0] - py) ** 2 + (c[1] - px) ** 2)
    return centers


def encode_centers(centers: Sequence[Tuple[int, int]]) -> str:
    return ";".join(f"{cy},{cx}" for cy, cx in centers)


def compute_centers(
    t0_path: str,
    lat0: float,
    lon0: float,
    center_crop_size: int,
    max_missing_ratio_t0: float,
    max_pos_per_plume: int,
) -> Dict[str, object]:
    crop_half = center_crop_size // 2
    with silence_fd2():
        ds = Dataset(t0_path, "r")
    try:
        prod = ds.groups["PRODUCT"]
        lat = get_2d(prod.variables["latitude"][:])
        lon = get_2d(prod.variables["longitude"][:])

        ch4_name = pick_ch4_var(prod)
        if ch4_name is None:
            raise RuntimeError("no_ch4")

        ch4_var = prod.variables[ch4_name]
        ch4 = to_nan_invalid(ch4_var[:], getattr(ch4_var, "__dict__", {}))
    finally:
        ds.close()

    py, px = nearest_iyix(lat, lon, lat0, lon0)
    height, width = ch4.shape
    kept: List[Tuple[int, int]] = []
    for cy, cx in candidate_pos_centers(py, px, height, width, crop_half):
        patch = crop_center(ch4, cy, cx, crop_half)
        if patch is None:
            continue
        if missing_ratio(patch) <= max_missing_ratio_t0:
            kept.append((cy, cx))
        if len(kept) >= max_pos_per_plume:
            break

    return {
        "nearest_iy": py,
        "nearest_ix": px,
        "pos_centers": encode_centers(kept),
        "ch4_var": ch4_name,
        "t0_ch4_shape": f"{height}x{width}",
        "center_status": "ok",
        "center_error": "",
    }


def parse_timepoints(value: str) -> List[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    bad = [tp for tp in out if tp not in TIMEPOINTS]
    if bad:
        raise ValueError(f"unknown timepoints: {bad}; allowed={TIMEPOINTS}")
    return out


def choose_row_for_timepoint(rows: pd.DataFrame, require_crop_qc: bool) -> Optional[pd.Series]:
    """Prefer crop-aware QC rows with an existing local raw path."""
    if rows.empty:
        return None

    scored = rows.copy()
    scored["_chosen_path"] = scored.apply(
        lambda r: existing_path(r.get("downloaded_path"), r.get("existing_raw_path"), r.get("processed_path")),
        axis=1,
    )
    with_path = scored[scored["_chosen_path"].astype(bool)].copy()
    if with_path.empty:
        return None if require_crop_qc else scored.iloc[-1]

    if require_crop_qc:
        status_col = "download_status" if "download_status" in with_path.columns else "status"
        crop_ok = with_path[with_path[status_col].astype(str).isin(CROP_OK_STATUSES)]
        if crop_ok.empty:
            return None
        return crop_ok.iloc[-1]

    status_col = "download_status" if "download_status" in with_path.columns else "status"
    crop_ok = with_path[with_path[status_col].astype(str).isin(CROP_OK_STATUSES)]
    if not crop_ok.empty:
        return crop_ok.iloc[-1]
    return with_path.iloc[-1]


def build_rows(args) -> Tuple[pd.DataFrame, Dict[str, int]]:
    manifest = pd.read_csv(args.manifest, low_memory=False)
    required = ["plume_id", "sensor", "timepoint", "event_time", "plume_latitude", "plume_longitude"]
    for col in required:
        if col not in manifest.columns:
            raise RuntimeError(f"missing manifest column: {col}")

    timepoints = parse_timepoints(args.timepoints)
    s5p = manifest[manifest["sensor"].astype(str).str.upper() == "S5P"].copy()
    s5p = s5p[s5p["timepoint"].isin(timepoints)].copy()

    rows: List[Dict[str, object]] = []
    stats = {
        "input_s5p_rows": int(len(s5p)),
        "plumes_seen": int(s5p["plume_id"].nunique()),
        "written": 0,
        "skipped_no_t0_path": 0,
        "center_ok": 0,
        "center_failed": 0,
    }

    grouped: Iterable[Tuple[str, pd.DataFrame]] = s5p.groupby("plume_id", sort=False)
    if args.limit:
        grouped = list(grouped)[: args.limit]

    for plume_id, group in grouped:
        by_tp: Dict[str, pd.Series] = {}
        for tp in timepoints:
            rows_tp = group[group["timepoint"] == tp]
            if rows_tp.empty:
                continue
            chosen = choose_row_for_timepoint(rows_tp, args.require_crop_qc)
            if chosen is not None:
                by_tp[tp] = chosen

        if "t0" not in by_tp:
            stats["skipped_no_t0_path"] += 1
            continue

        base = by_tp["t0"]
        t0_path = existing_path(base.get("downloaded_path"), base.get("existing_raw_path"), base.get("processed_path"))
        if not t0_path:
            stats["skipped_no_t0_path"] += 1
            continue

        lat0 = float(base["plume_latitude"])
        lon0 = float(base["plume_longitude"])
        out: Dict[str, object] = {
            "plume_id": plume_id,
            "plume_time": base.get("event_time", ""),
            "lat": lat0,
            "lon": lon0,
            "event_time": base.get("event_time", ""),
            "plume_latitude": lat0,
            "plume_longitude": lon0,
        }

        for tp in TIMEPOINTS:
            row = by_tp.get(tp)
            path = ""
            if row is not None:
                path = existing_path(row.get("downloaded_path"), row.get("existing_raw_path"), row.get("processed_path"))
            out[PATH_COLS[tp]] = path
            out[f"s5p_{tp}_image_time"] = "" if row is None else row.get("image_time", "")
            out[f"s5p_{tp}_product_id"] = "" if row is None else row.get("product_id", "")
            out[f"s5p_{tp}_product_name"] = "" if row is None else row.get("product_name", "")
            out[f"s5p_{tp}_download_status"] = "" if row is None else row.get("download_status", "")
            out[f"s5p_{tp}_selection_source"] = "" if row is None else row.get("selection_source", "")
            out[f"s5p_{tp}_target_time"] = "" if row is None else row.get("target_time", "")
            out[f"s5p_{tp}_time_delta_hours"] = "" if row is None else row.get("time_delta_hours", "")
            out[f"s5p_{tp}_qc_reason"] = "" if row is None else row.get("qc_reason", "")
            out[f"s5p_{tp}_qc_center_distance_km"] = "" if row is None else row.get("qc_center_distance_km", "")
            out[f"s5p_{tp}_qc_patch_missing_ratio"] = "" if row is None else row.get("qc_patch_missing_ratio", "")
            out[f"s5p_{tp}_qc_candidate_rank"] = "" if row is None else row.get("qc_candidate_rank", "")
            out[f"s5p_{tp}_qc_candidates_checked"] = "" if row is None else row.get("qc_candidates_checked", "")

        # Compatibility aliases used by old S5P scripts.
        out["S5p_path"] = out["s5p_t0_path"]
        out["s5p_minus90_path"] = out["s5p_seasonal_path"]
        out["s5p_minus360_path"] = out["s5p_year_path"]

        if args.skip_center_compute:
            out.update(
                {
                    "nearest_iy": np.nan,
                    "nearest_ix": np.nan,
                    "pos_centers": "",
                    "ch4_var": "",
                    "t0_ch4_shape": "",
                    "center_status": "skipped",
                    "center_error": "skip_center_compute",
                }
            )
        else:
            try:
                out.update(
                    compute_centers(
                        t0_path=t0_path,
                        lat0=lat0,
                        lon0=lon0,
                        center_crop_size=args.center_crop_size,
                        max_missing_ratio_t0=args.max_missing_ratio_t0,
                        max_pos_per_plume=args.max_pos_per_plume,
                    )
                )
                stats["center_ok"] += 1
            except Exception as exc:
                if args.strict:
                    raise
                out.update(
                    {
                        "nearest_iy": np.nan,
                        "nearest_ix": np.nan,
                        "pos_centers": "",
                        "ch4_var": "",
                        "t0_ch4_shape": "",
                        "center_status": "failed",
                        "center_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                stats["center_failed"] += 1

        rows.append(out)
        stats["written"] += 1

        if args.progress_every and stats["written"] % args.progress_every == 0:
            print(
                f"[progress] written={stats['written']} center_ok={stats['center_ok']} "
                f"center_failed={stats['center_failed']}",
                flush=True,
            )

    out_df = pd.DataFrame(rows)
    preferred = [
        "plume_id",
        "plume_time",
        "lat",
        "lon",
        "event_time",
        "plume_latitude",
        "plume_longitude",
        "S5p_path",
        "s5p_minus90_path",
        "s5p_minus360_path",
    ]
    for tp in TIMEPOINTS:
        preferred.extend(
            [
                f"s5p_{tp}_path",
                f"s5p_{tp}_image_time",
                f"s5p_{tp}_product_id",
                f"s5p_{tp}_product_name",
                f"s5p_{tp}_download_status",
                f"s5p_{tp}_selection_source",
                f"s5p_{tp}_target_time",
                f"s5p_{tp}_time_delta_hours",
                f"s5p_{tp}_qc_reason",
                f"s5p_{tp}_qc_center_distance_km",
                f"s5p_{tp}_qc_patch_missing_ratio",
                f"s5p_{tp}_qc_candidate_rank",
                f"s5p_{tp}_qc_candidates_checked",
            ]
        )
    preferred.extend(["nearest_iy", "nearest_ix", "pos_centers", "ch4_var", "t0_ch4_shape", "center_status", "center_error"])
    existing = [c for c in preferred if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in existing]
    return out_df[existing + remaining], stats


def parse_args():
    parser = argparse.ArgumentParser(description="Build S5P six-time center CSV from the upgraded manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--timepoints", default=",".join(TIMEPOINTS))
    parser.add_argument("--center-crop-size", type=int, default=5, help="Legacy check.ipynb used 5x5 to filter valid centers.")
    parser.add_argument("--max-missing-ratio-t0", type=float, default=0.50)
    parser.add_argument("--max-pos-per-plume", type=int, default=8)
    parser.add_argument("--require-crop-qc", action="store_true", help="Only use S5P rows with crop-aware QC success statuses.")
    parser.add_argument("--skip-center-compute", action="store_true", help="Do not reopen t0 nc to compute legacy pos_centers; final crop realigns by lat/lon.")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test plume limit.")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.center_crop_size % 2 != 1:
        raise SystemExit("--center-crop-size must be odd")
    return args


def main() -> int:
    args = parse_args()
    out_df, stats = build_rows(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"saved: {args.out_csv}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
