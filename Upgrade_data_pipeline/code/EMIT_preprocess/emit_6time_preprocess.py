#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PIPELINE_ROOT.parent
DEFAULT_INPUT_CSV = PIPELINE_ROOT / "csv" / "emit_6time_merged_paths.csv"
DEFAULT_512_ROOT = Path("/mnt/engg-niulab/Yuyao/preprocessed_512/emit")
DEFAULT_512_CSV = DEFAULT_512_ROOT / "emit_6time_512_manifest.csv"
DEFAULT_CROP_ROOT = Path("/mnt/engg-niulab/Yuyao/final_crop/emit")
DEFAULT_MASK_ROOTS = [
    Path("/mnt/engg-niulab/Yuyao/sensors_raw_data/CM"),
    Path("/mnt/engg-niulab/yuyao/sensors_raw_data/CM"),
    Path("/data2/yuyao/methane_emission/carbon_mapper_data_masks"),
]

WINDOW_SIZE = 512
SCALE_M = 60.0
PATCH_SIZE = 16
TARGET_SIZE = 224
CENTER_BOX = 6
MISSING_THRESH = 0.25
N_POS = 16
N_NEG = 16
DEFAULT_BAND_COUNT = 16
DEFAULT_DISTANCE_UPPER_BOUND_DEG = 0.0015


@dataclass(frozen=True)
class Timepoint:
    name: str
    npz_col: str
    path_col: str
    filename: str
    patch_name: str


TIMEPOINTS = [
    Timepoint("t0", "t0_npz_path", "t0_512_path", "emit_t0_512.tif", "emit_t0.tif"),
    Timepoint("prev1", "prev1_npz_path", "prev1_512_path", "emit_prev1_512.tif", "emit_prev1.tif"),
    Timepoint("prev2", "prev2_npz_path", "prev2_512_path", "emit_prev2_512.tif", "emit_prev2.tif"),
    Timepoint("prev3", "prev3_npz_path", "prev3_512_path", "emit_prev3_512.tif", "emit_prev3.tif"),
    Timepoint("seasonal", "seasonal_npz_path", "seasonal_512_path", "emit_seasonal_512.tif", "emit_seasonal.tif"),
    Timepoint("year", "year_npz_path", "year_512_path", "emit_year_512.tif", "emit_year.tif"),
]


def lazy_import_pandas():
    import pandas as pd

    return pd


def lazy_import_geo():
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from scipy.spatial import cKDTree

    return np, rasterio, Transformer, Affine, Resampling, reproject, cKDTree


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def path_exists(value: Any) -> bool:
    return has_value(value) and Path(str(value)).exists()


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def ensure_columns(df):
    for tp in TIMEPOINTS:
        for col in (tp.path_col, f"has_{tp.name}"):
            if col not in df.columns:
                df[col] = ""
    for col in ("std_ok", "bug", "emit_mask_512_path", "mask_path", "crop_ok"):
        if col not in df.columns:
            df[col] = ""
    return df


def first_existing_path(row: dict[str, Any], *cols: str) -> str:
    for col in cols:
        value = row.get(col, "")
        if path_exists(value):
            return str(value)
    return ""


def expected_512_path(out_root: Path, plume_id: str, tp: Timepoint) -> Path:
    return out_root / sanitize_filename(plume_id) / tp.filename


def expected_mask_path(out_root: Path, plume_id: str) -> Path:
    return out_root / sanitize_filename(plume_id) / "mask_60m_512.tif"


@lru_cache(maxsize=128)
def transformers_for_utm(lat_band: int, lon_band: int):
    _, _, Transformer, *_ = lazy_import_geo()
    lat = lat_band / 1000.0
    lon = lon_band / 1000.0
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    crs = f"EPSG:{epsg}"
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return crs, to_utm, to_wgs


def get_utm_for_center(lat: float, lon: float):
    return transformers_for_utm(int(round(lat * 1000)), int(round(lon * 1000)))


def npz_to_emit_512(
    npz_path: str,
    out_path: Path,
    center_lat: float,
    center_lon: float,
    band_count: int,
    overwrite: bool,
    distance_upper_bound_deg: float,
) -> tuple[bool, str]:
    np, rasterio, _, Affine, _, _, cKDTree = lazy_import_geo()
    src = Path(npz_path)
    if not src.exists():
        return False, "missing_npz"
    if out_path.exists() and not overwrite:
        return True, "existing"

    try:
        with np.load(src) as data:
            refl = data["reflectance_ch4"].astype("float32", copy=False)
            lat = data["lat"].astype("float64", copy=False)
            lon = data["lon"].astype("float64", copy=False)
            wavelengths = data["wavelengths_nm"].astype("float32", copy=False) if "wavelengths_nm" in data.files else None
            band_indices = data["band_indices"].astype("int16", copy=False) if "band_indices" in data.files else None

        if refl.ndim != 3:
            return False, f"bad_reflectance_shape:{refl.shape}"
        if refl.shape[:2] != lat.shape or refl.shape[:2] != lon.shape:
            return False, f"lat_lon_shape_mismatch:{refl.shape},{lat.shape},{lon.shape}"
        if band_count > 0:
            if refl.shape[2] < band_count:
                return False, f"not_enough_bands:{refl.shape[2]}<{band_count}"
            refl = refl[:, :, :band_count]
            if wavelengths is not None:
                wavelengths = wavelengths[:band_count]
            if band_indices is not None:
                band_indices = band_indices[:band_count]

        valid = np.isfinite(lat) & np.isfinite(lon)
        if not np.any(valid):
            return False, "no_valid_lat_lon"
        points = np.column_stack([lat[valid].ravel(), lon[valid].ravel()])
        values = refl[valid, :]
        tree = cKDTree(points)

        crs, to_utm, to_wgs = get_utm_for_center(center_lat, center_lon)
        center_x, center_y = to_utm.transform(center_lon, center_lat)
        half = (WINDOW_SIZE * SCALE_M) / 2.0
        top_left_x = center_x - half
        top_left_y = center_y + half
        xs = top_left_x + (np.arange(WINDOW_SIZE, dtype="float64") + 0.5) * SCALE_M
        ys = top_left_y - (np.arange(WINDOW_SIZE, dtype="float64") + 0.5) * SCALE_M
        mx, my = np.meshgrid(xs, ys)
        target_lon, target_lat = to_wgs.transform(mx, my)
        target_points = np.column_stack([target_lat.ravel(), target_lon.ravel()])
        dist, idx = tree.query(target_points, distance_upper_bound=distance_upper_bound_deg)
        good = np.isfinite(dist) & (idx < values.shape[0])

        bands = refl.shape[2]
        out = np.full((bands, WINDOW_SIZE * WINDOW_SIZE), np.nan, dtype="float32")
        if np.any(good):
            out[:, good] = values[idx[good], :].T
        out = out.reshape(bands, WINDOW_SIZE, WINDOW_SIZE)

        transform = Affine(SCALE_M, 0.0, top_left_x, 0.0, -SCALE_M, top_left_y)
        profile = {
            "driver": "GTiff",
            "height": WINDOW_SIZE,
            "width": WINDOW_SIZE,
            "count": bands,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "nodata": np.nan,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out)
            if wavelengths is not None:
                dst.update_tags(wavelengths_nm=",".join(str(float(x)) for x in wavelengths))
            if band_indices is not None:
                dst.update_tags(source_band_indices=",".join(str(int(x)) for x in band_indices))
            dst.update_tags(source_npz=str(src), source_subset="reflectance_ch4", scale_m=str(SCALE_M))
        return True, "success"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def candidate_source_mask(row: dict[str, Any], roots: list[Path]) -> str:
    for col in ("cm_plume_tif_local_path", "plume_mask_path", "local_plume_tif", "plume_tif_path", "plume_tif"):
        value = row.get(col, "")
        if has_value(value):
            text = str(value).strip()
            if text.startswith("/") and Path(text).exists():
                return text

    plume_id = str(row.get("plume_id", "")).strip()
    for root in roots:
        candidates = [
            root / plume_id / "plume.tif",
            root / plume_id / f"{plume_id}_plume.tif",
            root / f"{plume_id}.tif",
            root / f"{plume_id}_plume.tif",
        ]
        for path in candidates:
            if path.exists() and path.stat().st_size > 0:
                return str(path)
        plume_dir = root / plume_id
        if plume_dir.exists():
            for path in sorted(plume_dir.glob("*plume*.tif")):
                if path.exists() and path.stat().st_size > 0:
                    return str(path)
    return ""


def build_mask_512(row: dict[str, Any], ref_tif: str, mask_roots: list[Path], out_root: Path, overwrite: bool) -> tuple[bool, str, str]:
    np, rasterio, _, _, Resampling, reproject, _ = lazy_import_geo()
    plume_id = str(row.get("plume_id", "")).strip()
    out_path = expected_mask_path(out_root, plume_id)
    if out_path.exists() and not overwrite:
        return True, str(out_path), "existing"
    source_mask = candidate_source_mask(row, mask_roots)
    if not source_mask:
        return False, "", "missing_source_mask"
    if not Path(ref_tif).exists():
        return False, "", "missing_ref_tif"

    try:
        with rasterio.open(ref_tif) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_shape = (ref.height, ref.width)
            if dst_crs is None:
                return False, "", "ref_crs_missing"

        with rasterio.open(source_mask) as src:
            src_arr = src.read(1).astype("float32")
            out = np.zeros(dst_shape, dtype="float32")
            if src.crs is not None:
                reproject(
                    source=src_arr,
                    destination=out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0,
                )
            elif src_arr.shape == dst_shape:
                out = src_arr
            else:
                return False, "", "source_mask_crs_missing"

        mask = ((out > 0) & np.isfinite(out)).astype("uint8")
        profile = {
            "driver": "GTiff",
            "height": dst_shape[0],
            "width": dst_shape[1],
            "count": 1,
            "dtype": "uint8",
            "crs": dst_crs,
            "transform": dst_transform,
            "compress": "lzw",
            "nodata": 0,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask, 1)
            dst.update_tags(source_mask=source_mask)
        return True, str(out_path), "success"
    except Exception as exc:
        return False, "", f"{type(exc).__name__}:{exc}"


def build512_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    plume_id = str(row.get("plume_id", "")).strip()
    result: dict[str, Any] = {"plume_id": plume_id}
    bugs: list[str] = []
    try:
        lat = float(row.get("plume_latitude"))
        lon = float(row.get("plume_longitude"))
    except Exception:
        result["std_ok"] = 0
        result["bug"] = "invalid_lat_lon"
        for tp in TIMEPOINTS:
            result[f"has_{tp.name}"] = 0
        return result

    all_ok = True
    out_root = Path(args.out_root)
    for tp in TIMEPOINTS:
        existing = first_existing_path(row, tp.path_col)
        if existing and not args.overwrite:
            result[tp.path_col] = existing
            result[f"has_{tp.name}"] = 1
            continue

        out_path = expected_512_path(out_root, plume_id, tp)
        if out_path.exists() and not args.overwrite:
            result[tp.path_col] = str(out_path)
            result[f"has_{tp.name}"] = 1
            continue

        npz_path = str(row.get(tp.npz_col, "")).strip() if has_value(row.get(tp.npz_col, "")) else ""
        if not npz_path or not Path(npz_path).exists():
            all_ok = False
            result[tp.path_col] = ""
            result[f"has_{tp.name}"] = 0
            bugs.append(f"{tp.name}:missing_npz")
            continue

        ok, msg = npz_to_emit_512(
            npz_path=npz_path,
            out_path=out_path,
            center_lat=lat,
            center_lon=lon,
            band_count=int(args.band_count),
            overwrite=bool(args.overwrite),
            distance_upper_bound_deg=float(args.distance_upper_bound_deg),
        )
        if ok:
            result[tp.path_col] = str(out_path)
            result[f"has_{tp.name}"] = 1
        else:
            all_ok = False
            result[tp.path_col] = ""
            result[f"has_{tp.name}"] = 0
            bugs.append(f"{tp.name}:{msg}")

    mask_ok = False
    ref_tif = result.get("t0_512_path", "") or first_existing_path(row, "t0_512_path")
    if ref_tif:
        mask_ok, mask_path, mask_msg = build_mask_512(row, ref_tif, [Path(p) for p in args.mask_roots], out_root, args.overwrite)
        result["emit_mask_512_path"] = mask_path if mask_ok else ""
        result["mask_path"] = mask_path if mask_ok else ""
        if not mask_ok:
            bugs.append(f"mask:{mask_msg}")
    else:
        result["emit_mask_512_path"] = ""
        result["mask_path"] = ""
        bugs.append("mask:missing_t0_ref")

    result["std_ok"] = 1 if all_ok and mask_ok else 0
    result["bug"] = "; ".join(bugs)
    return result


def write_updates_by_plume_id(df, updates: list[dict[str, Any]], out_csv: Path) -> None:
    pd = lazy_import_pandas()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not updates:
        tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
        df.to_csv(tmp_csv, index=False)
        tmp_csv.replace(out_csv)
        return
    res_df = pd.DataFrame(updates)
    df = df.set_index("plume_id")
    res_df = res_df.set_index("plume_id")
    for col in res_df.columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")
    df.update(res_df)
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    df.reset_index().to_csv(tmp_csv, index=False)
    tmp_csv.replace(out_csv)


def merge_existing_build512_manifest(df, out_csv: Path, overwrite: bool):
    pd = lazy_import_pandas()
    if overwrite or not out_csv.exists():
        return df, 0
    old = pd.read_csv(out_csv, low_memory=False)
    if "plume_id" not in old.columns:
        return df, 0

    df = df.set_index("plume_id")
    old = old.drop_duplicates("plume_id", keep="last").set_index("plume_id")
    common = df.index.intersection(old.index)
    if common.empty:
        return df.reset_index(), 0

    merge_cols = []
    for tp in TIMEPOINTS:
        merge_cols.extend([tp.path_col, f"has_{tp.name}"])
    merge_cols.extend(["std_ok", "bug", "emit_mask_512_path", "mask_path", "crop_ok"])

    for col in merge_cols:
        if col not in old.columns:
            continue
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")
        values = old.loc[common, col]
        keep = values.notna() & values.astype(str).str.strip().ne("")
        if keep.any():
            df.loc[values.index[keep], col] = values[keep].astype("object")
    return df.reset_index(), len(common)


def run_build512(args: argparse.Namespace) -> int:
    pd = lazy_import_pandas()
    in_csv = Path(args.input_csv)
    out_csv = Path(args.out_csv)
    df = ensure_columns(pd.read_csv(in_csv, low_memory=False))
    df, merged_rows = merge_existing_build512_manifest(df, out_csv, bool(args.overwrite))
    df = ensure_columns(df)
    if args.limit:
        df = df.head(args.limit).copy()
    if args.require_all_npz:
        cols = [tp.npz_col for tp in TIMEPOINTS]
        df = df[df[cols].apply(lambda s: all(has_value(v) for v in s), axis=1)].copy()
    rows = df.to_dict("records")
    print(
        f"build512: rows={len(rows)} input={in_csv} out_root={args.out_root} "
        f"out_csv={out_csv} merged_existing_manifest_rows={merged_rows}",
        flush=True,
    )

    updates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(build512_row, row, args) for row in rows]
        try:
            for idx, fut in enumerate(as_completed(futures), start=1):
                updates.append(fut.result())
                if idx % args.progress_every == 0 or idx == len(futures):
                    ok = sum(1 for item in updates if int(item.get("std_ok", 0)) == 1)
                    write_updates_by_plume_id(df, updates, out_csv)
                    print(
                        f"build512: {idx}/{len(futures)} processed, "
                        f"std_ok_written={ok}, synced={out_csv}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            if updates:
                write_updates_by_plume_id(df, updates, out_csv)
                print(f"build512: interrupted; synced {len(updates)} completed rows to {out_csv}", flush=True)
            for fut in futures:
                fut.cancel()
            raise

    write_updates_by_plume_id(df, updates, out_csv)
    print(f"build512: wrote {out_csv}", flush=True)
    return 0


def read_chw(path: str, expect_bands: int):
    _, rasterio, *_ = lazy_import_geo()
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
    if expect_bands > 0 and arr.shape[0] != expect_bands:
        raise ValueError(f"expected {expect_bands} bands, got {arr.shape[0]}: {path}")
    return arr


def read_mask(path: str):
    _, rasterio, *_ = lazy_import_geo()
    with rasterio.open(path) as src:
        return src.read(1)


def resize_chw(crop, target_size: int):
    import cv2
    import numpy as np

    out = np.empty((crop.shape[0], target_size, target_size), dtype="float32")
    for band_idx in range(crop.shape[0]):
        src = np.nan_to_num(crop[band_idx], nan=0.0).astype("float32", copy=False)
        out[band_idx] = cv2.resize(src, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return out


def resize_mask(mask_crop, target_size: int):
    import cv2
    import numpy as np

    src = np.nan_to_num(mask_crop, nan=0.0).astype("uint8", copy=False)
    return cv2.resize(src, (target_size, target_size), interpolation=cv2.INTER_NEAREST).astype("uint8")


def write_chw_tif(path: Path, arr) -> None:
    import warnings

    _, rasterio, *_ = lazy_import_geo()
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
        "count": int(arr.shape[0]),
        "dtype": str(arr.dtype),
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "nodata": None,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr)


def write_mask_tif(path: Path, mask) -> None:
    import warnings

    _, rasterio, *_ = lazy_import_geo()
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "nodata": 0,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(mask.astype("uint8", copy=False), 1)


def is_valid_crop(crop) -> bool:
    import numpy as np

    if crop.size == 0 or crop.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
        return False
    missing = np.isnan(crop[0]) | (crop[0] == 0)
    return bool(missing.mean() <= MISSING_THRESH)


def write_sample(sample_id: int, label: int, plume_id: str, out_root: Path, images: dict[str, Any], mask, x: int, y: int, row: dict[str, Any]):
    import numpy as np

    crops = {tp.name: images[tp.name][:, y : y + PATCH_SIZE, x : x + PATCH_SIZE] for tp in TIMEPOINTS}
    if not all(is_valid_crop(crop) for crop in crops.values()):
        return None

    out_dir = out_root / f"sample_{sample_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for tp in TIMEPOINTS:
        dst = out_dir / tp.patch_name
        write_chw_tif(dst, resize_chw(crops[tp.name], TARGET_SIZE))
        paths[f"path_{tp.name}"] = str(dst)

    if label == 1:
        mask_up = resize_mask(mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE], TARGET_SIZE)
    else:
        mask_up = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype="uint8")
    mask_path = out_dir / "plume.tif"
    write_mask_tif(mask_path, mask_up)

    rec = {
        "sample_id": sample_id,
        "label": label,
        "plume_id": plume_id,
        "data_path": str(out_dir),
        "mask_path": str(mask_path),
        "crop_x": x,
        "crop_y": y,
        "latitude": row.get("plume_latitude", ""),
        "longitude": row.get("plume_longitude", ""),
        "event_time": row.get("event_time", ""),
    }
    for tp in TIMEPOINTS:
        rec[f"{tp.name}_image_time"] = row.get(f"{tp.name}_image_time", "")
    rec.update(paths)
    return rec


def stable_seed(seed: int, plume_id: str) -> int:
    digest = hashlib.sha1(plume_id.encode("utf-8", "ignore")).hexdigest()
    return seed + int(digest[:8], 16)


def crop_row(row: dict[str, Any], out_root: Path, id_counter: dict[str, int], lock: threading.Lock, args: argparse.Namespace) -> list[dict[str, Any]]:
    plume_id = str(row.get("plume_id", "")).strip()
    try:
        paths = {tp.name: first_existing_path(row, tp.path_col) for tp in TIMEPOINTS}
        if not all(paths.values()):
            return []
        mask_path = first_existing_path(row, "emit_mask_512_path", "mask_path")
        if not mask_path:
            return []
        images = {name: read_chw(path, int(args.expect_bands)) for name, path in paths.items()}
        mask = read_mask(mask_path)
    except Exception as exc:
        print(f"crop224: skip {plume_id}, read error: {exc}", flush=True)
        return []

    rng = random.Random(stable_seed(int(args.seed), plume_id))
    results: list[dict[str, Any]] = []

    def next_id() -> int:
        with lock:
            id_counter["value"] += 1
            return id_counter["value"]

    def valid_at(x: int, y: int) -> bool:
        return all(
            is_valid_crop(images[tp.name][:, y : y + PATCH_SIZE, x : x + PATCH_SIZE])
            for tp in TIMEPOINTS
        )

    pos_min = WINDOW_SIZE // 2 + (CENTER_BOX // 2) - PATCH_SIZE
    pos_max = WINDOW_SIZE // 2 - (CENTER_BOX // 2)
    pos_count = 0
    attempts = 0
    while pos_count < N_POS and attempts < int(args.max_attempts_pos):
        attempts += 1
        x = rng.randint(pos_min, pos_max)
        y = rng.randint(pos_min, pos_max)
        if not valid_at(x, y):
            continue
        sample = write_sample(next_id(), 1, plume_id, out_root, images, mask, x, y, row)
        if sample is not None:
            results.append(sample)
            pos_count += 1

    neg_count = 0
    attempts = 0
    while neg_count < N_NEG and attempts < int(args.max_attempts_neg):
        attempts += 1
        x = rng.randint(0, WINDOW_SIZE - PATCH_SIZE)
        y = rng.randint(0, WINDOW_SIZE - PATCH_SIZE)
        cx = WINDOW_SIZE // 2
        cy = WINDOW_SIZE // 2
        if (x <= cx <= x + PATCH_SIZE) and (y <= cy <= y + PATCH_SIZE):
            continue
        if not valid_at(x, y):
            continue
        sample = write_sample(next_id(), 0, plume_id, out_root, images, mask, x, y, row)
        if sample is not None:
            results.append(sample)
            neg_count += 1
    return results


def get_start_sample_id(out_root: Path, manifest_csv: Path, resume: bool) -> int:
    if not resume:
        return 0
    if manifest_csv.exists():
        pd = lazy_import_pandas()
        df = pd.read_csv(manifest_csv, usecols=["sample_id"])
        if len(df):
            return int(df["sample_id"].max())
    if out_root.exists():
        ids = []
        for path in out_root.iterdir():
            if path.is_dir() and path.name.startswith("sample_"):
                try:
                    ids.append(int(path.name.split("_")[-1]))
                except Exception:
                    pass
        return max(ids) if ids else 0
    return 0


def run_crop224(args: argparse.Namespace) -> int:
    pd = lazy_import_pandas()
    in_csv = Path(args.input_csv)
    out_root = Path(args.out_root)
    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else out_root / "dataset_manifest_temporal.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    df = ensure_columns(pd.read_csv(in_csv, low_memory=False))
    if args.only_complete:
        required = [tp.path_col for tp in TIMEPOINTS] + ["emit_mask_512_path"]
        df = df[df[required].apply(lambda s: all(path_exists(v) for v in s), axis=1)].copy()
    if args.resume and manifest_csv.exists():
        old = pd.read_csv(manifest_csv, usecols=["plume_id"])
        done = set(old["plume_id"].astype(str))
        df = df[~df["plume_id"].astype(str).isin(done)].copy()
    if args.limit:
        df = df.head(args.limit)

    rows = df.to_dict("records")
    id_counter = {"value": get_start_sample_id(out_root, manifest_csv, args.resume)}
    lock = threading.Lock()
    print("crop224: rows={} start_id={} out_root={}".format(len(rows), id_counter["value"], out_root), flush=True)

    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(crop_row, row, out_root, id_counter, lock, args) for row in rows]
        for idx, fut in enumerate(as_completed(futures), start=1):
            batch = fut.result()
            all_results.extend(batch)
            if idx % args.progress_every == 0 or idx == len(futures):
                print(f"crop224: {idx}/{len(futures)} rows, samples={len(all_results)}", flush=True)

    out_df = pd.DataFrame(all_results)
    if args.resume and manifest_csv.exists() and len(out_df):
        old = pd.read_csv(manifest_csv)
        out_df = pd.concat([old, out_df], ignore_index=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(manifest_csv, index=False)
    print(f"crop224: wrote samples={len(all_results)} manifest={manifest_csv}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build six-time EMIT SWIR 512 chips and 16-to-224 crops.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p512 = sub.add_parser("build512", help="Convert six-time EMIT SWIR npz files to centered 512x512 16-band GeoTIFFs and aligned masks.")
    p512.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    p512.add_argument("--out-csv", default=str(DEFAULT_512_CSV))
    p512.add_argument("--out-root", default=str(DEFAULT_512_ROOT))
    p512.add_argument("--mask-roots", nargs="+", default=[str(p) for p in DEFAULT_MASK_ROOTS])
    p512.add_argument("--band-count", type=int, default=DEFAULT_BAND_COUNT, help="Number of SWIR bands to write. Use 0 to keep all downloaded bands.")
    p512.add_argument("--distance-upper-bound-deg", type=float, default=DEFAULT_DISTANCE_UPPER_BOUND_DEG)
    p512.add_argument("--workers", type=int, default=2)
    p512.add_argument("--progress-every", type=int, default=50)
    p512.add_argument("--limit", type=int, default=0)
    p512.add_argument("--overwrite", action="store_true")
    p512.add_argument("--require-all-npz", action="store_true", help="Only process rows where all six npz paths exist.")
    p512.set_defaults(func=run_build512)

    pcrop = sub.add_parser("crop224", help="Crop six 512x512 EMIT TIFFs plus mask into 16x16 patches resized to 224.")
    pcrop.add_argument("--input-csv", default=str(DEFAULT_512_CSV))
    pcrop.add_argument("--out-root", default=str(DEFAULT_CROP_ROOT))
    pcrop.add_argument("--manifest-csv", default="")
    pcrop.add_argument("--expect-bands", type=int, default=DEFAULT_BAND_COUNT)
    pcrop.add_argument("--workers", type=int, default=4)
    pcrop.add_argument("--progress-every", type=int, default=50)
    pcrop.add_argument("--limit", type=int, default=0)
    pcrop.add_argument("--seed", type=int, default=20260706)
    pcrop.add_argument("--max-attempts-pos", type=int, default=100)
    pcrop.add_argument("--max-attempts-neg", type=int, default=200)
    pcrop.add_argument("--resume", action="store_true")
    pcrop.add_argument("--only-complete", action="store_true", default=True)
    pcrop.set_defaults(func=run_crop224)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
