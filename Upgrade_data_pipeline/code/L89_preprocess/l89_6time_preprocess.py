#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import math
import multiprocessing as mp
import os
import random
import re
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "Upgrade_data_pipeline" / "csv").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {here}")


REPO_ROOT = find_repo_root()
DEFAULT_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "l89_6time_complete_paths.csv"
DEFAULT_512_ROOT = Path("/mnt/engg-niulab/Yuyao/preprocessed_512/L89")
DEFAULT_MASK_OUT_ROOT = DEFAULT_512_ROOT / "plume_masks_l89_512"
DEFAULT_PATCH_OUT_ROOT = DEFAULT_512_ROOT / "l89_6time_temporal_16_resized_to_224"
DEFAULT_QC_DIR = REPO_ROOT / "Upgrade_data_pipeline" / "temp"
DEFAULT_CM_MASK_ROOTS = [
    Path("/mnt/engg-niulab/Yuyao/sensors_raw_data/CM"),
    Path("/mnt/engg-niulab/yuyao/sensors_raw_data/CM"),
    Path("/data2/yuyao/methane_emission/carbon_mapper_data_masks"),
]

WINDOW_SIZE = 512
MASK_TARGET_RES = 30
MASK_TARGET_SIZE = 512
PATCH_SIZE = 16
PATCH_TARGET_SIZE = 224
CENTER_BOX = 6
MISSING_THRESH = 0.25
N_POS = 16
N_NEG = 16
MAX_SAMPLE_IDS_PER_ROW = 300


@dataclass(frozen=True)
class Timepoint:
    name: str
    raw_col: str
    path_col: str
    legacy_col: str
    filename: str
    patch_name: str


TIMEPOINTS = [
    Timepoint("t0", "t0_raw_path", "t0_512_path", "l89_0_std_512", "l89_0_std_512.tif", "l89_0.tif"),
    Timepoint("prev1", "prev1_raw_path", "prev1_512_path", "l89_-7_std_512", "l89_-7_std_512.tif", "l89_prev1.tif"),
    Timepoint("prev2", "prev2_raw_path", "prev2_512_path", "l89_prev2_std_512", "l89_prev2_std_512.tif", "l89_prev2.tif"),
    Timepoint("prev3", "prev3_raw_path", "prev3_512_path", "l89_prev3_std_512", "l89_prev3_std_512.tif", "l89_prev3.tif"),
    Timepoint("seasonal", "seasonal_raw_path", "seasonal_512_path", "l89_-90_std_512", "l89_-90_std_512.tif", "l89_seasonal.tif"),
    Timepoint("year", "year_raw_path", "year_512_path", "l89_-360_std_512", "l89_-360_std_512.tif", "l89_year.tif"),
]

SAMPLE_MANIFEST_FIELDS = [
    "id",
    "label",
    "plume_id",
    "path",
    "path_plume",
    "source_x",
    "source_y",
    "latitude",
    "longitude",
    "event_time",
    *[f"{tp.name}_image_time" for tp in TIMEPOINTS],
    *[f"path_{tp.name}" for tp in TIMEPOINTS],
]


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


def lazy_import_pandas():
    import pandas as pd

    return pd


def lazy_import_rasterio():
    import numpy as np
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window

    return np, rasterio, None, Affine, Resampling, reproject, Window


@lru_cache(maxsize=128)
def transformer_for_crs(crs_text: str):
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", crs_text, always_xy=True)


def ensure_pipeline_columns(df):
    for tp in TIMEPOINTS:
        for col in (tp.path_col, tp.legacy_col, f"has_{tp.name}"):
            if col not in df.columns:
                df[col] = ""
    for col in ("std_ok", "bug", "l89_512_mask_path", "mask_path"):
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
    return out_root / plume_id / tp.filename


def process_chip_centered(in_path: str, out_path: Path, lat: float, lon: float) -> tuple[bool, str]:
    np, rasterio, _, _, _, _, Window = lazy_import_rasterio()

    if not Path(in_path).exists():
        return False, "missing_raw"

    try:
        with rasterio.open(in_path) as src:
            if src.crs is None:
                return False, "source_crs_missing"
            transformer = transformer_for_crs(src.crs.to_string())
            target_x, target_y = transformer.transform(lon, lat)

            row, col = src.index(target_x, target_y)
            win = Window(
                col_off=int(round(col - WINDOW_SIZE / 2)),
                row_off=int(round(row - WINDOW_SIZE / 2)),
                width=WINDOW_SIZE,
                height=WINDOW_SIZE,
            )

            arr = src.read(window=win, boundless=True, fill_value=0).astype("float32")
            new_transform = src.window_transform(win)

            nodata_mask = np.sum(arr, axis=0) == 0
            for band_idx in range(arr.shape[0]):
                arr[band_idx, nodata_mask] = np.nan

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=WINDOW_SIZE,
                width=WINDOW_SIZE,
                count=src.count,
                dtype="float32",
                transform=new_transform,
                crs=src.crs,
                compress="deflate",
                predictor=2,
                tiled=True,
                nodata=np.nan,
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr)
        return True, "success"
    except Exception as exc:
        return False, str(exc)


def crop_row_to_512(row: dict[str, Any], out_root: Path, overwrite: bool) -> dict[str, Any]:
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
    for tp in TIMEPOINTS:
        existing = first_existing_path(row, tp.path_col, tp.legacy_col)
        if existing and not overwrite:
            result[tp.path_col] = existing
            result[tp.legacy_col] = existing
            result[f"has_{tp.name}"] = 1
            continue

        out_path = expected_512_path(out_root, plume_id, tp)
        if out_path.exists() and not overwrite:
            result[tp.path_col] = str(out_path)
            result[tp.legacy_col] = str(out_path)
            result[f"has_{tp.name}"] = 1
            continue

        raw_path = str(row.get(tp.raw_col, "")).strip() if has_value(row.get(tp.raw_col, "")) else ""
        if not raw_path or not Path(raw_path).exists():
            all_ok = False
            result[tp.path_col] = ""
            result[tp.legacy_col] = ""
            result[f"has_{tp.name}"] = 0
            bugs.append(f"{tp.name}:missing_raw")
            continue

        ok, msg = process_chip_centered(raw_path, out_path, lat, lon)
        if ok:
            result[tp.path_col] = str(out_path)
            result[tp.legacy_col] = str(out_path)
            result[f"has_{tp.name}"] = 1
        else:
            all_ok = False
            result[tp.path_col] = ""
            result[tp.legacy_col] = ""
            result[f"has_{tp.name}"] = 0
            bugs.append(f"{tp.name}:{msg}")

    result["std_ok"] = 1 if all_ok else 0
    result["bug"] = "; ".join(bugs)
    return result


def write_updates_by_plume_id(df, updates: list[dict[str, Any]], out_csv: Path) -> None:
    pd = lazy_import_pandas()
    if not updates:
        df.to_csv(out_csv, index=False)
        return

    res_df = pd.DataFrame(updates)
    df = df.set_index("plume_id")
    res_df = res_df.set_index("plume_id")
    for col in res_df.columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")
    df.update(res_df)
    df.reset_index().to_csv(out_csv, index=False)


def run_crop512(args: argparse.Namespace) -> int:
    pd = lazy_import_pandas()
    in_csv = Path(args.input_csv)
    out_csv = Path(args.out_csv) if args.out_csv else in_csv
    out_root = Path(args.out_root)

    df = ensure_pipeline_columns(pd.read_csv(in_csv))
    rows = df.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    print(f"crop512: rows={len(rows)} input={in_csv} output_csv={out_csv} out_root={out_root}", flush=True)

    updates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(crop_row_to_512, row, out_root, args.overwrite) for row in rows]
        for idx, fut in enumerate(as_completed(futures), start=1):
            upd = fut.result()
            updates.append(upd)
            if idx % args.progress_every == 0 or idx == len(futures):
                ok = sum(1 for item in updates if int(item.get("std_ok", 0)) == 1)
                print(f"crop512: {idx}/{len(futures)} processed, std_ok_in_batch={ok}", flush=True)

    write_updates_by_plume_id(df, updates, out_csv)
    print(f"crop512: wrote {out_csv}", flush=True)
    return 0


def repair_row_paths(row: dict[str, Any], out_root: Path) -> dict[str, Any]:
    plume_id = str(row.get("plume_id", "")).strip()
    result: dict[str, Any] = {"plume_id": plume_id}
    all_ok = True
    for tp in TIMEPOINTS:
        existing = first_existing_path(row, tp.path_col, tp.legacy_col)
        expected = expected_512_path(out_root, plume_id, tp)
        final_path = existing or (str(expected) if expected.exists() else "")
        result[tp.path_col] = final_path
        result[tp.legacy_col] = final_path
        result[f"has_{tp.name}"] = 1 if final_path else 0
        if not final_path:
            all_ok = False
    result["std_ok"] = 1 if all_ok else 0
    return result


def run_repair512(args: argparse.Namespace) -> int:
    pd = lazy_import_pandas()
    in_csv = Path(args.input_csv)
    out_csv = Path(args.out_csv) if args.out_csv else in_csv
    out_root = Path(args.out_root)

    df = ensure_pipeline_columns(pd.read_csv(in_csv))
    updates = [repair_row_paths(row, out_root) for row in df.to_dict("records")]
    write_updates_by_plume_id(df, updates, out_csv)
    complete = sum(1 for item in updates if int(item.get("std_ok", 0)) == 1)
    print(f"repair512: complete_all6={complete}/{len(updates)} wrote {out_csv}", flush=True)
    return 0


def candidate_source_mask(row: dict[str, Any], roots: list[Path]) -> str:
    for col in ("cm_plume_tif_local_path", "plume_mask_path", "local_plume_tif", "plume_tif_path", "plume_tif"):
        value = row.get(col, "")
        if has_value(value):
            text = str(value).strip()
            if text.startswith("/") and Path(text).exists():
                return text

    plume_id = str(row.get("plume_id", "")).strip()
    for root in roots:
        direct_candidates = [
            root / plume_id / "plume.tif",
            root / plume_id / f"{plume_id}_plume.tif",
            root / f"{plume_id}.tif",
            root / f"{plume_id}_plume.tif",
        ]
        for path in direct_candidates:
            if path.exists() and path.stat().st_size > 0:
                return str(path)
        plume_dir = root / plume_id
        if plume_dir.exists():
            for path in sorted(plume_dir.glob("*plume*.tif")):
                if path.exists() and path.stat().st_size > 0:
                    return str(path)
    return ""


def process_l89_mask(
    plume_id: str,
    center_lat: float,
    center_lon: float,
    ref_tif_path: str,
    source_mask_path: str,
    out_root: Path,
    overwrite: bool,
) -> tuple[bool, str, str]:
    np, rasterio, _, Affine, Resampling, reproject, _ = lazy_import_rasterio()

    dst_path = out_root / plume_id / "mask_30m_512.tif"
    if dst_path.exists() and not overwrite:
        return True, str(dst_path), "existing"
    if not Path(source_mask_path).exists():
        return False, "", f"source_mask_missing:{source_mask_path}"
    if not Path(ref_tif_path).exists():
        return False, "", f"reference_l89_missing:{ref_tif_path}"

    try:
        with rasterio.open(ref_tif_path) as ref:
            if ref.crs is None:
                return False, "", "reference_crs_missing"
            target_crs = ref.crs

        with rasterio.open(source_mask_path) as src:
            if src.height == 0 or src.width == 0:
                return False, "", "empty_source_mask"

            transformer = transformer_for_crs(target_crs.to_string())
            center_x, center_y = transformer.transform(center_lon, center_lat)

            # This intentionally follows the old notebook mask-grid rule. It is
            # not exactly the same as src.index + round(pixel - 256) used for
            # the 512 chip crop, so qc-overlay should be checked after this step.
            center_x = round(center_x / MASK_TARGET_RES) * MASK_TARGET_RES
            center_y = round(center_y / MASK_TARGET_RES) * MASK_TARGET_RES
            top_left_x = center_x - (MASK_TARGET_SIZE // 2) * MASK_TARGET_RES
            top_left_y = center_y + (MASK_TARGET_SIZE // 2) * MASK_TARGET_RES
            dst_transform = Affine(MASK_TARGET_RES, 0, top_left_x, 0, -MASK_TARGET_RES, top_left_y)

            out_mask = np.zeros((1, MASK_TARGET_SIZE, MASK_TARGET_SIZE), dtype="uint8")
            reproject(
                source=rasterio.band(src, 1),
                destination=out_mask[0],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
            )

            profile = {
                "driver": "GTiff",
                "height": MASK_TARGET_SIZE,
                "width": MASK_TARGET_SIZE,
                "count": 1,
                "dtype": "uint8",
                "crs": target_crs,
                "transform": dst_transform,
                "compress": "lzw",
                "nodata": 0,
            }
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(out_mask)
        return True, str(dst_path), "success"
    except Exception as exc:
        return False, "", str(exc)


def mask_row(row: dict[str, Any], mask_roots: list[Path], out_root: Path, overwrite: bool) -> dict[str, Any]:
    plume_id = str(row.get("plume_id", "")).strip()
    result: dict[str, Any] = {"plume_id": plume_id}
    ref_path = first_existing_path(row, "t0_512_path", "l89_0_std_512")
    source_mask = candidate_source_mask(row, mask_roots)
    try:
        lat = float(row.get("plume_latitude"))
        lon = float(row.get("plume_longitude"))
    except Exception:
        result["l89_512_mask_path"] = ""
        result["mask_path"] = ""
        result["mask_bug"] = "invalid_lat_lon"
        return result

    ok, out_path, msg = process_l89_mask(plume_id, lat, lon, ref_path, source_mask, out_root, overwrite)
    result["l89_512_mask_path"] = out_path if ok else ""
    result["mask_path"] = out_path if ok else ""
    result["mask_bug"] = "" if ok else msg
    return result


def run_mask512(args: argparse.Namespace) -> int:
    pd = lazy_import_pandas()
    in_csv = Path(args.input_csv)
    out_csv = Path(args.out_csv) if args.out_csv else in_csv
    mask_roots = [Path(p) for p in args.cm_mask_roots]
    out_root = Path(args.out_root)

    df = ensure_pipeline_columns(pd.read_csv(in_csv))
    rows = df.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    print(f"mask512: rows={len(rows)} mask_out={out_root}", flush=True)

    updates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(mask_row, row, mask_roots, out_root, args.overwrite) for row in rows]
        for idx, fut in enumerate(as_completed(futures), start=1):
            updates.append(fut.result())
            if idx % args.progress_every == 0 or idx == len(futures):
                ok = sum(1 for item in updates if has_value(item.get("mask_path", "")))
                print(f"mask512: {idx}/{len(futures)} processed, mask_ok_in_batch={ok}", flush=True)

    write_updates_by_plume_id(df, updates, out_csv)
    print(f"mask512: wrote {out_csv}", flush=True)
    return 0


def normalize_first_band(path: str):
    np, rasterio, *_ = lazy_import_rasterio()
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
    valid = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        valid &= data != nodata
    valid &= data != 0
    if not np.any(valid):
        return np.zeros_like(data)
    vmin, vmax = np.nanpercentile(data[valid], [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.nan_to_num(data, nan=0.0)
    return (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin + 1e-6)


def read_single_band(path: str):
    np, rasterio, *_ = lazy_import_rasterio()
    if not Path(path).exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def run_qc_overlay(args: argparse.Namespace) -> int:
    pd = lazy_import_pandas()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    in_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    mask_roots = [Path(p) for p in args.cm_mask_roots]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_pipeline_columns(pd.read_csv(in_csv))
    if "std_ok" in df.columns:
        work = df[df["std_ok"].astype(str).isin({"1", "1.0", "True", "true"})].copy()
    else:
        work = df.copy()
    work = work.head(args.samples)

    made = 0
    for _, row in work.iterrows():
        row_dict = row.to_dict()
        plume_id = str(row_dict["plume_id"])
        l89_t0 = first_existing_path(row_dict, "t0_512_path", "l89_0_std_512")
        resized_mask_path = first_existing_path(row_dict, "l89_512_mask_path", "mask_path")
        airborne_mask_path = candidate_source_mask(row_dict, mask_roots)
        if not l89_t0 or not resized_mask_path:
            print(f"qc-overlay: skip {plume_id}, missing t0 512 or mask", flush=True)
            continue

        img_l89 = normalize_first_band(l89_t0)
        img_resized = read_single_band(resized_mask_path)
        img_airborne = read_single_band(airborne_mask_path) if airborne_mask_path else None
        if img_resized is None:
            print(f"qc-overlay: skip {plume_id}, mask read failed", flush=True)
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_airborne if img_airborne is not None else np.zeros((10, 10)), cmap="viridis")
        axes[0].set_title("Original Carbon Mapper mask")
        axes[1].imshow(img_l89, cmap="gray")
        axes[1].axhline(256, color="white", linestyle="--", alpha=0.3)
        axes[1].axvline(256, color="white", linestyle="--", alpha=0.3)
        axes[1].set_title("L89 t0 512 crop")
        axes[2].imshow(img_l89, cmap="gray")
        binary_mask = (img_resized > 0).astype(float)
        axes[2].imshow(np.ma.masked_where(binary_mask == 0, binary_mask), cmap="spring", alpha=0.8)
        axes[2].set_title(f"Overlay, mask pixels={int(np.count_nonzero(img_resized))}")
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        out_path = out_dir / f"l89_6time_mask_overlay_{sanitize_filename(plume_id)}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"qc-overlay: wrote {out_path}", flush=True)
        made += 1

    print(f"qc-overlay: wrote {made} images to {out_dir}", flush=True)
    return 0


def run_qa_stats(args: argparse.Namespace) -> int:
    pd = lazy_import_pandas()
    np, rasterio, *_ = lazy_import_rasterio()

    df = ensure_pipeline_columns(pd.read_csv(args.input_csv))
    sample_path = ""
    for _, row in df.iterrows():
        sample_path = first_existing_path(row.to_dict(), "t0_512_path", "l89_0_std_512")
        if sample_path:
            break
    if not sample_path:
        raise RuntimeError("No existing t0 512 path found.")

    with rasterio.open(sample_path) as src:
        img = src.read().astype("float32")
    print(f"sample_path: {sample_path}")
    print(f"min: {np.nanmin(img)}")
    print(f"max: {np.nanmax(img)}")
    print(f"nan_count: {int(np.isnan(img).sum())}")
    print(f"zero_count: {int((img == 0).sum())}")
    return 0


def read_chw(path: str):
    _, rasterio, *_ = lazy_import_rasterio()
    with rasterio.open(path) as src:
        return src.read().astype("float32")


def read_mask(path: str):
    _, rasterio, *_ = lazy_import_rasterio()
    with rasterio.open(path) as src:
        return src.read(1)


def resize_chw(crop, target_size: int):
    import numpy as np
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    out = np.empty((crop.shape[0], target_size, target_size), dtype="float32")
    for band_idx in range(crop.shape[0]):
        band = np.nan_to_num(crop[band_idx], nan=0.0).astype("float32", copy=False)
        image = Image.fromarray(band)
        out[band_idx] = np.asarray(image.resize((target_size, target_size), resample=resampling), dtype="float32")
    return out


def resize_mask(mask_crop, target_size: int):
    import numpy as np
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).NEAREST
    image = Image.fromarray(mask_crop.astype("uint8", copy=False))
    return np.asarray(image.resize((target_size, target_size), resample=resampling), dtype="uint8")


def is_valid_crop(crop) -> bool:
    import numpy as np

    if crop.size == 0 or crop.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
        return False
    return bool((np.isnan(crop[0]) | (crop[0] == 0)).mean() <= MISSING_THRESH)


def make_sample(
    sample_id: int,
    label: int,
    plume_id: str,
    out_root: Path,
    images: dict[str, Any],
    mask,
    x: int,
    y: int,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    import tifffile

    crops = {
        tp.name: images[tp.name][:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
        for tp in TIMEPOINTS
    }
    if not all(is_valid_crop(crop) for crop in crops.values()):
        return None

    out_dir = out_root / f"{sample_id:08d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, str] = {}
    for tp in TIMEPOINTS:
        patch = resize_chw(crops[tp.name], PATCH_TARGET_SIZE)
        dst = out_dir / tp.patch_name
        tifffile.imwrite(dst, patch)
        out_paths[f"path_{tp.name}"] = str(dst)

    if label == 1:
        mask_crop = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
        mask_up = resize_mask(mask_crop, PATCH_TARGET_SIZE)
    else:
        import numpy as np

        mask_up = np.zeros((PATCH_TARGET_SIZE, PATCH_TARGET_SIZE), dtype="uint8")
    plume_path = out_dir / "plume.tif"
    tifffile.imwrite(plume_path, mask_up)

    result = {
        "id": sample_id,
        "label": label,
        "plume_id": plume_id,
        "path": str(out_dir),
        "path_plume": str(plume_path),
        "source_x": x,
        "source_y": y,
        "latitude": row.get("plume_latitude", ""),
        "longitude": row.get("plume_longitude", ""),
        "event_time": row.get("event_time", ""),
    }
    for tp in TIMEPOINTS:
        result[f"{tp.name}_image_time"] = row.get(f"{tp.name}_image_time", "")
    result.update(out_paths)
    return result


def process_sampling_row(
    row: dict[str, Any],
    out_root: Path,
    id_counter: dict[str, int],
    counter_lock: threading.Lock,
    seed: int,
) -> list[dict[str, Any]]:
    plume_id = str(row.get("plume_id", "")).strip()
    try:
        paths = {tp.name: first_existing_path(row, tp.path_col, tp.legacy_col) for tp in TIMEPOINTS}
        if not all(paths.values()):
            return []
        mask_path = first_existing_path(row, "l89_512_mask_path", "mask_path")
        if not mask_path:
            return []
        images = {name: read_chw(path) for name, path in paths.items()}
        mask = read_mask(mask_path)
    except Exception as exc:
        print(f"sample224: skip {plume_id}, read error: {exc}", flush=True)
        return []

    rng = random.Random(seed + abs(hash(plume_id)) % 1_000_000_000)
    results: list[dict[str, Any]] = []

    def next_id() -> int:
        with counter_lock:
            id_counter["value"] += 1
            return id_counter["value"]

    pos_min = 256 + (CENTER_BOX // 2) - PATCH_SIZE
    pos_max = 256 - (CENTER_BOX // 2)
    pos_count = 0
    attempts = 0
    while pos_count < N_POS and attempts < 100:
        attempts += 1
        x = rng.randint(pos_min, pos_max)
        y = rng.randint(pos_min, pos_max)
        sample = make_sample(next_id(), 1, plume_id, out_root, images, mask, x, y, row)
        if sample is not None:
            results.append(sample)
            pos_count += 1

    neg_count = 0
    attempts = 0
    while neg_count < N_NEG and attempts < 200:
        attempts += 1
        x = rng.randint(0, WINDOW_SIZE - PATCH_SIZE)
        y = rng.randint(0, WINDOW_SIZE - PATCH_SIZE)
        if (x <= 256 <= x + PATCH_SIZE) and (y <= 256 <= y + PATCH_SIZE):
            continue
        sample = make_sample(next_id(), 0, plume_id, out_root, images, mask, x, y, row)
        if sample is not None:
            results.append(sample)
            neg_count += 1

    return results


def iter_dataframe_records(df):
    columns = list(df.columns)
    for values in df.itertuples(index=False, name=None):
        yield dict(zip(columns, values))


def append_sample_manifest(manifest_csv: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_csv.exists() or manifest_csv.stat().st_size == 0
    with manifest_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SAMPLE_MANIFEST_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def load_done_plume_ids(manifest_csv: Path) -> set[str]:
    if not manifest_csv.exists() or manifest_csv.stat().st_size == 0:
        return set()
    with manifest_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "plume_id" not in (reader.fieldnames or []):
            return set()
        return {str(row.get("plume_id", "")).strip() for row in reader if row.get("plume_id", "")}


def get_start_sample_id(out_root: Path, manifest_csv: Path, resume: bool) -> int:
    manifest_max_id = 0
    if resume and manifest_csv.exists() and manifest_csv.stat().st_size > 0:
        with manifest_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if "id" in (reader.fieldnames or []):
                for row in reader:
                    try:
                        manifest_max_id = max(manifest_max_id, int(float(row.get("id", 0))))
                    except (TypeError, ValueError):
                        continue
    dir_ids = [int(path.name) for path in out_root.iterdir() if path.is_dir() and path.name.isdigit()] if out_root.exists() else []
    dir_max_id = max(dir_ids) if dir_ids else 0
    return max(manifest_max_id, dir_max_id)


def process_sampling_row_block(payload: tuple[dict[str, Any], str, int, int]) -> tuple[list[dict[str, Any]], int]:
    row, out_root_text, start_id, seed = payload
    id_counter = {"value": start_id}
    records = process_sampling_row(row, Path(out_root_text), id_counter, threading.Lock(), seed)
    if id_counter["value"] > start_id + MAX_SAMPLE_IDS_PER_ROW:
        plume_id = str(row.get("plume_id", "")).strip()
        raise RuntimeError(f"sample id block exhausted for plume_id={plume_id}")
    return records, id_counter["value"]


def run_sample224(args: argparse.Namespace) -> int:
    pd = lazy_import_pandas()
    in_csv = Path(args.input_csv)
    out_root = Path(args.out_root)
    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else out_root / "dataset_manifest.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    df = ensure_pipeline_columns(pd.read_csv(in_csv))
    if args.resume and manifest_csv.exists():
        done = load_done_plume_ids(manifest_csv)
        if done:
            df = df[~df["plume_id"].astype(str).isin(done)].copy()
    if args.limit:
        df = df.head(args.limit).copy()

    total_rows = len(df)
    workers = max(1, int(args.workers))
    flush_every = max(1, int(args.flush_every))
    gc_every = max(0, int(args.gc_every))
    progress_every = max(1, int(args.progress_every))
    max_inflight = max(1, int(args.max_inflight)) if args.max_inflight else max(1, workers * max(1, int(args.inflight_multiplier)))
    isolate_rows = bool(args.isolate_rows) if args.isolate_rows is not None else workers == 1

    id_counter = {"value": get_start_sample_id(out_root, manifest_csv, args.resume)}
    counter_lock = threading.Lock()
    mode = "process-isolated" if isolate_rows else ("sequential" if workers == 1 else f"threaded max_inflight={max_inflight}")
    print(
        f"sample224: rows={total_rows} start_id={id_counter['value']} mode={mode} out_root={out_root}",
        flush=True,
    )

    buffer: list[dict[str, Any]] = []
    completed_rows = 0
    written_samples = 0

    def handle_records(records: list[dict[str, Any]]) -> None:
        nonlocal completed_rows, written_samples
        completed_rows += 1
        buffer.extend(records)
        written_samples += len(records)
        if completed_rows % flush_every == 0:
            append_sample_manifest(manifest_csv, buffer)
            buffer.clear()
        if completed_rows % progress_every == 0 or completed_rows == total_rows:
            print(
                f"sample224: {completed_rows}/{total_rows} rows, samples_written={written_samples}",
                flush=True,
            )
        if gc_every and completed_rows % gc_every == 0:
            gc.collect()

    if isolate_rows and workers == 1:
        ctx = mp.get_context(args.mp_start_method)
        with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
            for row in iter_dataframe_records(df):
                records, end_id = pool.apply(
                    process_sampling_row_block,
                    ((row, str(out_root), id_counter["value"], args.seed),),
                )
                id_counter["value"] = end_id
                handle_records(records)
    elif isolate_rows:
        start_id = id_counter["value"]
        payloads = (
            (row, str(out_root), start_id + row_idx * MAX_SAMPLE_IDS_PER_ROW, args.seed)
            for row_idx, row in enumerate(iter_dataframe_records(df))
        )
        highest_id = start_id
        ctx = mp.get_context(args.mp_start_method)
        with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
            for records, end_id in pool.imap_unordered(process_sampling_row_block, payloads, chunksize=1):
                highest_id = max(highest_id, end_id)
                handle_records(records)
        id_counter["value"] = highest_id
    elif workers == 1:
        for row in iter_dataframe_records(df):
            records = process_sampling_row(row, out_root, id_counter, counter_lock, args.seed)
            handle_records(records)
    else:
        row_iter = iter_dataframe_records(df)
        pending = set()
        submitted = 0

        def submit_next(pool: ThreadPoolExecutor) -> bool:
            nonlocal submitted
            try:
                row = next(row_iter)
            except StopIteration:
                return False
            fut = pool.submit(process_sampling_row, row, out_root, id_counter, counter_lock, args.seed)
            pending.add(fut)
            submitted += 1
            return True

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while len(pending) < max_inflight and submit_next(pool):
                pass
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    handle_records(fut.result())
                    while len(pending) < max_inflight and submit_next(pool):
                        pass

    append_sample_manifest(manifest_csv, buffer)
    buffer.clear()
    print(f"sample224: wrote {written_samples} new samples, manifest={manifest_csv}", flush=True)
    return 0


def add_common_csv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out-csv", default="", help="Default overwrites --input-csv.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "L89 six-time preprocessing copied from preprocess_dataset_L89/l89_90360_preprocess.ipynb "
            "and adapted for t0, prev1, prev2, prev3, seasonal, year."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_crop = sub.add_parser("crop512", help="Crop raw L89 TIF chips to centered 512x512 float32 GeoTIFFs.")
    add_common_csv_args(p_crop)
    p_crop.add_argument("--out-root", default=str(DEFAULT_512_ROOT))
    p_crop.add_argument("--workers", type=int, default=12)
    p_crop.add_argument("--limit", type=int, default=0)
    p_crop.add_argument("--progress-every", type=int, default=100)
    p_crop.add_argument("--overwrite", action="store_true")
    p_crop.set_defaults(func=run_crop512)

    p_repair = sub.add_parser("repair512", help="Repair/fill 512 path columns from existing files, without cropping.")
    add_common_csv_args(p_repair)
    p_repair.add_argument("--out-root", default=str(DEFAULT_512_ROOT))
    p_repair.set_defaults(func=run_repair512)

    p_mask = sub.add_parser("mask512", help="Generate 512x512 L89-aligned Carbon Mapper plume masks.")
    add_common_csv_args(p_mask)
    p_mask.add_argument("--out-root", default=str(DEFAULT_MASK_OUT_ROOT))
    p_mask.add_argument("--cm-mask-roots", nargs="+", default=[str(p) for p in DEFAULT_CM_MASK_ROOTS])
    p_mask.add_argument("--workers", type=int, default=8)
    p_mask.add_argument("--limit", type=int, default=0)
    p_mask.add_argument("--progress-every", type=int, default=100)
    p_mask.add_argument("--overwrite", action="store_true")
    p_mask.set_defaults(func=run_mask512)

    p_qc = sub.add_parser("qc-overlay", help="Save visual checks for original mask, L89 t0 512, and overlay.")
    p_qc.add_argument("--input-csv", default=str(DEFAULT_CSV))
    p_qc.add_argument("--out-dir", default=str(DEFAULT_QC_DIR))
    p_qc.add_argument("--cm-mask-roots", nargs="+", default=[str(p) for p in DEFAULT_CM_MASK_ROOTS])
    p_qc.add_argument("--samples", type=int, default=10)
    p_qc.set_defaults(func=run_qc_overlay)

    p_stats = sub.add_parser("qa-stats", help="Print min/max/NaN/zero counts for the first available t0 512 crop.")
    p_stats.add_argument("--input-csv", default=str(DEFAULT_CSV))
    p_stats.set_defaults(func=run_qa_stats)

    p_sample = sub.add_parser("sample224", help="Sample 16 positive and 16 negative patches per plume from all six 512 timepoints.")
    p_sample.add_argument("--input-csv", default=str(DEFAULT_CSV))
    p_sample.add_argument("--out-root", default=str(DEFAULT_PATCH_OUT_ROOT))
    p_sample.add_argument("--manifest-csv", default="")
    p_sample.add_argument("--workers", type=int, default=8)
    p_sample.add_argument("--limit", type=int, default=0)
    p_sample.add_argument("--progress-every", type=int, default=50)
    p_sample.add_argument("--flush-every", type=int, default=1, help="Append manifest rows after this many completed plumes.")
    p_sample.add_argument("--max-inflight", type=int, default=0, help="Cap submitted-but-unfinished rows; default is workers * inflight-multiplier.")
    p_sample.add_argument("--inflight-multiplier", type=int, default=2)
    p_sample.add_argument("--gc-every", type=int, default=25, help="Run gc.collect after this many completed plumes; 0 disables it.")
    p_sample.add_argument("--resume", action="store_true")
    p_sample.add_argument("--seed", type=int, default=1234)
    p_sample.add_argument("--isolate-rows", dest="isolate_rows", action="store_true", default=None, help="Run each plume in a recycled child process to avoid native memory growth.")
    p_sample.add_argument("--no-isolate-rows", dest="isolate_rows", action="store_false", help="Disable automatic row isolation when --workers 1 is used.")
    p_sample.add_argument("--mp-start-method", choices=("spawn", "fork", "forkserver"), default="spawn")
    p_sample.set_defaults(func=run_sample224)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
