#!/usr/bin/env python3
"""Rebuild the six-time S2 dataset from CDSE raw and legacy 512 assets.

The image standardization follows Cell 4 of
``preprocess_dataset_s2/carbon_mapper_sentinel2_plume_-7_download.ipynb``:
read each S2 stack as B,H,W float32, center crop or zero-pad it to 512, and
write a GDAL-readable multiband TIFF.  Existing standardized images may be
hard-linked after they have been traced back to the same source table.

Carbon Mapper masks are always rebuilt from the raw georeferenced plume TIFF
with the exact old 20 m reprojection and center-at-plume geometry.  Missing or
empty masks are fatal; they are never replaced with an all-zero mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
import tifffile
from affine import Affine
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform
from rasterio.windows import Window


METHANE_ROOT = Path("/home/yuyao/methane_train")
CSV_ROOT = METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_cdse_legacy512_exact"

DEFAULT_INPUT_CSV = (
    METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_all6_available_paths.csv"
)
DEFAULT_VALIDATED_512_CSV = (
    METHANE_ROOT
    / "Upgrade_data_pipeline"
    / "csv"
    / "s2_6time_all6_available_paths_std512_complete.csv"
)
DEFAULT_MAIN_CSV = (
    METHANE_ROOT
    / "Upgrade_data_pipeline"
    / "csv"
    / "carbon_mapper_plumes_20160101_20260530_with_t0_flags.csv"
)
DEFAULT_CM_ROOT = Path("/mnt/engg-niulab/yuyao/sensors_raw_data/CM")
DEFAULT_OUT_512_ROOT = Path(
    "/mnt/engg-niulab/yuyao/preprocessed_512/S2_6time_cdse_legacy512_exact"
)
DEFAULT_OUT_32_ROOT = Path(
    "/mnt/engg-niulab/yuyao/final_crop/s2_6time_cdse_legacy512_exact_32"
)
DEFAULT_OUT_224_ROOT = Path(
    "/mnt/engg-niulab/yuyao/final_crop/s2_6time_cdse_legacy512_exact_32_to_224"
)

DEFAULT_SOURCE_CSV = CSV_ROOT / "s2_6time_cdse_legacy512_sources.csv"
DEFAULT_SOURCE_AUDIT = CSV_ROOT / "s2_6time_cdse_legacy512_sources.audit.json"
DEFAULT_QA_512_CSV = CSV_ROOT / "s2_6time_cdse_legacy512_512_qa.csv"
DEFAULT_COMPLETE_512_CSV = CSV_ROOT / "s2_6time_cdse_legacy512_512_complete.csv"
DEFAULT_SPLIT_ROOT = CSV_ROOT / "temporal_split"

DEFAULT_RAW_PREFIX = "/mnt/engg-niulab/yuyao/sensors_raw_data/S2/"
DEFAULT_LEGACY_512_PREFIX = (
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
    "Dataset/plume_raw_s2_90360_fixed_512/"
)
DEFAULT_VALIDATED_512_PREFIX = "/mnt/engg-niulab/yuyao/preprocessed_512/S2/"

WINDOW_SIZE = 512
PATCH_SIZE = 32
TARGET_SIZE = 224
EXPECTED_BANDS = 12
EXPECTED_ROWS = 4450
EXPECTED_ALL_RAW_ROWS = 4407
EXPECTED_MIXED_ROWS = 43


@dataclass(frozen=True)
class Timepoint:
    name: str
    input_path_col: str
    input_kind_col: str
    std_col: str
    std_filename: str
    patch_filename: str
    image_time_col: str
    force_tifffile: bool = False


TIMEPOINTS = [
    Timepoint(
        "t0",
        "t0_input_path",
        "t0_input_kind",
        "s2_0_std_512",
        "s2_0_std_512.tif",
        "s2_0.tif",
        "t0_image_time",
        True,
    ),
    Timepoint(
        "prev1",
        "prev1_input_path",
        "prev1_input_kind",
        "s2_-7_std_512",
        "s2_-7_std_512.tif",
        "s2_prev1.tif",
        "prev1_image_time",
    ),
    Timepoint(
        "prev2",
        "prev2_input_path",
        "prev2_input_kind",
        "s2_prev2_std_512",
        "s2_prev2_std_512.tif",
        "s2_prev2.tif",
        "prev2_image_time",
    ),
    Timepoint(
        "prev3",
        "prev3_input_path",
        "prev3_input_kind",
        "s2_prev3_std_512",
        "s2_prev3_std_512.tif",
        "s2_prev3.tif",
        "prev3_image_time",
    ),
    Timepoint(
        "seasonal",
        "seasonal_input_path",
        "seasonal_input_kind",
        "s2_-90_std_512",
        "s2_-90_std_512.tif",
        "s2_seasonal.tif",
        "seasonal_image_time",
    ),
    Timepoint(
        "year",
        "year_input_path",
        "year_input_kind",
        "s2_-360_std_512",
        "s2_-360_std_512.tif",
        "s2_year.tif",
        "year_image_time",
    ),
]

PATH_COLUMNS = {timepoint.name: f"path_{timepoint.name}" for timepoint in TIMEPOINTS}
BANNED_SOURCE_MARKERS = ("/s2_gee_6time/", "cm_s2_l2a_6time_gee")
NOMINAL_OFFSETS_DAYS = {
    "t0": 0,
    "prev1": -7,
    "prev2": -14,
    "prev3": -21,
    "seasonal": -90,
    "year": -360,
}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def file_ok(value: Any) -> bool:
    text = clean(value)
    if not text:
        return False
    try:
        path = Path(text)
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_time(value: Any) -> pd.Timestamp:
    text = clean(value)
    if not text:
        return pd.NaT
    try:
        timestamp = pd.to_datetime(text, utc=True, errors="coerce", format="mixed")
    except TypeError:
        timestamp = pd.NaT
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(text, utc=True, errors="coerce")
    return timestamp


def fallback_group_id(plume_id: str) -> str:
    parts = plume_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1] and len(parts[1]) <= 4:
        return parts[0]
    return plume_id


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def parallel_file_flags(paths: Iterable[str], workers: int) -> dict[str, bool]:
    unique_paths = sorted({clean(path) for path in paths if clean(path)})
    if not unique_paths:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        values = pool.map(file_ok, unique_paths)
        return dict(zip(unique_paths, values))


def required_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def validate_source_path(path: str, kind: str, args: argparse.Namespace) -> str:
    lower_path = path.lower()
    if any(marker in lower_path for marker in BANNED_SOURCE_MARKERS):
        raise ValueError(f"unrelated export source is forbidden: {path}")
    if kind == "raw":
        if not path.startswith(str(args.raw_prefix)):
            raise ValueError(f"raw source is outside the approved CDSE root: {path}")
        return "raw"
    if kind == "512":
        if not path.startswith(str(args.legacy_512_prefix)):
            raise ValueError(f"512 source is outside the approved legacy root: {path}")
        return "512"
    raise ValueError(f"unsupported input kind {kind!r}: {path}")


def build_manifest(args: argparse.Namespace) -> int:
    input_path = Path(args.input_csv)
    validated_path = Path(args.validated_512_csv)
    main_path = Path(args.main_csv)
    source = pd.read_csv(input_path, low_memory=False)
    required_columns(
        source,
        [
            "plume_id",
            "event_time",
            "plume_latitude",
            "plume_longitude",
            "input_kind_pattern",
            *[timepoint.input_path_col for timepoint in TIMEPOINTS],
            *[timepoint.input_kind_col for timepoint in TIMEPOINTS],
            *[timepoint.image_time_col for timepoint in TIMEPOINTS],
        ],
        input_path,
    )
    if source["plume_id"].astype(str).duplicated().any():
        examples = source.loc[source["plume_id"].astype(str).duplicated(False), "plume_id"].head(10).tolist()
        raise ValueError(f"source table has duplicate plume_id values: {examples}")
    if int(args.expected_rows) > 0 and len(source) != int(args.expected_rows):
        raise ValueError(f"expected {args.expected_rows} source rows, found {len(source)}")

    main = pd.read_csv(
        main_path,
        usecols=lambda column: column
        in {"plume_id", "event_group_id", "datetime", "plume_latitude", "plume_longitude"},
        low_memory=False,
    )
    required_columns(main, ["plume_id", "event_group_id"], main_path)
    if main["plume_id"].astype(str).duplicated().any():
        raise ValueError(f"{main_path} has duplicate plume_id values")
    metadata = main[["plume_id", "event_group_id"]].copy()
    source = source.merge(metadata, on="plume_id", how="left", validate="one_to_one")
    source["event_group_id"] = source.apply(
        lambda row: clean(row.get("event_group_id")) or fallback_group_id(clean(row.get("plume_id"))),
        axis=1,
    )
    if source["event_group_id"].eq("").any():
        raise ValueError("at least one source row has no event_group_id")

    validated = pd.read_csv(validated_path, low_memory=False)
    validated_columns = ["plume_id", *[timepoint.std_col for timepoint in TIMEPOINTS]]
    required_columns(validated, validated_columns, validated_path)
    if validated["plume_id"].astype(str).duplicated().any():
        raise ValueError(f"{validated_path} has duplicate plume_id values")
    validated = validated[validated_columns].copy()
    validated = validated.rename(
        columns={
            timepoint.std_col: f"validated_{timepoint.name}_512_path" for timepoint in TIMEPOINTS
        }
    )
    source = source.merge(validated, on="plume_id", how="left", validate="one_to_one")

    source_records = source.to_dict("records")
    approved_paths: list[str] = []
    validated_paths: list[str] = []
    raw_cm_paths: list[str] = []
    output_root = Path(args.out_512_root)
    patterns: list[str] = []
    for row in source_records:
        plume_id = clean(row.get("plume_id"))
        event_timestamp = parse_time(row.get("event_time"))
        if pd.isna(event_timestamp):
            raise ValueError(f"invalid event_time for {plume_id}: {row.get('event_time')}")
        kinds: list[str] = []
        for timepoint in TIMEPOINTS:
            input_path_value = clean(row.get(timepoint.input_path_col))
            input_kind = clean(row.get(timepoint.input_kind_col)).lower()
            kinds.append(validate_source_path(input_path_value, input_kind, args))
            approved_paths.append(input_path_value)

            validated_value = clean(row.get(f"validated_{timepoint.name}_512_path"))
            if not validated_value.startswith(str(args.validated_512_prefix)):
                raise ValueError(
                    f"validated 512 path is outside the approved root: {validated_value}"
                )
            if any(marker in validated_value.lower() for marker in BANNED_SOURCE_MARKERS):
                raise ValueError(f"unrelated path in validated table: {validated_value}")
            validated_paths.append(validated_value)
            row[timepoint.std_col] = str(output_root / plume_id / timepoint.std_filename)
            if not clean(row.get(timepoint.image_time_col)):
                row[timepoint.image_time_col] = (
                    event_timestamp
                    + pd.Timedelta(days=NOMINAL_OFFSETS_DAYS[timepoint.name])
                ).isoformat()

        pattern = "+".join(kinds)
        patterns.append(pattern)
        if pattern not in {
            "raw+raw+raw+raw+raw+raw",
            "raw+raw+raw+raw+512+512",
        }:
            raise ValueError(f"unexpected source pattern for {plume_id}: {pattern}")
        if pattern != clean(row.get("input_kind_pattern")):
            raise ValueError(
                f"source pattern mismatch for {plume_id}: table={row.get('input_kind_pattern')} actual={pattern}"
            )
        row["source_pattern"] = pattern
        row["cohort"] = "cdse_raw_or_legacy512"
        row["raw_cm_mask_path"] = str(Path(args.cm_root) / plume_id / "plume.tif")
        row["resized_512x512_path"] = str(output_root / plume_id / "resized_512x512.tif")
        raw_cm_paths.append(row["raw_cm_mask_path"])

    pattern_counts = pd.Series(patterns).value_counts().to_dict()
    if int(args.expected_all_raw_rows) > 0:
        actual = int(pattern_counts.get("raw+raw+raw+raw+raw+raw", 0))
        if actual != int(args.expected_all_raw_rows):
            raise ValueError(f"expected {args.expected_all_raw_rows} all-raw rows, found {actual}")
    if int(args.expected_mixed_rows) > 0:
        actual = int(pattern_counts.get("raw+raw+raw+raw+512+512", 0))
        if actual != int(args.expected_mixed_rows):
            raise ValueError(f"expected {args.expected_mixed_rows} mixed rows, found {actual}")

    path_flags = parallel_file_flags(
        [*approved_paths, *validated_paths, *raw_cm_paths], int(args.stat_workers)
    )
    missing_approved = [path for path in approved_paths if not path_flags.get(path, False)]
    missing_validated = [path for path in validated_paths if not path_flags.get(path, False)]
    missing_masks = [path for path in raw_cm_paths if not path_flags.get(path, False)]
    if missing_approved or missing_validated or missing_masks:
        raise FileNotFoundError(
            "source census failed: "
            f"missing_inputs={len(missing_approved)} "
            f"missing_validated_512={len(missing_validated)} "
            f"missing_raw_masks={len(missing_masks)} "
            f"examples={(missing_approved + missing_validated + missing_masks)[:20]}"
        )

    output = pd.DataFrame(source_records)
    output["event_time_parsed"] = output["event_time"].map(parse_time)
    if output["event_time_parsed"].isna().any():
        examples = output.loc[
            output["event_time_parsed"].isna(), ["plume_id", "event_time"]
        ].head(10)
        raise ValueError(f"invalid event_time values: {examples.to_dict('records')}")
    output = output.sort_values(["event_time_parsed", "plume_id"], kind="stable").drop(
        columns="event_time_parsed"
    )
    atomic_csv(output, Path(args.source_csv))

    audit = {
        "input_csv": str(input_path),
        "validated_512_csv": str(validated_path),
        "rows": len(output),
        "unique_plumes": int(output["plume_id"].nunique()),
        "unique_event_groups": int(output["event_group_id"].nunique()),
        "source_patterns": {str(key): int(value) for key, value in pattern_counts.items()},
        "source_files": len(approved_paths),
        "source_files_present": len(approved_paths) - len(missing_approved),
        "validated_512_files": len(validated_paths),
        "validated_512_files_present": len(validated_paths) - len(missing_validated),
        "raw_cm_masks": len(raw_cm_paths),
        "raw_cm_masks_present": len(raw_cm_paths) - len(missing_masks),
        "uses_unrelated_export_pipeline": False,
        "output_csv": str(args.source_csv),
    }
    atomic_json(audit, Path(args.source_audit_json))
    log(json.dumps(audit, ensure_ascii=False))
    return 0


def to_bhw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"unexpected image shape {array.shape}")
    if array.shape[0] in {1, 3, 4, 12, 13}:
        return array
    if array.shape[-1] in {1, 3, 4, 12, 13}:
        return np.transpose(array, (2, 0, 1))
    return array


def center_crop_or_pad(array: np.ndarray, size: int = WINDOW_SIZE) -> tuple[np.ndarray, str]:
    bands, height, width = array.shape
    if height == size and width == size:
        return array, "noop"
    if height < size or width < size:
        output = np.zeros((bands, size, size), dtype=array.dtype)
        destination_y = max(0, (size - height) // 2)
        destination_x = max(0, (size - width) // 2)
        source_y = max(0, (height - size) // 2)
        source_x = max(0, (width - size) // 2)
        copy_height = min(height, size)
        copy_width = min(width, size)
        output[
            :,
            destination_y : destination_y + copy_height,
            destination_x : destination_x + copy_width,
        ] = array[
            :,
            source_y : source_y + copy_height,
            source_x : source_x + copy_width,
        ]
        return output, f"pad_center:{height}x{width}"
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return array[:, y0 : y0 + size, x0 : x0 + size], f"crop_center:{height}x{width}"


def read_rasterio_center(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        if dataset.width >= WINDOW_SIZE and dataset.height >= WINDOW_SIZE:
            x0 = (dataset.width - WINDOW_SIZE) // 2
            y0 = (dataset.height - WINDOW_SIZE) // 2
            array = dataset.read(window=Window(x0, y0, WINDOW_SIZE, WINDOW_SIZE))
        else:
            array = dataset.read()
    return to_bhw(array).astype(np.float32, copy=False)


def read_tifffile(path: Path) -> np.ndarray:
    return to_bhw(tifffile.imread(path)).astype(np.float32, copy=False)


def read_image_for_standardization(path: Path, force_tifffile: bool) -> tuple[np.ndarray, str]:
    if force_tifffile:
        return read_tifffile(path), "tifffile"
    try:
        array = read_rasterio_center(path)
        note = "rasterio"
        if array.shape[0] == 1:
            tif_array = read_tifffile(path)
            if tif_array.shape[0] in {12, 13}:
                array = tif_array
                note = "tifffile_after_rasterio_single_band"
        return array, note
    except Exception as rasterio_error:
        try:
            return read_tifffile(path), "tifffile_fallback"
        except Exception as tifffile_error:
            raise RuntimeError(
                f"rasterio={type(rasterio_error).__name__}:{rasterio_error}; "
                f"tifffile={type(tifffile_error).__name__}:{tifffile_error}"
            ) from tifffile_error


def write_multiband_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".part.{os.getpid()}.{random.randrange(1 << 30)}"
    )
    bands, height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": str(array.dtype),
        "transform": from_origin(0, 0, 1, 1),
        "compress": "deflate",
        "predictor": 2 if np.issubdtype(array.dtype, np.floating) else 1,
        "tiled": True,
        "blockxsize": min(256, width),
        "blockysize": min(256, height),
        "BIGTIFF": "IF_SAFER",
    }
    try:
        with rasterio.open(temporary, "w", **profile) as dataset:
            dataset.write(array)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def standardize_to_512(source: Path, destination: Path, force_tifffile: bool, overwrite: bool) -> str:
    if file_ok(destination) and not overwrite:
        return "exists"
    if not file_ok(source):
        raise FileNotFoundError(source)
    array, read_note = read_image_for_standardization(source, force_tifffile)
    if array.shape[0] != EXPECTED_BANDS:
        raise ValueError(f"expected {EXPECTED_BANDS} bands, got {array.shape}: {source}")
    array, geometry_note = center_crop_or_pad(array, WINDOW_SIZE)
    write_multiband_atomic(destination, array.astype(np.float32, copy=False))
    return ":".join([read_note, geometry_note])


def link_validated(source: Path, destination: Path, mode: str, overwrite: bool) -> str:
    if not file_ok(source):
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if not overwrite and file_ok(destination):
            return "exists"
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy-validated":
        temporary = destination.with_name(destination.name + f".part.{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return "copied_validated"
    if mode == "hardlink-validated":
        os.link(source, destination)
        return "hardlinked_validated"
    if mode == "symlink-validated":
        os.symlink(source, destination)
        return "symlinked_validated"
    raise ValueError(f"unsupported image mode: {mode}")


def paste_clipped(canvas: np.ndarray, source: np.ndarray, start_x: int, start_y: int) -> None:
    source_height, source_width = source.shape
    destination_height, destination_width = canvas.shape
    destination_x0 = max(0, start_x)
    destination_y0 = max(0, start_y)
    destination_x1 = min(destination_width, start_x + source_width)
    destination_y1 = min(destination_height, start_y + source_height)
    if destination_x0 >= destination_x1 or destination_y0 >= destination_y1:
        raise ValueError("reprojected plume mask does not intersect the 512 canvas")
    source_x0 = destination_x0 - start_x
    source_y0 = destination_y0 - start_y
    canvas[destination_y0:destination_y1, destination_x0:destination_x1] = source[
        source_y0 : source_y0 + (destination_y1 - destination_y0),
        source_x0 : source_x0 + (destination_x1 - destination_x0),
    ]


def build_exact_legacy_mask(
    raw_mask: Path,
    output_path: Path,
    latitude: float,
    longitude: float,
    overwrite: bool,
) -> dict[str, Any]:
    if file_ok(output_path) and not overwrite:
        with rasterio.open(output_path) as dataset:
            existing = (dataset.read(1) > 0).astype(np.uint8)
        return {
            "mask_action": "exists",
            "mask_positive_pixels": int(existing.sum()),
            "mask_center20_sum": int(existing[246:266, 246:266].sum()),
        }
    if not file_ok(raw_mask):
        raise FileNotFoundError(raw_mask)

    with rasterio.open(raw_mask) as source:
        if source.crs is None:
            raise ValueError(f"raw Carbon Mapper mask has no CRS: {raw_mask}")
        temporary_transform, _, _ = calculate_default_transform(
            source.crs,
            source.crs,
            source.width,
            source.height,
            resolution=(20, 20),
            left=source.bounds.left,
            bottom=source.bounds.bottom,
            right=source.bounds.right,
            top=source.bounds.top,
        )
        projected_width = max(1, int((source.bounds.right - source.bounds.left) / 20))
        projected_height = max(1, int((source.bounds.top - source.bounds.bottom) / 20))
        projected = np.zeros(
            (source.count, projected_height, projected_width), dtype=np.uint8
        )
        for band_index in range(source.count):
            reproject(
                source=source.read(band_index + 1),
                destination=projected[band_index],
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=temporary_transform,
                dst_crs=source.crs,
                resampling=Resampling.bilinear,
            )
        binary = np.max(projected > 0, axis=0).astype(np.uint8)
        center_x_values, center_y_values = transform(
            "EPSG:4326", source.crs, [float(longitude)], [float(latitude)]
        )
        center_x = float(center_x_values[0])
        center_y = float(center_y_values[0])
        center_pixel_x = int((center_x - source.bounds.left) / 20)
        center_pixel_y = int((source.bounds.top - center_y) / 20)
        start_x = WINDOW_SIZE // 2 - center_pixel_x
        start_y = WINDOW_SIZE // 2 - center_pixel_y
        canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
        paste_clipped(canvas, binary, start_x, start_y)
        output_transform = Affine(
            20,
            0,
            center_x - (WINDOW_SIZE // 2) * 20,
            0,
            -20,
            center_y + (WINDOW_SIZE // 2) * 20,
        )
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            transform=output_transform,
            crs=source.crs,
            width=WINDOW_SIZE,
            height=WINDOW_SIZE,
            count=1,
            dtype="uint8",
            compress="deflate",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        output_path.name + f".part.{os.getpid()}.{random.randrange(1 << 30)}"
    )
    try:
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(canvas, 1)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "mask_action": "rebuilt_exact_legacy_geometry",
        "mask_positive_pixels": int(canvas.sum()),
        "mask_center20_sum": int(canvas[246:266, 246:266].sum()),
    }


def validate_512_image(path: Path) -> tuple[bool, str]:
    if not file_ok(path):
        return False, "missing"
    try:
        with rasterio.open(path) as dataset:
            shape = (dataset.count, dataset.height, dataset.width)
            dtype = dataset.dtypes[0]
        if shape != (EXPECTED_BANDS, WINDOW_SIZE, WINDOW_SIZE):
            return False, f"shape={shape}"
        if dtype != "float32":
            return False, f"dtype={dtype}"
        return True, "ok"
    except Exception as error:
        return False, f"{type(error).__name__}:{error}"


def process_512_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    plume_id = clean(row.get("plume_id"))
    result: dict[str, Any] = {
        "plume_id": plume_id,
        "event_group_id": clean(row.get("event_group_id")),
        "source_pattern": clean(row.get("source_pattern")),
        "status": "fail",
        "reason": "",
    }
    errors: list[str] = []
    actions: list[str] = []
    for timepoint in TIMEPOINTS:
        destination = Path(clean(row.get(timepoint.std_col)))
        try:
            if args.image_mode == "standardize":
                source = Path(clean(row.get(timepoint.input_path_col)))
                action = standardize_to_512(
                    source,
                    destination,
                    timepoint.force_tifffile,
                    bool(args.overwrite),
                )
            elif args.image_mode == "reuse-validated":
                source = Path(clean(row.get(f"validated_{timepoint.name}_512_path")))
                destination = source
                action = "reused_validated_in_place"
            else:
                source = Path(clean(row.get(f"validated_{timepoint.name}_512_path")))
                action = link_validated(
                    source, destination, str(args.image_mode), bool(args.overwrite)
                )
            if args.image_mode != "reuse-validated":
                valid, message = validate_512_image(destination)
                if not valid:
                    raise ValueError(message)
            result[timepoint.std_col] = str(destination)
            actions.append(f"{timepoint.name}:{action}")
        except Exception as error:
            result[timepoint.std_col] = ""
            errors.append(f"{timepoint.name}:{type(error).__name__}:{error}")

    mask_path = Path(clean(row.get("resized_512x512_path")))
    try:
        mask_info = build_exact_legacy_mask(
            Path(clean(row.get("raw_cm_mask_path"))),
            mask_path,
            float(row["plume_latitude"]),
            float(row["plume_longitude"]),
            bool(args.overwrite_masks),
        )
        result.update(mask_info)
        result["resized_512x512_path"] = str(mask_path)
        if int(mask_info["mask_positive_pixels"]) <= 0:
            raise ValueError("zero_positive_pixels")
        if int(mask_info["mask_center20_sum"]) <= 0:
            raise ValueError("zero_positive_pixels_in_center20")
    except Exception as error:
        result["resized_512x512_path"] = ""
        errors.append(f"mask:{type(error).__name__}:{error}")

    result["actions"] = ";".join(actions)
    result["reason"] = ";".join(errors)
    result["status"] = "ok" if not errors else "fail"
    return result


def build_512(args: argparse.Namespace) -> int:
    source = pd.read_csv(args.source_csv, low_memory=False)
    if args.plume_id:
        source = source[source["plume_id"].astype(str).eq(str(args.plume_id))].copy()
    if int(args.limit) > 0:
        source = source.head(int(args.limit)).copy()
    if source.empty:
        raise ValueError("no source rows selected for build-512")

    rows = source.to_dict("records")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(process_512_row, row, args): clean(row.get("plume_id")) for row in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            plume_id = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "plume_id": plume_id,
                    "status": "fail",
                    "reason": f"worker:{type(error).__name__}:{error}",
                }
            results.append(result)
            if index % max(1, int(args.progress_every)) == 0 or index == len(futures):
                ok_count = sum(item.get("status") == "ok" for item in results)
                log(
                    f"build512 {index}/{len(futures)} ok={ok_count} "
                    f"fail={len(results) - ok_count}"
                )

    qa = pd.DataFrame(results).sort_values("plume_id", kind="stable")
    atomic_csv(qa, Path(args.qa_csv))
    successful = qa[qa["status"].eq("ok")].set_index("plume_id")
    complete = source[source["plume_id"].astype(str).isin(successful.index)].copy()
    for column in [
        *[timepoint.std_col for timepoint in TIMEPOINTS],
        "resized_512x512_path",
        "mask_positive_pixels",
        "mask_center20_sum",
        "mask_action",
    ]:
        if column in successful.columns:
            complete[column] = complete["plume_id"].astype(str).map(successful[column])
    complete["has_all6_512"] = 1
    complete = complete.sort_values(["event_time", "plume_id"], kind="stable")
    atomic_csv(complete, Path(args.complete_csv))
    log(
        f"build512 complete={len(complete)}/{len(source)} qa={args.qa_csv} "
        f"output={args.complete_csv}"
    )
    return 0 if len(complete) == len(source) else 2


def choose_temporal_cutoff(
    frame: pd.DataFrame,
    target_ratio: float,
    min_ratio: float,
    max_ratio: float,
) -> tuple[pd.Timestamp, set[str], dict[str, Any]]:
    work = frame.copy()
    work["_event_time"] = work["event_time"].map(parse_time)
    if work["_event_time"].isna().any():
        examples = work.loc[
            work["_event_time"].isna(), ["plume_id", "event_time"]
        ].head(10)
        raise ValueError(f"invalid event_time values: {examples.to_dict('records')}")
    group_stats = work.groupby("event_group_id", as_index=False).agg(
        group_time=("_event_time", "min"),
        group_time_max=("_event_time", "max"),
        row_count=("plume_id", "size"),
    )
    spans = group_stats["group_time_max"] - group_stats["group_time"]
    if (spans > pd.Timedelta(days=1)).any():
        examples = group_stats.loc[spans > pd.Timedelta(days=1)].head(10).to_dict("records")
        raise ValueError(f"event_group_id spans more than one day: {examples}")

    group_stats["date"] = group_stats["group_time"].dt.floor("D")
    by_date = (
        group_stats.groupby("date", as_index=False)["row_count"].sum().sort_values("date")
    )
    by_date["cumulative"] = by_date["row_count"].cumsum()
    total = int(by_date["row_count"].sum())
    candidates: list[tuple[float, pd.Timestamp, float]] = []
    for row in by_date.itertuples(index=False):
        ratio = int(row.cumulative) / total
        if min_ratio <= ratio <= max_ratio:
            cutoff = pd.Timestamp(row.date) + pd.Timedelta(days=1)
            candidates.append((abs(ratio - target_ratio), cutoff, ratio))
    if not candidates:
        raise ValueError(f"no temporal cutoff yields ratio in [{min_ratio}, {max_ratio}]")
    _, cutoff, ratio = min(candidates, key=lambda item: (item[0], item[1]))
    train_groups = set(
        group_stats.loc[
            group_stats["group_time"] < cutoff, "event_group_id"
        ].astype(str)
    )
    audit = {
        "cutoff_utc": cutoff.isoformat(),
        "rule": "train event_group_time < cutoff; test event_group_time >= cutoff",
        "target_ratio": float(target_ratio),
        "actual_ratio": float(ratio),
        "total_rows": total,
        "total_event_groups": int(len(group_stats)),
    }
    return cutoff, train_groups, audit


def split_temporal(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.complete_csv, low_memory=False)
    required_columns(frame, ["plume_id", "event_group_id", "event_time"], Path(args.complete_csv))
    cutoff, train_groups, audit = choose_temporal_cutoff(
        frame,
        float(args.target_ratio),
        float(args.min_ratio),
        float(args.max_ratio),
    )
    group_values = frame["event_group_id"].astype(str)
    train = frame[group_values.isin(train_groups)].copy()
    test = frame[~group_values.isin(train_groups)].copy()
    train["split"] = "train"
    test["split"] = "test"
    event_overlap = set(train["event_group_id"].astype(str)) & set(
        test["event_group_id"].astype(str)
    )
    plume_overlap = set(train["plume_id"].astype(str)) & set(test["plume_id"].astype(str))
    if event_overlap or plume_overlap:
        raise RuntimeError(
            f"split leakage detected: event_groups={len(event_overlap)} plumes={len(plume_overlap)}"
        )

    output_root = Path(args.split_root)
    cutoff_tag = cutoff.strftime("%Y-%m-%d")
    train_path = output_root / f"s2_cdse_legacy512_train_cutoff_{cutoff_tag}.csv"
    test_path = output_root / f"s2_cdse_legacy512_test_cutoff_{cutoff_tag}.csv"
    atomic_csv(train, train_path)
    atomic_csv(test, test_path)
    audit.update(
        {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_ratio": len(train) / len(frame),
            "train_event_groups": int(train["event_group_id"].nunique()),
            "test_event_groups": int(test["event_group_id"].nunique()),
            "event_group_overlap": len(event_overlap),
            "plume_id_overlap": len(plume_overlap),
            "train_csv": str(train_path),
            "test_csv": str(test_path),
        }
    )
    atomic_json(audit, output_root / "split_audit.json")
    log(json.dumps(audit, ensure_ascii=False))
    return 0


def read_chw_512(path: Path) -> np.ndarray:
    if not file_ok(path):
        raise FileNotFoundError(path)
    array = to_bhw(tifffile.imread(path)).astype(np.float32, copy=False)
    if array.shape != (EXPECTED_BANDS, WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError(f"expected 12x512x512, got {array.shape}: {path}")
    return array


def read_mask_512(path: Path) -> np.ndarray:
    if not file_ok(path):
        raise FileNotFoundError(path)
    array = tifffile.imread(path)
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            array = np.max(to_bhw(array), axis=0)
    if array.shape != (WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError(f"expected 512x512 mask, got {array.shape}: {path}")
    return (np.nan_to_num(array) > 0).astype(np.uint8)


def legacy_positive_crop(
    rng: random.Random,
    patch_size: int = PATCH_SIZE,
    center_size: int = 20,
) -> tuple[int, int]:
    center_x = WINDOW_SIZE // 2
    center_y = WINDOW_SIZE // 2
    center_left = center_x - center_size // 2
    center_right = center_x + center_size // 2
    center_top = center_y - center_size // 2
    center_bottom = center_y + center_size // 2
    left_min = max(0, center_right - patch_size)
    left_max = min(center_left, WINDOW_SIZE - patch_size)
    top_min = max(0, center_bottom - patch_size)
    top_max = min(center_top, WINDOW_SIZE - patch_size)
    if left_min > left_max or top_min > top_max:
        raise ValueError(
            f"positive crop is impossible: patch={patch_size} center={center_size}"
        )
    return rng.randint(left_min, left_max), rng.randint(top_min, top_max)


def legacy_center_contained(
    x: int,
    y: int,
    patch_size: int = PATCH_SIZE,
    center_size: int = 10,
) -> bool:
    center_x = WINDOW_SIZE // 2
    center_y = WINDOW_SIZE // 2
    center_x1 = center_x - center_size // 2
    center_y1 = center_y - center_size // 2
    center_x2 = center_x + center_size // 2
    center_y2 = center_y + center_size // 2
    return (
        x <= center_x1
        and y <= center_y1
        and x + patch_size >= center_x2
        and y + patch_size >= center_y2
    )


def band_zero_too_much(array: np.ndarray, band_index: int, threshold: float) -> bool:
    if band_index < 0 or band_index >= array.shape[0]:
        raise ValueError(f"band index {band_index} is invalid for shape {array.shape}")
    band = array[band_index]
    return float(np.count_nonzero(band == 0)) / float(band.size) >= threshold


def crop_array(array: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    return array[..., y : y + size, x : x + size]


def tifffile_write_atomic(path: Path, array: np.ndarray, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{random.randrange(1 << 30)}.part.tif"
    )
    kwargs = {} if compression == "none" else {"compression": compression}
    try:
        tifffile.imwrite(temporary, array, **kwargs)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def patch_record(
    row: dict[str, Any],
    split: str,
    kind: str,
    index: int,
    x: int,
    y: int,
    label: int,
    image_paths: dict[str, str],
    mask_path: str,
    mask_sum: int,
) -> dict[str, Any]:
    sample_id = f"{row['plume_id']}__{kind}_{index:02d}_x{x}_y{y}"
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "plume_id": row["plume_id"],
        "event_group_id": row["event_group_id"],
        "cohort": row.get("cohort", "cdse_raw_or_legacy512"),
        "source_pattern": row.get("source_pattern", row.get("input_kind_pattern", "")),
        "split": split,
        "label": int(label),
        "crop_kind": kind,
        "crop_index": int(index),
        "crop_x": int(x),
        "crop_y": int(y),
        "plume_mask_sum": int(mask_sum),
        "path_plume": mask_path,
        "plume_mask_path": mask_path,
        "mask_path": mask_path,
        "mask_path_512": row["resized_512x512_path"],
        "plume_latitude": row["plume_latitude"],
        "plume_longitude": row["plume_longitude"],
        "latitude": row["plume_latitude"],
        "longitude": row["plume_longitude"],
        "event_time": row["event_time"],
        "datetime": row["event_time"],
        "source": "legacy_notebook_center20" if kind == "positive" else "legacy_notebook_random",
    }
    for timepoint in TIMEPOINTS:
        record[PATH_COLUMNS[timepoint.name]] = image_paths[timepoint.name]
        record[timepoint.image_time_col] = row[timepoint.image_time_col]
    record["image_path"] = record["path_t0"]
    record["s2_path"] = record["path_t0"]
    record["s2_-7_path"] = record["path_prev1"]
    record["s2_pre_path"] = record["path_seasonal"]
    record["s2_pre_pre_path"] = record["path_year"]
    return record


def process_crop_row(
    row: dict[str, Any], split: str, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], str]:
    plume_id = clean(row.get("plume_id"))
    try:
        images = {
            timepoint.name: read_chw_512(Path(clean(row.get(timepoint.std_col))))
            for timepoint in TIMEPOINTS
        }
        mask = read_mask_512(Path(clean(row.get("resized_512x512_path"))))
    except Exception as error:
        return [], f"read:{type(error).__name__}:{error}"
    if int(mask.sum()) <= 0 or int(mask[246:266, 246:266].sum()) <= 0:
        return [], "mask_missing_positive_pixels_at_center"

    patch_size = int(args.patch_size)
    rng = random.Random(stable_seed(int(args.seed), split, plume_id))
    coordinates: list[tuple[str, int, int, int, int]] = []
    for index in range(int(args.n_pos)):
        x, y = legacy_positive_crop(
            rng, patch_size=patch_size, center_size=int(args.positive_center_size)
        )
        coordinates.append(("positive", index, x, y, 1))
    for index in range(int(args.n_random)):
        x = rng.randint(0, WINDOW_SIZE - patch_size)
        y = rng.randint(0, WINDOW_SIZE - patch_size)
        label = int(
            legacy_center_contained(
                x,
                y,
                patch_size=patch_size,
                center_size=int(args.label_center_size),
            )
        )
        coordinates.append(("random", index, x, y, label))

    output_root = Path(args.out_32_root) / split / plume_id
    records: list[dict[str, Any]] = []
    issue_counts: dict[str, int] = {}
    for kind, index, x, y, label in coordinates:
        crops = {
            name: crop_array(image, x, y, patch_size) for name, image in images.items()
        }
        if any(crop.shape[-2:] != (patch_size, patch_size) for crop in crops.values()):
            issue_counts["bad_shape"] = issue_counts.get("bad_shape", 0) + 1
            continue
        if any(
            band_zero_too_much(
                crop, int(args.band_index), float(args.zero_ratio_thresh)
            )
            for crop in crops.values()
        ):
            issue_counts["band_zero_filter"] = issue_counts.get("band_zero_filter", 0) + 1
            continue

        mask_crop = crop_array(mask, x, y, patch_size)
        if label == 1 and int(mask_crop.sum()) <= 0:
            issue_counts["positive_zero_mask"] = issue_counts.get("positive_zero_mask", 0) + 1
            continue
        if label == 0:
            mask_crop = np.zeros_like(mask_crop)

        sample_dir = output_root / f"{kind}_{index:02d}_x{x}_y{y}"
        image_paths: dict[str, str] = {}
        try:
            for timepoint in TIMEPOINTS:
                image_path = sample_dir / timepoint.patch_filename
                if bool(args.overwrite) or not file_ok(image_path):
                    tifffile_write_atomic(
                        image_path,
                        crops[timepoint.name].astype(np.float32, copy=False),
                        str(args.compression),
                    )
                image_paths[timepoint.name] = str(image_path)
            plume_path = sample_dir / "plume.tif"
            if bool(args.overwrite) or not file_ok(plume_path):
                tifffile_write_atomic(
                    plume_path,
                    mask_crop.astype(np.uint8, copy=False),
                    str(args.compression),
                )
            records.append(
                patch_record(
                    row,
                    split,
                    kind,
                    index,
                    x,
                    y,
                    label,
                    image_paths,
                    str(plume_path),
                    int(mask_crop.sum()),
                )
            )
        except Exception as error:
            return [], f"write:{type(error).__name__}:{error}"

    expected = int(args.n_pos) + int(args.n_random)
    if bool(args.require_full_counts) and len(records) != expected:
        return [], f"incomplete:{len(records)}/{expected};issues={issue_counts}"
    message = "" if not issue_counts else json.dumps(issue_counts, sort_keys=True)
    return records, message


def normalized_patch_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates("sample_id", keep="last")
    frame = frame.sort_values(
        ["event_group_id", "plume_id", "crop_kind", "crop_index"], kind="stable"
    ).reset_index(drop=True)
    frame["id"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    return frame


def crop_one_split(split: str, input_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(input_path, low_memory=False)
    required_columns(
        frame,
        [
            "plume_id",
            "event_group_id",
            "resized_512x512_path",
            *[timepoint.std_col for timepoint in TIMEPOINTS],
        ],
        input_path,
    )
    if int(args.limit) > 0:
        frame = frame.head(int(args.limit)).copy()

    output_csv = Path(args.out_32_root) / f"{split}_patches_32.csv"
    issue_csv = Path(args.out_32_root) / f"{split}_crop_issues.csv"
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if bool(args.resume) and not bool(args.overwrite):
        if output_csv.exists() and output_csv.stat().st_size > 0:
            records = pd.read_csv(output_csv, low_memory=False).to_dict("records")
        if issue_csv.exists() and issue_csv.stat().st_size > 0:
            issues = pd.read_csv(issue_csv, low_memory=False).to_dict("records")
    done_plumes = {clean(record.get("plume_id")) for record in records}
    done_plumes.update(clean(issue.get("plume_id")) for issue in issues)
    rows = [
        row
        for row in frame.to_dict("records")
        if clean(row.get("plume_id")) not in done_plumes
    ]
    log(
        f"crop32 {split}: rows={len(frame)} resume_done={len(done_plumes)} "
        f"to_process={len(rows)}"
    )

    def flush() -> None:
        atomic_csv(normalized_patch_frame(records), output_csv)
        issue_frame = pd.DataFrame(issues, columns=["plume_id", "message", "sample_count"])
        atomic_csv(issue_frame, issue_csv)

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(process_crop_row, row, split, args): clean(row.get("plume_id"))
            for row in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            plume_id = futures[future]
            try:
                new_records, message = future.result()
            except Exception as error:
                new_records, message = [], f"worker:{type(error).__name__}:{error}"
            records.extend(new_records)
            if message or not new_records:
                issues.append(
                    {
                        "plume_id": plume_id,
                        "message": message or "no_samples",
                        "sample_count": len(new_records),
                    }
                )
            if index % max(1, int(args.progress_every)) == 0 or index == len(futures):
                flush()
                log(
                    f"crop32 {split} {index}/{len(futures)} samples={len(records)} "
                    f"issues={len(issues)}"
                )
    if not rows:
        flush()
    output = normalized_patch_frame(records)
    label_counts = output["label"].value_counts().sort_index().to_dict() if not output.empty else {}
    log(f"crop32 wrote {output_csv} rows={len(output)} labels={label_counts}")
    return output


def crop_32(args: argparse.Namespace) -> int:
    train = crop_one_split("train", Path(args.train_csv), args)
    test = crop_one_split("test", Path(args.test_csv), args)
    combined = pd.concat([train, test], ignore_index=True)
    atomic_csv(combined, Path(args.out_32_root) / "all_patches_32.csv")
    return 0


def resize_bilinear_chw(
    image: np.ndarray, out_height: int = TARGET_SIZE, out_width: int = TARGET_SIZE
) -> np.ndarray:
    image = to_bhw(image).astype(np.float32, copy=False)
    _, height, width = image.shape
    y = np.linspace(0, height - 1, out_height)
    x = np.linspace(0, width - 1, out_width)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    weight_x = (x - x0)[None, :]
    weight_y = (y - y0)[:, None]
    top_left = image[:, y0[:, None], x0[None, :]]
    top_right = image[:, y0[:, None], x1[None, :]]
    bottom_left = image[:, y1[:, None], x0[None, :]]
    bottom_right = image[:, y1[:, None], x1[None, :]]
    output = (
        top_left * (1 - weight_x) * (1 - weight_y)
        + top_right * weight_x * (1 - weight_y)
        + bottom_left * (1 - weight_x) * weight_y
        + bottom_right * weight_x * weight_y
    )
    return output.astype(np.float32, copy=False)


def map_resized_path(source: str, old_root: Path, new_root: Path) -> Path:
    path = Path(source)
    try:
        relative = path.relative_to(old_root)
    except ValueError as error:
        raise ValueError(f"path is outside the 32 root: {path}") from error
    return new_root / relative


def resized_image_ok(path: Path, target_size: int) -> bool:
    if not file_ok(path):
        return False
    try:
        with tifffile.TiffFile(path) as tif:
            shape = tif.series[0].shape
        return shape == (EXPECTED_BANDS, target_size, target_size)
    except Exception:
        return False


def resize_one_image(
    source: Path,
    destination: Path,
    target_size: int,
    overwrite: bool,
    compression: str,
) -> tuple[bool, str]:
    try:
        if not overwrite and resized_image_ok(destination, target_size):
            return True, "exists"
        image = tifffile.imread(source)
        output = resize_bilinear_chw(image, target_size, target_size)
        tifffile_write_atomic(destination, output, compression)
        return True, "written"
    except Exception as error:
        return False, f"{type(error).__name__}:{error}"


def resize_one_split(split: str, args: argparse.Namespace) -> pd.DataFrame:
    old_root = Path(args.out_32_root)
    new_root = Path(args.out_224_root)
    input_csv = old_root / f"{split}_patches_32.csv"
    frame = pd.read_csv(input_csv, low_memory=False)
    if int(args.limit) > 0:
        frame = frame.head(int(args.limit)).copy()

    source_destination: dict[str, str] = {}
    for column in PATH_COLUMNS.values():
        new_values: list[str] = []
        for source in frame[column].astype(str):
            destination = map_resized_path(source, old_root, new_root)
            source_destination[source] = str(destination)
            new_values.append(str(destination))
        frame[column] = new_values
    frame["image_path"] = frame["path_t0"]
    frame["s2_path"] = frame["path_t0"]
    frame["s2_-7_path"] = frame["path_prev1"]
    frame["s2_pre_path"] = frame["path_seasonal"]
    frame["s2_pre_pre_path"] = frame["path_year"]

    tasks = sorted(source_destination.items())
    failures: list[dict[str, str]] = []
    completed = 0
    batch_size = max(1, int(args.batch_files))
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            futures = {
                pool.submit(
                    resize_one_image,
                    Path(source),
                    Path(destination),
                    int(args.target_size),
                    bool(args.overwrite),
                    str(args.compression),
                ): (source, destination)
                for source, destination in batch
            }
            for future in as_completed(futures):
                source, destination = futures[future]
                valid, message = future.result()
                completed += 1
                if not valid:
                    failures.append(
                        {"source": source, "destination": destination, "message": message}
                    )
                if completed % max(1, int(args.progress_every)) == 0:
                    log(
                        f"resize224 {split} {completed}/{len(tasks)} failures={len(failures)}"
                    )
    if failures:
        atomic_csv(pd.DataFrame(failures), new_root / f"{split}_resize_failures.csv")
        raise RuntimeError(f"resize224 {split} failed files={len(failures)}")
    output_csv = new_root / f"{split}_patches_224.csv"
    atomic_csv(frame, output_csv)
    log(f"resize224 wrote {output_csv} rows={len(frame)} images={len(tasks)}")
    return frame


def resize_224(args: argparse.Namespace) -> int:
    train = resize_one_split("train", args)
    test = resize_one_split("test", args)
    atomic_csv(
        pd.concat([train, test], ignore_index=True),
        Path(args.out_224_root) / "all_patches_224.csv",
    )
    return 0


def sampled_path_census(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rows_per_split: int,
    workers: int,
) -> dict[str, Any]:
    sampled_frames: list[pd.DataFrame] = []
    for frame in [train, test]:
        if rows_per_split <= 0 or len(frame) <= rows_per_split:
            sampled_frames.append(frame)
        else:
            sampled_frames.append(frame.sample(n=rows_per_split, random_state=0))
    sampled = pd.concat(sampled_frames, ignore_index=True)
    columns = [*PATH_COLUMNS.values(), "path_plume"]
    paths = [clean(value) for column in columns for value in sampled[column]]
    flags = parallel_file_flags(paths, workers)
    missing = sorted(path for path in set(paths) if not flags.get(path, False))
    return {
        "rows_sampled": len(sampled),
        "files_sampled": len(paths),
        "unique_files_sampled": len(set(paths)),
        "missing_files": len(missing),
        "missing_examples": missing[:20],
    }


def audit_dataset(args: argparse.Namespace) -> int:
    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    required = [
        "plume_id",
        "event_group_id",
        "label",
        "plume_mask_sum",
        *PATH_COLUMNS.values(),
        "path_plume",
        *[timepoint.image_time_col for timepoint in TIMEPOINTS],
    ]
    required_columns(train, required, Path(args.train_csv))
    required_columns(test, required, Path(args.test_csv))

    event_overlap = set(train["event_group_id"].astype(str)) & set(
        test["event_group_id"].astype(str)
    )
    plume_overlap = set(train["plume_id"].astype(str)) & set(test["plume_id"].astype(str))
    train_labels = set(pd.to_numeric(train["label"], errors="coerce").dropna().astype(int))
    test_labels = set(pd.to_numeric(test["label"], errors="coerce").dropna().astype(int))
    train_positive_zero = int(
        ((train["label"] == 1) & (train["plume_mask_sum"] <= 0)).sum()
    )
    test_positive_zero = int(
        ((test["label"] == 1) & (test["plume_mask_sum"] <= 0)).sum()
    )
    train_negative_nonzero = int(
        ((train["label"] == 0) & (train["plume_mask_sum"] > 0)).sum()
    )
    test_negative_nonzero = int(
        ((test["label"] == 0) & (test["plume_mask_sum"] > 0)).sum()
    )

    invalid_timestamps: dict[str, int] = {}
    for timepoint in TIMEPOINTS:
        combined = pd.concat(
            [train[timepoint.image_time_col], test[timepoint.image_time_col]], ignore_index=True
        )
        invalid_timestamps[timepoint.image_time_col] = int(
            combined.map(parse_time).isna().sum()
        )

    path_census = sampled_path_census(
        train,
        test,
        int(args.path_audit_rows),
        int(args.path_stat_workers),
    )
    audit = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_labels": train["label"].value_counts().sort_index().to_dict(),
        "test_labels": test["label"].value_counts().sort_index().to_dict(),
        "train_plumes": int(train["plume_id"].nunique()),
        "test_plumes": int(test["plume_id"].nunique()),
        "train_event_groups": int(train["event_group_id"].nunique()),
        "test_event_groups": int(test["event_group_id"].nunique()),
        "event_group_overlap": len(event_overlap),
        "plume_id_overlap": len(plume_overlap),
        "train_zero_mask_positive_labels": train_positive_zero,
        "test_zero_mask_positive_labels": test_positive_zero,
        "train_nonzero_mask_negative_labels": train_negative_nonzero,
        "test_nonzero_mask_negative_labels": test_negative_nonzero,
        "invalid_timestamps": invalid_timestamps,
        "path_census": path_census,
    }
    errors: list[str] = []
    if event_overlap or plume_overlap:
        errors.append("train/test event or plume leakage")
    if not train_labels.issubset({0, 1}) or not test_labels.issubset({0, 1}):
        errors.append(f"invalid labels train={train_labels} test={test_labels}")
    if train_positive_zero or test_positive_zero:
        errors.append("positive labels with empty plume masks")
    if train_negative_nonzero or test_negative_nonzero:
        errors.append("negative labels with non-empty plume masks")
    if any(invalid_timestamps.values()):
        errors.append(f"invalid timestamps: {invalid_timestamps}")
    if path_census["missing_files"]:
        errors.append(f"missing sampled files: {path_census['missing_examples']}")
    audit["status"] = "fail" if errors else "ok"
    audit["errors"] = errors
    atomic_json(audit, Path(args.audit_json))
    log(json.dumps(audit, ensure_ascii=False))
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "build-manifest", description="Validate and freeze the six-time source mapping."
    )
    manifest.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    manifest.add_argument("--validated-512-csv", default=str(DEFAULT_VALIDATED_512_CSV))
    manifest.add_argument("--main-csv", default=str(DEFAULT_MAIN_CSV))
    manifest.add_argument("--cm-root", default=str(DEFAULT_CM_ROOT))
    manifest.add_argument("--out-512-root", default=str(DEFAULT_OUT_512_ROOT))
    manifest.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    manifest.add_argument("--source-audit-json", default=str(DEFAULT_SOURCE_AUDIT))
    manifest.add_argument("--raw-prefix", default=DEFAULT_RAW_PREFIX)
    manifest.add_argument("--legacy-512-prefix", default=DEFAULT_LEGACY_512_PREFIX)
    manifest.add_argument("--validated-512-prefix", default=DEFAULT_VALIDATED_512_PREFIX)
    manifest.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    manifest.add_argument(
        "--expected-all-raw-rows", type=int, default=EXPECTED_ALL_RAW_ROWS
    )
    manifest.add_argument("--expected-mixed-rows", type=int, default=EXPECTED_MIXED_ROWS)
    manifest.add_argument("--stat-workers", type=int, default=32)
    manifest.set_defaults(func=build_manifest)

    build = subparsers.add_parser(
        "build-512",
        description="Create the six 512 images and exact legacy Carbon Mapper mask.",
    )
    build.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    build.add_argument("--qa-csv", default=str(DEFAULT_QA_512_CSV))
    build.add_argument("--complete-csv", default=str(DEFAULT_COMPLETE_512_CSV))
    build.add_argument(
        "--image-mode",
        choices=[
            "standardize",
            "reuse-validated",
            "hardlink-validated",
            "symlink-validated",
            "copy-validated",
        ],
        default="reuse-validated",
        help=(
            "standardize reruns notebook Cell 4 from raw/legacy inputs; reuse-validated "
            "references the verified Cell-4 outputs directly; other validated modes try "
            "to materialize links or copies under the new root."
        ),
    )
    build.add_argument("--workers", type=int, default=16)
    build.add_argument("--progress-every", type=int, default=50)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--overwrite-masks", action="store_true")
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--plume-id", default="")
    build.set_defaults(func=build_512)

    split = subparsers.add_parser(
        "split", description="Choose an event-safe temporal train/test cutoff."
    )
    split.add_argument("--complete-csv", default=str(DEFAULT_COMPLETE_512_CSV))
    split.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    split.add_argument("--target-ratio", type=float, default=0.85)
    split.add_argument("--min-ratio", type=float, default=0.80)
    split.add_argument("--max-ratio", type=float, default=0.90)
    split.set_defaults(func=split_temporal)

    crop = subparsers.add_parser(
        "crop-32", description="Copy the old notebook crop logic to all six timepoints."
    )
    crop.add_argument("--train-csv", required=True)
    crop.add_argument("--test-csv", required=True)
    crop.add_argument("--out-32-root", default=str(DEFAULT_OUT_32_ROOT))
    crop.add_argument("--workers", type=int, default=16)
    crop.add_argument("--progress-every", type=int, default=50)
    crop.add_argument("--seed", type=int, default=20260119)
    crop.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    crop.add_argument("--n-pos", type=int, default=16)
    crop.add_argument("--n-random", type=int, default=16)
    crop.add_argument("--positive-center-size", type=int, default=20)
    crop.add_argument("--label-center-size", type=int, default=10)
    crop.add_argument("--band-index", type=int, default=11)
    crop.add_argument("--zero-ratio-thresh", type=float, default=0.20)
    crop.add_argument(
        "--compression",
        choices=["none", "deflate", "zlib", "lzma", "zstd"],
        default="deflate",
    )
    crop.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    crop.add_argument("--overwrite", action="store_true")
    crop.add_argument("--require-full-counts", action="store_true")
    crop.add_argument("--limit", type=int, default=0)
    crop.set_defaults(func=crop_32)

    resize = subparsers.add_parser(
        "resize-224", description="Resize the six 32-pixel image stacks to 224."
    )
    resize.add_argument("--out-32-root", default=str(DEFAULT_OUT_32_ROOT))
    resize.add_argument("--out-224-root", default=str(DEFAULT_OUT_224_ROOT))
    resize.add_argument("--workers", type=int, default=8)
    resize.add_argument("--batch-files", type=int, default=512)
    resize.add_argument("--progress-every", type=int, default=1000)
    resize.add_argument("--target-size", type=int, default=TARGET_SIZE)
    resize.add_argument(
        "--compression",
        choices=["none", "deflate", "zlib", "lzma", "zstd"],
        default="deflate",
    )
    resize.add_argument("--overwrite", action="store_true")
    resize.add_argument("--limit", type=int, default=0)
    resize.set_defaults(func=resize_224)

    audit = subparsers.add_parser(
        "audit", description="Fail on leakage, invalid labels, timestamps, or sampled paths."
    )
    audit.add_argument("--train-csv", required=True)
    audit.add_argument("--test-csv", required=True)
    audit.add_argument("--audit-json", required=True)
    audit.add_argument("--path-audit-rows", type=int, default=1000)
    audit.add_argument("--path-stat-workers", type=int, default=32)
    audit.set_defaults(func=audit_dataset)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
