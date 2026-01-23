import argparse
import csv
import json
import re
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

BAND_REGEX = re.compile(r"B([0-9A-Za-z]+)_20m\.jp2$")
CSV_COLUMNS = ["plume_id", "slot", "s2_datetime", "safe_dir", "tif_path", "cloud_cover"]
DEFAULT_RAW_ROOT = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_S2")
DEFAULT_OUTPUT_ROOT = Path("/data2/yuyao/methane_emission/carbonmapper_data_s2_l2a_reclip")
DEFAULT_OUTPUT_CSV = Path("/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_s2_reclip.csv")
MISSING_RATIO_THRESHOLD = 0.8
WINDOW_SIZE = 512
BAND_COUNT = 12


def parse_iso_datetime(value: str) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_cloud_cover(safe_dir: Path) -> Optional[float]:
    mtd_path = safe_dir / "MTD_MSIL2A.xml"
    if not mtd_path.exists():
        return None
    try:
        tree = ET.parse(str(mtd_path))
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag.endswith("Cloud_Coverage_Assessment"):
                if elem.text and elem.text.strip():
                    return float(elem.text.strip())
    except Exception:
        return None
    return None


def find_safe_dir(path_str: str, raw_root: Path) -> Optional[Path]:
    if not isinstance(path_str, str) or len(path_str) == 0:
        return None
    given_path = Path(path_str)
    if given_path.exists():
        return given_path
    candidate = raw_root / given_path.name
    if candidate.exists():
        return candidate
    if given_path.name.endswith(".SAFE"):
        for cand in raw_root.glob(f"**/{given_path.name}"):
            if cand.is_dir():
                return cand
    return None


def load_boundaries(row: Dict) -> Optional[List[float]]:
    bounds_value = row.get("plume_bounds")
    if isinstance(bounds_value, str):
        try:
            return json.loads(bounds_value)
        except json.JSONDecodeError:
            return None
    if isinstance(bounds_value, Sequence):
        return list(bounds_value)
    return None


def compute_window(dataset, bounds: Sequence[float]) -> Optional[Window]:
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bounds
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    center_x, center_y = transformer.transform(center_lon, center_lat)
    center_col, center_row = (~dataset.transform) * (center_x, center_y)
    half = WINDOW_SIZE // 2
    col_start = center_col - half
    row_start = center_row - half
    return Window(col_start, row_start, WINDOW_SIZE, WINDOW_SIZE)


def clip_product(safe_dir: Path, bounds: Sequence[float]) -> Optional[tuple[np.ndarray, rasterio.Affine, object]]:
    jp2_paths = sorted(safe_dir.rglob("*_20m.jp2"))
    if not jp2_paths:
        return None
    stacked = None
    reference_transform = None
    reference_crs = None
    target_window: Optional[Window] = None
    fill_value = 0
    for jp2_path in jp2_paths:
        match = BAND_REGEX.search(jp2_path.name)
        if not match:
            continue
        band_token = match.group(1).upper()
        band_index = 8 if band_token == "8A" else int(band_token)
        if band_index < 1 or band_index > BAND_COUNT:
            continue
        with rasterio.open(jp2_path) as dataset:
            if dataset.crs is None:
                continue
            if target_window is None:
                target_window = compute_window(dataset, bounds)
                if target_window is None:
                    return None
                reference_transform = dataset.window_transform(target_window)
                reference_crs = dataset.crs
                fill_value = dataset.nodata if dataset.nodata is not None else 0
            clipped = dataset.read(1, window=target_window, boundless=True, fill_value=fill_value)
            if stacked is None:
                stacked = np.zeros((BAND_COUNT, WINDOW_SIZE, WINDOW_SIZE), dtype=clipped.dtype)
            stacked[band_index - 1] = clipped
    if stacked is None:
        return None
    nodata_mask = stacked == fill_value
    missing_ratio = nodata_mask.sum() / stacked.size
    if missing_ratio > MISSING_RATIO_THRESHOLD:
        return None
    return stacked, reference_transform, reference_crs


def append_record(record: Dict, args) -> None:
    with args.csv_lock:
        need_header = not args.csv_initialized
        with args.output_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if need_header:
                writer.writeheader()
                args.csv_initialized = True
            writer.writerow(record)


def process_row(row: Dict, args) -> None:
    plume_id = row.get("plume_id")
    plume_bounds = load_boundaries(row)
    if plume_bounds is None:
        return []
    event_dt = parse_iso_datetime(row.get("datetime"))
    if event_dt is None:
        return []
    for slot in range(1, 4):
        dt = parse_iso_datetime(row.get(f"s2_{slot}_datetime"))
        path_str = row.get(f"s2_{slot}_path")
        if dt is None or path_str is None:
            continue
        delta_hours = abs((dt - event_dt).total_seconds()) / 3600.0
        if delta_hours > 24.0:
            continue
        safe_dir = find_safe_dir(path_str, args.raw_root)
        if safe_dir is None:
            continue
        cloud_cover = load_cloud_cover(safe_dir)
        if cloud_cover is None or cloud_cover >= 20.0:
            continue
        clipped = clip_product(safe_dir, plume_bounds)
        if clipped is None:
            continue
        stacked, transform, crs = clipped
        plume_dir = args.output_root / str(plume_id)
        plume_dir.mkdir(parents=True, exist_ok=True)
        tif_stamp = dt.strftime("%Y%m%dT%H%M%SZ")
        tif_path = plume_dir / f"s2_{tif_stamp}.tif"
        tif_path_str = str(tif_path)
        with args.path_lock:
            if tif_path_str in args.recorded_paths:
                continue
            args.recorded_paths.add(tif_path_str)
        profile = {
            "driver": "GTiff",
            "height": WINDOW_SIZE,
            "width": WINDOW_SIZE,
            "count": BAND_COUNT,
            "dtype": stacked.dtype,
            "transform": transform,
            "crs": crs,
        }
        if not tif_path.exists():
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(stacked)
        record = {
            "plume_id": plume_id,
            "slot": slot,
            "s2_datetime": dt.isoformat().replace("+00:00", "Z"),
            "safe_dir": str(safe_dir),
            "tif_path": tif_path_str,
            "cloud_cover": cloud_cover,
        }
        append_record(record, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reclip Sentinel-2 SAFE archives into plume-centered chips.")
    parser.add_argument("--csv", required=True, help="Input merged CSV with plume metadata.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Directory containing SAFE folders.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination directory for new chips.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Where to write summary CSV.")
    parser.add_argument("--workers", type=int, default=4, help="Number of threads for parallel processing.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.raw_root = Path(args.raw_root)
    args.output_root = Path(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_csv = Path(args.output_csv)
    if args.output_csv.exists() and args.output_csv.stat().st_size > 0:
        existing_df = pd.read_csv(args.output_csv)
    else:
        existing_df = pd.DataFrame()
    existing_paths = set(existing_df["tif_path"].astype(str)) if not existing_df.empty else set()
    args.recorded_paths = set(existing_paths)
    initial_count = len(existing_paths)
    args.csv_lock = threading.Lock()
    args.path_lock = threading.Lock()
    args.csv_initialized = args.output_csv.exists() and args.output_csv.stat().st_size > 0
    df = pd.read_csv(args.csv)
    rows = df.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process_row, row, args) for row in rows]
        for fut in futures:
            fut.result()
    total_new = len(args.recorded_paths) - initial_count
    print(f"Completed reclipping. New entries: {max(0, total_new)}")


if __name__ == "__main__":
    main()
