"""Crop Sentinel-2 and Landsat-8/9 chips into fixed-size patches aligned with Carbon Mapper plumes."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt

import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    REPO_ROOT
    / "carbon_mapper_data"
    / "csvs"
    / "merged_file_with_s2_l8_filtered_with_flags_low_cloud_only.csv"
)
MASK_ROOT = REPO_ROOT / "carbon_mapper_data_masks"
S2_ROOT = REPO_ROOT / "carbonmapper_data_s2_l2a"
L89_ROOT = REPO_ROOT / "carbonmapper_data_l89_l2sp"
OUTPUT_S2_ROOT = REPO_ROOT / "CM_s2_l8_s2"
OUTPUT_L89_ROOT = REPO_ROOT / "CM_s2_l8_l89"
CSV_OUTPUT_DIR = REPO_ROOT / "data_csv"


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _parse_iso8601(value: object) -> Optional[datetime]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_float(value: object) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_timestamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _ensure_channel_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        if arr.shape[0] <= 64 and arr.shape[1] == arr.shape[2]:
            return arr
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Unexpected chip array shape: {arr.shape}")


def _is_valid_chip_path(path: Path) -> bool:
    """Return True if path points to an existing GeoTIFF chip."""
    try:
        return path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    except OSError:
        return False


def _read_chip(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    arr = _ensure_channel_first(arr)
    return arr


def _read_mask_with_profile(mask_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    try:
        with rasterio.open(mask_path) as src:
            data = src.read()
            mask_arr = data[-1] if data.shape[0] > 1 else data[0]
            profile = {
                "height": src.height,
                "width": src.width,
                "transform": src.transform,
                "crs": src.crs,
            }
            return mask_arr, profile
    except Exception:
        arr = tifffile.imread(mask_path)
        if arr.ndim == 3:
            arr = arr[..., -1]
        profile = {
            "height": arr.shape[0],
            "width": arr.shape[1],
            "transform": None,
            "crs": None,
        }
        return arr, profile


def _resize_mask_nearest(mask_arr: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if mask_arr.shape == (target_height, target_width):
        return mask_arr
    src_h, src_w = mask_arr.shape
    if src_h == 0 or src_w == 0 or target_height <= 0 or target_width <= 0:
        raise ValueError("Invalid mask resize request.")
    y_idx = np.linspace(0, src_h - 1, target_height)
    x_idx = np.linspace(0, src_w - 1, target_width)
    y_idx = np.clip(np.round(y_idx).astype(int), 0, src_h - 1)
    x_idx = np.clip(np.round(x_idx).astype(int), 0, src_w - 1)
    return mask_arr[np.ix_(y_idx, x_idx)]


def _load_mask_like_image(mask_path: Path, ref_height: int, ref_width: int) -> np.ndarray:
    mask_arr, mask_profile = _read_mask_with_profile(mask_path)
    if mask_profile["height"] == ref_height and mask_profile["width"] == ref_width:
        return mask_arr
    ref_profile = {
        "height": ref_height,
        "width": ref_width,
        "transform": None,
        "crs": None,
    }
    if (
        mask_profile["transform"] is not None
        and mask_profile["crs"] is not None
        and ref_profile["transform"] is not None
        and ref_profile["crs"] is not None
    ):
        dest = np.zeros((ref_height, ref_width), dtype=np.float32)
        reproject(
            source=mask_arr,
            destination=dest,
            src_transform=mask_profile["transform"],
            src_crs=mask_profile["crs"],
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
        return dest
    return _resize_mask_nearest(mask_arr, ref_height, ref_width)


def _clamp_window(top: int, left: int, height: int, width: int, size: int) -> Tuple[int, int]:
    max_top = max(0, height - size)
    max_left = max(0, width - size)
    return max(0, min(top, max_top)), max(0, min(left, max_left))


def _select_plume_windows(
    plume_mask: np.ndarray, crop_size: int, max_count: int
) -> List[Tuple[int, int]]:
    plume_pixels = np.argwhere(plume_mask)
    if plume_pixels.size == 0:
        return []
    height, width = plume_mask.shape
    rows = plume_pixels[:, 0]
    cols = plume_pixels[:, 1]
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    centroid_row = rows.mean()
    centroid_col = cols.mean()
    row_start = max(0, row_min - crop_size)
    row_end = min(height - crop_size, row_max)
    col_start = max(0, col_min - crop_size)
    col_end = min(width - crop_size, col_max)
    stride = max(4, crop_size // 2)
    candidates: List[Tuple[float, int, int]] = []
    for top in range(row_start, row_end + 1, stride):
        for left in range(col_start, col_end + 1, stride):
            mask_patch = plume_mask[top : top + crop_size, left : left + crop_size]
            cover = mask_patch.sum()
            if cover == 0:
                continue
            center_row = top + crop_size / 2.0
            center_col = left + crop_size / 2.0
            dist = math.hypot(center_row - centroid_row, center_col - centroid_col)
            score = float(cover) - 0.1 * dist
            candidates.append((score, top, left))
    if not candidates:
        center_top = int(round(centroid_row - crop_size / 2.0))
        center_left = int(round(centroid_col - crop_size / 2.0))
        center_top, center_left = _clamp_window(center_top, center_left, height, width, crop_size)
        mask_patch = plume_mask[center_top : center_top + crop_size, center_left : center_left + crop_size]
        if mask_patch.sum() > 0:
            candidates.append((float(mask_patch.sum()), center_top, center_left))
    candidates.sort(key=lambda item: item[0], reverse=True)
    unique_windows: List[Tuple[int, int]] = []
    seen = set()
    for _, top, left in candidates:
        key = (top, left)
        if key in seen:
            continue
        seen.add(key)
        unique_windows.append(key)
        if len(unique_windows) >= max_count:
            break
    return unique_windows


def _select_background_windows(
    plume_mask: np.ndarray, crop_size: int, max_count: int
) -> List[Tuple[int, int]]:
    height, width = plume_mask.shape
    if height < crop_size or width < crop_size:
        return []
    safe_mask = ~plume_mask
    distance_map = distance_transform_edt(safe_mask)
    stride = max(4, crop_size // 2)
    candidates: List[Tuple[float, float, int, int]] = []
    for top in range(0, height - crop_size + 1, stride):
        for left in range(0, width - crop_size + 1, stride):
            mask_patch = plume_mask[top : top + crop_size, left : left + crop_size]
            if mask_patch.any():
                continue
            dist_patch = distance_map[top : top + crop_size, left : left + crop_size]
            min_dist = dist_patch.min()
            mean_dist = dist_patch.mean()
            candidates.append((min_dist, mean_dist, top, left))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    unique_windows: List[Tuple[int, int]] = []
    seen = set()
    for _, _, top, left in candidates:
        key = (top, left)
        if key in seen:
            continue
        seen.add(key)
        unique_windows.append(key)
        if len(unique_windows) >= max_count:
            break
    return unique_windows


def _plume_centroid(plume_mask: np.ndarray) -> Optional[Tuple[float, float]]:
    plume_pixels = np.argwhere(plume_mask)
    if plume_pixels.size == 0:
        return None
    rows = plume_pixels[:, 0]
    cols = plume_pixels[:, 1]
    return rows.mean(), cols.mean()


def _center_biased_positive_windows(
    plume_mask: np.ndarray,
    crop_size: int,
    count: int,
) -> List[Tuple[int, int]]:
    if count <= 0:
        return []
    centroid = _plume_centroid(plume_mask)
    candidates = _select_plume_windows(plume_mask, crop_size, max(count * 4, count))
    if not candidates:
        return []
    if centroid is None:
        return candidates[:count]
    centroid_row, centroid_col = centroid
    candidates.sort(
        key=lambda coords: math.hypot(
            (coords[0] + crop_size / 2.0) - centroid_row,
            (coords[1] + crop_size / 2.0) - centroid_col,
        )
    )
    return candidates[:count]


def _random_background_windows(
    plume_mask: np.ndarray,
    crop_size: int,
    count: int,
) -> List[Tuple[int, int]]:
    if count <= 0:
        return []
    candidates = _select_background_windows(plume_mask, crop_size, max(count * 4, count))
    if not candidates:
        return []
    random.shuffle(candidates)
    return candidates[:count]


@dataclass
class ChipSlot:
    sensor: str  # "s2" or "l89"
    plume_id: str
    image_path: Path
    image_dt: datetime
    event_dt: datetime
    cloud_cover: Optional[float]
    scene_id: Optional[str]
    row: Dict[str, object]

    def image_tag(self) -> str:
        stamp = self.image_dt.strftime("%Y%m%dT%H%M%SZ")
        prefix = "s2" if self.sensor == "s2" else "l89"
        if self.scene_id:
            safe_scene = str(self.scene_id).replace("/", "_")
            return f"{prefix}_{safe_scene}_{stamp}"
        return f"{prefix}_{stamp}"

    def time_offset_hours(self) -> float:
        delta = self.image_dt - self.event_dt
        return abs(delta.total_seconds()) / 3600.0


def _gather_s2_slots(
    row: Dict[str, object],
    event_dt: datetime,
    hours_window: float,
    max_slots: int,
) -> List[ChipSlot]:
    plume_id = row["plume_id"]
    slots: List[ChipSlot] = []
    for idx in (1, 2, 3):
        dt = _parse_iso8601(row.get(f"s2_{idx}_datetime"))
        if dt is None:
            continue
        delta_seconds = (dt - event_dt).total_seconds()
        if delta_seconds < 0 or delta_seconds > hours_window * 3600.0:
            continue
        candidates: List[Path] = []
        path_hint = row.get(f"s2_{idx}_path")
        if isinstance(path_hint, str) and path_hint.strip():
            candidate_path = Path(path_hint.strip())
            if _is_valid_chip_path(candidate_path):
                candidates.append(candidate_path)
        stamp = dt.strftime("%Y%m%dT%H%M%SZ")
        candidates.append(S2_ROOT / plume_id / f"s2_{stamp}.tif")
        image_path = None
        for candidate in candidates:
            candidate_path = candidate
            if not isinstance(candidate_path, Path):
                candidate_path = Path(candidate_path)
            if _is_valid_chip_path(candidate_path):
                image_path = candidate_path
                break
        if image_path is None:
            print(f"[WARN] {plume_id}: missing S2 chip for {stamp}")
            continue
        slots.append(
            ChipSlot(
                sensor="s2",
                plume_id=plume_id,
                image_path=image_path,
                image_dt=dt,
                event_dt=event_dt,
                cloud_cover=_to_float(row.get(f"s2_{idx}_cloud_cover")),
                scene_id=None,
                row=row,
            )
        )
    slots.sort(key=lambda slot: slot.time_offset_hours())
    if max_slots is not None and max_slots > 0:
        return slots[:max_slots]
    return slots


def _gather_l89_slots(
    row: Dict[str, object],
    event_dt: datetime,
    hours_window: float,
) -> List[ChipSlot]:
    plume_id = row["plume_id"]
    slots: List[ChipSlot] = []
    for idx in (1, 2, 3):
        dt = _parse_iso8601(row.get(f"l8_{idx}_datetime"))
        if dt is None:
            continue
        if abs((dt - event_dt).total_seconds()) > hours_window * 3600.0:
            continue
        tif_path = row.get(f"l8_{idx}_tif")
        scene_id = row.get(f"l8_{idx}_scene_id")
        path: Optional[Path] = None
        if not _is_missing(tif_path):
            candidate = Path(str(tif_path).strip())
            if candidate.exists():
                path = candidate
        if path is None and not _is_missing(scene_id):
            fallback = L89_ROOT / plume_id / f"l8_{scene_id}.tif"
            if fallback.exists():
                path = fallback
        if path is None:
            print(f"[WARN] {plume_id}: missing L8/L9 chip for slot {idx}")
            continue
        slots.append(
            ChipSlot(
                sensor="l89",
                plume_id=plume_id,
                image_path=path,
                image_dt=dt,
                event_dt=event_dt,
                cloud_cover=_to_float(row.get(f"l8_{idx}_cloud_cover")),
                scene_id=scene_id,
                row=row,
            )
        )
    return slots


def _save_patch(image_patch: np.ndarray, mask_patch: np.ndarray, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    img_path = dest_dir / "image.tif"
    mask_path = dest_dir / "plume.tif"
    tifffile.imwrite(img_path, np.ascontiguousarray(image_patch))
    tifffile.imwrite(mask_path, np.ascontiguousarray(mask_patch))
    return img_path


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _collect_row_value(row: Dict[str, object], key: str) -> Optional[float]:
    return _to_float(row.get(key))


def _process_slot(
    slot: ChipSlot,
    mask_path: Path,
    crop_size: int,
    pos_count: int,
    neg_count: int,
    dest_root: Path,
    records: List[Dict[str, object]],
    sensor_label: str,
) -> None:
    if not mask_path.exists():
        print(f"[WARN] {slot.plume_id}: missing plume mask {mask_path}")
        return
    try:
        chip = _read_chip(slot.image_path)
    except Exception as exc:
        print(f"[WARN] failed to read chip {slot.image_path}: {exc}")
        return
    try:
        mask = _load_mask_like_image(mask_path, chip.shape[1], chip.shape[2])
    except Exception as exc:
        print(f"[WARN] failed to load mask {mask_path}: {exc}")
        return
    plume_mask = mask > 0
    pos_windows = _center_biased_positive_windows(plume_mask, crop_size, pos_count)
    bg_windows = _random_background_windows(plume_mask, crop_size, neg_count)
    chip_dir = dest_root / slot.plume_id / slot.image_tag()
    lat = _collect_row_value(slot.row, "plume_latitude")
    lon = _collect_row_value(slot.row, "plume_longitude")
    emission = _collect_row_value(slot.row, "emission_auto")
    emission_uncertainty = _collect_row_value(slot.row, "emission_uncertainty_auto")
    wind = _collect_row_value(slot.row, "wind_speed_avg_auto")
    plume_dt = _parse_iso8601(slot.row.get("datetime"))
    plume_dt = plume_dt if plume_dt is not None else slot.event_dt
    image_height, image_width = chip.shape[1], chip.shape[2]
    plume_dt_str = _format_timestamp(plume_dt)
    for kind, windows in (("fg", pos_windows), ("bg", bg_windows)):
        label = 1 if kind == "fg" else 0
        for idx, (top, left) in enumerate(windows, start=1):
            img_patch = chip[:, top : top + crop_size, left : left + crop_size]
            mask_patch = plume_mask[top : top + crop_size, left : left + crop_size]
            target_dir = chip_dir / f"{kind}_{idx:02d}"
            img_path = _save_patch(img_patch, mask_patch.astype(np.uint8) * 255, target_dir)
            record_emission = emission if label == 1 else 0.0
            record_uncertainty = emission_uncertainty if label == 1 else 0.0
            record_wind = wind if label == 1 else 0.0
            records.append(
                {
                    "label": label,
                    "image_path": _relative_path(img_path),
                    "image_timestamp": _format_timestamp(slot.image_dt),
                    "plume_datetime": plume_dt_str,
                    "plume_id": slot.plume_id,
                    "sensor": sensor_label,
                    "scene_id": slot.scene_id if slot.scene_id else "",
                    "plume_latitude": lat,
                    "plume_longitude": lon,
                    "cloud_cover": slot.cloud_cover,
                    "time_offset_hours": slot.time_offset_hours(),
                    "emission_auto": record_emission,
                    "emission_uncertainty_auto": record_uncertainty,
                    "wind_speed_avg_auto": record_wind,
                    "crop_kind": kind,
                    "crop_index": idx,
                    "crop_row": top,
                    "crop_col": left,
                    "crop_size": crop_size,
                    "plume_pixels_in_crop": int(mask_patch.sum()),
                    "image_height": image_height,
                    "image_width": image_width,
                }
            )


def _load_plume_ids(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    resolved = Path(path)
    if not resolved.exists():
        print(f"[WARN] split file {resolved} does not exist; ignoring.")
        return set()
    df = pd.read_csv(resolved)
    if "plume_id" in df.columns:
        series = df["plume_id"]
    else:
        series = df.iloc[:, 0]
    ids = set()
    for value in series.dropna():
        text = str(value).strip()
        if text:
            ids.add(text)
    return ids


def _group_by_split(
    records: List[Dict[str, object]],
    train_ids: set[str],
    test_ids: set[str],
) -> Dict[str, List[Dict[str, object]]]:
    if not train_ids and not test_ids:
        return {"all": records}
    train_records: List[Dict[str, object]] = []
    test_records: List[Dict[str, object]] = []
    other_records: List[Dict[str, object]] = []
    for record in records:
        plume_id = record.get("plume_id")
        if plume_id in train_ids:
            train_records.append(record)
        elif plume_id in test_ids:
            test_records.append(record)
        else:
            other_records.append(record)
    return {"train": train_records, "test": test_records, "unassigned": other_records}


def _write_records_with_splits(
    records: List[Dict[str, object]],
    base_name: str,
    output_dir: Path,
    train_ids: set[str],
    test_ids: set[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        print(f"[INFO] no records to write for {base_name}.")
        return
    grouped = _group_by_split(records, train_ids, test_ids)
    if "all" in grouped:
        out_path = output_dir / f"{base_name}.csv"
        pd.DataFrame(grouped["all"]).to_csv(out_path, index=False)
        print(f"[INFO] wrote {len(grouped['all']):5d} rows -> {out_path}")
        return
    if grouped["train"]:
        out_path = output_dir / f"{base_name}_train.csv"
        pd.DataFrame(grouped["train"]).to_csv(out_path, index=False)
        print(f"[INFO] wrote {len(grouped['train']):5d} train rows -> {out_path}")
    else:
        print(f"[WARN] no train records matched provided split for {base_name}.")
    if grouped["test"]:
        out_path = output_dir / f"{base_name}_test.csv"
        pd.DataFrame(grouped["test"]).to_csv(out_path, index=False)
        print(f"[INFO] wrote  {len(grouped['test']):5d} test rows -> {out_path}")
    else:
        print(f"[WARN] no test records matched provided split for {base_name}.")
    if grouped["unassigned"]:
        out_path = output_dir / f"{base_name}_unassigned.csv"
        pd.DataFrame(grouped["unassigned"]).to_csv(out_path, index=False)
        print(
            f"[WARN] {len(grouped['unassigned'])} rows did not match the provided split; "
            f"they were saved to {out_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop Sentinel-2 and Landsat-8/9 chips around Carbon Mapper plumes."
    )
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH, help="Input merged CSV path.")
    parser.add_argument("--hours-window", type=float, default=24.0, help="Time window in hours.")
    parser.add_argument("--crop-size", type=int, default=96, help="Crop size in pixels.")
    parser.add_argument(
        "--pos-count", type=int, default=8, help="Number of positive (foreground) crops per plume."
    )
    parser.add_argument(
        "--neg-count", type=int, default=16, help="Number of negative (background) crops per plume."
    )
    parser.add_argument(
        "--max-s2-slots",
        type=int,
        default=1,
        help="Maximum number of Sentinel-2 acquisitions to sample per plume.",
    )
    parser.add_argument(
        "--output-s2-dir",
        type=Path,
        default=OUTPUT_S2_ROOT,
        help="Destination for S2 crops.",
    )
    parser.add_argument(
        "--output-l89-dir",
        type=Path,
        default=OUTPUT_L89_ROOT,
        help="Destination for L8/9 crops.",
    )
    parser.add_argument(
        "--csv-output-dir",
        type=Path,
        default=CSV_OUTPUT_DIR,
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of plume rows to process.",
    )
    parser.add_argument(
        "--train-plume-ids",
        type=Path,
        default=None,
        help="CSV listing plume IDs assigned to the Carbon Mapper training split.",
    )
    parser.add_argument(
        "--test-plume-ids",
        type=Path,
        default=None,
        help="CSV listing plume IDs assigned to the Carbon Mapper test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    train_ids = _load_plume_ids(args.train_plume_ids)
    test_ids = _load_plume_ids(args.test_plume_ids)
    rows = df.to_dict("records")
    s2_records: List[Dict[str, object]] = []
    l89_records: List[Dict[str, object]] = []
    processed = 0
    for idx, row in enumerate(rows, start=1):
        if args.limit is not None and processed >= args.limit:
            break
        plume_id = row.get("plume_id")
        event_dt = _parse_iso8601(row.get("datetime"))
        if event_dt is None:
            continue
        mask_path = MASK_ROOT / str(plume_id) / "plume.tif"
        s2_slots = _gather_s2_slots(row, event_dt, args.hours_window, args.max_s2_slots)
        l89_slots = _gather_l89_slots(row, event_dt, args.hours_window)
        if not s2_slots and not l89_slots:
            continue
        for slot in s2_slots:
            _process_slot(
                slot,
                mask_path,
                args.crop_size,
                args.pos_count,
                args.neg_count,
                args.output_s2_dir,
                s2_records,
                "s2",
            )
        for slot in l89_slots:
            _process_slot(
                slot,
                mask_path,
                args.crop_size,
                args.pos_count,
                args.neg_count,
                args.output_l89_dir,
                l89_records,
                "l89",
            )
        processed += 1
        if processed % 100 == 0:
            print(f"[INFO] processed {processed} plume rows")
    args.csv_output_dir.mkdir(parents=True, exist_ok=True)
    s2_csv_path = args.csv_output_dir / "CM_s2_l8_s2.csv"
    l89_csv_path = args.csv_output_dir / "CM_s2_l8_l89.csv"
    if s2_records:
        _write_records_with_splits(s2_records, s2_csv_path.stem, args.csv_output_dir, train_ids, test_ids)
    else:
        print("[INFO] no S2 crops were generated.")
    if l89_records:
        _write_records_with_splits(l89_records, l89_csv_path.stem, args.csv_output_dir, train_ids, test_ids)
    else:
        print("[INFO] no L8/9 crops were generated.")


if __name__ == "__main__":
    main()
