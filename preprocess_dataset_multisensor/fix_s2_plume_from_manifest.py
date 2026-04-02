import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


MASK_ROOT = Path("/data2/yuyao/methane_emission/carbon_mapper_data_masks")
IMAGE_SIZE_512 = 512
CENTER = IMAGE_SIZE_512 // 2
TARGET_SIZE = 224

# Match preprocess_dataset_multisensor/crop.py
GSD = {
    "s2": 10.0,
    "l89": 30.0,
    "emit": 60.0,
    "s5p": 3500.0,
}

PATCH_SIZE = {
    "s2": 32,
    "l89": 16,
    "emit": 16,
    "s5p": 3,
}


def resize_hw(mask: np.ndarray, out_size: int = TARGET_SIZE) -> np.ndarray:
    h, w = mask.shape
    ys = np.linspace(0, h - 1, out_size).round().astype(np.int32)
    xs = np.linspace(0, w - 1, out_size).round().astype(np.int32)
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    return mask[ys[:, None], xs[None, :]].astype(np.float32)


def compute_s2_top_left(anchor_sensor: str, dx_anchor_px: int, dy_anchor_px: int) -> tuple[int, int]:
    ps = PATCH_SIZE["s2"]
    half = ps // 2
    dx_m = dx_anchor_px * GSD[anchor_sensor]
    dy_m = dy_anchor_px * GSD[anchor_sensor]
    dx = int(round(dx_m / GSD["s2"]))
    dy = int(round(dy_m / GSD["s2"]))
    cx = CENTER + dx
    cy = CENTER + dy
    return cx - half, cy - half


def crop_hw(arr: np.ndarray, x: int, y: int, size: int) -> np.ndarray | None:
    h, w = arr.shape
    if x < 0 or y < 0 or x + size > w or y + size > h:
        return None
    return arr[y : y + size, x : x + size]


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    return arr


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(0, 0, 1, 1),
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": min(256, mask.shape[1]),
        "blockysize": min(256, mask.shape[0]),
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.float32), 1)


def fix_manifest(manifest_csv: Path, mask_root: Path, dry_run: bool = False, limit: int = 0) -> None:
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        required_cols = ["plume_id", "anchor_sensor", "dx_anchor_px", "dy_anchor_px", "s2_plume_path"]
        missing_cols = [c for c in required_cols if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise RuntimeError(f"Missing required columns in {manifest_csv}: {missing_cols}")

        stats: dict[str, int] = {
            "rows_seen": 0,
            "rows_fixed": 0,
            "rows_skipped_no_path": 0,
            "rows_skipped_no_plume_id": 0,
            "rows_skipped_no_source_mask": 0,
            "rows_skipped_bad_anchor": 0,
            "rows_skipped_bad_offsets": 0,
            "rows_skipped_out_of_bounds": 0,
        }

        for row in reader:
            if limit and stats["rows_seen"] >= limit:
                break
            stats["rows_seen"] += 1

            plume_id = str(row.get("plume_id", "")).strip()
            out_path_str = str(row.get("s2_plume_path", "")).strip()
            anchor_sensor = str(row.get("anchor_sensor", "")).strip().lower()

            if not out_path_str or out_path_str.lower() == "nan":
                stats["rows_skipped_no_path"] += 1
                continue
            if not plume_id or plume_id.lower() == "nan":
                stats["rows_skipped_no_plume_id"] += 1
                continue
            if anchor_sensor not in GSD:
                stats["rows_skipped_bad_anchor"] += 1
                continue

            try:
                dx_anchor_px = int(float(row["dx_anchor_px"]))
                dy_anchor_px = int(float(row["dy_anchor_px"]))
            except Exception:
                stats["rows_skipped_bad_offsets"] += 1
                continue

            src_mask_path = mask_root / plume_id / "resized_512x512.tif"
            if not src_mask_path.exists():
                stats["rows_skipped_no_source_mask"] += 1
                continue

            x, y = compute_s2_top_left(anchor_sensor, dx_anchor_px, dy_anchor_px)
            cropped = crop_hw(read_mask(src_mask_path), x, y, PATCH_SIZE["s2"])
            if cropped is None:
                stats["rows_skipped_out_of_bounds"] += 1
                continue

            resized = resize_hw(cropped, TARGET_SIZE)
            if not dry_run:
                write_mask(Path(out_path_str), resized)
            stats["rows_fixed"] += 1

    print(f"manifest: {manifest_csv}")
    for k, v in stats.items():
        print(f"{k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild s2_plume.tif files from resized_512x512.tif using an existing multisensor manifest."
    )
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, default=MASK_ROOT)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows for smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fix_manifest(
        manifest_csv=args.manifest_csv,
        mask_root=args.mask_root,
        dry_run=args.dry_run,
        limit=max(0, int(args.limit)),
    )
