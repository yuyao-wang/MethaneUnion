import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio

from common import has_value, normalize_rgb, overlay_mask, read_csv_rows, resize_nearest, write_csv_rows, write_png


SENSORS = ("s2", "l89", "emit", "s5p")


def path_exists(row: Dict[str, str], col: str) -> bool:
    return has_value(row.get(col, "")) and Path(str(row[col])).exists()


def read_chw(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read()


def read_hw(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def sensor_panel(row: Dict[str, str], sensor: str, tile_size: int) -> np.ndarray:
    image_col = f"{sensor}_0_path"
    if not path_exists(row, image_col):
        return np.full((tile_size, tile_size, 3), 230, dtype=np.uint8)
    rgb = normalize_rgb(read_chw(Path(str(row[image_col]))))
    if sensor != "s5p" and path_exists(row, f"{sensor}_mask_path"):
        mask = read_hw(Path(str(row[f"{sensor}_mask_path"])))
        rgb = overlay_mask(rgb, mask)
    return resize_nearest(rgb, tile_size, tile_size)


def make_canvas(row: Dict[str, str], tile_size: int) -> np.ndarray:
    gap = 8
    h = tile_size
    w = tile_size * len(SENSORS) + gap * (len(SENSORS) - 1)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x = 0
    for sensor in SENSORS:
        panel = sensor_panel(row, sensor, tile_size)
        canvas[:, x : x + tile_size] = panel
        x += tile_size + gap
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize query crop samples with mask overlays.")
    p.add_argument("--manifest_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile_size", type=int, default=224)
    return p.parse_args()


def main() -> None:
    _, rows = read_csv_rows(args.manifest_csv)
    random.seed(args.seed)
    positives = [r for r in rows if str(r.get("label", "")).strip() == "1"]
    negatives = [r for r in rows if str(r.get("label", "")).strip() == "0"]
    random.shuffle(positives)
    random.shuffle(negatives)
    half = max(1, args.num_samples // 2)
    selected = positives[:half] + negatives[: max(0, args.num_samples - half)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, str]] = []
    for row in selected:
        sid = str(row.get("id", "unknown"))
        label = str(row.get("label", "x"))
        pid = str(row.get("plume_id", "sample")).replace("/", "_")
        out = args.out_dir / f"id{sid}_label{label}_{pid}.png"
        write_png(out, make_canvas(row, args.tile_size))
        written.append(row)
        print(f"wrote {out}")
    write_csv_rows(args.out_dir / "selected_query_samples.csv", list(rows[0].keys()) if rows else [], written)
    print(f"selected_written: {len(written)}")


if __name__ == "__main__":
    args = parse_args()
    main()
