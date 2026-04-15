import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio

from common import (
    SENSORS,
    has_value,
    normalize_rgb,
    overlay_mask,
    read_csv_rows,
    read_first_band,
    resize_nearest,
    write_png,
)


IMAGE_COLS = {
    "s2": "s2_0_512_path",
    "l89": "l89_0_512_path",
    "emit": "emit_0_512_path",
}
MASK_COLS = {
    "s2": "s2_mask_512_path",
    "l89": "l89_mask_512_path",
    "emit": "emit_mask_512_path",
}


def path_ok(row: Dict[str, str], col: str) -> bool:
    return has_value(row.get(col, "")) and Path(str(row[col])).exists()


def complete_four_sensor(row: Dict[str, str]) -> bool:
    if not all(path_ok(row, IMAGE_COLS[s]) and path_ok(row, MASK_COLS[s]) for s in SENSORS):
        return False
    return path_ok(row, "S5p_path") and path_ok(row, "s5p_minus90_path") and path_ok(row, "s5p_minus360_path")


def read_image_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        count = min(ds.count, 3)
        arr = ds.read(list(range(1, count + 1)))
    return normalize_rgb(arr)


def read_mask(path: Path) -> np.ndarray:
    return (read_first_band(path) > 0).astype(np.uint8)


def s5p_panel(row: Dict[str, str], size: int = 512) -> np.ndarray:
    # S5P quicklook is intentionally simple: show a black panel with a green
    # center marker because reading NetCDF requires netCDF4, which is not part
    # of the current lightweight environment. The path is still listed in the
    # sidecar CSV for checking.
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    cy = size // 2
    cx = size // 2
    panel[cy - 12 : cy + 13, cx - 2 : cx + 3, 1] = 255
    panel[cy - 2 : cy + 3, cx - 12 : cx + 13, 1] = 255
    return panel


def make_canvas(row: Dict[str, str], tile_size: int = 256) -> np.ndarray:
    top_tiles = []
    bottom_tiles = []
    for sensor in SENSORS:
        rgb = read_image_rgb(Path(str(row[IMAGE_COLS[sensor]])))
        mask = read_mask(Path(str(row[MASK_COLS[sensor]])))
        mask_rgb = np.zeros_like(rgb)
        m = resize_nearest(mask, rgb.shape[0], rgb.shape[1]) > 0
        mask_rgb[m] = np.array([255, 255, 255], dtype=np.uint8)
        top_tiles.append(resize_nearest(rgb, tile_size, tile_size))
        bottom_tiles.append(resize_nearest(overlay_mask(rgb, mask), tile_size, tile_size))

    s5p = s5p_panel(row)
    top_tiles.append(resize_nearest(s5p, tile_size, tile_size))
    bottom_tiles.append(resize_nearest(s5p, tile_size, tile_size))

    gap = 8
    h = tile_size * 2 + gap
    w = tile_size * 4 + gap * 3
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x = 0
    for tile in top_tiles:
        canvas[0:tile_size, x : x + tile_size] = tile
        x += tile_size + gap
    x = 0
    y = tile_size + gap
    for tile in bottom_tiles:
        canvas[y : y + tile_size, x : x + tile_size] = tile
        x += tile_size + gap
    return canvas


def write_sidecar(path: Path, rows: list[Dict[str, str]]) -> None:
    fields = [
        "plume_id",
        "latitude",
        "longitude",
        "datetime",
        "s2_0_512_path",
        "s2_mask_512_path",
        "l89_0_512_path",
        "l89_mask_512_path",
        "emit_0_512_path",
        "emit_mask_512_path",
        "S5p_path",
    ]
    from common import write_csv_rows

    write_csv_rows(path, fields, rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize random complete four-sensor raw512 samples.")
    p.add_argument("--manifest_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile_size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    _, rows = read_csv_rows(args.manifest_csv)
    candidates = [r for r in rows if complete_four_sensor(r)]
    print(f"complete_four_sensor_candidates: {len(candidates)}")
    random.seed(args.seed)
    random.shuffle(candidates)
    selected = candidates[: max(0, args.num_samples)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i, row in enumerate(selected, start=1):
        pid = str(row.get("plume_id", f"sample_{i}")).replace("/", "_")
        try:
            canvas = make_canvas(row, tile_size=args.tile_size)
            out = args.out_dir / f"{i:02d}_{pid}.png"
            write_png(out, canvas)
            print(f"wrote {out}")
            written.append(row)
        except Exception as e:
            print(f"[warn] failed plume_id={pid}: {e}")

    write_sidecar(args.out_dir / "selected_samples.csv", written)
    print(f"selected_written: {len(written)}")


if __name__ == "__main__":
    args = parse_args()
    main()
