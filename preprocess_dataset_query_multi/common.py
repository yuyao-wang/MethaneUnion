import csv
import math
import os
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio


SENSORS = ("s2", "l89", "emit")

SENSOR_SPECS = {
    "s2": {
        "image_cols": ["s2_0_std_512", "s2_-90_std_512", "s2_-360_std_512"],
        "out_image_cols": ["s2_0_512_path", "s2_-90_512_path", "s2_-360_512_path"],
        "source_mask_col": "s2_plume_mask_512_path",
        "out_mask_col": "s2_mask_512_path",
        "out_mask_name": "s2_mask_512.tif",
    },
    "l89": {
        "image_cols": ["l89_0_std_512", "l89_-90_std_512", "l89_-360_std_512"],
        "out_image_cols": ["l89_0_512_path", "l89_-90_512_path", "l89_-360_512_path"],
        "source_mask_col": "l89_mask_512_path",
        "out_mask_col": "l89_mask_512_path",
        "out_mask_name": "l89_mask_512.tif",
    },
    "emit": {
        "image_cols": ["emit_0_simulated_512_path", "emit_-90_simulated_512_path", "emit_-180_simulated_512_path"],
        "out_image_cols": ["emit_0_512_path", "emit_-90_512_path", "emit_-180_512_path"],
        "source_mask_col": "emit_mask_512_path",
        "out_mask_col": "emit_mask_512_path",
        "out_mask_name": "emit_mask_512.tif",
    },
}


RAW_MASK_ROOT = Path("/data2/yuyao/methane_emission/carbon_mapper_data_masks")
EMIT_MASK_ROOT = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_masks_wv3_512")
EMIT_MASK_NAME = "mask_60m_512.tif"


def has_value(value: object) -> bool:
    s = str(value or "").strip()
    return s != "" and s.lower() != "nan" and s.lower() != "none"


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"No CSV header found: {path}")
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def first_existing_path(row: Dict[str, str], cols: Iterable[str]) -> Optional[Path]:
    for col in cols:
        value = row.get(col, "")
        if has_value(value):
            path = Path(str(value))
            if path.exists():
                return path
    return None


def raw_plume_path(plume_id: str, raw_mask_root: Path = RAW_MASK_ROOT) -> Path:
    return raw_mask_root / plume_id / "plume.tif"


def emit_mask_path(plume_id: str, emit_mask_root: Path = EMIT_MASK_ROOT) -> Path:
    return emit_mask_root / plume_id / EMIT_MASK_NAME


def dataset_shape(path: Path) -> Optional[Tuple[int, int, int]]:
    try:
        with rasterio.open(path) as ds:
            return ds.count, ds.height, ds.width
    except Exception:
        return None


def read_first_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def binary_array(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    return ((arr > 0) & finite).astype(np.uint8)


def write_binary_tif(path: Path, mask: np.ndarray, profile_ref: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if profile_ref is None:
        profile = {
            "driver": "GTiff",
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "count": 1,
            "dtype": "uint8",
            "compress": "deflate",
            "tiled": True,
            "blockxsize": min(256, int(mask.shape[1])),
            "blockysize": min(256, int(mask.shape[0])),
            "BIGTIFF": "IF_SAFER",
        }
    else:
        profile = dict(profile_ref)
        profile.update(
            driver="GTiff",
            height=int(mask.shape[0]),
            width=int(mask.shape[1]),
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            tiled=True,
            blockxsize=min(256, int(mask.shape[1])),
            blockysize=min(256, int(mask.shape[0])),
            BIGTIFF="IF_SAFER",
        )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def mask_stats(path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {
        "mask_exists": False,
        "mask_height": "",
        "mask_width": "",
        "mask_dtype": "",
        "mask_is_binary": False,
        "mask_positive_pixels": "",
        "mask_positive_fraction": "",
    }
    if not path.exists():
        return out
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        out["mask_exists"] = True
        out["mask_height"] = ds.height
        out["mask_width"] = ds.width
        out["mask_dtype"] = str(arr.dtype)
    vals = np.unique(arr[np.isfinite(arr)])
    out["mask_is_binary"] = bool(np.all(np.isin(vals, [0, 1])))
    pos = int(((arr > 0) & np.isfinite(arr)).sum())
    total = int(arr.size)
    out["mask_positive_pixels"] = pos
    out["mask_positive_fraction"] = pos / total if total else 0.0
    return out


def normalize_rgb(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 2:
        rgb = np.stack([data, data, data], axis=-1)
    elif data.ndim == 3:
        if data.shape[0] <= 32 and data.shape[0] < data.shape[-1]:
            data = np.transpose(data, (1, 2, 0))
        if data.shape[-1] >= 3:
            rgb = data[..., :3]
        else:
            rgb = np.repeat(data[..., :1], 3, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape for RGB: {data.shape}")

    out = np.zeros_like(rgb[..., :3], dtype=np.float32)
    for c in range(3):
        band = rgb[..., c]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        lo, hi = np.nanpercentile(band[finite], [2, 98])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            hi = float(np.nanmax(band[finite]))
            lo = float(np.nanmin(band[finite]))
        if hi <= lo:
            continue
        out[..., c] = np.clip((band - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    ys = np.clip(np.round(np.linspace(0, h - 1, out_h)).astype(np.int64), 0, h - 1)
    xs = np.clip(np.round(np.linspace(0, w - 1, out_w)).astype(np.int64), 0, w - 1)
    if img.ndim == 2:
        return img[ys[:, None], xs[None, :]]
    return img[ys[:, None], xs[None, :], :]


def overlay_mask(rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    out = rgb.copy()
    if mask is None:
        return out
    m = resize_nearest((mask > 0).astype(np.uint8), out.shape[0], out.shape[1]) > 0
    out[m, 0] = 255
    out[m, 1] = (out[m, 1] * 0.25).astype(np.uint8)
    out[m, 2] = (out[m, 2] * 0.25).astype(np.uint8)
    return out


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an RGB uint8 PNG without Pillow/matplotlib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"PNG writer expects HxWx3 uint8, got {arr.shape}")
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
