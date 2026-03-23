import argparse
import math
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

try:
    from netCDF4 import Dataset
except Exception:
    Dataset = None

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


# =========================
# Config
# =========================
MASTER_CSV = Path("/home/yuyao/methane_train/preprocess_dataset_multisensor/master_multisensor_outer_join.csv")
OUT_ROOT = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset")
OUT_CSV = OUT_ROOT / "manifest_multisensor_crop.csv"

EMIT_MASK_ROOT = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_masks_wv3_512")
EMIT_MASK_NAME = "mask_60m_512.tif"

# physical GSD (meter / pixel)
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

TARGET_SIZE = 224
IMAGE_SIZE_512 = 512
CENTER = IMAGE_SIZE_512 // 2
N_POS = 16
N_NEG = 16

# quality thresholds (copied from legacy scripts)
S2_BAND_INDEX = 11
S2_ZERO_RATIO_THRESH = 0.20
L89_MISSING_THRESH = 0.25
EMIT_MISSING_THRESH = 0.25
S5P_MISSING_THRESH = 0.50

# S5P variables
CH4_CANDIDATES = [
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
]


# =========================
# Utilities
# =========================

def is_valid_path(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"


def normalize_datetime_iso(v) -> Optional[str]:
    """Normalize mixed datetime formats to UTC ISO string: YYYY-MM-DDTHH:MM:SSZ."""
    if pd.isna(v):
        return None
    try:
        ts = pd.to_datetime(v, utc=True, errors="coerce", format="mixed")
    except TypeError:
        # For older pandas versions without format='mixed'
        ts = pd.to_datetime(v, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_chw(data: np.ndarray) -> Optional[np.ndarray]:
    if data.ndim == 2:
        return data[None, :, :]
    if data.ndim != 3:
        return None
    if data.shape[0] <= 32:
        return data
    return data.transpose(2, 0, 1)


def resize_chw(img: np.ndarray, out_size: int = TARGET_SIZE) -> np.ndarray:
    if torch is not None and F is not None:
        t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)  # 1,C,H,W
        out = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    # fallback: simple per-channel linear interpolation using numpy indexing
    c, h, w = img.shape
    ys = np.linspace(0, h - 1, out_size)
    xs = np.linspace(0, w - 1, out_size)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    out = np.empty((c, out_size, out_size), dtype=np.float32)
    for i in range(c):
        a = img[i][y0[:, None], x0[None, :]]
        b = img[i][y0[:, None], x1[None, :]]
        c1 = img[i][y1[:, None], x0[None, :]]
        d = img[i][y1[:, None], x1[None, :]]
        out[i] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c1 * (1 - wx) * wy + d * wx * wy
    return out


def resize_hw(mask: np.ndarray, out_size: int = TARGET_SIZE) -> np.ndarray:
    if torch is not None and F is not None:
        t = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]  # 1,1,H,W
        out = F.interpolate(t, size=(out_size, out_size), mode="nearest")
        return out.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    h, w = mask.shape
    ys = np.linspace(0, h - 1, out_size).round().astype(np.int32)
    xs = np.linspace(0, w - 1, out_size).round().astype(np.int32)
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    return mask[ys[:, None], xs[None, :]].astype(np.float32)


def crop_chw(arr: np.ndarray, x: int, y: int, size: int) -> Optional[np.ndarray]:
    c, h, w = arr.shape
    if x < 0 or y < 0 or x + size > w or y + size > h:
        return None
    return arr[:, y : y + size, x : x + size]


def crop_hw(arr: np.ndarray, x: int, y: int, size: int) -> Optional[np.ndarray]:
    h, w = arr.shape
    if x < 0 or y < 0 or x + size > w or y + size > h:
        return None
    return arr[y : y + size, x : x + size]


def center_contained(x: int, y: int, size: int, center_box: int = 10) -> bool:
    cx1 = CENTER - center_box // 2
    cy1 = CENTER - center_box // 2
    cx2 = CENTER + center_box // 2
    cy2 = CENTER + center_box // 2
    x1, y1 = x, y
    x2, y2 = x + size, y + size
    return x1 <= cx1 and y1 <= cy1 and x2 >= cx2 and y2 >= cy2


def s2_valid_crop(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> bool:
    def zero_ratio_ok(a: np.ndarray) -> bool:
        if a.shape[0] <= S2_BAND_INDEX:
            return False
        band = a[S2_BAND_INDEX]
        return float((band == 0).mean()) < S2_ZERO_RATIO_THRESH

    return zero_ratio_ok(c0) and zero_ratio_ok(c1) and zero_ratio_ok(c2)


def missing_ratio_valid(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, thresh: float) -> bool:
    def one(a: np.ndarray) -> bool:
        if a.size == 0:
            return False
        first = a[0]
        miss = np.isnan(first) | (first == 0)
        return float(miss.mean()) <= thresh

    return one(c0) and one(c1) and one(c2)


def pick_ch4_var(prod) -> Optional[str]:
    for k in CH4_CANDIDATES:
        if k in prod.variables:
            return k
    return None


def to_nan_invalid(arr, attrs=None):
    a = np.array(arr, dtype=np.float32, copy=False)
    attrs = attrs or {}
    fv = attrs.get("_FillValue", None)
    mv = attrs.get("missing_value", None)
    if fv is not None:
        a = np.where(a == np.float32(fv), np.nan, a)
    if mv is not None:
        a = np.where(a == np.float32(mv), np.nan, a)
    a = np.where(np.abs(a) > 1e20, np.nan, a)
    if a.ndim == 3:
        a = a[0]
    return a


def read_s5p(path_nc: str, ch4_hint: Optional[str] = None) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
    if Dataset is None:
        return None, False, ch4_hint
    if (not path_nc) or (str(path_nc).lower() == "nan"):
        return None, False, ch4_hint
    if not Path(path_nc).exists():
        return None, False, ch4_hint

    try:
        ds = Dataset(str(path_nc), "r")
        prod = ds.groups["PRODUCT"]
        ch4 = ch4_hint if (ch4_hint and ch4_hint in prod.variables) else pick_ch4_var(prod)
        if ch4 is None:
            ds.close()
            return None, False, ch4_hint
        v = prod.variables[ch4]
        arr = to_nan_invalid(v[:], getattr(v, "__dict__", {}))
        ds.close()
        return arr, True, ch4
    except Exception:
        return None, False, ch4_hint


def s5p_crop_and_resize(arr2d: np.ndarray, cx: int, cy: int, size: int = 3) -> Optional[np.ndarray]:
    half = size // 2
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    if y0 < 0 or x0 < 0 or y1 > arr2d.shape[0] or x1 > arr2d.shape[1]:
        return None
    small = arr2d[y0:y1, x0:x1]
    if (1.0 - np.isfinite(small).sum() / small.size) > S5P_MISSING_THRESH:
        return None
    # nan-aware resize
    fin = np.isfinite(small)
    v = np.where(fin, small, 0.0).astype(np.float32)
    w = fin.astype(np.float32)
    if torch is not None and F is not None:
        vt = torch.from_numpy(v)[None, None, :, :].float()
        wt = torch.from_numpy(w)[None, None, :, :].float()
        vr = F.interpolate(vt, size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        wr = F.interpolate(wt, size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    else:
        ys = np.linspace(0, v.shape[0] - 1, TARGET_SIZE)
        xs = np.linspace(0, v.shape[1] - 1, TARGET_SIZE)
        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, v.shape[0] - 1)
        x1 = np.clip(x0 + 1, 0, v.shape[1] - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]
        va = v[y0[:, None], x0[None, :]]
        vb = v[y0[:, None], x1[None, :]]
        vc = v[y1[:, None], x0[None, :]]
        vd = v[y1[:, None], x1[None, :]]
        wa = w[y0[:, None], x0[None, :]]
        wb = w[y0[:, None], x1[None, :]]
        wc = w[y1[:, None], x0[None, :]]
        wd = w[y1[:, None], x1[None, :]]
        vr = va * (1 - wx) * (1 - wy) + vb * wx * (1 - wy) + vc * (1 - wx) * wy + vd * wx * wy
        wr = wa * (1 - wx) * (1 - wy) + wb * wx * (1 - wy) + wc * (1 - wx) * wy + wd * wx * wy
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(wr > 1e-6, vr / wr, np.nan)
    return out.astype(np.float32)


@dataclass
class SensorBundle:
    name: str
    t0: Optional[np.ndarray]
    t90: Optional[np.ndarray]
    t360: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    nearest_ix: Optional[int] = None
    nearest_iy: Optional[int] = None


class Counter:
    def __init__(self, start: int = 0):
        self.lock = threading.Lock()
        self.n = start

    def next(self) -> int:
        with self.lock:
            self.n += 1
            return self.n


def load_row_sensors(row: pd.Series) -> Dict[str, SensorBundle]:
    out: Dict[str, SensorBundle] = {}

    # S2
    if bool(row.get("has_s2", False)) and all(is_valid_path(row.get(c)) for c in ["s2_0_std_512", "s2_-90_std_512", "s2_-360_std_512"]):
        try:
            t0 = to_chw(tifffile.imread(str(row["s2_0_std_512"])))
            t90 = to_chw(tifffile.imread(str(row["s2_-90_std_512"])))
            t360 = to_chw(tifffile.imread(str(row["s2_-360_std_512"])))
            if is_valid_path(row.get("s2_plume_mask_512_path")) and Path(str(row["s2_plume_mask_512_path"])).exists():
                m = tifffile.imread(str(row["s2_plume_mask_512_path"]))
            else:
                m = np.zeros((IMAGE_SIZE_512, IMAGE_SIZE_512), dtype=np.uint8)
            if t0 is not None and t90 is not None and t360 is not None and m is not None:
                out["s2"] = SensorBundle("s2", t0, t90, t360, m)
        except Exception:
            pass

    # L89
    if bool(row.get("has_l89", False)) and all(is_valid_path(row.get(c)) for c in ["l89_0_std_512", "l89_-90_std_512", "l89_-360_std_512"]):
        try:
            t0 = to_chw(tifffile.imread(str(row["l89_0_std_512"])))
            t90 = to_chw(tifffile.imread(str(row["l89_-90_std_512"])))
            t360 = to_chw(tifffile.imread(str(row["l89_-360_std_512"])))
            if is_valid_path(row.get("l89_mask_512_path")) and Path(str(row["l89_mask_512_path"])).exists():
                m = tifffile.imread(str(row["l89_mask_512_path"]))
            else:
                m = np.zeros((IMAGE_SIZE_512, IMAGE_SIZE_512), dtype=np.uint8)
            if t0 is not None and t90 is not None and t360 is not None and m is not None:
                out["l89"] = SensorBundle("l89", t0, t90, t360, m)
        except Exception:
            pass

    # EMIT
    if bool(row.get("has_emit", False)) and all(is_valid_path(row.get(c)) for c in ["emit_0_simulated_512_path", "emit_-90_simulated_512_path", "emit_-180_simulated_512_path"]):
        try:
            t0 = to_chw(tifffile.imread(str(row["emit_0_simulated_512_path"])))
            t90 = to_chw(tifffile.imread(str(row["emit_-90_simulated_512_path"])))
            t360 = to_chw(tifffile.imread(str(row["emit_-180_simulated_512_path"])))
            mpath = EMIT_MASK_ROOT / str(row["plume_id"]) / EMIT_MASK_NAME
            m = tifffile.imread(str(mpath)) if mpath.exists() else np.zeros((IMAGE_SIZE_512, IMAGE_SIZE_512), dtype=np.uint8)
            if t0 is not None and t90 is not None and t360 is not None:
                out["emit"] = SensorBundle("emit", t0, t90, t360, m)
        except Exception:
            pass

    # S5P
    if bool(row.get("has_s5p", False)) and all(is_valid_path(row.get(c)) for c in ["S5p_path", "s5p_minus90_path", "s5p_minus360_path"]):
        try:
            ch4_hint = None
            t0, ok0, ch4_hint = read_s5p(str(row["S5p_path"]), ch4_hint)
            t90, ok90, ch4_hint = read_s5p(str(row["s5p_minus90_path"]), ch4_hint)
            t360, ok360, ch4_hint = read_s5p(str(row["s5p_minus360_path"]), ch4_hint)
            if ok0 and ok90 and ok360 and t0 is not None and t90 is not None and t360 is not None:
                ix = int(float(row["nearest_ix"])) if is_valid_path(row.get("nearest_ix")) else t0.shape[1] // 2
                iy = int(float(row["nearest_iy"])) if is_valid_path(row.get("nearest_iy")) else t0.shape[0] // 2
                out["s5p"] = SensorBundle("s5p", t0, t90, t360, None, nearest_ix=ix, nearest_iy=iy)
        except Exception:
            pass

    return out


def anchor_sensor(loaded: Dict[str, SensorBundle]) -> str:
    if "s2" in loaded:
        return "s2"
    return sorted(loaded.keys(), key=lambda s: GSD[s])[0]


def sample_offsets_overlap(anchor: str, n_pos: int = N_POS, n_neg: int = N_NEG) -> List[Tuple[int, int, int]]:
    # returns list[(label, dx_anchor_px, dy_anchor_px)]
    out: List[Tuple[int, int, int]] = []
    ps = PATCH_SIZE[anchor]
    half = ps // 2

    # positive: keep center in crop
    pos_min = -(half - 1)
    pos_max = half - 1
    for _ in range(n_pos):
        dx = random.randint(pos_min, pos_max)
        dy = random.randint(pos_min, pos_max)
        out.append((1, dx, dy))

    # negative: anywhere but center not contained
    while len(out) < n_pos + n_neg:
        dx = random.randint(-(CENTER - half), (IMAGE_SIZE_512 - 1 - half) - CENTER)
        dy = random.randint(-(CENTER - half), (IMAGE_SIZE_512 - 1 - half) - CENTER)
        x = CENTER + dx - half
        y = CENTER + dy - half
        if not center_contained(x, y, ps):
            out.append((0, dx, dy))
    return out


def sample_offsets_single(sensor: str, n_pos: int = N_POS, n_neg: int = N_NEG) -> List[Tuple[int, int, int]]:
    if sensor == "s5p":
        return [(1, 0, 0), (0, 0, 0)]

    out: List[Tuple[int, int, int]] = []
    ps = PATCH_SIZE[sensor]
    half = ps // 2

    if sensor == "s2":
        pos_min = -(half - 1)
        pos_max = half - 1
    else:
        # l89/emit legacy center jitter ~CENTER_BOX=6
        pos_min = -3
        pos_max = 3

    for _ in range(n_pos):
        out.append((1, random.randint(pos_min, pos_max), random.randint(pos_min, pos_max)))

    while len(out) < n_pos + n_neg:
        dx = random.randint(-(CENTER - half), (IMAGE_SIZE_512 - 1 - half) - CENTER)
        dy = random.randint(-(CENTER - half), (IMAGE_SIZE_512 - 1 - half) - CENTER)
        x = CENTER + dx - half
        y = CENTER + dy - half
        if not center_contained(x, y, ps):
            out.append((0, dx, dy))

    return out


def compute_top_left(sensor: str, anchor: str, dx_anchor: int, dy_anchor: int, overlap_mode: bool, s5p_bundle: Optional[SensorBundle] = None) -> Tuple[int, int]:
    ps = PATCH_SIZE[sensor]
    half = ps // 2

    if sensor == "s5p" and overlap_mode:
        # Overlap mode: S5P fixed center crop as requested
        cx = s5p_bundle.nearest_ix if s5p_bundle and s5p_bundle.nearest_ix is not None else CENTER
        cy = s5p_bundle.nearest_iy if s5p_bundle and s5p_bundle.nearest_iy is not None else CENTER
        return cx - half, cy - half

    dx_m = dx_anchor * GSD[anchor]
    dy_m = dy_anchor * GSD[anchor]
    dx = int(round(dx_m / GSD[sensor]))
    dy = int(round(dy_m / GSD[sensor]))

    cx = CENTER + dx
    cy = CENTER + dy
    return cx - half, cy - half


def write_sensor_files(sample_dir: Path, sensor: str, c0, c90, c360, cmask) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    p0 = sample_dir / f"{sensor}_0.tif"
    p90 = sample_dir / f"{sensor}_90.tif"
    p360 = sample_dir / f"{sensor}_360.tif"
    pm = sample_dir / f"{sensor}_plume.tif"

    tifffile.imwrite(str(p0), c0)
    tifffile.imwrite(str(p90), c90)
    tifffile.imwrite(str(p360), c360)
    tifffile.imwrite(str(pm), cmask)

    paths[f"{sensor}_0_path"] = str(p0)
    paths[f"{sensor}_90_path"] = str(p90)
    paths[f"{sensor}_360_path"] = str(p360)
    paths[f"{sensor}_plume_path"] = str(pm)
    return paths


def process_row(row: pd.Series, counter: Counter) -> Tuple[List[Dict], Dict[str, object]]:
    loaded = load_row_sensors(row)
    dbg: Dict[str, object] = {
        "plume_id": row.get("plume_id"),
        "loaded_sensors": sorted(list(loaded.keys())),
        "reason": "ok",
    }
    if len(loaded) == 0:
        dbg["reason"] = "no_sensor_loaded"
        return [], dbg

    overlap_mode = len(loaded) >= 2
    anchor = anchor_sensor(loaded)
    dbg["overlap_mode"] = overlap_mode
    dbg["anchor_sensor"] = anchor
    offsets = sample_offsets_overlap(anchor) if overlap_mode else sample_offsets_single(next(iter(loaded.keys())))

    out_rows: List[Dict] = []

    for label, dx_anchor, dy_anchor in offsets:
        sample_id = counter.next()
        sample_dir = OUT_ROOT / f"group_{sample_id:08d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        rec = {
            "id": sample_id,
            "plume_id": row["plume_id"],
            "label": int(label),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "datetime": normalize_datetime_iso(row.get("datetime")),
            "overlap_mode": bool(overlap_mode),
            "anchor_sensor": anchor,
            "dx_anchor_px": int(dx_anchor),
            "dy_anchor_px": int(dy_anchor),
        }

        ok = True
        for sensor, bundle in loaded.items():
            x, y = compute_top_left(sensor, anchor, dx_anchor, dy_anchor, overlap_mode, bundle if sensor == "s5p" else None)

            if sensor == "s5p":
                cx = x + PATCH_SIZE["s5p"] // 2
                cy = y + PATCH_SIZE["s5p"] // 2
                c0 = s5p_crop_and_resize(bundle.t0, cx, cy, PATCH_SIZE["s5p"])
                c90 = s5p_crop_and_resize(bundle.t90, cx, cy, PATCH_SIZE["s5p"])
                c360 = s5p_crop_and_resize(bundle.t360, cx, cy, PATCH_SIZE["s5p"])
                if c0 is None or c90 is None or c360 is None:
                    ok = False
                    break
                c0 = c0[None, :, :]
                c90 = c90[None, :, :]
                c360 = c360[None, :, :]
                # No source plume mask from table for S5P; use center marker mask.
                sm = np.zeros((PATCH_SIZE["s5p"], PATCH_SIZE["s5p"]), dtype=np.float32)
                sm[PATCH_SIZE["s5p"] // 2, PATCH_SIZE["s5p"] // 2] = 1.0
                cm = resize_hw(sm, TARGET_SIZE)
            else:
                ps = PATCH_SIZE[sensor]
                c0_raw = crop_chw(bundle.t0, x, y, ps)
                c90_raw = crop_chw(bundle.t90, x, y, ps)
                c360_raw = crop_chw(bundle.t360, x, y, ps)
                m_raw = crop_hw(bundle.mask, x, y, ps) if bundle.mask is not None else None
                if c0_raw is None or c90_raw is None or c360_raw is None or m_raw is None:
                    ok = False
                    break

                # sensor-specific quality check
                if sensor == "s2":
                    if not s2_valid_crop(c0_raw, c90_raw, c360_raw):
                        ok = False
                        break
                elif sensor == "l89":
                    if not missing_ratio_valid(c0_raw, c90_raw, c360_raw, L89_MISSING_THRESH):
                        ok = False
                        break
                elif sensor == "emit":
                    if not missing_ratio_valid(c0_raw, c90_raw, c360_raw, EMIT_MISSING_THRESH):
                        ok = False
                        break

                c0 = resize_chw(c0_raw, TARGET_SIZE)
                c90 = resize_chw(c90_raw, TARGET_SIZE)
                c360 = resize_chw(c360_raw, TARGET_SIZE)
                cm = resize_hw(m_raw, TARGET_SIZE)

            rec.update(write_sensor_files(sample_dir, sensor, c0, c90, c360, cm))

        if ok:
            out_rows.append(rec)
        else:
            # best effort cleanup for failed sample
            try:
                for f in sample_dir.glob("*"):
                    f.unlink(missing_ok=True)
                sample_dir.rmdir()
            except Exception:
                pass

    if len(out_rows) == 0:
        dbg["reason"] = "all_crops_filtered_or_failed"
    dbg["sample_count"] = len(out_rows)
    return out_rows, dbg


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master_csv, low_memory=False)
    if args.max_rows > 0:
        df = df.head(args.max_rows)

    existing_df = pd.DataFrame()
    processed_plume_ids = set()
    start_id = 0
    if args.resume and Path(args.out_csv).exists():
        try:
            existing_df = pd.read_csv(args.out_csv, low_memory=False)
            if "plume_id" in existing_df.columns:
                processed_plume_ids = set(existing_df["plume_id"].dropna().astype(str).unique())
            if "id" in existing_df.columns and len(existing_df) > 0:
                start_id = int(pd.to_numeric(existing_df["id"], errors="coerce").max())
            print(f"[resume] loaded existing manifest rows={len(existing_df)}, processed_plume_ids={len(processed_plume_ids)}, start_id={start_id}")
        except Exception as e:
            print(f"[resume] failed to load existing manifest: {e}. start fresh.")

    if processed_plume_ids:
        before = len(df)
        df = df[~df["plume_id"].astype(str).isin(processed_plume_ids)].copy()
        print(f"[resume] skip already processed plume_id: {before - len(df)} / {before}")

    counter = Counter(start=start_id)
    all_rows: List[Dict] = existing_df.to_dict("records") if len(existing_df) > 0 else []

    rows = [row for _, row in df.iterrows()]
    total = len(rows)

    def _work(r):
        return process_row(r, counter)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, result in enumerate(ex.map(_work, rows), start=1):
            recs, dbg = result
            if recs:
                all_rows.extend(recs)
            if args.debug and (i <= args.debug_limit or i % args.debug_every == 0 or dbg.get("reason") != "ok"):
                print(
                    f"[debug] {i}/{total} plume_id={dbg.get('plume_id')} "
                    f"loaded={dbg.get('loaded_sensors')} overlap={dbg.get('overlap_mode')} "
                    f"anchor={dbg.get('anchor_sensor')} reason={dbg.get('reason')} "
                    f"samples={dbg.get('sample_count', len(recs))}"
                )
            if i % 50 == 0 or i == total:
                print(f"processed {i}/{total}, samples={len(all_rows)}")

    out_df = pd.DataFrame(all_rows)

    preferred_cols = [
        "id",
        "plume_id",
        "label",
        "latitude",
        "longitude",
        "datetime",
        "overlap_mode",
        "anchor_sensor",
        "dx_anchor_px",
        "dy_anchor_px",
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
        "s2_plume_path",
        "l89_0_path",
        "l89_90_path",
        "l89_360_path",
        "l89_plume_path",
        "emit_0_path",
        "emit_90_path",
        "emit_360_path",
        "emit_plume_path",
        "s5p_0_path",
        "s5p_90_path",
        "s5p_360_path",
        "s5p_plume_path",
    ]

    for c in preferred_cols:
        if c not in out_df.columns:
            out_df[c] = pd.NA
    out_df = out_df[preferred_cols]

    out_df.to_csv(args.out_csv, index=False)
    print(f"saved: {args.out_csv}")
    print(f"rows: {len(out_df)}")
    if len(out_df) > 0:
        print("label sum:", int(out_df["label"].sum()), "/", len(out_df))


def parse_args():
    p = argparse.ArgumentParser(description="Unified multisensor crop with physical-anchor alignment.")
    p.add_argument("--master_csv", type=Path, default=MASTER_CSV)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--out_csv", type=Path, default=OUT_CSV)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=0, help="for quick smoke test")
    p.add_argument("--resume", action="store_true", help="resume from existing manifest and skip processed plume_id")
    p.add_argument("--debug", action="store_true", help="print debug diagnostics")
    p.add_argument("--debug_every", type=int, default=50, help="debug print frequency")
    p.add_argument("--debug_limit", type=int, default=20, help="always print debug for first N rows")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUT_ROOT = args.out_root
    OUT_CSV = args.out_csv
    run(args)
