import os
import random
import warnings
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
from netCDF4 import Dataset


# ============== input ==============
IN_CSV = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/s5p_all_OFFL_with_centers.csv"

OUT_ROOT = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_5x5_to_32_offl_triplet")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_CSV = "./s5p_samples_5x5_to_32_triplet.csv"

POS_DIR = OUT_ROOT / "samples2" / "pos"
NEG_DIR = OUT_ROOT / "samples2" / "neg"
POS_DIR.mkdir(parents=True, exist_ok=True)
NEG_DIR.mkdir(parents=True, exist_ok=True)

# ============== crop config ==============
CROP_SIZE = 5
CROP_HALF = CROP_SIZE // 2
OUT_SIZE = 32

MAX_MISSING_RATIO_T0 = 0.50

NEG_EXCLUDE_HALF = 5           # outside 11x11
NEG_RANDOM_TRIES = 50          # ✅ 按你说的：随机 50 次
# corners are tried first

# parallel
MAX_WORKERS = 8                # 网络盘建议 4~8
PRINT_EVERY = 50

CH4_CANDIDATES = [
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
]

warnings.filterwarnings("ignore", category=RuntimeWarning)


@contextmanager
def silence_fd2():
    """Kill HDF5/getfattr spam."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(devnull)
        os.close(old)


def get_2d(a):
    a = np.asarray(a)
    if a.ndim == 3:
        return a[0]
    if a.ndim == 2:
        return a
    raise ValueError(f"Unexpected dims: {a.shape}")


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
    return get_2d(a)


def crop_center(a2d, cy, cx, half):
    H, W = a2d.shape
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return a2d[y0:y1, x0:x1]


def missing_ratio(patch2d):
    return 1.0 - (np.isfinite(patch2d).sum() / patch2d.size)


def nan_out():
    return np.full((OUT_SIZE, OUT_SIZE), np.nan, dtype=np.float32)


def resize_nan_aware(src2d):
    src = src2d.astype(np.float32, copy=False)
    fin = np.isfinite(src)
    v = np.where(fin, src, 0.0).astype(np.float32)
    w = fin.astype(np.float32)

    v_r = cv2.resize(v, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    w_r = cv2.resize(w, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w_r > 1e-6, v_r / w_r, np.nan).astype(np.float32)


def parse_centers(s):
    # "cy,cx;cy,cx" -> [(cy,cx),...]
    s = str(s) if s is not None else ""
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        cy, cx = part.split(",")
        out.append((int(float(cy)), int(float(cx))))
    return out


def pick_ch4_var(prod):
    for k in CH4_CANDIDATES:
        if k in prod.variables:
            return k
    return None


def read_ch4(path_nc, ch4name_hint=None):
    """
    Read PRODUCT/ch4 -> 2D float32 with NaNs
    returns (ch4_2d or None, ok_bool, ch4name_used)
    """
    if not path_nc or str(path_nc).lower() == "nan":
        return None, False, ch4name_hint

    p = Path(str(path_nc))
    if not p.exists():
        return None, False, ch4name_hint

    try:
        with silence_fd2():
            ds = Dataset(str(p), "r")
        prod = ds.groups["PRODUCT"]

        ch4name = ch4name_hint if (ch4name_hint and ch4name_hint in prod.variables) else pick_ch4_var(prod)
        if ch4name is None:
            ds.close()
            return None, False, ch4name_hint

        v = prod.variables[ch4name]
        a = to_nan_invalid(v[:], getattr(v, "__dict__", {}))
        ds.close()
        return a, True, ch4name
    except Exception:
        return None, False, ch4name_hint


def outside_exclude_11x11(py, px, cy, cx):
    return not (abs(cy - py) <= NEG_EXCLUDE_HALF and abs(cx - px) <= NEG_EXCLUDE_HALF)


def find_neg_center(ch4_t0, py, px, seed):
    """
    Neg selection on t0 only:
    1) try 4 corners (in-bounds and outside 11x11)
    2) random try 50 times; first valid -> stop
    validity: crop exists and missing_ratio<=0.5 on t0
    """
    H, W = ch4_t0.shape
    y_min, y_max = CROP_HALF, H - CROP_HALF - 1
    x_min, x_max = CROP_HALF, W - CROP_HALF - 1
    if y_min > y_max or x_min > x_max:
        return None

    corners = [(y_min, x_min), (y_min, x_max), (y_max, x_min), (y_max, x_max)]
    # far first
    corners.sort(key=lambda c: (c[0]-py)**2 + (c[1]-px)**2, reverse=True)

    for cy, cx in corners:
        if not outside_exclude_11x11(py, px, cy, cx):
            continue
        p = crop_center(ch4_t0, cy, cx, CROP_HALF)
        if p is not None and missing_ratio(p) <= MAX_MISSING_RATIO_T0:
            return cy, cx

    rng = random.Random(seed)
    for _ in range(NEG_RANDOM_TRIES):
        cy = rng.randint(y_min, y_max)
        cx = rng.randint(x_min, x_max)
        if not outside_exclude_11x11(py, px, cy, cx):
            continue
        p = crop_center(ch4_t0, cy, cx, CROP_HALF)
        if p is not None and missing_ratio(p) <= MAX_MISSING_RATIO_T0:
            return cy, cx

    return None


def process_one(args):
    idx, row, pos_base, neg_base = args

    plume_id = str(row["plume_id"])
    plume_time = str(row["plume_time"])
    lat0 = float(row["lat"])
    lon0 = float(row["lon"])

    p0 = str(row["S5p_path"])
    p90 = str(row["s5p_minus90_path"])
    p360 = str(row["s5p_minus360_path"])

    py = int(float(row["nearest_iy"]))
    px = int(float(row["nearest_ix"]))
    pos_centers = parse_centers(row.get("pos_centers", ""))

    # must have at least one pos center
    if not pos_centers:
        raise RuntimeError("no_pos_centers_precomputed")

    # Read 3 arrays (each at most once)
    ch4_t0, ok0, ch4name_used = read_ch4(p0, row.get("ch4_var", None))
    if not ok0 or ch4_t0 is None:
        raise RuntimeError("open_t0_fail")

    ch4_90, ok90, _ = read_ch4(p90, ch4name_used)
    ch4_360, ok360, _ = read_ch4(p360, ch4name_used)

    H, W = ch4_t0.shape

    out_rows = []
    pos_dir = Path(pos_base) / plume_id
    neg_dir = Path(neg_base) / plume_id
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # for each pos, save pos; try neg
    for j, (cy, cx) in enumerate(pos_centers):
        pos0_small = crop_center(ch4_t0, cy, cx, CROP_HALF)
        if pos0_small is None or missing_ratio(pos0_small) > MAX_MISSING_RATIO_T0:
            continue

        pos0 = resize_nan_aware(pos0_small)

        if ok90 and ch4_90 is not None:
            p = crop_center(ch4_90, cy, cx, CROP_HALF)
            pos90 = resize_nan_aware(p) if p is not None else nan_out()
            has90 = True
        else:
            pos90 = nan_out()
            has90 = False

        if ok360 and ch4_360 is not None:
            p = crop_center(ch4_360, cy, cx, CROP_HALF)
            pos360 = resize_nan_aware(p) if p is not None else nan_out()
            has360 = True
        else:
            pos360 = nan_out()
            has360 = False

        pos_stack = np.stack([pos0, pos90, pos360], axis=0).astype(np.float32)
        pos_npz = pos_dir / f"s5p_pos_{j:02d}.npz"
        np.savez_compressed(
            pos_npz,
            ch4=pos_stack,
            meta={"label": 1, "plume_id": plume_id, "ch4_var": ch4name_used,
                  "center_iy": int(cy), "center_ix": int(cx),
                  "nearest_iy": int(py), "nearest_ix": int(px),
                  "has_90": bool(has90), "has_360": bool(has360)}
        )

        out_rows.append({
            "plume_id": plume_id, "plume_time": plume_time, "lat": lat0, "lon": lon0,
            "has_90": bool(has90), "has_360": bool(has360), "ch4_var": ch4name_used,
            "image_path": str(pos_npz), "center_iy": int(cy), "center_ix": int(cx), "label": 1
        })

        # neg on t0 only; if fail -> keep only pos
        neg_center = find_neg_center(ch4_t0, py, px, seed=idx * 1000 + j)
        if neg_center is None:
            continue
        ncy, ncx = neg_center
        neg0_small = crop_center(ch4_t0, ncy, ncx, CROP_HALF)
        if neg0_small is None or missing_ratio(neg0_small) > MAX_MISSING_RATIO_T0:
            continue

        neg0 = resize_nan_aware(neg0_small)

        if ok90 and ch4_90 is not None:
            p = crop_center(ch4_90, ncy, ncx, CROP_HALF)
            neg90 = resize_nan_aware(p) if p is not None else nan_out()
        else:
            neg90 = nan_out()

        if ok360 and ch4_360 is not None:
            p = crop_center(ch4_360, ncy, ncx, CROP_HALF)
            neg360 = resize_nan_aware(p) if p is not None else nan_out()
        else:
            neg360 = nan_out()

        neg_stack = np.stack([neg0, neg90, neg360], axis=0).astype(np.float32)
        neg_npz = neg_dir / f"s5p_neg_{j:02d}.npz"
        np.savez_compressed(
            neg_npz,
            ch4=neg_stack,
            meta={"label": 0, "plume_id": plume_id, "ch4_var": ch4name_used,
                  "center_iy": int(ncy), "center_ix": int(ncx),
                  "nearest_iy": int(py), "nearest_ix": int(px),
                  "has_90": bool(has90), "has_360": bool(has360)}
        )

        out_rows.append({
            "plume_id": plume_id, "plume_time": plume_time, "lat": lat0, "lon": lon0,
            "has_90": bool(has90), "has_360": bool(has360), "ch4_var": ch4name_used,
            "image_path": str(neg_npz), "center_iy": int(ncy), "center_ix": int(ncx), "label": 0
        })

    return out_rows


def main():
    df = pd.read_csv(IN_CSV, low_memory=False)

    req = ["plume_id","plume_time","lat","lon","S5p_path","s5p_minus90_path","s5p_minus360_path",
           "nearest_iy","nearest_ix","pos_centers"]
    for c in req:
        if c not in df.columns:
            raise RuntimeError(f"Missing col {c}")

    rows = df.to_dict("records")
    total = len(rows)

    tasks = [(i, rows[i], str(POS_DIR), str(NEG_DIR)) for i in range(total)]

    all_samples = []
    err_cnt = Counter()
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for fut in as_completed(futs):
            done += 1
            try:
                all_samples.extend(fut.result())
            except Exception as e:
                err_cnt[str(e)] += 1

            if (done % PRINT_EVERY) == 0 or done == total:
                pct = 100.0 * done / total
                print(f"[{done}/{total} | {pct:5.1f}%] samples={len(all_samples)} errors={sum(err_cnt.values())}")

    out = pd.DataFrame(all_samples)
    out = out[["plume_id","plume_time","lat","lon","has_90","has_360","ch4_var","image_path","center_iy","center_ix","label"]]
    out.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("Total plumes:", total, "Total samples:", len(out))
    print("Top errors:", err_cnt.most_common(10))


if __name__ == "__main__":
    main()
