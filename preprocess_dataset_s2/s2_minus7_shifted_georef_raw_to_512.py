import os
import re
import time
from pathlib import Path
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform
from tqdm import tqdm

# =========================
# Translated comment
# =========================
CSV_PATH = "/data2/yuyao/methane_emission/preprocess_dataset_s2/CM_S2_L2A_gee90360.csv"
MANIFEST_MINUS7 = "/data2/yuyao/methane_emission/preprocess_dataset_s2/manifest_minus7_plume_to_safe.csv"
RAW_DIR_MINUS7 = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7"

OUT_BASE = "/data2/yuyao/methane_emission/carbon_mapper_data/CM_S2_L2A"
OUT_NAME = "s2_-7.tif"

PATCH_SIZE = 512
HALF = PATCH_SIZE // 2

MAX_WORKERS = 12

# Translated comment
PRINT_EVERY = 200  # Translated comment
MAX_FAIL_DETAIL = 10  # Translated comment

# Translated comment
SLOT_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B8A","B09","B10","B11","B12"]

JP2_NAME_RE = re.compile(r"^(T\d{2}[A-Z]{3})_(\d{8}T\d{6})_(B(?:0[1-9]|1[0-2]|8A))_20m\.jp2$")


# =========================
# Helpers
# =========================
def to_rel_under_root(abs_path: str) -> str:
    root = "/data2/yuyao/methane_emission/"
    return abs_path[len(root):] if abs_path.startswith(root) else abs_path

def find_any_jp2_in_safe(safe_dir: str):
    p = Path(safe_dir)
    for pat in ["*_B11_20m.jp2", "*_B12_20m.jp2", "*_B8A_20m.jp2", "*_B02_20m.jp2", "*_20m.jp2"]:
        hits = list(p.glob(pat))
        if hits:
            return str(hits[0])
    return None

def parse_tile_sensing_from_any_jp2(jp2_path: str):
    base = os.path.basename(jp2_path)
    m = JP2_NAME_RE.match(base)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def band_jp2_path(safe_dir: str, tile: str, sensing: str, band: str) -> str | None:
    p = Path(safe_dir) / f"{tile}_{sensing}_{band}_20m.jp2"
    return str(p) if p.exists() else None

def lonlat_to_pixel(ds, lon, lat):
    xs, ys = rio_transform("EPSG:4326", ds.crs, [lon], [lat])
    x, y = xs[0], ys[0]
    col, row = ~ds.transform * (x, y)
    return float(row), float(col)

def compute_window_shift_clamp(ds, center_row, center_col):
    if ds.width < PATCH_SIZE or ds.height < PATCH_SIZE:
        return None, None, None

    row0 = int(np.floor(center_row)) - HALF
    col0 = int(np.floor(center_col)) - HALF

    row0c = min(max(row0, 0), ds.height - PATCH_SIZE)
    col0c = min(max(col0, 0), ds.width - PATCH_SIZE)

    shift_r = abs(row0c - row0)
    shift_c = abs(col0c - col0)

    win = Window(col0c, row0c, PATCH_SIZE, PATCH_SIZE)
    return win, int(shift_r), int(shift_c)

def write_stack_geotiff(out_path: str, stack_bhw: np.ndarray, ref_ds, win: Window):
    meta = ref_ds.meta.copy()
    meta.update({
        "driver": "GTiff",
        "count": stack_bhw.shape[0],
        "height": stack_bhw.shape[1],
        "width": stack_bhw.shape[2],
        "crs": ref_ds.crs,
        "transform": ref_ds.window_transform(win),
        "dtype": stack_bhw.dtype,
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(stack_bhw)

def build_minus7_for_plume(plume_id: str, lat: float, lon: float, safe_name: str):
    """
 success dict; failure dict(ok=0, reason=...)
    """
    safe_dir = os.path.join(RAW_DIR_MINUS7, safe_name)
    if not os.path.isdir(safe_dir):
        return {"plume_id": plume_id, "ok": 0, "reason": "safe_dir_missing", "safe": safe_name}

    any_jp2 = find_any_jp2_in_safe(safe_dir)
    if any_jp2 is None:
        return {"plume_id": plume_id, "ok": 0, "reason": "no_jp2_in_safe", "safe": safe_name}

    tile, sensing = parse_tile_sensing_from_any_jp2(any_jp2)
    if tile is None:
        return {"plume_id": plume_id, "ok": 0, "reason": "jp2_name_parse_fail", "safe": safe_name, "jp2": any_jp2}

    ref_path = band_jp2_path(safe_dir, tile, sensing, "B11") or any_jp2
    if not os.path.exists(ref_path):
        return {"plume_id": plume_id, "ok": 0, "reason": "ref_jp2_missing", "safe": safe_name}

    try:
        with rasterio.open(ref_path) as ref_ds:
            if ref_ds.crs is None:
                return {"plume_id": plume_id, "ok": 0, "reason": "ref_crs_none", "safe": safe_name, "ref": ref_path}

            center_row, center_col = lonlat_to_pixel(ref_ds, lon, lat)

            if not (0 <= center_row < ref_ds.height and 0 <= center_col < ref_ds.width):
                return {
                    "plume_id": plume_id, "ok": 0, "reason": "point_outside_tile",
                    "safe": safe_name, "ref": ref_path, "row": center_row, "col": center_col,
                    "h": ref_ds.height, "w": ref_ds.width
                }

            win, shift_r, shift_c = compute_window_shift_clamp(ref_ds, center_row, center_col)
            if win is None:
                return {"plume_id": plume_id, "ok": 0, "reason": "tile_smaller_than_512", "safe": safe_name}

            stack = np.zeros((12, PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
            missing = []

            for i, band in enumerate(SLOT_BANDS):
                p = band_jp2_path(safe_dir, tile, sensing, band)
                if p is None:
                    missing.append(band)
                    continue
                with rasterio.open(p) as ds:
                    arr = ds.read(1, window=win)
                    if arr.shape != (PATCH_SIZE, PATCH_SIZE):
                        return {"plume_id": plume_id, "ok": 0, "reason": "window_read_bad_shape", "safe": safe_name, "band": band, "shape": str(arr.shape)}
                    stack[i] = arr.astype(np.uint16, copy=False)

            out_abs = os.path.join(OUT_BASE, plume_id, OUT_NAME)
            write_stack_geotiff(out_abs, stack, ref_ds, win)

            return {
                "plume_id": plume_id,
                "ok": 1,
                "safe": safe_name,
                "tile": tile,
                "sensing": sensing,
                "out_abs": out_abs,
                "out_rel": to_rel_under_root(out_abs),
                "shift_row_px": shift_r,
                "shift_col_px": shift_c,
                "missing_bands": ",".join(missing),
            }
    except Exception as e:
        return {"plume_id": plume_id, "ok": 0, "reason": f"exception:{e}", "safe": safe_name, "ref": ref_path}


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(CSV_PATH)
    man = pd.read_csv(MANIFEST_MINUS7)

    man = man[(man["ok"] == 1) & (man["raw_dir_exists"] == 1)]
    m = man[["plume_id", "s2_minus7_product_name"]].dropna().drop_duplicates("plume_id")
    safe_map = dict(zip(m["plume_id"].astype(str), m["s2_minus7_product_name"].astype(str)))

    if "s2_-7_path" not in df.columns:
        df["s2_-7_path"] = ""

    # worklist
    tasks = []
    for i, r in df.iterrows():
        pid = str(r.get("plume_id", ""))
        if pid in safe_map:
            tasks.append((i, pid, float(r["plume_latitude"]), float(r["plume_longitude"]), safe_map[pid]))

    print(f"[INFO] CSV rows={len(df)} | tasks={len(tasks)} | RAW_DIR_MINUS7={RAW_DIR_MINUS7}")

    fail_counter = Counter()
    last200 = deque(maxlen=200)
    fail_detail_printed = 0

    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(build_minus7_for_plume, pid, lat, lon, safe): (idx, pid, safe)
                for (idx, pid, lat, lon, safe) in tasks}

        pbar = tqdm(total=len(futs), desc="Build s2_-7.tif (512x512 georef, shift if needed)", ncols=100)

        done = ok = 0
        for fut in as_completed(futs):
            idx, pid, safe = futs[fut]
            res = fut.result()
            results.append({**res, "idx": idx})

            done += 1
            if res.get("ok") == 1:
                ok += 1
                last200.append(1)
            else:
                last200.append(0)
                reason = res.get("reason", "unknown_fail")
                fail_counter[reason] += 1

                # Translated comment
                if fail_detail_printed < MAX_FAIL_DETAIL:
                    tqdm.write(f"[FAIL] plume={pid} safe={safe} reason={reason} extra={ {k:v for k,v in res.items() if k not in ('ok','plume_id','safe','reason')} }")
                    fail_detail_printed += 1

            # tqdm postfix
            if done % 10 == 0:
                recent_ok = (sum(last200) / len(last200)) if len(last200) else 0.0
                pbar.set_postfix({
                    "ok": ok,
                    "fail": done - ok,
                    "ok_rate_200": f"{recent_ok:.2%}"
                })

            # Translated comment
            if done % PRINT_EVERY == 0:
                elapsed = time.time() - t0
                recent_ok = (sum(last200) / len(last200)) if len(last200) else 0.0
                top5 = fail_counter.most_common(5)
                tqdm.write(f"[STAT] done={done}/{len(futs)} ok={ok} ({ok/done:.2%}) recent200_ok={recent_ok:.2%} elapsed={elapsed/60:.1f} min")
                tqdm.write(f"[STAT] top_fail_reasons={top5}")

            pbar.update(1)

        pbar.close()

    res_df = pd.DataFrame(results)

    # Translated comment
    ok_df = res_df[res_df["ok"] == 1].set_index("idx")
    for idx, row in ok_df.iterrows():
        df.at[idx, "s2_-7_path"] = row["out_rel"]

    # Translated comment
    df.to_csv(CSV_PATH, index=False)

    # Translated comment
    log_csv = os.path.join(os.path.dirname(CSV_PATH), "build_s2_minus7_log.csv")
    res_df.to_csv(log_csv, index=False)

    print("\n==== DONE ====")
    print("Updated CSV:", CSV_PATH)
    print("Log CSV    :", log_csv)
    print("ok:", int((res_df["ok"] == 1).sum()), "/", len(res_df))
    print("Top fail reasons:", fail_counter.most_common(10))

if __name__ == "__main__":
    main()
