import os
import time
import random
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

# =========================
# Config (EDIT THESE)
# =========================
TRAIN_CSV  = "/data2/yuyao/methane_emission/data_csv/s2_-790360_temporal_CDSE0_gee90360_2024/train.csv"
TEST_CSV   = "/data2/yuyao/methane_emission/data_csv/s2_-790360_temporal_CDSE0_gee90360_2024/test.csv"

OUT_DIR    = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_s2_-790360_32_2024"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# plume masks root (your aligned resized_512)
MASK_ROOT = "/data2/yuyao/methane_emission/carbon_mapper_data_masks"
MASK_FILENAME = "resized_512x512.tif"   # confirm your exact filename

PATCH_SIZE = 32
CENTER_BOX = 256          # sample crop center inside center 256x256 box
MAX_TRIES_POS = 120       # pos sample retry cap per plume
MAX_TRIES_NEG = 120       # neg sample retry cap per plume
N_POS = 16
N_NEG = 16

MAX_WORKERS = 16
FLUSH_EVERY_SEC = 60
STAT_EVERY = 200

# optional zero filter
BAND_INDEX = 11           # B12 (0-based)
ZERO_RATIO_THRESH = 0.20

# make NEG cleaner by requiring mask==0 in the patch
NEG_REQUIRE_MASK_EMPTY = True

# =========================
# Utils
# =========================
def debug(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][pid:{os.getpid()}][tid:{threading.get_ident()}] {msg}", flush=True)

def to_chw(arr):
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected ndim={arr.ndim}, shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4, 12, 13):   # CHW
        return arr
    if arr.shape[-1] in (1, 3, 4, 12, 13):  # HWC
        return np.transpose(arr, (2, 0, 1))
    # fallback
    if arr.shape[0] <= 20 and arr.shape[0] < arr.shape[-1]:
        return arr
    return np.transpose(arr, (2, 0, 1))

def band_zero_too_much(chw):
    if chw is None or chw.shape[0] <= BAND_INDEX:
        return False
    total = chw[BAND_INDEX].size
    if total == 0:
        return True
    return (chw[BAND_INDEX] == 0).sum() / total >= ZERO_RATIO_THRESH

def crop_chw(chw, x, y, size):
    return chw[:, y:y+size, x:x+size]

def crop_hw(hw, x, y, size):
    return hw[y:y+size, x:x+size]

def sample_center_xy(width=512, height=512, patch_size=32, center_box=256):
    cx = width // 2
    cy = height // 2
    half_box = center_box // 2

    min_center_x = cx - half_box
    max_center_x = cx + half_box
    min_center_y = cy - half_box
    max_center_y = cy + half_box

    min_x = max(0, min_center_x - patch_size // 2)
    max_x = min(width - patch_size, max_center_x - patch_size // 2)
    min_y = max(0, min_center_y - patch_size // 2)
    max_y = min(height - patch_size, max_center_y - patch_size // 2)

    if min_x > max_x or min_y > max_y:
        min_x, max_x = 0, width - patch_size
        min_y, max_y = 0, height - patch_size

    return random.randint(min_x, max_x), random.randint(min_y, max_y)

def ensure_abs(p):
    if not isinstance(p, str):
        return ""
    p = p.strip()
    if not p:
        return ""
    return p if os.path.isabs(p) else os.path.abspath(p)

def mask_path_for_plume(plume_id: str) -> str:
    # common layout: <MASK_ROOT>/<plume_id>/resized_512x512.tif
    return os.path.join(MASK_ROOT, plume_id, MASK_FILENAME)

def read_chw_512(path, tag, bug):
    path = ensure_abs(path)
    if not path or not os.path.exists(path):
        bug.append(f"{tag}_missing:{path}")
        return None
    try:
        arr = tifffile.imread(path)
        chw = to_chw(arr).astype(np.float32)
        if chw.shape[-2:] != (512, 512):
            bug.append(f"{tag}_not512:{chw.shape}")
            return None
        return chw
    except Exception as e:
        bug.append(f"{tag}_read_err:{e}")
        return None

def read_mask_512(plume_id, bug):
    mp = mask_path_for_plume(plume_id)
    mp = ensure_abs(mp)
    if not mp or not os.path.exists(mp):
        bug.append(f"mask_missing:{mp}")
        return None, mp
    try:
        m = tifffile.imread(mp)
        if m.shape != (512, 512):
            bug.append(f"mask_not512:{m.shape}")
            return None, mp
        return m, mp
    except Exception as e:
        bug.append(f"mask_read_err:{e}")
        return None, mp

# =========================
# Per-plume crop
# =========================
def crop_one_plume(row, out_root, split_name):
    pid = str(row.get("plume_id", "")).strip()
    if not pid:
        return [], "missing_plume_id"

    bug = []

    # IMPORTANT: use std512 columns from train/test.csv directly
    p_t0   = ensure_abs(row.get("s2_0_std_512", ""))
    p_m7   = ensure_abs(row.get("s2_-7_std_512", ""))
    p_m90  = ensure_abs(row.get("s2_-90_std_512", ""))
    p_m360 = ensure_abs(row.get("s2_-360_std_512", ""))

    # record raw values for debugging
    rawvals = f"rawvals:s2_0_std_512={row.get('s2_0_std_512','')}|s2_-7_std_512={row.get('s2_-7_std_512','')}|s2_-90_std_512={row.get('s2_-90_std_512','')}|s2_-360_std_512={row.get('s2_-360_std_512','')}"

    t0   = read_chw_512(p_t0,   "t0",   bug)
    m7   = read_chw_512(p_m7,   "m7",   bug)
    m90  = read_chw_512(p_m90,  "m90",  bug)
    m360 = read_chw_512(p_m360, "m360", bug)

    if t0 is None or m7 is None or m90 is None or m360 is None:
        # add exists checks (super helpful)
        exinfo = f"exists:t0={os.path.exists(p_t0) if p_t0 else False},m7={os.path.exists(p_m7) if p_m7 else False},m90={os.path.exists(p_m90) if p_m90 else False},m360={os.path.exists(p_m360) if p_m360 else False}"
        return [], ";".join(bug + [rawvals, exinfo])

    mask512, maskp = read_mask_512(pid, bug)
    if mask512 is None:
        # without mask we cannot guarantee "pos must contain plume"
        return [], ";".join(bug + [f"mask_path={maskp}", rawvals])

    plume_dir = Path(out_root) / split_name / pid
    plume_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    def write_sample(kind, k, x, y, img_chw, pre90_chw, pre360_chw, label):
        d = plume_dir / f"{kind}_{k:02d}_x{x}_y{y}"
        d.mkdir(parents=True, exist_ok=True)

        img_path  = str(d / "image.tif")      # t0 or -7
        p90_path  = str(d / "s2_-90.tif")
        p360_path = str(d / "s2_-360.tif")

        tifffile.imwrite(img_path,  img_chw)
        tifffile.imwrite(p90_path,  pre90_chw)
        tifffile.imwrite(p360_path, pre360_chw)

        samples.append({
            "plume_id": pid,
            "split": split_name,
            "label": int(label),
            "image_path": img_path,
            "s2_pre_path": p90_path,
            "s2_pre_pre_path": p360_path,
            "crop_x": int(x),
            "crop_y": int(y),
            "source": "t0" if label == 1 else "m7",
            "mask_path_512": maskp,
        })

    # -------- POS: require plume pixels in mask patch --------
    pos_written = 0
    tries = 0
    while pos_written < N_POS and tries < MAX_TRIES_POS:
        tries += 1
        x, y = sample_center_xy(512, 512, PATCH_SIZE, CENTER_BOX)

        cm = crop_hw(mask512, x, y, PATCH_SIZE)
        if cm.shape != (PATCH_SIZE, PATCH_SIZE):
            continue
        if cm.sum() <= 0:
            continue

        c0   = crop_chw(t0,   x, y, PATCH_SIZE)
        c90  = crop_chw(m90,  x, y, PATCH_SIZE)
        c360 = crop_chw(m360, x, y, PATCH_SIZE)
        if c0.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
            continue
        if band_zero_too_much(c0) or band_zero_too_much(c90) or band_zero_too_much(c360):
            continue

        write_sample("pos", pos_written, x, y, c0, c90, c360, label=1)
        pos_written += 1

    if pos_written < N_POS:
        bug.append(f"pos_insufficient:{pos_written}/{N_POS}")

    # -------- NEG: sample in center; optionally require no plume pixels --------
    neg_written = 0
    tries = 0
    while neg_written < N_NEG and tries < MAX_TRIES_NEG:
        tries += 1
        x, y = sample_center_xy(512, 512, PATCH_SIZE, CENTER_BOX)

        if NEG_REQUIRE_MASK_EMPTY:
            cm = crop_hw(mask512, x, y, PATCH_SIZE)
            if cm.shape != (PATCH_SIZE, PATCH_SIZE):
                continue
            if cm.sum() > 0:
                continue

        c7   = crop_chw(m7,   x, y, PATCH_SIZE)
        c90  = crop_chw(m90,  x, y, PATCH_SIZE)
        c360 = crop_chw(m360, x, y, PATCH_SIZE)
        if c7.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
            continue
        if band_zero_too_much(c7) or band_zero_too_much(c90) or band_zero_too_much(c360):
            continue

        write_sample("neg", neg_written, x, y, c7, c90, c360, label=0)
        neg_written += 1

    if neg_written < N_NEG:
        bug.append(f"neg_insufficient:{neg_written}/{N_NEG}")

    # must have full set
    if len(samples) < (N_POS + N_NEG):
        return samples, ";".join(bug + [rawvals])

    return samples, ";".join(bug) if bug else ""

# =========================
# Runner (skip/stat/flush/resume)
# =========================
def run_split(split_name, split_csv, out_csv_path):
    df = pd.read_csv(split_csv)
    if "plume_id" not in df.columns:
        raise RuntimeError(f"{split_csv} missing plume_id")

    need = ["s2_0_std_512", "s2_-7_std_512", "s2_-90_std_512", "s2_-360_std_512"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"{split_csv} missing columns: {miss}")

    debug(f"[{split_name}] rows={len(df)} from {split_csv}")

    out_rows = []
    out_lock = threading.Lock()

    # resume
    if os.path.exists(out_csv_path):
        try:
            prev = pd.read_csv(out_csv_path)
            out_rows.extend(prev.to_dict("records"))
            debug(f"[{split_name}] resume loaded rows={len(prev)} from {out_csv_path}")
        except Exception as e:
            debug(f"[{split_name}] resume load failed (start fresh): {e}")

    done_plumes = set()
    if out_rows:
        tmp = pd.DataFrame(out_rows)
        cnts = tmp.groupby("plume_id").size()
        done_plumes = set(cnts[cnts >= (N_POS + N_NEG)].index.astype(str).tolist())
        debug(f"[{split_name}] resume done_plumes={len(done_plumes)}")

    flush_state = {"last": time.time()}
    def flush(force=False):
        now = time.time()
        if force or (now - flush_state["last"] >= FLUSH_EVERY_SEC):
            tmp_path = out_csv_path + ".part"
            with out_lock:
                pd.DataFrame(out_rows).to_csv(tmp_path, index=False)
                os.replace(tmp_path, out_csv_path)
            flush_state["last"] = now
            debug(f"[{split_name}] flushed: {out_csv_path}")

    tasks = []
    skip = 0
    for _, r in df.iterrows():
        pid = str(r["plume_id"])
        if pid in done_plumes:
            skip += 1
            continue
        tasks.append(r.to_dict())
    debug(f"[{split_name}] to_process={len(tasks)} skip={skip}")

    ok = fail = 0
    recent = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(crop_one_plume, row, OUT_DIR, split_name): row.get("plume_id") for row in tasks}

        for i, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc=f"{split_name}: crop pos/neg", mininterval=1.0), start=1):
            pid = str(futs[fut])
            try:
                samples, bug = fut.result()
            except Exception as e:
                samples, bug = [], f"worker_exception:{e}"

            if len(samples) >= (N_POS + N_NEG):
                ok += 1
                recent.append(1)
                with out_lock:
                    out_rows.extend(samples)
            else:
                fail += 1
                recent.append(0)
                debug(f"[{split_name}][{pid}] FAIL bug={bug}")

            if len(recent) > STAT_EVERY:
                recent = recent[-STAT_EVERY:]

            if (i % STAT_EVERY) == 0:
                recent_ok = sum(recent) / max(1, len(recent))
                elapsed_min = (time.time() - start) / 60
                debug(f"[{split_name}][STAT] done={i}/{len(tasks)} ok={ok} fail={fail} skip={skip} recent{STAT_EVERY}_ok={recent_ok:.2%} elapsed={elapsed_min:.1f} min")
                flush(False)

    flush(True)
    debug(f"[{split_name}] DONE ok={ok} fail={fail} skip={skip} total_out_rows={len(out_rows)}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    out_train = os.path.join(OUT_DIR, "train_patches.csv")
    out_test  = os.path.join(OUT_DIR, "test_patches.csv")

    run_split("train", TRAIN_CSV, out_train)
    run_split("test",  TEST_CSV,  out_test)

    debug("ALL DONE.")
