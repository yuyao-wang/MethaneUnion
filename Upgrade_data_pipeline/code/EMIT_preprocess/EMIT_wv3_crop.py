import os
import random
import threading

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm

# =========================
# Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Translated comment
INPUT_CSV = "./merged_with_emit_tag.csv"

# Translated comment
MASK_ROOT = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_masks_wv3_512"
# Translated comment
MASK_NAME = "mask_60m_512.tif"

# Translated comment
SIM_OUTPUT_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/EMIT_simulated_WV3_L2A_60resolution_NOnorm"

# Translated comment
BASE_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224"
os.makedirs(BASE_DIR, exist_ok=True)

# Translated comment
CHIP_SIZE_PX = 512  # Translated comment
SCALE_M = 60  # Translated comment

PATCH_SIZE = 16  # Translated comment
TARGET_SIZE = 224  # Translated comment
CENTER_BOX = 6  # Translated comment
MISSING_THRESH = 0.25
N_POS = 16  # Translated comment
N_NEG = 16  # Translated comment
NUM_WORKERS = 18

counter_lock = threading.Lock()
global_cnt = 0  # Translated comment

# =========================
# Helpers
# =========================

def ensure_chw(img: np.ndarray) -> np.ndarray:
    """Translated to English."""
    if img.ndim == 2:
        return img[np.newaxis, ...]
    if img.ndim == 3 and img.shape[0] > 100:  # HWC -> CHW
        return img.transpose(2, 0, 1)
    return img

def gpu_upsample(crop_np: np.ndarray) -> np.ndarray:
    """Translated to English."""
    with torch.no_grad():
        img_t = torch.from_numpy(crop_np.astype(np.float32)).to(DEVICE).unsqueeze(0)
        img_t = torch.nan_to_num(img_t, nan=0.0)
        up = F.interpolate(img_t, size=(TARGET_SIZE, TARGET_SIZE),
                           mode="bilinear", align_corners=False)
        return up.squeeze(0).cpu().numpy()

def upsample_mask(mask_16: np.ndarray) -> np.ndarray:
    """Translated to English."""
    f = TARGET_SIZE // PATCH_SIZE  # 224/16=14
    return np.repeat(np.repeat(mask_16, f, axis=0), f, axis=1).astype("uint8")

def is_valid_crop(crop: np.ndarray) -> bool:
    """Translated to English."""
    if crop.size == 0:
        return False
    band0 = crop[0]
    missing = (np.isnan(band0) | (band0 == 0))
    return missing.mean() <= MISSING_THRESH

def get_three_paths(row) -> tuple[str, str, str] | tuple[None, None, None]:
    """Translated to English."""
    p0 = str(row["simulated_512_path"])
    if not p0.endswith("_sim_WV3.tif"):
        return None, None, None
    p90 = p0.replace("_sim_WV3.tif", "_-90_sim_WV3.tif")
    p180 = p0.replace("_sim_WV3.tif", "_-180_sim_WV3.tif")
    if not (os.path.exists(p0) and os.path.exists(p90) and os.path.exists(p180)):
        return None, None, None
    return p0, p90, p180

def get_mask_path(plume_id: str) -> str | None:
    mp = os.path.join(MASK_ROOT, plume_id, MASK_NAME)
    return mp if os.path.exists(mp) else None

def get_start_cnt(base_dir: str) -> int:
    """Translated to English."""
    ids = []
    for d in os.listdir(base_dir):
        if d.startswith("sample_"):
            try:
                ids.append(int(d.split("_")[-1]))
            except Exception:
                pass
    return max(ids) if ids else 0

# =========================
# Worker
# =========================

def process_single_row(row):
    global global_cnt
    plume_id = str(row["plume_id"])
    t0_path, t90_path, t180_path = get_three_paths(row)
    if t0_path is None:
        return []

    mask_path = get_mask_path(plume_id)
    if mask_path is None:
        return []

    try:
        t0 = ensure_chw(tifffile.imread(t0_path))
        t90 = ensure_chw(tifffile.imread(t90_path))
        t180 = ensure_chw(tifffile.imread(t180_path))
        full_mask = tifffile.imread(mask_path)
    except Exception as e:
        print(f"[READ ERROR] {plume_id}: {e}")
        return []

    if full_mask.shape != (CHIP_SIZE_PX, CHIP_SIZE_PX):
        return []

    local_meta = []

    # Translated comment
    p_start_min = CHIP_SIZE_PX // 2 + (CENTER_BOX // 2) - PATCH_SIZE
    p_start_max = CHIP_SIZE_PX // 2 - (CENTER_BOX // 2)
    pos_count, attempts = 0, 0

    while pos_count < N_POS and attempts < 100:
        attempts += 1
        x = random.randint(p_start_min, p_start_max)
        y = random.randint(p_start_min, p_start_max)

        c0   = t0[:,   y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        c90  = t90[:,  y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        c180 = t180[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        if not (is_valid_crop(c0) and is_valid_crop(c90) and is_valid_crop(c180)):
            continue

        m_crop = full_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        up0   = gpu_upsample(c0)
        up90  = gpu_upsample(c90)
        up180 = gpu_upsample(c180)
        m_up  = upsample_mask(m_crop)

        with counter_lock:
            global_cnt += 1
            sid = global_cnt

        out_dir = os.path.join(BASE_DIR, f"sample_{sid:06d}")
        os.makedirs(out_dir, exist_ok=True)

        p0   = os.path.join(out_dir, "wv3_t0.tif")
        p90  = os.path.join(out_dir, "wv3_-90.tif")
        p180 = os.path.join(out_dir, "wv3_-180.tif")
        pmask = os.path.join(out_dir, "plume.tif")

        tifffile.imwrite(p0,   up0)
        tifffile.imwrite(p90,  up90)
        tifffile.imwrite(p180, up180)
        tifffile.imwrite(pmask, m_up)

        local_meta.append({
            "sample_id": sid,
            "label": 1,
            "plume_id": plume_id,
            "data_path": out_dir,
            "path_t0": p0,
            "path_t90": p90,
            "path_t180": p180,
            "mask_path": pmask,
            "crop_x": x,
            "crop_y": y,
        })
        pos_count += 1

    # Translated comment
    neg_count, attempts = 0, 0
    while neg_count < N_NEG and attempts < 200:
        attempts += 1
        x = random.randint(0, CHIP_SIZE_PX - PATCH_SIZE)
        y = random.randint(0, CHIP_SIZE_PX - PATCH_SIZE)

        # Translated comment
        cx = CHIP_SIZE_PX // 2
        cy = CHIP_SIZE_PX // 2
        if (x <= cx <= x + PATCH_SIZE) and (y <= cy <= y + PATCH_SIZE):
            continue

        c0   = t0[:,   y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        c90  = t90[:,  y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        c180 = t180[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        if not (is_valid_crop(c0) and is_valid_crop(c90) and is_valid_crop(c180)):
            continue

        up0   = gpu_upsample(c0)
        up90  = gpu_upsample(c90)
        up180 = gpu_upsample(c180)
        m_up  = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype="uint8")

        with counter_lock:
            global_cnt += 1
            sid = global_cnt

        out_dir = os.path.join(BASE_DIR, f"sample_{sid:06d}")
        os.makedirs(out_dir, exist_ok=True)

        p0   = os.path.join(out_dir, "wv3_t0.tif")
        p90  = os.path.join(out_dir, "wv3_-90.tif")
        p180 = os.path.join(out_dir, "wv3_-180.tif")
        pmask = os.path.join(out_dir, "plume.tif")

        tifffile.imwrite(p0,   up0)
        tifffile.imwrite(p90,  up90)
        tifffile.imwrite(p180, up180)
        tifffile.imwrite(pmask, m_up)

        local_meta.append({
            "sample_id": sid,
            "label": 0,
            "plume_id": plume_id,
            "data_path": out_dir,
            "path_t0": p0,
            "path_t90": p90,
            "path_t180": p180,
            "mask_path": pmask,
            "crop_x": x,
            "crop_y": y,
        })
        neg_count += 1

    return local_meta

# =========================
# Main
# =========================

def main():
    global global_cnt

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Translated comment
    if "simulated_512_path" not in df.columns:
        df["simulated_512_path"] = df["plume_id"].astype(str).apply(
            lambda pid: os.path.join(SIM_OUTPUT_DIR, f"{pid}_sim_WV3.tif")
        )

    # Translated comment
    keep = []
    for _, row in df.iterrows():
        t0, t90, t180 = get_three_paths(row)
        if t0 is None:
            continue
        mp = get_mask_path(str(row["plume_id"]))
        if mp is None:
            continue
        keep.append(row)
    df_valid = pd.DataFrame(keep)
    print(f"Rows with all 3 sims + mask: {len(df_valid)}")

    # Translated comment
    global_cnt = get_start_cnt(BASE_DIR)
    print(f"Start sample_id from {global_cnt + 1}")

    all_meta = []
    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid)):
        all_meta.extend(process_single_row(row))

    manifest_path = os.path.join(BASE_DIR, "dataset_manifest_temporal.csv")
    pd.DataFrame(all_meta).to_csv(manifest_path, index=False)
    print(f"Done. Total cropped samples: {len(all_meta)}")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
