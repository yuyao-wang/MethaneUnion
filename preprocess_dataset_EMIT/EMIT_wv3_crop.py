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

# 1) 输入 CSV：包含 plume_id（此脚本会自动构造 simulated_512_path）
INPUT_CSV = "./merged_with_emit_tag.csv"

# 2) EMIT WV3 模拟影像与掩膜
MASK_ROOT = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_masks_wv3_512"
# 掩膜文件名：每个 plume_id 目录下的 512x512 掩膜
MASK_NAME = "mask_60m_512.tif"

# 原始 EMIT WV3 60m, 512x512 模拟影像所在目录（由 EMIT_wv3*.py 生成）
SIM_OUTPUT_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/EMIT_simulated_WV3_L2A_60resolution_NOnorm"

# 3) 输出目录（新建）
BASE_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224"
os.makedirs(BASE_DIR, exist_ok=True)

# --- 裁剪参数 ---
CHIP_SIZE_PX = 512   # 对齐 EMIT_wv3.py
SCALE_M = 60         # 对齐 EMIT_wv3.py（这里只是记录物理尺寸用）

PATCH_SIZE = 16      # 和 L89 temporal 一样的 OS_SIZE
TARGET_SIZE = 224    # 统一输入尺寸
CENTER_BOX = 6       # 中心抖动范围（像素）
MISSING_THRESH = 0.25
N_POS = 16           # 每个 plume 的正样本数
N_NEG = 16           # 每个 plume 的负样本数
NUM_WORKERS = 18

counter_lock = threading.Lock()
global_cnt = 0  # 全局样本 id 计数

# =========================
# Helpers
# =========================

def ensure_chw(img: np.ndarray) -> np.ndarray:
    """保证输出为 (C, H, W)"""
    if img.ndim == 2:
        return img[np.newaxis, ...]
    if img.ndim == 3 and img.shape[0] > 100:  # HWC -> CHW
        return img.transpose(2, 0, 1)
    return img

def gpu_upsample(crop_np: np.ndarray) -> np.ndarray:
    """16x16 -> 224x224 双线性插值"""
    with torch.no_grad():
        img_t = torch.from_numpy(crop_np.astype(np.float32)).to(DEVICE).unsqueeze(0)
        img_t = torch.nan_to_num(img_t, nan=0.0)
        up = F.interpolate(img_t, size=(TARGET_SIZE, TARGET_SIZE),
                           mode="bilinear", align_corners=False)
        return up.squeeze(0).cpu().numpy()

def upsample_mask(mask_16: np.ndarray) -> np.ndarray:
    """最近邻插值 mask 到 224x224"""
    f = TARGET_SIZE // PATCH_SIZE  # 224/16=14
    return np.repeat(np.repeat(mask_16, f, axis=0), f, axis=1).astype("uint8")

def is_valid_crop(crop: np.ndarray) -> bool:
    """检查单个时相切片是否有效：0 或 NaN 比例不能太高"""
    if crop.size == 0:
        return False
    band0 = crop[0]
    missing = (np.isnan(band0) | (band0 == 0))
    return missing.mean() <= MISSING_THRESH

def get_three_paths(row) -> tuple[str, str, str] | tuple[None, None, None]:
    """从 simulated_512_path 推导 t0/-90/-180 三个路径"""
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
    """已有样本目录下继续编号（可选）"""
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

    # --- POSITIVE SAMPLES（中心附近抖动） ---
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

    # --- NEGATIVE SAMPLES（避开中心） ---
    neg_count, attempts = 0, 0
    while neg_count < N_NEG and attempts < 200:
        attempts += 1
        x = random.randint(0, CHIP_SIZE_PX - PATCH_SIZE)
        y = random.randint(0, CHIP_SIZE_PX - PATCH_SIZE)

        # 避免覆盖中心区域
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

    # 如果 CSV 中没有 simulated_512_path 列，则根据 plume_id 和固定目录自动构造
    if "simulated_512_path" not in df.columns:
        df["simulated_512_path"] = df["plume_id"].astype(str).apply(
            lambda pid: os.path.join(SIM_OUTPUT_DIR, f"{pid}_sim_WV3.tif")
        )

    # 只保留三张 sim 图都存在的样本
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

    # 从已有输出继续编号（可选）
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
