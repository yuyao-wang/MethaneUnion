# (-7, -90, -360) negative samples
import os
import pandas as pd
import numpy as np
import tifffile
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

# =========================
# 1. Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# output directory
BASE_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_-790360_16_resized_to_224_CRSfixed"
INPUT_CSV = "./CM_L89_L2SR_std512.csv"

OS_SIZE = 16          # original patch size (16x16)
TARGET_SIZE = 224     # training input size (224x224)
CENTER_BOX = 6        # center jitter range
MISSING_THRESH = 0.20 # missing-pixel ratio threshold
N_PAIRS = 16          # number of sample pairs generated per plume ID (16Pos, 16Neg)
NUM_WORKERS = 18       # number of worker threads

os.makedirs(BASE_DIR, exist_ok=True)
counter_lock = threading.Lock()
global_cnt = 0

# =========================
# 2. Helpers
# =========================

def ensure_chw(img):
    """Ensure the image layout is (Channels, Height, Width)"""
    if img.ndim == 2: return img[np.newaxis, ...]
    if img.shape[0] > 100: return img.transpose(2, 0, 1)
    return img

def gpu_upsample(crop_np):
    """Use the GPU to upsample 16x16 crops to 224x224"""
    with torch.no_grad():
        img_t = torch.from_numpy(crop_np.astype(np.float32)).to(DEVICE).unsqueeze(0)
        img_t = torch.nan_to_num(img_t, nan=0.0)
        upsampled = F.interpolate(img_t, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear', align_corners=False)
        return upsampled.squeeze(0).cpu().numpy()

def is_valid_crop(crops_list):
    """Check whether all temporal crops satisfy the missing-pixel ratio requirement"""
    for crop in crops_list:
        if crop.size == 0: return False
        # Compute the ratio of NaN or zero pixels
        missing_ratio = (np.isnan(crop[0]) | (crop[0] == 0)).mean()
        if missing_ratio > MISSING_THRESH:
            return False
    return True

# =========================
# 3. Core Logic
# =========================

def save_sample(c_list, m_crop, label, plume_id, tag=""):
    """Save a sample crop to disk and return its metadata"""
    global global_cnt
    # c_list expected order: [Target_T, Pre1_T, Pre2_T]
    up0 = gpu_upsample(c_list[0])
    up1 = gpu_upsample(c_list[1])
    up2 = gpu_upsample(c_list[2])
    
    # Translated comment
    m_up = F.interpolate(torch.from_numpy(m_crop.astype(np.float32)).unsqueeze(0).unsqueeze(0), 
                         size=(TARGET_SIZE, TARGET_SIZE), mode='nearest').numpy()[0,0].astype('uint8')
    
    with counter_lock:
        global_cnt += 1
        this_id = global_cnt

    sample_dir = os.path.join(BASE_DIR, str(this_id))
    os.makedirs(sample_dir, exist_ok=True)
    
    # write files to disk
    tifffile.imwrite(os.path.join(sample_dir, "target.tif"), up0)
    tifffile.imwrite(os.path.join(sample_dir, "pre1.tif"), up1)
    tifffile.imwrite(os.path.join(sample_dir, "pre2.tif"), up2)
    tifffile.imwrite(os.path.join(sample_dir, "plume.tif"), m_up)

    return {
        "id": this_id, 
        "label": label, 
        "plume_id": plume_id, 
        "path": sample_dir,
        "neg_type": tag  # record whether the negative sample came from the same location or an edge fallback
    }

def process_single_row(row):
    """Process one row from the CSV"""
    local_results = []
    
    try:
        # Load four 512x512 source images and the mask
        t0 = ensure_chw(tifffile.imread(row["l89_path"]))           # T0
        t7 = ensure_chw(tifffile.imread(row["l89_-7_path"]))        # T-7
        t90 = ensure_chw(tifffile.imread(row["l89_pre_path"]))      # T-90
        t360 = ensure_chw(tifffile.imread(row["l89_pre_pre_path"])) # T-360
        full_mask = tifffile.imread(row["mask_path"])
    except Exception as e:
        return []

    # --- Step 1: sample positive crops (must stay near the center and include the plume) ---
    pos_samples = []
    attempts = 0
    while len(pos_samples) < N_PAIRS and attempts < 150:
        attempts += 1
        # Apply a small jitter inside the center region
        p_min = 256 - (CENTER_BOX // 2) - OS_SIZE // 2
        p_max = 256 + (CENTER_BOX // 2) - OS_SIZE // 2
        x, y = random.randint(p_min, p_max), random.randint(p_min, p_max)
        
        c0 = t0[:, y:y+OS_SIZE, x:x+OS_SIZE]
        c90 = t90[:, y:y+OS_SIZE, x:x+OS_SIZE]
        c360 = t360[:, y:y+OS_SIZE, x:x+OS_SIZE]
        m_crop = full_mask[y:y+OS_SIZE, x:x+OS_SIZE]

        if is_valid_crop([c0, c90, c360]):
            res = save_sample([c0, c90, c360], m_crop, 1, row["plume_id"], tag="positive")
            res['x'], res['y'] = x, y  # record the location for negative-sample alignment
            pos_samples.append(res)

    # --- Step 2: generate one matching negative sample for each positive sample (Label 0) ---
    for ps in pos_samples:
        x, y = ps['x'], ps['y']
        
        # First try sampling T-7 at the same coordinates
        c7_same = t7[:, y:y+OS_SIZE, x:x+OS_SIZE]
        c90_same = t90[:, y:y+OS_SIZE, x:x+OS_SIZE]
        c360_same = t360[:, y:y+OS_SIZE, x:x+OS_SIZE]
        
        if is_valid_crop([c7_same, c90_same, c360_same]):
            # Case 1: same-location sampling succeeds
            ns = save_sample([c7_same, c90_same, c360_same], np.zeros((OS_SIZE, OS_SIZE)), 0, row["plume_id"], tag="same_loc")
            local_results.append(ps)
            local_results.append(ns)
        else:
            # Case 2: T-7 is too incomplete at the same location, so fall back to an edge crop
            found_edge = False
            for _ in range(30):
                ex, ey = random.randint(0, 512-OS_SIZE), random.randint(0, 512-OS_SIZE)
                # Avoid the central plume area
                if abs(ex-256) < 60 and abs(ey-256) < 60: continue
                
                ec7 = t7[:, ey:ey+OS_SIZE, ex:ex+OS_SIZE]
                ec90 = t90[:, ey:ey+OS_SIZE, ex:ex+OS_SIZE]
                ec360 = t360[:, ey:ey+OS_SIZE, ex:ex+OS_SIZE]
                
                if is_valid_crop([ec7, ec90, ec360]):
                    ns = save_sample([ec7, ec90, ec360], np.zeros((OS_SIZE, OS_SIZE)), 0, row["plume_id"], tag="edge_escape")
                    local_results.append(ps)
                    local_results.append(ns)
                    found_edge = True
                    break
            # Translated comment
    
    return local_results

# =========================
# 4. Main Execution
# =========================

def process_all():
    global global_cnt
    df = pd.read_csv(INPUT_CSV).dropna(subset=['mask_path', 'l89_path', 'l89_-7_path'])
    manifest_path = os.path.join(BASE_DIR, "dataset_manifest_temporal.csv")
    
    all_data = []
    if os.path.exists(manifest_path):
        old_df = pd.read_csv(manifest_path)
        processed_pids = set(old_df['plume_id'].unique())
        df = df[~df['plume_id'].isin(processed_pids)]
        global_cnt = old_df['id'].max() if len(old_df) > 0 else 0
        all_data = old_df.to_dict('records')
        print(f"Resuming... {len(processed_pids)} plumes already done. Max ID: {global_cnt}")

    print(f"Processing {len(df)} new plumes...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        rows_list = [row for _, row in df.iterrows()]
        pbar = tqdm(total=len(rows_list))
        
        for res_list in executor.map(process_single_row, rows_list):
            if res_list:
                all_data.extend(res_list)
            pbar.update(1)
            
            # Save progress every 50 plumes
            if pbar.n % 50 == 0:
                pd.DataFrame(all_data).to_csv(manifest_path, index=False)
    
    pd.DataFrame(all_data).to_csv(manifest_path, index=False)
    print(f"All done. Total samples: {len(all_data)}")

if __name__ == "__main__":
    process_all()