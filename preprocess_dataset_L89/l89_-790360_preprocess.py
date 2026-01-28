import os
import pandas as pd
import numpy as np
import tifffile
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import threading

# =========================
# Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_32_resized_to_224_updated"
INPUT_CSV = "./CM_L89_L2SR_std512.csv"

OS_SIZE = 32
TARGET_SIZE = 224
CENTER_BOX = 6
MISSING_THRESH = 0.25
N_POS, N_NEG = 16, 16
NUM_WORKERS = 8  # Increase this to speed up remote disk reading

os.makedirs(BASE_DIR, exist_ok=True)
counter_lock = threading.Lock()
global_cnt = 0
visual_done = 0

# =========================
# GPU & Format Helpers
# =========================

def ensure_chw(img):
    if img.ndim == 2: return img[np.newaxis, ...]
    if img.ndim == 3 and img.shape[0] > 100: return img.transpose(2, 0, 1)
    return img

def gpu_upsample(crop_np):
    """Safe GPU upsampling."""
    with torch.no_grad():
        img_t = torch.from_numpy(crop_np.astype(np.float32)).to(DEVICE).unsqueeze(0)
        img_t = torch.nan_to_num(img_t, nan=0.0)
        upsampled = F.interpolate(img_t, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear', align_corners=False)
        return upsampled.squeeze(0).cpu().numpy()

# =========================
# Worker Function (Modified)
# =========================

def process_single_row(row):
    global global_cnt, visual_done
    local_results = []
    
    try:
        # 读取原始时序数据
        t0 = ensure_chw(tifffile.imread(row["l89_0_std_512"]))
        t1 = ensure_chw(tifffile.imread(row["l89_-90_std_512"]))
        t2 = ensure_chw(tifffile.imread(row["l89_-360_std_512"]))
        full_mask = tifffile.imread(row["mask_path"])
    except Exception as e:
        return []

    # 辅助函数：检查切片是否有效（无缺失像素）
    def is_valid_crop(crop):
        if crop.size == 0: return False
        return (np.isnan(crop[0]) | (crop[0] == 0)).mean() <= MISSING_THRESH

    # --- POSITIVE SAMPLES ---
    p_start_min = 256 + (CENTER_BOX // 2) - OS_SIZE
    p_start_max = 256 - (CENTER_BOX // 2)
    pos_count, attempts = 0, 0
    
    while pos_count < N_POS and attempts < 100:
        attempts += 1
        x, y = random.randint(p_start_min, p_start_max), random.randint(p_start_min, p_start_max)
        c0, c1, c2 = t0[:, y:y+OS_SIZE, x:x+OS_SIZE], t1[:, y:y+OS_SIZE, x:x+OS_SIZE], t2[:, y:y+OS_SIZE, x:x+OS_SIZE]
        
        # 正样本必须保证三个时段都相对完整
        if not (is_valid_crop(c0) and is_valid_crop(c1) and is_valid_crop(c2)):
            continue
            
        m_crop = full_mask[y:y+OS_SIZE, x:x+OS_SIZE]
        up0, up1, up2 = gpu_upsample(c0), gpu_upsample(c1), gpu_upsample(c2)
        m_up = np.repeat(np.repeat(m_crop, 7, axis=0), 7, axis=1)

        with counter_lock:
            global_cnt += 1
            this_id = global_cnt

        out_path = os.path.join(BASE_DIR, str(this_id))
        os.makedirs(out_path, exist_ok=True)
        
        # 物理保存文件
        p0, p90, p360 = os.path.join(out_path, "l89_0.tif"), os.path.join(out_path, "l89_90.tif"), os.path.join(out_path, "l89_360.tif")
        tifffile.imwrite(p0, up0); tifffile.imwrite(p90, up1); tifffile.imwrite(p360, up2)
        tifffile.imwrite(os.path.join(out_path, "plume.tif"), m_up)

        local_results.append({
            "id": this_id, "label": 1, "plume_id": row["plume_id"], "path": out_path,
            "path_t0": p0, "path_t90": p90, "path_t360": p360,
            "latitude": row["plume_latitude"], "longitude": row["plume_longitude"]
        })
        pos_count += 1

    # --- NEGATIVE SAMPLES ---
    neg_count, attempts = 0, 0
    while neg_count < N_NEG and attempts < 200: # 负样本不合格时通过循环重新 Crop
        attempts += 1
        x, y = random.randint(0, 512-OS_SIZE), random.randint(0, 512-OS_SIZE)
        if (x <= 256 <= x+OS_SIZE) and (y <= 256 <= y+OS_SIZE): continue

        c0, c1, c2 = t0[:, y:y+OS_SIZE, x:x+OS_SIZE], t1[:, y:y+OS_SIZE, x:x+OS_SIZE], t2[:, y:y+OS_SIZE, x:x+OS_SIZE]
        
        # 关键改进：如果负样本时序不全，重新采样
        if not (is_valid_crop(c0) and is_valid_crop(c1) and is_valid_crop(c2)):
            continue

        up0, up1, up2 = gpu_upsample(c0), gpu_upsample(c1), gpu_upsample(c2)
        
        with counter_lock:
            global_cnt += 1
            this_id = global_cnt

        out_path = os.path.join(BASE_DIR, str(this_id))
        os.makedirs(out_path, exist_ok=True)
        
        p0, p90, p360 = os.path.join(out_path, "l89_0.tif"), os.path.join(out_path, "l89_90.tif"), os.path.join(out_path, "l89_360.tif")
        tifffile.imwrite(p0, up0); tifffile.imwrite(p90, up1); tifffile.imwrite(p360, up2)
        tifffile.imwrite(os.path.join(out_path, "plume.tif"), np.zeros((224, 224), dtype='uint8'))
        
        local_results.append({
            "id": this_id, "label": 0, "plume_id": row["plume_id"], "path": out_path,
            "path_t0": p0, "path_t90": p90, "path_t360": p360,
            "latitude": row["plume_latitude"], "longitude": row["plume_longitude"]
        })
        neg_count += 1

    return local_results

def save_visual_check(idx, t0, t0_up, mask, mask_up):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(t0[0], cmap='viridis'); axes[0].set_title("32x32 Raw")
    axes[1].imshow(t0_up[0], cmap='viridis'); axes[1].set_title("224x224 Up")
    axes[2].imshow(mask, cmap='gray'); axes[3].imshow(mask_up, cmap='gray')
    plt.savefig(os.path.join(BASE_DIR, f"check_{idx}.png"))
    plt.close()

# =========================
# Main execution
# =========================
def get_start_cnt(base_dir):
    """检查已经生成的文件夹，防止覆盖"""
    existing_ids = [int(d) for d in os.listdir(base_dir) if d.isdigit()]
    return max(existing_ids) if existing_ids else 0

# 在执行前初始化
os.makedirs(BASE_DIR, exist_ok=True)
global_cnt = get_start_cnt(BASE_DIR)
print(f"Current max ID in storage: {global_cnt}. New samples will start from {global_cnt + 1}")

counter_lock = threading.Lock()
visual_done = 0 # 视觉检查可以重新生成

# =========================
# 修改后的主执行函数
# =========================
def process_all():
    global global_cnt
    
    # 1. 读取输入并加载已有的 manifest (如果存在)
    df = pd.read_csv(INPUT_CSV).dropna(subset=['mask_path'])
    manifest_path = os.path.join(BASE_DIR, "dataset_manifest.csv")
    
    processed_plume_ids = set()
    all_data = []

    if os.path.exists(manifest_path):
        try:
            old_df = pd.read_csv(manifest_path)
            all_data = old_df.to_dict('records')
            processed_plume_ids = set(old_df['plume_id'].unique())
            print(f"Resuming: {len(processed_plume_ids)} plume_ids already processed.")
        except:
            print("Manifest exists but could not be read. Starting fresh.")

    # 2. 过滤掉已经处理过的 plume_id
    df_to_process = df[~df['plume_id'].isin(processed_plume_ids)]
    
    if len(df_to_process) == 0:
        print("All rows already processed!")
        return

    print(f"Remaining rows to process: {len(df_to_process)}")

    # 3. 并行处理
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        rows = [row for _, row in df_to_process.iterrows()]
        
        # 使用 as_completed 或简单的循环来实时获取结果
        pbar = tqdm(total=len(rows))
        for res_list in executor.map(process_single_row, rows):
            if res_list:
                all_data.extend(res_list)
            
            pbar.update(1)
            
            # 每处理 50 个原始行，强制存一次盘，防止崩溃
            if pbar.n % 50 == 0:
                pd.DataFrame(all_data).to_csv(manifest_path, index=False)
    
    # 最后存一次全量数据
    final_df = pd.DataFrame(all_data)
    final_df.to_csv(manifest_path, index=False)
    print(f"Finished. Total samples in manifest: {len(final_df)}")

if __name__ == "__main__":
    process_all()