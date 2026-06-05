import os
import pandas as pd
import numpy as np
import tifffile
import random

# Translated comment
IMG_SIZE = 512  # Translated comment
CENTER_SIZE = 30  # Translated comment
CROP_SIZE = 96  # Translated comment
INNER_MARGIN = 128  # Translated comment

base_dir = '/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/train_tryout_32/carbonmapper_data_temporal_split_classification'
train_csv_path = os.path.join(base_dir, 'train.csv')
test_csv_path = os.path.join(base_dir, 'test.csv')
origin_train_csv = '/data2/yuyao/methane_emission/data_csv/train.csv'
origin_test_csv = '/data2/yuyao/methane_emission/data_csv/test.csv'


def get_crop(width=IMG_SIZE, height=IMG_SIZE,
             center_size=CENTER_SIZE,
             crop_width=CROP_SIZE, crop_height=CROP_SIZE):
    """
 (top, left), crop_width x crop_height  center center_size x center_size .
    """
    center_x = width // 2
    center_y = height // 2
    center_left = center_x - center_size // 2
    center_right = center_x + center_size // 2
    center_top = center_y - center_size // 2
    center_bottom = center_y + center_size // 2

    # Translated comment
    min_left = max(0, center_right - crop_width)
    max_left = min(center_left, width - crop_width)
    min_top = max(0, center_bottom - crop_height)
    max_top = min(center_top, height - crop_height)

    # Translated comment
    left = random.randint(min_left, max_left)
    top = random.randint(min_top, max_top)
    return top, left  # Translated comment


def random_inner_crop():
    """
 image, CROP_SIZE x CROP_SIZE figure,  edge INNER_MARGIN .
    """
    max_row = IMG_SIZE - CROP_SIZE - INNER_MARGIN
    max_col = IMG_SIZE - CROP_SIZE - INNER_MARGIN

    # Translated comment
    if max_row < INNER_MARGIN or max_col < INNER_MARGIN:
        row = random.randint(0, IMG_SIZE - CROP_SIZE)
        col = random.randint(0, IMG_SIZE - CROP_SIZE)
    else:
        row = random.randint(INNER_MARGIN, max_row)
        col = random.randint(INNER_MARGIN, max_col)
    return row, col


# ================= train =================
data = []
cnt = 0
org_train_df = pd.read_csv(origin_train_csv)

for index, row in org_train_df.iterrows():
    t_data = tifffile.imread(row['s2_path'])
    if t_data.ndim == 3 and t_data.shape[-1] == 12:
        t_data = np.transpose(t_data, (2, 0, 1))

    mask = tifffile.imread(row['plume_mask_path'])

    crop_list = []
    # Translated comment
    for i in range(8):
        crop_list.append(get_crop())

    # Translated comment
    for i in range(16):
        crop_list.append(random_inner_crop())

    for crop in crop_list:
        r, c = crop
        nt_data = t_data[:, r:r+CROP_SIZE, c:c+CROP_SIZE]
        n_mask = mask[r:r+CROP_SIZE, c:c+CROP_SIZE]

        dir_path = os.path.join(base_dir, str(cnt))
        os.makedirs(dir_path, exist_ok=True)

        nt_path = os.path.join(dir_path, "s2.tif")
        n_mask_path = os.path.join(dir_path, "plume.tif")
        tifffile.imwrite(nt_path, nt_data)
        tifffile.imwrite(n_mask_path, n_mask)

        mask_sum = np.sum(n_mask)
        data.append({
            "id": cnt,
            "s2_path": nt_path,
            "plume_mask_path": n_mask_path,
            "label": 0 if mask_sum == 0 else 1,
            "emission_auto": 0 if mask_sum == 0 else row['emission_auto'],
            "emission_uncertainty_auto": 0 if mask_sum == 0 else row['emission_uncertainty_auto']
        })
        cnt += 1

train_df = pd.DataFrame(data)
train_df.to_csv(train_csv_path, index=False)
print(f'training set count: {cnt}')


# ================= test =================
data_test = []
org_test_df = pd.read_csv(origin_test_csv)

for index, row in org_test_df.iterrows():
    t_data = tifffile.imread(row['s2_path'])
    if t_data.ndim == 3 and t_data.shape[-1] == 12:
        t_data = np.transpose(t_data, (2, 0, 1))

    mask = tifffile.imread(row['plume_mask_path'])

    crop_list = []
    for i in range(8):
        crop_list.append(get_crop())
    for i in range(16):
        crop_list.append(random_inner_crop())

    for crop in crop_list:
        r, c = crop
        nt_data = t_data[:, r:r+CROP_SIZE, c:c+CROP_SIZE]
        n_mask = mask[r:r+CROP_SIZE, c:c+CROP_SIZE]

        dir_path = os.path.join(base_dir, str(cnt))
        os.makedirs(dir_path, exist_ok=True)

        nt_path = os.path.join(dir_path, "s2.tif")
        n_mask_path = os.path.join(dir_path, "plume.tif")
        tifffile.imwrite(nt_path, nt_data)
        tifffile.imwrite(n_mask_path, n_mask)

        mask_sum = np.sum(n_mask)
        data_test.append({
            "id": cnt,
            "s2_path": nt_path,
            "plume_mask_path": n_mask_path,
            "label": 0 if mask_sum == 0 else 1,
            "emission_auto": 0 if mask_sum == 0 else row['emission_auto'],
            "emission_uncertainty_auto": 0 if mask_sum == 0 else row['emission_uncertainty_auto']
        })
        cnt += 1

test_df = pd.DataFrame(data_test)
test_df.to_csv(test_csv_path, index=False)
print(f'total count: {cnt}')

print(f"train label=1: {train_df['label'].sum()} / {len(train_df)}")
print(f"test  label=1: {test_df['label'].sum()} / {len(test_df)}")
