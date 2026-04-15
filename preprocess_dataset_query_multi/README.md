# Query multisensor preprocessing

This folder prepares the raw 512x512 layer used before query crop generation.

The intended order is:

1. Generate sensor-aligned binary 512 masks and a raw manifest:

```bash
python3 preprocess_dataset_query_multi/prepare_raw512_masks.py \
  --master_csv preprocess_dataset_multisensor/master_multisensor_outer_join.csv \
  --out_root /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512 \
  --out_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512.csv
```

2. Run QA and generate a clean manifest:

```bash
python3 preprocess_dataset_query_multi/qa_raw512_manifest.py \
  --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512.csv \
  --clean_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512_clean.csv \
  --qa_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/qa_raw512.csv
```

3. Visualize a few complete four-sensor samples:

```bash
python3 preprocess_dataset_query_multi/visualize_raw512_samples.py \
  --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512_clean.csv \
  --out_dir /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/quicklooks \
  --num_samples 8
```

The scripts avoid pandas/matplotlib so they can run in the current lightweight environment.

4. Generate query crops. Example for 120m:

```bash
python3 preprocess_dataset_query_multi/make_query_crops.py \
  --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512_clean.csv \
  --out_root /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/query_120m/crops \
  --out_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/query_120m/manifest_query_120m.csv \
  --query_size_m 120 \
  --n_pos 16 \
  --n_neg 16 \
  --workers 4
```

5. Visualize query crops:

```bash
python3 preprocess_dataset_query_multi/visualize_query_crops.py \
  --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/query_120m/manifest_query_120m.csv \
  --out_dir /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/query_120m/quicklooks \
  --num_samples 24
```
