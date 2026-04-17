
# python preprocess_dataset_query_multi/prepare_raw512_masks.py \
#   --master_csv preprocess_dataset_multisensor/master_multisensor_outer_join.csv \
#   --out_root /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512 \
#   --out_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512.csv \
#   --qa_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/prepare_raw512_masks_log.csv \
#   --workers 8 \
#   --overwrite

# python preprocess_dataset_query_multi/qa_raw512_manifest.py \
#   --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512.csv \
#   --clean_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512_clean.csv \
#   --qa_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/qa_raw512.csv \
#   --workers 12

# python preprocess_dataset_query_multi/visualize_raw512_samples.py \
#   --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512_clean.csv \
#   --out_dir /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/quicklooks \
#   --num_samples 8

# python preprocess_dataset_query_multi/qa_raw512_manifest.py \
#   --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512.csv \
#   --clean_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512_clean.csv \
#   --qa_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/qa_raw512.csv \
#   --workers 12

# python preprocess_dataset_query_multi/make_query_crops.py \
#   --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512_clean.csv \
#   --out_root /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/query_480m/crops \
#   --out_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/query_480m/manifest_query_480m.csv \
#   --query_size_m 480 \
#   --n_pos 16 \
#   --n_neg 16 \
#   --max_attempts_pos 200 \
#   --max_attempts_neg 800 \
#   --save_every 200 \
#   --workers 8 \
#   --target_size 512 \
#   --center_box_m 500 \
#   --resume \
#   --local_stage_root /diniuvol/yuyao/make_query_crops_stage \
#   --stage_max_gb 100 \
#   --stage_wait_sec 2

#[config] {'target_size': 224, 'n_pos': 16, 'n_neg': 16, 'center_box_px': 10, 'patch_size': {'s2': 36, 'l89': 12, 'emit': 6, 's5p': 3}, 's5p_stack_output': True}
python preprocess_dataset_query_multi/crop_legacy_param.py \
  --master_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512/manifest_raw512_clean.csv \
  --out_root /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_120m/crops \
  --out_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_120m/manifest.csv \
  --query_size_m 120 \
  --target_size 224 \
  --n_pos 16 \
  --n_neg 16 \
  --center_box_px 10 \
  --s5p_stack_output \
  --workers 8 \
  --debug \
  --debug_every 20 \
  --resume

# python preprocess_dataset_query_multi/visualize_query_crops.py \
#   --manifest_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/query_120m/manifest_query_120m_smoke.csv \
#   --out_dir /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/query_120m/quicklooks_smoke \
#   --num_samples 24
