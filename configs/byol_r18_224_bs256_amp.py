# configs/byol_r18_224_bs256_amp.py
default_scope = 'mmselfsup'

# ===== Model =====
model = dict(
    type='BYOL',
    backbone=dict(
        type='ResNet',
        depth=18,                 # ← change to 34 for ResNet-34
        in_channels=12,
        out_indices=[4],
        norm_eval=True),
    neck=dict(                    # projector
        type='NonLinearNeck',
        in_channels=512,          # last feature channels of ResNet-18/34
        hid_channels=4096,
        out_channels=256,
        with_avg_pool=True,
        with_last_bn=True,
        with_last_bn_affine=False),
    head=dict(                    # predictor
        type='BYOLHead',
        in_channels=256,
        pred_hidden_channels=4096,
        pred_out_channels=256),
    target_ema=0.996,             # cosine-scheduled toward ~1.0 during training
)

# ===== Data augmentation (2 views) =====
# Note: example shown in mmcv-style; you may also directly use the official BYOL pipeline cfg
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', degree=90, prob=0.5),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=1.0),  # strong for view A
    dict(type='RandomGrayscale', prob=0.2),
    dict(type='PackSelfSupInputs')
]
# For view B, reduce blur intensity and optionally add solarize; see mmselfsup BYOL template

train_dataloader = dict(
    batch_size=128,                  # per GPU
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CustomDataset',        # or folder-based image dataset
        data_root='/data/s2_rgb/train',
        pipeline=train_pipeline)
)

# ===== Optimizer / Training strategy =====
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='LARS',
        lr=0.2,                      # already scaled assuming effective batch=256
        weight_decay=1.5e-6,
        momentum=0.9),
    accumulative_counts=2            # gradient accumulation ×2 → effective batch 256
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=10),
    dict(type='CosineAnnealingLR', T_max=190, by_epoch=True, begin=10, end=200)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50)
)
env_cfg = dict(cudnn_benchmark=True)

# ===== Runner (single GPU; specify via command-line later) =====
