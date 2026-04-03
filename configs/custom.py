_base_ = '../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# ========================
# DATASET
# ========================
dataset_type = 'CocoDataset'
data_root = 'data/'

classes = ('drawing', 'table', 'note')

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False)
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes),
        test_mode=True
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/annotations/train.json',
    metric='bbox'
)

# ========================
# MODEL (CHỈ SỬA CLASS)
# ========================
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=3
        )
    )
)

# ========================
# TRAINING
# ========================
train_cfg = dict(
    max_epochs=30,   # 🔥 giảm xuống 30 là đủ
    val_interval=1
)

# ========================
# OPTIMIZER
# ========================
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.002,   # 🔥 giảm LR cho dataset nhỏ
        momentum=0.9,
        weight_decay=0.0001
    )
)

# ========================
# LR SCHEDULER
# ========================
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[20, 25],
        gamma=0.1
    )
]

# ========================
# AUGMENTATION (QUAN TRỌNG)
# ========================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(type='RandomFlip', prob=0.5),

    dict(
        type='RandomResize',
        scale=[(1333, 640), (1333, 800)],
        keep_ratio=True
    ),

    dict(type='PackDetInputs')
]

# ========================
# MISC
# ========================
default_scope = 'mmdet'
work_dir = './work_dir'