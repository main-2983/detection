_base_ = [
    '../configs/_base_/default_runtime.py',
    '../configs/_base_/schedules/schedule_1x.py'
]

img_scale = (640, 640)  # height, width

# dataset settings
dataset_type = 'CocoDataset'
classes = [
    "Beetroot", "Avocado", "Kiwi", "Peach", "Mandarine", "Orange", "Ginger",
    "Banana", "Kumquats", "Onion", "Cactus", "Plum", "Kaki", "Tomato", "Pineapple",
    "Cauliflower", "Pepper", "Melon", "Nectarine", "Papaya", "Pear", "Redcurrant",
    "Redcurrant", "Apple", "Huckleberry", "Guava", "Limes", "Granadilla", "Lemon",
    "Mango", "Strawberry", "Physalis", "Quince", "Kohlrabi", "Pepino", "Rambutan",
    "Salak", "Eggplant", "Maracuja", "Nut", "Walnut", "Grapefruit", "Mangostan",
    "Pomegranate", "Hazelnut", "Mulberry", "Tamarillo", "Tangelo", "Cantaloupe",
    "Potato", "Chestnut", "Cherry", "Clementine", "Lychee", "Apricot", "Dates",
    "Cocos", "Pomelo", "Grape", "Passion", "Carambula", "Blueberry", "Pitahaya", "Raspberry"
]
data_root = '../Dataset/FruitCOCO/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', pad_to_square=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/_annotations.coco.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid/_annotations.coco.json',
        img_prefix=data_root + 'valid/',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=1, metric='bbox')

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=64, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])