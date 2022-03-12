_base_ = './yolov3_d53_mstrain-608_273e_coco.py'
# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0,0], std=[255., 255., 255., 255.], to_rgb=True)
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet_rgb_depth_a',
        depth=53,
        out_indices=(3, 4, 5),
        pretrained=None,
        init_cfg=None),
    bbox_head=dict(
        num_classes=1))
train_pipeline = [
    dict(type='LoadImageFromFile_rgb_depth'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (416, 416)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile_rgb_depth'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../data/ht_cumt_rgbd/'
classes=('person',)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2014.json',
        img_prefix=data_root + 'train2014/',
        img_prefix_depth=data_root + 'depth_train/',
        pipeline=train_pipeline,
    classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        img_prefix_depth=data_root + 'depth_val/',
        pipeline=test_pipeline,
    classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        img_prefix_depth=data_root + 'depth_val/',
        pipeline=test_pipeline,
    classes=classes))
# optimizer
optimizer = dict(type='SGD', lr=0.000125, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))