_base_ = './configs/htc/htc_r50_fpn_1x_coco.py'

# learning policy
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[18, 36])
total_epochs = 48

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')),
    roi_head=dict(
        _delete_=True,
        type='PointRendRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='concat',
            roi_layer=dict(
                type='SimpleRoIAlign', output_size=14),
            out_channels=256,
            featmap_strides=[4]),
        mask_head=dict(
            type='CoarseMaskHead',
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=20,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        point_head=dict(
            type='MaskPointHead',
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=20,
            coarse_pred_each_layer=True,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))

# model training and testing settings
train_cfg = dict(
    _delete_=True,
    rcnn=dict(
        mask_size=7,
        num_points=14 * 14,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False),
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    
    )



test_cfg = dict(
    rcnn=
      dict(
        subdivision_steps=5, subdivision_num_points=28 * 28, scale_factor=2))

dataset_type = 'CocoDataset'
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='AutoAugment',
#         policies=[
#                   [
#                       dict(
#                           type='Shear',
#                           level=5
#                       ),
#                       dict(
#                           type='Rotate',
#                           level=5
#                       ),
#                       dict(
#                           type='Translate',
#                           level=5
#                       ),
#                       dict(
#                           type='RandomCrop',
#                           crop_type='relative_range',
#                           crop_size=(0.9, 0.9),
#                           allow_negative_crop=True),
#                       dict(
#                           type='BrightnessTransform',
#                           level=3
#                       ),
#                       dict(
#                           type='ContrastTransform',
#                           level=3
#                       )
#                   ]]),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(
#         type='Collect',
#         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]


# 800, 1000, 1200, 1400, 1600
# 1333, 1666, 2000, 2333, 2666
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=[(1333, 800), (1666, 1000), (2000, 1200),
#         (2333, 1400), (2666, 1600)],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip', flip_ratio=0.5),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

root = './'
data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./pascal_train.json',
        img_prefix='./train_images/',
        #seg_prefix='',
        pipeline=train_pipeline),
    #),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./pascal_train.json',
        img_prefix='./train_images/',
        #seg_prefix='',
        pipeline=test_pipeline),
    #),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='./test_images/',
        ann_file='./test.json',
        #seg_prefix='',
        pipeline=test_pipeline))