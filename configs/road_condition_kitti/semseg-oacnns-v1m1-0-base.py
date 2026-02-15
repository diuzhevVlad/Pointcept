_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 24  # total bs across all gpus; tune based on memory
mix_prob = 0.8
empty_cache = False
enable_amp = True
sync_bn = True

# enable per-class TensorBoard logging
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=1),
    dict(type="PreciseEvaluator", test_last=False),
]

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="OACNNs",
        in_channels=4,  # coord(3) + strength(1)
        num_classes=8,
        embed_channels=64,
        enc_channels=[64, 64, 128, 256],
        groups=[4, 4, 8, 16],
        enc_depth=[3, 3, 9, 8],
        dec_channels=[256, 256, 256, 256],
        point_grid_size=[[8, 12, 16, 16], [6, 9, 12, 12], [4, 6, 8, 8], [3, 4, 6, 6]],
        dec_depth=[2, 2, 2, 2],
        enc_num_ref=[16, 16, 16, 16],
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "RoadConditionKITTIDataset"
data_root = "/data"
ignore_index = -1
names = [
    "nonroad",
    "dry",
    "wet",
    "snow",
    "pothole",
    "hill",
    "slush",
    "moisture",
]

data = dict(
    num_classes=8,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomVerticalCrop", min_height_threshold=-0.5, p=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 12, 1 / 12], axis="x", p=0.2),
            dict(type="RandomRotate", angle=[-1 / 12, 1 / 12], axis="y", p=0.2),
            dict(type="RandomRotate", angle=[-1 / 12, 1 / 12], axis="x", p=0.2),
            dict(type="RandomRotate", angle=[-1 / 12, 1 / 12], axis="y", p=0.2),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomShift", shift=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
            dict(type="RandomFlip", p=0.1),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)
