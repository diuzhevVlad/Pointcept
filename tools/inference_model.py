data_config = dict(
    type="CustomDataset",
    data_root="data/custom",
    transform=[
        dict(
            type="GridSample",
            grid_size=0.05,
            hash_type="fnv",
            mode="spec_test",
            keys=("coord", "strength", "segment"),
            return_grid_coord=True,
        ),
        # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            feat_keys=("coord", "strength"),
        ),
    ],
    test_mode=False,
)

data_config = dict(
    type="SemanticKITTIDataset",
    split="val",
    data_root="data/semantic_kitti",
    transform=[
        dict(
            type="GridSample",
            grid_size=0.05,
            hash_type="fnv",
            mode="spec_test",
            keys=("coord", "strength", "segment"),
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
    ignore_index=-1,
)

model_config = dict(
    type="DefaultSegmentorV2",
    num_classes=19,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
)

checkpoint_path = "exp/semantic_kitti/semseg-pt-v3m1-0/model/model_best.pth"
from pointcept.models import build_model
from pointcept.datasets import build_dataset
import torch
from collections import OrderedDict
import numpy as np
import tqdm


if __name__ == "__main__":
    model = build_model(model_config).cuda()
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage.cuda()
    )

    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        weight[key.replace("module.", "")] = value

    load_state_info = model.load_state_dict(weight, strict=True)

    dataset = build_dataset(data_config)

    with torch.no_grad():
        ds = tqdm.tqdm(range(len(dataset)))
        for i in ds:
            data = dataset[i]
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(non_blocking=True)

            output = model(data)["loss"]
            probs = torch.softmax(output, dim=1)
            labels = torch.argmax(probs, dim=1).cpu().numpy()
            labels = np.vectorize(dataset.learning_map_inv.__getitem__)(
                labels & 0xFFFF
            ).astype(np.int32)
            ds.set_description(str(np.unique(labels)))
            # print(labels.shape)
            labels.astype(np.int32).tofile(
                "data/labels/{}.label".format(str(i).zfill(6))
            )
            # data["feat"].cpu().numpy().reshape(-1).tofile(
            #     "data/velodyne/{}.bin".format(str(i).zfill(6))
            # )

            del data, output, probs, labels
            torch.cuda.empty_cache()
