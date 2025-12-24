"""
Point Transformer - V3 Mode1 with image fusion

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Modified by: Codex
"""

import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point
from .point_transformer_v3m1_base import PointTransformerV3


class SmallImageEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, images):
        return self.pool(self.features(images))


@MODELS.register_module("PT-v3m1-img")
class PointTransformerV3Image(PointTransformerV3):
    def __init__(
        self,
        img_in_channels=3,
        img_feat_channels=32,
        img_patch_size=14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_feat_channels = img_feat_channels
        self.img_patch_size = img_patch_size
        self.img_encoder = SmallImageEncoder(
            img_in_channels, img_feat_channels, img_patch_size
        )

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)

        if self._has_image_inputs(data_dict):
            img_feat = self.img_encoder(data_dict["images"].to(point.feat.device))
            point_img_feat = self._gather_image_features(
                img_feat,
                data_dict["correspondence"].to(point.feat.device),
                data_dict["img_num"].to(point.feat.device),
                point.offset,
            )
            point.feat = torch.cat([point.feat, point_img_feat], dim=1)
        return point

    @staticmethod
    def _has_image_inputs(data_dict):
        return all(
            key in data_dict for key in ("images", "correspondence", "img_num")
        )

    def _gather_image_features(self, image_feat, correspondence, img_num, offset):
        num_points = correspondence.shape[0]
        if image_feat.shape[0] == 0 or num_points == 0:
            return image_feat.new_zeros((num_points, self.img_feat_channels))

        batch = offset2batch(offset)
        img_num = img_num.view(-1).to(offset.device)
        img_offsets = torch.cumsum(img_num, dim=0) - img_num
        img_offsets = img_offsets.to(correspondence.device)

        valid_mask = (correspondence[..., 0] >= 0) & (correspondence[..., 1] >= 0)
        if not valid_mask.any():
            return image_feat.new_zeros((num_points, self.img_feat_channels))

        point_idx, view_idx = torch.where(valid_mask)
        global_img = img_offsets[batch[point_idx]] + view_idx
        max_img = image_feat.shape[0]
        valid_img = global_img < max_img
        if not valid_img.all():
            point_idx = point_idx[valid_img]
            view_idx = view_idx[valid_img]
            global_img = global_img[valid_img]

        coords = correspondence[point_idx, view_idx]
        h, w = image_feat.shape[2], image_feat.shape[3]
        y = coords[:, 0].clamp(0, h - 1).long()
        x = coords[:, 1].clamp(0, w - 1).long()

        gathered = image_feat[global_img, :, y, x]
        return torch_scatter.scatter_mean(
            gathered, point_idx, dim=0, dim_size=num_points
        )
