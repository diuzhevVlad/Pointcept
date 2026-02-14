import torch
import torch.nn as nn
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict
import os

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from pointcept.utils import comm
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self._nan_hook_fired = False
        self._enable_nan_hooks = os.getenv("POINTCEPT_NAN_DEBUG", "1") == "1"
        if self._enable_nan_hooks:
            self._register_nan_hooks()

    @staticmethod
    def _is_finite_tensor(tensor):
        return torch.is_tensor(tensor) and torch.isfinite(tensor).all().item()

    @staticmethod
    def _collect_tensors(obj, prefix=""):
        tensors = []
        if torch.is_tensor(obj):
            tensors.append((prefix, obj))
        elif isinstance(obj, Point):
            for key in obj.keys():
                value = obj.get(key)
                tensors.extend(DefaultSegmentorV2._collect_tensors(value, f"{prefix}.{key}"))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                tensors.extend(DefaultSegmentorV2._collect_tensors(value, f"{prefix}.{key}"))
        elif isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                tensors.extend(DefaultSegmentorV2._collect_tensors(value, f"{prefix}[{idx}]"))
        return tensors

    @staticmethod
    def _log_tensor_stats(name, tensor):
        if not torch.is_tensor(tensor):
            print(f"{name}: not a tensor ({type(tensor)})")
            return
        finite = torch.isfinite(tensor).all().item()
        t_min = tensor.min().item() if finite else "non-finite"
        t_max = tensor.max().item() if finite else "non-finite"
        print(f"{name} finite: {finite}, min/max: {t_min}/{t_max}, shape: {tuple(tensor.shape)}")

    def _log_nan_context(self, input_dict, seg_logits, loss, feat=None):
        if not comm.is_main_process():
            return
        print("NaN/Inf detected in loss. Dumping debug context...")
        if torch.is_tensor(loss):
            print(f"loss finite: {torch.isfinite(loss).all().item()}, value: {loss}")
        else:
            print(f"loss type: {type(loss)}")

        self._log_tensor_stats("seg_logits", seg_logits)
        if feat is not None:
            self._log_tensor_stats("feat", feat)

        # Check input tensors for non-finite values
        for key in ("coord", "strength", "segment"):
            value = input_dict.get(key, None)
            if torch.is_tensor(value):
                self._log_tensor_stats(f"input {key}", value)

        # Validate segment label range
        segment = input_dict.get("segment", None)
        if torch.is_tensor(segment) and torch.is_tensor(seg_logits):
            num_classes = seg_logits.shape[-1]
            ignore_index = -1
            if hasattr(self.criteria, "criteria") and len(self.criteria.criteria) > 0:
                ignore_index = getattr(self.criteria.criteria[0], "ignore_index", -1)
            invalid_mask = ~(
                (segment == ignore_index)
                | ((segment >= 0) & (segment < num_classes))
            )
            invalid_count = invalid_mask.sum().item()
            if invalid_count > 0:
                print(
                    f"invalid labels: {invalid_count} out of {segment.numel()}, "
                    f"ignore_index={ignore_index}, num_classes={num_classes}"
                )

        # Compute per-criterion losses for isolation
        if hasattr(self.criteria, "criteria") and torch.is_tensor(seg_logits):
            for idx, crit in enumerate(self.criteria.criteria):
                try:
                    val = crit(seg_logits, input_dict["segment"])
                    finite = torch.isfinite(val).all().item()
                    print(f"criterion[{idx}] {crit.__class__.__name__}: finite={finite}, value={val}")
                except Exception as exc:
                    print(f"criterion[{idx}] {crit.__class__.__name__} raised: {exc}")

    def _register_nan_hooks(self):
        def hook_fn(module, inputs, outputs):
            if self._nan_hook_fired or not comm.is_main_process():
                return
            input_tensors = self._collect_tensors(inputs, "input")
            output_tensors = self._collect_tensors(outputs, "output")
            for name, tensor in input_tensors + output_tensors:
                if torch.is_tensor(tensor) and not torch.isfinite(tensor).all().item():
                    self._nan_hook_fired = True
                    print(f"NaN/Inf detected in module: {module.__class__.__name__}")
                    self._log_tensor_stats(name, tensor)
                    # Extra detail for Linear layers to isolate source
                    if isinstance(module, nn.Linear):
                        if len(input_tensors) > 0:
                            self._log_tensor_stats("linear.input", input_tensors[0][1])
                        if torch.is_tensor(module.weight):
                            self._log_tensor_stats("linear.weight", module.weight)
                        if module.bias is not None:
                            self._log_tensor_stats("linear.bias", module.bias)
                    # Log only; allow training loop to handle non-finite loss.
                    return

        for name, module in self.named_modules():
            if module is self:
                continue
            module.register_forward_hook(hook_fn)

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            if torch.is_tensor(loss) and not torch.isfinite(loss).all().item():
                self._log_nan_context(input_dict, seg_logits, loss, feat=feat)
            return_dict["loss"] = loss
            # Expose detached logits for training metrics/logging hooks.
            return_dict["seg_logits"] = seg_logits.detach()
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            if torch.is_tensor(loss) and not torch.isfinite(loss).all().item():
                self._log_nan_context(input_dict, seg_logits, loss, feat=feat)
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
