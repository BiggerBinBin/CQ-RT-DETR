"""Scale-aware RT-DETRv2 box criterion for CottonWeedDet12."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .rtdetrv2_criterion import RTDETRCriterionv2


@register()
class RTDETRCriterionv2ScaleAware(RTDETRCriterionv2):
    """RT-DETRv2 criterion with mild scale-aware box weighting.

    Only matched positive boxes are reweighted. Classification/VFL,
    Hungarian matching, model architecture and inference are unchanged.
    The mean matched-box weight is normalized to 1 for every loss call,
    keeping the global loss magnitude close to the official criterion.
    """

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        boxes_weight_format=None,
        share_matched_indices=False,
        area_ref=0.01,
        area_power=0.25,
        area_min_weight=0.75,
        area_max_weight=1.50,
        normalize_area_weight=True,
    ):
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
        )
        if area_ref <= 0:
            raise ValueError(f"area_ref must be positive, got {area_ref}")
        if area_power < 0:
            raise ValueError(f"area_power must be non-negative, got {area_power}")
        if area_min_weight <= 0 or area_max_weight < area_min_weight:
            raise ValueError(
                "Invalid area weight bounds: "
                f"min={area_min_weight}, max={area_max_weight}"
            )

        self.area_ref = float(area_ref)
        self.area_power = float(area_power)
        self.area_min_weight = float(area_min_weight)
        self.area_max_weight = float(area_max_weight)
        self.normalize_area_weight = bool(normalize_area_weight)

    def _get_area_weights(self, target_boxes: torch.Tensor) -> torch.Tensor:
        # target_boxes are normalized cx, cy, w, h.
        target_area = (
            target_boxes[:, 2].clamp_min(1e-8)
            * target_boxes[:, 3].clamp_min(1e-8)
        )
        weights = (self.area_ref / target_area).pow(self.area_power)
        weights = weights.clamp(
            min=self.area_min_weight,
            max=self.area_max_weight,
        )
        if self.normalize_area_weight and weights.numel() > 0:
            weights = weights / weights.mean().clamp_min(1e-8)
        return weights.detach()

    def loss_boxes(
        self,
        outputs,
        targets,
        indices,
        num_boxes,
        boxes_weight=None,
    ):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]

        matched_target_boxes = [
            target["boxes"][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ]

        # Be robust to a batch with no matched targets.
        if not matched_target_boxes or sum(x.shape[0] for x in matched_target_boxes) == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        target_boxes = torch.cat(matched_target_boxes, dim=0)
        scale_weight = self._get_area_weights(target_boxes)

        loss_bbox_per_box = F.l1_loss(
            src_boxes,
            target_boxes,
            reduction="none",
        ).sum(dim=-1)
        loss_bbox = (loss_bbox_per_box * scale_weight).sum() / num_boxes

        loss_giou_per_box = 1.0 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )

        giou_weight = scale_weight
        if boxes_weight is not None:
            giou_weight = giou_weight * boxes_weight
        loss_giou = (loss_giou_per_box * giou_weight).sum() / num_boxes

        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
