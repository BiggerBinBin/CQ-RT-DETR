#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory-Quality Replay Supervision (TQRS) for RT-DETRv2.

TQRS is a training-only criterion extension. It keeps:
- the original RT-DETRv2 model;
- the original final one-to-one Hungarian matching;
- the original VFL + L1 + GIoU losses;
- the original inference graph and TensorRT engine.

It adds an auxiliary replay supervision on intermediate decoder layers.
Candidate queries are mined using their complete decoder trajectory instead
of a single-layer score.

For query i and target j:
    final quality:
        q_ij = p_ij^alpha * IoU_ij^beta * exp(-gamma * center_distance_ij)

    localization progress:
        gain_ij = relu(IoU_ij^L - IoU_ij^1)

    trajectory stability:
        stability_ij = exp(-Var_l(IoU_ij^l) / tau)

    replay score:
        r_ij = q_ij * (1 + gain_weight * gain_ij) * stability_ij

The number of replay positives for each target is ambiguity-adaptive:
targets with several similarly plausible queries receive more replay
supervision, while easy targets retain nearly one-to-one behavior.

The final decoder layer is never converted to one-to-many supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    generalized_box_iou,
)
from .rtdetrv2_criterion import RTDETRCriterionv2


__all__ = ["TQRSCriterionv2"]


@dataclass
class ReplayPair:
    query_index: int
    target_index: int
    score: float
    quality_target: float
    is_hungarian: bool


def _pairwise_iou_cxcywh(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """Pairwise IoU, shapes [Q,4] and [M,4] -> [Q,M]."""
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return pred_boxes.new_zeros(
            (pred_boxes.shape[0], target_boxes.shape[0])
        )

    iou, _ = box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes),
    )
    return iou


def _pairwise_center_distance(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    GT-size-normalized squared center distance.

    This avoids using an absolute image-size threshold and makes the geometric
    term comparable across object scales.
    """
    pred_center = pred_boxes[:, None, :2]
    target_center = target_boxes[None, :, :2]
    target_size = target_boxes[None, :, 2:].clamp_min(eps)
    offset = (pred_center - target_center) / target_size
    return offset.square().sum(dim=-1)


@register()
class TQRSCriterionv2(RTDETRCriterionv2):
    """
    RT-DETRv2 criterion with cross-layer query trajectory replay supervision.

    The base criterion still computes the complete official RT-DETRv2 loss.
    TQRS only appends:
        loss_tqrs_cls
        loss_tqrs_bbox
        loss_tqrs_giou

    Set any corresponding weight in weight_dict to 0 to disable that term.
    """

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        losses: Sequence[str],
        alpha: float = 0.75,
        gamma: float = 2.0,
        num_classes: int = 80,
        boxes_weight_format: str | None = None,
        share_matched_indices: bool = False,
        # TQRS mining
        tqrs_prob_alpha: float = 1.0,
        tqrs_iou_beta: float = 2.0,
        tqrs_center_gamma: float = 0.25,
        tqrs_gain_weight: float = 1.0,
        tqrs_stability_tau: float = 0.02,
        tqrs_max_pos: int = 3,
        tqrs_min_score: float = 0.02,
        tqrs_min_iou: float = 0.20,
        tqrs_min_gain: float = 0.05,
        tqrs_quality_floor: float = 0.05,
        tqrs_layer_power: float = 1.0,
        tqrs_use_trajectory: bool = True,
        tqrs_adaptive_k: bool = True,
        tqrs_include_hungarian: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=list(losses),
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
        )

        if tqrs_max_pos < 1:
            raise ValueError("tqrs_max_pos must be >= 1.")
        if tqrs_stability_tau <= 0:
            raise ValueError("tqrs_stability_tau must be positive.")
        if tqrs_quality_floor < 0 or tqrs_quality_floor > 1:
            raise ValueError("tqrs_quality_floor must be in [0, 1].")

        self.tqrs_prob_alpha = float(tqrs_prob_alpha)
        self.tqrs_iou_beta = float(tqrs_iou_beta)
        self.tqrs_center_gamma = float(tqrs_center_gamma)
        self.tqrs_gain_weight = float(tqrs_gain_weight)
        self.tqrs_stability_tau = float(tqrs_stability_tau)
        self.tqrs_max_pos = int(tqrs_max_pos)
        self.tqrs_min_score = float(tqrs_min_score)
        self.tqrs_min_iou = float(tqrs_min_iou)
        self.tqrs_min_gain = float(tqrs_min_gain)
        self.tqrs_quality_floor = float(tqrs_quality_floor)
        self.tqrs_layer_power = float(tqrs_layer_power)
        self.tqrs_use_trajectory = bool(tqrs_use_trajectory)
        self.tqrs_adaptive_k = bool(tqrs_adaptive_k)
        self.tqrs_include_hungarian = bool(tqrs_include_hungarian)
        self.eps = float(eps)

    def _layer_outputs(
        self,
        outputs: Dict[str, Any],
    ) -> List[Dict[str, torch.Tensor]]:
        layers: List[Dict[str, torch.Tensor]] = []

        for auxiliary in outputs.get("aux_outputs", []):
            layers.append(
                {
                    "pred_logits": auxiliary["pred_logits"],
                    "pred_boxes": auxiliary["pred_boxes"],
                }
            )

        layers.append(
            {
                "pred_logits": outputs["pred_logits"],
                "pred_boxes": outputs["pred_boxes"],
            }
        )
        return layers

    def _dynamic_budget(self, scores: torch.Tensor) -> int:
        """
        Ambiguity-adaptive positive budget.

        If the best candidate strongly dominates the second candidate,
        the target is easy and K approaches 1. If several candidates have
        similar scores, K approaches tqrs_max_pos.
        """
        if not self.tqrs_adaptive_k or self.tqrs_max_pos == 1:
            return self.tqrs_max_pos

        if scores.numel() <= 1:
            return 1

        top2 = torch.topk(scores, k=2, largest=True, sorted=True).values
        top1 = float(top2[0].detach())
        second = float(top2[1].detach())

        if top1 <= self.tqrs_min_score:
            return 1

        relative_gap = max(min((top1 - second) / (top1 + self.eps), 1.0), 0.0)
        ambiguity = 1.0 - relative_gap
        budget = 1 + int(round((self.tqrs_max_pos - 1) * ambiguity))
        return max(1, min(self.tqrs_max_pos, budget))

    def _mine_pairs_single_image(
        self,
        layer_logits: List[torch.Tensor],
        layer_boxes: List[torch.Tensor],
        target: Dict[str, torch.Tensor],
        hungarian_pair: Tuple[torch.Tensor, torch.Tensor],
    ) -> List[ReplayPair]:
        labels = target["labels"]
        target_boxes = target["boxes"]

        num_targets = int(labels.numel())
        num_queries = int(layer_logits[-1].shape[0])

        if num_targets == 0 or num_queries == 0:
            return []

        iou_layers: List[torch.Tensor] = []
        prob_layers: List[torch.Tensor] = []

        for logits, boxes in zip(layer_logits, layer_boxes):
            iou_layers.append(_pairwise_iou_cxcywh(boxes, target_boxes))
            # [Q,C][:, labels] -> [Q,M]
            prob_layers.append(logits.sigmoid()[:, labels])

        iou_stack = torch.stack(iou_layers, dim=0)
        final_iou = iou_stack[-1]
        first_iou = iou_stack[0]
        final_prob = prob_layers[-1]

        center_distance = _pairwise_center_distance(
            layer_boxes[-1],
            target_boxes,
            eps=self.eps,
        )

        quality = (
            final_prob.clamp_min(self.eps).pow(self.tqrs_prob_alpha)
            * final_iou.clamp_min(0.0).pow(self.tqrs_iou_beta)
            * torch.exp(-self.tqrs_center_gamma * center_distance)
        )

        if self.tqrs_use_trajectory and len(iou_layers) > 1:
            gain = (final_iou - first_iou).clamp_min(0.0)
            variance = iou_stack.var(dim=0, unbiased=False)
            stability = torch.exp(
                -variance / max(self.tqrs_stability_tau, self.eps)
            )
        else:
            gain = torch.zeros_like(final_iou)
            stability = torch.ones_like(final_iou)

        replay_score = (
            quality
            * (1.0 + self.tqrs_gain_weight * gain)
            * stability
        )

        quality_target = (
            final_iou * stability
        ).detach().clamp(
            min=self.tqrs_quality_floor,
            max=1.0,
        )

        budgets = [
            self._dynamic_budget(replay_score[:, target_index])
            for target_index in range(num_targets)
        ]

        selected: List[ReplayPair] = []
        used_queries: set[int] = set()
        target_counts = [0 for _ in range(num_targets)]

        # Always preserve the final Hungarian winner as the trajectory anchor.
        if self.tqrs_include_hungarian:
            src_indices, target_indices = hungarian_pair
            for query_tensor, target_tensor in zip(
                src_indices.tolist(),
                target_indices.tolist(),
            ):
                query_index = int(query_tensor)
                target_index = int(target_tensor)

                if (
                    query_index < 0
                    or query_index >= num_queries
                    or target_index < 0
                    or target_index >= num_targets
                ):
                    continue

                if query_index in used_queries:
                    continue

                selected.append(
                    ReplayPair(
                        query_index=query_index,
                        target_index=target_index,
                        score=float(
                            replay_score[query_index, target_index].detach()
                        ),
                        quality_target=float(
                            quality_target[query_index, target_index].detach()
                        ),
                        is_hungarian=True,
                    )
                )
                used_queries.add(query_index)
                target_counts[target_index] += 1

        candidates: List[Tuple[float, int, int]] = []

        for query_index in range(num_queries):
            for target_index in range(num_targets):
                score = float(
                    replay_score[query_index, target_index].detach()
                )
                current_iou = float(
                    final_iou[query_index, target_index].detach()
                )
                current_gain = float(
                    gain[query_index, target_index].detach()
                )

                valid_geometry = (
                    current_iou >= self.tqrs_min_iou
                    or current_gain >= self.tqrs_min_gain
                )

                if score >= self.tqrs_min_score and valid_geometry:
                    candidates.append(
                        (score, query_index, target_index)
                    )

        candidates.sort(key=lambda item: item[0], reverse=True)

        # Global greedy assignment keeps one query associated with one target.
        for score, query_index, target_index in candidates:
            if query_index in used_queries:
                continue
            if target_counts[target_index] >= budgets[target_index]:
                continue

            selected.append(
                ReplayPair(
                    query_index=query_index,
                    target_index=target_index,
                    score=score,
                    quality_target=float(
                        quality_target[query_index, target_index].detach()
                    ),
                    is_hungarian=False,
                )
            )
            used_queries.add(query_index)
            target_counts[target_index] += 1

        return selected

    def _tqrs_losses(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        zero = outputs["pred_logits"].sum() * 0.0
        losses = {
            "loss_tqrs_cls": zero,
            "loss_tqrs_bbox": zero,
            "loss_tqrs_giou": zero,
        }

        layers = self._layer_outputs(outputs)

        # There is no earlier decoder layer to replay when num_layers == 1.
        if len(layers) <= 1:
            return losses

        final_for_matcher = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        final_matching = self.matcher(final_for_matcher, targets)["indices"]

        total_cls = zero
        total_bbox = zero
        total_giou = zero
        total_weight = zero.new_tensor(0.0)

        num_replay_layers = len(layers) - 1

        for batch_index, target in enumerate(targets):
            layer_logits = [
                layer["pred_logits"][batch_index]
                for layer in layers
            ]
            layer_boxes = [
                layer["pred_boxes"][batch_index]
                for layer in layers
            ]

            pairs = self._mine_pairs_single_image(
                layer_logits=layer_logits,
                layer_boxes=layer_boxes,
                target=target,
                hungarian_pair=final_matching[batch_index],
            )

            if not pairs:
                continue

            query_indices = torch.tensor(
                [pair.query_index for pair in pairs],
                dtype=torch.long,
                device=layer_logits[-1].device,
            )
            target_indices = torch.tensor(
                [pair.target_index for pair in pairs],
                dtype=torch.long,
                device=layer_logits[-1].device,
            )
            quality_targets = torch.tensor(
                [pair.quality_target for pair in pairs],
                dtype=layer_logits[-1].dtype,
                device=layer_logits[-1].device,
            )

            class_indices = target["labels"][target_indices]
            replay_target_boxes = target["boxes"][target_indices]

            for layer_index in range(num_replay_layers):
                # Later intermediate layers receive more weight because their
                # trajectories are closer to the final output.
                normalized_position = (
                    float(layer_index + 1) / float(num_replay_layers)
                )
                layer_weight = normalized_position ** self.tqrs_layer_power

                logits = layer_logits[layer_index]
                boxes = layer_boxes[layer_index]

                selected_logits = logits[
                    query_indices,
                    class_indices,
                ]
                cls_loss = F.binary_cross_entropy_with_logits(
                    selected_logits,
                    quality_targets,
                    reduction="none",
                )
                cls_loss = (
                    cls_loss * quality_targets
                ).sum()

                selected_boxes = boxes[query_indices]
                bbox_loss = F.l1_loss(
                    selected_boxes,
                    replay_target_boxes,
                    reduction="none",
                ).sum(dim=-1)
                bbox_loss = (
                    bbox_loss * quality_targets
                ).sum()

                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(selected_boxes),
                    box_cxcywh_to_xyxy(replay_target_boxes),
                )
                giou_loss = (
                    (1.0 - torch.diag(giou))
                    * quality_targets
                ).sum()

                total_cls = total_cls + layer_weight * cls_loss
                total_bbox = total_bbox + layer_weight * bbox_loss
                total_giou = total_giou + layer_weight * giou_loss
                total_weight = (
                    total_weight
                    + layer_weight
                    * quality_targets.sum().clamp_min(self.eps)
                )

        if float(total_weight.detach()) <= 0:
            return losses

        losses["loss_tqrs_cls"] = total_cls / total_weight
        losses["loss_tqrs_bbox"] = total_bbox / total_weight
        losses["loss_tqrs_giou"] = total_giou / total_weight
        return losses

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, torch.Tensor]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        # Official RT-DETRv2 losses and matching remain untouched.
        losses = super().forward(outputs, targets, **kwargs)

        replay_losses = self._tqrs_losses(outputs, targets)
        for name, value in replay_losses.items():
            coefficient = float(self.weight_dict.get(name, 0.0))
            losses[name] = coefficient * value

        return losses
