#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory-Consistent Competition Margin (TCCM) for RT-DETRv2.

TCCM fixes the main failure mode of the previous TQRS/R1 experiment:
unmatched queries are never relabeled as positives on the primary one-to-one
classification head. Instead, TCCM identifies persistent competitors around a
final Hungarian winner and applies a small, delayed, budget-capped pairwise
margin so the winner ranks above those duplicates.

The inference graph is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .rtdetrv2_criterion import RTDETRCriterionv2

__all__ = ["TCCMCriterionv2"]


def pairwise_iou_cxcywh(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return pred_boxes.new_zeros((pred_boxes.shape[0], target_boxes.shape[0]))
    iou, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))
    return iou


@register()
class TCCMCriterionv2(RTDETRCriterionv2):
    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict: Dict[str, float],
        losses: Sequence[str],
        alpha: float = 0.75,
        gamma: float = 2.0,
        num_classes: int = 80,
        boxes_weight_format: str | None = None,
        share_matched_indices: bool = False,
        tccm_start_epoch: int = 20,
        tccm_ramp_epochs: int = 10,
        tccm_min_iou: float = 0.25,
        tccm_min_prob: float = 0.03,
        tccm_max_competitors: int = 2,
        tccm_stability_tau: float = 0.03,
        tccm_use_trajectory: bool = True,
        tccm_winner_iou_tolerance: float = 0.02,
        tccm_rank_weight: float = 0.20,
        tccm_rank_budget_fraction: float = 0.10,
        tccm_margin_base: float = 0.05,
        tccm_margin_scale: float = 0.20,
        tccm_temperature: float = 0.20,
        tccm_mono_weight: float = 0.00,
        tccm_mono_budget_fraction: float = 0.05,
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
        if tccm_start_epoch < 0 or tccm_ramp_epochs < 0:
            raise ValueError("Invalid TCCM schedule")
        if tccm_max_competitors < 1:
            raise ValueError("tccm_max_competitors must be >= 1")
        if tccm_stability_tau <= 0 or tccm_temperature <= 0:
            raise ValueError("tau and temperature must be positive")

        self.tccm_start_epoch = int(tccm_start_epoch)
        self.tccm_ramp_epochs = int(tccm_ramp_epochs)
        self.tccm_min_iou = float(tccm_min_iou)
        self.tccm_min_prob = float(tccm_min_prob)
        self.tccm_max_competitors = int(tccm_max_competitors)
        self.tccm_stability_tau = float(tccm_stability_tau)
        self.tccm_use_trajectory = bool(tccm_use_trajectory)
        self.tccm_winner_iou_tolerance = float(tccm_winner_iou_tolerance)
        self.tccm_rank_weight = float(tccm_rank_weight)
        self.tccm_rank_budget_fraction = float(tccm_rank_budget_fraction)
        self.tccm_margin_base = float(tccm_margin_base)
        self.tccm_margin_scale = float(tccm_margin_scale)
        self.tccm_temperature = float(tccm_temperature)
        self.tccm_mono_weight = float(tccm_mono_weight)
        self.tccm_mono_budget_fraction = float(tccm_mono_budget_fraction)
        self.eps = float(eps)

    def _schedule(self, epoch: int | None) -> float:
        if epoch is None:
            return 1.0
        if epoch < self.tccm_start_epoch:
            return 0.0
        if self.tccm_ramp_epochs == 0:
            return 1.0
        value = (epoch - self.tccm_start_epoch + 1) / float(self.tccm_ramp_epochs)
        return max(0.0, min(value, 1.0))

    @staticmethod
    def _decoder_layers(outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        layers: List[Dict[str, torch.Tensor]] = []
        for auxiliary in outputs.get("aux_outputs", []):
            layers.append({"pred_logits": auxiliary["pred_logits"], "pred_boxes": auxiliary["pred_boxes"]})
        layers.append({"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]})
        return layers

    def _rank_loss(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, torch.Tensor]],
        final_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        zero = outputs["pred_logits"].sum() * 0.0
        layers = self._decoder_layers(outputs)
        total_loss = zero
        total_weight = zero.new_tensor(0.0)

        for batch_index, target in enumerate(targets):
            labels, target_boxes = target["labels"], target["boxes"]
            if labels.numel() == 0:
                continue
            src_indices, target_indices = final_indices[batch_index]
            if src_indices.numel() == 0:
                continue

            layer_boxes = [layer["pred_boxes"][batch_index] for layer in layers]
            iou_stack = torch.stack([pairwise_iou_cxcywh(boxes, target_boxes) for boxes in layer_boxes], dim=0)
            final_iou = iou_stack[-1]
            final_logits = layers[-1]["pred_logits"][batch_index]
            final_probabilities = final_logits.sigmoid()
            winner_queries = {int(index) for index in src_indices.tolist()}

            for query_value, target_value in zip(src_indices.tolist(), target_indices.tolist()):
                winner_query = int(query_value)
                target_index = int(target_value)
                class_index = int(labels[target_index])
                winner_iou = final_iou[winner_query, target_index].detach()
                if float(winner_iou) < self.tccm_min_iou:
                    continue

                mask = torch.ones(final_logits.shape[0], dtype=torch.bool, device=final_logits.device)
                for matched_query in winner_queries:
                    mask[matched_query] = False
                mask &= final_iou[:, target_index].detach() >= self.tccm_min_iou
                mask &= final_probabilities[:, class_index].detach() >= self.tccm_min_prob
                mask &= final_iou[:, target_index].detach() <= winner_iou + self.tccm_winner_iou_tolerance

                # Avoid cross-object suppression: a competitor is associated
                # with this GT only when this GT is its best-overlap target.
                # This matters when two nearby weeds overlap or share a class.
                best_target_for_query = final_iou.detach().argmax(dim=1)
                mask &= best_target_for_query == target_index
                candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
                if candidate_indices.numel() == 0:
                    continue

                trajectory = iou_stack[:, candidate_indices, target_index].detach()
                if self.tccm_use_trajectory and trajectory.shape[0] > 1:
                    variance = trajectory.var(dim=0, unbiased=False)
                    stability = torch.exp(-variance / self.tccm_stability_tau)
                    trajectory_mean = trajectory.mean(dim=0)
                else:
                    stability = torch.ones_like(trajectory[-1])
                    trajectory_mean = trajectory[-1]

                persistence = (trajectory[-1] * stability * (0.5 + 0.5 * trajectory_mean)).clamp(0.0, 1.0)
                keep = min(self.tccm_max_competitors, int(candidate_indices.numel()))
                values, positions = torch.topk(persistence, k=keep, largest=True, sorted=True)
                competitors = candidate_indices[positions]

                winner_logit = final_logits[winner_query, class_index]
                competitor_logits = final_logits[competitors, class_index]
                margins = (self.tccm_margin_base + self.tccm_margin_scale * values).detach()
                pair_loss = self.tccm_temperature * F.softplus(
                    (competitor_logits - winner_logit + margins) / self.tccm_temperature
                )
                pair_weights = (values * winner_iou).detach().clamp_min(self.eps)
                total_loss = total_loss + (pair_loss * pair_weights).sum()
                total_weight = total_weight + pair_weights.sum()

        if float(total_weight.detach()) <= 0:
            return zero
        return total_loss / total_weight.clamp_min(self.eps)

    def _monotonic_loss(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, torch.Tensor]],
        final_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        zero = outputs["pred_boxes"].sum() * 0.0
        layers = self._decoder_layers(outputs)
        if len(layers) <= 1:
            return zero

        total_loss = zero
        total_weight = zero.new_tensor(0.0)
        for batch_index, target in enumerate(targets):
            target_boxes = target["boxes"]
            src_indices, target_indices = final_indices[batch_index]
            if target_boxes.numel() == 0 or src_indices.numel() == 0:
                continue
            matched_targets = target_boxes[target_indices]
            winner_ious = []
            for layer in layers:
                boxes = layer["pred_boxes"][batch_index][src_indices]
                winner_ious.append(torch.diag(pairwise_iou_cxcywh(boxes, matched_targets)))
            iou_stack = torch.stack(winner_ious, dim=0)
            final_quality = iou_stack[-1].detach().clamp_min(0.1)
            for layer_index in range(iou_stack.shape[0] - 1):
                deterioration = F.relu(iou_stack[layer_index].detach() - iou_stack[layer_index + 1])
                total_loss = total_loss + (deterioration * final_quality).sum()
                total_weight = total_weight + final_quality.sum()

        if float(total_weight.detach()) <= 0:
            return zero
        return total_loss / total_weight.clamp_min(self.eps)

    def _budgeted_loss(
        self,
        raw_loss: torch.Tensor,
        base_loss: torch.Tensor,
        weight: float,
        budget_fraction: float,
        schedule: float,
    ) -> torch.Tensor:
        if weight <= 0 or budget_fraction <= 0 or schedule <= 0:
            return raw_loss * 0.0
        weighted = raw_loss * (weight * schedule)
        maximum = base_loss.detach().abs() * budget_fraction * schedule
        scale = torch.clamp(maximum / (weighted.detach().abs() + self.eps), max=1.0)
        return weighted * scale

    def forward(self, outputs: Dict[str, Any], targets: List[Dict[str, torch.Tensor]], **kwargs: Any) -> Dict[str, torch.Tensor]:
        losses = super().forward(outputs, targets, **kwargs)
        epoch_value = kwargs.get("epoch")
        epoch = int(epoch_value) if epoch_value is not None else None
        schedule = self._schedule(epoch)

        final_output = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}
        final_indices = self.matcher(final_output, targets)["indices"]
        raw_rank = self._rank_loss(outputs, targets, final_indices)
        raw_mono = self._monotonic_loss(outputs, targets, final_indices)

        zero_logits = outputs["pred_logits"].sum() * 0.0
        zero_boxes = outputs["pred_boxes"].sum() * 0.0
        base_vfl = losses.get("loss_vfl", zero_logits)
        base_localization = losses.get("loss_bbox", zero_boxes) + losses.get("loss_giou", zero_boxes)

        losses["loss_tccm_rank"] = self._budgeted_loss(
            raw_rank, base_vfl, self.tccm_rank_weight, self.tccm_rank_budget_fraction, schedule
        )
        losses["loss_tccm_mono"] = self._budgeted_loss(
            raw_mono, base_localization, self.tccm_mono_weight, self.tccm_mono_budget_fraction, schedule
        )
        return losses
