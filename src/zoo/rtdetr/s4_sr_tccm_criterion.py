#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4 SR-TCCM for RT-DETRv2
========================
Scale-conditioned soft-quality Trajectory-Consistent Competition Margin.

S4 is a training-only criterion extension developed from the S2/S3 results:
- medium and large targets retain the S2 competition rule;
- small targets receive a continuous quality-protection factor;
- unlike S3, no competitor is hard-deleted by a quality threshold.

This is a training-only criterion extension. It does NOT:
- create extra positive queries;
- modify Hungarian assignment;
- modify the detector architecture or inference graph;
- add parameters, FLOPs, ONNX nodes, or TensorRT latency.

It only calibrates the final-layer competition between the Hungarian winner
and persistent same-target competitors. The ranking direction is aligned with
RT-DETRv2 VFL: the matched winner is pushed above unmatched competitors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .rtdetrv2_criterion import RTDETRCriterionv2

__all__ = ["S4SRTCCMCriterionv2"]


def pairwise_iou_cxcywh(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """Pairwise IoU: [Q,4] x [M,4] -> [Q,M]."""
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return pred_boxes.new_zeros((pred_boxes.shape[0], target_boxes.shape[0]))
    iou, _ = box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes),
    )
    return iou


@register()
class S4SRTCCMCriterionv2(RTDETRCriterionv2):
    """Original RT-DETRv2 losses plus a conservative query-ranking loss."""

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
        # Schedule and loss budget
        sr_start_epoch: int = 30,
        sr_ramp_epochs: int = 20,
        sr_rank_weight: float = 0.10,
        sr_budget_fraction: float = 0.03,
        # Competitor mining
        sr_min_iou: float = 0.25,
        sr_min_prob: float = 0.03,
        sr_max_competitors: int = 2,
        sr_stability_tau: float = 0.03,
        sr_use_trajectory: bool = True,
        # Reliability gates
        sr_use_scale_reliability: bool = False,
        sr_area_reference: float = 0.01,
        sr_scale_floor: float = 0.10,
        # S4 conditional soft quality protection.
        sr_conditional_quality_power: float = 2.0,
        sr_quality_prob_alpha: float = 1.0,
        sr_quality_iou_beta: float = 2.0,
        sr_quality_tau: float = 0.05,
        sr_winner_iou_tau: float = 0.05,
        # Margin
        sr_margin_base: float = 0.02,
        sr_margin_scale: float = 0.10,
        sr_temperature: float = 0.20,
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

        if sr_start_epoch < 0 or sr_ramp_epochs < 0:
            raise ValueError("Epoch schedule values must be non-negative.")
        if sr_max_competitors < 1:
            raise ValueError("sr_max_competitors must be >= 1.")
        if sr_stability_tau <= 0 or sr_temperature <= 0:
            raise ValueError("Temperature values must be positive.")
        if sr_area_reference <= 0:
            raise ValueError("sr_area_reference must be positive.")
        if not 0 <= sr_scale_floor <= 1:
            raise ValueError("sr_scale_floor must be in [0,1].")
        if sr_conditional_quality_power <= 0:
            raise ValueError("sr_conditional_quality_power must be positive.")

        self.sr_start_epoch = int(sr_start_epoch)
        self.sr_ramp_epochs = int(sr_ramp_epochs)
        self.sr_rank_weight = float(sr_rank_weight)
        self.sr_budget_fraction = float(sr_budget_fraction)

        self.sr_min_iou = float(sr_min_iou)
        self.sr_min_prob = float(sr_min_prob)
        self.sr_max_competitors = int(sr_max_competitors)
        self.sr_stability_tau = float(sr_stability_tau)
        self.sr_use_trajectory = bool(sr_use_trajectory)

        self.sr_use_scale_reliability = bool(sr_use_scale_reliability)
        self.sr_area_reference = float(sr_area_reference)
        self.sr_scale_floor = float(sr_scale_floor)
        self.sr_conditional_quality_power = float(
            sr_conditional_quality_power
        )
        self.sr_quality_prob_alpha = float(sr_quality_prob_alpha)
        self.sr_quality_iou_beta = float(sr_quality_iou_beta)
        self.sr_quality_tau = float(sr_quality_tau)
        self.sr_winner_iou_tau = float(sr_winner_iou_tau)

        self.sr_margin_base = float(sr_margin_base)
        self.sr_margin_scale = float(sr_margin_scale)
        self.sr_temperature = float(sr_temperature)
        self.eps = float(eps)

    def _schedule(self, epoch: int | None) -> float:
        if epoch is None:
            return 1.0  # direct unit/smoke test
        if epoch < self.sr_start_epoch:
            return 0.0
        if self.sr_ramp_epochs == 0:
            return 1.0
        progress = (epoch - self.sr_start_epoch + 1) / float(self.sr_ramp_epochs)
        return max(0.0, min(progress, 1.0))

    @staticmethod
    def _decoder_layers(outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        layers: List[Dict[str, torch.Tensor]] = []
        for aux in outputs.get("aux_outputs", []):
            layers.append({"pred_logits": aux["pred_logits"], "pred_boxes": aux["pred_boxes"]})
        layers.append({"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]})
        return layers

    def _scale_terms(
        self,
        target_box: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return S2 scale weight and unfloored relative target scale.

        `relative_scale` equals one when a target area is at least the training
        set median. S4 therefore becomes exactly S2 for those targets.
        """
        if not self.sr_use_scale_reliability:
            one = target_box.new_tensor(1.0)
            return one, one

        area = (target_box[2] * target_box[3]).detach().clamp_min(self.eps)
        relative_scale = torch.sqrt(
            area / self.sr_area_reference
        ).clamp(max=1.0)
        scale_gate = self.sr_scale_floor + (
            1.0 - self.sr_scale_floor
        ) * relative_scale
        return scale_gate, relative_scale

    def _conditional_quality_gate(
        self,
        quality_delta: torch.Tensor,
        relative_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Blend S2 and soft quality protection according to target scale.

        h = (1-relative_scale)^kappa
        gate = (1-h) + h*sigmoid((q_w-q_c)/tau)
        """
        soft_quality = torch.sigmoid(
            quality_delta / self.sr_quality_tau
        )
        smallness = (
            1.0 - relative_scale
        ).clamp(0.0, 1.0).pow(
            self.sr_conditional_quality_power
        )
        return (1.0 - smallness) + smallness * soft_quality

    def _quality(
        self,
        probability: torch.Tensor,
        iou: torch.Tensor,
    ) -> torch.Tensor:
        return (
            probability.detach().clamp_min(self.eps).pow(self.sr_quality_prob_alpha)
            * iou.detach().clamp_min(0.0).pow(self.sr_quality_iou_beta)
        )

    def _raw_rank_loss(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, torch.Tensor]],
        final_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = outputs["pred_logits"].sum() * 0.0
        layers = self._decoder_layers(outputs)
        total_loss = zero
        total_weight = zero.new_tensor(0.0)

        selected_pairs = 0
        selected_targets = 0
        mean_scale_gate = zero.new_tensor(0.0)
        mean_quality_gate = zero.new_tensor(0.0)
        mean_winner_gate = zero.new_tensor(0.0)
        gate_count = 0

        for batch_index, target in enumerate(targets):
            labels = target["labels"]
            gt_boxes = target["boxes"]
            if labels.numel() == 0:
                continue

            src_indices, tgt_indices = final_indices[batch_index]
            if src_indices.numel() == 0:
                continue

            layer_boxes = [layer["pred_boxes"][batch_index] for layer in layers]
            iou_stack = torch.stack(
                [pairwise_iou_cxcywh(boxes, gt_boxes) for boxes in layer_boxes],
                dim=0,
            )
            final_iou = iou_stack[-1]
            final_logits = layers[-1]["pred_logits"][batch_index]
            final_probs = final_logits.sigmoid()

            # A query assigned to any GT must never be suppressed as a competitor.
            matched_queries = {int(q) for q in src_indices.tolist()}
            best_gt_per_query = final_iou.detach().argmax(dim=1)

            for winner_q_raw, target_j_raw in zip(src_indices.tolist(), tgt_indices.tolist()):
                winner_q = int(winner_q_raw)
                target_j = int(target_j_raw)
                class_j = int(labels[target_j])

                winner_iou = final_iou[winner_q, target_j].detach()
                winner_prob = final_probs[winner_q, class_j].detach()
                if float(winner_iou) < self.sr_min_iou:
                    continue

                candidate_mask = torch.ones(
                    final_logits.shape[0], dtype=torch.bool, device=final_logits.device
                )
                for matched_q in matched_queries:
                    candidate_mask[matched_q] = False

                candidate_mask &= best_gt_per_query.eq(target_j)
                candidate_mask &= final_iou[:, target_j].detach().ge(self.sr_min_iou)
                candidate_mask &= final_probs[:, class_j].detach().ge(self.sr_min_prob)

                # Strict winner protection: never suppress a better-localized query.
                candidate_mask &= final_iou[:, target_j].detach().le(winner_iou)

                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                if candidate_indices.numel() == 0:
                    continue

                candidate_traj = iou_stack[:, candidate_indices, target_j].detach()
                if self.sr_use_trajectory and candidate_traj.shape[0] > 1:
                    variance = candidate_traj.var(dim=0, unbiased=False)
                    stability = torch.exp(-variance / self.sr_stability_tau)
                    trajectory_mean = candidate_traj.mean(dim=0)
                else:
                    stability = torch.ones_like(candidate_traj[-1])
                    trajectory_mean = candidate_traj[-1]

                candidate_iou = candidate_traj[-1]
                persistence = (
                    candidate_iou * stability * (0.5 + 0.5 * trajectory_mean)
                ).clamp(0.0, 1.0)

                candidate_prob = final_probs[candidate_indices, class_j].detach()
                winner_quality = self._quality(winner_prob, winner_iou)
                candidate_quality = self._quality(candidate_prob, candidate_iou)
                quality_delta = winner_quality - candidate_quality

                scale_gate, relative_scale = self._scale_terms(
                    gt_boxes[target_j]
                )
                quality_gate = self._conditional_quality_gate(
                    quality_delta=quality_delta,
                    relative_scale=relative_scale,
                )

                keep = min(self.sr_max_competitors, int(candidate_indices.numel()))
                selection_score = persistence * quality_gate
                _, top_pos = torch.topk(
                    selection_score,
                    k=keep,
                    largest=True,
                    sorted=True,
                )
                competitors = candidate_indices[top_pos]
                competitor_iou = candidate_iou[top_pos]
                selected_persistence = persistence[top_pos]
                competitor_quality_gate = quality_gate[top_pos]

                winner_gate = torch.sigmoid(
                    (winner_iou - competitor_iou) / self.sr_winner_iou_tau
                )
                # Apply the S4 quality gate exactly once. The selection score
                # chooses competitors; the pair weight controls optimization.
                pair_weight = (
                    selected_persistence
                    * scale_gate
                    * winner_gate
                    * competitor_quality_gate
                ).detach().clamp_min(self.eps)

                winner_logit = final_logits[winner_q, class_j]
                competitor_logits = final_logits[competitors, class_j]

                # Keep the S2 margin itself unchanged; only its pair weight is
                # conditioned by target scale and joint-quality reliability.
                margin = (
                    self.sr_margin_base
                    + self.sr_margin_scale * selected_persistence.detach()
                )
                pair_loss = self.sr_temperature * F.softplus(
                    (competitor_logits - winner_logit + margin) / self.sr_temperature
                )

                total_loss = total_loss + (pair_loss * pair_weight).sum()
                total_weight = total_weight + pair_weight.sum()
                selected_pairs += keep
                selected_targets += 1
                mean_scale_gate = mean_scale_gate + scale_gate.detach()
                mean_quality_gate = mean_quality_gate + competitor_quality_gate.mean().detach()
                mean_winner_gate = mean_winner_gate + winner_gate.mean().detach()
                gate_count += 1

        if float(total_weight.detach()) <= 0:
            metrics = {
                "sr_selected_pairs": zero,
                "sr_selected_targets": zero,
                "sr_scale_gate": zero,
                "sr_quality_gate": zero,
                "sr_winner_gate": zero,
            }
            return zero, metrics

        denominator = float(max(gate_count, 1))
        metrics = {
            "sr_selected_pairs": zero.new_tensor(float(selected_pairs)),
            "sr_selected_targets": zero.new_tensor(float(selected_targets)),
            "sr_scale_gate": mean_scale_gate / denominator,
            "sr_quality_gate": mean_quality_gate / denominator,
            "sr_winner_gate": mean_winner_gate / denominator,
        }
        return total_loss / total_weight.clamp_min(self.eps), metrics

    def _budgeted_loss(
        self,
        raw_loss: torch.Tensor,
        base_vfl: torch.Tensor,
        schedule: float,
    ) -> torch.Tensor:
        if schedule <= 0 or self.sr_rank_weight <= 0 or self.sr_budget_fraction <= 0:
            return raw_loss * 0.0
        weighted = raw_loss * (self.sr_rank_weight * schedule)
        maximum = base_vfl.detach().abs() * self.sr_budget_fraction * schedule
        scale = torch.clamp(
            maximum / (weighted.detach().abs() + self.eps),
            max=1.0,
        )
        return weighted * scale

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, torch.Tensor]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        losses = super().forward(outputs, targets, **kwargs)

        epoch_value = kwargs.get("epoch")
        epoch = int(epoch_value) if epoch_value is not None else None
        schedule = self._schedule(epoch)

        final_match = self.matcher(
            {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]},
            targets,
        )["indices"]
        raw_rank, diagnostics = self._raw_rank_loss(outputs, targets, final_match)

        base_vfl = losses.get("loss_vfl", outputs["pred_logits"].sum() * 0.0)
        losses["loss_sr_tccm"] = self._budgeted_loss(raw_rank, base_vfl, schedule)

        # Diagnostic values are multiplied by zero so they appear in logs without
        # changing optimization. Their numerical values remain available in logs.
        for key, value in diagnostics.items():
            losses[key] = value.detach() * 0.0
        losses["sr_schedule"] = outputs["pred_logits"].sum() * 0.0
        return losses
