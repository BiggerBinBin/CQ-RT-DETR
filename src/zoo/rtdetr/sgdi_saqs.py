#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGDI + SAQS extensions for the official RT-DETRv2 PyTorch implementation.

Target repository layout:
    rtdetrv2_pytorch/
      src/zoo/rtdetr/
        hybrid_encoder.py
        rtdetrv2_decoder.py
        rtdetrv2_criterion.py

Components
----------
1. SGDIHybridEncoder
   Semantic-Guided Detail Injection:
   - receives PResNet C2/C3/C4/C5
   - injects gated C2 residual detail into C3
   - then executes the original three-level HybridEncoder

2. ScaleAwareRTDETRCriterionv2
   - applies bounded area-aware weights to L1 and GIoU losses
   - optionally supervises SAQS survival logits

3. SAQSRTDETRTransformerv2
   Scale-Aware Query Survival:
   - training keeps all matching queries, preserving standard auxiliary losses
   - inference/export hard-prunes queries between decoder layers
   - denoising queries are excluded from survival supervision
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .denoising import get_contrastive_denoising_training_group
from .hybrid_encoder import HybridEncoder
from .rtdetrv2_criterion import RTDETRCriterionv2
from .rtdetrv2_decoder import (
    MLP,
    RTDETRTransformerv2,
    TransformerDecoderLayer,
)
from .utils import inverse_sigmoid


__all__ = [
    "SGDIHybridEncoder",
    "ScaleAwareRTDETRCriterionv2",
    "SAQSRTDETRTransformerv2",
]


class ConvBNAct(nn.Module):
    """Small ONNX/TensorRT-friendly convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SemanticGuidedDetailInjection(nn.Module):
    """
    Inject C2 local residual details into C3 under C3/C4 semantic gating.

    Equations:
        R2 = |C2 - AvgPool(C2)|
        D3 = PWConv(DWConv_s2([C2, R2]))
        G3 = sigmoid(Conv([Proj(C3), Up(Proj(C4))]))
        C3_hat = C3 + alpha * G3 * D3

    alpha is initialized to zero, so a newly created model initially behaves
    exactly like the original C3 path.
    """

    def __init__(
        self,
        c2_channels: int = 64,
        c3_channels: int = 128,
        c4_channels: int = 256,
        gate_channels: int = 64,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        detail_in = 2 * c2_channels

        self.detail_dw = ConvBNAct(
            detail_in,
            detail_in,
            kernel_size=3,
            stride=2,
            groups=detail_in,
            act=True,
        )
        self.detail_pw = ConvBNAct(
            detail_in,
            c3_channels,
            kernel_size=1,
            stride=1,
            act=False,
        )

        self.c3_proj = ConvBNAct(
            c3_channels,
            gate_channels,
            kernel_size=1,
            stride=1,
            act=True,
        )
        self.c4_proj = ConvBNAct(
            c4_channels,
            gate_channels,
            kernel_size=1,
            stride=1,
            act=True,
        )
        self.gate = nn.Sequential(
            ConvBNAct(
                2 * gate_channels,
                gate_channels,
                kernel_size=3,
                stride=1,
                act=True,
            ),
            nn.Conv2d(gate_channels, c3_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
    ) -> torch.Tensor:
        smooth = F.avg_pool2d(c2, kernel_size=3, stride=1, padding=1)
        residual = torch.abs(c2 - smooth)

        detail = torch.cat([c2, residual], dim=1)
        detail = self.detail_pw(self.detail_dw(detail))

        c4_up = F.interpolate(
            self.c4_proj(c4),
            size=c3.shape[-2:],
            mode="nearest",
        )
        gate = self.gate(torch.cat([self.c3_proj(c3), c4_up], dim=1))

        return c3 + self.alpha * gate * detail


@register()
class SGDIHybridEncoder(HybridEncoder):
    """
    HybridEncoder-compatible subclass that preserves original parameter names.

    Because the original encoder layers remain directly under this module
    (input_proj, encoder, fpn_blocks, ...), an E_lite3_cb checkpoint can be
    loaded with RT-DETRv2's `--tuning` mechanism. Only the new `sgdi.*`
    parameters are unmatched.
    """

    __share__ = ["eval_spatial_size"]

    def __init__(
        self,
        in_channels: Sequence[int] = (64, 128, 256, 512),
        feat_strides: Sequence[int] = (4, 8, 16, 32),
        hidden_dim: int = 192,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_act: str = "gelu",
        use_encoder_idx: Sequence[int] = (2,),
        num_encoder_layers: int = 1,
        pe_temperature: float = 10000,
        expansion: float = 0.5,
        depth_mult: float = 1.0,
        act: str = "silu",
        eval_spatial_size: Optional[Sequence[int]] = None,
        version: str = "v2",
        gate_channels: int = 64,
        alpha_init: float = 0.0,
    ) -> None:
        if len(in_channels) != 4 or len(feat_strides) != 4:
            raise ValueError(
                "SGDIHybridEncoder expects C2/C3/C4/C5 channels and strides."
            )

        self.sgdi_input_channels = list(in_channels)
        self.sgdi_input_strides = list(feat_strides)

        # The inherited HybridEncoder still operates on C3/C4/C5.
        super().__init__(
            in_channels=list(in_channels[1:]),
            feat_strides=list(feat_strides[1:]),
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=list(use_encoder_idx),
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
        )

        self.sgdi = SemanticGuidedDetailInjection(
            c2_channels=int(in_channels[0]),
            c3_channels=int(in_channels[1]),
            c4_channels=int(in_channels[2]),
            gate_channels=gate_channels,
            alpha_init=alpha_init,
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != 4:
            raise ValueError(
                f"Expected four PResNet features C2-C5, received {len(feats)}."
            )

        c2, c3, c4, c5 = feats
        enhanced_c3 = self.sgdi(c2, c3, c4)
        return super().forward([enhanced_c3, c4, c5])


class QuerySurvivalHead(nn.Module):
    """Learnable query validity head."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        mid = max(hidden_dim // 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, 1),
        )
        init.xavier_uniform_(self.net[1].weight)
        init.zeros_(self.net[1].bias)
        init.xavier_uniform_(self.net[3].weight)
        # Low prior probability because only a few of 100 queries are positives.
        init.constant_(self.net[3].bias, math.log(0.05 / 0.95))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.net(query).squeeze(-1)


class SAQSTransformerDecoder(nn.Module):
    """
    Decoder with train-time survival supervision and eval-time hard pruning.

    keep_queries defines the number of matching queries entering each layer.
    Example [100, 75, 50]:
        layer 1 receives 100
        layer 2 receives 75
        layer 3 receives 50

    During training all matching queries are retained. This keeps the official
    auxiliary and denoising losses shape-compatible. During eval/ONNX export,
    fixed Top-K pruning is performed after layers 1 and 2.
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: nn.Module,
        num_layers: int,
        eval_idx: int = -1,
        keep_queries: Sequence[int] = (100, 75, 50),
        survival_beta: float = 0.5,
        survival_area_lambda: float = 0.25,
        survival_area_tau: float = 0.05,
        score_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(keep_queries) != num_layers:
            raise ValueError(
                f"keep_queries length {len(keep_queries)} != num_layers {num_layers}"
            )
        if any(k <= 0 for k in keep_queries):
            raise ValueError("All keep_queries values must be positive.")
        if any(keep_queries[i + 1] > keep_queries[i] for i in range(num_layers - 1)):
            raise ValueError("keep_queries must be non-increasing.")

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.keep_queries = [int(k) for k in keep_queries]

        self.survival_heads = nn.ModuleList(
            [QuerySurvivalHead(hidden_dim) for _ in range(num_layers - 1)]
        )
        self.survival_beta = float(survival_beta)
        self.survival_area_lambda = float(survival_area_lambda)
        self.survival_area_tau = float(survival_area_tau)
        self.score_eps = float(score_eps)

    @staticmethod
    def _gather_queries(
        tensor: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        index = indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
        return tensor.gather(dim=1, index=index)

    def _survival_score(
        self,
        layer_index: int,
        query: torch.Tensor,
        logits: torch.Tensor,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        survival_logits = self.survival_heads[layer_index](query)
        confidence = logits.sigmoid().amax(dim=-1).clamp_min(self.score_eps)
        area = (boxes[..., 2] * boxes[..., 3]).clamp_min(0.0)

        score = (
            survival_logits
            + self.survival_beta * torch.log(confidence)
            + self.survival_area_lambda
            * torch.exp(-area / max(self.survival_area_tau, self.score_eps))
        )
        return score, survival_logits

    def forward(
        self,
        target: torch.Tensor,
        ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: List[List[int]],
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        attn_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        dn_count: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        dec_out_bboxes: List[torch.Tensor] = []
        dec_out_logits: List[torch.Tensor] = []
        survival_outputs: List[Dict[str, torch.Tensor]] = []

        ref_points_detach = F.sigmoid(ref_points_unact)
        output = target
        ref_points: Optional[torch.Tensor] = None

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            bbox_delta = bbox_head[i](output)
            inter_ref_bbox = F.sigmoid(
                bbox_delta + inverse_sigmoid(ref_points_detach)
            )
            layer_logits = score_head[i](output)

            if self.training:
                if i == 0 or ref_points is None:
                    layer_boxes = inter_ref_bbox
                else:
                    layer_boxes = F.sigmoid(
                        bbox_delta + inverse_sigmoid(ref_points)
                    )

                dec_out_logits.append(layer_logits)
                dec_out_bboxes.append(layer_boxes)

                if i < self.num_layers - 1:
                    # Denoising queries are at the beginning. They retain their
                    # official losses and are excluded from survival supervision.
                    match_query = output[:, dn_count:]
                    match_logits = layer_logits[:, dn_count:]
                    match_boxes = layer_boxes[:, dn_count:]
                    survival_outputs.append(
                        {
                            "pred_survival": self.survival_heads[i](match_query),
                            "pred_logits": match_logits,
                            "pred_boxes": match_boxes,
                        }
                    )

                ref_points = inter_ref_bbox
                ref_points_detach = inter_ref_bbox.detach()
                continue

            # Evaluation/export: no denoising queries and no attention mask.
            if dn_count != 0:
                raise RuntimeError("dn_count must be zero during evaluation.")
            if attn_mask is not None:
                raise RuntimeError(
                    "Hard query pruning does not support an eval attention mask."
                )

            if i == self.eval_idx:
                dec_out_logits.append(layer_logits)
                dec_out_bboxes.append(inter_ref_bbox)
                break

            next_k = min(self.keep_queries[i + 1], output.shape[1])
            if next_k < output.shape[1]:
                score, _ = self._survival_score(
                    i,
                    output,
                    layer_logits,
                    inter_ref_bbox,
                )
                topk_indices = torch.topk(
                    score,
                    k=next_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices

                output = self._gather_queries(output, topk_indices)
                inter_ref_bbox = self._gather_queries(
                    inter_ref_bbox,
                    topk_indices,
                )

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            survival_outputs,
        )


@register()
class SAQSRTDETRTransformerv2(RTDETRTransformerv2):
    """RT-DETRv2 decoder with scale-aware query survival."""

    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 192,
        num_queries: int = 100,
        feat_channels: Sequence[int] = (192, 192, 192),
        feat_strides: Sequence[int] = (8, 16, 32),
        num_levels: int = 3,
        num_points: Any = (4, 4, 4),
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learn_query_content: bool = False,
        eval_spatial_size: Optional[Sequence[int]] = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
        aux_loss: bool = True,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        keep_queries: Sequence[int] = (100, 75, 50),
        survival_beta: float = 0.5,
        survival_area_lambda: float = 0.25,
        survival_area_tau: float = 0.05,
    ) -> None:
        # Copy mutable arguments before the official constructor can modify them.
        feat_channels = list(feat_channels)
        feat_strides = list(feat_strides)
        num_points_value = list(num_points) if isinstance(num_points, (list, tuple)) else num_points

        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points_value,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points_value,
            cross_attn_method=cross_attn_method,
        )
        self.decoder = SAQSTransformerDecoder(
            hidden_dim=hidden_dim,
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            eval_idx=eval_idx,
            keep_queries=keep_queries,
            survival_beta=survival_beta,
            survival_area_lambda=survival_area_lambda,
            survival_area_tau=survival_area_tau,
        )

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        memory, spatial_shapes = self._get_encoder_input(list(feats))

        if self.training and self.num_denoising > 0:
            if targets is None:
                raise ValueError("targets are required during denoising training.")
            (
                denoising_logits,
                denoising_bbox_unact,
                attn_mask,
                dn_meta,
            ) = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_logits = None
            denoising_bbox_unact = None
            attn_mask = None
            dn_meta = None

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        ) = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )

        dn_count = 0
        if self.training and dn_meta is not None:
            dn_count = int(dn_meta["dn_num_split"][0])

        out_bboxes, out_logits, survival_outputs = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            dn_count=dn_count,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes,
                dn_meta["dn_num_split"],
                dim=2,
            )
            dn_out_logits, out_logits = torch.split(
                out_logits,
                dn_meta["dn_num_split"],
                dim=2,
            )

        out: Dict[str, Any] = {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
        }

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                out_logits[:-1],
                out_bboxes[:-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list,
                enc_topk_bboxes_list,
            )
            out["enc_meta"] = {
                "class_agnostic": self.query_select_method == "agnostic"
            }

            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits,
                    dn_out_bboxes,
                )
                out["dn_meta"] = dn_meta

            if survival_outputs:
                out["survival_outputs"] = survival_outputs

        return out


@register()
class ScaleAwareRTDETRCriterionv2(RTDETRCriterionv2):
    """
    RT-DETRv2 criterion with bounded small-object localization weighting and
    optional query-survival supervision.
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
        boxes_weight_format: Optional[str] = None,
        share_matched_indices: bool = False,
        scale_lambda: float = 0.5,
        scale_gamma: float = 1.0,
        scale_weight_max: float = 1.5,
        survival_focal_alpha: float = 0.75,
        survival_focal_gamma: float = 2.0,
        survival_small_lambda: float = 0.5,
        survival_small_gamma: float = 1.0,
        survival_weight_max: float = 1.5,
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
        self.scale_lambda = float(scale_lambda)
        self.scale_gamma = float(scale_gamma)
        self.scale_weight_max = float(scale_weight_max)

        self.survival_focal_alpha = float(survival_focal_alpha)
        self.survival_focal_gamma = float(survival_focal_gamma)
        self.survival_small_lambda = float(survival_small_lambda)
        self.survival_small_gamma = float(survival_small_gamma)
        self.survival_weight_max = float(survival_weight_max)

    def _area_weight(
        self,
        target_boxes: torch.Tensor,
        strength: float,
        exponent: float,
        maximum: float,
    ) -> torch.Tensor:
        area = (
            target_boxes[:, 2].clamp_min(0.0)
            * target_boxes[:, 3].clamp_min(0.0)
        )
        weight = 1.0 + strength * torch.pow(
            1.0 - torch.sqrt(area.clamp(max=1.0)),
            exponent,
        )
        return weight.clamp(max=maximum)

    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
        boxes_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if "pred_boxes" not in outputs:
            raise KeyError("pred_boxes is required.")

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [target["boxes"][target_idx]
             for target, (_, target_idx) in zip(targets, indices)],
            dim=0,
        )

        if target_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        scale_weight = self._area_weight(
            target_boxes,
            strength=self.scale_lambda,
            exponent=self.scale_gamma,
            maximum=self.scale_weight_max,
        )

        l1 = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = (l1 * scale_weight.unsqueeze(-1)).sum() / num_boxes

        loss_giou = 1.0 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = loss_giou * scale_weight
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight
        loss_giou = loss_giou.sum() / num_boxes

        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

    def _survival_loss(
        self,
        survival_output: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        match_input = {
            "pred_logits": survival_output["pred_logits"],
            "pred_boxes": survival_output["pred_boxes"],
        }
        indices = self.matcher(match_input, targets)["indices"]
        logits = survival_output["pred_survival"]

        target_survival = torch.zeros_like(logits)
        sample_weight = torch.ones_like(logits)

        for batch_index, (src_index, target_index) in enumerate(indices):
            if src_index.numel() == 0:
                continue

            target_survival[batch_index, src_index] = 1.0
            gt_boxes = targets[batch_index]["boxes"][target_index]
            positive_weight = self._area_weight(
                gt_boxes,
                strength=self.survival_small_lambda,
                exponent=self.survival_small_gamma,
                maximum=self.survival_weight_max,
            )
            sample_weight[batch_index, src_index] = positive_weight

        focal = torchvision.ops.sigmoid_focal_loss(
            logits,
            target_survival,
            alpha=self.survival_focal_alpha,
            gamma=self.survival_focal_gamma,
            reduction="none",
        )
        return (focal * sample_weight).mean()

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, torch.Tensor]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        survival_outputs = outputs.get("survival_outputs", None)

        # Keep the official matcher/loss pipeline isolated from custom records.
        base_outputs = dict(outputs)
        base_outputs.pop("survival_outputs", None)
        losses = super().forward(base_outputs, targets, **kwargs)

        if survival_outputs:
            coefficient = float(self.weight_dict.get("loss_survival", 1.0))
            for index, survival_output in enumerate(survival_outputs):
                loss = self._survival_loss(survival_output, targets)
                losses[f"loss_survival_aux_{index}"] = coefficient * loss

        return losses
