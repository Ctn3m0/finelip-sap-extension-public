"""Subcaption-Aggregated Patch (SAP) objective for FineLIP."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def aggregate_subcaption_patches(patch_features, subcaption_features):
    """Aggregate patches for variable-count subcaptions.

    Args:
        patch_features: Tensor shaped ``[B, P, D]`` without a CLS token.
        subcaption_features: List of ``B`` tensors shaped ``[M_i, D]``.

    Returns:
        A pair of concatenated normalized tensors, each ``[sum(M_i), D]``.
    """
    if patch_features.ndim != 3:
        raise ValueError("patch_features must have shape [B, P, D]")
    if len(subcaption_features) != patch_features.shape[0]:
        raise ValueError("one subcaption tensor is required per image")

    patch_features = F.normalize(patch_features, dim=-1)
    pooled_visual = []
    normalized_text = []
    scale = math.sqrt(patch_features.shape[-1])

    for patches, subcaptions in zip(patch_features, subcaption_features):
        if subcaptions.ndim != 2:
            raise ValueError("each subcaption tensor must have shape [M_i, D]")
        if subcaptions.shape[-1] != patches.shape[-1]:
            raise ValueError("patch and subcaption embedding widths must match")
        if subcaptions.shape[0] == 0:
            continue

        subcaptions = F.normalize(subcaptions, dim=-1)
        attention = torch.softmax((subcaptions @ patches.T) / scale, dim=-1)
        pooled = F.normalize(attention @ patches, dim=-1)
        pooled_visual.append(pooled)
        normalized_text.append(subcaptions)

    if not pooled_visual:
        width = patch_features.shape[-1]
        empty = patch_features.new_empty((0, width))
        return empty, empty
    return torch.cat(pooled_visual, dim=0), torch.cat(normalized_text, dim=0)


class SAPLoss(nn.Module):
    """Local-rank sigmoid contrastive loss for subcaption/patch pairs."""

    def __init__(self, embed_dim, init_logit_scale=math.log(100.0), init_logit_bias=-10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(init_logit_bias)))

    def contrastive_loss(self, visual_features, text_features):
        if visual_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError("contrastive features must have shape [N, D]")
        if visual_features.shape != text_features.shape:
            raise ValueError("visual and text features must have identical shapes")
        if visual_features.shape[0] == 0:
            return (visual_features.sum() + text_features.sum()) * 0.0

        logits = self.logit_scale.exp() * (visual_features @ text_features.T)
        logits = logits + self.logit_bias
        labels = 2.0 * torch.eye(
            logits.shape[0], device=logits.device, dtype=logits.dtype
        ) - 1.0
        return -F.logsigmoid(labels * logits).mean()

    def forward(self, patch_features, subcaption_features):
        if patch_features.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected embedding width {self.embed_dim}, got {patch_features.shape[-1]}"
            )
        pooled_visual, normalized_text = aggregate_subcaption_patches(
            patch_features, subcaption_features
        )
        return self.contrastive_loss(pooled_visual, normalized_text)
