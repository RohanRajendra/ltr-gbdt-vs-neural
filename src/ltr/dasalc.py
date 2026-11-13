"""DASALC (Qin et al., ICLR 2021) in PyTorch.

DASALC = Data Augmentation + Self-Attention + Latent Cross, on top of a plain
feed-forward ranker. Pieces:
  1. Feature transform: x -> sign(x) * log(1 + |x|), then BatchNorm (no learned affine).
  2. Data augmentation: add Gaussian noise to the transformed features at train time.
  3. Per-document tower: L x [Linear -> BatchNorm -> ReLU -> Dropout].
  4. Listwise context: multi-head self-attention over all docs of the same query,
     producing a context vector a_i per doc.
  5. Latent cross: h_i * (1 + a_i), then a linear layer gives the score.
  6. Loss: softmax cross-entropy over each query's list (ListNet-style).
TODO(verify against paper PDF): exact noise std, layer sizes, attention depth/heads.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def log1p_transform(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(X) * np.log1p(np.abs(X))


class DASALC(nn.Module):
    def __init__(self, n_features: int, hidden: int = 512, layers: int = 3, dropout: float = 0.3,
                 heads: int = 4, attn_layers: int = 2, noise_std: float = 1.5):
        super().__init__()
        self.noise_std = noise_std
        self.in_bn = nn.BatchNorm1d(n_features, affine=False)
        blocks, d = [], n_features
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.tower = nn.Sequential(*blocks)
        enc = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(enc, attn_layers)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) padded lists, mask: (B, L) True for real docs. Returns scores (B, L)."""
        B, L, Fdim = x.shape
        flat = x[mask]                                   # only real docs go through BatchNorm
        flat = self.in_bn(flat)
        if self.training and self.noise_std > 0:
            flat = flat + self.noise_std * torch.randn_like(flat)
        h_flat = self.tower(flat)
        h = x.new_zeros(B, L, h_flat.shape[-1])
        h[mask] = h_flat
        a = self.attn(h, src_key_padding_mask=~mask)     # listwise context per doc
        s = self.out(h * (1 + a)).squeeze(-1)            # latent cross
        return s.masked_fill(~mask, float("-inf"))


def softmax_ce_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ListNet-style: target distribution = labels normalised per query.
    Queries with no relevant docs carry no signal and are skipped."""
    labels = labels.masked_fill(~mask, 0)
    tot = labels.sum(1, keepdim=True)
    keep = tot.squeeze(1) > 0
    if not keep.any():
        return scores.sum() * 0
    target = labels[keep] / tot[keep]
    logp = F.log_softmax(scores[keep], dim=1).masked_fill(~mask[keep], 0)
    return -(target * logp).sum(1).mean()
