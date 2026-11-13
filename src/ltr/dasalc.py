"""DASALC (Qin et al., ICLR 2021) in PyTorch.

DASALC = Data Augmentation + Self-Attention + Latent Cross, on top of a plain
feed-forward ranker. Pieces:
  1. Feature transform: x -> sign(x) * log(1 + |x|), then BatchNorm (no learned affine).
  2. Data augmentation: add Gaussian noise to the transformed features at train time.
  3. Per-document tower: L x [Linear -> BatchNorm -> ReLU -> Dropout].
  4. Listwise context: multi-head self-attention over the transformed input features
     of all docs of the same query, producing a context vector a_i per doc.
  5. Latent cross: h_i * (1 + W a_i), then a linear layer gives the score.
  6. Loss: softmax cross-entropy over each query's list (ListNet-style).
Paper tuning ranges (App. A): hidden {256..4096}, layers {3..6}, attention layers {3..6},
heads {2..5}, noise sigma in [0, 5] (1.5 best on Web30K).
Bug fixed 2026-11-13: attention was first applied to the tower output; with the encoder's
residual connection that made the latent cross ~ h + h^2 and training unstable.
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
                 heads: int = 4, attn_layers: int = 2, noise_std: float = 1.5, use_attn: bool = True,
                 attn_dim: int = 128):
        super().__init__()
        self.noise_std = noise_std
        self.use_attn = use_attn  # False = plain DNN baseline (no listwise context / latent cross)
        self.in_bn = nn.BatchNorm1d(n_features, affine=False)
        blocks, d = [], n_features
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.tower = nn.Sequential(*blocks)
        # Paper Sec 4.2: self-attention over the (transformed) INPUT features, L layers of
        # MHSA + LayerNorm, then a linear projection to the hidden size for the latent cross.
        self.attn_in = nn.Linear(n_features, attn_dim)
        enc = nn.TransformerEncoderLayer(attn_dim, heads, dim_feedforward=attn_dim, dropout=dropout,
                                         batch_first=True)
        self.attn = nn.TransformerEncoder(enc, attn_layers, enable_nested_tensor=False)
        self.attn_out = nn.Linear(attn_dim, hidden)
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
        if self.use_attn:
            f = x.new_zeros(B, L, Fdim)
            f[mask] = flat                                # transformed (+noised) inputs
            a = self.attn(self.attn_in(f), src_key_padding_mask=~mask)  # listwise context
            h = h * (1 + self.attn_out(a))                # latent cross, eq. (11)
        s = self.out(h).squeeze(-1)
        return s.masked_fill(~mask, float("-inf"))


def softmax_ce_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ListNet-style: target distribution = labels normalised per query.
    Queries with no relevant docs carry no signal and are skipped."""
    labels = labels.masked_fill(~mask, 0)
    tot = labels.sum(1, keepdim=True)
    keep = tot.squeeze(1) > 0
    if not keep.any():
        return scores[mask].sum() * 0  # padded scores are -inf, and -inf * 0 = NaN
    target = labels[keep] / tot[keep]
    logp = F.log_softmax(scores[keep], dim=1).masked_fill(~mask[keep], 0)
    return -(target * logp).sum(1).mean()
