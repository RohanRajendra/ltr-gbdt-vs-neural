"""DASALC (Qin et al., ICLR 2021) in PyTorch, following the paper's Sec. 4, Fig. 1 and eq. 13.

DASALC = Data Augmentation + Self-Attention + Latent Cross. For each query's list of documents:
  1. Input: x -> sign(x) * log(1 + |x|)  (eq. 6). The paper skips this for Yahoo, whose
     features are already normalised; see `log1p_transform` and run_dasalc.py --log1p.
     Then BatchNorm without a learned scale/shift.
  2. Data augmentation: add N(0, sigma^2) noise to every element, during training only
     (eqs. 7-8). The result f feeds both branches below.
  3. Document tower: `layers` x [FC -> ReLU -> BN] (Fig. 1), no dropout; output h_out(x_i).
  4. Listwise context: `attn_layers` layers of multi-head self-attention (MHSA) over the list,
     each followed by LayerNorm (Sec. 4.2.2, eqs. 9-10), applied directly to f. Each head
     projects its input to head size z; the H heads are concatenated and projected to z by
     W_out. A residual connection is added when the input and output sizes match (layers
     2 onwards), "similarly to Vaswani et al." as the paper says.
  5. Latent cross: (1 + W a_i) * h_out(x_i)  (eq. 11; W projects a_i to the hidden size).
  6. Score: s_i = FC(ReLU(latent cross_i))  (eq. 13, Fig. 1).
  7. Loss: softmax cross-entropy on raw labels, -sum_i y_i log softmax(s)_i (Sec. 4.3),
     averaged over queries that have at least one relevant document. This weights each query
     by its total label, like TF-Ranking's softmax loss.
Paper tuning ranges (App. A): hidden {256..4096}, layers {3..6}, attention layers {3..6},
heads {2..5}, noise sigma in [0, 5] (1.5 best on Web30K, App. B.4).
Not given in the paper (my choices): attention head size z, optimizer details, batch size,
and the list length used in training.
History (2026-11-13): (1) attention moved from the hidden layer to the inputs, a bug fix;
(2) loss weighting, ReLU before the output layer, FC-ReLU-BN order without dropout, and
attention without an extra input projection or feed-forward sublayer, to match the paper.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def log1p_transform(X: np.ndarray, use_log1p: bool = True) -> np.ndarray:
    """eq. 6. Missing values (NaN, e.g. Istella's DBL_MAX sentinel) become 0 first."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(X) * np.log1p(np.abs(X)) if use_log1p else X


class MultiHeadSelfAttention(nn.Module):
    """eqs. 9-10: H heads, each projecting the input to head size z (W^Q, W^K, W^V),
    softmax(Q K^T / sqrt(z)) V per head, then concat -> W_out -> size z."""

    def __init__(self, in_dim: int, heads: int, head_size: int):
        super().__init__()
        self.h, self.z = heads, head_size
        self.qkv = nn.Linear(in_dim, 3 * heads * head_size)
        self.w_out = nn.Linear(heads * head_size, head_size)

    def forward(self, f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = f.shape
        q, k, v = self.qkv(f).view(B, L, 3, self.h, self.z).permute(2, 0, 3, 1, 4)  # (B, H, L, z)
        # Padded documents are masked out as keys, so they never affect real documents.
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return self.w_out(o.transpose(1, 2).reshape(B, L, self.h * self.z))


class DASALC(nn.Module):
    def __init__(self, n_features: int, hidden: int = 512, layers: int = 3, noise_std: float = 1.5,
                 use_attn: bool = True, attn_layers: int = 3, heads: int = 4, head_size: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.noise_std = noise_std
        self.use_attn = use_attn  # False = plain DNN baseline (no listwise context / latent cross)
        self.hidden = hidden
        self.in_bn = nn.BatchNorm1d(n_features, affine=False)
        blocks, d = [], n_features
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU(), nn.BatchNorm1d(hidden)]  # FC-ReLU-BN, Fig. 1
            if dropout > 0:  # not in the paper; off by default
                blocks.append(nn.Dropout(dropout))
            d = hidden
        self.tower = nn.Sequential(*blocks)
        if use_attn:
            dims = [n_features] + [head_size] * attn_layers
            self.attn = nn.ModuleList(MultiHeadSelfAttention(dims[i], heads, head_size)
                                      for i in range(attn_layers))
            self.attn_norm = nn.ModuleList(nn.LayerNorm(head_size) for _ in range(attn_layers))
            self.attn_proj = nn.Linear(head_size, hidden)  # "a_i goes through a linear projection"
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) padded lists, mask: (B, L) True for real docs. Returns scores (B, L)."""
        B, L, Fdim = x.shape
        f = self.in_bn(x[mask])  # only real docs go through BatchNorm
        if self.training and self.noise_std > 0:
            f = f + self.noise_std * torch.randn_like(f)
        h = x.new_zeros(B, L, self.hidden)
        h[mask] = self.tower(f)
        if self.use_attn:
            a = x.new_zeros(B, L, Fdim)
            a[mask] = f  # transformed (+noised) inputs, eq. 8
            for mhsa, norm in zip(self.attn, self.attn_norm):
                o = mhsa(a, mask)
                a = norm(o + a) if o.shape[-1] == a.shape[-1] else norm(o)
            h = h * (1 + self.attn_proj(a))  # latent cross, eq. 11
        s = self.out(F.relu(h)).squeeze(-1)  # eq. 13
        return s.masked_fill(~mask, float("-inf"))


def softmax_ce_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Paper Sec. 4.3: l = -sum_i y_i log softmax(s)_i with raw labels, averaged over queries
    that have at least one relevant document (queries without one carry no signal)."""
    labels = labels.masked_fill(~mask, 0)
    keep = labels.sum(1) > 0
    if not keep.any():
        return scores[mask].sum() * 0  # padded scores are -inf, and -inf * 0 = NaN
    logp = F.log_softmax(scores[keep], dim=1).masked_fill(~mask[keep], 0)
    return -(labels[keep] * logp).sum() / keep.sum()
