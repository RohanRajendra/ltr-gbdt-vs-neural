"""NDCG@k computed by hand, so the definition is explicit and checkable.

gain = 2^rel - 1, discount = 1 / log2(rank + 1), ranks start at 1.
Queries with no relevant document have IDCG = 0. How to score them is a
convention that changes the reported number, so it is a parameter:
  - "one":  count as NDCG = 1 (LightGBM's default behaviour)
  - "zero": count as NDCG = 0
  - "skip": drop them from the average
Ties in predicted score are broken by original order (stable sort).
"""
from __future__ import annotations

import numpy as np


def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    return float(np.sum((2.0 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_per_query(y: np.ndarray, score: np.ndarray, group: np.ndarray, k: int) -> np.ndarray:
    """Returns NDCG@k per query; NaN where the query has no relevant docs."""
    out = np.empty(len(group))
    start = 0
    for i, n in enumerate(group):
        yy, ss = y[start:start + n], score[start:start + n]
        start += n
        idcg = dcg_at_k(np.sort(yy)[::-1], k)
        if idcg == 0:
            out[i] = np.nan
            continue
        order = np.argsort(-ss, kind="stable")
        out[i] = dcg_at_k(yy[order], k) / idcg
    return out


def mean_ndcg(y, score, group, k: int, empty: str = "one") -> float:
    v = ndcg_per_query(y, score, group, k)
    if empty == "skip":
        return float(np.nanmean(v))
    return float(np.mean(np.nan_to_num(v, nan=1.0 if empty == "one" else 0.0)))
