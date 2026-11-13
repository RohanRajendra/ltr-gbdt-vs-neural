"""Hand-check of LambdaMART on one short MQ2008 query.

Check 1: |dNDCG| for swapping docs i and j, computed two ways:
    (a) directly: swap the two docs in the ranking, recompute NDCG
    (b) shortcut used by LambdaMART: (gain_i - gain_j) * |disc(r_i) - disc(r_j)| / maxDCG
Check 2: the per-document gradients (lambdas) and Hessians, computed by hand
    following LightGBM's lambdarank rules, against what LightGBM actually produces.

LightGBM 4.x lambdarank, per pair (high label h, low label l), sigma = 1:
    p        = 1 / (1 + exp(s_h - s_l))
    dNDCG    = |dNDCG(h, l)|, divided by (0.01 + |s_h - s_l|)   [lambdarank_norm=true]
    lambda_h -= p * dNDCG ;  lambda_l += p * dNDCG        (gradient = -lambda push)
    hess_h, hess_l += p * (1 - p) * dNDCG
then all lambdas and Hessians are rescaled by log2(1 + S) / S, S = sum of 2 * p * dNDCG.
To read LightGBM's numbers we train ONE tree, learning rate 1, no regularisation, with the
doc index as the only feature and one leaf per doc. Each leaf value is then -g_i / h_i.

Run: uv run python scripts/hand_check_lambdas.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from ltr.data import RESULTS, load_split
from ltr.metrics import dcg_at_k

QID = 10686
S0 = np.array([0.2, 0.9, -0.4, 0.5, -0.1, 0.3, 0.6])  # fixed starting scores (a poor ranking)

tr = load_split("MQ2008", "train", 1)
rows = tr.qid == QID
y = tr.y[rows].astype(int)
n = len(y)
assert n == len(S0)

gain = 2.0 ** y - 1
disc = 1 / np.log2(np.arange(n) + 2)          # discount at rank 1..n
max_dcg = dcg_at_k(np.sort(y.astype(float))[::-1], n)
order = np.argsort(-S0, kind="stable")
rank = np.empty(n, int)
rank[order] = np.arange(n)                     # 0-based rank of each doc


def ndcg_of(r):  # r: 0-based rank per doc
    return float(np.sum(gain * disc[r]) / max_dcg)


ndcg0 = ndcg_of(rank)

# ---- Check 1: dNDCG direct vs shortcut ----
pairs, worst1 = [], 0.0
for i in range(n):
    for j in range(i + 1, n):
        if y[i] == y[j]:
            continue
        r2 = rank.copy()
        r2[i], r2[j] = rank[j], rank[i]
        direct = abs(ndcg_of(r2) - ndcg0)
        shortcut = abs(gain[i] - gain[j]) * abs(disc[rank[i]] - disc[rank[j]]) / max_dcg
        worst1 = max(worst1, abs(direct - shortcut))
        pairs.append((i, j, direct, shortcut))

# ---- Check 2: lambdas by hand ----
g = np.zeros(n)
h = np.zeros(n)
S = 0.0
for i, j, d, _ in pairs:
    hi, lo = (i, j) if y[i] > y[j] else (j, i)
    ds = S0[hi] - S0[lo]
    dn = d / (0.01 + abs(ds))
    p = 1 / (1 + np.exp(ds))
    g[hi] -= p * dn
    g[lo] += p * dn
    h[hi] += p * (1 - p) * dn
    h[lo] += p * (1 - p) * dn
    S += 2 * p * dn
scale = np.log2(1 + S) / S
g *= scale
h *= scale
hand_step = -g / h                             # Newton step per doc = leaf value

# ---- LightGBM's own numbers ----
X = np.arange(n, dtype=float).reshape(-1, 1)
ds_lgb = lgb.Dataset(X, y, group=[n], init_score=S0,
                     params=dict(min_data_in_bin=1, max_bin=n + 1))
params = dict(objective="lambdarank", learning_rate=1.0, num_leaves=n, min_data_in_leaf=1,
              min_sum_hessian_in_leaf=0.0, lambda_l1=0.0, lambda_l2=0.0, min_gain_to_split=0.0,
              max_bin=n + 1, min_data_in_bin=1, lambdarank_truncation_level=n,
              lambdarank_norm=True, sigmoid=1.0, verbose=-1, deterministic=True)
bst = lgb.train(params, ds_lgb, num_boost_round=1)
lgb_step = bst.predict(X)                      # init_score is not included -> just the tree
leaves = bst.predict(X, pred_leaf=True).ravel()
# If LightGBM puts several docs in one leaf, its value is -(sum g)/(sum h) over those docs.
hand_leaf = np.array([-g[leaves == lf].sum() / h[leaves == lf].sum() for lf in leaves])
worst2 = float(np.max(np.abs(lgb_step - hand_leaf)))

print(f"query {QID}: labels {y.tolist()}, start scores {S0.tolist()}")
print(f"ranking by score (doc ids): {order.tolist()}  NDCG = {ndcg0:.4f}, maxDCG = {max_dcg:.4f}")
print("\nCheck 1: |dNDCG| direct vs shortcut")
for i, j, d, s in pairs:
    print(f"  docs ({i},{j}) labels ({y[i]},{y[j]}): direct {d:.6f}  shortcut {s:.6f}")
print(f"  max difference: {worst1:.2e}")
print("\nCheck 2: per-doc gradient, Hessian, and step -g/h")
print("  doc label score  grad(hand)  hess(hand)  leaf  leaf step(hand)  leaf step(LightGBM)")
for k in range(n):
    print(f"  {k:3d} {y[k]:5d} {S0[k]:5.1f}  {g[k]:+.6f}  {h[k]:.6f}  {leaves[k]:4d}  {hand_leaf[k]:+.6f}       {lgb_step[k]:+.6f}")
print(f"  distinct leaves used: {len(set(leaves))} of {n}")
print(f"  max |leaf step(hand) - leaf step(LightGBM)| = {worst2:.2e}")

out = RESULTS / "hand_check_mq2008.json"
out.write_text(json.dumps(dict(
    qid=QID, labels=y.tolist(), start_scores=S0.tolist(), ndcg_start=ndcg0, max_dcg=max_dcg,
    pairs=[dict(i=i, j=j, direct=d, shortcut=s) for i, j, d, s in pairs],
    max_diff_dndcg=worst1, grad_hand=g.tolist(), hess_hand=h.tolist(),
    step_hand=hand_step.tolist(), leaf=leaves.tolist(), leaf_step_hand=hand_leaf.tolist(), step_lightgbm=lgb_step.tolist(),
    max_diff_step=worst2, leaves=len(set(leaves))), indent=2))
print("saved", out)
