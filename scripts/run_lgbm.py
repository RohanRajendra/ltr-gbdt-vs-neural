"""LightGBM lambdarank (LambdaMART) baseline on one dataset/fold.

Example:
    uv run python scripts/run_lgbm.py MQ2008 --fold 1 --leaves 31
Writes results/lgbm/<dataset>_f<fold>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ltr.data import REPO, load_split
from ltr.metrics import mean_ndcg

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("--fold", type=int, default=1)
p.add_argument("--leaves", type=int, default=255)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--min-leaf", type=int, default=50)
p.add_argument("--rounds", type=int, default=2000)
p.add_argument("--early-stop", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

t0 = time.time()
tr, va, te = (load_split(a.dataset, s, a.fold) for s in ("train", "vali", "test"))
t_load = time.time() - t0
print(f"{a.dataset} fold {a.fold}: train {tr.n_queries} q / {len(tr.y)} docs, "
      f"vali {va.n_queries} q, test {te.n_queries} q, {tr.X.shape[1]} feats  (load {t_load:.0f}s)")

params = dict(objective="lambdarank", metric="ndcg", eval_at=[5, 10],
              learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
              lambdarank_truncation_level=20, seed=a.seed, verbose=-1, num_threads=0)
dtr = lgb.Dataset(tr.X, tr.y, group=tr.group)
dva = lgb.Dataset(va.X, va.y, group=va.group, reference=dtr)

t0 = time.time()
model = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva], valid_names=["vali"],
                  callbacks=[lgb.early_stopping(a.early_stop, first_metric_only=True, verbose=False),
                             lgb.log_evaluation(100)])
t_train = time.time() - t0

t0 = time.time()
score = model.predict(te.X, num_iteration=model.best_iteration)
t_pred = time.time() - t0

res = dict(dataset=a.dataset, fold=a.fold, params=params, best_iter=model.best_iteration,
           train_sec=round(t_train, 1), predict_ms_per_query=round(1000 * t_pred / te.n_queries, 4),
           n_test_queries=te.n_queries, frac_test_queries_no_rel=float(np.mean(
               [te.y[s:s + n].max() == 0 for s, n in zip(np.cumsum(np.r_[0, te.group[:-1]]), te.group)])))
for k in (1, 5, 10):
    for empty in ("one", "zero", "skip"):
        res[f"ndcg@{k}_{empty}"] = round(mean_ndcg(te.y, score, te.group, k, empty), 4)

out = REPO / "results" / "lgbm" / f"{a.dataset}_f{a.fold}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "params"}, indent=2))
