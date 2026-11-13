"""Train DASALC on one dataset/fold; same splits and metric code as the LightGBM baseline.

Example:
    uv run python scripts/run_dasalc.py MQ2008 --fold 1 --epochs 30
Writes results/dasalc/<dataset>_f<fold>.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ltr.dasalc import DASALC, log1p_transform, softmax_ce_loss
from ltr.data import RESULTS, load_split
from ltr.metrics import mean_ndcg

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("--fold", type=int, default=1)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch", type=int, default=64, help="queries per batch")
p.add_argument("--list-size", type=int, default=200, help="docs sampled per query in training")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--hidden", type=int, default=512)
p.add_argument("--layers", type=int, default=3)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--noise", type=float, default=1.5)
p.add_argument("--patience", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--attn-layers", type=int, default=2)
p.add_argument("--no-attn", action="store_true", help="plain DNN: no self-attention / latent cross")
p.add_argument("--tag", default="", help="suffix for the results file name")
a = p.parse_args()

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def to_lists(split):
    X = log1p_transform(split.X).astype(np.float32)
    starts = np.r_[0, np.cumsum(split.group)[:-1]]
    return [(X[s:s + n], split.y[s:s + n]) for s, n in zip(starts, split.group)]


def batches(lists, bs, cap=None, shuffle=False):
    idx = rng.permutation(len(lists)) if shuffle else np.arange(len(lists))
    for i in range(0, len(idx), bs):
        chunk = []
        for j in idx[i:i + bs]:
            x, y = lists[j]
            if cap and len(y) > cap:
                pick = rng.choice(len(y), cap, replace=False)
                x, y = x[pick], y[pick]
            chunk.append((x, y))
        L = max(len(y) for _, y in chunk)
        X = np.zeros((len(chunk), L, chunk[0][0].shape[1]), np.float32)
        Y = np.zeros((len(chunk), L), np.float32)
        M = np.zeros((len(chunk), L), bool)
        for k, (x, y) in enumerate(chunk):
            X[k, :len(y)], Y[k, :len(y)], M[k, :len(y)] = x, y, True
        yield (torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev), torch.from_numpy(M).to(dev))


@torch.no_grad()
def predict(model, lists):
    model.eval()
    out = []
    for X, _, M in batches(lists, 32):
        s = model(X, M)
        out += [s[k][M[k]].cpu().numpy() for k in range(len(s))]
    return np.concatenate(out)


t0 = time.time()
tr, va, te = (load_split(a.dataset, s, a.fold) for s in ("train", "vali", "test"))
Ltr, Lva, Lte = to_lists(tr), to_lists(va), to_lists(te)
print(f"{a.dataset} f{a.fold} on {dev}: {len(Ltr)} train q, load {time.time() - t0:.0f}s")

model = DASALC(tr.X.shape[1], hidden=a.hidden, layers=a.layers, dropout=a.dropout,
               noise_std=a.noise, attn_layers=a.attn_layers, use_attn=not a.no_attn).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)

best, best_state, bad, t0 = -1.0, None, 0, time.time()
for ep in range(a.epochs):
    model.train()
    losses = []
    for X, Y, M in batches(Ltr, a.batch, cap=a.list_size, shuffle=True):
        if M.sum() < 2:
            continue
        loss = softmax_ce_loss(model(X, M), Y, M)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    v = mean_ndcg(va.y, predict(model, Lva), va.group, 5)
    print(f"epoch {ep + 1}: loss {np.mean(losses):.4f}  vali ndcg@5 {v:.4f}")
    if v > best:
        best, bad = v, 0
        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience:
            break
t_train = time.time() - t0

model.load_state_dict(best_state)
t0 = time.time()
score = predict(model, Lte)
t_pred = time.time() - t0

res = dict(dataset=a.dataset, fold=a.fold, args=vars(a), device=dev, epochs_run=ep + 1,
           best_vali_ndcg5=round(best, 4), train_sec=round(t_train, 1),
           predict_ms_per_query=round(1000 * t_pred / te.n_queries, 4))
for k in (1, 5, 10):
    for empty in ("one", "zero", "skip"):
        res[f"ndcg@{k}_{empty}"] = round(mean_ndcg(te.y, score, te.group, k, empty), 4)
out = RESULTS / "dasalc" / f"{a.dataset}_f{a.fold}{a.tag}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=2))
