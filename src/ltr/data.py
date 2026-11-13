"""Loading LTR datasets (SVMlight / LETOR format) into dense float32 arrays.

Every dataset is a list of (label, qid, features) rows, grouped by query.
We return X (n_docs x n_features), y (n_docs,), qid (n_docs,), and
group sizes (n_queries,) in the order the queries appear in the file.
Parsed splits are cached as .npz under data/cache/ for fast reloads.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
CACHE = REPO / "data" / "cache"

# name -> (n_features, {split: relative path template}); {fold} filled in where used
DATASETS: dict[str, tuple[int, dict[str, str]]] = {
    "MQ2007": (46, {s: "MQ2007/Fold{fold}/%s.txt" % s for s in ("train", "vali", "test")}),
    "MQ2008": (46, {s: "MQ2008/Fold{fold}/%s.txt" % s for s in ("train", "vali", "test")}),
    "MSLR-WEB10K": (136, {s: "MSLR-WEB10K/Fold{fold}/%s.txt" % s for s in ("train", "vali", "test")}),
    "MSLR-WEB30K": (136, {s: "MSLR-WEB30K/Fold{fold}/%s.txt" % s for s in ("train", "vali", "test")}),
    "yahoo-set1": (700, {"train": "yahoo/set1.train.txt", "vali": "yahoo/set1.valid.txt",
                         "test": "yahoo/set1.test.txt"}),
    "istella-s": (220, {"train": "istella-s/sample/train.txt", "vali": "istella-s/sample/vali.txt",
                        "test": "istella-s/sample/test.txt"}),
    "istella22": (220, {"train": "istella22/train.svm.gz", "vali": "istella22/valid.svm.gz",
                        "test": "istella22/test.svm.gz"}),
}


@dataclass
class Split:
    X: np.ndarray       # float32, (n_docs, n_features)
    y: np.ndarray       # float32 relevance labels
    qid: np.ndarray     # int64 query ids
    group: np.ndarray   # int64 docs-per-query, in file order

    @property
    def n_queries(self) -> int:
        return len(self.group)


def _groups(qid: np.ndarray) -> np.ndarray:
    """Docs per query, assuming rows of one query are contiguous (true for all these files)."""
    change = np.flatnonzero(np.diff(qid)) + 1
    bounds = np.concatenate([[0], change, [len(qid)]])
    sizes = np.diff(bounds)
    if len(np.unique(qid)) != len(sizes):
        raise ValueError("query rows are not contiguous")
    return sizes


def load_split(name: str, split: str, fold: int = 1, use_cache: bool = True) -> Split:
    n_features, paths = DATASETS[name]
    path = RAW / paths[split].format(fold=fold)
    tag = f"{name}_f{fold}_{split}" if "{fold}" in paths[split] else f"{name}_{split}"
    cache = CACHE / f"{tag}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return Split(d["X"], d["y"], d["qid"], d["group"])

    f = gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")
    with f:
        X, y, qid = load_svmlight_file(f, n_features=n_features, query_id=True,
                                       dtype=np.float32, zero_based=False)
    X = X.toarray()
    # Istella uses DBL_MAX (1.8e308) as a sentinel; it overflows float32 to inf.
    X[~np.isfinite(X)] = np.nan
    s = Split(X, y.astype(np.float32), qid.astype(np.int64), _groups(qid))
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=s.X, y=s.y, qid=s.qid, group=s.group)
    return s
