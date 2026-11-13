"""Loading LTR datasets (SVMlight / LETOR format) into dense float32 arrays.

Every dataset is a list of (label, qid, features) rows, grouped by query.
We return X (n_docs x n_features), y (n_docs,), qid (n_docs,), and
group sizes (n_queries,) in the order the queries appear in the file.
Parsed splits are cached as .npz under data/cache/ for fast reloads.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file

import os

REPO = Path(__file__).resolve().parents[2]
# Overridable so the same code runs on Kaggle (read-only inputs, writable /kaggle/working).
RAW = Path(os.environ.get("LTR_DATA_ROOT", REPO / "data" / "raw"))
CACHE = Path(os.environ.get("LTR_CACHE_DIR", REPO / "data" / "cache"))
RESULTS = Path(os.environ.get("LTR_RESULTS_DIR", REPO / "results"))

# Full Istella ships only train/test. Qin et al. (Table 1) hold out 2,318 of 23,219 train
# queries (~10%) for validation; their exact queries are unknown, so we use a seeded 10% split.
ISTELLA_VALI_FRAC = 0.10
ISTELLA_VALI_SEED = 0

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
    # "trainfull" is the official train file; "train"/"vali" are carved from it (see above).
    "istella": (220, {"trainfull": "istella/full/train.txt", "test": "istella/full/test.txt"}),
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


_FULL: dict = {}
CHUNK_LINES = 500_000


def _parse_chunked(path: Path, n_features: int):
    """Parse an SVMlight file in chunks of lines, densifying each chunk.
    Parsing a whole 11 GB file at once builds sparse temp arrays ~2-3x the dense size;
    chunking keeps peak memory near 2x the final dense array."""
    Xs, ys, qs = [], [], []

    def flush(buf):
        X, y, q = load_svmlight_file(io.BytesIO(b"".join(buf)), n_features=n_features,
                                     query_id=True, dtype=np.float32, zero_based=False)
        Xs.append(X.toarray()); ys.append(y); qs.append(q)

    f = gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")
    with f:
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) == CHUNK_LINES:
                flush(buf)
                buf = []
        if buf:
            flush(buf)
    X = np.concatenate(Xs) if len(Xs) > 1 else Xs[0]
    del Xs
    # Istella uses DBL_MAX (1.8e308) as a sentinel; it overflows float32 to inf.
    X[~np.isfinite(X)] = np.nan
    return X, np.concatenate(ys), np.concatenate(qs)


def _subset(s: Split, query_mask: np.ndarray) -> Split:
    row_mask = np.repeat(query_mask, s.group)
    return Split(s.X[row_mask], s.y[row_mask], s.qid[row_mask], s.group[query_mask])


def load_split(name: str, split: str, fold: int = 1, use_cache: bool = True) -> Split:
    use_cache = use_cache and os.environ.get("LTR_CACHE", "1") != "0"
    n_features, paths = DATASETS[name]
    if split not in paths and "trainfull" in paths and split in ("train", "vali"):
        key = (name, fold)
        if key not in _FULL:  # parse the big file once per process, not once per split
            _FULL[key] = load_split(name, "trainfull", fold, use_cache)
        full = _FULL[key]
        rng = np.random.default_rng(ISTELLA_VALI_SEED)
        is_vali = rng.random(full.n_queries) < ISTELLA_VALI_FRAC
        if split == "vali":  # scripts load train then vali; free the big copy afterwards
            _FULL.pop(key)
        return _subset(full, is_vali if split == "vali" else ~is_vali)
    path = RAW / paths[split].format(fold=fold)
    if not path.exists() and Path(str(path) + ".gz").exists():  # Kaggle copy is gzipped
        path = Path(str(path) + ".gz")
    tag = f"{name}_f{fold}_{split}" if "{fold}" in paths[split] else f"{name}_{split}"
    cache = CACHE / f"{tag}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return Split(d["X"], d["y"], d["qid"], d["group"])

    X, y, qid = _parse_chunked(path, n_features)
    s = Split(X, y.astype(np.float32), qid.astype(np.int64), _groups(qid))
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=s.X, y=s.y, qid=s.qid, group=s.group)
    return s
