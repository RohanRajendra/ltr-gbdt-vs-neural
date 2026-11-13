# GBDT vs. Neural Rankers, and Embeddings as Tree Features

CS6140 term project (Track 1: replicate and critique). Reproduces the LambdaMART vs. DASALC
comparison of Qin et al. (ICLR 2021) across six learning-to-rank benchmarks, then tests whether
ColBERTv2 signals work as LambdaMART features on Istella22.

## Layout
- `src/ltr/` – library code (loaders, models, metrics)
- `configs/` – per-dataset / per-model settings
- `results/` – metrics tables and plots (small files only)
- `docs/` – proposal, checkpoint report, dataset download notes
- `notebooks/` – exploration
- `data/` – git-ignored; see `data/README.md`

## Setup
    uv sync
    uv run python -c "import lightgbm, torch; print('ok')"
