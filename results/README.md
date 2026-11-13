# results/

- `lgbm/<dataset>_f<fold>.json`: LightGBM lambdarank baseline (`scripts/run_lgbm.py`).
  `istella_f1.json` ran on Kaggle (`kaggle/istella_lgbm/`).
- `dasalc/<dataset>_f<fold><tag>.json`: DASALC and plain-network runs (`scripts/run_dasalc.py`).
  Tags, oldest first:
  - `_prefix_bug`: first version, with attention on the hidden layer (a bug). Don't use.
  - `_dasalc_fix`, `_dasalc_fix_noise0`, `_dnn` (WEB10K) and `_fix_noise0` (WEB30K): after the
    attention bug fix, before the paper-matching changes.
  - `_v2_*`: the current paper-matched model (see the docstring of `src/ltr/dasalc.py`).
- `hand_check_mq2008.json`: the LambdaMART hand check (`scripts/hand_check_lambdas.py`).
- `log_*.txt`: training logs for the runs above.

Each JSON stores NDCG@1/5/10 under three rules for queries with no relevant document:
`_one` (LightGBM's rule), `_zero` (TF-Ranking's rule) and `_skip` (left out; used in the report).
