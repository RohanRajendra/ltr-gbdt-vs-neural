# Kaggle setup

Kaggle gives free GPUs (T4 / P100, about 30 GPU-hours per week) and about 30 GB of RAM per notebook.
We need it for jobs that don't fit on the Mac (16 GB RAM), starting with full Istella.

## One-time setup
1. **Phone verification.** Profile picture → **Settings** → **Phone verification**.
   Without this, GPUs and internet in notebooks stay locked.
2. **API token.** Same Settings page → **API** → **Create New Token**. Kaggle shows the token
   and how to install it, either as an environment variable or as a file under `~/.kaggle/`.
   Follow what the page says. Older accounts got a `kaggle.json` download instead.
   Never paste the token into chat or commit it to git.

3. **Test it:**

       PYENV_VERSION=3.11.15 uvx --with six kaggle datasets list -m

   "No datasets found" means the token works. The `--with six` works around a missing dependency in the current kaggle package.
4. **Test a GPU notebook.** **Create → New Notebook**. In the right panel, set **Accelerator** to "GPU T4 x2" or "P100",
   turn **Internet** on, and run `!nvidia-smi`.

## Rules
- Upload our data only as a **private** Kaggle Dataset. The licenses (MSLR, Istella, Yahoo) are non-commercial and don't allow redistribution.
- Your weekly GPU quota and session time limit are shown in Settings. Check them before long runs.

## Job 1: LightGBM on full Istella
Full Istella's training split (7.3M rows) needs about 20 GB RAM at peak, which is more than the Mac has.

What gets uploaded: `data/kaggle/ltr-istella-full/` holds the gzipped `train.txt.gz` and `test.txt.gz`
(about 1.3 GB), a flat copy of the loader and metric code, and `dataset-metadata.json`.
It is git-ignored. If the code changes, re-copy it with `cp src/ltr/data.py src/ltr/metrics.py scripts/run_lgbm.py data/kaggle/ltr-istella-full/`.

1. Upload as a **private** dataset (the default; never pass `--public`):

       PYENV_VERSION=3.11.15 uvx --with six kaggle datasets create -p data/kaggle/ltr-istella-full

   Wait until the dataset page shows it as ready. Processing 1.3 GB can take a few minutes.
2. Push and start the job (a CPU script, no GPU needed):

       PYENV_VERSION=3.11.15 uvx --with six kaggle kernels push -p kaggle/istella_lgbm

3. Check progress:

       PYENV_VERSION=3.11.15 uvx --with six kaggle kernels status rohanrajendra12/istella-lgbm

4. When it says complete, download results:

       PYENV_VERSION=3.11.15 uvx --with six kaggle kernels output rohanrajendra12/istella-lgbm -p results/kaggle_istella

Expected time: about 30 min parsing plus 1 to 2 h training on Kaggle's 4 CPU cores.
The validation split is a seeded 10% of the official train queries, matching the paper's split
sizes; the paper's exact queries are unknown.
