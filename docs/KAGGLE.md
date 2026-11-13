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

       PYENV_VERSION=3.11.15 uvx kaggle datasets list -m

   An empty list with no error means the token works.
4. **Test a GPU notebook.** **Create → New Notebook**. In the right panel, set **Accelerator** to "GPU T4 x2" or "P100",
   turn **Internet** on, and run `!nvidia-smi`.

## Rules
- Upload our data only as a **private** Kaggle Dataset. The licenses (MSLR, Istella, Yahoo) are non-commercial and don't allow redistribution.
- Your weekly GPU quota and session time limit are shown in Settings. Check them before long runs.

## Next (commands given when needed)
- Upload `data/raw/istella/full` as a private dataset.
- Run `scripts/run_lgbm.py istella` in a CPU notebook with high RAM.
