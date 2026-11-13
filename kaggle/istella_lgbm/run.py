"""Kaggle job: LightGBM lambdarank on full Istella.

Expects the private dataset rohanrajendra12/ltr-istella-full, which holds
train.txt and test.txt (Kaggle unzips the uploaded .gz files) and a flat copy of ltr/data.py, ltr/metrics.py, run_lgbm.py.
Results land in /kaggle/working/results/lgbm/istella_f1.json (download from the Output tab).
"""
import glob
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

# Find the dataset wherever Kaggle mounted it.
hits = glob.glob("/kaggle/input/**/run_lgbm.py", recursive=True)
assert hits, "dataset not attached: add rohanrajendra12/ltr-istella-full as input"
src = Path(hits[0]).parent
print("dataset at", src)

work = Path("/kaggle/working")
pkg = work / "code" / "ltr"
pkg.mkdir(parents=True, exist_ok=True)
for f in ("__init__.py", "data.py", "metrics.py"):
    shutil.copy(src / f, pkg / f)

raw = work / "raw" / "istella" / "full"
raw.mkdir(parents=True, exist_ok=True)
# Kaggle decompresses .gz on upload, so accept either train.txt or train.txt.gz.
for stem in ("train.txt", "test.txt"):
    found = [src / n for n in (stem, stem + ".gz") if (src / n).exists()]
    assert found, f"{stem} missing in {src}: {sorted(p.name for p in src.iterdir())}"
    link = raw / found[0].name
    if not link.exists():
        os.symlink(found[0], link)
    print("linked", link, "->", found[0])

os.environ.update(LTR_DATA_ROOT=str(work / "raw"), LTR_RESULTS_DIR=str(work / "results"),
                  LTR_CACHE="0")  # no .npz cache: saves disk and the time to write 7 GB
sys.path.insert(0, str(work / "code"))
subprocess.run(["free", "-g"])
subprocess.run(["nproc"])

sys.argv = ["run_lgbm.py", "istella", "--fold", "1"]
runpy.run_path(str(src / "run_lgbm.py"), run_name="__main__")
