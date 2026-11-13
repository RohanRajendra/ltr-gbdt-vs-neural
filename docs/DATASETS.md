# Dataset download guide

Put every download in the OneDrive datasets folder (linked as `data/raw`). Never commit data.

## Status
| Dataset | Status |
|---|---|
| Istella-S, Istella (full), Istella22, `test-svm-docIds.gz` | downloaded |
| MSLR-WEB10K / WEB30K | downloaded |
| MQ2007 / MQ2008 (LETOR 4.0) | downloaded |
| Yahoo! C14B Set 1 | downloaded (Hugging Face) |

## 1. MSLR-WEB10K and WEB30K
1. Open https://www.microsoft.com/en-us/research/project/mslr/
2. Accept the license agreement on the page.
3. Download `MSLR-WEB10K.zip` (~1.2 GB) and `MSLR-WEB30K.zip` (~3.7 GB).
4. Verify with `md5 <file>`:
   - WEB10K: `97c5d4e7c171e475c91d7031e4fd8e79`
   - WEB30K: `4beae4bee0cd244fc9b2aff355a61555`
5. Unzip. Each has `Fold1`..`Fold5`, each with `train.txt`, `vali.txt`, `test.txt`.

Microsoft's server can be very slow. Start this first and let it run.

## 2. MQ2007 / MQ2008 (LETOR 4.0)
1. Open https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/
2. Find "LETOR 4.0" and download `MQ2008.rar` and `MQ2007.rar`. Direct links may have moved, so use the page's links.
3. The extraction script (section 4) unpacks the `.rar` files with macOS's built-in `bsdtar`, so nothing extra needs installing.
4. Use the `Fold1..Fold5` folders. For supervised ranking, use the files with the 46 features.

Fallback if the links are dead: ask me and I'll look for a mirror.

## 3. Yahoo! Learning to Rank Challenge (C14B Set 1)
1. Open https://huggingface.co/datasets/YahooResearch/Yahoo-Learning-to-Rank-Challenge/tree/main
2. Download `yahoo-ltrc-c14-dataset.tgz` and `readme-yahoo-ltrc-c14-dataset.md`. The archive contains a nested `ltrc_yahoo.tar.bz2`.
3. Use Set 1 (~20K train, ~3K valid, ~7K test queries).

## 4. Extract
Run from the repo root:

    ./scripts/extract_datasets.sh

This checks the MSLR MD5s and unpacks each archive into `data/raw/<name>/`. It is safe to re-run, because finished datasets are skipped. Istella22's inner `.gz` files stay compressed.

In Finder, set the OneDrive `datasets` folder to "Always Keep on This Device" so files aren't moved to the cloud mid-training.
