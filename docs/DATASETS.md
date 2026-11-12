# Dataset download guide

Put every download in the OneDrive datasets folder (linked as `data/raw`). Never commit data.

## Status
| Dataset | Status |
|---|---|
| Istella-S, Istella (full), Istella22, `test-svm-docIds.gz` | downloaded (Istella22 archive not yet integrity-checked) |
| MSLR-WEB10K / WEB30K | TODO |
| MQ2007 / MQ2008 (LETOR 4.0) | TODO |
| Yahoo! C14B Set 1 | TODO (needs approval, may take days) |

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
3. Unpack with `brew install unar` then `unar MQ2008.rar`.
4. Use the `Fold1..Fold5` folders. For supervised ranking, use the files with the 46 features.

Fallback if the links are dead: ask me and I'll look for a mirror.

## 3. Yahoo! Learning to Rank Challenge (C14B Set 1)
1. Create or sign in to a Yahoo account.
2. Request access at https://research.yahoo.com/datasets (the old Webscope catalog is `webscope.sandbox.yahoo.com`).
3. Search for "Learning to Rank Challenge" (C14B). Use your Northeastern address and state academic, non-commercial use.
4. Wait for approval, then download the archive and use Set 1 (~20K train, ~3K valid, ~7K test queries).

I couldn't confirm the current approval process. If it isn't approved before the checkpoint, report it as "requested, pending".

## 4. Check what we already have
- `tar -tzf istella22.tar.gz > /dev/null` (several minutes) to confirm the 26 GB archive isn't truncated.
