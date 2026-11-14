<!-- Update for BRC-A-20 "Datasets" — add the section below at the end (or above "Dataset creation" as the current state). The old split (train 1379 / test 344) and the old tile datasets (BRC-A-22, BRC-A-23) stay as history and should be marked superseded. -->

## 2026-11 cohort (current)

Rebuilt by [`scripts/make_cohort.py`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/scripts/make_cohort.py)
from the MOU Excel (morphology 8500, non-neoadjuvant, grade ≠ 9, T/N slide assignment), then filtered
to the 1735 slides that have embeddings (4 slides dropped for missing QC masks; all four classes still
present in every split). Labels CSV: `/mnt/projects/breast_cancer/brc_grading/labels.csv`
(`slide_id, grade, split`).

| split | N | G1 | G2 | G3 | total |
| --- | --- | --- | --- | --- | --- |
| train | 593 | 68 | 254 | 293 | 1208 |
| val | 126 | 16 | 55 | 64 | 261 |
| test | 131 | 14 | 58 | 63 | **266** |
| total | 850 | 98 | 367 | 420 | 1735 |

* **Case-disjoint, measured not assumed** ([`scripts/case_leakage.py`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/scripts/case_leakage.py)):
  1735 slides carry 1734 distinct case prefixes; 0 of 266 test and 0 of 261 val slides share a case with train.
* **5-fold cross-validation cohort** (the measurement ruler; single splits carry a ±2.6-point sampling
  noise floor and ~6-point seed spread): `/mnt/projects/breast_cancer/brc_grading/kfold/labels-kf{0..4}.csv`,
  each train 1174–1178 / val 291–295 / test 266, built by
  [`scripts/make_kfold_labels.py`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/scripts/make_kfold_labels.py).
* Tiles per slide are strongly class-correlated (p25 2.3k, median 6.5k, p75 12.3k, max 27k; class-0
  median 2.5k vs 9–11k for tumour grades) — log tile count alone scores 0.579 raw accuracy on test,
  so bag-length-dependent tricks are always checked against that baseline.
* G1 is 5.6% of training slides — the reason balanced accuracy is the headline metric.

**Superseded:** the old train-1379/test-344 split and the BRC-A-22/A-23 tile datasets (stride 112,
epithelium/tissue threshold filtering) are historical. Training now consumes whole-tissue Virchow2
embedding parquets (BRC-A-15, 2026-11 rebuild); the old experiment-37 models also differed in label
space (8-class) and are not comparable to anything current.
