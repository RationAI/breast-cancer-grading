## The cohort (bc20-24, 2026-11)

Rebuilt by [`scripts/make_cohort.py`](https://github.com/RationAI/breast-cancer-grading/blob/master/scripts/make_cohort.py) from the MOU Excel (morphology 8500, non-neoadjuvant, grade ≠ 9, T/N slide assignment), filtered to the 1735 slides that have embeddings (4 dropped for missing QC masks; all classes still in every split). Labels CSV: `/mnt/projects/breast_cancer/brc_grading/labels.csv`.

| split | N | G1 | G2 | G3 | total |
|---|---|---|---|---|---|
| train | 593 | 68 | 254 | 293 | 1208 |
| val | 126 | 16 | 55 | 64 | 261 |
| test | 131 | 14 | 58 | 63 | **266** |
| total | 850 | 98 | 367 | 420 | 1735 |

* **Label semantics:** `grade` **0 = no tumour epithelium** (an `-N` slide), 1/2/3 = Nottingham G1/G2/G3. `-T` slides carry the case's Nottingham grade, `-N` slides are class 0. (An older note calling the classes "B-RAD grades 2–5" was wrong — counts unaffected, names were.)
* **Case-disjoint, measured not assumed** ([`scripts/case_leakage.py`](https://github.com/RationAI/breast-cancer-grading/blob/master/scripts/case_leakage.py)): 1735 slides carry 1734 distinct case prefixes; 0 of 266 test and 0 of 261 val slides share a case with train.
* **5-fold CV cohort** (the measurement ruler — single splits carry a ±2.6-point sampling noise floor and ~6-point seed spread): `/mnt/projects/breast_cancer/brc_grading/kfold/labels-kf{0..4}.csv`, each train 1174–1178 / val 291–295 / test 266, built by [`scripts/make_kfold_labels.py`](https://github.com/RationAI/breast-cancer-grading/blob/master/scripts/make_kfold_labels.py).
* Tiles per slide are strongly class-correlated (p25 2.3k, median 6.5k, p75 12.3k, max 27k; class-0 median 2.5k vs 9–11k for tumour grades) — log tile count *alone* scores 0.579 raw accuracy on test, so bag-length-dependent tricks are always checked against that baseline.
* **G1 is 5.6% of training slides** — why balanced accuracy is the headline metric and why G1 recall (14 test slides, 7.1-point steps) is the least stable number reported.

## Label defects found by expert review (2026-11-14)

The pathologist reviewed the 8 worst test errors (notes: [`reports/zprava_pro_patologa_komentare.md`](https://github.com/RationAI/breast-cancer-grading/blob/master/reports/zprava_pro_patologa_komentare.md)). Verdicts: 3 outright label errors (the model was right), 3 defensible label-semantics cases, 1 genuine borderline, only **1 real model mistake**. Concrete defects for a cohort v2:

| defect | example | note |
|---|---|---|
| multi-tumour grade assignment | `2021_02976-08-T1` | a breast with G1+G2+G3 tumours; the slide is the G3 one but carries the G1 label |
| morphological code mismatch (8500 vs 8480) | `2021_00353-21-T2` | mucinous carcinoma (8480) labelled with the G3 of an 8500 ductal; should be G2 |
| **in-situ lesions (DCIS) inside class N** | `2021_01085-02-N` | labelled N but contains DCIS — the model called it G3 and was right; class N is not "tumour-free" by this definition |
| Nottingham score vs grade inconsistency | `2020_00168-02-T` | score 7 = G2, labelled G3 (undercounted mitoses) |

**Implications:** reported accuracies are **lower bounds** — part of the measured "model error" is reference-standard noise (published Nottingham inter-reader agreement is ~38–77% exact). A v2 cohort must re-grade **blind to the model's predictions** — fixing only the slides the model got wrong would contaminate the test set.

## External data

* **BRACS** (public, 547 WSIs, subtypes N/PB/UDH/FEA/ADH/DCIS/IC, **no Nottingham grades**): data card BRC-A-43. Plan: embed with our own pipeline (0.5 mpp rescales + QC + tissue masks already exist), run the frozen artifact, test N-vs-rest + DCIS behaviour label-free; the paper-comparable grading validation needs a pathologist to grade the IC slides.
* MammaPrint / Breast MMCI Full Test: **not usable as grading benchmarks** (no joinable grades; 36/100 MMCI negatives already in train; different mpp/stride) — details in [`REPORT.md` §11](https://github.com/RationAI/breast-cancer-grading/blob/master/REPORT.md).
