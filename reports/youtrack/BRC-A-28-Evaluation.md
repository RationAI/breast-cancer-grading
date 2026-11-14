<!-- Update for BRC-A-28 "Evaluation" — replace the whole article body with the text below. The old content (one-of-five model selection, tile-level model as primary, 344-slide test, experiment 37) is superseded. -->

# Evaluation (2026-11-14)

## Shipped model

5-fold average of one recipe (attention pooling, head dropout 0.2, whole bags, plain CE), **no decision
rule**. Weights: `/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/`
(sha256-verified manifest, git sha, per-fold label CSVs). Training runs: MLflow
[experiment 82 "BRC Grading"](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/82),
variants `kf0-cand` … `kf4-cand`. Per-slide predictions for all variants:
`…/brc_grading/variants/` (CSVs with per-class probabilities).

## Test results (266 held-out slides, zero test-set choices)

| balanced acc | AUROC | raw acc | κ (QWK) | recalls N/G1/G2/G3 | undercalls |
| --- | --- | --- | --- | --- | --- |
| 0.7449 | 0.9528 | 0.8534 | 0.928 | 0.99 / 0.50 / 0.74 / 0.75 | 25 / 266 (9.4%) |

Confusion matrix (rows = truth, columns = prediction):

| | N | G1 | G2 | G3 |
| --- | --- | --- | --- | --- |
| **N** | **130** | 0 | 0 | 1 |
| **G1** | 0 | **7** | 5 | 2 |
| **G2** | 0 | 9 | **43** | 6 |
| **G3** | 1 | 0 | 15 | **47** |

Out-of-fold (1469 slides) the same recipe scores balanced 0.7588 / raw 0.8530 / κ 0.924; 84% of the 216
errors are exactly one grade off, adjacent errors are symmetric (93 under / 89 over), and the four
adjacent pairs G3→G2, G2→G3, G2→G1, G1→G2 carry 82% of all error. The tumour/no-tumour boundary is
nearly free (98% detection, 15 missed tumours in 750); **the hard part is naming the grade inside
tumours — a label-noise-shaped, adjacent-grade-tie population.**

## Expert review of the worst errors (pathologist, 2026-11-14)

Of the 8 largest grade-jump errors: **3 label errors where the model was right** (a DCIS slide labelled
N; a multi-tumour slide carrying the wrong tumour's grade; a G2 slide that is G1), 3 defensible
label-semantics cases, 1 genuine borderline, **1 real model mistake**. The headline numbers above are
therefore **lower bounds**. Notes preserved in
[`reports/zprava_pro_patologa_komentare.md`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/reports/zprava_pro_patologa_komentare.md).

## Overlays (xOpat) and reporting

* All 266 test slides, attention as grey overlay: MLflow exp 82, run
  [`c9c975a3`](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/82/runs/c9c975a3/artifacts)
  → `test_grey.html` (same file on the PVC: `…/brc_grading/reports/test_grey.html`)
* Per-slide table: `masks/test/summary.csv`
* Report for the pathologist (Czech):
  [`reports/zprava_pro_patologa.md`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/reports/zprava_pro_patologa.md)
* One-page summary of what is shipped:
  [`reports/latest_results.md`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/reports/latest_results.md)

## Operating points (inference-time, no retrain)

A class-prior rule (λ ≈ 0.9) lifts balanced accuracy to 0.7959 but is **significantly worse slide by
slide** (McNemar p = 0.0135, ~15 new undercalls per one removed) — rejected. If "never report a slide
too low" becomes the instruction, escalation β ≈ 0.5 is the right knob (108 → 76 undercalls for 11
extra false alarms). Full price tables in `REPORT.md` §9.

## Comparison to literature

Vs. the ACMIL+UNI paper (Breast Cancer Res 2025, 10.1186/s13058-025-02019-4, 3-class on TCGA-BRCA,
macro-F1 0.731 / AUROC 0.835): restricted to our 135 tumour test slides we score macro AUROC 0.865
(G1/G2/G3 = 0.86/0.82/0.91 — ahead on every grade) but macro F1 0.658 — they name the grade better,
we rank it better, both gaps within single-split noise. External validation pending (BRACS — needs
pathologist grading, see BRC-A-11 status).

## Reproduction

```bash
# out-of-fold report for the fold runs (CPU pod, needs the PVC):
REPORT_JOB_NAME=brc-grading-oof-final REPORT_SCRIPT=ensemble_variants.py \
REPORT_VARIANTS=kf0-cand,kf1-cand,kf2-cand,kf3-cand,kf4-cand \
REPORT_ARGS='--oof --train-counts 575,67,247,286' uv run --no-sync python scripts/run_report.py
```
