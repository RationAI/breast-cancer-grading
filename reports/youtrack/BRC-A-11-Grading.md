<!-- Update for BRC-A-11 "Grading" (parent) — replace the intro paragraph with the text below, or append it as a "Status (2026-11)" section. -->

Develop a multiclass classifier to predict histologic grades (0–3) for Whole Slide Images (WSIs). The data is restricted to non-neoadjuvant breast cancer cases specifically categorized under morphology code 8500.

* Grades 1-3 correspond to Nottingham Grading system; grade 0 represents benign tissue (no cancer).

## Status (2026-11-14)

The solution is built and frozen. A **whole-slide attention-MIL over frozen Virchow2 tile embeddings**
(no epithelium pre-filtering — whole tissue bags) is trained with **stratified 5-fold cross-validation**,
and the shipped model is the **average of the 5 fold models' probabilities** — no single-run selection,
no decision rule. On the held-out 266-slide test split (zero test-set choices):

| balanced acc | AUROC | raw acc | κ (QWK) | recalls N/G1/G2/G3 |
|---|---|---|---|---|
| 0.7449 | 0.9528 | 0.8534 | 0.928 | 0.99 / 0.50 / 0.74 / 0.75 |

The tumour/no-tumour boundary is essentially solved (98% detection); the remaining error is almost
entirely the G1/G2/G3 boundary — adjacent-grade ties, symmetric, i.e. label-noise shaped. An expert
pathologist review of the 8 worst test errors found **only 1 real model mistake**; 3 were label errors
where the model was right.

* Artifact (weights + manifest): `/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/`
* Overlays for all 266 test slides (xOpat): MLflow [experiment 82 "BRC Grading"](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/82), run `c9c975a3` → Artifacts → `test_grey.html`
* Full write-up: [`REPORT.md`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/REPORT.md), one-page summary [`reports/latest_results.md`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/reports/latest_results.md)
* Updated children: **BRC-A-20** (datasets/cohort), **BRC-A-21** (methodology), **BRC-A-28** (evaluation).
* Pending: external validation on BRACS (needs a pathologist to grade the BRACS IC slides — the dataset
  ships no Nottingham grades), and a label-noise measurement over all test errors.
