Predict the **Nottingham (Elston–Ellis) histologic grade** of breast-carcinoma whole-slide images (bc20-24 cohort): a 4-class task — **0 = no tumour epithelium (an `-N` slide), 1/2/3 = G1/G2/G3** — solved with whole-slide **attention-MIL over frozen Virchow2 tile embeddings**. Data restricted to non-neoadjuvant cases, morphology code 8500, grade 9 (not assessable) filtered out.

* **Code:** https://github.com/RationAI/breast-cancer-grading (uv + Hydra + PyTorch Lightning + MLflow, provenance-tracked)
* **Deep write-up:** [`REPORT.md`](https://github.com/RationAI/breast-cancer-grading/blob/master/REPORT.md) — every number, what it means, and the evidence behind it
* **Working state:** [`STATUS.md`](https://github.com/RationAI/breast-cancer-grading/blob/master/STATUS.md) · one-page summary: [`reports/latest_results.md`](https://github.com/RationAI/breast-cancer-grading/blob/master/reports/latest_results.md)

## Status (2026-11-14) — solution shipped

The model is the **average of 5 cross-validation fold models' probabilities** (attention pooling, head dropout 0.2, whole tissue bags, plain CE, no decision rule — no seed choice, no single-run selection). On the held-out **266-slide test split**, with zero test-set choices:

| balanced acc | AUROC | raw acc | κ (QWK) | recalls N/G1/G2/G3 |
|---|---|---|---|---|
| 0.7449 | 0.9528 | 0.8534 | 0.928 | 0.99 / 0.50 / 0.74 / 0.75 |

The tumour/no-tumour boundary is essentially solved (98% detection, 15 missed tumours in 750); the remaining error lives almost entirely on the **G1/G2/G3 boundary** — adjacent-grade ties, symmetric, i.e. label-noise shaped. An expert pathologist review of the 8 worst test errors found **only 1 real model mistake** (3 were label errors where the model was right), so these numbers are **lower bounds**.

* **Frozen artifact** (weights + sha256 manifest): `/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/`
* **Overlays** for all 266 test slides (xOpat, attention as grey veil): MLflow [experiment 82 "BRC Grading"](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/82), run `c9c975a3` → `test_grey.html`

**Pending (decisions, not compute):** external validation on BRACS (needs a pathologist to Elston-Ellis-grade the BRACS IC slides — the dataset ships no Nottingham grades; data card: BRC-A-43); label-noise measurement over all test errors; attention-vs-annotation test on the Breast MMCI tile parquets.

## Children

* **[Pipeline & Code](https://youtrack.rationai.cloud.e-infra.cz/articles/BRC-A-67/Pipeline-Code)** — how to run everything: stages, scripts, configs, jobs, reproduction
* **[Cohort & Labels](https://youtrack.rationai.cloud.e-infra.cz/articles/BRC-A-68/Cohort-Labels)** — bc20-24 cohort, splits, k-folds, label semantics and known label defects
* **[Methodology & Evaluation](https://youtrack.rationai.cloud.e-infra.cz/articles/BRC-A-69/Methodology-Evaluation)** — the model, training recipe, results, operating points, expert review, artifacts

*(Pre-2026-11 history — the tile-level model era, the old split and the old tiling — is archived in the previous Grading tree.)*
