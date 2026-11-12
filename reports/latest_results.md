# Latest results — what we are using

**12. 11. 2026 · one-page summary.** Full evidence and definitions: [`REPORT.md`](../REPORT.md).
Ops facts: [`STATUS.md`](../STATUS.md).

## What we are using

**A 5-fold-averaged MIL (multiple-instance learning) classifier for the Nottingham grade.**

| | |
|---|---|
| **Task** | 4 classes: **N** (no tumour epithelium) / **G1** / **G2** / **G3** on whole slides |
| **Input** | per-tile Virchow2 embeddings (one parquet per slide), whole bags — no tile cap |
| **Architecture** | attention pooling (not mean pooling), head dropout **0.2** |
| **Training** | plain cross-entropy, 5-fold cross-validation, patience 5 |
| **Inference** | average of the **5 fold models'** softmax probabilities — no seed choice, no decision rule (λ = 0) |
| **Frozen artifact** | `/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/` (five `best.ckpt` + `manifest.json`) |
| **Also from these weights** | the xOpat attention overlays and the pathologist report — same model, same numbers |

## Latest results — shipped artifact, 266-slide test split (zero test-set choices)

| balanced accuracy | AUROC | raw accuracy | quadratic κ | undercalls |
|---|---|---|---|---|
| **0.7449** | **0.9528** | **0.8534** | **0.928** | 25 / 266 (9.4 %) |

Recall per class (from the actual per-slide predictions of the shipped ensemble):

| | N | G1 | G2 | G3 |
|---|---|---|---|---|
| recall | **0.99** (130/131) | **0.50** (7/14) | **0.74** (43/58) | **0.75** (47/63) |

- **Raw accuracy 85.3 % is inflated** by the 131 tumour-free slides (99 % correct).
  On the 135 tumour-bearing slides the grade is right in **72 %**.
- G1 has only 14 test slides — one slide moves its recall 7 points, so G1 ≈ 50 % is
  a small-sample number, not necessarily a wall.
- In plain language: nearly always separates tumour-free from tumour (98 %+),
  ranking is strong (AUROC 0.95), naming the exact grade — especially G1 — is the weak part.

## What we rejected recently (why the numbers above did not move)

| experiment | result | verdict |
|---|---|---|
| prior rule λ ≈ 0.9 | +5.0 balanced accuracy, but **significantly worse per slide** (McNemar p = 0.0135), ~15 new undercalls per one removed | rejected — ship λ = 0 |
| bag cap 1024 / 4096 (newest MLflow runs, e.g. `kf1_tf4096` κ 0.912) | whole bags win: −1.6 (p = 0.062) / **−3.0 (p = 0.0008)** | rejected — whole bags stay |
| ordinal cross-entropy, class-weighted CE, learned boundary, mean pooling | no gain / worse (REPORT.md §6–§8, full tables) | rejected |

## Where to look

- **MLflow**: experiment **82 „BRC Grading"** (`mlflow-s3.rationai-mlflow`) — graded runs carry
  `test/kappa`; the report pages are runs named `MIL overlay report …` (newest: `c9c975a3`).
- **REPORT.md** — every metric's definition, noise floor (±2.6 points on the test split), and the
  graveyard of rejected arms.

> Note: `REPORT.md` §4 prints recalls `0.98 / 0.55 / 0.69 / 0.83` in the shipped row — that column is
> stale (it does not average to the stated balanced accuracy 0.7449). The table above is recomputed
> from the shipped ensemble's per-slide predictions and matches the confusion matrix the pathologist
> report was built from.
