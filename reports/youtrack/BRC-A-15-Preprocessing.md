<!-- Update for BRC-A-15 "Preprocessing" — add the section below at the end. The tissue/QC/epithelium mask documentation above stays as the (still valid) foundation; the 2026-11 rebuild replaces the tiling + adds embeddings. -->

## 2026-11 rebuild: tiling + embeddings (current pipeline)

The project repo was rebuilt (same GitLab project) and the chain ends at **per-slide Virchow2
embedding parquets**, which is what training consumes. All runs are in MLflow
[experiment 82 "BRC Grading"](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/82).

| step | MLflow | script |
| --- | --- | --- |
| Tiling | [run `bef4e85173d24c3992fb171429c71fd6`](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/82/runs/bef4e85173d24c3992fb171429c71fd6/artifacts) (dataset `artifacts/bc20-24`) | [`preprocessing/tiling`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/preprocessing/tiling/__main__.py) |
| Virchow2 embeddings | [run `96532a9ff29241e4b65d57dbc379e944`](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/82/runs/96532a9ff29241e4b65d57dbc379e944/artifacts) (all 1735 per-slide parquets) | [`preprocessing/embeddings.py`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/preprocessing/embeddings.py) |

Key differences from the tiling documented in BRC-A-22 (old, stride 112 with epithelium/tissue
threshold filtering):

* **224 px tiles at 0.5 µm/px, stride 224** (non-overlapping) — Virchow2's native input
* **no epithelium filtering**: MIL trains/evaluates on whole tissue bags. The old MIL flaw
  ("only epithelium tiles were selected — wrong for MIL, which looks at whole slides") is fixed by
  construction; tiles are filtered by tissue + QC only
* 4 slides excluded for missing QC masks (`2020_01089-02-T`, `2020_01166-01-N`, `2020_02743-02-T`,
  `2020_06097-01-N`) — listed in
  [`configs/experiment/preprocessing/embeddings/virchow2.yaml`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/configs/experiment/preprocessing/embeddings/virchow2.yaml)
* one parquet per slide (median bag 61 MB, p90 144 MB, max 336 MB; 2560-d Virchow2 features), cached on
  the PVC at `/mnt/projects/breast_cancer/brc_grading/embeddings/bc20-24/`

Every step is provenance-tracked (`log_provenance` + the training run's `ProvenanceCallback`), and each
is submitted as a kube-job (`scripts/run_tiling.py`, `scripts/run_embeddings.py`, …) — see the repo
README, "Preprocessing".
