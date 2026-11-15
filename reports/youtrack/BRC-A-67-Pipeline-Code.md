How to run the grading solution end to end. Repo: https://github.com/RationAI/breast-cancer-grading — built on the RationAI machine-learning template (**uv** + **Hydra** + **PyTorch Lightning** + **MLflow**, provenance-tracked via [rationai-provenance](https://github.com/RationAI/provenance)). Every run logs to MLflow **experiment 82 "BRC Grading"** (https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/82) and carries provenance (user, git sha, dataset, image, hardware); an unregistered user fails the run at startup.

## Pipeline stages, in order

| stage | what it does | local command | kube-jobs submitter |
|---|---|---|---|
| tissue masks | ray-based tissue detection | `uv run -m preprocessing.tissue_masks` | `scripts/run_tissue_masks.py` |
| QC masks | blur / folding / residual via the QC service | `uv run -m preprocessing.quality_control` | `scripts/run_qc.py` |
| tiling | **224 px @ 0.5 µm/px, stride 224** (non-overlapping); per-tile tissue+QC overlaps; no epithelium filtering — MIL sees whole tissue bags | `uv run -m preprocessing.tiling +experiment=preprocessing/tiling/bc20_24_224` | `scripts/run_tiling.py` |
| embeddings | frozen **Virchow2** → one parquet per slide (2560-d features); all 1735 slides cached on the PVC | `uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings/virchow2` | `uv run --extra job python scripts/run_embeddings.py` (uses the huggingface-secret) |
| training | attention-MIL head over the bags | `uv run -m ml.train +experiment=ml/mil_attention` | `scripts/run_train.py` |
| scoring / reports | fold averaging, OOF reports, operating points | see `scripts/ensemble_variants.py` | `scripts/run_report.py` |
| overlays | attention BigTIFFs + clickable xOpat html | — | `scripts/run_masks.py` |

Key MLflow runs: tiling `bef4e85173d24c3992fb171429c71fd6`, embeddings `96532a9ff29241e4b65d57dbc379e944`, test overlays `c9c975a3` (→ `test_grey.html`). Four slides are excluded for missing QC masks (`2020_01089-02-T`, `2020_01166-01-N`, `2020_02743-02-T`, `2020_06097-01-N`) — the list lives in [configs/experiment/preprocessing/embeddings/virchow2.yaml](https://github.com/RationAI/breast-cancer-grading/blob/master/configs/experiment/preprocessing/embeddings/virchow2.yaml).

## Labels CSV contract

`GradingDataset` reads one CSV ([configs/data/grading.yaml](https://github.com/RationAI/breast-cancer-grading/blob/master/configs/data/grading.yaml)):

| column | meaning |
|---|---|
| `slide_id` | matches `<slide_id>.parquet` in the embeddings dir |
| `grade` | 0 (N) / 1 (G1) / 2 (G2) / 3 (G3) |
| `split` | `train` / `val` / `test` |

## Scripts

| script | purpose |
|---|---|
| `make_cohort.py` | build `labels.csv` from the MOU Excel (`--require-embeddings` keeps it consistent with what exists) |
| `make_kfold_labels.py` | write `kfold/labels-kf{0..4}.csv` (5-fold CV cohort) |
| `case_leakage.py` | verify splits are case-disjoint (measured, not assumed) |
| `run_train.py` | submit MIL training; resumable (`last.ckpt` on the PVC), 12 CPU / 48 Gi / 8 workers |
| `ensemble_variants.py` | fold averaging, out-of-fold reports, λ/β operating points (McNemar) |
| `freeze_artifact.py` | freeze weights + manifest.json (sha256, git sha, recipe); refuses to overwrite |
| `run_masks.py` / `attention_masks.py` / `xopat_report.py` | attention overlays (BigTIFF), per-slide focus stats, clickable `test_grey.html` |
| `all_runs_table.py` | every MLflow run with recalls, from the laptop |
| `report_variants.py` / `run_report.py` | per-variant prediction CSVs; run report scripts on the cluster |
| `create_user.py` / `register_dataset.py` / `register_embeddings.py` | provenance registration (see README) |
| `make_demo_data.py` | synthetic quickstart (CPU, no cluster) |

## Quickstart & reproduction

```bash
uv sync                                   # or: uv sync --extra job  (kube-jobs submission)
uv run python scripts/make_demo_data.py   # synthetic data
uv run -m ml.train +experiment=ml/demo    # CPU pipeline check
```

Full out-of-fold report for the shipped fold runs (from a CPU pod with the PVC):

```bash
REPORT_JOB_NAME=brc-grading-oof-final REPORT_SCRIPT=ensemble_variants.py \
REPORT_VARIANTS=kf0-cand,kf1-cand,kf2-cand,kf3-cand,kf4-cand \
REPORT_ARGS='--oof --train-counts 575,67,247,286' uv run --no-sync python scripts/run_report.py
```

Scoring a new cohort with the frozen artifact: run `ml.train` per checkpoint with `checkpoint=… dataset.paths.labels=<cohort csv> eval.split=test`, then average the per-slide softmaxes — the same averaging `scripts/ensemble_variants.py` does. No new code needed.

## Operational notes (each cost a failed job — see STATUS.md)

* bag decoding (~61 MB parquet/slide) dominates the ~10 ms GPU step → throughput scales with dataloader workers; 8 workers + persistent workers, fp16 bag cache
* a pod whose training died must not report `Succeeded` (guarded in the submitter); `EMB_CACHE_GB` is per-worker-per-split, not a total
* trust logged params, never a variant name — early "t2048-*" arms trained whole bags because `max_tiles` was never passed
