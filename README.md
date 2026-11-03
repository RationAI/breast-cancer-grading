# Breast Cancer Grading (BRC)

Histopathological **grading of breast carcinoma** from whole-slide images (bc20-24
cohort): a 4-class multiple-instance-learning (MIL) task over Virchow2 tile
embeddings. Built on the [machine-learning](https://github.com/RationAI/machine-learning/tree/provenance)
template (PyTorch Lightning + Hydra + MLflow + uv, provenance-tracked).

## Task

Predict the **Nottingham (Elston–Ellis) grade** of each slide:

| label | class                     | ~frequency (bc20-24) |
| ----- | ------------------------- | --------------------- |
| 0     | no tumour epithelium (N)  | 50%                   |
| 1     | G1 — well differentiated  | 5%                    |
| 2     | G2 — moderately diff.     | 20%                   |
| 3     | G3 — poorly differentiated| 25%                   |

Class-imbalanced, ordered classes → macro AUROC/F1 + **quadratic weighted kappa** as
headline metrics, class-balanced sampling during training. Each slide is a *bag* of
tile embeddings; per-tile attention from the aggregator doubles as an
interpretability signal (tile-level heatmaps), which is the end goal — see the
[Karolinska approach](https://www-nature-com.proxy.ics.muni.cz/articles/s41516-025-00882-2)
of explaining a genomic-assay model (MammaPrint) through a morphological grading
model, and the YouTrack **BRC** project for the full task definition, data
provenance and evaluation protocol (BRC-A-2 … BRC-A-28).

## Installation

```bash
uv sync            # or: uv sync --extra job   (adds kube-jobs submission)
```

## Quickstart — verify the pipeline on synthetic data (CPU, no cluster)

```bash
uv run python scripts/make_demo_data.py   # data/demo/{labels.csv, embeddings/}
uv run -m ml.train +experiment=ml/demo    # trains up to 60 epochs, tests best.ckpt
```

## Usage

```bash
uv run -m ml.train +experiment=ml/mil_attention                 # fit (real data)
uv run -m ml.train +experiment=ml/mil_attention mode=test checkpoint=best.ckpt
uv run -m ml.train +experiment=ml/mil_attention mode=predict checkpoint=best.ckpt
```

`mode` selects the Lightning stage (`fit|validate|test|predict`). After `fit`,
the best checkpoint is automatically tested on the held-out split
(`test_after_fit: true`). All configuration is Hydra-based — override anything on
the command line (e.g. `ml.aggregator.attention_dim=256 trainer.max_epochs=50`).

### Experiment configs

- `configs/experiment/ml/demo.yaml` — synthetic data, CPU, pipeline check.
- `configs/experiment/ml/mil_attention.yaml` — attention MIL on Virchow2
  embeddings, the grading baseline. Swap `/ml/aggregator: mean` for the
  parameter-free baseline.
- `configs/experiment/preprocessing/…` — tiling / embeddings runs.

## Preprocessing (real bc20-24 data)

Follows the BRC methodology: MOU slides → tissue masks → QC masks (folding,
blur, residual) → optional epithelium masks → tiling at 0.5 µm/px with 224 px
tiles (Virchow2) → per-slide Virchow2 embeddings.

```bash
uv run -m preprocessing.tissue_masks        # ray tissue masks -> MLflow
uv run -m preprocessing.quality_control      # QC masks via the QC service
uv run -m preprocessing.tiling +experiment=preprocessing/tiling/mou_224
uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings/virchow2
```

Each step logs artifacts to MLflow and calls `log_provenance`. The tiling run
prints the dataset URI for the embeddings run; paste it into
`configs/experiment/preprocessing/embeddings/virchow2.yaml` (`dataset.path`).

### Labels

`GradingDataset` reads one labels CSV (`configs/data/grading.yaml` →
`dataset.paths.labels`) with columns:

| column    | meaning                                            |
| --------- | -------------------------------------------------- |
| `slide_id`| matches `<slide_id>.parquet` in the embeddings dir  |
| `grade`   | 0 (N) / 1 (G1) / 2 (G2) / 3 (G3)                   |
| `split`   | `train` / `val` / `test` — one dataset, split column (BRC-A-20) |

The cohort comes from the bc20-24 MOU Excel (morphology-filtered, non-neoadjuvant,
grade ≠ 9; slides per case split T/N — see BRC-A-5, BRC-A-14, BRC-A-15). Splits are
**stratified by grade and grouped by case** — slides of one case never span splits
(BRC-A-16).

## Provenance

Every run in this repo is provenance-tracked via [rationai-provenance](https://github.com/RationAI/provenance):
preprocessing apps call `log_provenance`, and the training run gets the
`ProvenanceCallback` (wired in [`configs/trainer/default.yaml`](configs/trainer/default.yaml)).
Runs link to the user who started them, the dataset registration, the git commit,
the Docker image and the hardware — an unregistered user fails the run at startup.

Prerequisites — as kube-jobs, same style as the other `scripts/`:
`uv run python scripts/create_user.py`, `scripts/register_dataset.py`,
`scripts/register_embeddings.py`:

```bash
export MLFLOW_TRACKING_URI=...  # your MLflow server
export MLFLOW_USER=...          # must be your exact Rancher username

# register yourself (once ever)
uv run python -c "from rationai.provenance import register_new_user; register_new_user(
    username='...', real_name='...', email='...',
    organization='...', lead_name='...', lead_email='...')"

# register the dataset (data manager, once per dataset version) — prints the run_id
uv run python -c "from rationai.provenance import register_dataset; print(register_dataset(
    'cohort.csv', path_column='path', dataset_name='bc20-24'))"
```

Paste the printed `run_id` (and your split URIs, if your split pipeline produces
train/test CSVs) into [`configs/ml.yaml`](configs/ml.yaml):

```yaml
dataset:
  run_id: f0c2a1…
  split_uris: {train: 'mlflow-artifacts:/<exp_id>/<run_id>/artifacts/train.csv',
               test: 'mlflow-artifacts:/<exp_id>/<run_id>/artifacts/test.csv'}
```

Training then verifies the data on disk against the registration (`fail_fast`
aborts on mismatch) and references the splits instead of recomputing them.

## Cluster

All preprocessing and training runs submit as kube-jobs:

```bash
uv run python scripts/run_tissue_masks.py
uv run python scripts/run_qc.py
uv run python scripts/run_tiling.py
uv run python scripts/run_embeddings.py   # needs HF_TOKEN (Virchow2 weights)
uv run python scripts/run_train.py
```

## Linting, Formatting and Type Checking

```bash
uvx ruff check  # Check and fix linting issues
uvx ruff format # Format code
uv run mypy .     # Run mypy
```
