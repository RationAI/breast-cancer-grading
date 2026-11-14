<!-- Update for BRC-A-21 "Methodology" — replace the whole article body with the text below (both planned methodologies below describe what actually shipped: Methodology B, extended). -->

# Methodology (as built, 2026-11)

## Input representation

* Each WSI → a **bag of tile embeddings**: 224 px tiles at 0.5 µm/px, **frozen Virchow2** encoder
  (2560-d per tile), whole tissue bags (tissue + QC filtered, **no epithelium pre-filtering**).
* One parquet per slide, pre-computed once (the expensive part); training only ever reads parquets.

## Model

* **Attention-pooling MIL**: gated attention over the bag (~17 k trainable parameters) + linear
  classification head (4 classes), head dropout 0.2. The encoder never moves — an "arm" trains the head.
* Mean pooling is the parameter-free baseline it beat: mean pooling cannot see G1 at all
  (recall 0.29–0.43) because averaging over a stroma-dominated bag washes the rare tumour signal out.

## Training

* **Stratified 5-fold cross-validation**, case-disjoint folds (`kfold/labels-kf{0..4}.csv`),
  5 runs of the same recipe (variant `kf0-cand` … `kf4-cand`)
* Loss: plain cross-entropy on slides; class-balanced batch sampling; whole bags (`max_tiles: null`)
* Early stopping on `valid/loss`, patience 5 (best epoch was 0–5 in all runs), best checkpoint per fold
* 12 CPU / 48 Gi pods — decoding a ~61 MB parquet bag dominates the ~10 ms GPU step, so throughput
  scales with dataloader workers

## Inference

* **Fold average**: the per-slide softmax of the 5 fold models is averaged; argmax of that. No seed
  choice, no member selection, **no decision rule** (a class-prior rule λ was tested and rejected —
  see BRC-A-28).
* Frozen artifact with sha256 manifest:
  `/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/`
  (5 × `best.ckpt`, fold label CSVs, `manifest.json`). To score a new cohort: run `ml.train` per
  checkpoint with `checkpoint=… dataset.paths.labels=<cohort csv> eval.split=test`, then average the
  per-slide softmaxes — the same averaging `scripts/ensemble_variants.py` does.

## Evaluation protocol (why, in one list)

* single runs cannot rank anything here: ±2.6-point sampling noise on 266 slides, ~6-point same-config
  seed spread → **≥3 seeds or the out-of-fold pool (1469 slides) + paired McNemar** for any claim
* choose on validation, report on test, once — every inference-time knob (λ, thresholds, member sets)
  was fitted on validation only or rejected

## Tested and rejected (full evidence in REPORT.md §6–§7)

mean pooling · ordinal CE (better AUROC, worse κ — ranker only) · class-weighted CE · focal loss ·
class-prior rule λ (balanced accuracy +5 but significantly worse per slide, p = 0.0135) · learned
per-class thresholds · bag caps 1024/4096 (whole bags win: −1.57 / −3.00) · bag augmentation
(rejected before spending: nothing positional exists for it to break).
