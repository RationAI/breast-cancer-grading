# Nottingham grading from Virchow2 embeddings — what we know (2026-07-29)

Working notes, not committed. Everything below is checked against MLflow (experiment **82 "BRC
Grading"**), the labels CSV on the share, or a pod log; the two places where a number is an
estimate say so.

## Headline

- The model works: **balanced (macro-recall) test accuracy 0.749–0.774, AUROC 0.942–0.952,
  quadratic-weighted kappa 0.913–0.921** on 4-class Nottingham grading, from frozen Virchow2
  tile embeddings, ~20 epochs of an attention-MIL head (17 k trainable parameters).
- **`test/accuracy` is balanced accuracy, not raw accuracy** — it is exactly the mean of the four
  class recalls in every finished run ((.99+.79+.62+.70)/4 = 0.775 for `7da26a15`). Raw accuracy
  is ~0.83, because class 0 (no tumour epithelium, 131 of 266 test slides) sits at recall 0.98–0.99
  and is nearly free. The hard classes are **G1 (c1) at 0.43–0.79** and **G2 (c2) at 0.62–0.72**.
- **We cannot tell yet whether attention pooling beats mean pooling.** The two candidate runs
  differ by 6 accuracy points but also differ in seed, and the test set is too small to see
  anything smaller than ~3 points. Four runs are in flight to settle it at matched seeds.
- Almost all the day's cost went to cluster plumbing, not to the model. The plumbing facts are
  at the bottom; they are the part worth keeping.

## The data (bc20-24, from `labels.csv` on the share)

1735 slides, one parquet of per-tile Virchow2 embeddings each (median bag ~61 MB, p90 144 MB,
max 336 MB; ~21 k tiles in the example slide, of which `max_tiles=2048` samples per epoch).

| split | c0 | c1 | c2 | c3 | total |
|---|---|---|---|---|---|
| train | 593 | 68 | 254 | 293 | 1208 |
| val | 126 | 16 | 55 | 64 | 261 |
| test | 131 | 14 | 58 | 63 | **266** |

Classes 0–3 are B-RAD/Nottingham grades 2–5. **c1 is 5.6 % of the data and 14 test slides** —
one test slide moves that class's recall by 7 points, which is why its recall jumps around.

## Finished runs (test set, 266 slides)

Balanced accuracy (`test/accuracy`) = mean of the four per-class recalls; per-class recalls are on
the right, against test counts 131 / 14 / 58 / 63.

| run | variant | seed | tiles/epoch | pooling | epochs | best val loss | bal. acc | AUROC | F1 | kappa | recall c0/c1/c2/c3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `c53175d6` | baseline | not logged | full bag (no `max_tiles` key existed) | attention | 22 | 0.4451 | **0.7612** | 0.9462 | 0.7335 | 0.9132 | .98 / .64 / .66 / .76 |
| `33c7de37` | `tiles2048_s123` | 123 | full bag (logged before the key existed) | attention | 23 | 0.4643 | 0.7486 | 0.9423 | 0.7472 | 0.9213 | .99 / .50 / .72 / .78 |
| `7da26a15` | `tiles2048_s456` | 456 | 2048 ✓ | attention | 14 | 0.4792 | **0.7743** | 0.9522 | 0.7260 | 0.9202 | .99 / .79 / .62 / .70 |
| `7cd593f0` | `meanpool_s123` | 123 | 2048 ✓ | mean | 35 | 0.4195 | 0.7135 | 0.9382 | 0.7038 | 0.9197 | .99 / .43 / .66 / .78 |
| `58076001` | `att2048_s789` | 789 | 2048 ✓ | attention | 15 | 0.4066 | 0.7147 | 0.9489 | 0.7168 | 0.9080 | .98 / **.36** / .81 / .71 |

## What the numbers will and will not support

- **Noise floor: ~±2.6 accuracy points.** n = 266 test slides, p ≈ 0.75 → binomial SE 0.027.
  The observed spread (0.7486 / 0.7612 / 0.7743 for the same attention config) is ordinary
  sampling noise, not signal. Any claimed improvement under ~3 points is invisible here.
- **The pooling comparison as it stands is confounded**: the honest 2048-tile pair is attention
  seed 456 (0.7743) vs mean seed 123 (0.7135) — pooling and seed change together. Nothing can be
  concluded from it.
- **The names lie about one knob.** `baseline` and `tiles2048_s123` have no `max_tiles` in their
  logged params, i.e. they trained *before* sub-sampling existed and saw whole bags. So
  "2048 vs full bag" was never actually measured; the three full-bag attention runs are the only
  repeat, and they span 0.749–0.774.
- **Class 1 is the weak spot in every run** (recall 0.36–0.79) while class 0 sits at 0.98–0.99.
  With 68 training slides for c1 that is a data-imbalance story, and it is the same story in the
  overfitting signal seen earlier (train acc ~0.92 vs val ~0.79, flat from epoch ~5).
- **The kappa column is saturated, the accuracy column is not.** Every finished run lands at
  kappa 0.908–0.921, mean pooling included, while balanced accuracy spans 0.71–0.77. Judge changes
  on balanced accuracy and the per-class recalls; kappa cannot resolve them.
- **One test slide of class 1 moves the headline numbers a lot.** c1 has 14 test slides, so its
  recall moves in 7.1-point steps — 1.8 points of balanced accuracy, most of the ±2.6-point noise
  floor. This is why the next batch dumps per-slide predictions: a paired (McNemar) comparison on
  the same 266 slides separates signal from that step size, an unpaired comparison cannot.
- Older runs in experiment 37 ("Breast Cancer Grading", 8-class, batch 20) and the 5-epoch probe
  are **not comparable** to any of the above — different label space and different batch regime.

## In flight right now (3 of 4 left on their A40s, submitted ~15:20 UTC;
## `att2048-s789` finished at epoch 15 and is in the results table)

| variant | seed | pooling | resume |
|---|---|---|---|
| `att2048-s123` | 123 | attention | from its own epoch-4 checkpoint |
| `att2048-s789` | 789 | attention | from epoch 4 |
| `mean-s456` | 456 | mean | yes |
| `mean-s789` | 789 | mean | yes |

Same 2048-tile, 4-class config as the finished runs; only seed/pooling differ. Together with
`tiles2048_s456` (attention) and `meanpool_s123` (mean) this completes a 2 × 3 attention-vs-mean
grid at matched seeds — the first comparison of pooling that is not confounded.

- Pod shape: 6 CPU, 24 Gi, `num_workers=4`, bag cache 768/512/512 MB. Whole-A40 because MIG is
  unusable (below).
- Epoch **~2–3.4 min**, early stopping (patience **10** on `valid/loss`) fires far earlier than the
  40-epoch cap: all four stop around epoch 13–15, with the best val loss already at epoch ~5, so a
  finished job is ~35 min of training (`58076001`: 15 epochs in 34 min). GPU utilisation is 0–15 %;
  these pods are I/O- and RAM-bound, which is exactly what made MIG tempting.
- Checkpoints go to `/mnt/projects/breast_cancer/brc_grading/checkpoints/ml_attention-<variant>/`
  and every pod resumes from its own if present, so a kill is not a loss.

## Prepared and committed, not submitted (`6b602e2`)

1. **Whole bags** — `datamodule.train.max_tiles: null`. Tiles per slide are p25 2.3k / median 6.5k
   / p75 12.3k / max 27k and strongly class-correlated (c0 median 2.5k, tumour grades 9–11k), so a
   fixed 2048-tile bag read c0 almost completely and the three hard classes at ~20 %: bag length on
   its own scores 0.579 raw accuracy / 0.391 balanced accuracy / 0.509 kappa on the test split. The
   val/test loaders always used whole bags, so training optimised a pooling the evaluation never
   applies. Cost: ~4× the tiles per step.
2. **`OrdinalCrossEntropy`** (`ml/loss=ordinal`) — reads the existing 4 logits as 3 cumulative
   `grade > k` questions, so a two-grade slip pays four times a one-grade slip instead of the same
   price. No new parameters, head unchanged. The current `configs/ml/metrics/quadratic_kappa.yaml`
   *is* quadratic-weighted (`weights: quadratic`), so this loss and `test/kappa` line up.
3. **Per-slide prediction dumps** — `GradingModule(predictions_dir=…)` writes
   `<stage>_epoch<N>.csv` (slide_id, label, 4 logits). `run_train.py` points it at
   `<ckpt-dir>/predictions`, which is per variant, so no new plumbing. Ensembling and a paired
   McNemar test between two runs both need these and nothing kept them before.

Measured before committing, in a live pod: `GradingDataset(max_tiles=None)` + attention model on CPU
→ 6 905 tiles/bag mean, 835 ms/bag cold **while three training jobs were reading the same share**,
4.4 GB peak RSS, 8 steps × 8 bags in 21 s → expect ~3–4 min/epoch against 2–3.4 min today.

Submit once the four in-flight jobs release their cards (~17:15):

```bash
cd ~/Documents/rationAI/breast-cancer-grading && git push origin master   # pods clone master
for spec in \
    "full-s456:seed=456" \
    "full-ord-s456:seed=456 ml/loss=ordinal" \
    "full-ord-s789:seed=789 ml/loss=ordinal" \
    "t2048-ord-s456:seed=456 ml/loss=ordinal datamodule.train.max_tiles=2048"; do
  TRAIN_VARIANT=${spec%%:*} TRAIN_OVERRIDES="${spec#*:} datamodule.num_workers=4" \
  TRAIN_EMB_CACHE_GB=2 TRAIN_MEMORY=32Gi TRAIN_CPU=8 \
    uv run --no-sync python scripts/run_train.py
done
```

32 Gi, and **explicit per-split cache sizes** (`datamodule.train.cache_mb=768
datamodule.val.cache_mb=512 datamodule.test.cache_mb=512`). The first submission of this batch used
`TRAIN_EMB_CACHE_GB=2` and **two pods came back `OOMKilled` in under six minutes**: `EMB_CACHE_GB`
sets `cache_mb` for *every* split, so 2 GB × 3 splits × 4 workers = 24 GiB of bag cache inside a
32 Gi pod, before any batch is in flight. The explicit trio above is 7.0 GiB (verified with
`--cfg job --resolve`), leaving room for a prefetched batch of 8 bags at the 27k-tile maximum
(8 × 27k × 2560 float16 ≈ 1.1 GB per worker, × 4 workers × prefetch 2).

| pair | isolates |
|---|---|
| `full_s456` vs `tiles2048_s456` (0.7743) | all the data vs a fifth of it, same seed and config |
| `full-ord-s456` vs `full_s456` | the ordinal loss at the new data budget |
| `t2048-ord-s456` vs `tiles2048_s456` | the ordinal loss at the old budget |
| `full-ord-s789` vs `full-ord-s456` | seed spread at the new setting |

Read them off `test/accuracy` and the per-class recalls; `test/kappa` is saturated at 0.908–0.921
and will not resolve any of this. Two things deliberately left alone: the checkpoint/early-stopping
monitor stays `valid/loss` (changing it alongside the loss would confound the ablation, though
selecting on `valid/kappa` is the obvious next candidate), and nothing yet ensembles the runs — the
prediction dumps make that possible, the script is not written.

## Plumbing facts worth keeping (each one cost a failed job)

- **Bag cache memory = `cache_mb` × 3 splits × `num_workers`.** 8 workers × (3072+1024+1024) MB
  ≈ 41 GB blew a 48 Gi pod at epoch ~4; 4 workers × (768+512+512) MB ≈ 7 GB fits a 24 Gi pod.
  This, not the model, decides the pod size (VRAM is only 1.9–3.5 GiB).
- **`EMB_CACHE_GB` (= `TRAIN_EMB_CACHE_GB`) is not a total budget** — it becomes `cache_mb` for each
  of the three splits, per worker. `EMB_CACHE_GB=2` with 4 workers is 24 GiB of cache, which OOMKilled
  two 32 Gi pods on 2026-11-10. Pass the three `datamodule.*.cache_mb` keys explicitly instead, and
  confirm with `--cfg job --resolve` before paying for a pod.
- **MLflow is reachable from the laptop**, not just in-cluster: `MLFLOW_TRACKING_URI=https://
  mlflow-s3.rationai.cloud.trusted.e-infra.cz`, our runs live in experiment **82 "BRC Grading"**
  (in-cluster the same server is `http://mlflow-s3.rationai-mlflow`, which kube-jobs injects as
  `MLFLOW_TRACKING_URI`). Only the shared PVC with the checkpoints and prediction CSVs needs a pod,
  and the checkpoint directories are spelled with **dashes** (`ml_attention-full-ord-s456`) while
  the MLflow `variant` param uses underscores (`full_ord_s456`).
- **A pod used to report `Succeeded` with training dead in its first minutes** — the script
  continues with `sleep 300; nvidia-smi; wait` after training to sample the GPU, so the pod's
  exit code came from `wait`. Training now records its code in `/tmp/train_rc` and the last line
  fails the job. It turned four fake successes into visible failures the same hour (`0c02029`).
- **Verify Hydra keys against the composed config**, `python -m ml.train --cfg job --resolve`.
  Invented keys (`ml.batch_size`) kill the run at config-parse time:
  `Could not override 'ml.batch_size'`. Real ones: `datamodule.batch_size`,
  `datamodule.num_workers`, `datamodule.{train,val,test}.cache_mb`, `datamodule.train.max_tiles`,
  `ml/aggregator=mean`, `seed`.
- **kube-jobs**: joins script lines with `;` (a line ending in `&` becomes `&;` → bash syntax
  error, close with `true`); resubmitting a name needs the old job deleted first; ~10 min to
  schedule; `TRAIN_CPU`/`TRAIN_MEMORY` are the packing knobs.
- **Experiment 82 holds ~27 phantom `RUNNING` runs.** The provenance autologger opens side runs
  (tags `user_run_id`, `prov_start_time`; metrics `epoch`, `lr-Adam`, `train/loss_step` only) that
  inherit the `variant` hyperparam and never finish. Filter on the presence of `test/kappa`, or on
  `epoch >= 8`, when reading a leaderboard — `/tmp/runs_table.py` does the former.
- **The `variant` hyperparam is a param, not a tag, and it is ASCII-safe**: `att2048-s789` is stored
  as `att2048_s789`. Search `params.variant = 'att2048_s789'`; `tags.variant` returns nothing.
- **`kubectl exec` into a job pod does not get the training venv** (`python` resolves to linuxbrew).
  Use `cd /home/jovyan/workdir && . .venv/bin/activate && python …`.
- **Pods must clone the repo under its real name** `RationAI/breast-cancer-grading`; GitHub's
  redirect from the old underscore name died mid-day, so pods trained on nothing until the clone
  was fixed and made to fail loudly (`d857670`, `491413a`).
- **Concurrent `uv run` inside one pod drops to the image's system python** — call
  `.venv/bin/python` directly (`d15cc78`).
- **MIG is not an option today**: 11 nodes report "cannot allocate all claims" (no free slices)
  and the nodes with free slices have no free CPU. Whole A40s only.
- **MLflow REST from a pod** (no local egress): `POST /api/2.0/mlflow/runs/search` with
  `experiment_ids` as a real JSON **list**; `params`/`metrics` come back as **lists** of
  key/value; experiment id is `82`, metric names are `test/accuracy`, `test/auroc`, `test/f1`,
  `test/kappa`, `test/recall_class_<i>`, `valid/loss`, `epoch`.
- NAS throughput measured 565 MB/s aggregate — not the bottleneck; per-file parquet decode is
  (that is what the bag cache exists for, `e416621`).

## Next steps, cheapest first

1. Read the 2 × 3 pooling grid when these four finish (~17:30 UTC, most likely). Expected honest
   answer: "within noise", which is itself worth knowing.
2. **Class 1 is the real problem.** Cheapest fixes: class-weighted loss or resampling (a Hydra
   flag away), then report macro-F1 / per-class recall as the headline rather than accuracy.
3. **Make the test set bigger before running more model variants**: only 15 % of the 1735 slides
   are held out. k-fold or a larger held-out set buys resolution that no seed repetition can.
4. Then, and only then, model changes: pretrained aggregator / transcript head, tile-level
   multiple instance, or a stronger regulariser for the train-acc-0.92 overfit.

## Not known

- Whether 2048-tile sub-sampling helps or hurts accuracy — never measured against full bags.
- Whether the attention weights attend to anything morphologically meaningful (no attention map
  inspection done, no slide-level visualisation).
- Calibration, and behaviour on external data (bc20-24 only, single scanner estate).
- Any comparison to the published BCI / bc20-24 baselines — no external baseline has been run.
