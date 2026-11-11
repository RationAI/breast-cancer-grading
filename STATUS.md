# Nottingham grading — working state and plumbing

**Findings live in [`REPORT.md`](REPORT.md).** This file is deliberately only: what is true right now,
what is pending, and the operational facts that each cost a failed job. Results, metric definitions and
the evidence for any claim belong in `REPORT.md`, not here.

Updated 2026-11-12. Numbers verified against per-slide prediction CSVs, MLflow experiment **82 "BRC
Grading"**, or a pod/`kubectl` log. `STATUS.md` history is in git; the sections deleted from it (12-arm
sweeps, per-wave writeups) are summarised in `REPORT.md` §6, §7 and Appendices A–B.

## Where we are

- **Shipped candidate**: 5-fold average of `cand` (attention pooling, `ml.head.dropout=0.2`, whole bags,
  plain CE), **no decision rule**. On test: balanced 0.7449 / AUROC 0.9528 / κ 0.928 / raw 0.8534,
  recalls N/G1/G2/G3 = 0.98 / 0.55 / 0.69 / 0.83, 25 undercalls of 266.
- **Frozen artifact**: `/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/` — five
  `best.ckpt`, fold label CSVs, `manifest.json` (sha256, git sha, recipe). Built by
  `scripts/freeze_artifact.py` (refuses to overwrite a directory that has a manifest).
  **The manifest's note still says "prior rule lambda fitted out-of-fold" — that is superseded**
  (`REPORT.md` §9: ship λ = 0). Rewrite the manifest when the winning arm is frozen; no new directory for
  a note.
- **Label semantics** (`scripts/make_cohort.py`, `README.md`): grade **0 = no tumour epithelium (N),
  1/2/3 = Nottingham G1/G2/G3**; grade 9 and neoadjuvant cases filtered. Old notes calling these
  "B-RAD grades 2–5" were wrong — counts unaffected, names were.
- **Cohort**: `/mnt/projects/breast_cancer/brc_grading/labels.csv`, 1735 slides, case-disjoint splits
  (measured, `scripts/case_leakage.py`). Fold CSVs: `…/brc_grading/kfold/labels-kf{0..4}.csv`, each
  train 1174–1178 / val 291–295 / test 266, train counts ≈ 575/67/247/286.

## Pending — decisions, not compute

1. **λ vs β is the pathologist's call, not ours.** λ = 0 by default; if "never report a slide too low" is
   the instruction, β ≈ 0.5 (76 vs 108 undercalls for 11 extra false alarms). What unblocks it is a cost
   per (true, predicted) pair. `REPORT.md` §9 has both price tables and the paired tests.
2. **Price the "report a grade range" option** (§9, §13): at β = 0.5, how many of the 1469 slides need a
   two-grade range and what is the miss rate inside the band. One free CPU job.
3. **Attention-vs-annotation test** — the project's stated end goal, and the biggest open item
   (`REPORT.md` §10–§11): attention is non-degenerate but below chance against the available tumour flag.
   The `Breast MMCI Full Test` tile parquets carry per-tile `carcinoma_roi_percentage` on 50 unseen
   slides; needs one pass over 86 GB, so a job, not the notebook.
4. **External grading test needs grades.** MammaPrint/MMCI slides have embeddings but **no joinable
   Nottingham grades**, and 36 of the MMCI negatives are in our train set. Skip as a benchmark
   (`REPORT.md` §11). For a real generalisation test: BRACS (public, ~198 graded slides) or another site's
   BRC slides.
5. **Rule for new arms** (`REPORT.md` §13): changes the input, not the head; ≥3 seeds on the OOF pool or it
   is not believed. Encoder change = funding decision, not an arm.

## What a run costs

| shape | 2048 tiles/epoch | whole bags |
|---|---|---|
| pod | 6–8 CPU, 24–32 Gi, 1 whole A40 | same |
| epoch | 2.0–3.4 min | 2.9–10.4 min (same config, different node → 3× spread; the share is the variable) |
| wall clock to early stop | 34–48 min typical | 40–211 min |
| VRAM | 1.9–3.5 GiB of 46 GiB | ~10 GiB (6 GiB measured when packed) |
| GPU utilisation, sampled | 0–15 % — I/O- and RAM-bound, not compute-bound | same |

`patience=5` is the committed default (`configs/trainer/default.yaml`); it cannot change which checkpoint
wins and cut ~40 % of epochs. Wall clock for 5 folds × 2 recipes: **89 min per pod** for two concurrent
runs, vs 3 h 33 m for the same shape on the previous split. Note `kf0-tf1024`'s best epoch was ~10 (vs 0–5
for whole bags) — capping delays the plateau, so patience 5 is near the minimum safe value.

**Pack shape that works**: 2 whole-bag runs per pod, `datamodule.batch_size=4`, `num_workers=2`, 32 Gi,
explicit `cache_mb` trio 768/512/512 (~4 min/epoch, 294 steps over 1175 slides). 3 per pod in 24 Gi is
**OOMKilled in epoch 0 at ~7 min** — the hog is batches in flight, not the cache.

## Cluster headroom (read from monitoring, not folklore)

A40 pool: 7 nodes, 9 cards (kub-as10/as11 have 2 each), mean **18 % utilisation**; per node 496 Gi RAM with
**352–430 Gi free**, 64 cores with 31–51 free. Host RAM was never the node limit. The real ceiling is the
**namespace quota**: `requests.memory` hard = **1 554 GiB** for `rationai-jobs-ns` (746 GiB used by others
at the time of reading); `requests.cpu` 402 hard / 156 used. A 48 Gi pod failing with "resource is out of
usage: memory" is that budget momentarily spent. **Check `kube_resourcequota` before a fat submission.**
Read it via Grafana's anonymous datasource proxy (see the `check-cluster-headroom` skill).

MIG is unavailable here (11 nodes report "cannot allocate all claims"; nodes with free slices have no free
CPU) → whole A40s only. These runs are parquet-decode bound: a 2-run pod uses 2–4 of its 8 cores. Chasing
"fill the VRAM" is wrong — framebuffer held is not work done. NAS throughput measured 565 MB/s aggregate,
not the bottleneck; per-file parquet decode is (that is what the bag cache is for, `e416621`).

## Plumbing facts (each one cost a failed job)

- **Bag-cache memory = `cache_mb` × 3 splits × `num_workers`.** `TRAIN_EMB_CACHE_GB` is **not a total
  budget** — it becomes `cache_mb` per split per worker, and a decoded bag is float32 while the parquet is
  float16, so double it. `EMB_CACHE_GB=2` with 4 workers = 24 GiB of cache; two 32 Gi pods came back
  `OOMKilled` (137) inside six minutes, before epoch 0 finished, so the job looks fine at 15 min and dies
  between polls. 8 workers × (3072+1024+1024) MB ≈ 41 GB blew a 48 Gi pod at epoch ~4. The default config
  is 3072 MB × 3 × 8 ≈ **144 GiB** — always pass the trio. `scripts/run_train.py` now computes this and
  refuses anything over half the pod (`check_cache_budget`); the shape that has never failed is
  768/512/512 with 4 workers in a **32 Gi** pod.
- **A pod used to report `Succeeded` with training dead in minutes** (the script ended in `wait`, so the
  exit code came from the GPU sampler). Training records its code in `/tmp/train_rc` and the last line
  fails the job (`0c02029`) — that turned four fake successes into visible failures the same hour.
- **`kubectl exec` into a job pod does not get the training venv** (`python` → linuxbrew):
  `cd /home/jovyan/workdir && . .venv/bin/activate && python …`.
- **Concurrent `uv run` inside one pod drops to the image's system python** — call `.venv/bin/python`
  directly (`d15cc78`). Locally `uv run` can pick a system Python 3.14 without torch: use `.venv/bin/python`
  or `uv run --no-sync`.
- **`uv add <pkg>` prunes packages it does not know about.** It removed `openslide-bin`, silently breaking
  `import openslide` (the declared dep is `openslide-client`; you pick the provider). `uv add
  openslide-bin` restores it.
- **Pods must clone the real repo name** `RationAI/breast-cancer-grading`; GitHub's redirect from the old
  underscore name died mid-day, so pods trained on nothing until the clone was fixed and made to fail
  loudly (`d857670`, `491413a`).
- **kube-jobs**: joins script lines with `;` (a line ending in `&` becomes `&;` → syntax error; close
  background jobs with `true`); **resubmitting a name needs the old job deleted first**; ~10 min to
  schedule; `TRAIN_CPU`/`TRAIN_MEMORY` are the packing knobs; the cluster injects
  `MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow` into every job.
- **MLflow is reachable from the laptop**: `https://mlflow-s3.rationai.cloud.trusted.e-infra.cz` (same
  server as in-cluster `http://mlflow-s3.rationai-mlflow`). Only the shared PVC — checkpoints, prediction
  CSVs — needs a pod. Keys: `test/accuracy` (**balanced**, not raw), `test/auroc`, `test/f1` (macro),
  `test/kappa`, `test/recall_class_<0..3>`, `valid/loss`, `epoch`.
- **Experiment 82 holds ~27 phantom `RUNNING` runs** from the provenance autologger (tags `user_run_id`,
  metrics `epoch`/`lr-Adam`/`train/loss_step` only). Filter on the presence of `test/kappa`, or `epoch >= 8`.
- **`variant` is a param, not a tag**, and `_`/`-` differ between systems: MLflow stores `full_ord_s456`,
  the checkpoint dir is `ml_attention-full-ord-s456`. Search `params.variant = '…'`.
- **Verify Hydra keys against the composed config** (`python -m ml.train --cfg job --resolve`); invented
  keys kill the run at parse time. Real ones: `datamodule.batch_size`, `datamodule.num_workers`,
  `datamodule.{train,val,test}.cache_mb`, `datamodule.{train,val,test}.max_tiles` (`null` = whole bags),
  `ml/aggregator=mean`, `ml/loss=ordinal`, `ml.head.dropout`, `seed`, `model.predictions_dir`.
- **The prediction dumps are numbered by different epochs**: the test frame is written at `epoch+1`
  (Lightning bumps `current_epoch` before the final test) and `best.ckpt`'s filename carries no epoch — read
  it out of the checkpoint (`ckpt["epoch"]`). The OOF pool = the five `valid_epoch<best>.csv`; the test
  frames are all the same 266 slides. Getting this wrong produced a "no dumps" report that ran to completion.
- **Tests here are import-time assert scripts, not pytest**: run `./tests/run_all.sh` (it runs each file with
  `PYTHONPATH=.:scripts` and prints the line each ends with). `pytest -q tests` collects **zero items**, so
  **"no tests ran" is not a pass** — that is how `tests/test_kfold_labels.py` shipped half a day with zero
  assertions in it.
- **Gap on `master`**: the flags used to submit the last waves (`--recipes`, `--run-name`, `--overrides`,
  and the `dataset.max_tiles` → `datamodule.{train,val,test}.max_tiles` expansion) lived only in a scratch
  working copy that has since been deleted. The results are valid (the pod spec is the record), but
  re-running a wave from `master` needs the flags re-added or the overrides typed out:

  ```
  ml.head.dropout=0.2 datamodule.num_workers=2 datamodule.batch_size=4
  datamodule.train.cache_mb=512 datamodule.val.cache_mb=256 datamodule.test.cache_mb=256
  datamodule.train.max_tiles=<cap> datamodule.val.max_tiles=<cap> datamodule.test.max_tiles=<cap>
  ```

## Submitting

```bash
cd ~/Documents/rationAI/breast-cancer-grading && git push origin master   # pods clone master
for spec in "kf0-tf1024:seed=42 datamodule.train.max_tiles=1024" \
            "kf0-tf4096:seed=42 datamodule.train.max_tiles=4096"; do
  TRAIN_VARIANT=${spec%%:*} TRAIN_OVERRIDES="${spec#*:}" \
  TRAIN_MEMORY=32Gi TRAIN_CPU=8 TRAIN_EMB_CACHE_GB=0.5 \
    uv run --no-sync python scripts/run_train.py
done
```

### xOpat overlays for the pathologist (`scripts/run_masks.py`)

`scripts/attention_masks.py` writes what xOpat needs, ported from Tile-Morph-Clustering so it behaves the
way the LDA masks they already reviewed behave: per-slide **uint8 pyramid BigTIFF at level-0 dimensions**
(blocks drawn in level-1 space — the space the tile coordinates are in, verified — then 2x pixel
replication, byte-identical to drawing at level 0 for a quarter of the memory) plus
`report_conf/reporter/attention_<split>.yaml` for `python -m report`.

```bash
uv run --no-sync python scripts/run_masks.py                  # test split -> job
MASKS_SPLIT=train uv run --no-sync python scripts/run_masks.py
kubectl logs -n rationai-jobs-ns job/brc-grading-masks-test-jiri-buchta
# report YAML only, from whatever masks are already on the PVC:
#   --stage report   (it is written at the end of a full run, so a killed run has none)
```

- Output: `/mnt/projects/breast_cancer/brc_grading/masks/<split>/{<slide>.tiff, attention_scores.csv,
  summary.csv}` (≈350 MB per slide — every slide in this estate is ~105 000 × 220 000 px — so ~90 GB for
  the 266 test slides; the PVC had 609 TB free when measured).
- **Resumable**: existing masks and `score_cache/` (keyed by parquet bytes + which weights) are skipped,
  so a resubmit continues. Measured 2.4 min per slide on the 8-CPU pod (every slide here is ~23 Gpx at
  level 0, and most of that is LZW encoding 99 % background) -> **~10.5 h for the 266 test slides**.
  `MASKS_CPU` is the knob (libvips encodes one TIFF tile per thread, `VIPS_CONCURRENCY=$(nproc)` is set by
  the job); to go faster, shard the slide list across jobs instead of raising the CPU — the encode is
  sequential per slide.
- Values are an **absolute** scale so a colour means the same on every slide: 0 = no tile, 1 = an equal
  share of that slide's attention, 128 = 10x, 255 = 100x or more. Pale = the head did not focus.
- Verified on the job's own output: 12-level pyramid, level-0 dims equal the slide's, and the mask value at
  `(2x, 2y)` equals the `attention_scores.csv` value for the same tile. **Registration tested**: tile rects
  built from the coordinates sit on tissue 49.8 % of the time vs 1.3-3.8 % for the same rects shifted
  200/500/1500 px at level 6 (whole slide 1.7 %) — the overlay lands where the tiles are, so the
  level-1-coordinate assumption is right. Odd level-0 dims need `embed`, not `crop` (a 1-px
  `bad extract area` failure otherwise). Stage 1 independently reproduces the shipped test accuracy
  **0.8534** on all 266 slides in ~4 min.
- Show the overlay as "where the grade head looked", **not** "where the tumour is": §10 measured
  tumour-vs-stroma contrast 0.48 (below chance). The caveat is written into the report YAML too.

`TRAIN_EMB_CACHE_GB` is refused when it exceeds half the pod; prefer the explicit `cache_mb` trio. Report
jobs: `REPORT_JOB_NAME=… REPORT_SCRIPT=ensemble_variants.py REPORT_VARIANTS=… REPORT_ARGS='--oof
--train-counts 575,67,247,286' uv run --no-sync python scripts/run_report.py` (delete a same-named job
before resubmitting). `scripts/report_variants.py` **refuses to guess a checkpoint root** — pass
`REPORT_CKPT_ROOT=/mnt/projects/breast_cancer/brc_grading/checkpoints`. Reproduction commands and why
`--train-counts` must come from the fold CSVs: `REPORT.md` Appendix C. Verified: the CSV recompute
reproduces MLflow exactly for the variants that have dumps.

## Notebook access (no job, no GPU)

`kubectl exec -n rationai-notebooks-ns jupyter-jiri-5fbuchta` — has the PVC, **1 CPU**, no GPU. Venv with
torch/pandas: `/home/jovyan/Tile-Morph-Clustering/.venv/bin/python`. Use heredocs (`python -` over stdin);
`kubectl cp` into the PVC is refused by an operator webhook. Good for CSV/dump analysis and small `.pt`
bags; anything scanning 86 GB of tile parquets belongs in a job.
