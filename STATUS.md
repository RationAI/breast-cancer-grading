# Nottingham grading from Virchow2 embeddings — what we know

Working notes (updated 2026-11-10, after 12 finished 4-class runs). Every number is from MLflow
(experiment **82 "BRC Grading"**), from a per-slide prediction CSV on the shared PVC, or from a pod
log. Estimates are labelled as such.

## Headline

- The model works: **4-class Nottingham grading from frozen Virchow2 tile embeddings with a
  17 k-parameter attention-MIL head** reaches **balanced accuracy 0.71–0.77, AUROC 0.93–0.95,
  quadratic kappa 0.89–0.92** on 266 held-out slides. Best single run: `tiles2048_s456`, 0.7743 /
  0.9522 / 0.9202 (`7da26a15`).
- **`test/accuracy` is balanced accuracy** (mean of the four class recalls), not raw accuracy — it
  reproduces exactly in every run. Raw accuracy is ~0.83–0.84 because class 0 (131 of 266 test
  slides) sits at recall 0.98–0.99 and is nearly free.
- **Attention pooling beats mean pooling.** Three seeds at matched config: +9.5 / +4.8 / +4.3
  balanced-accuracy points, 3 wins out of 3. The confounded pair that made this look unknowable is
  gone.
- **Using every tile every epoch is worse than using ~20 %.** Paired at two seeds: −3.7 and −1.3
  points for whole bags, and ~3× slower per epoch. The random sub-sample acts as bag augmentation
  and keeps train bags the same length as the eval bags.
- **The ordinal loss is a wash on accuracy, the best ranker we have, and worse on kappa.** See
  "Settled by the last batch".
- **Class 1 (G1) is the whole problem.** Its recall spans 0.29–0.79 across runs while class 0 never
  drops below 0.98. 68 training slides, 14 test slides.

## The data (bc20-24, from `labels.csv` on the share)

1735 slides, one parquet of per-tile Virchow2 embeddings each (median bag ~61 MB, p90 144 MB, max
336 MB). Tiles per slide: p25 2.3k, median 6.5k, p75 12.3k, max 27k — and **strongly
class-correlated** (c0 median 2.5k vs 9–11k for the tumour grades), which is why bag length alone
scores 0.579 raw accuracy / 0.391 balanced accuracy / 0.509 kappa on the test split.

| split | c0 | c1 | c2 | c3 | total |
|---|---|---|---|---|---|
| train | 593 | 68 | 254 | 293 | 1208 |
| val | 126 | 16 | 55 | 64 | 261 |
| test | 131 | 14 | 58 | 63 | **266** |

Classes 0–3 are B-RAD/Nottingham grades 2–5. **Only 15 % of slides are held out.**

## All finished runs (test set, 266 slides), best-val-loss checkpoint

`bal` = `test/accuracy` = balanced accuracy; recalls are c0/c1/c2/c3 against counts 131/14/58/63.
Val loss is **not comparable across loss families** (CE ≈ 0.42–0.50, ordinal ≈ 0.14–0.18).

| run | variant | seed | tiles/epoch | pooling | loss | ep | min | best val | bal | AUROC | F1 | kappa | recalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `7da26a15` | tiles2048_s456 | 456 | 2048 | attn | CE | 14 | 48 | 0.4792 | **0.7743** | 0.9522 | 0.7260 | 0.9202 | .99/.79/.62/.70 |
| `2a8a70e3` | t2048_ord_s456 | 456 | 2048 | attn | ord | 14 | 96 | 0.1769 | 0.7693 | **0.9544** | 0.7187 | 0.8994 | .99/.79/.57/.73 |
| `ba362f65` | full_ord_s456 | 456 | whole | attn | ord | 14 | 40 | 0.1797 | 0.7637 | **0.9544** | 0.7251 | 0.8995 | .99/.71/.59/.76 |
| `c53175d6` | baseline (unnamed) | n/l | whole | attn | CE | 22 | n/l | 0.4451 | 0.7612 | 0.9462 | 0.7335 | 0.9132 | .98/.64/.66/.76 |
| `44897d1a` | att2048_s123 | 123 | 2048 | attn | CE | 40 (cap) | 211 | 0.4423 | 0.7618 | 0.9406 | **0.7538** | 0.9193 | .99/.57/.69/.79 |
| `33c7de37` | tiles2048_s123 | 123 | **whole** (name lies: logged before `max_tiles` existed) | attn | CE | 23 | 54 | 0.4643 | 0.7486 | 0.9423 | 0.7472 | **0.9213** | .99/.50/.72/.78 |
| `907fbfde` | full_ord_s789 | 789 | whole | attn | ord | 17 | 105 | 0.1678 | 0.7432 | 0.9511 | 0.7161 | 0.9174 | .99/.64/.66/.68 |
| `1ce8c392` | full_s456 | 456 | whole | attn | CE | 18 | 124 | 0.4991 | 0.7372 | 0.9482 | 0.7309 | 0.9107 | .98/.50/.74/.73 |
| `58076001` | att2048_s789 | 789 | 2048 | attn | CE | 15 | 34 | 0.4792 | 0.7147 | 0.9489 | 0.7168 | 0.9080 | .98/**.36**/.81/.71 |
| `7cd593f0` | meanpool_s123 | 123 | 2048 | mean | CE | 35 | 119 | 0.4195 | 0.7135 | 0.9382 | 0.7038 | 0.9197 | .99/.43/.66/.78 |
| `d16d2e01` | mean_s456 | 456 | 2048 | mean | CE | 23 | 137 | 0.4245 | 0.6791 | 0.9391 | 0.6781 | 0.9160 | .99/**.29**/.72/.71 |
| `49efd009` | mean_s789 | 789 | 2048 | mean | CE | 16 | 78 | 0.4337 | 0.6718 | 0.9327 | 0.6632 | 0.8949 | .98/.36/.55/.79 |

## Settled by the last batch

**1. Attention vs mean pooling — attention, at every seed we tried.** Both at 2048 tiles, CE:

| seed | attention | mean | Δ |
|---|---|---|---|
| 456 | 0.7743 | 0.6791 | **+9.5** |
| 123 | 0.7618 | 0.7135 | **+4.8** |
| 789 | 0.7147 | 0.6718 | **+4.3** |

Mean pooling's failure mode is visible in the recalls: it never learns c1 (0.29–0.43) and its c2
swings 0.55–0.72. Mean over a bag dominated by stroma washes the rare tumour signal out; the
attention weights do not have to.

**2. Whole bags vs the 2048-tile sub-sample — the sub-sample wins** (attention, CE):

| seed | 2048 tiles | whole bags | Δ |
|---|---|---|---|
| 456 | 0.7743 | 0.7372 | **+3.7** |
| 123 | 0.7618 | 0.7486 | **+1.3** |

The hypothesis that motivated whole bags was real — the fixed 2048 cap reads c0 almost completely
and the three hard classes at ~20 %, and val/test always used whole bags, so training optimised a
pooling the evaluation never applies. But the random sub-sample per epoch behaves like bag
augmentation and the equalised bag lengths evidently matter more. Whole bags also cost ~3–4× the
tiles per step for a worse result. **Revert to 2048.**

**3. Ordinal cross-entropy — no accuracy win, best ranking, kappa worse.** `OrdinalCrossEntropy`
(`ml/loss=ordinal`) reads the same 4 logits as 3 cumulative `grade > k` questions, so a two-grade
slip costs four times a one-grade slip. No new parameters, head unchanged.

| pair (same seed + data budget) | CE | ordinal |
|---|---|---|
| s456 @ 2048 | **0.7743** | 0.7693 (−0.5) |
| s456 @ whole | 0.7372 | **0.7637** (+2.7) |
| AUROC, same two pairs | 0.9522 / 0.9482 | **0.9544 / 0.9544** |
| kappa, same two pairs | **0.9202 / 0.9107** | 0.8994 / 0.8995 |

One result each way on balanced accuracy → no effect established. But **all three ordinal runs are
the top three on AUROC** (0.9511–0.9544), which is a 3/3 pattern, and its recall profile is
flatter — c1 at 0.59–0.79 and c2 at 0.57–0.66, versus CE's c1 at 0.36–0.50 with a compensating c2.
That is the shape you want for a screening score; the QWK regression is the price. Not adopted, not
rejected — a second seed at the config we keep would settle it for one job.

## What the numbers will and will not support

- **Seed noise is bigger than sampling noise.** Identical config, three seeds (attention @ 2048 +
  CE): 0.7743 / 0.7618 / 0.7147 — a **6.0-point range**, against a binomial floor of ±2.6 points
  (n = 266, p ≈ 0.75 → SE 0.027). Two or three seeds per arm is the minimum for any claim; a single
  run cannot rank anything.
- **Paired comparisons are the only cheap way to see small effects.** The same 266 slides under both
  arms removes most of the binomial term; that is what the per-slide prediction dumps are for.
- **One c1 slide moves the headline ~1.8 points.** c1 has 14 test slides, so its recall moves in
  7.1-point steps — most of the noise floor. c2 (58) and c3 (63) are ~4× steadier. Expect c1 to look
  erratic even between identical runs; trust it only over ≥3 seeds.
- **Kappa is saturated, balanced accuracy is not.** Every run lands at QWK 0.89–0.92 including mean
  pooling at 0.67 balanced accuracy. Judge on balanced accuracy + per-class recalls (+ macro-F1,
  which is the one metric that still separates runs: 0.66–0.75).
- **The `variant` names lie twice.** `tiles2048_s123` trained on whole bags (before `max_tiles`
  existed); `baseline` has no seed logged. Trust the logged params, not the name.
- **Older runs are not comparable**: experiment 37 ("Breast Cancer Grading", 8-class, batch 20) and
  the 5-epoch probe differ in label space and batch regime.

## What a run costs

| shape | 2048 tiles/epoch | whole bags |
|---|---|---|
| pod | 6–8 CPU, 24–32 Gi, 1 whole A40 | same |
| epoch | 2.0–3.4 min | 2.9–10.4 min (same config, different node → 3× spread; the share is the variable) |
| wall clock to early stop | 34–48 min typical (patience 10 on `valid/loss`; best epoch usually ~5) | 40–211 min (the 211 is `att2048_s123`, the only run that hit the 40-epoch cap) |
| VRAM | 1.9–3.5 GiB of 46 GiB | ~10 GiB |
| GPU utilisation, sampled | 0–15 % — these jobs are I/O- and RAM-bound, not compute-bound | same |

Whole A40s only: MIG is unavailable on this cluster (11 nodes report "cannot allocate all claims",
and the nodes with free slices have no free CPU).

## Next batch

(The arms above are submitted; these are the ones after them. Cheapest first, one job each.)

1. **Whatever the arms above confirm, confirm it with a second seed.** Regularisation and
   checkpoint-selection moves are expected to be worth 1-3 points, which is *under* the 6-point
   same-config seed spread — a single favourable arm is a hypothesis, not a recipe.
2. **Ensemble the runs that have prediction dumps** (mean of the four softmax probability vectors,
   majority vote) — the dumps already exist, so this costs CPU minutes, not GPU:

   ```bash
   cd ~/Documents/rationAI/breast-cancer-grading && git push origin master   # pods clone master
   REPORT_VARIANTS='full_s456,full_ord_s456,full_ord_s789,t2048_ord_s456' \
   REPORT_EPOCH=best REPORT_SPLIT=test REPORT_CKPT_ROOT=/mnt/projects/breast_cancer/brc_grading/checkpoints \
     REPORT_MLFLOW=off uv run --no-sync python scripts/run_report.py
   ```

   Expected ~+1 point from 3–4 members. Also the natural vehicle for a paired McNemar test between
   two arms (identical slide set, so the comparison cancels most of the binomial noise).
3. **Then fix class 1 with data, not architecture**: class-weighted CE (a Hydra flag away), c1
   oversampling, or a c1-specific threshold — c1 is 5.6 % of training slides.
4. **Then make the test set bigger** (k-fold over the 1735 slides, or simply a larger held-out
   split): only 15 % is held out, and no amount of seed repetition buys resolution the split size
   cannot. Until then the honest resolution limit is ~3 points.

Submit template (pods clone **master**, so push first):

```bash
cd ~/Documents/rationAI/breast-cancer-grading && git push origin master
for spec in \
    "t2048-ord-s123:seed=123 ml/loss=ordinal" \
    "t2048-ord-s789:seed=789 ml/loss=ordinal"; do
  TRAIN_VARIANT=${spec%%:*} TRAIN_OVERRIDES="${spec#*:} datamodule.num_workers=4" \
  TRAIN_MEMORY=24Gi TRAIN_CPU=6 \
  TRAIN_EMB_CACHE_GB=0.75 \
    uv run --no-sync python scripts/run_train.py
done
```

`TRAIN_EMB_CACHE_GB=0.75` is the only safe shortcut form (0.75 GB × 3 splits × 4 workers = 9 GiB in
a 24 Gi pod). For anything bigger, pass the three keys explicitly and check
`python -m ml.train --cfg job --resolve` first — see the OOM story below.

## Reading results

`scripts/report_variants.py` (+ `scripts/run_report.py`, the kube-jobs wrapper) prints, per variant:
MLflow test metrics **and** metrics recomputed from the per-slide CSVs, with the best-val epoch
cross-checked. It refuses to guess a checkpoint root — pass `REPORT_CKPT_ROOT` explicitly.

```bash
# in-cluster (needs the PVC for the CSVs; MLflow optional)
REPORT_VARIANTS=... REPORT_CKPT_ROOT=... uv run --no-sync python scripts/run_report.py

# from the laptop: MLflow only, no pod
export MLFLOW_TRACKING_URI=https://mlflow-s3.rationai.cloud.trusted.e-infra.cz
PYTHONPATH=. .venv/bin/python scripts/report_variants.py --variants tiles2048_s456 --mlflow-only
```

Verified: the CSV recompute reproduces MLflow exactly for the four variants that have dumps
(`full_ord_s456` → bal 0.7637 / kappa 0.8995 / AUROC 0.9544 both ways).

## Overnight batch: 8 arms, one knob each (submitted 2026-11-10 ~23:00 UTC)

All at the config the last batch settled on — **2048 tiles/epoch, attention pooling** — so every
arm is paired against a finished run at the **same seed**, and all of them dump per-slide
predictions so they can join the ensemble afterwards. Pod shape 6 CPU / 32 Gi / whole A40,
`cache_mb` 768/512/512, 4 workers.

| arm | the one knob | paired against | what it asks |
|---|---|---|---|
| `t2048-w-s456` | class-weighted CE, `[0.28, 2.48, 0.66, 0.58]` | `tiles2048_s456` **0.7743** | does upweighting G1 lift balanced accuracy without wrecking c0/c2 |
| `t2048-w-s123` | same, second seed | `att2048_s123` **0.7618** | is that reproducible |
| `t2048-ord-s123` | ordinal loss, second seed | `att2048_s123` **0.7618** (CE) | is the ordinal AUROC edge real, is accuracy really a wash |
| `t2048-acc-s456` | select/early-stop on `valid/accuracy` (mode max) instead of `valid/loss` | `tiles2048_s456` **0.7743** | we pick the epoch by a loss but report accuracy; the best-loss epoch is usually ~5 of 15 |
| `t2048-acc-s123` | same, second seed | `att2048_s123` **0.7618** | reproducible |
| `t2048-wd2-s456` | Adam `weight_decay` 1e-3 → 1e-2 | 0.7743 | the documented overfit (train acc 0.92 vs val 0.79) |
| `t2048-dr02-s456` | head `dropout` 0.5 → 0.2 | 0.7743 | …or is 0.5 too strong for a 10 k-param head |
| `t2048-t4096-s456` | `datamodule.train.max_tiles` 2048 → 4096 | 0.7743 | the budget was never swept: only 2048 vs *all* has been measured |

The loss-selection knob deliberately leaves the LR scheduler on `valid/loss`: optimise on the
loss, *choose* by the metric we report. `valid/accuracy` is balanced accuracy on 261 val slides,
where c1 has 16 slides (6.25-point steps), so selection gets noisier — that is the risk in that
pair of arms, not a bug.

**Reading them in the morning** — MLflow is reachable from the laptop:

```bash
cd ~/Documents/rationAI/breast-cancer-grading
export MLFLOW_TRACKING_URI=https://mlflow-s3.rationai.cloud.trusted.e-infra.cz
MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=. .venv/bin/python /tmp/full_table.py   # all runs, recalls included
# then re-ensemble everything that now has dumps (CPU pod, ~3 min):
REPORT_JOB_NAME=brc-grading-ensemble2 REPORT_SCRIPT=ensemble_variants.py \
REPORT_VARIANTS=full_s456,full_ord_s456,full_ord_s789,t2048_ord_s456,t2048_w_s456,t2048_ord_s123 \
  uv run --no-sync python scripts/run_report.py
```

The first ensemble job (`brc-grading-ensemble`) finished at 23:4x UTC and **its output has not been
read yet** — `kubectl logs -n rationai-jobs-ns <pod>` while the pod is still there.

## Plumbing facts worth keeping (each one cost a failed job)

- **Bag-cache memory = `cache_mb` × 3 splits × `num_workers`.** `TRAIN_EMB_CACHE_GB`
  (= `EMB_CACHE_GB`) is **not a total budget**: it becomes `cache_mb` for *each* split, per worker.
  `EMB_CACHE_GB=2` with 4 workers = 24 GiB of cache, and **two 32 Gi pods came back `OOMKilled`
  (exit 137) inside six minutes on 2026-11-10** — before epoch 0 finished, so the job looks fine at
  15 min and dies between two polls. Earlier, 8 workers × (3072+1024+1024) MB ≈ 41 GB blew a 48 Gi
  pod at epoch ~4. The model needs 2–10 GiB of VRAM; **the cache decides the pod size.**
- **The cache shortcut bit me twice; the submitter now refuses it.** `EMB_CACHE_GB=0.75` with 4
  workers OOMKilled **3 of 3 pods in epoch 0 (exit 137, no traceback)** in 24 Gi pods on
  2026-11-10. The arithmetic: the budget is *per worker per split* (0.75 × 3 × 4 = 9 GiB) **and a
  decoded bag is float32 while the parquet is float16, so double it** → ~18 GiB wanted inside a
  24 Gi pod. `scripts/run_train.py` now computes that and refuses anything over half the pod
  (`cache_budget_gib` / `check_cache_budget`); the shape that has never failed is the explicit
  `cache_mb` trio 768/512/512 with 4 workers in a **32 Gi** pod (14 GiB by that arithmetic).
  The untouched config default is 3072 MB × 3 splits × 8 workers, i.e. ~144 GiB — always pass the
  trio explicitly.
- **A pod used to report `Succeeded` with training dead in its first minutes** — the script continued
  with `sleep 300; nvidia-smi; wait` to sample the GPU, so the exit code came from `wait`. Training
  now records its code in `/tmp/train_rc` and the last line fails the job; that turned four fake
  successes into visible failures the same hour (`0c02029`).
- **`kubectl exec` into a job pod does not get the training venv** (`python` → linuxbrew). Use
  `cd /home/jovyan/workdir && . .venv/bin/activate && python …`.
- **Concurrent `uv run` inside one pod drops to the image's system python** — call `.venv/bin/python`
  directly (`d15cc78`). Locally, `uv run` can pick up a system Python 3.14 with no torch: use
  `.venv/bin/python` or `uv run --no-sync`.
- **`uv add <pkg>` prunes packages it does not know about.** It removed `openslide-bin`, silently
  breaking `import openslide` in the local venv (the declared dependency is `openslide-client`, whose
  provider you pick yourself). `uv add openslide-bin` puts it back.
- **Pods must clone the repo under its real name** `RationAI/breast-cancer-grading`; GitHub's redirect
  from the old underscore name died mid-day, so pods trained on nothing until the clone was fixed and
  made to fail loudly (`d857670`, `491413a`).
- **kube-jobs**: joins script lines with `;` (a line ending in `&` becomes `&;` → syntax error, close
  background jobs with `true`); resubmitting a name needs the old job deleted first; ~10 min to
  schedule; `TRAIN_CPU`/`TRAIN_MEMORY` are the packing knobs; the cluster injects
  `MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow` into every job.
- **MLflow is reachable from the laptop**, not only in-cluster:
  `https://mlflow-s3.rationai.cloud.trusted.e-infra.cz` (same server as the in-cluster
  `http://mlflow-s3.rationai-mlflow`). Only the shared PVC — checkpoints and prediction CSVs — needs a
  pod. Metric keys are `test/accuracy` (balanced), `test/auroc`, `test/f1` (macro), `test/kappa`,
  `test/recall_class_<0..3>`, `valid/loss`, `epoch`.
- **Experiment 82 holds ~27 phantom `RUNNING` runs.** The provenance autologger opens side runs (tags
  `user_run_id`, `prov_start_time`; metrics `epoch`, `lr-Adam`, `train/loss_step` only) that inherit
  the `variant` hyperparam and never finish. Filter on the presence of `test/kappa`, or `epoch >= 8`.
- **`variant` is a param, not a tag**, and underscores/dashes differ between systems: MLflow stores
  `full_ord_s456`, the checkpoint directory is `ml_attention-full-ord-s456`. Search
  `params.variant = '…'`; `tags.variant` returns nothing.
- **Verify Hydra keys against the composed config** (`python -m ml.train --cfg job --resolve`).
  Invented keys (`ml.batch_size`) kill the run at parse time. Real ones: `datamodule.batch_size`,
  `datamodule.num_workers`, `datamodule.{train,val,test}.cache_mb`, `datamodule.train.max_tiles`
  (`null` = whole bags), `ml/aggregator=mean`, `ml/loss=ordinal`, `seed`,
  `model.predictions_dir`.
- NAS throughput measured 565 MB/s aggregate — not the bottleneck; per-file parquet decode is (that is
  what the bag cache exists for, `e416621`).
- **Tests here are import-time assert scripts, not pytest functions**: run each file directly
  (`.venv/bin/python tests/test_x.py`). `pytest tests` collects **nothing** (no `__init__.py`,
  non-recursive default) — `consider_namespace_packages = true` in `pyproject.toml` fixes that, and
  `pytest` is now a declared dev dependency (`pytest -q tests` → 55 passed).

## Not known

- Whether an ensemble of seeds beats the best single run here — the dumps exist for four variants,
  the ensemble/McNemar script does not.
- Whether 2048 tiles is the right budget (only 2048 vs "all" have been compared; 1024 and 4096 untested),
  or whether attention weights attend to anything morphologically meaningful (no attention-map
  inspection, no slide-level visualisation).
- Calibration, and behaviour on external data (bc20-24 only, single scanner estate).
- Any comparison to published BCI / bc20-24 baselines — no external baseline has been run.
