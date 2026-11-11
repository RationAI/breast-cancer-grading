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

**The split is case-disjoint, measured rather than assumed** (`scripts/case_leakage.py`, job
`brc-grading-leak`): 1735 slides carry **1734 distinct case prefixes** (the one duplicated case sits
twice in `train`), and **0 of 266 test slides and 0 of 261 val slides share a case with train**. So
the headline numbers are not inflated by two sections of one patient sitting on both sides — the
usual way a slide-level grading score is fake. It cannot detect same-patient slides carrying
different case prefixes, which the ID scheme would have to hide.

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

All at the config the last batch settled on — **that is what was intended; see the correction below** —
so every arm was meant to be paired against a finished run at the **same seed**, and all of them dump
per-slide predictions so they can join the ensemble afterwards. Pod shape 6 CPU / 32 Gi / whole A40,
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

**Correction, read this before quoting any number below: 7 of the 8 arms never got the tile knob.**
The submitted command lines contain `ml/aggregator`, the loss, `seed`, `cache_mb`, `model.…` — and
`max_tiles` **only for `t4096`** (checked in the job YAML on 2026-11-11, `.spec…command`). The other
seven trained on **whole bags**, the config default `max_tiles: null`, despite the `t2048-` prefix in
their variant names, so MLflow logs `max_tiles=None` for them and the `t2048_*` names are a lie. Their
baselines are therefore **`full_s456` 0.7372** and, at seed 123, `tiles2048_s123` **0.7486** — which is
itself mis-named and also ran whole bags (`max_tiles=None`), the only whole-bag CE run at that seed.
Paired against the right baselines, and ordered by the gain:

| arm | test bal. | true baseline (same seed, whole bags, same loss) | Δ |
|---|---|---|---|
| `t2048_dr02_s456` (head dropout 0.5 → 0.2) | **0.7956** | 0.7372 | **+5.8** — and the best single run we have |
| `t2048_acc_s123` (select on `valid/accuracy`) | 0.7846 | 0.7486 | +3.6 |
| `t2048_w_s123` (class-weighted CE) | 0.7798 | 0.7486 | +3.1 |
| `t2048_acc_s456` | 0.7661 | 0.7372 | +2.9 |
| `t2048_ord_s123` (ordinal) | 0.7690 | 0.7486 | +2.0 |
| `t2048_t4096_s456` (4096 tiles — the only arm that got its knob) | 0.7556 | 0.7372 whole / 0.7743 at 2048 | +1.8 / −1.9 |
| `t2048_wd2_s456` (`weight_decay` 1e-2) | 0.7437 | 0.7372 | +0.7 |
| `t2048_w_s456` (class-weighted CE) | 0.7475 | 0.7372 | +1.0 |

Every one of those Δs is measured against a **single** baseline run at one seed, and the seed spread on
an identical config is ~6 points — so a favourable Δ is a hypothesis until the arm has its own second
seed, which is exactly the rule below. The 2048-vs-whole-bags question is also not settled by this
batch, because it never tested 2048.

The loss-selection knob deliberately leaves the LR scheduler on `valid/loss`: optimise on the
loss, *choose* by the metric we report. `valid/accuracy` is balanced accuracy on 261 val slides,
where c1 has 16 slides (6.25-point steps), so selection gets noisier — that is the risk in that
pair of arms, not a bug.

### First ensemble numbers (`brc-grading-ensemble3`, 2026-11-11, CPU only)

With the epoch pairing fixed, the four variants that have dumps (whole bags: `full_s456`,
`full_ord_s456`, `full_ord_s789`, `t2048_ord_s456`) averaged their softmax probabilities and were
judged on the same 266 test slides:

| | test bal. | AUROC | macro-F1 | kappa | recalls |
|---|---|---|---|---|---|
| best single member (`t2048_ord_s456`) | 0.7693 | 0.9544 | 0.7187 | 0.8994 | 0.99/0.79/0.57/0.73 |
| **ensemble of 4** | **0.7856** | **0.9562** | **0.7550** | **0.9195** | 0.99/0.71/0.69/0.75 |

+1.6 points over its own best member, the best AUROC and macro-F1 in the whole experiment, for CPU
minutes and no training. It does **not** yet beat `t2048_dr02_s456` (0.7956), which is one seed of one
unconfirmed knob — so the honest statement is: ensembling is free and it works, and the open question
is whether dropout-0.2 runs join it or stand alone.

The uniform-prior decision rule is member-dependent, which is the finding:

* `full_s456` (plain CE): λ\* = 0.8 from validation → test **0.7372 → 0.7768**, +4.0 points, one run.
* `full_ord_s789`: λ\* = 0.1 → test 0.7432 → 0.7611, +1.8. `full_ord_s456` and `t2048_ord_s456`: λ\* =
  0.0 — where the ordinal loss already moved the boundary, there is nothing left to correct.
* the ensemble: λ\* = 0.0, validation curve monotonically *down* from 0.820, so averaging probabilities
  had already absorbed the prior bias the rule corrects. Applying λ = 1 to it costs 3.9 points.

So "the prior rule buys 4 points" is true of the CE runs and false of the ensemble and the ordinals;
report it per recipe, never as a blanket gain. Two more ensemble jobs are queued to see whether the
overnight arms (`brc-grading-ens-new`, all 8) and a best-4 blend (`brc-grading-ens-best4`) beat 0.7856.

**Reading them in the morning** — MLflow is reachable from the laptop:

```bash
cd ~/Documents/rationAI/breast-cancer-grading
MLFLOW_URL=https://mlflow-s3.rationai.cloud.trusted.e-infra.cz MLFLOW_DISABLE_AGENT_HINT=1 \
  PYTHONPATH=. .venv/bin/python scripts/all_runs_table.py    # every finished run, recalls included
# then re-ensemble everything that now has dumps (CPU pod, ~3 min; delete the old job first):
REPORT_JOB_NAME=brc-grading-ensemble3 REPORT_SCRIPT=ensemble_variants.py \
REPORT_VARIANTS=full_s456,full_ord_s456,full_ord_s789,tiles2048_s456,t2048_ord_s456,t2048_w_s456,t2048_ord_s123 \
  uv run --no-sync python scripts/run_report.py
```

### The ensemble script found "no dumps" — my off-by-one, now fixed

`brc-grading-ensemble` ran to completion and reported **no dumps for all four variants**, while
`report_variants.py` listed test CSVs for the same variants minutes earlier. I settled it by
`kubectl exec` into a live training pod with the share mounted (no job spent): the share holds
`test_epoch18.csv` next to `valid_epoch0..17.csv` — Lightning bumps `current_epoch` before the final
test — and `load_variant()` demanded a *valid* dump with the **same number** as the test one, so it
returned `None` for every variant. Fixed to take the test frame from the single `test_epoch<N>.csv`
and pick the validation frame by **best validation balanced accuracy** (which is also the honest way
to choose what λ is fitted on). `tests/test_ensemble_pairing.py` reproduces the exact numbering
mismatch. So: **no ensemble or prior-correction number exists yet** — that is the first thing the
rerun above produces.

## What we do with the results (bar fixed before we look)

Every arm is judged **paired against its own seed's baseline**, on balanced accuracy plus the four
recalls and macro-F1, with AUROC and kappa checked for regression. Adopt only if **both seeds agree
and the gain is ≥3 points** and c0's recall is not what moved; below that the finding is written as
"no effect", not as "promising, let's re-run" — the same config spans 6 points across three seeds, so
a single favourable run is a hypothesis. λ and any threshold are fitted on validation only; the test
split is touched once per configuration. Whatever is adopted goes into `configs/` as the new default
recipe, not into a note, and the retired arms stay listed here so nobody pays for them twice.

| result | what we do next | cost |
|---|---|---|
| prior rule and/or ensemble ≥3 points | becomes **post-processing, not a config**: fit λ on val, emit per-slide grades + a screening score, apply it retroactively to every run with dumps, and report AUROC + corrected recalls as the headline. It also means the model ranks better than its argmax is used → ranking losses / calibration get another shot | CPU minutes |
| class weights win at both seeds | third seed + an oversampling arm to prove the mechanism, then change the default loss and the roadmap becomes "more class-1 slides", not "better head" | 2–3 A40 jobs |
| selecting on `valid/accuracy` wins | config-only, becomes the default monitor; the best recipe gets a fresh 3-seed run under it and older numbers are marked not comparable | 3 A40 jobs |
| weight decay / dropout / 4096 tiles win | one more point on that curve + second seed, then it is the recipe | 2–4 A40 jobs |
| **nothing clears the bar** (the likely outcome) | the head is not the limit → stop tuning heads, fund **k-fold** (resolution) and a **features probe** instead | see below |

## Agreed as the next candidates, not queued

1. **Learning-curve probe — train on 50 % vs 100 % of the training slides, everything else equal.**
   The only test in the plan that addresses the big question: flat ⇒ frozen Virchow2 tiles at one
   magnification are the ceiling and the next money goes to features; climbing ⇒ data volume is the
   lever. Needs a slide-subset knob (`datamodule.train` has no such parameter today), ~30 lines.
2. **k-fold over the 1735 slides.** Test is 266 slides, so the ruler reads ±2.6 points while the seed
   spread is 6 — half of what we queue exists only to find out whether effects are measurable. Five
   folds turn a ±2.6 ruler into ~±1 and make one job per arm decisive. ~1 day of split code.
3. **Random-length sub-sample (uniform 1024–4096 tiles per bag per epoch) to clear the 2048 win.**
   Tile count alone scores 0.579 accuracy / 0.509 QWK on test (earlier probe, not re-verified this
   session), i.e. tumour area is genuine signal; a *fixed* 2048 cap truncates the three tumour grades
   to ~20 % of their tiles and leaves c0's small bags alone, so part of the 2048 advantage could be
   bag-length handling rather than bag augmentation. One job. Not the confound people assume: a
   fixed cap *compresses* the size distribution, it does not encode the label.

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
- **Tests here are import-time assert scripts, not pytest functions**: run `./tests/run_all.sh`, which
  runs every file with `PYTHONPATH=.:scripts` and prints the line each one ends with (`tests/test_x.py`
  on its own works too). `pytest -q tests` collects **zero items** even with `pythonpath`
  set in `pyproject.toml` (tried, reverted), because the files contain no `test_*` functions — so
  **"no tests ran" is not a pass, and a pass count from pytest has never described these checks**.
  That is how `tests/test_kfold_labels.py` shipped for half a day with zero assertions in it: a heredoc
  mishap replaced the fold checks with a helper, and every run of the suite reported nothing wrong.
  `run_all.sh` exists so the next such hole cannot stay silent; a new test file is only proven once it
  prints its own line there.

## Post-hoc analysis, 2026-11-11: what survives being chosen on validation

The overnight arms plus the older four were blended three ways, and the first finding is about the
blending itself, not the model. Until today `ensemble_variants.py` fitted the decision rule on
whatever validation epoch scored best, while the test frame comes from `best.ckpt` — a *different*
epoch — so the rule was fitted on one model and applied to another's output, and validation numbers
were flattered by cherry-picking an epoch. `best.ckpt`'s filename carries no epoch (`filename: best`)
so it is read out of the checkpoint now; the val numbers dropped 2–4 points across the board. On the
19-epoch run the tested checkpoint is epoch 17 of 18, which is why the first ensemble job found
"no dumps": it looked for `valid_epoch18`.

Chosen on validation, evaluated on test once (`brc-grading-ens-greedy`, 12 members):

| recipe (all selection on val only) | val | test bal. | AUROC | kappa | recalls |
|---|---|---|---|---|---|
| greedy blend = `acc_s456` + 2×`acc_s123`, λ=0.2 | 0.8451 | **0.7896** | 0.9468 | 0.9005 | 0.98/0.86/0.59/0.73 |
| all 12 averaged, λ=0.5 | 0.8244 | 0.7873 | **0.9525** | 0.8896 | 0.98/0.86/0.57/0.75 |
| best single on val (`t2048_acc_s456`) + bias/temperature | 0.8244 | 0.7849 | 0.9465 | 0.9037 | 0.98/0.86/0.55/0.75 |

- **Yesterday's 0.8002 was selection luck.** Picking the eight-member set by eye after reading test
  gave 0.8002; the same procedure with the member set chosen on validation gives **0.7896**, and the
  zero-choice version (average everything, λ from val) gives 0.7873. The honest post-hoc gain over
  the best single run (0.7743) is **+1.3 points**, not +2.6.
- **Per-class thresholds (idea #2) did nothing.** Fitting a temperature + four class offsets — the
  general form the scalar prior rule is a special case of — gave 0.7896, identical to the scalar λ,
  and made the 12-member blend *worse* (0.7782 vs 0.7873). Five parameters on 261 validation slides
  (16 of them grade 1) overfits exactly as feared. Stop paying for it; the scalar rule is enough.
- **`dr02_s456` (dropout 0.2, test 0.7956) is not chosen by validation at all** — its corrected val
  is 0.7769, mid-pack. Either its test edge is luck or val and test disagree; a single seed cannot
  tell those apart, which is the argument for the folds below.
- Greedy picked the two `valid/accuracy`-monitored runs and nothing else, i.e. validation prefers a
  different recipe than test does. With a ±2.6-point ruler that disagreement is unresolvable — it is
  not a signal to act on, it is the noise floor.

## k-fold: the ruler, built

The test set is 266 slides, so every number above carries ±2.6 points at 95%, while the same config
moves ~6 points across seeds. Most of the overnight arms could not have answered their own question.

- `scripts/make_kfold_labels.py` writes stratified folds as ordinary labels CSVs, so nothing under
  `ml/` changed; folds are stratified because grade 1 has 68 training slides and an unstratified fold
  can come out with none. `tests/test_kfold_labels.py` checks the folds are disjoint, cover the pool,
  leave `test` alone, and load through `GradingDataset`.
- Written on the share by job `brc-grading-kfold`: `/mnt/projects/breast_cancer/brc_grading/kfold/labels-kf{0..4}.csv`,
  each train 1174–1178 / val 291–295 / test 266, per-fold train counts ≈ 575/67/247/285.
- `ensemble_variants.py --oof` concatenates the folds' held-out fifths (~1469 slides) and fits the
  decision rule leave-one-fold-out, so the pooled number includes fitting the rule. It also averages
  the folds' test dumps into one k-fold model, with λ from the out-of-fold rows only. Pass
  `--train-counts` — a fold trains on four fifths, so its prior is not the full-cohort one.
- Cost of the ruler, measured rather than assumed (2026-11-11). A whole-bag MIL run takes **4-5 GB of
  the 48 GB card at 0-12 % utilization** — parquet-decode bound, not GPU bound — so the packing limit is
  host RAM and CPU, not VRAM. **3 concurrent whole-bag runs in a 24 Gi pod were OOMKilled in epoch 0 at
  7 min**: the hog is batches in flight (`batch_size` × bags of 61-336 MB fp32 × workers), not the bag
  cache. **2 runs per pod at `datamodule.batch_size=4`, `num_workers=2`, 32 Gi** run at ~4 min/epoch
  (294 steps over 1175 slides). One job per fold carrying both recipes is therefore the cheap shape:
  **5 jobs for 10 trainings**, paired inside each pod, instead of 10 jobs.
- Ran: jobs `brc-grading-train-mil-attention-kf{0..4}`, each packing `<fold>-ctl` (today's
  defaults) and `<fold>-cand` (`ml.head.dropout=0.2`) on the same fold CSV. The candidate is dropout
  only — `configs/data/grading.yaml` documents that sub-sampling to 2048 tiles truncates the three
  tumour grades to ~20 % of their tiles while leaving grade 0 nearly intact, so bag size itself leaks the
  label (log tile count alone scores 0.579 accuracy / 0.509 QWK on test). The 2048-tile idea is dead on
  that evidence, not on the −1.9 measured by the one arm that actually got the knob.
- Reading it when they finish: `REPORT_JOB_NAME=brc-grading-oof REPORT_SCRIPT=ensemble_variants.py
  REPORT_VARIANTS=kf0-ctl,kf1-ctl,kf2-ctl,kf3-ctl,kf4-ctl REPORT_ARGS="--oof --train-counts
  575,67,247,285" uv run --no-sync python scripts/run_report.py`, and the same with `-cand`. The
  `--train-counts` is the fold's **train** counts (1174-1178 slides, not the shipped 940); its val
  split is 291-295 slides. A real effect is a few points, not one: a
  fold is 295 slides, and even the whole 1469-slide pool only reaches ±1.

## Not known

- ~~Whether an ensemble of seeds beats the best single run~~ **settled by the folds**: the 5-fold
  average of one recipe scores 0.7959 on test against 0.7743 for the best single seed and 0.7896 for
  the validation-chosen 12-run blend, and it needs no seed choice at all. Take the fold average.
- Whether 2048 tiles is the right budget (only 2048 vs "all" have been compared; 1024 and 4096 untested,
  and the 5-fold wave tested **neither** - `cand` was dropout only, no tile override in the job spec),
  or whether attention weights attend to anything morphologically meaningful (no attention-map
  inspection, no slide-level visualisation).
- Calibration, and behaviour on external data (bc20-24 only, single scanner estate).
- Any comparison to published BCI / bc20-24 baselines — no external baseline has been run.

## Cluster headroom, read from monitoring instead of guessed (2026-11-11)

`run_train.py` says a 48 Gi request "does not schedule on a GPU node" and the first pack today was sized
at 24 Gi from that memory. Both are wrong about *why*. The A40 pool, read from the cluster's Prometheus
(via Grafana's anonymous datasource proxy — see `STATUS.md` recipe below / the `check-cluster-headroom`
skill):

| | per node | across the 7 A40 nodes |
|---|---|---|
| RAM | 496 Gi total, **352–430 Gi free** | 2 816 Gi free |
| CPU | 64 cores, **31–51 free** | 288 cores free |
| GPUs | 1, except `kub-as10` / `kub-as11` with **2 A40s each** | 9 cards, mean **18 % utilization** |

So host RAM is *not* scarce and never was the node limit. The real ceiling is the **namespace quota**:
`requests.memory` hard = **1 554 GiB** for `rationai-jobs-ns`, 746 GiB of it used by other jobs at the
time of reading (`requests.cpu` 402 hard / 156 used). A 48 Gi pod getting "resource is out of usage:
memory" was that budget being momentarily spent, not a 64-core node with 400 Gi free refusing it.
**Check `kube_resourcequota` before a fat submission, not folklore.**

What the cards are actually doing while our five fold pods run: each pod holds a whole A40 and uses
**11–20 GiB of 46 GiB at 0–56 % utilization**; a `nuclei-graph` job elsewhere on the pool is at 100 %.
So "fill the VRAM" is the wrong target — framebuffer held is not work done. These runs are parquet-decode
bound (a 2-run pod uses 2–4 of its 8 cores, i.e. the workers wait on I/O, not on the SMs), and the metric
to chase is utilization and slides/s. Concurrency does lift it, so the shape for the next batch is one
card per pod with **6 runs per pod** (measured ~6 GiB framebuffer and 8–12 GiB RAM per whole-bag run →
36 GiB VRAM, ~70 Gi RAM, ~16 cores): 5 folds × 2 recipes in **2 pods** instead of 5, with the quota
checked first. 100 % utilization is not reachable for this workload by packing alone.

## The 5-fold wave (10 runs, 1469 out-of-fold slides): the candidate wins, the prior rule mostly does not

Both recipes pooled in one report, and for the first time compared **paired, slide by slide**
(`scripts/ensemble_variants.py --oof` now reads fold from the name prefix and recipe from the
suffix, so `kf3-ctl` and `kf3-cand` are pooled separately instead of averaged or refused). Every λ
is fitted leave-one-fold-out; the 266 test slides were read once, at the end.

| recipe | OOF argmax | OOF + rule | rule Δ | test 5-fold avg, no rule | test + rule | test AUROC | κ |
|---|---|---|---|---|---|---|---|
| `ctl` (defaults) | 0.7612 | 0.7593 | **−0.2** | 0.7566 | 0.7861 | 0.9470 | 0.895 |
| `cand` (dropout 0.2 — **nothing else**, no tile override in the job spec) | 0.7588 | **0.7775** | **+1.9** | 0.7449 | **0.7959** | 0.9528 | 0.904 |

**Paired on the same 1469 out-of-fold slides: cand 0.8393 vs ctl 0.8230 accuracy, 62/38
discordant, McNemar p = 0.021.** **`ml.head.dropout=0.2` is a small but real +1.6-point win** - and
it is the whole candidate: the job spec carries no `max_tiles`/tile override, so this wave says
nothing about tile budgets (the first draft of this section said "2048 train tiles", which is wrong).
The 4-fold interim said +0.8 OOF and −1.2 test — a sign flip, i.e. exactly the noise the pairing was
built to remove. Fold-to-fold spread is 0.044 (ctl) / 0.055 (cand): larger than the whole effect, so
no ±2-point claim from a single split has ever meant anything in this project.

1. **Ship the fold average, not a seed.** cand 5-fold average + rule = **0.7959** balanced accuracy
   / AUROC 0.9528 / κ 0.904, recalls 0.98/0.86/0.60/0.75, with λ from out-of-fold rows and zero
   test-set choices. It matches the hand-picked 12-run blend (0.7896) and beats the best single
   seed (0.7743) without the seed lottery.
2. **The prior rule is not a free +3 points.** At cohort scale: +1.9 (cand), −0.2 (ctl). It buys
   grade 1 (recall 0.50 → 0.86) and spends grade 2 (0.74 → 0.60) — it slides the operating point
   along the grade 1/2 boundary and adds no information (AUROC 0.9528 → 0.9478). Whether that trade
   is the right one is clinical, not a metric question. Not yet paired-tested: the paired block
   compares recipes, not decision rules.
3. **`patience=10` spends about half the GPU time on nothing.** Measured off the checkpoints: best
   epoch was 0–5 in all ten runs (3,0,0,5,5,4,2,2,2,4), and EarlyStopping then waited out 10 more, so the runs ended
   at 15–28 epochs. `patience=5` cannot change which checkpoint wins (`best.ckpt` is already saved)
   and would cut ~40 % of the epoch count. Epoch time itself ranged 4.8–9.5 min across the five pods
   on the same work, so these runs wait on the NAS, not on the A40.

Reproduce: `REPORT_JOB_NAME=brc-grading-oof-final REPORT_SCRIPT=ensemble_variants.py
REPORT_VARIANTS=kf0-ctl,...,kf4-cand REPORT_ARGS='--oof --train-counts 575,67,247,286'
uv run --no-sync python scripts/run_report.py`. The prior must come from the fold CSVs
(`<project>/kfold/labels-kf{K}.csv` → 575,67,247,286 ±1 slide per fold), not the shipped 1175.

## Which way the mistakes point (2026-11-11, job `brc-grading-trade`)

Balanced accuracy cannot see the *direction* of an error. Reading it off the 1469-slide
out-of-fold pool (`cand` = attention pooling, dropout 0.2, whole bags), confusion rows = the
pathologist, cols = the model:

```
no rule                                   + rule (lambda ~0.9, fitted out-of-fold)
      p0    p1    p2    p3                      p0    p1    p2    p3
t0    699     3    13     4               t0    690     9    13     7
t1      3    46    33     2               t1      3    60    19     2
t2      7    37   212    53               t2      4    65   191    49
t3      5     3    53   296               t3      5     8    52   292
```

- **Class 2 is the sink.** It is the only row that loses to *both* neighbours (37 up, 53 down) and
  it is where class 1's 33 errors land. Class 0 is essentially never confused with anything
  (20 errors in 719 slides), so "0.76 balanced accuracy" is really "near-perfect at separating
  class 0, and a coin-flip-ish 1/2/3 boundary inside the tumour grades".
- **The rule is a 1↔2 boundary move and nothing else**: class 1 goes 46 → 60 correct, class 2 goes
  212 → 191. 65 class-2 slides get read as class 1.
- **Direction, out-of-fold, `cand`**: no rule = exact 85.3 %, undercall 7.4 % (108), overcall
  7.4 % (108) — symmetric. With the shipped rule = exact 83.9 %, **undercall 9.4 % (141)**,
  overcall 6.3 % (101). On the 266 test slides the shipped artifact undercalls **34 of 266
  (12.8 %)** and overcalls 10 (3.8 %); undercalls concentrate on classes 2 and 3 (18/58, 16/63).
- **Lambda ladder** (out-of-fold rows, shown not chosen): λ=0 → bal 0.759, κ 0.924, under 108 /
  over 108. λ=1.0 → 0.774, κ 0.912, under 141 / over 101. λ=1.5 → 0.774, κ 0.899, under 163 /
  over 110. λ=2.0 → 0.753, under 202 / over 126. Balanced accuracy peaks at λ≈1-1.5 and *then
  falls while undercalls keep climbing*, so the metric cannot police this choice.

**Open decision, and it belongs to the pathologist, not to us:** if reporting a slide one grade
too low is the dangerous error, the shipped λ≈0.9 rule triples it (10 → 34 on test) in exchange
for class-1 recall 0.55 → 0.86, and **λ = 0 is the symmetric, safer artifact** at 1.9 points less
balanced accuracy and *better* κ (0.924 vs 0.913). Asking for a cost per (true, predicted) pair
turns this into arithmetic; guessing it is how we got the retracted +3.

## Frozen artifact (2026-11-11)

`/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/` — the five `cand` fold
`best.ckpt` files, their fold label CSVs and `manifest.json` (sha256, size, source path, git sha,
recipe text). Built by `scripts/freeze_artifact.py`, which refuses to overwrite a directory that
already has a manifest. Scoring a new cohort needs no new code: run `ml.train` per checkpoint with
`checkpoint=... dataset.paths.labels=<cohort csv> eval.split=test`, then average the per-slide
softmaxes — the same averaging `scripts/ensemble_variants.py` does across folds.

## The paired lambda test: ship lambda = 0 unless the pathologist asks otherwise (2026-11-11, job `brc-grading-lambda`)

Same 1469 out-of-fold slides, same model, two operating points. Discordant pairs, not averages:

| recipe | raw accuracy, no rule vs rule | discordant | McNemar | flips **into** undercall | out of undercall | under calls |
|---|---|---|---|---|---|---|
| cand (dropout 0.2) | **0.8530 vs 0.8393** | 40 / 20 | **p = 0.0135** | **31** | 2 | 108 → 137 |
| ctl | **0.8482 vs 0.8230** | 49 / 12 | **p < 0.0001** | **42** | 1 | 125 → 166 |

On the 266 test slides the same direction, not significant on its own: cand 11/6 (p = 0.33), 9 new
undercalls and none removed; ctl 11/4 (p = 0.12).

**Read this against the earlier "+1.9 points from the rule".** Both are true and they disagree
because the metrics disagree. Balanced accuracy *rises* 0.7588 → 0.7775 because the rule trades 21
slides of the 309-slide class 2 for 14 slides of the 84-slide class 1 and balanced accuracy weights
those classes equally. Counted slide by slide the rule is **significantly worse at naming the grade
correctly**, in both recipes, and it creates roughly fifteen undercalls for every one it removes.
So the honest sentence is: *the rule buys recall on the rarest grade by spending per-slide
correctness, and it spends it in the direction that matters.*

Recommendation for the handoff, pending a cost matrix from the pathologist: **ship the 5-fold
average with no prior correction.** On test that artifact is balanced accuracy 0.7449, AUROC
0.9528, quadratic kappa 0.928, raw accuracy 0.8534, undercalls 25/266 (9.4 %) against 34/266
(12.8 %) for the corrected one. Choosing λ = 1.5 for a better-looking balanced-accuracy column
would triple the dangerous errors on 1469 slides, which the ladder showed and the paired test
confirms. The frozen manifest's note currently says "prior rule lambda fitted out-of-fold"; treat
that as superseded by this section - the manifest gets rewritten when the winning arm is frozen
after the tile wave, and no new directory is needed for a note.

## "Miss nothing, few false alarms" = escalate, and the price list (2026-11-11, job `brc-grading-escalate`)

The stated objective is: never report a slide lower than it is, at a modest cost in false alarms.
Two things had to be settled before spending GPU on it.

**1. The words "miss grade 3" point two opposite ways, and the model's errors are not where the
phrase suggests.** Out-of-fold confusion, row class 1 (= B-RAD grade 3), 84 slides:
`3 -> c0, 46 correct, 33 -> c2, 2 -> c3`. A true grade-3 slide is called benign/grade-2 **3 times in
84**; three quarters of its errors go *upward* to grade 4. Detection of "anything above grade 2" is
already **98.0 %** (15 misses in 750 tumour slides) and only 20 of 719 bottom-class slides are
escalated. Where the dangerous direction actually lives is the top of the scale: 44/309 class-2 and
61/357 class-3 slides are read *lower* than they are. So "do not miss grade 3" is nearly free
already; "do not miss anything" means the grade-4 and grade-5 undercalls.

**2. The prior rule is the wrong knob for it and escalation is the right one.** λ>0 moves mass
toward the *rarest* class, which sits below class 2 - it manufactures grade-4-read-as-grade-3
errors. Escalation (`log p_c + beta * c`, in `ensemble_variants.escalate`) biases every call
upward. Price list, 1469 out-of-fold slides, `cand`, base λ = 0:

| β | detect >c0 | missed | false alarms | undercalls | overcalls | bal | κ |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.9800 | 15 | 20 | 108 | 108 | 0.7588 | 0.924 |
| 0.20 | 0.9827 | 13 | 25 | 92 | 125 | 0.7508 | 0.920 |
| **0.50** | 0.9827 | 13 | **31** | **76** | 143 | 0.7443 | 0.912 |
| 0.75 | 0.9867 | 10 | 37 | 63 | 165 | 0.7305 | 0.904 |
| 1.00 | 0.9880 | 9 | 55 | 57 | 200 | 0.7108 | 0.871 |

"False alarm" here = a grade-2 slide escalated to grade 3 or above, i.e. the ones that cost the
pathologist time on a benign slide: 31 at β = 0.5 is 4.3 % of the 719 bottom-class slides.

**Reading: β ≈ 0.5 is the knee.** It removes 30 % of the dangerous errors (108 → 76) for 11 extra
false alarms and 1.4 points of balanced accuracy; β = 0.75 removes 42 % for 17. β = 1.0 costs 35
false alarms and 4.8 points to save the last 51 → 9 misses, and it *reduces* class-1 recall to 0.39
because escalating pushes real grade-3 slides into grade 4 - the one way the stated goal and
"correct grade name" genuinely conflict. Note also that **99 % detection is unreachable with this
bias** (best 98.8 % at the grid maximum), so the report's auto-pick printed β = 1.0; that is a grid
artefact, now labelled as one in the output.

Nothing was trained for this: one model, one frozen artifact, and an operating point that is a
addition of `beta * class_index` in log space at inference. Decide β, and it can be written into the
manifest when the winning arm is frozen.
