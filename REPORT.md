# Nottingham grading from Virchow2 embeddings — what we know, and what each number means

Findings file, as of 2026-11-12. `STATUS.md` holds current state, job plumbing and what to run next;
this file holds the results and explains them. Every number here comes from MLflow (experiment **82
"BRC Grading"**), a per-slide prediction CSV on the shared PVC, a pod log, or a `kubectl` reading.
Estimates are labelled. Nothing is quoted from memory of a paper — where a published figure is used it
was fetched and is cited inline.

---

## 1. The task, in one paragraph

One whole-slide image (WSI) of breast cancer goes in; one Nottingham (Elston–Ellis) grade comes out.
The slide is cut into 224 px tiles at 0.5 µm/px, each tile is embedded into 2560 dimensions by a
**frozen** Virchow2 encoder (that part costs the most money, so it is pre-computed once and cached),
and what we train is a **17 k-parameter attention head** over the bag of tiles — multiple-instance
learning (MIL): the slide has one label, the tiles have none, and the head learns which tiles matter.
Training an arm means training that head; the encoder never moves. The whole model per slide is thus
"look at ~6500 tile embeddings, weight them, name the grade".

## 2. What the classes are — and a correction to old notes

From `scripts/make_cohort.py` and `README.md`: `grade` is **0 = no tumour epithelium (an `-N` slide),
1 = Nottingham G1, 2 = G2, 3 = G3**. Grade 9 (not assessable) and neoadjuvant cases are filtered out;
`-T` slides carry the case's Nottingham grade, `-N` slides are class 0.

> **Old notes in `STATUS.md` said "classes 0–3 are B-RAD/Nottingham grades 2–5", and one section called
> class 1 "B-RAD grade 3". Both are wrong.** The counts and percentages in those sections are unaffected
> — only the grade names attached to them were. Where this file says G1/G2/G3 it means Nottingham
> grade 1/2/3, and class 0 means "this slide shows no tumour epithelium".

That correction also sharpens what the model is good at: the biggest separation in the data is **tumour
versus no tumour**, which the model has nearly solved, and the hard part is the **grade inside the
tumours**, which is where all the remaining error lives (§8).

## 3. How to read every number in this file

These are the five metrics we log, what each can and cannot see, and the size of the noise that any
one of them carries. Read this before quoting any result.

| metric | what it is | what it is good for | what it hides |
|---|---|---|---|
| **balanced accuracy** | mean of the four per-class recalls. **This is what MLflow logs as `test/accuracy`** — it is not raw accuracy | rare classes matter as much as common ones, which is right when G1 is 5 % of the cohort | it weights a class of 84 slides exactly like a class of 719, so it can improve while the model gets *fewer* slides right (§9) |
| **raw accuracy** | fraction of slides named correctly | what a pathologist would count on a stack of slides | dominated by class 0; our raw accuracy is ~0.85 while balanced is ~0.76 |
| **per-class recall** | correct share within each class | the only way to see *which* grade the model cannot do | nothing — always read these together with balanced accuracy |
| **AUROC** | how well the probability scores rank a slide above its neighbours, ignoring the threshold | whether the model *knows* something even where its argmax is wrong; the number to compare between loss functions | nothing about where the decision boundary sits |
| **quadratic kappa (κ)** | agreement corrected for chance, penalising a two-grade slip four times a one-grade slip | ordinal tasks — grading, not colour-classification | it saturates here: every run we ever trained lands at κ 0.89–0.92, including runs with balanced accuracy 0.67 |

**The noise floor — the most important number in this file.** On 266 held-out slides, the binomial
95 % error bar on a balanced accuracy of 0.75 is **±2.6 points**. But retraining the *identical*
configuration at three different random seeds gave 0.7743 / 0.7618 / 0.7147 — a **6.0-point spread**.
Seed noise is larger than sampling noise, so **a single run has never been able to rank anything in this
project**, and no ±2-point claim from one split means anything. What we do about it:

- **Two or three seeds minimum** before a result is called real; one favourable arm is a hypothesis.
- **Paired tests, not averages.** Both arms judge the *same* slides, so most of the binomial term
  cancels. We count the slides where the arms disagree (discordant pairs) and run **McNemar's test**;
  that is how dropout 0.2 was proven at p = 0.021 while other "wins" of 3 points evaporated.
- **One 5-fold wave instead of a 12-arm sweep.** Five folds over 1469 slides give a ~±1-point ruler
  where the single split gave ±2.6. Most of the arms we ran before the folds existed could not have
  answered their own question — that is why the graveyard in §7 is so large.
- **Choose on validation, report on test, once.** λ, per-class thresholds and member sets must be
  picked on validation *only*. When we broke this rule we published a +2.6-point ensemble that
  redemocratised to +1.3 (§14).
- **One slide of G1 moves the headline 1.8 points.** G1 has 14 test slides, so its recall moves in
  7.1-point steps. Expect G1 to look erratic even between identical runs.

## 4. Headline: what we would ship today

**The 5-fold average of one recipe — attention pooling, dropout 0.2, whole bags, plain CE — with no
decision rule bolted on.** On the 266-slide test split, with every choice (weights, dropout, averaging)
made out-of-fold and **zero test-set choices**:

| | balanced | AUROC | raw acc | κ | recalls (N/G1/G2/G3) | undercalls |
|---|---|---|---|---|---|---|
| shipped artifact | **0.7449** | **0.9528** | 0.8534 | **0.928** | 0.98 / 0.55 / 0.69 / 0.83 | 25 / 266 (9.4 %) |
| same weights + prior rule λ≈0.9 | 0.7959 | 0.9478 | 0.8391 | 0.913 | 0.98 / **0.86** / 0.60 / 0.75 | 34 / 266 (12.8 %) |

In plain language: **it almost never mistakes a tumour slide for a tumour-free one (98 %), it is
reliably right about G3 (83 %), it is mediocre on G2 (69 %), and it can name G1 correctly only about
half the time.** Ranking is strong — AUROC 0.95 and κ 0.928 mean that even when the name is wrong it is
usually the *neighbouring* grade, and the scores do order slides sensibly. The name is the weak part,
not the ranking.

The second row is the tempting one: 5 more points of balanced accuracy. **We recommend against it** —
counted slide by slide it is *significantly worse* at naming grades (McNemar p = 0.0135) and it creates
about fifteen undercalls for every one it removes (§9). Choosing the higher balanced-accuracy column
triples the errors that matter. That finding was retracted once and re-derived twice (§14); it is the
single most important thing on this page after the recall row.

Location: `/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/` — five
`best.ckpt`, their fold label CSVs, `manifest.json` (sha256, git sha, recipe). To score a new cohort:
run `ml.train` per checkpoint with `checkpoint=… dataset.paths.labels=<cohort csv> eval.split=test`,
then average the per-slide softmaxes — the same averaging `scripts/ensemble_variants.py` does. No new
code needed.

## 5. The data (bc20-24)

1735 slides from one scanner estate, one parquet of per-tile Virchow2 embeddings each (median bag 61 MB,
p90 144 MB, max 336 MB).

| split | N | G1 | G2 | G3 | total |
|---|---|---|---|---|---|
| train | 593 | 68 | 254 | 293 | 1208 |
| val | 126 | 16 | 55 | 64 | 261 |
| test | 131 | 14 | 58 | 63 | **266** |

- **G1 is 5.6 % of training slides.** This is why balanced accuracy is the headline metric and why G1's
  recall is the least stable number we report.
- **Tiles per slide are strongly class-correlated**: p25 2.3 k, median 6.5 k, p75 12.3 k, max 27 k, with
  class-0 bags at a median 2.5 k versus 9–11 k for the tumour grades. Log tile count *alone* scores
  0.579 raw accuracy / 0.391 balanced / 0.509 κ on test. Tumour area is genuine signal, and any
  bag-length-dependent trick gets checked against that baseline.
- **Case-disjoint, measured not assumed** (`scripts/case_leakage.py`, job `brc-grading-leak`): 1735
  slides carry 1734 distinct case prefixes, and **0 of 266 test and 0 of 261 val slides share a case
  with train**. This is the usual way a slide-level grading score turns out to be fake, and it is not
  happening here. It cannot detect same-patient slides carrying different case prefixes.
- **Only 15 % is held out**, which is the resolution problem §3 describes. The k-fold cohort (five
  folds over the remaining 1469 slides) is our answer for measurement; it is not extra data.

## 6. What changed the model, and by how much

Four things moved the needle. Three are structural, one is a one-number knob.

| change | evidence | size | verdict |
|---|---|---|---|
| **attention pooling instead of mean pooling** | 3 seeds at matched config: +9.5 / +4.8 / +4.3 balanced points, 3 wins of 3 | **large** | shipped. Mean pooling cannot see G1 at all (recall 0.29–0.43) because averaging over a bag dominated by stroma washes the rare tumour signal out |
| **head dropout 0.5 → 0.2** | paired on 1469 out-of-fold slides: 0.8393 vs 0.8230 raw accuracy, 62/38 discordant, **McNemar p = 0.021**; replicated on an independent split | **+1.6** | shipped. The only hyperparameter in this project that beat its own null |
| **averaging the 5 folds instead of shipping one run** | fold average 0.7959 (with rule) / 0.7449 (without) on test vs **0.7743** for the best single seed and 0.7896 for a hand-picked 12-run blend | **+1.3** vs best seed | shipped, and it needs no seed choice at all. "Which seed?" stops being a question |
| **ordinal cross-entropy** (reads the same 4 logits as 3 cumulative "grade > k" questions, so a two-grade slip costs 4× a one-grade slip; no new parameters) | all three ordinal runs took the top three AUROCs (0.9511–0.9544) and flatter recalls, but accuracy was a wash (one result each way) and κ got worse | ranker only | not adopted, not rejected. A second seed at the kept config would settle it for one job |

Two measurement notes that matter more than the effects: whole-bag training **won** on the paired split
(the earlier "sub-sample to 2048 is better" result came from an unequal comparison — §7, §14), and
`patience=10` wasted about half the GPU time: best epoch was 0–5 in all ten runs and early stopping then
waited out ten more, so patience 5 is now the committed default.

## 7. What did not change it: the graveyard

Fifteen arms, ~84 pod-hours. Each row is a claim we no longer have to test twice. Δ is paired where
pairing was possible.

| arm | hypothesis | result |
|---|---|---|
| class-weighted CE `[0.28,2.48,0.66,0.58]` | upweighting the rare G1 lifts balanced accuracy | seeds disagreed (+3.1 / +1.0); with a paired pool it is **−2.8 points** |
| focal loss γ = 2 | hard examples drive the gradient | **−1.0** |
| per-class thresholds (temperature + 4 offsets fitted on val) | generalises the scalar prior rule for free | **no change** (identical to scalar λ); *worse* on the 12-member blend. Five parameters on 261 val slides (16 of them G1) overfit exactly as feared |
| boundary fitted for accuracy directly, leave-one-fold-out | the most learned threshold the setting allows — beats a fixed boundary | **+0.13 / −0.20, both noise** (McNemar p = 0.85 / 0.75) |
| `weight_decay` 1e-3 → 1e-2 | the documented overfit (train 0.92 vs val 0.79) | +0.7 on one seed, unmeasurable paired |
| bag cap 1024 / 4096 tiles | sweep the tile budget | whole bags win: **1024 −1.57 (p = 0.062), 4096 −3.00 (p = 0.0008)**. Dose-response is non-monotone, so these are not three points on a curve. 1024 is ~6× cheaper per epoch and statistically indistinguishable from whole bags — the knob to pull if throughput ever matters, not if accuracy does. **Confounded**: the cap was applied to train *and* val/test, while the shipped config deliberately leaves evaluation uncapped ("evaluation always sees the whole slide"), so this wave cannot separate bag augmentation during training from truncated bags at evaluation. Run it train-only if the question ever matters again |
| whole bags vs fixed 2048 | the pooling the evaluation applies should be the pooling you train | **whole bags win, paired** — the old −3.7/−1.3 "sub-sample wins" reading came from an unequal comparison |
| select/early-stop on `valid/accuracy` instead of `valid/loss` | choose by the metric we report | +3.6 / +2.9 single-seed, **greedy blend preferred those two runs and the blend did win (0.7896)** — but on a ±2.6 ruler the val/test disagreement is unresolvable noise, not a signal |
| bag augmentation (jitter/flip) | kills bag-length leakage | **rejected before spending**: no positional input exists, attention is permutation-invariant, and fixed-length tensors cannot be padded at random without adding masking. There is nothing for augmentation to break |
| slide-level features (age, laterality, tumour size) | clinical covariates add signal | open — `train_meta.parquet` was found to carry them but no arm has used them |
| encoder-swap as an arm | new features from a different encoder | **closed as an arm**: the encoder is external and pre-computed; it is a funding decision, not an experiment (§13) |

## 8. Anatomy of the mistakes

Out-of-fold confusion for the shipped recipe, 1469 slides. **Rows = the pathologist, columns = the
model.**

```
              N      G1     G2     G3
  N         699      3     13      4     (719 slides)
  G1          3     46     33      2     (84)
  G2          7     37    212     53     (309)
  G3          5      3     53    296     (357)
```

How to read it:

- **The tumour / no-tumour boundary is essentially free.** 20 errors in 719 class-0 slides, and G1 is
  called tumour-free only **3 times in 84**. "Any carcinoma present" is detected at **98.0 %** — 15
  misses in 750 tumour slides. Whatever the model is doing with tile embeddings, it can find the cancer.
- **G2 is the sink.** It is the only row that loses to *both* neighbours (37 down, 53 up) and it is
  where G1's 33 errors land. G2's recall of 0.69 is the model's real weakness, and **181 of the 216
  errors (84 %) sit inside the G1/G2/G3 block**; only 35 involve class N.
- **Balanced accuracy 0.76 is really two numbers glued together**: near-perfect at separating class 0,
  and close to a coin flip on which of G1/G2/G3 to name.
- **84 % of the errors are exactly one grade off — there are two separate error populations.** Re-derived
  straight from the per-slide dumps (not from a summary), 216 wrong slides:

| distance | slides | share of the 216 errors | share of all 1469 slides |
|---|---|---|---|
| **1 grade off** | **182** | **84.3 %** | 12.4 % |
| 2 grades off | 25 | 11.6 % | 1.7 % |
| 3 grades off | 9 | 4.2 % | 0.6 % |

  Adjacent errors are symmetric (93 under, 89 over), and **four adjacent pairs carry 82 % of every error**:
  *G3→G2* 53, *G2→G3* 53, *G2→G1* 37, *G1→G2* 33. Per class, the share of a class's errors that land next
  door: **G1 95 %, G2 93 %, G3 87 %** — but **class N only 15 %**, because a false alarm by definition
  skips the grade below it (N→G2 13, N→G3 4).

  So the two populations are: **(a) 182 grade-tie slides**, one grade off, symmetric, all inside the G1/G2/G3
  block — the label-noise-shaped population; and **(b) 34 far-miss slides (2.3 %)**, which are the clinical
  risk: **15 tumour slides read as tumour-free** (7 G2→N, 5 G3→N, 3 G1→N = 2.0 % of the 750 tumour slides),
  of which 12 are read two or three grades low, plus 20 tumour-free slides given a grade.
  *Caveat when reading the distance column:* the numeric coding makes "tumour-free read as G2" 2 grades
  away and "G3 read as G2" 1 grade away, which conflates a missed cancer with a one-grade slip. Use the
  two-population split, not the raw distance, when thinking about harm.
- **Undercalls versus overcalls are balanced at λ = 0**: exact 85.3 %, under 7.4 % (108), over 7.4 %
  (108). Symmetric — which is why §9 matters: any rule that tilts this is a *choice*, not an improvement.
- **The dangerous direction lives at the top of the scale.** "Do not miss a cancer" is already nearly
  free. What is left is G2 and G3 being read low: **44 of 309 G2 and 61 of 357 G3 slides**.

## 9. Operating points: three knobs, all at inference, none needing a retrain

All three take the frozen model's probabilities and change only the *decision*. We tested all three
paired on the same 1469 out-of-fold slides.

**λ — the class-prior correction** (`log p_c − λ·log π_c`). Corrects the prior the training split taught
the argmax. **β — escalation** (`log p_c + β·c`). Biases every call upward by class index. **Learned
boundary** — five parameters on the cumulative softmax fitted for accuracy, leave-one-fold-out.

**What we found first is that λ is the wrong knob for "never report a slide too low".** λ > 0 pushes
probability mass toward the *rarest* class — G1, which sits *below* G2 — so it manufactures
G2-read-as-G1 errors. Escalation is the knob that actually answers the question, and it has a price
list (shipped recipe, base λ = 0, 1469 slides; "false alarm" = a tumour-free slide called a grade):

| β | detect any tumour | missed tumour | false alarms | undercalls | overcalls | balanced | κ |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.9800 | 15 | 20 | 108 | 108 | 0.7588 | 0.924 |
| 0.20 | 0.9827 | 13 | 25 | 92 | 125 | 0.7508 | 0.920 |
| **0.50** | 0.9827 | 13 | **31** | **76** | 143 | 0.7443 | 0.912 |
| 0.75 | 0.9867 | 10 | 37 | 63 | 165 | 0.7305 | 0.904 |
| 1.00 | 0.9880 | 9 | 55 | 57 | 200 | 0.7108 | 0.871 |

**β ≈ 0.5 is the knee**: 30 % fewer dangerous errors (108 → 76) for 11 extra false alarms — i.e. 1.5 %
of the 719 tumour-free slides get a grade they shouldn't — and 1.4 points of balanced accuracy. β = 1.0
costs 35 false alarms and 4.8 points to go from 51 to 9 remaining misses, and it starts *converting real
G1 slides into G2* (G1 recall falls to 0.39). **99 % tumour detection is unreachable by biasing** (best
98.8 % at the grid maximum), so the report's auto-pick printing β = 1.0 was a grid artefact, now
labelled as one in the output.

**One thing the price list hides, and it matters**: β does not only move errors between neighbours. At
β = 0.5 the total error count barely moves (216 → 219) but the *far* errors grow from **34 to 41** and the
adjacent share falls 84.3 % → 81.3 %. Escalation buys fewer undercalls partly by taking bigger risks on
individual slides.

**The conflict between the two goals is real and irreducible.** The only slides β ever changes are, by
construction, the ones the model could not decide between two adjacent grades. On that pile, being wrong
high and being wrong low are mutually exclusive. The way out is to stop forcing one grade on a tie —
report a range — and we already have the per-slide probabilities that identify the ties. That option is
**not yet priced**; it is one free CPU job (§13).

**λ is member-dependent, so never report it as a blanket gain.** Fitted on validation per recipe: plain
CE `full_s456` took λ* = 0.8 and gained **+4.0** on test (0.7372 → 0.7768, one run); `full_ord_s789` took
λ* = 0.1 for +1.8; the two ordinal `s456` runs took **λ* = 0** — where the ordinal loss already moved the
boundary there is nothing left to correct; and the 4-member ensemble took λ* = 0 with a validation curve
monotonically *down* from 0.820, i.e. averaging probabilities had already absorbed the prior bias the rule
corrects. Applying λ = 1 to the ensemble **costs** 3.9 points.

**λ, the paired verdict.** Same slides, same model, two operating points, discordant pairs:

| recipe | raw acc, no rule → rule | discordant | McNemar | flips *into* undercall | out of undercall |
|---|---|---|---|---|---|
| dropout 0.2 (shipped) | 0.8530 → 0.8393 | 40 / 20 | **p = 0.0135** | **31** | 2 |
| defaults (control) | 0.8482 → 0.8230 | 49 / 12 | **p < 0.0001** | **42** | 1 |

Balanced accuracy *rises* 0.7588 → 0.7775 under the rule, because it trades 21 slides of the 309-slide
G2 class for 14 slides of the 84-slide G1 class and weights them equally. Counted slide by slide the
rule is significantly worse, in **both** recipes, and creates ~15 undercalls for every one it removes.
The λ ladder makes the point unpoliciable by metrics: λ = 0 → bal 0.759 / κ 0.924; λ = 1.0 → 0.774 /
0.912; λ = 1.5 → 0.774 / 0.899; λ = 2.0 → 0.753. Balanced accuracy peaks then *falls while undercalls
keep climbing*.

**Learned boundary: no gain.** 0.8530 vs 0.8543 (p = 0.85) and 0.8482 vs 0.8462 (p = 0.75). With β, with
λ, and now with a boundary fitted for accuracy directly, the same conclusion keeps holding: **the
ranking is where the information is; the threshold step only relocates errors between adjacent grades.**
`--oof` prints this test automatically now.

**Recommendation, pending a cost matrix from the pathologist:** ship the 5-fold average with **no
correction** (λ = 0). If asked for "never report low", use **β ≈ 0.5**, not λ. Ask for a cost per
(true, predicted) pair and this becomes arithmetic instead of a judgment call.

## 10. Can the model explain itself? (attention weights, verified)

The project's stated end-goal is to explain a genomic assay (MammaPrint) through morphology, which
requires the attention weights to mean something. Measured off the shipped artifact (job
`brc-grading-attn`, one fold, 53 slides, 43 457 tiles, 600 tiles/s on 1 CPU):

- **The weights are not degenerate**: participation (inverse perplexity of the weight distribution) is
  20–415 tiles of 866, and max weight 0.04–0.94 across slides. Attention *is* selecting.
- **It is not prognostically meaningful yet**: area under tumour-vs-stroma contrast = **0.48 — below
  chance** — at every level we tried: raw tile scores, 250 µm grid cells, and after binarising to the
  pathologist's own tumour mask (`tumor_roi`, 10× and 40× levels).
- **The tumour mask is not usable as ground truth**: it is empty on 11 of 53 slides and its polygon is
  118 µs/px to rasterise (a 25× time tax over the model itself).
- Caveat on the contrast: the available tumour flag does not discriminate tumour epithelium from
  stroma in a slide where the model attends to ~25 % of tiles, so the honest reading is "no
  demonstrable prognostic content", **not** "anti-correlated".

So attention is a real, working mechanism that currently selects tissue we cannot yet show to be the
right tissue. Fixing that needs annotation-grade tumour labels, not more head tuning — see §11, which
is where a usable label source actually exists.

## 11. External data: what exists, and what it can and cannot prove

**MammaPrint slides (`bc/mammaprint`, 88 slides, Virchow2 `.pt` embeddings ready).** 79 of 88 slide IDs
appear in our `data_mapping.csv`, but **0 of 88 are in `labels.csv`** — genuinely unseen, and the 46
`P2023_*` IDs are a different accession vintage from our `2022_*`/`2024_*`. **No Nottingham grades are
joinable to them**: the grade column in `Mammaprint_anonymizovany.xlsx` keys on record/biopsy numbers
(2025/4060 …, 856 assay records) and no sheet or mapping file connects those to slide IDs. A grading
model with no grades on the target is not a test.

**`tiling_parquets_sharded/Breast MMCI Full Test` (100 slides, 50 carcinoma / 50 not, tile-level
`carcinoma_roi_percentage`).** Three problems as a grading benchmark: **36 of the 100 are already in our
training labels — and all 36 are on the negative side**, leaving 14 clean negatives; no grades joinable
either; and the tiles are 224 px at mpp 0.4678/0.4856 with **stride 112** (ours: mpp 0.5, stride 224), so
3–7 % different field of view and a doubled, overlapping grid (24–36 k tiles vs our median 6.5 k — the
overlap is decimable, the mpp shift is not, short of re-embedding).

**Verdict: skip both as grading benchmarks.** What they are good for:

- *Collapse check* (~30 min, runs in the notebook): score the 50 unseen carcinoma slides from their
  ready `.pt` bags with the frozen artifact and compare `1 − p(N)` and the grade spread against our
  in-distribution test slides. Not a benchmark — the check that would embarrass us if it failed (if a
  fifth of genuinely unseen tumours read as tumour-free, that changes what we ship). Everything for it is
  already verified in place.
- *Attention-vs-annotation test* — **the project's actual end goal, needing no grades and no clean
  negatives.** The tile parquets carry `carcinoma_roi_percentage` per tile, so on 50 unseen slides we can
  measure whether the model's attention mass lands inside annotated carcinoma. This is the one thing on
  this list that could give points rather than decimals, and it is the §10 gap with a usable label source.
  Cost: one pass over 86 GB of tile parquets, which wants a job, not the 1-CPU notebook pod.

**For a real generalisation test of the grader** we need a *graded* external cohort — BRACS (public,
~198 slides with Nottingham grades), or another site's BRC slides that are not in bc20-24.

## 12. Limits a better model cannot get past

- **The label ceiling.** G1 recall 0.55 and G2 0.69, against published Nottingham inter-reader
  agreement of **~38 % exact and 32 % adjacent-grade** in the 24-observer CAP/NCI survey
  (doi:10.5858/arpa.2013.0212-CP — fetched, not recalled) and 55–77 % exact elsewhere
  (Mod Pathol 2024;37:100423, N = 515). We are inside the human band. **A model that reads 84 G1 slides
  like a pathologist will not get 84/84.** The 0.85 raw accuracy partly reflects a single-reader
  reference standard; part of what looks like model error is reference-standard noise. Multi-reader
  labels on the ambiguous G1/G2 band are the only thing that moves this — a data problem, not compute.
- **The cohort is the constraint.** G1 is 68 training slides and 14 test slides. No seed repetition buys
  resolution the split size cannot provide.
- **No external baseline has been run**, no published BCI / bc20-24 comparison exists, and calibration
  has never been checked. All three are required before any clinical claim.

## 13. Where the remaining points are, ranked by expected value

1. **Encoder features — the only untested lever that could give points, not decimals.** A different
   magnification (5× instead of 20×), a different tile size, a tile-level task-aware encoder, or an
   encoder that saw breast tissue. **Not an "arm"**: the encoder is external and pre-computed, so this is
   a funding/storage decision (one re-embedding run plus a new bag set), not something to queue. The
   attention-vs-annotation test (§11) tells us whether morphology attention is even pointed at tumour
   before paying for a second embedding set.
2. **Harvest the seed variance.** 6 points of spread, averageable; the ceiling is roughly the +1.3
   already banked, 15 runs ≈ 2 h. Small but real, and it buys error bars on what we ship.
3. **Price the "report a range" option** (§9): at β = 0.5, how many slides need a two-grade range, and
   what is the miss rate inside the band? One free CPU job, and it decides whether ranges are cheap or
   absurd.
4. **Get better labels.** Multi-reader on the G1/G2 ambiguous band. Not compute.
5. **Slide-level clinical features** (age, laterality, tumour size in `train_meta.parquet`) — untested,
   cheap, plausibly additive with morphology.
6. **Learning-curve probe: train on 50 % vs 100 % of the training slides, everything else equal.** The only
   test in the plan that answers the big question — flat means frozen Virchow2 tiles at one magnification
   are the ceiling and the next money goes to features; climbing means data volume is the lever. Needs a
   slide-subset knob (`datamodule.train` has no such parameter today), ~30 lines, 2 jobs.

**Two rules so we stop buying what we already own:**
(i) no new arm unless it changes the **input** rather than the head;
(ii) no arm is believed without ≥3 seeds judged on the out-of-fold pool.
Those two rules would have skipped most of the 84 pod-hours and kept every conclusion.

## 14. Methodology mistakes we made, so they don't get repeated

Each of these cost a wrong number that we then had to retract. The mechanisms, not the embarrassment:

1. **Variant names lied about their knobs.** Seven "t2048-*" arms trained on whole bags because
   `max_tiles` was never passed; MLflow correctly logged `max_tiles=None`. Fixed in the submitter (a
   knob that reaches training is now logged back). **Trust logged params, never a name.**
2. **Baselines were mislabelled the same way** — "the baseline was whole bags" turned out to be
   `max_tiles=2048`, which made "whole bags beat sub-sampling by 2.4 points" mean the exact opposite of
   what we filed. Several comparisons were re-measured before they were believed.
3. **Ensembles were fitted on a different epoch than they were tested on.** `best.ckpt` carries no epoch
   in its filename; the rule was fitted on one model's validation and applied to another's output.
   Validation numbers dropped 2–4 points once fixed.
4. **A "+2.6-point ensemble" was selection luck** — choosing the member set after seeing test. Same
   procedure with members chosen on validation: +1.3. Related, and unresolved: the best test run we have
   (dropout 0.2, 0.7956) is **not chosen by validation at all** (corrected val 0.7769, mid-pack), while
   greedy selection on validation picked the two `valid/accuracy`-monitored runs instead. On a ±2.6-point
   ruler that val/test disagreement is the noise floor, not a signal to act on.
5. **"Miss nothing" scored 1.0 on a degenerate model** — constant-escalate satisfies "never too low"
   perfectly. The objective now reports a recall/specificity curve instead of a single thresholded
   number.
6. **λ looked like a free +3 points** on balanced accuracy and was significantly *worse* per slide
   (§9). Retracted, then re-derived twice with paired tests.
7. **Silent failure modes** in the harness: a pod could report `Succeeded` with training dead in minutes
   (exit code came from `wait`); `EMB_CACHE_GB` is per-worker-per-split, not a total, and OOMKilled pods
   in epoch 0 with no traceback; the test frame is written at `epoch+1`, so "no dumps" was an off-by-one.
   Each is now guarded in code, and each is written up in `STATUS.md`.

---

## Appendix A — every finished run (test split, 266 slides), best-val-loss checkpoint

`bal` = MLflow `test/accuracy` = balanced accuracy. Recalls are N/G1/G2/G3 against 131/14/58/63. Val
loss is **not comparable across loss families** (CE ≈ 0.42–0.50, ordinal ≈ 0.14–0.18).

| run | variant | seed | tiles/epoch | pooling | loss | ep | min | best val | bal | AUROC | F1 | κ | recalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `7da26a15` | tiles2048_s456 | 456 | 2048 | attn | CE | 14 | 48 | 0.4792 | **0.7743** | 0.9522 | 0.7260 | 0.9202 | .99/.79/.62/.70 |
| `2a8a70e3` | t2048_ord_s456 | 456 | 2048 | attn | ord | 14 | 96 | 0.1769 | 0.7693 | **0.9544** | 0.7187 | 0.8994 | .99/.79/.57/.73 |
| `ba362f65` | full_ord_s456 | 456 | whole | attn | ord | 14 | 40 | 0.1797 | 0.7637 | **0.9544** | 0.7251 | 0.8995 | .99/.71/.59/.76 |
| `c53175d6` | baseline (unnamed) | n/l | whole | attn | CE | 22 | n/l | 0.4451 | 0.7612 | 0.9462 | 0.7335 | 0.9132 | .98/.64/.66/.76 |
| `44897d1a` | att2048_s123 | 123 | 2048 | attn | CE | 40 (cap) | 211 | 0.4423 | 0.7618 | 0.9406 | **0.7538** | 0.9193 | .99/.57/.69/.79 |
| `33c7de37` | tiles2048_s123 | 123 | **whole** (name lies) | attn | CE | 23 | 54 | 0.4643 | 0.7486 | 0.9423 | 0.7472 | **0.9213** | .99/.50/.72/.78 |
| `907fbfde` | full_ord_s789 | 789 | whole | attn | ord | 17 | 105 | 0.1678 | 0.7432 | 0.9511 | 0.7161 | 0.9174 | .99/.64/.66/.68 |
| `1ce8c392` | full_s456 | 456 | whole | attn | CE | 18 | 124 | 0.4991 | 0.7372 | 0.9482 | 0.7309 | 0.9107 | .98/.50/.74/.73 |
| `58076001` | att2048_s789 | 789 | 2048 | attn | CE | 15 | 34 | 0.4792 | 0.7147 | 0.9489 | 0.7168 | 0.9080 | .98/**.36**/.81/.71 |
| `7cd593f0` | meanpool_s123 | 123 | 2048 | mean | CE | 35 | 119 | 0.4195 | 0.7135 | 0.9382 | 0.7038 | 0.9197 | .99/.43/.66/.78 |
| `d16d2e01` | mean_s456 | 456 | 2048 | mean | CE | 23 | 137 | 0.4245 | 0.6791 | 0.9391 | 0.6781 | 0.9160 | .99/**.29**/.72/.71 |
| `49efd009` | mean_s789 | 789 | 2048 | mean | CE | 16 | 78 | 0.4337 | 0.6718 | 0.9327 | 0.6632 | 0.8949 | .98/.36/.55/.79 |

Mean pooling's failure mode is visible in those recalls: it never learns G1 (0.29–0.43) and its G2 swings
0.55–0.72. Older runs (experiment 37, 8-class; the 5-epoch probe) differ in label space and batch regime
and are **not comparable**.

## Appendix B — the eight overnight arms (2026-11-10) and the knob that never landed

All eight were meant to be paired at the same seed against the 2048-tile config; **seven never received
the tile knob** (checked in the job YAML), so their true baselines are the whole-bag runs — `full_s456`
0.7372 and, at seed 123, `tiles2048_s123` 0.7486 (itself mis-named and also whole-bag).

| arm | test bal. | true baseline | Δ |
|---|---|---|---|
| dropout 0.5 → 0.2 | **0.7956** | 0.7372 | **+5.8** — best single run we have, and the one knob later proven paired (p = 0.021) |
| select on `valid/accuracy` | 0.7846 / 0.7661 | 0.7486 / 0.7372 | +3.6 / +2.9 |
| class-weighted CE | 0.7798 / 0.7475 | 0.7486 / 0.7372 | +3.1 / +1.0 (seeds disagree → no effect) |
| ordinal loss, 2nd seed | 0.7690 | 0.7486 | +2.0 |
| 4096 tiles (the only arm that got its knob) | 0.7556 | 0.7372 whole / 0.7743 at 2048 | +1.8 / −1.9 |
| `weight_decay` 1e-2 | 0.7437 | 0.7372 | +0.7 |

Every Δ here is against a **single** baseline at one seed while same-config seed spread is ~6 points —
which is the empirical case for §3's rules and for the fold waves that replaced this style of experiment.

## Appendix C — reproduction

```bash
# Full out-of-fold report for a set of fold runs (needs the PVC, so a CPU pod):
REPORT_JOB_NAME=brc-grading-oof-final REPORT_SCRIPT=ensemble_variants.py \
REPORT_VARIANTS=kf0-ctl,...,kf4-cand \
REPORT_ARGS='--oof --train-counts 575,67,247,286' uv run --no-sync python scripts/run_report.py

# Every finished run with recalls, MLflow only, from the laptop:
MLFLOW_URL=https://mlflow-s3.rationai.cloud.trusted.e-infra.cz MLFLOW_DISABLE_AGENT_HINT=1 \
  PYTHONPATH=. .venv/bin/python scripts/all_runs_table.py
```

The prior must come from the fold CSVs (`<project>/kfold/labels-kf{K}.csv` → 575,67,247,286 ±1 slide per
fold), not the shipped 1175 — a fold trains on four fifths, so its prior is not the cohort's. The CSV
recompute was verified to reproduce MLflow exactly (`full_ord_s456` → bal 0.7637 / κ 0.8995 / AUROC
0.9544 both ways).
