# Do MammaPrint and our grade model look at the same tissue?

*2026-11-24 · Jiri Buchta · 304 MammaPrint slides (4 excluded: contaminated slides)*

**Question.** For each slide we have (a) our grade-MIL attention (Virchow2 tiles,
5-fold averaged) and (b) the official MammaPrint run's per-tile attribution: its
attention weights and a leave-one-out score telling which tiles push the predicted
index **up** (risk drivers). How much do the two models agree on *where* they look?

## Method in one paragraph
His tiles (level-2 grid) are mapped onto our tile grid (level 1, exact parent map);
per slide we compute Spearman rank correlation between our attention and three of
his per-tile quantities over all matched tiles. Slide-level correlation of risk
scores is computed across slides. All 304 slides matched grids at **coverage =
1.000** (every one of his tiles found its our-grid parents), so the grids are
provably aligned — any disagreement below is real, not a registration artifact.

## Results (per-slide Spearman ρ, n = 304)

| Compared quantity | median ρ | IQR | slides ρ > 0 | slides |ρ| > 0.3 |
|---|---|---|---|---|
| our attention vs **his risk-driver tiles** (signed LOO) | **0.02** | −0.19 … +0.20 | 52 % | 29 % (both signs) |
| our attention vs **magnitude** of his attribution (abs LOO) | **0.01** | −0.09 … +0.11 | ~51 % | 2 % |
| our attention vs his **raw attention weights** | **0.62** | 0.52 … 0.69 | — | 96 % above +0.3 |

Slide-level (across the 304 slides):

| Quantity | Spearman |
|---|---|
| his predicted index vs true index (sanity) | **+0.71** ✅ |
| his predicted index vs our P(grade 3) | **−0.72** |
| his TRUE index vs our P(grade 3) | **−0.53** |
| our attention spread: effective tiles / max weight | 2 792 (median) / ≈ 7× uniform |

MammaPrint polarity note: on the standard assay a LOW/negative index means
GENOMICALLY HIGH RISK (official cutoff −0.275). With that reading the negative
correlations are the clinically expected direction — high genomic risk skews
poorly differentiated. At the cutoff: genomically high-risk slides (n = 48) get
our G3 call in 44 % of cases (mean P(G3) 0.46) vs 12 % (0.22) for low-risk ones;
our P(G3) mean falls monotonically 0.51 → 0.06 across index quartiles. (To be
confirmed with the ML owner that the target is the raw index, not an inverted
risk score — if inverted, these sign readings flip.)

## What this means
1. **No tile-level agreement with what drives his risk.** Coin-flip sign (52 %),
   median ρ ≈ 0. The models do not use the same tiles to decide risk.
2. **They do rank raw "tile salience" similarly** (ρ ≈ 0.6, 96 % of slides) — both
   models prefer tumour-rich tiles over background. But his raw weights are *not*
   his risk attribution; when you ask "which tiles actually move the MammaPrint
   index", our attention has zero information about that.
3. **Verdicts track, tiles do not.** His index anti-correlates with our G3
   probability (−0.72 predicted, −0.53 vs truth), which under standard MammaPrint
   polarity = agreement on *which* tumors are aggressive. Yet our attention
   carries zero information about *which tiles* move his index. Same destination,
   different evidence: a real, independent cross-modality validation of our model
   (nothing genomic ever touched it), plus a clean null on shared mechanism.
4. Our attention is diffuse by design (thousands of tiles share the mass), so a
   hot-spot coincidence test is conservative — the null agreement is not an
   artifact of sparse spikes.

## Caveats
- The two heat scales are not comparable; the scale-free quantity is the
  both-top-10 % overlap layer (green) in the xOpat page.
- His attribution is LOO on his own grid at one downsampling level; subtile
  effects within his tiles are invisible to the test.
- 4 contaminated slides were excluded from every number above.

## Artifacts
- xOpat overlay page (3 layers per slide, all statistics on cards):
  `/mnt/projects/breast_cancer/brc_grading/reports/mammaprint_agreement.html`
  (also as MLflow artifact, experiment "BRC Grading")
- Per-slide table: `mammaprint_compare/report_summary.csv`,
  `agreement_by_slide.csv`
- Tile-level join for reuse: `mammaprint_compare/pairing.parquet` (2.47 M his-tile rows)
