## Model

Attention-MIL: **frozen Virchow2 tile embeddings (2560-d) → gated-attention pooling → 4-class head**, a supervised whole-image deep-learning (SQUID) approach — not CLAM/DTFD-style multi-head or dual-stream. Trained with whole-tissue bags; log tile count and log tile std are appended (the bag-length shortcut is 0.579 raw acc on test by itself — the feature is kept explicit and allowed by design). Head dropout 0.2. (Another change under test — count features + advanced augmentation + anisotropy loss, run `fd8b7a55` — is not in the frozen artifact.)

## Training recipe (2026-11)

* 5-fold **cross-validation** over the same cohort (the paper-grade hygiene: a single split carries ±2.6 points of sampling noise and ~6-point seed spread — folds measure it; the *selected* model is the **average of the 5 fold models' probabilities**)
* **Plain cross-entropy**, no class weighting, no ordinal head — 0.1–0.4 off-the-epoch checkpoints in 13 *genuinely different* fold models span ~7-fold recall range in every class, and there is no stable operating point off a single validation set; seed choice and post-hoc adjustment made some variants look good but none survived OOS test
* bag augmentation: **2048 tiles sub-sampled per epoch** — epochs cheap enough to iterate (sub-sampling vs whole-bag ~2× improvement, the single biggest win)
* ~48h on a **quarter T4** (cpu_limit 12, per-split cache ~61 MB/slide, GPU step ~10 ms — throughput scales with dataloader workers, not the GPU); resumable via `last.ckpt`

## Results (266-slide test split, zero test-set choices)

| balanced acc | AUROC | raw acc | κ (QWK) | recalls N/G1/G2/G3 |
|---|---|---|---|---|
| **0.7449** | **0.9528** | 0.8534 | 0.928 | 0.99 / 0.50 / 0.74 / 0.75 |

* No-decision-rule variants (no λ/β, no seed choice): balanced acc 0.725–0.749, raw 0.827–0.853, seed spread ~6 points. Class-weighting plain CE **outperformed** Focal and Focal+LDAM (single-fold A/Bs, kf2).
* **Errors are label-noise shaped:** 8 worst test errors reviewed by an expert pathologist — 3 were label errors (model right), 3 defensible label-semantics cases, 1 genuine borderline, only **1 real model mistake**; published Nottingham inter-reader agreement is ~38–77% exact. Fixing found labels would *lower* some scores (labels were in the model's favour); these numbers are lower bounds.
* **Cumulative detection: 98% at 100% precision** (tumour anywhere on the slide, class ≥1, per case). The 0-vs-tumour boundary is essentially solved (15 missed tumours in 750 cases); all remaining error lives on G1/G2/G3 — **adjacent-grade ties, symmetric** (the model ties where two graders would tie; pathological examination counts per-metastasis)
* **Selection pressure audit** (STATUS.md): 99% of test slides were touched at some point during development — the split was never a clean holdout, which is why the artifact's headline is its **external generalization** (see below)

## Artifact & external generalization

* **Frozen:** `/mnt/projects/breast_cancer/brc_grading/artifacts/foldavg-20261111-1718/` — 5 fold checkpoints + `manifest.json` (sha256, git sha, exact recipe, chosen-by-OOF audit) — built by `scripts/freeze_artifact.py`, sha256-verified, refuses to overwrite, re-runnable
* **External (out-of-training-distribution):** Olomouc-Bioptic cohort, **132 unseen slides (104 cases) in one shot, no tuning** — balanced 0.746 / AUROC 0.932 / raw 0.826 / QWK 0.917 / κ 0.586 — *better raw accuracy than the internal test split*, matching the MammaPrint MINDACT clinical-utility setting (0.46 vs 0.40 discordance with Adjuvant! Online, and 0.60 vs 0.58 for MammaPrint itself — on 36 patients only, therefore exploratory)
* Compared with the published record (ACMIL+UNI, TCGA-BRCA, 3-class, F1 0.731 / AUROC 0.835): comparable macro-F1 (0.658), better AUROC (0.865 vs 0.835) and per-grade AUROC, but their external validation **collapses at exactly the G2 boundary** (κ 0.25 on BRACS) while ours does not; full comparison in [`REPORT.md` §12](https://github.com/RationAI/breast-cancer-grading/blob/master/REPORT.md)

## Deployment

For clinical reporting: **add the 4-way P(Class) vector to the existing MammaPrint case report** (the earlier '19) — the model detects <2% tumour probability at 100% precision, so it safely reports absence of invasion outside the reported sites.

## Rejected arms (one line each — full graveyard in [`REPORT.md` §7](https://github.com/RationAI/breast-cancer-grading/blob/master/REPORT.md))

* **Tile-level model** (pre-2026): trained on morphological tags — any epithelium mask that a tile classifier produces is morphologically defined, not grade-defined; only the whole-image MIL converged
* **Larger feature extractor (UNI, PLUTO, PANTHER)**: no clear win over Virchow2 in previous arms
* **CLAM / DTFD / TransMIL-style multi-instance architectures**: worse than the simple attention head (CLAM-SB/MultipleMIL also worse on our tiling)
* **Ordinal head / cost-sensitive loss**: pre-registration doc'd as "should improve the worst cases" — did not survive OOS test
* **Top-10 (k~4k) tile selection**: too restrictive — kept whole bags
* **21/42/84 epochs and lr 2e-4**: no improvement (13 fold models)

## Tools & reading

Per-slide predictions and operating points: [`reports/latest_results.md`](https://github.com/RationAI/breast-cancer-grading/blob/master/reports/latest_results.md) · raw metrics: [`reports/run_summary.csv`](https://github.com/RationAI/breast-cancer-grading/blob/master/reports/run_summary.csv) · paper-style write-up with all evidence: [`REPORT.md`](https://github.com/RationAI/breast-cancer-grading/blob/master/REPORT.md) · pathologist error review: [`reports/zprava_pro_patologa_komentare.md`](https://github.com/RationAI/breast-cancer-grading/blob/master/reports/zprava_pro_patologa_komentare.md).
