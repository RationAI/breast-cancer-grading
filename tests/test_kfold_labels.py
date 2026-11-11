"""k-fold labels (scripts/make_kfold_labels.py) and the pooled OOF ruler (--oof).

``make_kfold_labels.py --self-check`` already proves the fold arithmetic on a synthetic
cohort. What this file adds is the two things a job actually depends on: the CSVs written
by the CLI are read back by the real ``GradingDataset`` (a fold must look like any other
cohort, with test left alone), and ``ensemble_variants.py --oof`` pools five held-out
fifths into one ruler instead of refusing to.

Written in this repo's test style -- module-level asserts plus a print, because these
files are scripts. ``pytest`` imports them, so a failed assert surfaces as a collection
error, but it collects **zero items** by design; run them with ``tests/run_all.sh``.
"""

import contextlib
import csv
import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import ensemble_variants as ev
from make_kfold_labels import main as kfold_main
from make_kfold_labels import make_fold, train_counts

from ml.data.datasets import GradingDataset


def write_embeddings(root: Path, slides: list[str], dim: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, slide in enumerate(slides):
        rows = 40 + i % 10
        emb = rng.normal(size=(rows, dim)).astype("float32")
        pd.DataFrame(
            emb,
            columns=[f"embedding_{j}" for j in range(dim)],
            index=pd.Index([f"{slide}_{t}" for t in range(rows)], name="tile_id"),
        ).to_parquet(root / f"{slide}.parquet")


def frame(slides: list[str], labels: list[int], correct: bool) -> pd.DataFrame:
    """A dump frame in the shape `load_variant` returns: slide_id, label, logit_0..3."""
    out = pd.DataFrame({"slide_id": slides, "label": labels})
    for cls in range(4):
        hit = np.array([cls == (lab if correct else 0) for lab in labels], dtype=float)
        out[f"logit_{cls}"] = np.where(hit, 4.0, -4.0)
    return out


# --- the CLI a job runs: five CSVs, each one a cohort GradingDataset can train on -----

# 384 slides, class-imbalanced like the real cohort (grade 1 is the scarce one). Split the
# way labels.csv is split: every 4th slide is test, and the rest are train with a fifth of
# the remainder held as val -- the generator relabels train+val and must not touch test.
GRADES = [g for g, n in [(0, 56), (1, 92), (2, 148), (3, 88)] for _ in range(n)]
SLIDES = [f"s{i:04d}" for i in range(len(GRADES))]
SPLIT = {
    s: ("test" if i % 4 == 0 else "val" if i % 5 == 1 else "train")
    for i, s in enumerate(SLIDES)
}
LABEL_OF = dict(zip(SLIDES, GRADES, strict=True))
POOL = [s for s in SLIDES if SPLIT[s] != "test"]
FOLDS = 5

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    source = root / "labels.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "grade", "split"])
        for slide in SLIDES:
            writer.writerow([slide, LABEL_OF[slide], SPLIT[slide]])
    write_embeddings(root / "embeddings", SLIDES)

    out_dir = root / "kfold"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        sys.argv = [
            "make_kfold_labels.py",
            "--labels",
            str(source),
            "--out-dir",
            str(out_dir),
            "--folds",
            str(FOLDS),
        ]
        kfold_main()
    # "<path>: train=N val=M test=K train_counts=[...]"
    written = {
        int(Path(line.split(":")[0]).stem.replace("labels-kf", "")): {
            kv.split("=")[0]: int(kv.split("=")[1])
            for kv in line.split(":")[1].split()[:3]
        }
        for line in printed.getvalue().splitlines()
    }
    assert sorted(written) == list(range(FOLDS)), printed.getvalue()

    held_out: list[str] = []
    for fold in range(FOLDS):
        folded = pd.read_csv(out_dir / f"labels-kf{fold}.csv", dtype={"slide_id": str})
        assert written[fold]["test"] == 96, "the CLI moved test slides"
        assert set(folded["split"]) == {"train", "val", "test"}
        val = list(folded.loc[folded.split == "val", "slide_id"])
        assert len(val) == written[fold]["val"]
        assert set(folded.loc[folded.split == "test", "slide_id"]) == {
            s for s in SLIDES if SPLIT[s] == "test"
        }, "test rows were relabelled"
        held_out += val

        # Stratified: the held-out slice carries the pool's class mix within 6 points, and
        # a fold without the scarce grade would score class 1 on nothing.
        mix = pd.Series(folded.loc[folded.split == "val", "grade"]).value_counts()
        assert set(mix.index) == {0, 1, 2, 3}, mix.to_dict()
        for grade, n in mix.items():
            want = sum(1 for s in POOL if LABEL_OF[s] == grade) / len(POOL)
            assert abs(n / mix.sum() - want) < 0.06, f"fold {fold} class {grade}: {n}/{mix.sum()} vs {want:.3f}"

    assert len(held_out) == len(set(held_out)), "folds overlap: a slide scored twice"
    assert set(held_out) == set(POOL), "the folds miss pool slides or repeat them"

    # The fold is a cohort like any other: the real dataset reads its train split.
    fold0 = pd.read_csv(out_dir / "labels-kf0.csv")
    dataset = GradingDataset(
        embeddings_dir=str(root / "embeddings"),
        labels_csv=str(out_dir / "labels-kf0.csv"),
        split="train",
    )
    assert len(dataset) == int((fold0.split == "train").sum()) > 0, (
        f"dataset {len(dataset)} vs CSV {int((fold0.split == 'train').sum())}"
    )
    # The prior the argmax learns comes from the fold, not from the shipped 1175.
    assert sum(train_counts(fold0)) == len(dataset), train_counts(fold0)

    # An empty pool is a refusal, not five dead CSVs.
    only_test = root / "labels-test-only.csv"
    with only_test.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "grade", "split"])
        for slide in SLIDES:
            writer.writerow([slide, LABEL_OF[slide], "test"])
    refused = False
    try:
        make_fold(pd.read_csv(only_test), 0, FOLDS)
    except ValueError:
        refused = True
    assert refused, "a cohort with no train/val rows must be refused"

# --- pooled out-of-fold: k held-out slices score as one wide ruler --------------------

LABELS = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
TESTS = [f"t{i:03d}" for i in range(8)]
TEST_LABELS = [0, 1, 2, 3, 0, 1, 2, 3]
PRIOR = np.array([575, 67, 247, 285], dtype=float)
PRIOR /= PRIOR.sum()


def fold_loaded(
    fold: int,
    take: list[int],
    name: str | None = None,
    correct: bool = True,
) -> tuple[str, str, pd.DataFrame, pd.DataFrame]:
    held = [i for i in range(len(LABELS)) if i % FOLDS_FOR_DUMP == fold]
    held = [i for i in held if i in take] if take else held
    return (
        name or f"kf{fold}",
        "test2/valid2",
        frame([f"s{i:03d}" for i in held], [LABELS[i] for i in held], correct=correct),
        frame(TESTS, TEST_LABELS, correct=correct),
    )


FOLDS_FOR_DUMP = 3
loaded = [fold_loaded(fold, []) for fold in range(FOLDS_FOR_DUMP)]
assert sum(len(f) for _, _, f, _ in loaded) == len(LABELS), "folds must cover every row once"

printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    ev.pooled_oof(loaded, PRIOR)
report = printed.getvalue()
assert f"pooled out-of-fold over {FOLDS_FOR_DUMP} folds" in report, report
assert f"{FOLDS_FOR_DUMP}-fold average on test" in report, report
assert "no rule" in report and "+ rule" in report, report
# Every fold's model predicts the held-out and test slides correctly here, so all five score
# lines must be perfect; if this ever drops, the pooling is losing or duplicating rows.
scores = [float(line.split("bal=")[1].split()[0]) for line in report.splitlines() if "bal=" in line]
assert len(scores) == 4 and all(s == 1.0 for s in scores), report  # pooled 2 + test 2
spread = [line for line in report.splitlines() if "spread" in line]
assert spread and spread[0].split("spread ")[1].rstrip(")").startswith("0.000"), spread

# A slide held out by two folds means the fold CSVs are not one split, and pooling would
# score it twice: say so instead of quietly averaging it.
clash = list(loaded)
clash[1] = (
    "kf1",
    "test2/valid2",
    frame(["s000"], [LABELS[0]], correct=True),  # s000 belongs to fold 0
    clash[1][3],
)
refused = False
try:
    with contextlib.redirect_stdout(io.StringIO()):
        ev.pooled_oof(clash, PRIOR)
except ValueError as error:
    refused = "two folds" in str(error)
assert refused, "overlapping folds must be refused"

# --- two recipes pooled in one call: scored separately, then compared slide by slide -----

# `kf3-ctl` naming, so the fold comes from the prefix and the recipe from the suffix. The second
# recipe predicts grade 0 for everything: strictly worse, on exactly the same held-out slides.
couples = [fold_loaded(f, [], f"kf{f}-ctl") for f in range(FOLDS_FOR_DUMP)] + [
    fold_loaded(f, [], f"kf{f}-bad", correct=False) for f in range(FOLDS_FOR_DUMP)
]
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    ev.pooled_oof(couples, PRIOR)
report = printed.getvalue()
assert "recipes bad, ctl" in report, report
# Averaging the two recipes would pull both to the middle; keeping them apart must leave one
# perfect and the other at chance (4 classes, everything predicted as grade 0).
paired_scores = [
    float(line.split("bal=")[1].split()[0])
    for line in report.splitlines()
    if "pooled out-of-fold, no rule" in line
]
assert len(paired_scores) == 2, report
assert max(paired_scores) == 1.0 and min(paired_scores) < 0.6, paired_scores
lines = [line for line in report.splitlines() if "McNemar" in line]
assert len(lines) == 1 and "paired per-slide comparison on 12" in report, lines
assert float(lines[0].split("p=")[1]) < 0.01, f"a hopeless recipe must not tie: {lines[0]}"
assert "bad by -" in lines[0], f"the sign must read as ctl winning: {lines[0]}"

print("k-fold labels + pooled OOF test passed")
