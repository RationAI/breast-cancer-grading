"""scripts/make_kfold_labels.py — folds are valid CV splits that `GradingDataset` can read.

The point of generating a fold as a plain labels CSV is that nothing under `ml/` has to
change, so this instantiates a real dataset from a generated fold instead of trusting that.
"""

import contextlib
import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_kfold_labels

from ml.data.datasets import GradingDataset


make_kfold_labels.self_check()

LABELS = pd.read_csv(ROOT / "data/demo/labels.csv")
POOL = LABELS["split"].isin(["train", "val"])
TEST_SLIDES = set(LABELS.loc[LABELS["split"] == "test", "slide_id"])

with TemporaryDirectory() as directory:
    held: set[str] = set()
    for fold in range(5):
        folded = make_kfold_labels.make_fold(LABELS, fold)
        path = Path(directory) / f"labels-kf{fold}.csv"
        folded.to_csv(path, index=False)

        assert set(folded.loc[folded["split"] == "test", "slide_id"]) == TEST_SLIDES
        val = set(folded.loc[folded["split"] == "val", "slide_id"])
        assert not (val & held), (
            f"fold {fold} repeats a slide another fold already held out"
        )
        held |= val

        dataset = GradingDataset(ROOT / "data/demo/embeddings", path, "val")
        assert set(dataset.slides["slide_id"]) == val, (
            "dataset disagrees with the fold CSV"
        )
        assert len(dataset.grades) == len(val)

    assert len(held) == int(POOL.sum()), (
        "folds do not cover the train+val pool exactly once"
    )

print(
    f"ok: {5} folds cover {len(held)} pool slides, test untouched, dataset loads them"
)


# --- pooled out-of-fold: five disjoint held-out fifths score as one ~1469-slide ruler ---
import ensemble_variants


def dump_predictions(
    path: Path, slides: list[str], labels: list[int], right: bool
) -> None:
    lines = ["slide_id,label,logit_0,logit_1,logit_2,logit_3"]
    for slide, label in zip(slides, labels, strict=True):
        predicted = label if right else 0
        logits = [-2.0] * 4
        logits[predicted] = 2.0
        lines.append(f"{slide},{label}," + ",".join(f"{v:.6f}" for v in logits))
    path.write_text("\n".join(lines))


with TemporaryDirectory() as directory:
    root = Path(directory)
    total = 0
    for fold in range(3):
        held_out = [f"f{fold}s{i}" for i in range(9)]
        # Fold 0 is right on its fifth, the others are wrong: the pooled number must average
        # the three, not quietly keep the good one.
        labels = list(np.tile([0, 1, 2, 3], 3)[: len(held_out)])
        predictions = root / f"ml_attention-kf{fold}-s0" / "predictions"
        predictions.mkdir(parents=True)
        dump_predictions(
            predictions / "valid_epoch3.csv", held_out, labels, right=fold == 0
        )
        # All four grades: macro-AUROC over 3 of 4 classes is undefined, and test has four.
        dump_predictions(
            predictions / "test_epoch4.csv", ["t0", "t1", "t2", "t3"], [0, 1, 2, 3], True
        )
        total += len(held_out)

    argv = sys.argv
    sys.argv = [
        "ensemble_variants.py",
        "--checkpoint-root",
        str(root),
        "--variants",
        "kf0-s0,kf1-s0,kf2-s0",
        "--train-counts",
        "36,36,36,36",
        "--oof",
    ]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ensemble_variants.main()
    finally:
        sys.argv = argv
    report = out.getvalue()
    assert f"n={total:3d}" in report, report
    assert "fitted leave-one-fold-out" in report, report

print(
    f"ok: {total} out-of-fold slides pooled from 3 folds, rule fitted leave-one-fold-out"
)
