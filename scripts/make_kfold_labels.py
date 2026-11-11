"""Cross-validation splits: relabel train+val into k folds, never touching test.

The test set is 266 slides, so every number in ``STATUS.md`` is measured with a ±2.6-point
ruler while the *same config* moves ~6 points across seeds - half the experiments here exist
only to find out whether their own effect is measurable at all. Five folds over the 1469
train+val slides give ~1469 out-of-fold predictions and a ~1-point ruler, with the 266 test
slides left intact as a final holdout.

Each fold CSV uses the schema the training config already expects, so nothing in ``ml/``
changes: a fold's ``train`` is four fifths of the train+val pool, its ``val`` is the fifth it
scores out of, and ``test`` is copied through untouched. Folds are stratified by grade, which
matters because class 1 has 68 training slides - unstratified folds can leave a fold without
any.

    python scripts/make_kfold_labels.py --labels labels.csv --out-dir /mnt/projects/breast_cancer/brc_grading/kfold
    python scripts/make_kfold_labels.py --self-check

Train one fold with ``dataset.paths.labels=/mnt/…/kfold/labels-kf0.csv`` and point
``ensemble_variants.py --oof`` at the five run names to pool their held-out fifths.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ml.data.datasets.grading import GRADE_COLUMN, SLIDE_COLUMN, SPLIT_COLUMN


POOL = ("train", "val")


def make_fold(
    labels: pd.DataFrame, fold: int, folds: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Relabel the train+val pool into ``train``/``val`` for one fold; ``test`` passes through."""
    if not 0 <= fold < folds:
        raise ValueError(f"fold {fold} outside 0..{folds - 1}")
    pool = labels.loc[labels[SPLIT_COLUMN].isin(POOL)]
    if pool.empty:
        raise ValueError("Labels CSV has no train/val slides to split.")
    rng = np.random.default_rng(seed)
    assignment = np.full(len(labels), -1)
    rows = pool.index.to_numpy()
    for grade in sorted(pool[GRADE_COLUMN].unique()):
        same_grade = rows[(pool[GRADE_COLUMN] == grade).to_numpy()]
        rng.shuffle(same_grade)
        # Deal the class round-robin over folds: sizes differ by at most one slide per class.
        assignment[same_grade] = np.arange(len(same_grade)) % folds
    out = labels.copy()
    out.loc[rows, SPLIT_COLUMN] = np.where(assignment[rows] == fold, "val", "train")
    return out


def train_counts(labels: pd.DataFrame) -> list[float]:
    """Slide counts per class in ``train`` — the prior the argmax learns, fold by fold."""
    train = labels.loc[labels[SPLIT_COLUMN] == "train"]
    return [
        float((train[GRADE_COLUMN] == grade).sum())
        for grade in sorted(train[GRADE_COLUMN].unique())
    ]


def self_check() -> None:
    labels = pd.DataFrame(
        {
            SLIDE_COLUMN: [f"s{i}" for i in range(200)],
            GRADE_COLUMN: np.tile([0, 1, 2, 3], 50),
            SPLIT_COLUMN: ["train"] * 150 + ["val"] * 30 + ["test"] * 20,
        }
    )
    pool = labels.loc[labels[SPLIT_COLUMN].isin(POOL), SLIDE_COLUMN]
    seen: set[str] = set()
    for fold in range(5):
        folded = make_fold(labels, fold)
        assert set(folded.loc[folded[SPLIT_COLUMN] == "test", SLIDE_COLUMN]) == set(
            labels.loc[labels[SPLIT_COLUMN] == "test", SLIDE_COLUMN]
        ), "test slides were relabelled"
        held = set(folded.loc[folded[SPLIT_COLUMN] == "val", SLIDE_COLUMN])
        assert not seen & held, (
            f"fold {fold} repeats slides other folds already held out"
        )
        seen |= held
        # Stratified: each class deals evenly, and the rarest class still appears everywhere.
        grades = folded.loc[folded[SPLIT_COLUMN] == "val", GRADE_COLUMN].value_counts()
        assert set(grades.index) == {0, 1, 2, 3}, grades.to_dict()
        assert grades.max() - grades.min() <= 1, grades.to_dict()
    assert seen == set(pool), "the folds do not cover the train+val pool exactly once"

    counts = [train_counts(make_fold(labels, fold)) for fold in range(5)]
    # 180 pool slides, 45 per grade, 5 folds: 36 train slides of each grade in every fold.
    assert all(fold_counts == [36.0, 36.0, 36.0, 36.0] for fold_counts in counts), (
        counts
    )
    pd.testing.assert_frame_equal(make_fold(labels, 1), make_fold(labels, 1))
    print(
        "self-check ok: folds are disjoint, stratified, cover the pool, leave test alone"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", default="/mnt/projects/breast_cancer/brc_grading/labels.csv"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Write every fold here, from one seed, so the folds cannot drift apart between jobs.",
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    labels = pd.read_csv(args.labels)
    for fold in range(args.folds):
        folded = make_fold(labels, fold, args.folds, args.seed)
        out = (
            os.path.join(args.out_dir, f"labels-kf{fold}.csv")
            if args.out_dir
            else (args.out or f"labels-kf{fold}.csv")
        )
        folded.to_csv(out, index=False)
        train = folded.loc[folded[SPLIT_COLUMN] == "train"]
        print(
            f"{out}: train={len(train)} "
            f"val={int((folded[SPLIT_COLUMN] == 'val').sum())} "
            f"test={int((folded[SPLIT_COLUMN] == 'test').sum())} "
            f"train_counts={train_counts(folded)}"
        )


if __name__ == "__main__":
    main()
