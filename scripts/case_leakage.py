"""Does the same *case* sit in train and test? bc20-24 films several slides per case.

Slide-level splits are what `labels.csv` ships, but two slides of one case share tissue, staining
and the pathologist's mind, so a model that has seen one is not blind to the other. This prints how
much of the test set is already known at case level, because that is the difference between
"0.79 balanced accuracy on held-out slides" and "0.79 balanced accuracy on held-out patients".

Run as a report job:
    REPORT_JOB_NAME=brc-grading-leak REPORT_SCRIPT=case_leakage.py \
      uv run --no-sync python scripts/run_report.py
"""

from __future__ import annotations

import argparse

import pandas as pd


LABELS = "/mnt/projects/breast_cancer/brc_grading/labels.csv"


def case_of(slide_id: str) -> str:
    """`2019_08728-01` and `2019_08728-02` are one case: the block before the last dash."""
    return slide_id.rsplit("-", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-labels", default=LABELS)
    args = parser.parse_args()

    labels = pd.read_csv(args.path_labels)
    labels["case"] = labels["slide_id"].map(case_of)
    by_split = {s: set(g["case"]) for s, g in labels.groupby("split")}
    print(f"{len(labels)} slides, {labels['case'].nunique()} cases")
    for split, group in labels.groupby("split"):
        print(
            f"  {split:6s} {len(group):5d} slides, {group['case'].nunique():5d} cases"
            f" ({len(group) / group['case'].nunique():.2f} slides per case)"
        )

    for a, b in [("train", "test"), ("val", "test"), ("train", "val")]:
        shared = by_split.get(a, set()) & by_split.get(b, set())
        rows = labels[(labels["split"] == b) & labels["case"].isin(shared)]
        print(
            f"{a} x {b}: {len(shared)} cases on both sides -> "
            f"{len(rows)} of {int((labels.split == b).sum())} {b} slides seen at case level"
            f" ({len(rows) / max(1, int((labels.split == b).sum())):.1%})"
        )

    # What a case-disjoint split would cost, by class: if it cannot be built without dropping most
    # of a class, the honest headline is "case-level" and the small classes need saying so.
    train = labels[labels["split"] != "test"]
    held = labels[labels["split"] == "test"]
    fresh = held[~held["case"].isin(set(train["case"]))]
    print(f"\ncase-fresh test slides available: {len(fresh)} of {len(held)}")
    for grade in sorted(held["grade"].unique()):
        print(
            f"  grade {grade}: {int((held.grade == grade).sum())} held out, "
            f"{int((fresh.grade == grade).sum())} of them case-fresh"
        )


if __name__ == "__main__":
    main()
