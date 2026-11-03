"""Generate synthetic demo data for the grading pipeline (labels + embeddings).

Creates a tiny synthetic cohort that mirrors the real bc20-24 shape — heavily
imbalanced 4 classes (N ~50%, G1 ~5%, G2 ~20%, G3 ~25%), multiple slides per
case, split by case — so the full training path can be verified on a laptop:

    uv run python scripts/make_demo_data.py
    uv run -m ml.train +experiment=ml/demo

Grade signal is a class-conditional Gaussian over 128-dim tile embeddings,
monotone along a fixed direction, so a correct model reaches high quadratic
kappa within a few epochs.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
EMBEDDING_DIM = 128
OUTPUT_DIR = Path("data/demo")
SLIDES_PER_CASE = (1, 3)  # inclusive
TILES_PER_SLIDE = (20, 80)  # inclusive
CLASS_CASES = {0: 50, 1: 5, 2: 20, 3: 25}  # cases per grade: N, G1, G2, G3
GRADE_NAMES = {0: "N", 1: "G1", 2: "G2", 3: "G3"}
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def main() -> None:
    rng = np.random.default_rng(SEED)

    # 100 cases with explicit per-grade counts (G1 needs >= 3 cases so every
    # split can keep at least one).
    grades = np.concatenate(
        [np.full(count, grade) for grade, count in CLASS_CASES.items()]
    )
    rng.shuffle(grades)

    # Monotone class signal: fixed unit direction, centers spread along it.
    direction = np.zeros(EMBEDDING_DIM)
    direction[:8] = 1.0
    direction /= np.linalg.norm(direction)
    centers = {g: (g - 1.5) * 3.0 * direction for g in GRADE_NAMES}

    rows: list[dict[str, object]] = []
    embeddings_dir = OUTPUT_DIR / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    slide_no = 0
    for case_no, grade in enumerate(grades):
        for _ in range(rng.integers(*SLIDES_PER_CASE)):
            slide_id = f"slide_{slide_no:03d}"
            slide_no += 1

            # Split at the case level (no slide leakage between splits).
            rows.append(
                {
                    "slide_id": slide_id,
                    "case": f"case_{case_no:03d}",
                    "grade": int(grade),
                    "split": "train",  # assigned below
                }
            )

            n_tiles = int(rng.integers(*TILES_PER_SLIDE))
            slide_mean = centers[grade] + rng.normal(0, 0.5, EMBEDDING_DIM)
            tiles = slide_mean + rng.normal(0, 1.0, (n_tiles, EMBEDDING_DIM))

            frame = pd.DataFrame(
                {
                    "x": (np.arange(n_tiles) % 10) * 224,
                    "y": (np.arange(n_tiles) // 10) * 224,
                    "embedding": list(tiles.astype(np.float32)),
                }
            )
            frame.to_parquet(embeddings_dir / f"{slide_id}.parquet", index=False)

    # Assign splits per grade: guarantee every split >= 1 case of each class,
    # then fill the rest by the target fractions.
    df = pd.DataFrame(rows)
    cases = df.drop_duplicates("case").set_index("case")["grade"].to_dict()
    for grade in GRADE_NAMES:
        grade_cases = [c for c, g in cases.items() if g == grade]
        rng.shuffle(grade_cases)
        if len(grade_cases) >= 2:
            df.loc[df["case"] == grade_cases[0], "split"] = "val"
            df.loc[df["case"] == grade_cases[1], "split"] = "test"
        for case in grade_cases[2:]:
            split = str(
                rng.choice(list(SPLIT_FRACTIONS), p=list(SPLIT_FRACTIONS.values()))
            )
            df.loc[df["case"] == case, "split"] = split

    df.to_csv(OUTPUT_DIR / "labels.csv", index=False)

    # Self-check: format compatibility + every split covers all grades.
    check = pd.read_parquet(embeddings_dir / "slide_000.parquet")
    assert set(check.columns) == {"x", "y", "embedding"}, check.columns
    assert len(check) > 0
    for split in SPLIT_FRACTIONS:
        grades_in_split = set(df.loc[df["split"] == split, "grade"])
        assert grades_in_split == set(GRADE_NAMES), (split, grades_in_split)
    print(df.groupby(["split", "grade"]).size().unstack(fill_value=0))
    print(f"\nWrote {len(df)} slides ({df['case'].nunique()} cases) to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
