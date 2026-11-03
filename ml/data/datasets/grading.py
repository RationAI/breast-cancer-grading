"""Slide-level grading dataset over precomputed per-tile embeddings.

Each slide is one *bag* in the multiple-instance-learning sense: a parquet file
with one row per tile (columns ``x``, ``y``, ``embedding``), exactly the format
written by :mod:`preprocessing.embeddings`. Labels and splits come from a single
CSV with columns ``slide_id``, ``grade``, ``split`` (see README, "Labels"),
following the BRC decision to keep one dataset with a split column instead of
physically separate split artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.typing import MILSample, SlideMetadata

SLIDE_COLUMN = "slide_id"
GRADE_COLUMN = "grade"
SPLIT_COLUMN = "split"


class GradingDataset(Dataset[MILSample]):
    """Serves one split of the grading cohort as slide-level MIL samples.

    Args:
        embeddings_dir: Directory with one ``<slide_id>.parquet`` per slide, as
            produced by ``preprocessing/embeddings.py``.
        labels_csv: CSV with columns ``slide_id``, ``grade`` (0 = negative,
            1 = G1, 2 = G2, 3 = G3), ``split`` (train/val/test).
        split: Which split to serve — rows with a different ``split`` are dropped.
    """

    def __init__(
        self, embeddings_dir: str | Path, labels_csv: str | Path, split: str
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.split = split
        self.slides = (
            pd.read_csv(labels_csv)
            .loc[lambda df: df[SPLIT_COLUMN] == split]
            .reset_index(drop=True)
        )
        if self.slides.empty:
            raise ValueError(f"Labels CSV has no '{split}' slides.")

        missing = [
            f"{slide_id}.parquet"
            for slide_id in self.slides[SLIDE_COLUMN]
            if not (self.embeddings_dir / f"{slide_id}.parquet").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} slides missing in {self.embeddings_dir}, "
                f"e.g. {missing[:3]}. Run preprocessing/embeddings.py first."
            )

    def __len__(self) -> int:
        return len(self.slides)

    @property
    def grades(self) -> np.ndarray:
        """Grades of this split's slides, aligned with ``__getitem__`` indices."""
        return self.slides[GRADE_COLUMN].to_numpy()

    def __getitem__(self, idx: int) -> MILSample:
        row = self.slides.iloc[idx]
        frame = pd.read_parquet(self.embeddings_dir / f"{row[SLIDE_COLUMN]}.parquet")
        bag = torch.from_numpy(np.stack(frame["embedding"].to_numpy())).float()
        label = torch.tensor(int(row[GRADE_COLUMN]), dtype=torch.long)
        return bag, label, SlideMetadata(slide_id=str(row[SLIDE_COLUMN]))
