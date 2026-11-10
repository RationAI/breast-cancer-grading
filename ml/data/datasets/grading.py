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
        max_tiles: Cap the bag at this many tiles, sampled afresh every epoch.
            Train-only in practice (set it on the train split alone): it is both
            the standard MIL bag augmentation and what makes an epoch affordable —
            a median slide carries ~7k tiles / 61 MB of parquet.
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        labels_csv: str | Path,
        split: str,
        max_tiles: int | None = None,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.split = split
        self.max_tiles = max_tiles
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
        if self.max_tiles is not None and bag.size(0) > self.max_tiles:
            # A different tile subset per epoch, so the attention cannot memorise
            # individual tiles. Reproducible: Lightning's ``seed_everything
            # (workers=True)`` reseeds each worker per epoch, so the same seed
            # replays the same schedule of subsets.
            keep = torch.randperm(bag.size(0))[: self.max_tiles]
            bag = bag[keep]
        label = torch.tensor(int(row[GRADE_COLUMN]), dtype=torch.long)
        return bag, label, SlideMetadata(slide_id=str(row[SLIDE_COLUMN]))
