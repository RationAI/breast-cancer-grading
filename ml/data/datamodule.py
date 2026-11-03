"""Thin Lightning datamodule that builds its split datasets lazily.

Each split (``train``/``val``/``test``) is passed as an *unresolved* Hydra config
node and instantiated on demand in :meth:`setup`, only for the running stage — so a
test-only run never loads the train/val embeddings, and vice versa. The entrypoint
must instantiate this with ``_recursive_=False`` so the split nodes arrive as
``DictConfig`` rather than pre-built datasets.

Bags have variable tile counts, so batches stay a list of samples
(:func:`mil_collate`). Training optionally uses class-balanced sampling
(``WeightedRandomSampler``) because the grading classes are heavily imbalanced
(~50% negative, ~5% G1, ~20% G2, ~25% G3 in the bc20-24 cohort).
"""

from collections.abc import Sequence

import lightning.pytorch as pl
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ml.typing import MILSample


type Sample = MILSample


def mil_collate[S](batch: Sequence[S]) -> list[S]:
    """Collate variable-length bags by keeping the batch as a list of samples."""
    return list(batch)


class DataModule(pl.LightningDataModule):
    """Builds train/val/test datasets lazily from unresolved config nodes.

    Args:
        batch_size: Slides (bags) per batch.
        num_workers: Dataloader worker processes.
        pin_memory: Pin host memory for faster host->device copies.
        persistent_workers: Keep workers alive between epochs.
        class_balanced: Train with a ``WeightedRandomSampler`` inversely
            proportional to class frequency (val/test always stay unweighted).
        **datasets: One entry per split (``train``/``val``/``test``), each an
            unresolved dataset ``DictConfig`` instantiated in :meth:`setup`.
            Prediction reuses the test split.
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        class_balanced: bool = False,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.class_balanced = class_balanced
        self._dataset_cfgs = datasets
        self._datasets: dict[str, Dataset[Sample]] = {}

    def _build(self, split: str) -> None:
        if split in self._datasets:
            return
        cfg = self._dataset_cfgs.get(split)
        if cfg is None:
            raise KeyError(f"No dataset config provided for split '{split}'.")
        self._datasets[split] = instantiate(cfg)

    def setup(self, stage: str | None = None) -> None:
        needed = {
            "fit": ("train", "val"),
            "validate": ("val",),
            "test": ("test",),
            "predict": ("test",),
        }.get(stage or "", ("train", "val", "test"))
        for split in needed:
            self._build(split)

    def _dataloader(
        self, split: str, shuffle: bool, sampler: WeightedRandomSampler | None = None
    ) -> DataLoader[Sample]:
        return DataLoader(
            self._datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=mil_collate,
        )

    def train_dataloader(self) -> DataLoader[Sample]:
        if self.class_balanced:
            grades: np.ndarray = self._datasets["train"].grades  # type: ignore[attr-defined]
            counts = np.bincount(grades)
            weights = (1.0 / counts[grades]).tolist()
            sampler = WeightedRandomSampler(weights, num_samples=len(grades))
            return self._dataloader("train", shuffle=False, sampler=sampler)
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("test", shuffle=False)


__all__ = ["DataModule", "mil_collate"]
