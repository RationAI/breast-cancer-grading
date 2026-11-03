"""Generic breast-cancer grading LightningModule.

Wires three independently swappable pieces — an :class:`~ml.models.encoders.base.Encoder`,
an :class:`~ml.models.aggregators.base.Aggregator`, and a :class:`~ml.models.heads.base.Head` —
into a multiple-instance-learning pipeline::

    bag -> encoder (per tile) -> aggregator (bag -> vector) -> head -> logits

The task is 4-class Nottingham grading: ``0`` = no tumour epithelium (negative
slide), ``1`` = G1 (well differentiated), ``2`` = G2 (moderately), ``3`` = G3
(poorly). Loss, optimizer, LR scheduler and metrics are config-instantiated.
Metrics receive softmax probabilities (``torchmetrics`` multiclass AUROC expects
scores; argmax-based metrics are invariant to the monotone map).
"""

import functools
import logging
from collections.abc import Sequence
from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics
from torch import Tensor, nn

from ml.models.aggregators.base import Aggregator
from ml.models.encoders.base import Encoder
from ml.models.heads.base import Head
from ml.typing import Bag, MILSample, SlideMetadata


logger = logging.getLogger(__name__)


class GradingModule(pl.LightningModule):
    """MIL module composing encoder -> aggregator -> head for slide grading.

    Args:
        encoder: Per-tile feature extractor. ``IdentityEncoder`` on the
            precomputed-embeddings path; an image backbone on the end-to-end path.
        aggregator: Bag pooling reducing per-tile features to a slide vector.
        head: Prediction head mapping the slide vector to class logits.
        loss: Loss module (e.g. ``torch.nn.CrossEntropyLoss`` on logits).
            ``None`` disables training (predict-only).
        optimizer: A ``functools.partial`` of an optimizer, bound to the model
            parameters in :meth:`configure_optimizers`.
        lr_scheduler: Optional dict with a ``scheduler`` partial and an optional
            ``monitor`` key.
        metrics: Torchmetrics keyed by name; cloned per ``train``/``valid``/``test``.
    """

    def __init__(
        self,
        encoder: Encoder,
        aggregator: Aggregator[Bag],
        head: Head,
        loss: nn.Module | None = None,
        optimizer: functools.partial[torch.optim.Optimizer] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        metrics: dict[str, torchmetrics.Metric] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.head = head
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        base = torchmetrics.MetricCollection(dict(metrics) if metrics else {})
        self.train_metrics = base.clone(prefix="train/")
        self.valid_metrics = base.clone(prefix="valid/")
        self.test_metrics = base.clone(prefix="test/")

        # The confusion matrix is not scalar-loggable, so it lives outside the
        # collections: per-class recalls are logged, the full matrix printed.
        self.confusion = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=head.out_dim
        )

    def forward(self, bag: Bag) -> tuple[Tensor, Tensor | None]:
        """Run one slide through encoder -> aggregator -> head.

        Args:
            bag: One slide's flat tile tensor ``(N, D)`` / ``(N, C, H, W)``.

        Returns:
            ``(logits, attention)`` where ``logits`` has shape ``(out_dim,)``
            and ``attention`` is the aggregator's per-tile weights or ``None``.
        """
        encoded = self.encoder(bag)  # (N, ...) -> (N, encoder.out_dim)
        pooled, attention = self.aggregator(encoded)  # (aggregator.out_dim,), attn|None
        logits = self.head(pooled)  # (num_classes,)
        return logits, attention

    def _forward_batch(self, bags: Sequence[Bag]) -> tuple[Tensor, list[Tensor | None]]:
        """Run a batch of variable-length bags, stacking per-slide logits."""
        logits: list[Tensor] = []
        attentions: list[Tensor | None] = []
        for bag in bags:
            prediction, attention = self(bag)
            logits.append(prediction)
            attentions.append(attention)
        return torch.stack(logits), attentions

    def _step(
        self,
        batch: list[MILSample],
        stage: str,
        metrics: torchmetrics.MetricCollection,
    ) -> Tensor:
        bags, labels, _ = _unpack_batch(batch)
        y = torch.stack(labels).to(self.device)
        y_pred, _ = self._forward_batch(bags)

        if self.loss is None:
            raise RuntimeError("A loss is required for training/validation steps.")

        loss = self.loss(y_pred, y)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=len(batch),
        )

        metrics.update(torch.softmax(y_pred, dim=-1), y)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        if stage != "train":
            self.confusion.update(torch.softmax(y_pred, dim=-1), y)
        return loss

    def _log_confusion(self, stage: str) -> None:
        """Log per-class recalls and print the confusion matrix for this epoch."""
        cm = self.confusion.compute()
        recalls = cm.diagonal() / cm.sum(dim=1).clamp_min(1)
        for i, recall in enumerate(recalls):
            self.log(f"{stage}/recall_class_{i}", recall)
        if stage == "test":
            print(f"\n{stage} confusion matrix (rows = true class 0..3):\n{cm}")
        self.confusion.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_confusion("valid")

    def on_test_epoch_end(self) -> None:
        self._log_confusion("test")

    def training_step(self, batch: list[MILSample], batch_idx: int) -> Tensor:
        return self._step(batch, "train", self.train_metrics)

    def validation_step(self, batch: list[MILSample], batch_idx: int) -> Tensor:
        return self._step(batch, "valid", self.valid_metrics)

    def test_step(self, batch: list[MILSample], batch_idx: int) -> Tensor:
        return self._step(batch, "test", self.test_metrics)

    def predict_step(self, batch: list[MILSample], batch_idx: int) -> dict[str, Any]:
        """Predict grades + per-tile attention (for tile-level heatmaps)."""
        bags, _, metadata = _unpack_batch(batch)
        predictions, attentions = self._forward_batch(bags)
        return {
            "predictions": predictions,
            "attentions": attentions,
            "metadata": metadata,
        }

    def configure_optimizers(self) -> Any:
        if self.optimizer is None:
            return None

        optimizer = self.optimizer(self.parameters())
        config: dict[str, Any] = {"optimizer": optimizer}

        if self.lr_scheduler:
            scheduler = {"scheduler": self.lr_scheduler["scheduler"](optimizer)}
            if monitor := self.lr_scheduler.get("monitor"):
                scheduler["monitor"] = monitor
            config["lr_scheduler"] = scheduler

        return config


def _unpack_batch(
    batch: list[MILSample],
) -> tuple[list[Bag], list[Tensor], list[SlideMetadata]]:
    """Split a list-of-samples batch into parallel lists of bags/labels/metadata."""
    bags = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    metadata = [sample[2] for sample in batch]
    return bags, labels, metadata


__all__ = ["GradingModule"]
