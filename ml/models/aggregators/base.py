"""Aggregator abstract base class.

An aggregator reduces one slide's tiles to a single slide-level vector
``(D',)``. This is the multiple-instance-learning (MIL) pooling step; swapping the
aggregator (mean, gated attention, ...) is the primary experimental axis.
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Aggregator[InputT](nn.Module, ABC):
    """Reduces one slide's tiles to a single slide vector.

    Generic over its input type: flat aggregators specialise as
    ``Aggregator[Bag]`` (a tile tensor ``(N, D)``). Implementations return the
    pooled vector and, optionally, per-tile weights (``None`` for parameter-free
    poolings such as mean/max), surfaced for interpretability (heatmaps).
    """

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Dimensionality of the aggregated slide vector."""

    @abstractmethod
    def forward(self, bag: InputT) -> tuple[Tensor, Tensor | None]:
        """Aggregate one slide's tiles.

        Args:
            bag: A flat per-tile feature tensor ``(N, D)``.

        Returns:
            A tuple ``(pooled, attention)`` where ``pooled`` has shape
            ``(out_dim,)`` and ``attention`` has shape ``(N,)`` or is ``None``
            if the aggregator assigns no weights.
        """


__all__ = ["Aggregator"]
