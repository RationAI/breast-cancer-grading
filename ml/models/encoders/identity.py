"""Identity encoder for the precomputed-embeddings path."""

from torch import Tensor

from ml.models.encoders.base import Encoder


class IdentityEncoder(Encoder):
    """Passes precomputed tile embeddings through unchanged.

    Used when training a head/aggregator on frozen per-slide embedding bags
    produced by :mod:`preprocessing.embeddings` (e.g. Virchow2, 2560-dim).

    Args:
        out_dim: Dimensionality of the incoming embeddings (e.g. ``2560`` for
            Virchow2).
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self._out_dim = out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, tiles: Tensor) -> Tensor:
        return tiles


__all__ = ["IdentityEncoder"]
