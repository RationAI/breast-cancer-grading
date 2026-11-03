"""Mean-pool MIL aggregator."""

from torch import Tensor

from ml.models.aggregators.base import Aggregator
from ml.typing import Bag


class MeanAggregator(Aggregator[Bag]):
    """Averages the tile features of a slide — the parameter-free MIL baseline."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self._feature_dim = feature_dim

    @property
    def out_dim(self) -> int:
        return self._feature_dim

    def forward(self, bag: Bag) -> tuple[Tensor, None]:
        return bag.mean(dim=0), None


__all__ = ["MeanAggregator"]
