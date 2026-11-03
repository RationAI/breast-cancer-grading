"""Head abstract base class.

A head maps a slide-level vector to the task prediction — logits for
classification.
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Head(nn.Module, ABC):
    """Maps a slide vector ``(in_dim,)`` to a prediction ``(out_dim,)``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Predict from the slide-level features.

        Args:
            features: Aggregated slide vector, shape ``(in_dim,)`` or a batch
                ``(B, in_dim)``.

        Returns:
            Prediction of shape ``(out_dim,)`` (or ``(B, out_dim)``).
        """


__all__ = ["Head"]
