"""Linear prediction head."""

from torch import Tensor, nn

from ml.models.heads.base import Head


class LinearHead(Head):
    """Dropout followed by a single linear layer.

    For ``C``-class classification the output is raw logits; the loss
    (cross-entropy) applies softmax internally.

    Args:
        in_dim: Dimensionality of the incoming slide vector.
        out_dim: Number of classes.
        dropout: Dropout probability applied before the linear layer.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_dim, out_dim)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(self.dropout(features))


__all__ = ["LinearHead"]
