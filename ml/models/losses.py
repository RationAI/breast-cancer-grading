"""Losses for the ordered Nottingham grades.

``torch.nn.CrossEntropyLoss`` scores ``G1 -> G3`` exactly as harshly as ``G1 -> G2``,
which is wrong for a task the metric reads as quadratic-weighted kappa and the clinic
reads as "one grade off is common, two grades off is a misdiagnosis".
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class OrdinalCrossEntropy(nn.Module):
    """Cumulative-link loss over ordered classes (CORAL-style, no extra parameters).

    The ``K`` logits are read as ``K - 1`` nested binary questions ``P(grade > k)``,
    each scored against ``target > k`` with binary cross-entropy. A one-grade slip can
    only move one of those questions, a two-grade slip moves two, so far errors cost
    more than adjacent ones. Class 0 = no tumour epithelium sits at the bottom of the
    same order, so ``grade > 0`` is "has tumour".

    Works on the module's existing head output: no threshold parameters, no change to
    ``LinearHead``, so a run can be switched to it with ``ml/loss=ordinal`` alone.

    Args:
        num_classes: Number of ordered grades (``K``); must match the head's
            ``out_dim``.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = num_classes

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Cumulative binary cross-entropy over ``P(grade > k)``, k = 0 .. K-2.

        Args:
            logits: ``(B, K)`` class logits.
            target: ``(B,)`` class indices in ``0 .. K-1``.

        Returns:
            Scalar mean loss.
        """
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"expected logits of shape (B, {self.num_classes}), got {tuple(logits.shape)}"
            )

        logp = torch.log_softmax(logits, dim=-1)
        # Both tails as cumulative log-sum-exps, so nothing subtracts two exponents
        # and the extremes (k at either end) stay exact rather than clamped. The
        # reversed cumsum at index k is P(grade >= k), hence the offset slice below.
        log_le = torch.logcumsumexp(logp, dim=-1)[
            :, :-1
        ]  # log P(grade <= k), k = 0..K-2
        log_gt = torch.logcumsumexp(logp.flip(-1), dim=-1).flip(-1)[
            :, 1:
        ]  # log P(grade > k)

        thresholds = torch.arange(self.num_classes - 1, device=logits.device)
        pos = (target[:, None] > thresholds).to(logits.dtype)
        return -(pos * log_gt + (1.0 - pos) * log_le).mean()


__all__ = ["OrdinalCrossEntropy"]
