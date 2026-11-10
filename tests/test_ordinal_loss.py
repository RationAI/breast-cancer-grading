"""Ordinal loss for the ordered Nottingham grades (ml/models/losses.py)."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.models.losses import OrdinalCrossEntropy


loss = OrdinalCrossEntropy(num_classes=4)
perfect = torch.tensor([[8.0, -8.0, -8.0, -8.0], [-8.0, -8.0, 8.0, -8.0]])

# Two K-1 binary tasks: with two classes the loss must be exactly BCE-with-logits on
# the logit difference, which pins the ordering of `target > k` down to no slack.
two = OrdinalCrossEntropy(num_classes=2)
logits2 = torch.randn(16, 2, requires_grad=True)
target2 = torch.randint(0, 2, (16,))
assert torch.allclose(
    two(logits2, target2),
    F.binary_cross_entropy_with_logits(logits2[:, 1] - logits2[:, 0], target2.float()),
    atol=1e-6,
), "K=2 must reduce to binary cross-entropy"

# A far error must cost more than an adjacent one; cross-entropy cannot tell them apart.
near = loss(torch.tensor([[4.0, 2.0, -2.0, -2.0]]), torch.tensor([0]))
far = loss(torch.tensor([[4.0, -2.0, -2.0, 2.0]]), torch.tensor([0]))
assert far > 2 * near, f"adjacent {near:.3f} vs far {far:.3f}: ordering not encoded"
ce = torch.nn.CrossEntropyLoss()
assert (
    abs(
        ce(torch.tensor([[4.0, 2.0, -2.0, -2.0]]), torch.tensor([0])).item()
        - ce(torch.tensor([[4.0, -2.0, -2.0, 2.0]]), torch.tensor([0])).item()
    )
    < 1e-4
), "the comparison only means something because CE is blind to the difference"

# Correct predictions approach zero, and confidence on a wrong grade keeps paying.
assert loss(perfect, torch.tensor([0, 2])) < 1e-3
assert loss(perfect, torch.tensor([3, 0])) > 3 * loss(perfect, torch.tensor([1, 2]))

# Every threshold task contributes, so the loss is finite for extreme logits (the
# cumulative tails are log-sum-exp, not subtracted exponentials).
assert torch.isfinite(loss(torch.full((4, 4), 60.0), torch.arange(4)))
assert torch.isfinite(loss(torch.full((4, 4), -60.0), torch.arange(4)))

graded = perfect.clone().requires_grad_(True)
loss(graded, torch.tensor([0, 2])).backward()
assert graded.grad is not None and torch.isfinite(graded.grad).any()
assert (graded.grad != 0).any(), (
    "a correct prediction under a wrong label must still train"
)

for bad, want in (
    (lambda: loss(torch.randn(3, 3), torch.tensor([0, 1, 2])), "expected logits"),
    (lambda: OrdinalCrossEntropy(num_classes=1), "num_classes"),
):
    try:
        bad()
    except ValueError as exc:
        assert want in str(exc), f"wrong message: {exc}"
    else:
        raise AssertionError(f"{want!r} must be rejected")

print("ordinal loss test passed")
