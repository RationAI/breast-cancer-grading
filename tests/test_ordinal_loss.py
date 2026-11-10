"""The two loss variants in ml/models/losses.py (ordinal, class-weighted CE)."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.models.losses import OrdinalCrossEntropy, weighted_cross_entropy


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

# Class weights have to arrive as a Tensor: Hydra gives CrossEntropyLoss a ListConfig
# and torch refuses it at instantiation, i.e. 10 minutes into a pod.
weighted = weighted_cross_entropy([0.28, 2.48, 0.66, 0.58])
assert torch.is_tensor(weighted.weight) and weighted.weight.dtype is torch.float32
assert weighted.weight.shape == (4,)
plain = torch.nn.CrossEntropyLoss()
uniform = torch.nn.CrossEntropyLoss(weight=torch.ones(4))
flat = torch.tensor([[2.0, -2.0, 0.0, 0.0]])
assert torch.allclose(
    weighted_cross_entropy([1.0, 1.0, 1.0, 1.0])(flat, torch.tensor([1])),
    plain(flat, torch.tensor([1])),
), "uniform weights must be plain CE"

# A batch of two slides: the rare class (1) answered wrongly, class 0 answered right.
# PyTorch's weighted "mean" divides by the *sum of the selected weights*, so weights
# rebalance classes against each other instead of scaling the loss (a single-sample
# batch cancels them exactly), which is why this needs a mixed-class batch.
x = torch.tensor([[2.0, -2.0, 0.0, 0.0], [2.0, 0.0, -2.0, 0.0]])
y = torch.tensor([1, 0])
assert weighted(x, y) > uniform(x, y), "upweighting the missed class must cost more"
assert weighted_cross_entropy([2.48, 0.28, 0.66, 0.58])(x, y) < uniform(x, y), (
    "and upweighting the class the batch gets right must cost less"
)
assert float(weighted(x, y)) > 1.5 * float(uniform(x, y)), (
    f"the rare-class miss should dominate: {float(weighted(x, y)):.2f}"
    f" vs {float(uniform(x, y)):.2f}"
)

# The shipped config group has to instantiate, not just compose.
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


with initialize_config_dir(
    config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None
):
    cfg = compose(
        config_name="ml",
        overrides=["+experiment=ml/mil_attention", "ml/loss=weighted_cross_entropy"],
    )
    built = instantiate(cfg.ml.loss)
assert isinstance(built, torch.nn.CrossEntropyLoss), built
assert torch.allclose(built.weight, torch.tensor([0.28, 2.48, 0.66, 0.58])), (
    built.weight
)

print("loss tests passed (ordinal + weighted CE)")
