"""Per-slide prediction dumps (GradingModule.predictions_dir).

Ensembling runs and paired significance tests need per-slide logits, which the
epoch-level metrics in MLflow do not keep, so the module writes them per epoch.
"""

import csv
import sys
import tempfile
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.models import GradingModule
from ml.models.aggregators import MeanAggregator
from ml.models.encoders import IdentityEncoder
from ml.models.heads import LinearHead


def build(predictions_dir: str | None) -> GradingModule:
    return GradingModule(
        encoder=IdentityEncoder(out_dim=8),
        aggregator=MeanAggregator(feature_dim=8),
        head=LinearHead(in_dim=8, out_dim=4),
        predictions_dir=predictions_dir,
    )


# Unset (Hydra renders ${oc.env:PREDICTIONS_DIR,null} as the string "null") stays inert.
assert build(None).predictions_dir is None
assert build("null").predictions_dir is None
assert build("").predictions_dir is None
assert str(build("/tmp/preds").predictions_dir) == "/tmp/preds"

with tempfile.TemporaryDirectory() as tmp:
    module = build(tmp)
    module._pred_rows["test"] = [
        ("slide_a", 2, torch.tensor([0.1, -1.0, 3.5, -0.25])),
        ("slide_b", 0, torch.zeros(4)),
    ]
    module._dump_predictions("test")

    with Path(tmp, "test_epoch0.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["slide_id", "label"] + [f"logit_{i}" for i in range(4)], rows[0]
    assert [r[0] for r in rows[1:]] == ["slide_a", "slide_b"], (
        "one row per slide, in step order"
    )
    assert (
        rows[1][1] == "2" and rows[1][2] == "0.100000" and rows[1][5] == "-0.250000"
    ), rows[1]
    assert not module._pred_rows["test"], (
        "rows are cleared, so epochs do not accumulate"
    )

    # Nothing collected (predictions off, or an empty split) writes nothing at all.
    module._dump_predictions("test")
    assert list(Path(tmp).glob("*.csv")) == [Path(tmp, "test_epoch0.csv")]

with tempfile.TemporaryDirectory() as tmp:
    silent = build(None)
    silent._pred_rows["valid"] = [("slide_a", 1, torch.zeros(4))]
    silent._dump_predictions("valid")  # must not raise, must not write anywhere
    assert list(Path(tmp).iterdir()) == []

print("prediction dump test passed")
