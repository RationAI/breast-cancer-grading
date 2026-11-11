"""Epoch pairing and end-to-end selection in scripts/ensemble_variants.py.

Lightning bumps ``current_epoch`` before the final test, so a run dumps
``test_epoch18.csv`` beside ``valid_epoch0..17.csv``. Pairing those by equal number matched
nothing and the first ensemble job reported "no dumps" for variants that plainly had them.
The epoch that produced the test frame is the one stored in ``best.ckpt``; without a readable
checkpoint the fallback is the last dumped validation epoch.
"""

import contextlib
import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ensemble_variants


LABELS = [0, 0, 1, 2, 3]


def dump(path: Path, labels: list[int], correct: bool) -> None:
    """One predictions CSV: every slide right, or every slide predicted class 0."""
    lines = ["slide_id,label,logit_0,logit_1,logit_2,logit_3"]
    for i, label in enumerate(labels):
        predicted = label if correct else 0
        logits = [-2.0] * 4
        logits[predicted] = 2.0
        lines.append(f"slide{i},{label}," + ",".join(f"{v:.6f}" for v in logits))
    path.write_text("\n".join(lines))


def make_variant(root: Path, name: str, correct: bool, epochs: int = 18) -> Path:
    """Dumps as on the share: ``test_epoch<epochs>.csv`` beside ``valid_epoch0..<epochs-1>``."""
    predictions = root / f"ml_attention-{name.replace('_', '-')}" / "predictions"
    predictions.mkdir(parents=True)
    dump(predictions / f"test_epoch{epochs}.csv", LABELS, correct=correct)
    for epoch in range(epochs):
        dump(predictions / f"valid_epoch{epoch}.csv", LABELS, correct=correct)
    return predictions


with TemporaryDirectory() as directory:
    root = Path(directory)
    predictions = make_variant(root, "full_ord_s456", correct=True)

    # No checkpoint to ask: fall back to the last validation dump, never to epoch 18.
    epoch, valid, test = ensemble_variants.load_variant(str(root), "full_ord_s456")
    assert epoch == "test18/valid17", epoch
    assert len(valid) == 5 and len(test) == 5

    # With one, take exactly the epoch it was saved at - even if an earlier epoch looks better
    # on validation, since the decision rule must be fitted on the model that produced test.
    torch.save({"epoch": 7}, predictions.parent / "best.ckpt")
    epoch, _, _ = ensemble_variants.load_variant(str(root), "full_ord_s456")
    assert epoch == "test18/valid7", epoch

    missing = root / "ml_attention-nothing-s000" / "predictions"
    missing.mkdir(parents=True)
    dump(missing / "test_epoch3.csv", LABELS, correct=True)
    assert ensemble_variants.load_variant(str(root), "nothing_s000") is None

print("ok: test/valid dumps pair on the tested checkpoint's epoch, with a fallback")

# --- main() end to end: selection must pick the informative member ---
with TemporaryDirectory() as directory:
    root = Path(directory)
    make_variant(root, "good_s1", correct=True, epochs=4)
    make_variant(root, "bad_s2", correct=False, epochs=4)

    argv = sys.argv
    sys.argv = [
        "ensemble_variants.py",
        "--checkpoint-root",
        str(root),
        "--variants",
        "bad_s2,good_s1",
    ]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ensemble_variants.main()
    finally:
        sys.argv = argv
    report = out.getvalue()
    assert "members: ['good_s1'" in report, report
    selected = [line for line in report.splitlines() if "selected blend" in line]
    assert selected and all("bal=1.0000" in line for line in selected), selected
    assert "greedy val curve" in report and "bias/temp" in report

print(
    "ok: validation-only selection picks the informative member and reports it once on test"
)
