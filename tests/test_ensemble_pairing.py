"""Epoch pairing in scripts/ensemble_variants.py.

Lightning bumps ``current_epoch`` before the final test, so a run dumps
``test_epoch18.csv`` beside ``valid_epoch0..17.csv``. Pairing those by equal number
matched nothing and the first ensemble job reported "no dumps" for variants that
plainly had them.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ensemble_variants


def dump(path: Path, labels: list[int], correct: bool) -> None:
    """One predictions CSV: every slide right, or every slide predicted class 0."""
    lines = ["slide_id,label,logit_0,logit_1,logit_2,logit_3"]
    for i, label in enumerate(labels):
        predicted = label if correct else 0
        logits = [-2.0] * 4
        logits[predicted] = 2.0
        lines.append(f"slide{i},{label}," + ",".join(f"{v:.6f}" for v in logits))
    path.write_text("\n".join(lines))


LABELS = [0, 0, 1, 2, 3]

with TemporaryDirectory() as directory:
    root = Path(directory)
    predictions = root / "ml_attention-full-ord-s456" / "predictions"
    predictions.mkdir(parents=True)
    # Test dump numbered one past the last validation dump, as on the share.
    dump(predictions / "test_epoch18.csv", LABELS, correct=True)
    for epoch in range(18):
        # Only epoch 7 gets every class right; the rest are all-class-0.
        dump(predictions / f"valid_epoch{epoch}.csv", LABELS, correct=epoch == 7)

    got = ensemble_variants.load_variant(str(root), "full_ord_s456")
    assert got is not None, "the off-by-one pairing is back"
    epoch, valid, test = got
    assert epoch == "test18/valid1.0000", epoch
    assert len(valid) == 5 and len(test) == 5, (len(valid), len(test))
    assert valid["logit_1"].iloc[2] > 0, "wrong validation epoch selected"

    missing = root / "ml_attention-nothing-s000" / "predictions"
    missing.mkdir(parents=True)
    dump(missing / "test_epoch3.csv", LABELS, correct=True)
    assert ensemble_variants.load_variant(str(root), "nothing_s000") is None

print("ok: test/valid epoch dumps pair by validation score, not by equal number")
