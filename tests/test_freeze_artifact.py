"""scripts/freeze_artifact.py: the manifest must describe the bytes it actually copied.

A frozen artifact is what leaves the building, so the two things worth asserting are that the
sha256 in the manifest is the sha256 of the copied file (not of the source it was meant to
copy), and that re-freezing into a directory that already has a manifest is refused rather than
silently replacing weights someone may already be scoring.
"""

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import freeze_artifact


def make_checkpoint(root: Path, variant: str, payload: bytes) -> Path:
    directory = root / f"ml_attention-{variant}" / "checkpoints"
    directory.mkdir(parents=True)
    path = directory / "best.ckpt"
    path.write_bytes(payload)
    return path


with TemporaryDirectory() as directory:
    root = Path(directory) / "checkpoints"
    kfold = Path(directory) / "kfold"
    kfold.mkdir()
    (kfold / "labels-kf0.csv").write_text("slide_id,grade,split\ns0,1,train\n")
    make_checkpoint(root, "kf0-cand", b"weights-of-kf0")
    make_checkpoint(root, "kf1-cand", b"weights-of-kf1-longer")
    out = Path(directory) / "artifact"

    argv = sys.argv
    sys.argv = [
        "freeze_artifact.py",
        "--variants",
        "kf0-cand,kf1-cand",
        "--checkpoint-root",
        str(root),
        "--kfold-dir",
        str(kfold),
        "--out",
        str(out),
        "--recipe",
        "test recipe",
    ]
    try:
        freeze_artifact.main()
    finally:
        sys.argv = argv

    manifest = json.loads((out / "manifest.json").read_text())
    assert [w["variant"] for w in manifest["weights"]] == ["kf0-cand", "kf1-cand"]
    assert manifest["recipe"] == "test recipe"
    for entry, payload in zip(manifest["weights"], [b"weights-of-kf0", b"weights-of-kf1-longer"]):
        copied = (out / entry["file"]).read_bytes()
        assert copied == payload, entry
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest(), entry
        assert entry["bytes"] == len(payload), entry
    assert manifest["weights"][0]["labels_csv"] == "labels/labels-kf0.csv"
    assert "labels_csv" not in manifest["weights"][1]  # kf1-cand's CSV was not there to copy

    # A second freeze into the same directory must not quietly replace shipped weights.
    sys.argv = ["freeze_artifact.py", "--variants", "kf0-cand", "--out", str(out)]
    try:
        freeze_artifact.main()
        raise AssertionError("re-freezing into a frozen directory should refuse")
    except SystemExit as error:
        assert "refusing to overwrite" in str(error), error

# Ambiguous weights are an error, not a coin flip on glob order.
with TemporaryDirectory() as directory:
    root = Path(directory)
    make_checkpoint(root, "kf0-cand", b"a")
    (root / "ml_attention-kf0-cand" / "old" / "checkpoints").mkdir(parents=True)
    (root / "ml_attention-kf0-cand" / "old" / "checkpoints" / "best.ckpt").write_bytes(b"b")
    try:
        freeze_artifact.find_checkpoint(str(root), "kf0-cand")
        raise AssertionError("two best.ckpt files should be refused")
    except ValueError as error:
        assert "ambiguous" in str(error), error

print("ok: freeze writes a manifest that matches the copied bytes and refuses to overwrite")
