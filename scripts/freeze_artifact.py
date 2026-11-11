"""Freeze a shipped artifact: copy the fold checkpoints into one dated directory plus a manifest.

The weights that scored 0.796 live in five directories named after their job
(``checkpoints/ml_attention-kf3-cand/``), which is fine for iterating and dangerous for handing
over: nothing says what they are, and a later wave with the same tag would overwrite them. This
copies them, the labels CSV they trained on, and a manifest of sha256s into one directory that is
safe to point a pathologist or an external cohort at.

Run as a report job (CPU only, reads and writes the share):
    REPORT_JOB_NAME=brc-grading-freeze REPORT_SCRIPT=freeze_artifact.py \
      REPORT_VARIANTS=kf0-cand,kf1-cand,kf2-cand,kf3-cand,kf4-cand \
      REPORT_ARGS="--recipe 'attention pooling, dropout 0.2, whole bags, patience 5' \
                   --note 'rule lambda fitted out-of-fold; see MLflow experiment 82'" \
      uv run --no-sync python scripts/run_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from glob import glob
from pathlib import Path


CHECKPOINT_ROOT = "/mnt/projects/breast_cancer/brc_grading/checkpoints"
KFOLD_DIR = "/mnt/projects/breast_cancer/brc_grading/kfold"
ARTIFACT_DIR = "/mnt/projects/breast_cancer/brc_grading/artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_checkpoint(root: str, variant: str) -> Path:
    """The ``best.ckpt`` of one variant. Globbed, because Lightning's dirpath has moved before.

    Several matches are an error rather than a coin flip: two candidate weights for one variant
    means the manifest would freeze whichever one glob happened to like.
    """
    hits = sorted(glob(f"{root}/ml_attention-{variant}/**/best.ckpt", recursive=True))
    if not hits:
        raise FileNotFoundError(f"no best.ckpt under {root}/ml_attention-{variant}")
    if len(hits) > 1:
        raise ValueError(f"{variant}: {len(hits)} best.ckpt files, ambiguous: {hits}")
    return Path(hits[0])


def fold_of(variant: str) -> str:
    """``kf3-cand`` was trained on ``labels-kf3.csv``; a bare name has no fold CSV."""
    fold, sep, _ = variant.partition("-")
    return fold if sep and re.fullmatch(r"kf\d+", fold) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True, help="comma-separated job variants")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--kfold-dir", default=KFOLD_DIR)
    parser.add_argument("--out", default="", help="default artifacts/foldavg-<today>")
    parser.add_argument("--recipe", default="", help="free text: what these weights are")
    parser.add_argument("--note", default="", help="free text: how to use them")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out = Path(
        args.out
        or f"{ARTIFACT_DIR}/foldavg-{datetime.now(UTC):%Y%m%d-%H%M}"
    )
    if (out / "manifest.json").exists():
        raise SystemExit(f"FATAL: {out} is already frozen; refusing to overwrite a shipped artifact")
    (out / "weights").mkdir(parents=True, exist_ok=True)

    entries = []
    for variant in variants:
        source = find_checkpoint(args.checkpoint_root, variant)
        target = out / "weights" / f"{variant}.ckpt"
        shutil.copyfile(source, target)
        entry = {
            "variant": variant,
            "source": str(source),
            "file": str(target.relative_to(out)),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
        fold = fold_of(variant)
        if fold:
            csv = Path(args.kfold_dir) / f"labels-{fold}.csv"
            if csv.exists():
                (out / "labels").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(csv, out / "labels" / csv.name)
                entry["labels_csv"] = f"labels/{csv.name}"
        entries.append(entry)
        print(f"froze {variant}: {entry['sha256'][:16]} {entry['bytes'] / 1e6:.1f} MB")

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": os.getenv("GIT_SHA")
        or os.getenv("CI_COMMIT_SHA")
        or subprocess_sha(),
        "recipe": args.recipe,
        "note": args.note,
        "weights": entries,
        "load": (
            "score each checkpoint over the new cohort (ml.train with checkpoint=..., "
            "eval.split=test, dataset.paths.labels=<cohort csv>), then average the softmax of the "
            "per-slide logits across the files below - the same averaging "
            "scripts/ensemble_variants.py does over folds"
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {out / 'manifest.json'}")


def subprocess_sha() -> str:
    """Best effort: the pod clones the repo, so HEAD is the commit the artifact was built from."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
