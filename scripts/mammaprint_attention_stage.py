"""Fold-average attention (our MIL) over the 308 MammaPrint slides.

Second half of the overnight job, after scripts/embed_mammaprint.py wrote
embeddings/mou-bc/. Reuses scripts/attention_masks.py stage functions verbatim
(so the TIFF overlays are byte-comparable with the cohort masks) but the slide
selection comes from kissmi's MLflow explanations/slides.parquet instead of
labels.csv -- these slides have no Nottingham grade, grade is logged as -1 and
the printed accuracy is meaningless for this subset.

Masks land in <root>/masks/mammaprint/<slide_id>.tiff. The xOpat report YAML is
skipped: the point here is the tile-level TIFFs for the comparison.
"""

import argparse
import sys
import types
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd


ROOT = Path("/mnt/projects/breast_cancer/brc_grading")
EXPLAIN_RUN = "6f27144cbb574e7794b997003069b95f"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--emb-dir", default=None, help="default <root>/embeddings/mou-bc")
    ap.add_argument("--mask-dir", default=None, help="default <root>/masks/mammaprint")
    args = ap.parse_args()

    import attention_masks as am
    from mlflow.tracking import MlflowClient
    from report_variants import use_cluster_mlflow

    client = MlflowClient(tracking_uri=use_cluster_mlflow())
    df = pd.read_parquet(client.download_artifacts(EXPLAIN_RUN, "explanations/slides.parquet"))
    df = df[df.status == "ok"]
    sel = pd.DataFrame({
        "slide_id": df.slide_id,
        "path": df.slide_path,
        "case": df.record_num,
        "grade": -1,  # no Nottingham grade here; truth/correct columns are noise
    })

    root = Path(args.root)
    arts = sorted(root.glob("artifacts/foldavg-*"))
    if not arts:
        raise SystemExit(f"no foldavg artifact under {root}/artifacts")
    a = types.SimpleNamespace(
        checkpoints=sorted(str(p) for p in arts[-1].glob("weights/*.ckpt")),
        dropout=0.2,
        scale="log",
        cache_dir=str(root / "score_cache"),
        mask_dir=args.mask_dir or str(root / "masks" / "mammaprint"),
        limit=0,
    )
    print(f"[mm] {len(sel)} slides, weights {Path(a.checkpoints[0]).parent}", flush=True)
    tile_df, _ = am.stage_scores(a, sel, Path(args.emb_dir or root / "embeddings" / "mou-bc"))
    am.stage_masks(a, tile_df, sel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
