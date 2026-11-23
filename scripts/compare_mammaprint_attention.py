"""Does the MammaPrint index look where our grade model looks? Tile-level join.

Joins OUR fold-averaged attention (score_cache npz written by
``mammaprint_attention_stage.py``, raw un-saturated weights) with HIS per-tile
attributions (MLflow ``explanations/tiles/<slide>.parquet``: signed
``leave_one_out`` = how much the predicted index moves when the tile is removed,
plus his own MIL ``attention``).

Grids: his tiles are level 2, 224 px, STRIDE 112 (overlapping grid, mpp
0.97119); ours are level 1 224 px (mpp 0.4856 = half). His tile at X spans L1
[2X, 2X+448), so it contains exactly our columns {2X, 2X+224}: every his tile
covers a 2x2 block of ours and every our tile sits in up to 4 of his.

Per slide we report Spearman rho between our mean subtile attention and
``loo_drives = -leave_one_out`` (>0 means: our high-attention tiles are the ones
whose removal LOWERS the MammaPrint index, i.e. tiles actively pushing risk up),
against ``|LOO|`` (importance overlap regardless of direction) and against his
MIL attention. The 4 training-contaminated slides (check_mammaprint_overlap.py)
are computed but flagged and excluded from the headline.

Also writes, for the xOpat page (``mammaprint_report.py``) and for Jiri:
  masks/mammaprint-loo-lowers/<slide>.tiff  his negative.ome.tif renamed for layer matching
  masks/mammaprint-agree/<slide>.tiff       our top-10% AND his top-10% driving tiles
  mammaprint_compare/pairing.parquet        slide_id, x, y (HIS grid), our attention
  mammaprint_compare/agreement_by_slide.csv per-slide stats (all 308, flagged)
  mammaprint_compare/report_summary.csv     what mammaprint_report.py renders

Run as a CPU job once the mmattn job finished embedding::

    REPORT_JOB_NAME=brc-grading-mmattn-compare REPORT_SCRIPT=compare_mammaprint_attention.py \
      REPORT_CPU=4 REPORT_MEMORY=16Gi uv run --no-sync python scripts/run_report.py

Self-test without PVC/MLflow: ``--selftest``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/mnt/projects/breast_cancer/brc_grading")
DEFAULT_RUN = "6f27144cbb574e7794b997003069b95f"
# Slides present in BOTH cohorts: their MIL weights saw these during training, so
# any agreement there is memorisation, not signal (scripts/check_mammaprint_overlap.py).
CONTAMINATED = ["P2023_02960", "P2023_02864", "P2024_04450", "P2024_05043"]
HIS_TILES = "explanations/tiles"
HIS_MASK = "explanations/masks/leave_one_out/mammaprint_index"
MIN_TILES = 200


def his_parents(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray):
    """Our level-1 top-left -> the level-2 (224px, stride 112) tiles containing it.

    His window spans 448 of our px and starts every 224 of them, so our corner
    sits in his tiles starting at x and x-224 (up to 4 parents, fewer at 0).
    """
    x, y = np.asarray(x), np.asarray(y)
    for d in (0, 224):
        for e in (0, 224):
            yield np.maximum(x - d, 0) // 2, np.maximum(y - e, 0) // 2


def join_slide(ours: pd.DataFrame, his: pd.DataFrame) -> pd.DataFrame:
    """One row per HIS tile: our mean attention over the (up to 4) subtiles inside."""
    parts = [pd.DataFrame({"hx": hx, "hy": hy, "w": ours["w"].to_numpy()})
             for hx, hy in his_parents(ours["x"], ours["y"])]
    agg = (pd.concat(parts, ignore_index=True).groupby(["hx", "hy"])
           .agg(wsum=("w", "sum"), n_sub=("w", "size")).reset_index())
    m = his.merge(agg, left_on=["x", "y"], right_on=["hx", "hy"], how="left")
    m["our_attn"] = m["wsum"] / m["n_sub"]
    return m


def spearman(a, b) -> float:
    ok = pd.notna(a) & pd.notna(b)
    a, b = np.asarray(a, float)[ok], np.asarray(b, float)[ok]
    if len(a) < MIN_TILES or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def slide_stats(m: pd.DataFrame, *, contaminated: bool) -> dict:
    """Per-slide agreement numbers. Rho uses his tiles with >=1 of our subtiles."""
    cov = float((m["n_sub"].fillna(0) > 0).mean())        # before any filtering
    m = m[m["n_sub"].fillna(0) >= 1].copy()
    m["drives"] = -m["leave_one_out"]          # >0 = tile pushes the index UP
    return {
        "his_tiles": len(m),
        "coverage": cov,
        "loo_rho": spearman(m["our_attn"], m["drives"]),
        "absloo_rho": spearman(m["our_attn"], m["drives"].abs()),
        "attn_rho": spearman(m["our_attn"], m["attention"]),
        "contaminated": int(contaminated),
    }


def load_ours(npz: Path) -> pd.DataFrame:
    z = np.load(npz)
    return pd.DataFrame({"x": z["x"], "y": z["y"], "w": z["w"]})


def _selftest() -> int:
    global MIN_TILES
    MIN_TILES = 8                            # synthetic slides are tiny
    rng = np.random.default_rng(0)
    # slide ~4704 of our px: our grid 0..4256 step 224; his 0..2128 step 112
    his = pd.DataFrame({"x": np.arange(0, 2240, 112), "y": np.zeros(20, dtype=int)})
    his["leave_one_out"] = -his["x"]          # driving tiles = high x
    his["attention"] = his["leave_one_out"]
    ours = pd.DataFrame({"x": np.arange(0, 4480, 224), "y": np.zeros(20, dtype=int)})
    ours["w"] = 2 * ours["x"] + rng.normal(0, 1e2, len(ours))
    m = join_slide(ours, his)
    assert m["n_sub"].notna().all(), "every his tile must contain at least one of ours"
    inner = m[(m["x"] > 0) & (m["x"] < 2128)]
    assert (inner["n_sub"] == 4).all(), "interior his tiles hold exactly 4 of ours"
    s = slide_stats(m, contaminated=False)
    assert s["coverage"] == 1.0 and s["loo_rho"] > 0.98, s      # aligned -> strong +
    inv = ours.copy()
    inv["w"] = -inv["w"]
    assert slide_stats(join_slide(inv, his), contaminated=False)["loo_rho"] < -0.98
    # one of our tiles moved -> some his tiles lose a child (partial coverage kept)
    off = ours.copy()
    off.loc[0, "x"] = 8960
    assert (join_slide(off, his)["n_sub"] < 4).any()
    print("selftest ok: overlap-aware 4-child map, rho +-1 on synthetic data")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--run", default=DEFAULT_RUN, help="his explain MLflow run id")
    ap.add_argument("--checkpoints", default=None,
                    help="must match the attention stage (score-cache tag); default: same")
    ap.add_argument("--q", type=float, default=0.9, help="top-decile cutoff for the agree layer")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-agree-tiffs", action="store_true",
                    help="skip rendering the agreement layer TIFFs")
    ap.add_argument("--no-masks", action="store_true", help="skip downloading his LOO masks")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import attention_masks as am
    from report_variants import use_cluster_mlflow

    if args.checkpoints:
        checkpoints = args.checkpoints.split(",")
    else:  # same default as mammaprint_attention_stage: newest foldavg-* ensemble
        arts = sorted(args.root.glob("artifacts/foldavg-*"))
        checkpoints = sorted(str(p) for p in arts[-1].glob("weights/*.ckpt"))
    tag = hashlib.md5("|".join(str(Path(c).parent.name) for c in checkpoints).encode()).hexdigest()[:8]

    import mlflow
    from mlflow.tracking import MlflowClient

    use_cluster_mlflow()
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    out = args.root / "mammaprint_compare"
    out.mkdir(parents=True, exist_ok=True)
    loo_dir = args.root / "masks" / "mammaprint-loo-lowers"
    agree_dir = args.root / "masks" / "mammaprint-agree"
    score_dir = args.root / "score_cache"

    slides = pd.read_parquet(client.download_artifacts(args.run, "explanations/slides.parquet"))
    slides = slides.rename(columns={"slide_path": "path",
                                    "prediction_raw/mammaprint_index": "prediction",
                                    "label/mammaprint_index": "label"})
    ours_summary = pd.read_csv(args.root / "masks" / "mammaprint" / "summary.csv")
    print(f"comparing {len(slides)} slides, score tag {tag}", flush=True)

    rows, pairing = [], []
    agree_frames, sel_rows, done = [], [], 0
    for r in slides.itertuples():
        sid = r.slide_id
        npzs = sorted(score_dir.glob(f"{sid}__*__{tag}.npz"))
        if not npzs:
            print(f"[warn] {sid}: no score cache (embedding/attention stage missing it?)", flush=True)
            continue
        try:
            his = pd.read_parquet(client.download_artifacts(args.run, f"{HIS_TILES}/{sid}.parquet"))
        except Exception as exc:
            print(f"[warn] {sid}: no his tiles ({type(exc).__name__})", flush=True)
            continue
        m = join_slide(load_ours(npzs[0]), his)
        m["drives"] = -m["leave_one_out"]
        st = slide_stats(m, contaminated=sid in CONTAMINATED)
        rows.append({"slide_name": sid, "path": r.path, "prediction": r.prediction,
                     "label": r.label, **st})
        keep = m[m["n_sub"].fillna(0) >= 1]
        pairing.append(keep[["slide_id", "x", "y", "n_sub", "our_attn",
                             "leave_one_out", "attention"]].copy())
        # agreement layer on OUR grid: our tile is "agreed" when its own w is top-q
        # AND its containing his-tile drives the index top-q on this slide
        sub = m[m["n_sub"].fillna(0) >= 1]
        if len(sub) >= MIN_TILES:
            w_q = sub["w"].quantile(args.q)
            d_q = sub["drives"].quantile(args.q)
            ours = load_ours(npzs[0])
            driving = sub[sub["drives"] >= d_q][["x", "y"]].rename(columns={"x": "hx", "y": "hy"})
            driving_idx = pd.MultiIndex.from_frame(driving)
            hit = np.zeros(len(ours), dtype=bool)
            for hx, hy in his_parents(ours["x"], ours["y"]):
                hit |= pd.MultiIndex.from_arrays([hx, hy]).isin(driving_idx).to_numpy()
            flag = (ours["w"] >= w_q).to_numpy() & hit
            if flag.any():
                agree_frames.append(pd.DataFrame({
                    "slide_name": str(sid),
                    "x": ours["x"].to_numpy().astype(int)[flag],
                    "y": ours["y"].to_numpy().astype(int)[flag], "value": np.uint8(255)}))
            sel_rows.append((sid, r.path))
        done += 1
        if not args.no_masks:
            loo_dir.mkdir(parents=True, exist_ok=True)
            dst = loo_dir / f"{sid}.tiff"
            if not dst.exists():
                try:
                    src = Path(client.download_artifacts(args.run, f"{HIS_MASK}/{sid}_negative.ome.tif"))
                    dst.write_bytes(src.read_bytes())   # renamed: layer item_name == slide stem
                except Exception as exc:
                    print(f"[warn] {sid}: LOO mask not fetched ({type(exc).__name__})", flush=True)
        if done % 40 == 0:
            print(f"  ...{done}/{len(slides)} slides", flush=True)
        if args.limit and done >= args.limit:
            break

    by_slide = pd.DataFrame(rows)
    clean = by_slide[by_slide["contaminated"] == 0]
    pair = pd.concat(pairing, ignore_index=True)
    pair.to_parquet(out / "pairing.parquet", index=False)

    # slide-level: does his continuous index track our grade probability at all?
    s = clean.merge(ours_summary[["slide_name", "eff_tiles", "max_weight", "probs"]],
                    left_on="slide_name", right_on="slide_name", how="left")
    s["p_g3"] = s["probs"].map(lambda p: json.loads(p)[3] if pd.notna(p) else np.nan)
    s = s.drop(columns="probs")
    by_slide.to_csv(out / "agreement_by_slide.csv", index=False)
    s.to_csv(out / "report_summary.csv", index=False)
    print(f"\n{len(by_slide)} slides joined; {len(clean)} clean "
          f"({len(by_slide) - len(clean)} contaminated excluded)")
    for col in ("loo_rho", "absloo_rho", "attn_rho"):
        print(f"  median {col:<10} {clean[col].median():+.3f}  "
              f"(IQR {clean[col].quantile(.25):+.3f} .. {clean[col].quantile(.75):+.3f}; "
              f"share > 0: {float((clean[col] > 0).mean()):.2f})")
    print(f"  mean tile coverage {clean['coverage'].mean():.3f}")
    print(f"  slide-level spearman(prediction, our p_G3) = "
          f"{s[['prediction', 'p_g3']].corr(method='spearman').iloc[0, 1]:+.3f}")

    if agree_frames and not args.no_agree_tiffs:
        agree = pd.concat(agree_frames, ignore_index=True)
        sel = pd.DataFrame(sel_rows, columns=["slide_id", "path"])
        am.stage_masks(argparse.Namespace(mask_dir=str(agree_dir), limit=0), agree, sel)
    print(f"-> {out}/agreement_by_slide.csv, report_summary.csv, pairing.parquet "
          f"({len(pair):,} his-grid rows); layers in masks/mammaprint-agree "
          f"(+ downloaded LOO masks in masks/mammaprint-loo-lowers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
