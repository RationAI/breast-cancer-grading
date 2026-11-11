"""xOpat overlays: per-tile attention scores -> grayscale mask BigTIFFs.

Port of Tile-Morph-Clustering's make_mask_tiff.py + its report-config generator,
so the pathologist reviews this model the same way they reviewed the LDA masks
there. Stage 1 scores every bag with the fold-average MIL (no gradients, no
rendering), stage 2 rasterizes, stage 3 writes the xOpat report YAML.

Coordinates: tile x,y in the parquet/bag files are LEVEL-1 pixels -- tiles are
224 px at mpp 0.4678 while these slides are 0.2339 mpp at level 0 (verified on
2020_00782: 16 895 tiles reaching x=50 176 of a level-1 width of 52 592). So
blocks are laid down on a level-1 raster and the raster is upscaled 2x with
pixel replication, which is byte-identical to drawing at level 0 for a quarter
of the memory (same trick as make_mask_tiff.py).

Value encoding (repeated in the report YAML) -- one absolute scale for all
slides, so a colour means the same thing on every slide:
    0     no tile here (background / QC-filtered)
    1     the tile got exactly an equal share (1/N) of the slide's attention
    128   10x an equal share
    255   100x an equal share or more (saturated)
i.e. value = decades of attention above an equal share. A slide where nothing
stands out therefore stays pale, which is the point: how FOCUSED a slide was is
in summary.csv as eff_tiles = 1/sum(w^2).

Usage (cluster):
    python scripts/attention_masks.py --split test --stage all --limit 3   # smoke test
    python scripts/attention_masks.py                                      # the whole test split

CPU only: the head is 17 k parameters and the embeddings are already on disk
(600 tiles/s/core measured in job brc-grading-attn), so this needs no GPU.

Read this before showing the overlays: the weights are verified NOT degenerate
(20-415 tiles of a bag participate) but they are NOT yet shown to prefer tumour
epithelium over stroma -- the tumour-vs-stroma contrast measured 0.48 AUC, i.e.
below chance (REPORT.md 10). Present the overlay as "where the grade head looked",
not as "where the tumour is".
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TILE = 224    # tile footprint in level-1 px; stride == size, so tiles never overlap
FEATURE_DIM = 2560  # Virchow2
GRADES = ("N", "G1", "G2", "G3")  # class 0 = no tumour epithelium, 1-3 = Nottingham
BASE = Path("/mnt/projects/breast_cancer/brc_grading")
ARTIFACT = BASE / "artifacts" / "foldavg-20261111-1718"


def load_models(checkpoints, dropout: float = 0.2):
    """Rebuild the shipped architecture and load each fold's weights, strictly.

    The checkpoints carry no hyper_parameters (the module never calls
    save_hyperparameters), so the architecture is rebuilt by hand and
    ``strict=True`` makes a wrong layer width fail loudly instead of silently
    giving another model's attention.
    """
    from ml.models import AttentionMIL, IdentityEncoder, LinearHead
    from ml.models.module import GradingModule

    models = []
    for path in checkpoints:
        module = GradingModule(
            encoder=IdentityEncoder(out_dim=FEATURE_DIM),
            aggregator=AttentionMIL(feature_dim=FEATURE_DIM, attention_dim=512),
            head=LinearHead(in_dim=FEATURE_DIM, out_dim=len(GRADES), dropout=dropout),
        )
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        module.load_state_dict(state, strict=True)
        models.append(module.eval())
        print(f"[stage1] loaded {Path(path).parent.name} ({len(state)} tensors)", flush=True)
    return models


def score_bag(models, feats: np.ndarray):
    """Fold-average softmax probabilities and per-tile attention over one bag.

    Averaging the softmax is what the shipped manifest prescribes; both averages
    are taken in probability space because every fold normalises over the same bag.
    """
    bag = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))
    probs, attn = [], []
    with torch.no_grad():
        for module in models:
            logits, weight = module(bag)
            probs.append(torch.softmax(logits, dim=-1).numpy())
            attn.append(weight.numpy())
    p = np.mean(probs, axis=0)
    a = np.mean(attn, axis=0)
    return p / p.sum(), a / a.sum()


def participation(weight: np.ndarray) -> float:
    """How many tiles the attention actually spreads over (inverse perplexity)."""
    return float(1.0 / max(float((weight ** 2).sum()), 1e-12))


def encode_values(w: np.ndarray) -> np.ndarray:
    """Attention weights -> uint8 mask values (0 = no tile, 255 = 100x an equal share).

    Absolute scale, identical on every slide: value = decades of attention above
    an equal share (1/N), so 1 = an average tile, 128 = 10x, 255 = 100x or more.
    """
    n = len(w)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    r = np.where(w > 0, np.log10(np.maximum(w, 1e-12) * n), -np.inf)  # decades above an equal share
    return np.where(w > 0, 1 + np.round(254.0 * np.clip(r / 2.0, 0.0, 1.0)), 0).astype(np.uint8)


def stage_scores(args, sel, emb_dir):
    """One row per tile (fold-averaged attention) + one row per slide."""
    models = load_models(args.checkpoints, dropout=args.dropout)
    # cache is keyed by slide content AND by which weights produced the scores
    tag = hashlib.md5("|".join(str(Path(c).parent.name) for c in args.checkpoints).encode()).hexdigest()[:8]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tiles, slides = [], []
    for i, row in enumerate(sel.itertuples(), 1):
        src = emb_dir / f"{row.slide_id}.parquet"
        if not src.exists():
            print(f"[warn] no embeddings for {row.slide_id}", flush=True)
            continue
        st = src.stat()
        cache = cache_dir / f"{row.slide_id}__{st.st_size}__{int(st.st_mtime)}__{tag}.npz"
        if cache.exists():
            z = np.load(cache)
            w, probs, xs, ys = z["w"], z["probs"], z["x"], z["y"]
        else:
            bag = pd.read_parquet(src, columns=["x", "y", "embedding"])
            xs, ys = bag.x.to_numpy(), bag.y.to_numpy()
            probs, w = score_bag(models, np.stack(bag.embedding.values))
            del bag
            np.savez_compressed(cache, w=w, probs=probs, x=xs, ys=ys)
        p = int(probs.argmax())
        t = int(row.grade)
        slides.append({"slide_name": row.slide_id, "case": row.case, "truth": t, "pred": p,
                       "correct": int(t == p), "n_tiles": len(w),
                       "max_weight": round(float(w.max()), 5),
                       "eff_tiles": round(participation(w), 1),
                       "probs": json.dumps([round(float(v), 4) for v in probs])})
        tiles.append(pd.DataFrame({"slide_name": row.slide_id, "x": xs, "y": ys,
                                   "value": encode_values(w)}))
        if i % 25 == 0:
            print(f"[stage1] {i}/{len(sel)} slides", flush=True)
    out = Path(args.mask_dir)
    out.mkdir(parents=True, exist_ok=True)
    tile_df = pd.concat(tiles, ignore_index=True)
    summary = pd.DataFrame(slides)
    tile_df.to_csv(out / "attention_scores.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(f"[stage1] {len(tile_df)} tile rows, accuracy {summary.correct.mean():.4f} "
          f"on {len(summary)} slides -> {out/'attention_scores.csv'}", flush=True)
    return tile_df, summary


def stage_masks(args, tile_df, sel):
    """One level-0 uint8 pyramid BigTIFF per slide (resumable: existing files skipped)."""
    import openslide
    import pyvips

    wsi = {r.slide_id: (r.path if Path(r.path).exists() else r.path + ".mrxs") for r in sel.itertuples()}
    out_dir = Path(args.mask_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(tile_df.slide_name.drop_duplicates())
    if args.limit:
        names = names[: args.limit]
    for i, name in enumerate(names, 1):
        path = out_dir / f"{name}.tiff"
        if path.exists():
            continue
        slide_path = wsi.get(name)
        if slide_path is None or not Path(slide_path).exists():
            print(f"[skip] no WSI for {name}", flush=True)
            continue
        sl = openslide.OpenSlide(slide_path)
        w0, h0 = sl.dimensions
        w1, h1 = sl.level_dimensions[1]
        fx, fy = round(w0 / w1), round(h0 / h1)
        sl.close()
        mask = np.zeros((h1, w1), dtype=np.uint8)
        sub = tile_df[tile_df.slide_name == name]
        for x, y, v in zip(sub.x.to_numpy(), sub.y.to_numpy(), sub.value.to_numpy()):
            dx, dy = min(int(x), w1 - TILE), min(int(y), h1 - TILE)
            if dx < 0 or dy < 0:      # slide smaller than one tile at level 1
                continue
            mask[dy:dy + TILE, dx:dx + TILE] = v
        img = pyvips.Image.new_from_array(mask)
        del mask
        if fx != 1 or fy != 1:
            img = img.zoom(fx, fy)
        # odd level-0 dims: 2*level1 can be 1 px short of level 0 -> pad with background
        # (cropping there would fail with "bad extract area")
        if img.width < w0 or img.height < h0:
            img = img.embed(0, 0, max(w0, img.width), max(h0, img.height), extend="black")
        if img.width != w0 or img.height != h0:
            img = img.crop(0, 0, w0, h0)
        img.tiffsave(str(path), tile=True, pyramid=True, bigtiff=True, compression="lzw")
        print(f"[stage2] {i}/{len(names)} {name}: {len(sub)} tiles -> "
              f"{path.name} {w0}x{h0} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)


def stage_report(args, summary, sel):
    """xOpat report YAML for the masks (same shape as Tile-Morph's lda_morphology.yaml)."""
    out_dir = Path(args.mask_dir)
    have = {t.stem for t in sorted(out_dir.glob("*.tiff"))}
    if not have:
        raise SystemExit(f"no masks in {out_dir}")
    rows = [r for r in sel.itertuples() if r.slide_id in have]
    paths = "\n".join(f"    - {r.path}" for r in rows)
    cfg = f"""# xOpat report: MIL attention masks (fold-averaged attention, {len(rows)} slides)
#
# Masks are uint8 grayscale (0-255):
#   0    = no tile here (background / QC-filtered)
#   1    = an equal share of this slide's attention (1/N)
#   128  = 10x an equal share
#   255  = 100x an equal share or more
# One absolute scale for every slide, so a colour means the same thing everywhere;
# a pale slide is one where the grade head did not focus anywhere in particular.
# Focus per slide (eff_tiles = 1/sum(w^2)) is in summary.csv next to the masks.
# Caveat (REPORT.md 10): these weights are verified not-degenerate, but a
tumour-vs-stroma contrast of 0.48 AUC means we cannot claim they mark tumour.
# Show them as "where the model looked", not "where the tumour is".
#
# Run:
#     python -m report --config-dir report_conf reporter=attention_{out_dir.name} user=<you>

defaults:
  - _self_

title: Breast cancer Nottingham grade - MIL attention

background:
  _target_: report.masks.SlideRetriever
  paths:
{paths}
  layer_name: MAMMAPRINT WSI (MIRAX)

mask_retrievers:
  - _target_: report.masks.DriveMasksRetriever
    dir_name: {out_dir}
    layer_name: Attention (fold average)
    shader_conf:
      type: colormap
      params:
        use_channel0: r
        opacity: 0.55
        connect:
          type: bool
          interactive: false
          default: true
        color:
          type: custom_colormap
          default:
          - ''            # 0 = no tile, hidden
          - "#2166AC"     # weak  (< ~5x an equal share)
          - "#F7F7F7"     # medium (5-40x)
          - "#B2182B"     # strong (> 40x an equal share)
          steps: 4
          mode: qualitative
          interactive: true
          title: 'Attention:'
          continuous: false
        threshold:
          breaks:
          - 0.003922     # value 0 = no tile
          - 0.25         # value 64  ~ 5x an equal share
          - 0.63         # value 160 ~ 40x an equal share
          mask:
          - 0            # background: hidden
          - 1            # weak: shown
          - 1            # medium: shown
          - 1            # strong: shown
          interactive: false
          inverted: true
          maskOnly: true
          toggleMask: true
          title: Categories
          min: 0
          max: 1
          minGap: 0.05
          step: null
          pips:
            mode: count
            values: 4
            density: 4
            stepped: true

save:
  _target_: report.save.MLFlowReportStorer
  artifact_path: report.html
  metadata:
    experiment_name: Breast Cancer
    run_name: MIL attention masks
    description: Fold-averaged attention of the 4-class Nottingham grade MIL over MAMMAPRINT WSIs
    user: ${{user}}
"""
    dest = Path(args.report_yaml) if args.report_yaml else BASE / "report_conf" / "reporter" / f"attention_{out_dir.name}.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(cfg)
    print(f"[stage3] {dest} ({len(rows)} slides)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="default: weights/*.ckpt of the newest <root>/artifacts/foldavg-* (the shipped ensemble)")
    ap.add_argument("--root", default="/mnt/projects/breast_cancer/brc_grading")
    ap.add_argument("--mask-dir", default=None, help="default <root>/masks/<split>")
    ap.add_argument("--report-yaml", default=None)
    ap.add_argument("--cache-dir", default=None, help="per-slide score cache (default <root>/score_cache)")
    ap.add_argument("--dropout", type=float, default=0.2, help="must match the shipped recipe")
    ap.add_argument("--stage", choices=["scores", "masks", "report", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--slides", default=None, help="comma-separated slide ids (overrides --split/--limit)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        v = encode_values(np.array([0.0, 0.5, 0.001, 0.499]))
        assert v[0] == 0 and v[2] == 1 and v[1] > v[2], v   # no tile / equal share / above it
        assert set(encode_values(np.full(100, 0.01)).tolist()) == {1}  # every tile an equal share
        ten = encode_values(np.array([0.1] + [0.9 / 99] * 99))         # one tile at 10x the share
        assert abs(int(ten[0]) - 128) <= 1, ten
        big = encode_values(np.array([0.5] + [0.5 / 999] * 999))       # 500x -> saturated
        assert big[0] == 255 and big[1] == 1, big
        share = 1.0 / 250                                    # equal share of a 250-tile bag
        bag = np.full(250, share)
        bag[:4] = [share, 3 * share, 10 * share, 30 * share]
        ramp = encode_values(bag)[:4]
        assert np.all(np.diff(ramp) > 0), ramp               # strictly increasing while unsaturated
        assert np.all(np.diff(encode_values(np.array([1e-6, 1e-4, 1e-2, 0.5]))) >= 0)  # never decreases
        assert encode_values(np.zeros(0)).size == 0
        print("selftest ok", v.tolist(), ramp.tolist())
        return

    labels = pd.read_csv(Path(args.root) / "labels.csv")
    mapping = pd.read_csv(Path(args.root) / "data_mapping.csv")
    mapping["slide_name"] = mapping.path.str.split("/").str[-2]
    emb_dir = Path(args.root) / "embeddings" / "bc20-24"
    args.cache_dir = args.cache_dir or str(Path(args.root) / "score_cache")
    args.mask_dir = args.mask_dir or str(Path(args.root) / "masks" / args.split)
    if not args.checkpoints:
        arts = sorted(Path(args.root).glob("artifacts/foldavg-*"))
        args.checkpoints = sorted(str(p) for p in arts[-1].glob("weights/*.ckpt")) if arts else []
    if not args.checkpoints:
        raise SystemExit("no checkpoints: pass --checkpoints or keep the foldavg artifact")
    print(f"[init] {len(args.checkpoints)} models from "
          f"{Path(args.checkpoints[0]).parent}", flush=True)

    sel = labels[labels.split == args.split].merge(
        mapping[["slide_name", "path", "case"]], left_on="slide_id", right_on="slide_name")
    if args.slides:
        want = {s.strip() for s in args.slides.split(",")}
        sel = sel[sel.slide_id.isin(want)]
    if args.limit:
        sel = sel.head(args.limit)
    print(f"[init] {len(sel)} slides split={args.split} stage={args.stage}", flush=True)

    if args.stage in ("scores", "all"):
        stage_scores(args, sel, emb_dir)
    if args.stage in ("masks", "all"):
        tile_df = pd.read_csv(Path(args.mask_dir) / "attention_scores.csv")
        stage_masks(args, tile_df, sel)
    if args.stage in ("report", "all"):
        summary = pd.read_csv(Path(args.mask_dir) / "summary.csv")
        stage_report(args, summary, sel)


if __name__ == "__main__":
    main()
