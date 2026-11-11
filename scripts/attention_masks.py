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

Value encoding (repeated in the report YAML):
    0     no tile here (background / QC-filtered)
    1-255 log10(attention / equal share), rescaled per slide so that the
          brightest tile on that slide is 255 and a tile with exactly an
          equal share of the attention is 1. Reading: "how many times more
          than an average tile did the model look here".
    Per-slide rescaling is deliberate -- attention is a softmax over one slide,
    so absolute weights are not comparable across slides. How FOCUSED a slide
    was is reported separately as eff_tiles = 1/sum(w^2) in summary.csv.

Usage (cluster):
    python scripts/attention_masks.py --root /mnt/projects/breast_cancer/brc_grading \
        --split test --stage all --limit 3
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.models.module import ModelModule

TILE = 224   # tile footprint in level-1 px; stride == size, so tiles never overlap
UP = 2       # level-1 -> level-0 linear factor on these slides (0.2339 / 0.4678 mpp)


def encode_values(w: np.ndarray) -> np.ndarray:
    """Attention weights -> uint8 mask values (0 = no tile, 255 = brightest)."""
    n = len(w)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    r = np.where(w > 0, np.log10(np.maximum(w, 1e-12) * n), -np.inf)  # decades above an equal share
    denom = max(math.log10(max(float(w.max()) * n, 10.0)), 1e-9)
    return np.where(w > 0, 1 + np.round(254.0 * np.clip(r / denom, 0.0, 1.0)), 0).astype(np.uint8)


def stage_scores(args, sel, emb_dir, device):
    """One row per tile (mean attention over folds) + one row per slide."""
    models = []
    for ckpt in args.checkpoints:
        conf = OmegaConf.load(Path(ckpt).parent / "config.yaml")
        m = ModelModule.load_from_checkpoint(ckpt, cfg=conf, map_location="cpu")
        models.append(m.eval().to(device))
    print(f"[stage1] {len(models)} models on {device}", flush=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tiles, slides = [], []
    for i, row in enumerate(sel.itertuples(), 1):
        src = emb_dir / row.slide_id / f"{row.slide_id}.pt"
        if not src.exists():
            print(f"[warn] no bag for {row.slide_id}", flush=True)
            continue
        st = src.stat()
        cache = cache_dir / f"{row.slide_id}__{st.st_size}__{int(st.st_mtime)}.npz"
        if cache.exists():
            z = np.load(cache)
            w, logits, xs, ys = z["w"], z["logits"], z["x"], z["y"]
        else:
            bags = torch.load(src, map_location="cpu", weights_only=False)
            if isinstance(bags, dict):
                bags = [bags]
            bag = bags[0]
            xs, ys = bag["x"].numpy(), bag["y"].numpy()
            emb = bag["embedding"].unsqueeze(0).to(device).float()
            ws, ls = [], []
            with torch.no_grad():
                for m in models:
                    feats = m.model(emb).float()
                    wt, _ = m.aggregator(feats, None)
                    ws.append(wt.squeeze(0).float().cpu().numpy())
                    ls.append(m.head(feats).squeeze(0).float().cpu().numpy())
            w = np.mean(ws, axis=0)                       # fold-average attention
            logits = np.mean(ls, axis=0)                  # fold-average logits, then softmax
            del bags, emb
            np.savez_compressed(cache, w=w, logits=logits, x=xs, ys=ys)
        probs = logits.softmax(0) if logits.ndim == 1 else logits.mean(0).softmax(0)
        p = int(probs.argmax())
        t = int(row.grade)
        slides.append({"slide_name": row.slide_id, "case": row.case, "truth": t, "pred": p,
                       "correct": int(t == p), "n_tiles": len(w),
                       "max_weight": round(float(w.max()), 5),
                       "eff_tiles": round(1.0 / max(float((w ** 2).sum()), 1e-12), 1),
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
#   1-255 = log10(attention / equal share of this slide's attention), rescaled
#           per slide so the brightest tile is 255. Red = the tiles the grade
#           head leaned on most; blue = barely above an average tile.
# Focus per slide (eff_tiles = 1/sum(w^2)) is in summary.csv next to the masks.
#
# Run:
#     python -m report --config-dir report_conf reporter=attention user=<you>

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
          - "#2166AC"     # weak  (near an equal share)
          - "#F7F7F7"     # medium
          - "#B2182B"     # strong (the tiles the grade head leaned on)
          steps: 4
          mode: qualitative
          interactive: true
          title: 'Attention:'
          continuous: false
        threshold:
          breaks:
          - 0.003922     # value 0 = no tile
          - 0.35
          - 0.70
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
    dest = Path(args.report_yaml) if args.report_yaml else out_dir.parent / "report_conf" / "reporter" / "attention.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(cfg)
    print(f"[stage3] {dest} ({len(rows)} slides)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--root", default="/mnt/projects/breast_cancer/brc_grading")
    ap.add_argument("--mask-dir", default=None, help="default <root>/masks/<split>")
    ap.add_argument("--report-yaml", default=None)
    ap.add_argument("--cache-dir", default=None, help="per-slide score cache (default <root>/score_cache)")
    ap.add_argument("--stage", choices=["scores", "masks", "report", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        w = np.array([0.0, 0.5, 0.001, 0.499])
        v = encode_values(w)
        assert v.max() == 255 and v[0] == 0, v            # brightest -> 255, zero weight -> 0
        assert v[1] == 255 and v[3] == 255, v             # the two dominant tiles saturate
        flat = encode_values(np.full(100, 0.01))          # every tile an equal share
        assert set(flat.tolist()) == {1}, flat
        ramp = encode_values(np.array([1e-6, 1e-4, 1e-2, 0.5]))
        assert np.all(np.diff(ramp) > 0), ramp            # monotone in weight
        assert encode_values(np.zeros(0)).size == 0
        print("selftest ok", v.tolist(), ramp.tolist())
        return

    cfg = OmegaConf.load(args.config)
    labels = pd.read_csv(Path(args.root) / "labels.csv")
    mapping = pd.read_csv(Path(args.root) / "data_mapping.csv")
    mapping["slide_name"] = mapping.path.str.split("/").str[-2]
    emb_dir = Path(args.root) / "embeddings" / "bc20-24"
    args.cache_dir = args.cache_dir or str(Path(args.root) / "score_cache")
    args.mask_dir = args.mask_dir or str(Path(args.root) / "masks" / args.split)
    args.checkpoints = args.checkpoints or [
        str(Path(cfg.paths.output_dir) / f"attention-kf{k}" / "last.ckpt") for k in range(5)]

    sel = labels[labels.split == args.split].merge(
        mapping[["slide_name", "path", "case"]], left_on="slide_id", right_on="slide_name")
    if args.limit:
        sel = sel.head(args.limit)
    print(f"[init] {len(sel)} slides split={args.split} stage={args.stage}", flush=True)

    if args.stage in ("scores", "all"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        stage_scores(args, sel, emb_dir, device)
    if args.stage in ("masks", "all"):
        tile_df = pd.read_csv(Path(args.mask_dir) / "attention_scores.csv")
        stage_masks(args, tile_df, sel)
    if args.stage in ("report", "all"):
        summary = pd.read_csv(Path(args.mask_dir) / "summary.csv")
        stage_report(args, summary, sel)


if __name__ == "__main__":
    main()
