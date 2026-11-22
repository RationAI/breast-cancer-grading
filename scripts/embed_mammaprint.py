"""Virchow2 embeddings for kissmi's 308 MammaPrint slides at OUR magnification.

Not the preprocessing.embeddings job: that one needs the tiling->MLflow chain and
tissue/QC masks which exist only for the grading cohort. We replicate instead the
exact recipe the bc20-24 features were made with, so the MIL sees the same
feature distribution it was trained on:

* level = OpenSlide.closest_level(0.5)  (same level choice as ratiopath read_slides)
* grid  = ratiopath grid_tiles(..., last="keep") with tile 224 / stride 224
* bag   = tiles whose central 112x112 ROI has tissue_overlap > 0
          (preprocessing/tiling.py rule; tissue mask computed at level 3 exactly
          like preprocessing/tissue_masks.py, ROI scaled by the level-downsample
          ratio -- shortcut: mask geometry follows openslide downsample factors,
          the original mapped by mpp, identical up to rounding)
* pixels/tensor: ml.data.datasets.tiles.TileDataset + A.Normalize + ToTensorV2,
  model + save: preprocessing.embeddings.{Virchow2, save_embeddings}

Output: /mnt/projects/breast_cancer/brc_grading/embeddings/mou-bc/<slide_id>.parquet
with columns x, y, embedding (level-px coords, float32 2560-d) -- the same layout
as embeddings/bc20-24, which is what scripts/attention_masks.py reads.

Resumable: a slide is skipped when its parquet exists with the expected row count.

Usage (cluster job with cuda + HF_TOKEN + HF_HOME, storage DATA+PROJECTS):
    python scripts/embed_mammaprint.py --batch 512
"""

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/projects/breast_cancer/brc_grading")
EXPLAIN_RUN = "6f27144cbb574e7794b997003069b95f"
MPP = 0.5  # the bc20-24 tiling target, NOT kissmi's 0.97119
TILE = 224
STRIDE = 224
MASK_LEVEL = 3  # configs/preprocessing/tissue_masks.yaml


def tissue_keep(slide_path: str, level: int, tiles: list[tuple[int, int]]) -> list[int]:
    """Indices of tiles with any tissue in the central ROI (training bag rule)."""
    import numpy as np
    import pyvips
    from rationai.masks import tissue_mask
    from ratiopath.openslide import OpenSlide

    with OpenSlide(slide_path) as sl:
        mask_level = min(MASK_LEVEL, sl.level_count - 1)
        f = round(sl.level_downsamples[mask_level] / sl.level_downsamples[level])
        mpp_x, mpp_y = sl.slide_resolution(mask_level)
        img = pyvips.Image.new_from_file(slide_path, level=mask_level)
    m = np.asarray(tissue_mask(img, mpp=(mpp_x + mpp_y) / 2))
    if m.ndim == 3:
        m = m[..., 0]
    off, size = TILE // 4, TILE // 2  # centered 50 % ROI in tile px
    return [
        i
        for i, (x, y) in enumerate(tiles)
        if m[(y + off) // f : (y + off + size) // f, (x + off) // f : (x + off + size) // f].any()
    ]


def slide_tiles(slide_path: str) -> tuple[int, list[tuple[int, int]]]:
    """(level, kept tile coords) exactly as the training tiling would produce."""
    from ratiopath.openslide import OpenSlide
    from ratiopath.tiling import grid_tiles

    with OpenSlide(slide_path) as sl:
        level = sl.closest_level(MPP)
        w, h = sl.level_dimensions[level]
    grid = list(grid_tiles(slide_extent=(w, h), tile_extent=(TILE, TILE),
                           stride=(STRIDE, STRIDE), last="keep"))
    return level, [grid[i] for i in tissue_keep(slide_path, level, grid)]


def embed_slide(model: torch.nn.Module, slide_path: str, out: Path,
                batch: int, device: torch.device, workers: int) -> int:
    import albumentations as A
    from datasets import Dataset as HFDataset

    from ml.data.datasets.tiles import TileDataset
    from preprocessing.embeddings import embeddings_are_complete, save_embeddings

    level, tiles = slide_tiles(slide_path)
    if out.exists() and embeddings_are_complete(out, len(tiles)):
        return 0
    ds = TileDataset(
        slide_path=slide_path, level=level, tile_extent_x=TILE, tile_extent_y=TILE,
        tiles=HFDataset.from_list([{"x": int(x), "y": int(y)} for x, y in tiles]),
        transforms=A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))]),
    )
    dl = DataLoader(ds, batch_size=batch, num_workers=workers,
                    pin_memory=device.type == "cuda")
    embs = torch.zeros((len(tiles), model.embed_dim), dtype=torch.float32)
    xs = np.zeros(len(tiles), dtype=np.int32)
    ys = np.zeros(len(tiles), dtype=np.int32)
    for i, (x, metadata) in enumerate(dl):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            e = model(x)
        s = i * batch
        embs[s:s + e.size(0)] = e.float().cpu()
        xs[s:s + e.size(0)] = metadata["x"].numpy()
        ys[s:s + e.size(0)] = metadata["y"].numpy()
    save_embeddings(embs, torch.from_numpy(xs), torch.from_numpy(ys), out)
    return len(tiles)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)  # 512 A40, 1024 H100, 128 mig-1g
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "embeddings" / "mou-bc"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        # ROI -> mask-pixel mapping: 112 px window at tile level, mask at 4x coarser
        # ds (f=4), must be the same 14 px window, and edge tiles must not crash.
        m = np.zeros((100, 100), dtype=np.uint8)
        f = 4  # mask level ds / tile level ds
        m[21:35, 21:35] = 255  # tissue at mask px 21..34 = tile px 84..136 (ROI 56..168)
        roi = lambda x, y: m[(y + 56) // f : (y + 168) // f, (x + 56) // f : (x + 168) // f]
        assert roi(0, 0).any()  # tile (0,0): ROI 56..168 -> mask 14..42 hits 21..34
        assert not roi(224, 0).any()
        assert not roi(0, 300).any()  # edge tile beyond mask -> empty slice, dropped
        print("self-test ok")
        return 0

    from mlflow.tracking import MlflowClient
    from report_variants import use_cluster_mlflow

    client = MlflowClient(tracking_uri=use_cluster_mlflow())
    slides = pd.read_parquet(client.download_artifacts(EXPLAIN_RUN, "explanations/slides.parquet"))
    slides = slides[slides.status == "ok"].sort_values("slide_id")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from preprocessing.embeddings import Virchow2

    model = Virchow2().to(device).eval()

    todo = [(r.slide_id, r.slide_path) for r in slides.itertuples()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"slides: {len(todo)} -> {out_dir}", flush=True)
    t0 = time.monotonic()
    for n, (sid, path) in enumerate(todo, 1):
        t1 = time.monotonic()
        try:
            k = embed_slide(model, path, out_dir / f"{sid}.parquet",
                            args.batch, device, args.workers)
            msg = f"{k} tiles" if k else "complete, skipped"
        except Exception as e:  # one broken WSI must not kill the night's job
            msg = f"FAILED: {type(e).__name__}: {e}"
        rate = n / (time.monotonic() - t0)
        eta = datetime.now(UTC) + timedelta(hours=(len(todo) - n) / rate / 3600)
        print(f"[{n}/{len(todo)}] {sid}: {msg} ({time.monotonic() - t1:.0f}s, "
              f"ETA {eta:%H:%M} UTC)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
