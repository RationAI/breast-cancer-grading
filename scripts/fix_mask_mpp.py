"""Restamp real slide resolution onto masks written before stage_masks set xres.

Numpy->tiffsave pyramids defaulted to xres=1 px/cm (= "10 mm/px"), which makes
xOpat warn "possibly faulty layers" on the grade-attention and agreement layers.
Pixels are correct; only the TIFF resolution tags are wrong. This re-saves each
mask with the background slide's own xres/yres (masks are stamped on the level-0
grid, so the slide's level-0 resolution is the right value). Idempotent: files
whose xres is already plausible (>10 px/cm) are skipped.

Job (run AFTER compare_mammaprint_attention.py finished, or it races new files)::

    REPORT_JOB_NAME=brc-grading-mmmask-fix REPORT_SCRIPT=fix_mask_mpp.py REPORT_DATA=1 \
      REPORT_CPU=8 REPORT_MEMORY=16Gi uv run --no-sync python scripts/run_report.py
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyvips


DEFAULT_ROOT = Path("/mnt/projects/breast_cancer/brc_grading")


def slide_res(path: str) -> tuple[float, float]:
    p = Path(path)
    if not p.exists():
        p = Path(str(path) + ".mrxs")
    im = pyvips.Image.new_from_file(p, access="sequential")
    return float(im.xres), float(im.yres)


def fix_one(args: tuple[Path, float, float]) -> str:
    tiff, xres, yres = args
    im = pyvips.Image.new_from_file(tiff, access="sequential")
    if im.xres > 10:
        return f"skip {tiff.name}"
    tmp = tiff.with_suffix(".fix.tiff")
    im.copy(xres=xres, yres=yres).tiffsave(
        str(tmp), tile=True, pyramid=True, bigtiff=True, compression="lzw")
    tmp.replace(tiff)  # atomic on the same fs; a crash leaves the original intact
    return f"fixed {tiff.name} xres={xres:.1f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--dirs", nargs="+", default=["masks/mammaprint", "masks/mammaprint-agree"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    # same slide -> his-WSI-path table the mask render stage used
    slides = pd.read_csv(args.root / "mammaprint_compare" / "report_summary.csv",
                         usecols=["slide_name", "path"])
    paths = dict(zip(slides.slide_name, slides.path, strict=True))
    res: dict[str, tuple[float, float]] = {}
    todo: list[tuple[Path, float, float]] = []
    missing: list[str] = []
    for d in args.dirs:
        for tiff in sorted((args.root / d).glob("*.tiff")):
            name = tiff.stem
            if name not in res:
                if name not in paths:
                    missing.append(name)
                    continue
                try:
                    res[name] = slide_res(paths[name])
                except Exception as e:
                    print(f"[warn] no res for {name}: {e}", flush=True)
                    missing.append(name)
                    continue
            x, y = res[name]
            if x > 1:
                todo.append((tiff, x, y))
    print(f"{len(todo)} masks to restamp, {len(missing)} without slide res: {missing[:5]}",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, msg in enumerate(pool.map(fix_one, todo), 1):
            if i % 20 == 0 or (msg.startswith("fixed") and i < 3):
                print(f"[{i}/{len(todo)}] {msg}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
