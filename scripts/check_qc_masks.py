"""Check QC mask files for OpenSlide/TiffFile readability on the cluster."""
from collections import Counter
from pathlib import Path

import tifffile
from openslide import OpenSlide


BASE = Path("/mnt/data/work/home/jovyan/k8s/rationai/namespaces/dp-quality-control/shared-volumes/qc-mask-storage")

MASKS = {
    "blur": BASE / "Piqe/blur_score_per_pixel",
    "folding": BASE / "FoldingFunction/folding_per_pixel",
    "residual": BASE / "ResidualArtifactsAndCoverage/artifacts_coverage",
}

stats = {name: Counter() for name in MASKS}
failures = {name: [] for name in MASKS}

for name, mask_dir in MASKS.items():
    if not mask_dir.exists():
        print(f"WARNING: {mask_dir} does not exist")
        continue

    tiff_files = sorted(mask_dir.glob("*.tiff"))
    print(f"\n{'='*60}")
    print(f"{name}: {len(tiff_files)} .tiff files in {mask_dir.name}")
    print(f"{'='*60}")

    for path in tiff_files:
        # Check tifffile
        try:
            with tifffile.TiffFile(str(path)) as tf:
                is_ome = tf.is_ome
                n_pages = len(tf.pages)
                n_levels = len(tf.series[0].levels) if tf.series else 0
                shape = tf.series[0].shape if tf.series else None
                stats[name]["tifffile_ok"] += 1
                if is_ome:
                    stats[name]["is_ome"] += 1
                if n_levels > 1:
                    stats[name]["has_pyramid"] += 1
        except Exception as e:
            stats[name]["tifffile_fail"] += 1
            failures[name].append((path.name, f"tifffile: {e}"))
            continue

        # Check OpenSlide
        try:
            with OpenSlide(str(path)) as slide:
                dims = slide.dimensions
                n_os_levels = len(slide.level_dimensions)
                stats[name]["openslide_ok"] += 1
        except Exception as e:
            stats[name]["openslide_fail"] += 1
            failures[name].append((path.name, f"openslide: {e}"))

    print(f"  tifffile OK:    {stats[name]['tifffile_ok']}/{len(tiff_files)}")
    print(f"  is OME:         {stats[name]['is_ome']}/{len(tiff_files)}")
    print(f"  has pyramid:    {stats[name]['has_pyramid']}/{len(tiff_files)}")
    print(f"  OpenSlide OK:   {stats[name]['openslide_ok']}/{len(tiff_files)}")
    print(f"  OpenSlide FAIL: {stats[name]['openslide_fail']}/{len(tiff_files)}")

    if failures[name]:
        print("\n  First 5 failures:")
        for fname, err in failures[name][:5]:
            print(f"    {fname}: {err}")
        if len(failures[name]) > 5:
            print(f"    ... and {len(failures[name]) - 5} more")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
any_ok = False
for name in MASKS:
    total = stats[name].get("tifffile_ok", 0) + stats[name].get("openslide_fail", 0)
    os_ok = stats[name]["openslide_ok"]
    os_fail = stats[name]["openslide_fail"]
    if os_ok > 0:
        any_ok = True
    print(f"  {name}: {os_ok} OpenSlide-readable, {os_fail} OpenSlide-unreadable")
    if stats[name]["is_ome"] > 0:
        print(f"    ({stats[name]['is_ome']} are OME-TIFF, would work with .ome.tiff extension)")

if not any_ok:
    print("\n  None are OpenSlide-readable. Fix: read QC masks with TiffFile, not OpenSlide.")
