"""MammaPrint agreement report as native xOpat **v3** sessions (ReportFast).

Why this exists next to mammaprint_report.py: the old page emits
``/xopat/redirect.php#...`` links, and that mount serves the **v2** viewer --
redirect.php was deleted upstream and v2 parses our sessions through its
legacy shim, which is what flags our layers as "possibly faulty". ReportFast
emits ``/v3/#<session json>`` (report_fast.xopat.DEFAULT_BASE_URL), which the
v3 viewer parses natively. Same data, same three layers as the v2 report:

  1. MammaPrint risk drivers (his signed LOO, orange, class 1 only)
  2. our grade attention (grey 7-step ramp, log-encoded 0..255)
  3. agreement (green where both rank a tile top-decile) -- drawn by default

Run inside the cluster report job::

    REPORT_JOB_NAME=brc-grading-mmatttn-v3 REPORT_SCRIPT=mammaprint_report_fast.py \
      REPORT_DATA=1 uv run --no-sync python scripts/run_report.py

The ``report_fast`` package is vendored under vendor/ (the pod clones this repo
anonymously and cannot reach the private ReportFast repo).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

from report_fast import Report, SlideGrid, XopatSession  # noqa: E402
from report_fast.layer import colormap_layer  # noqa: E402

ROOT = Path(os.getenv("MAMMAPRINT_ROOT", "/mnt/projects/breast_cancer/brc_grading"))
OUT = Path(os.getenv("MAMMAPRINT_OUTPUT",
                     str(ROOT / "reports" / "mammaprint_agreement_v3.html")))
ATTN = ROOT / "masks" / "mammaprint"
LOO = ROOT / "masks" / "mammaprint-loo-lowers"
AGREE = ROOT / "masks" / "mammaprint-agree"

# Our attention TIFFs encode log1p(6w)/log1p(6) scaled to 0..255, so the colour
# breaks are 1/6..5/6 of that scale, with 0 dropped (same ramp as the v2 report).
GRAY = ["#000000", "#2b2b2b", "#555555", "#808080",
        "#aaaaaa", "#d4d4d4", "#ffffff"]
GRAY_BREAKS = [0.0039, 0.1667, 0.3333, 0.5, 0.6667, 0.8333]


def layers_for(stem: str) -> list[dict]:
    """The three overlays for one slide, skipping any TIFF not on disk."""
    out: list[dict] = []
    loo = LOO / f"{stem}.tiff"
    if loo.exists():
        out.append(colormap_layer(
            str(loo), name="MammaPrint risk drivers (LOO)", classes=2,
            palette=["#000000", "#ff8c00"], breaks=[0.5], mask=[0, 1],
            opacity=0.55))
    attn = ATTN / f"{stem}.tiff"
    if attn.exists():
        out.append(colormap_layer(
            str(attn), name="our grade attention", classes=7, palette=GRAY,
            breaks=GRAY_BREAKS, opacity=0.45))
    agree = AGREE / f"{stem}.tiff"
    if agree.exists():
        lay = colormap_layer(
            str(agree), name="agreement (both top-decile)", classes=2,
            palette=["#000000", "#00e05a"], breaks=[0.5], mask=[0, 1],
            opacity=0.7)
        lay["visible"] = True  # the one layer the reader should see first
        out.append(lay)
    for lay in out:
        lay.setdefault("visible", False)
    return out


def main() -> None:
    rows = pd.read_csv(ROOT / "mammaprint_compare" / "report_summary.csv")
    rows = rows.sort_values("loo_rho", ascending=False, na_position="last")

    grid = SlideGrid()
    cards = 0
    for r in rows.itertuples():
        layers = layers_for(r.slide_name)
        session = XopatSession.from_slide(
            r.path, layers=layers, name=r.slide_name,
            params={"ui": {"toolBar": "essential"}})
        note = (f"P(G3) {r.p_g3:.2f} · ρ drivers {r.loo_rho:+.2f} · "
                f"ρ his-attn {r.attn_rho:.2f}")
        if bool(r.contaminated):
            note += " · RESAMPLE (contaminated)"
        grid.add(session, note=note)
        cards += 1

    clean = rows[~rows.contaminated]
    report = Report(
        title="MammaPrint risk drivers × our grade attention",
        subtitle=f"{cards} slides · xOpat v3",
        preamble=(
            "Orange = tiles that drive his MammaPrint prediction (signed LOO). "
            "Grey = our attention (darker = higher). Green (drawn by default) = "
            "tiles both models put in their top decile. Tile-level agreement is "
            f"absent by any measure: median ρ(ours, his drivers) = "
            f"{clean.loo_rho.median():+.2f}, while both models' raw attention "
            f"correlate at {clean.attn_rho.median():+.2f} — they rank tumour "
            "similarly but disagree on what matters. If a card opens black or "
            "warns about a layer, note the slide: this page was rebuilt for the "
            "v3 viewer and the links are unverified from the build machine."
        ),
    )
    report.add(grid)
    path = report.write(OUT)
    sizes = pd.Series([len(layers_for(s)) for s in rows.slide_name])
    print(f"-> {path} ({path.stat().st_size / 1e6:.1f} MB), {cards} cards, "
          f"layers/slide: {sizes.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
