"""xOpat page: MammaPrint index drivers vs grade-model attention, stacked layers.

Three toggleable layers per slide (same geometry, different grids -- both masks
carry their own mpp so the viewer aligns them):
  1. "MammaPrint index drivers"   -- HIS leave-one-out mask (tiles whose removal
     LOWERS the predicted index, i.e. tiles actively pushing risk up), taken
     verbatim from his MLflow run by compare_mammaprint_attention.py.
  2. "Grade MIL attention"        -- OUR fold-averaged attention, grey ramp.
  3. "Agreement"                  -- tiles BOTH models call top-10%: the only
     layer that is scale-free across models; eyeball THIS one, not colour vs
     colour (the two heat scales mean nothing to each other).

Per-slide agreement numbers (Spearman rhos from the tile-level join) sit on each
card; run compare_mammaprint_attention.py first -- it writes
mammaprint_compare/report_summary.csv and the mask dirs this page links.

Job::

    REPORT_JOB_NAME=brc-grading-mmattn-report REPORT_SCRIPT=mammaprint_report.py \
      REPORT_CPU=2 REPORT_MEMORY=8Gi uv run --no-sync python scripts/run_report.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
import xopat_report as xr


DEFAULT_ROOT = xr.DEFAULT_ROOT


def solid_shader(color: str, opacity: float, title: str) -> dict:
    """Hide zeros, flat colour above -- for binary/magnitude attribution masks."""
    return {
        "type": "colormap",
        "params": {
            "use_channel0": "r",
            "opacity": opacity,
            "color": {"type": "custom_colormap", "default": ["", color], "steps": 2,
                      "mode": "qualitative", "interactive": False,
                      "title": title, "continuous": False},
            "threshold": {"breaks": [1 / 255, 1.0], "mask": [0, 1], "interactive": False,
                          "inverted": True, "maskOnly": True, "toggleMask": True,
                          "title": title, "min": 0, "max": 1, "minGap": 0.05, "step": None},
        },
    }


def load_rows(compare_dir: Path, attn_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(compare_dir / "report_summary.csv")
    have = {p.stem for p in attn_dir.glob("*.tiff")}
    rows = rows[rows["slide_name"].isin(have) & rows["path"].notna()].reset_index(drop=True)
    if rows.empty:
        raise SystemExit("no slides have both report_summary.csv and an attention TIFF")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sort", choices=["rho-desc", "rho-asc", "name"], default="rho-desc")
    ap.add_argument("--clean-only", action="store_true", help="drop the 4 contaminated slides")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)

    xr.ensure_reporting(args.root)
    from fasthtml.common import P, to_xml
    from rationai.reporting import (
        Report,
        ReportGrid,
        ReportHeader,
        ReportSection,
        RunList,
        SummaryTable,
        attach_layers,
        layers_from_disk,
        slide_card,
        slides_from_paths,
    )

    compare_dir = args.root / "mammaprint_compare"
    attn_dir = args.root / "masks" / "mammaprint"
    rows = load_rows(compare_dir, attn_dir)
    if args.clean_only:
        rows = rows[rows["contaminated"] == 0].reset_index(drop=True)
    rows = rows.sort_values(
        "loo_rho" if args.sort != "name" else "slide_name",
        ascending=args.sort == "rho-asc", na_position="last").reset_index(drop=True)
    if args.limit:
        rows = rows.head(args.limit)

    layer_dirs = [
        (args.root / "masks" / "mammaprint-loo-lowers",
         "MammaPrint index drivers (his LOO)", solid_shader("#ff8c00", 0.45, "Index drivers:")),
        (attn_dir, "Grade MIL attention (ours)", xr.grey_shader(0.45)),
        (args.root / "masks" / "mammaprint-agree",
         "Agreement: both top 10%", solid_shader("#00e05a", 0.7, "Both models:")),
    ]
    layers = [l for d, name, sh in layer_dirs if d.is_dir()
              for l in layers_from_disk(d, globs=("*.tiff",), layer_name=name, shader_conf=sh)]
    slides = attach_layers(slides_from_paths(rows["path"].tolist(), layer_name="WSI (Mirax)"), layers)
    by_name = {s["name"]: s for s in slides}

    def card(row) -> dict:
        return {
            "MammaPrint index (pred / true)": [f"{row.prediction:+.3f} / {row.label:+.3f}"],
            "rho attention vs index drivers": [round(float(row.loo_rho), 3)],
            "rho vs |LOO| / his attention": [f"{row.absloo_rho:.3f} / {row.attn_rho:.3f}"],
            "tile coverage (his -> ours)": [round(float(row.coverage), 3)],
            "eff_tiles (ours)": [round(float(row.eff_tiles), 1)],
            "our P(G3)": [round(float(row.p_g3), 3)],
        }

    cards = [slide_card({**by_name[Path(r.path).stem],
                         "ground_truth": "CONTAMINATED" if r.contaminated else ""},
                        metrics=card(r))
             for r in rows.itertuples()]

    clean = rows[rows["contaminated"] == 0]
    glob = {
        "slides": [len(rows)],
        "median rho (attention vs index drivers, clean)": [round(float(clean["loo_rho"].median()), 3)],
        "share of slides rho > 0": [round(float((clean["loo_rho"] > 0).mean()), 2)],
        "median rho vs |LOO|": [round(float(clean["absloo_rho"].median()), 3)],
        "median rho vs his attention": [round(float(clean["attn_rho"].median()), 3)],
        "mean tile coverage": [round(float(clean["coverage"].mean()), 3)],
        "spearman(index pred, our P(G3)) over slides": [
            round(float(clean[["prediction", "p_g3"]].corr(method="spearman").iloc[0, 1]), 3)],
    }
    page = Report(
        "MammaPrint x grade MIL overlay",
        ReportHeader(f"MammaPrint index drivers vs grade attention ({len(rows)} slides)",
                     RunList([{"name": "MIL runs (MLflow)", "href": xr.MLFLOW_UI}])),
        SummaryTable(glob, title="Tile-level agreement (Spearman on matched tiles)"),
        ReportSection("How to read this", P(
            "Orange = tiles whose removal LOWERS his predicted MammaPrint index (they drive "
            "risk up; his attribution, level-2 grid). White = where our grade MIL put "
            "attention (1 = an equal share, 128 = 10x, 255 = 100x). Green = tiles both models "
            "rank in their slide's top 10% -- the two heat scales are not comparable, so "
            "judge agreement by the green layer and the per-card rho, not by colour. ",
            xr.TUMOUR_CAVEAT)),
        ReportSection(f"Slides ({len(cards)})", ReportGrid(*cards)),
        title="MammaPrint x grade MIL overlay",
    )

    out = args.out or args.root / "reports" / "mammaprint_agreement.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_xml(page), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(rows)} slides)")
    if not args.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        mlflow.set_experiment(xr.EXPERIMENT)
        with mlflow.start_run(
                run_name=f"MammaPrint agreement report {datetime.now(UTC):%Y%m%d-%H%M}") as run:
            mlflow.log_artifact(str(out))
            mlflow.log_artifact(str(compare_dir / "report_summary.csv"))
            mlflow.log_metric("slides", len(rows))
        print(f"MLflow run {run.info.run_id} in experiment '{xr.EXPERIMENT}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
