"""Build the xOpat overlay report: heatmaps + the per-slide numbers, in one HTML page.

Replaces the `report` (Hydra) gallery, which drew the masks with hard colour
thresholds and could not show any per-slide numbers. This version puts
`summary.csv` next to every thumbnail (truth, prediction with its probability,
tiles, how focused the bag was) and links the WSI with the mask as a layer:
`--shader grey` (default) for a translucent black -> white brightness ramp at
`--opacity`, `--shader heatmap` for xOpat's coloured heatmap.

Run it as a job (the PVC and MLflow are only reachable in the cluster)::

    REPORT_JOB_NAME=brc-grading-report-heatmap REPORT_SCRIPT=xopat_report.py \
      REPORT_ARGS='--split test' uv run --no-sync python scripts/run_report.py
    kubectl logs -n rationai-jobs-ns job/brc-grading-report-heatmap-jiri-buchta

Local smoke test without the PVC (writes /tmp/xopat-report-selftest.html)::

    uv run --no-sync python scripts/xopat_report.py --self-test

Needs `rationai-reporting` (private GitHub repo, no key in pods): copy of the
`init` branch sits at ``<root>/vendor/rationai-reporting`` and the job installs it
with uv, which also pulls `python-fasthtml`.
shortcut: vendored snapshot of a branch that is still moving; re-copy when
upstream changes. Do not PYTHONPATH it instead -- `rationai` in site-packages is
a regular package (it has an `__init__.py`), so a loose `rationai/` dir on the
path is shadowed, not merged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("/mnt/projects/breast_cancer/brc_grading")
# Must be the experiment the MIL runs live in (82, 40 runs with test/kappa).
# "Breast Cancer Grading" (37) is a metric-less SDK-boilerplate experiment, and the
# vendored `report` app's 107 sits on a third server entirely.
EXPERIMENT = "BRC Grading"
MLFLOW_UI = "http://mlflow-s3.rationai-mlflow/#/experiments/82"
CLASSES = 4

TUMOUR_CAVEAT = (
    "The overlay is where the grade head *looked*, not where the tumour is: the "
    "tumour-vs-stroma contrast measured 0.48 AUC (below chance, REPORT.md §10). "
)

# One line per mask value scale (see attention_masks.py --scale), so the page never
# explains a ramp the TIFFs do not use.
BRIGHTNESS = {
    "log": "Brightness is one absolute scale for every slide: the mask stores 0 = no tile, "
           "1 = an equal share of that slide's attention (1/N), 128 = 10x that share, "
           "255 = 100x or more.",
    "pct": "Brightness is per slide: black is where the model does not look (no tile, or under "
           "0.2% of the peak), white is that slide's brightest tile, linear in attention in "
           "between. Compare slides by `eff_tiles`, not by colour.",
    "sqrt": "Brightness is per slide, square-root compressed: black is where the model does not "
            "look, white is that slide's brightest tile, an average tile around 20% grey. "
            "Compare slides by `eff_tiles`, not by colour.",
}

CAVEAT = TUMOUR_CAVEAT + BRIGHTNESS["log"] + (
    " `eff_tiles` (= 1/sum(w²)) is how many tiles the bag effectively spent its "
    "attention on: ~N means it looked everywhere, a few hundred means it committed to a spot."
)

# xOpat `colormap` shader: a black -> white brightness ramp over the whole slide, drawn
# at a partial opacity so the tissue underneath stays visible. Value 0 (no tile) is the
# black end, NOT transparent -- the ask was "black where it does not look, up to white at
# what it looks at", and a transparent floor just leaves the slide showing through. Only
# the keys Tile-Morph's reviewed LDA config is known to work with (breaks = fraction of
# 255, one colour per bin).
GREY_RAMP = ["#000000", "#333333", "#555555", "#777777", "#999999", "#bbbbbb", "#ffffff"]


def grey_shader(opacity: float) -> dict:
    """Translucent black -> white brightness overlay for the mask layer."""
    n = len(GREY_RAMP) - 1                       # bins above the black floor
    return {
        "type": "colormap",
        "params": {
            "use_channel0": "r",
            "opacity": opacity,
            "color": {
                "type": "custom_colormap",
                "default": GREY_RAMP,
                "steps": len(GREY_RAMP),
                "mode": "qualitative",
                "interactive": True,
                "title": "Attention:",
                "continuous": False,
            },
            "threshold": {
                "breaks": [1 / 255] + [i / n for i in range(1, n)],
                "mask": [1] * len(GREY_RAMP),
                "interactive": False,
                "inverted": True,
                "maskOnly": True,
                "toggleMask": True,
                "title": "Attention:",
                "min": 0,
                "max": 1,
                "minGap": 0.05,
                "step": None,
                "pips": {"mode": "count", "values": len(GREY_RAMP), "density": len(GREY_RAMP),
                         "stepped": True},
            },
        },
    }


def load_rows(root: Path, mask_dir: Path) -> pd.DataFrame:
    """One row per slide that has BOTH a mask on disk and a row in summary.csv.

    Ordering is left to `sort_rows`; missing masks are dropped on purpose -- a card
    whose overlay 404s is worse than no card.
    """
    summary = pd.read_csv(mask_dir / "summary.csv")
    have = {p.stem for p in sorted(mask_dir.glob("*.tiff"))}
    rows = summary[summary.slide_name.isin(have)].copy()
    if rows.empty:
        raise SystemExit(f"no slide has both a mask in {mask_dir} and a summary.csv row")
    rows["probs"] = rows.probs.map(json.loads)
    for c in range(CLASSES):
        rows[f"p{c}"] = rows.probs.str[c]
    mapping = pd.read_csv(root / "data_mapping.csv")
    mapping["slide_name"] = mapping.path.str.split("/").str[-2]
    rows = rows.merge(mapping[["slide_name", "path"]], on="slide_name", how="left")
    missing = int(rows.path.isna().sum())
    if missing:
        print(f"warning: {missing} slides have no path in data_mapping.csv, dropped")
        rows = rows[rows.path.notna()]
    return rows.reset_index(drop=True)


def sort_rows(rows: pd.DataFrame, how: str) -> pd.DataFrame:
    """Review order: the interesting slides first."""
    if how == "name":
        return rows.sort_values("slide_name")
    if how == "eff-desc":
        return rows.sort_values(["eff_tiles", "slide_name"], ascending=[False, True])
    # wrong-first: misclassified slides, most-focused ones of those first
    return rows.sort_values(["correct", "eff_tiles"], ascending=[True, False])


def global_metrics(rows: pd.DataFrame) -> dict[str, list]:
    """Cohort-level numbers for the table under the header."""
    truth, pred = rows.truth.to_numpy(), rows.pred.to_numpy()
    per_class = {}
    for c in range(CLASSES):
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (tp + fp) and (tp + fn) and (precision + recall)
            else float("nan")
        )
        per_class[f"class {c} — precision / recall / F1"] = [f"{precision:.2f} / {recall:.2f} / {f1:.2f}"]
        per_class[f"class {c} — slides (true)"] = [int((truth == c).sum())]
    return {
        "slides with masks": [len(rows)],
        "accuracy": [float(rows.correct.mean())],
        "mean eff_tiles (attention spread, tiles)": [round(float(rows.eff_tiles.mean()), 1)],
        "median eff_tiles": [round(float(rows.eff_tiles.median()), 1)],
        "mean tiles per slide": [round(float(rows.n_tiles.mean()), 1)],
        **per_class,
    }


def card_metrics(row: pd.Series) -> dict[str, list]:
    """Per-slide table: what the model said, how confidently, where it looked."""
    probs = row.probs
    return {
        "prediction": [f"G{int(row.pred)}" + ("" if row.correct else " ✗")],
        "probabilities (G0…G3)": [" / ".join(f"{p:.3f}" for p in probs)],
        "top probability": [round(float(max(probs)), 4)],
        "tiles": [int(row.n_tiles)],
        "eff_tiles": [round(float(row.eff_tiles), 1)],
        "max tile weight": [round(float(row.max_weight), 5)],
    }


def build_page(rows: pd.DataFrame, mask_dir: Path, title: str, *, shader: dict | None = None,
               scale: str = "log") -> object:
    from fasthtml.common import P
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

    slides = attach_layers(
        slides_from_paths(rows.path.tolist(), layer_name="WSI (Mirax)"),
        layers_from_disk(mask_dir, globs=("*.tiff",), layer_name="Attention",
                         shader_conf=shader or {}),
    )
    by_name = {s["name"]: s for s in slides}
    cards = []
    for row in rows.itertuples():
        cards.append(
            slide_card(
                {**by_name[Path(row.path).stem], "ground_truth": f"G{int(row.truth)}"},
                metrics=card_metrics(row),
            )
        )
    runs = [{"name": "MIL runs (MLflow)", "href": MLFLOW_UI}]
    return Report(
        "Overlay report",
        ReportHeader(title, RunList(runs)),
        SummaryTable(global_metrics(rows), title="Cohort and agreement"),
        ReportSection("How to read the overlay",
                      P(TUMOUR_CAVEAT + BRIGHTNESS[scale] +
                        " `eff_tiles` (= 1/sum(w²)) is how many tiles the bag effectively spent "
                        "its attention on: ~N means it looked everywhere, a few hundred means it "
                        "committed to a spot.")),
        ReportSection(f"Slides ({len(cards)}) — worst-agreement first", ReportGrid(*cards)),
        title=title,
    )


def upload(html_path: Path, summary_csv: Path, split: str, n_slides: int, params: dict) -> str:
    """Log the page (and the `summary.csv` it was built from) to MLflow, next to the runs."""
    import mlflow

    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    with mlflow.start_run(run_name=f"MIL overlay report {html_path.stem} {stamp}") as run:
        mlflow.log_artifact(str(html_path))
        if summary_csv.exists():
            mlflow.log_artifact(str(summary_csv))
        mlflow.log_params(params)
        mlflow.log_metric("slides", n_slides)
        return run.info.run_id


def ensure_reporting(root: Path) -> None:
    """Install `rationai-reporting` from the PVC copy if the venv lacks it.

    Upstream is a private GitHub repo and pods have no key for it, so the job gets
    the vendored copy (which also pulls `python-fasthtml` from the normal index).
    """
    try:
        import rationai.reporting  # noqa: F401

        return
    except ImportError:
        pass
    src = root / "vendor" / "rationai-reporting"
    if not src.is_dir():
        raise SystemExit(f"rationai.reporting not installed and no vendored copy at {src}")
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, str(src)],
        check=True,
    )


def self_test() -> int:
    """Render a page from two fake rows; checks the pieces the user asked for."""
    ensure_reporting(DEFAULT_ROOT)
    root = Path("/tmp/xopat-selftest")
    mask_dir = root / "masks" / "test"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for name in ("slideA-1-T", "slideB-2-N"):
        (mask_dir / f"{name}.tiff").touch()
    pd.DataFrame(
        {
            "slide_name": ["slideA-1-T", "slideB-2-N"],
            "case": ["slideA", "slideB"],
            "truth": [0, 2],
            "pred": [0, 3],
            "correct": [1, 0],
            "n_tiles": [4000, 900],
            "max_weight": [0.01, 0.4],
            "eff_tiles": [1500.0, 12.0],
            "probs": ['[0.9,0.05,0.03,0.02]', '[0.01,0.04,0.35,0.6]'],
        }
    ).to_csv(mask_dir / "summary.csv", index=False)
    (root / "data_mapping.csv").write_text(
        "path,grade,case\n"
        "/mnt/bioptic_tree/2020/01/001/slideA-1-T/slideA-1-T.mrxs,0,slideA\n"
        "/mnt/bioptic_tree/2020/01/002/slideB-2-N/slideB-2-N.mrxs,2,slideB\n"
    )

    rows = sort_rows(load_rows(root, mask_dir), "wrong-first")
    assert rows.slide_name.tolist() == ["slideB-2-N", "slideA-1-T"], rows.slide_name.tolist()
    page = build_page(rows, mask_dir, "selftest")
    grey = build_page(rows, mask_dir, "selftest-grey", shader=grey_shader(0.5), scale="sqrt")
    from fasthtml.common import to_xml

    html = to_xml(page)
    assert "slideB-2-N" in html and "slideA-1-T" in html
    assert "heatmap" in html, "overlay must be a heatmap layer"
    ghtml = to_xml(grey)
    assert "colormap" in ghtml, "grey shader must reach the link"
    # Decode a real link and check what xOpat is actually handed: the shader JSON is
    # percent-encoded in the href, so asserting on colour strings proves nothing.
    import json
    import re
    import urllib.parse

    link = json.loads(urllib.parse.unquote(
        re.search(r"xopat/redirect\.php#([^\"']+)", ghtml).group(1)))
    shader_conf = link["visualizations"][0]["shaders"]["layer_shader_0"]["params"]
    assert shader_conf["opacity"] == 0.5, shader_conf["opacity"]
    assert shader_conf["color"]["default"][0] == "#000000", "value 0 must be black, not transparent"
    assert shader_conf["color"]["default"][-1] == "#ffffff", "the peak must be white"
    assert set(shader_conf["threshold"]["mask"]) == {1}, "no bin may be hidden"
    assert "black is where the model does not look" in ghtml, "legend must match the shader"
    assert "sqrt" in ghtml.lower() or "brightest tile" in ghtml, "legend must match the scale"
    assert "bioptic_tree/2020/01/002/slideB-2-N/slideB-2-N.mrxs" in html  # mount-relative
    assert "masks/test/slideB-2-N.tiff" in html
    assert "0.350 / 0.600" in html, "per-slide probabilities from summary.csv"
    assert "G3 ✗" in html and "G2" in html
    assert "accuracy" in html and "eff_tiles" in html
    out = Path("/tmp/xopat-report-selftest.html")
    out.write_text(html, encoding="utf-8")
    print(f"selftest ok: {len(rows)} rows, {len(html) // 1024} KB -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--masks-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sort", choices=["wrong-first", "eff-desc", "name"], default="wrong-first")
    ap.add_argument("--shader", choices=["grey", "heatmap"], default="grey",
                    help="grey = translucent black->white ramp; heatmap = xOpat's built-in colours")
    ap.add_argument("--opacity", type=float, default=0.5, help="layer opacity for --shader grey")
    ap.add_argument("--scale", choices=["log", "pct", "sqrt"], default="log",
                    help="how the masks in --masks-dir were encoded; only changes the legend")
    ap.add_argument("--no-mlflow", action="store_true", help="Write the HTML only.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    ensure_reporting(args.root)
    mask_dir = args.masks_dir or args.root / "masks" / args.split
    rows = sort_rows(load_rows(args.root, mask_dir), args.sort)
    if args.limit:
        rows = rows.head(args.limit)
    title = f"Breast cancer Nottingham grade — MIL attention ({args.split}, {len(rows)} slides)"
    shader = grey_shader(args.opacity) if args.shader == "grey" else None
    page = build_page(rows, mask_dir, title, shader=shader, scale=args.scale)

    from fasthtml.common import to_xml

    out = args.out or args.root / "reports" / (
        f"{args.split}_{args.shader}.html" if args.scale == "log"
        else f"{args.split}_{args.shader}{args.scale}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_xml(page), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(rows)} slides)")
    print(f"accuracy {rows.correct.mean():.4f} on these {len(rows)} slides")
    if not args.no_mlflow:
        run_id = upload(out, mask_dir / "summary.csv", args.split, len(rows), {
            "shader": args.shader, "opacity": args.opacity, "scale": args.scale,
            "sort": args.sort, "masks_dir": str(mask_dir)})
        print(f"MLflow run {run_id} in experiment '{EXPERIMENT}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
