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

import json
import os
import sys
from pathlib import Path

import pandas as pd
from fasthtml.common import (
    Div,
    Input,
    Label,
    Option,
    Script,
    Select,
    Span,
    Style,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

from report_fast import Report, SlideGrid, XopatSession  # noqa: E402
from report_fast.layer import colormap_layer  # noqa: E402

ROOT = Path(os.getenv("MAMMAPRINT_ROOT", "/mnt/projects/breast_cancer/brc_grading"))
OUT = Path(os.getenv("MAMMAPRINT_OUTPUT",
                     str(ROOT / "reports" / "mammaprint_agreement_v3.html")))
ATTN = ROOT / "masks" / "mammaprint"
LOO = ROOT / "masks" / "mammaprint-loo-lowers"
AGREE = ROOT / "masks" / "mammaprint-agree"

# Our attention TIFFs encode decades of attention above an equal share: 1 = an
# average tile, 128 = 10x, 255 = 100x or more -- so BRIGHTER grey = higher
# attention (a slide where nothing stands out stays pale). The 7 colour bins are
# even cuts of 0..255 (1x, 3x, 6x, 10x, 18x, 32x, 100x), 0 = no tile dropped
# (same ramp as the v2 report).
GRAY = ["#000000", "#2b2b2b", "#555555", "#808080",
        "#aaaaaa", "#d4d4d4", "#ffffff"]
GRAY_BREAKS = [0.0039, 0.1667, 0.3333, 0.5, 0.6667, 0.8333]

TOOLBAR_CSS = """
#mm-toolbar { display: flex; gap: 16px; flex-wrap: wrap; align-items: center;
              font-size: 0.92rem; margin: 10px 0 14px; }
#mm-toolbar label { display: inline-flex; gap: 5px; align-items: center; }
#mm-toolbar select { padding: 2px 6px; }
#mm-count { color: var(--rf-muted); }
"""

# Sorting/filtering the vendored library cannot do (it is a no-JS page): the
# card name is the join key, the numbers ride along as a literal JS map.
TOOLBAR_JS = """
const MM = %s;
(function () {
  const grid = document.querySelector('.rf-slide-grid');
  if (!grid) return;
  const cards = Array.from(grid.children);
  const nameOf = c => (c.querySelector('.rf-name') || {}).textContent || '';
  const meta = c => MM[nameOf(c).trim()];
  const f = { grade: -1, disc: false, hideBad: false, sort: 'absSurprise' };
  const KEYS = {
    absSurprise: m => -Math.abs(m.s),
    surpriseUp: m => m.s,
    surpriseDown: m => -m.s,
    trueUp: m => m.t,
    gradeUp: m => m.g,
    name: (m, n) => n
  };
  const origin = new Map(cards.map((c, i) => [c, i]));
  function sortCards() {
    const key = KEYS[f.sort];
    if (!key) return;
    cards.sort((a, b) => {
      const A = meta(a), B = meta(b);
      const ka = A ? key(A, nameOf(a)) : Infinity;
      const kb = B ? key(B, nameOf(b)) : Infinity;
      return ka < kb ? -1 : ka > kb ? 1 : origin.get(a) - origin.get(b);
    });
    cards.forEach(c => grid.appendChild(c));
  }
  function paint() {
    let shown = 0;
    for (const c of cards) {
      const m = meta(c);
      const ok = m
        && (f.grade < 0 || m.g === f.grade)
        && (!f.disc || Math.abs(m.s) >= 0.35)
        && (!f.hideBad || !m.bad);
      c.style.display = ok ? '' : 'none';
      if (ok) shown++;
      if (m) {
        const line = Array.from(c.querySelectorAll('.rf-note'))
          .find(p => p.textContent.startsWith('surprise'));
        if (line) line.style.color =
          m.s <= -0.35 ? '#c0392b' : m.s >= 0.35 ? '#188038' : '';
      }
    }
    const el = document.getElementById('mm-count');
    if (el) el.textContent = shown + ' / ' + cards.length + ' slides';
  }
  const on = (id, ev, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(ev, fn);
  };
  on('mm-grade', 'change', e => { f.grade = +e.target.value; paint(); });
  on('mm-disc', 'change', e => { f.disc = e.target.checked; paint(); });
  on('mm-hidebad', 'change', e => { f.hideBad = e.target.checked; paint(); });
  on('mm-sort', 'change', e => { f.sort = e.target.value; sortCards(); });
  sortCards();
  paint();
})();
"""


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
    # the job's pandas keeps "False"/"True" as strings; ~ on that is nonsense
    rows["contaminated"] = rows.contaminated.astype(str).str.lower().eq("true")
    # report_summary.csv kept only P(G3); the full posterior sits on the
    # attention stage's summary as a JSON list [p(G0)..p(G3)]
    summ = ROOT / "masks" / "mammaprint" / "summary.csv"
    if summ.exists():
        post = (pd.read_csv(summ)[["slide_name", "probs"]]
                .drop_duplicates("slide_name").set_index("slide_name").probs)
        for i in range(4):
            rows[f"p_g{i}"] = rows.slide_name.map(
                post.map(lambda j, i=i: json.loads(j)[i]))
    # grade call + how far each slide strays from its grade's usual index
    rows["our_grade"] = rows[[f"p_g{i}" for i in range(4)]].values.argmax(axis=1)
    rows["expected"] = rows.groupby("our_grade").label.transform("median")
    rows["surprise"] = rows.label - rows.expected
    meta = {
        r.slide_name: dict(g=int(r.our_grade), t=round(float(r.label), 3),
                           s=round(float(r.surprise), 3),
                           d=float(r.loo_rho), bad=bool(r.contaminated))
        for r in rows.itertuples()
    }
    rows = rows.sort_values("loo_rho", ascending=False, na_position="last")

    grid = SlideGrid()
    cards = 0
    for r in rows.itertuples():
        layers = layers_for(r.slide_name)
        session = XopatSession.from_slide(
            r.path, layers=layers, name=r.slide_name,
            # registered slide_protocols name on this deployment; without it v3
            # falls back to the default (IIP) and mrxs backgrounds never load
            protocol="wsi_service",
            params={"ui": {"toolBar": "essential"}})
        note = [
            f"grade {int(r.our_grade)} · P(G0-3) "
            + "/".join(f"{getattr(r, f'p_g{i}'):.2f}" for i in range(4)),
            f"MM index measured {r.label:+.2f} · predicted from image {r.prediction:+.2f}",
            f"surprise {r.surprise:+.2f} (typical for grade {r.expected:+.2f})",
            f"ρ drivers {r.loo_rho:+.2f} · ρ attention {r.attn_rho:.2f}",
        ]
        if bool(r.contaminated):
            note.append("RESAMPLE (contaminated)")
        grid.add(session, note=note)
        cards += 1

    clean = rows[~rows.contaminated]
    report = Report(
        theme="light",  # BASE_CSS dark-scope bug: its dark vars sit on a Div,
        # body keeps the light ones -> dark scheme paints #e6e9ed text on white
        title="MammaPrint risk drivers × our grade attention",
        subtitle=f"{cards} slides · xOpat v3",
        preamble=(
            "Orange = tiles that drive his MammaPrint prediction (removing them "
            "lowers his risk index). Grey = our model's attention: THE BRIGHTER "
            "a tile, the more our grade model looks at it (whitest = 100x an "
            "average tile). Green (drawn by default) = tiles both models rank in "
            "their own top decile. ρ = Spearman rank correlation, -1..+1, between "
            "how the two models ORDER this slide's tiles: -1 exact reversal, "
            "0 unrelated (coin flip), +1 identical order; |ρ| < 0.2 is no real "
            "agreement. Card numbers, line by line: our grade call and the full "
            "grade probabilities; his MammaPrint index as measured and as "
            "predicted from this image; surprise = measured index minus the "
            "median index of our slides of the same grade (negative = riskier "
            "than the grade suggests, printed red from |0.35| up, safer = "
            "green); and the two per-slide ρ. The toolbar sorts and filters "
            "the cards; by default the most discordant slides come first. "
            f"Cohort medians: ρ vs his risk drivers = {clean.loo_rho.median():+.2f} "
            f"(coin flip), ρ vs his raw attention = {clean.attn_rho.median():+.2f} "
            "- both models look at tumour, but disagree on what matters. If a "
            "card opens black or warns about a layer, note the slide: links are "
            "unverified from the build machine."
        ),
    )
    report.add(Style(TOOLBAR_CSS))
    report.add(Div(
        Label("grade ", Select(
            Option("all", value="-1"),
            *[Option(f"G{i}", value=str(i)) for i in range(4)], id="mm-grade")),
        Label(Input(type="checkbox", id="mm-disc"),
              "only discordant (|surprise| ≥ 0.35)"),
        Label(Input(type="checkbox", id="mm-hidebad"), "hide resampled"),
        Label("sort ", Select(
            Option("|surprise| desc", value="absSurprise"),
            Option("surprise asc (risk surprises first)", value="surpriseUp"),
            Option("surprise desc", value="surpriseDown"),
            Option("measured index asc", value="trueUp"),
            Option("grade asc", value="gradeUp"),
            Option("name", value="name"), id="mm-sort")),
        Span(id="mm-count"), id="mm-toolbar"))
    report.add(grid)
    report.add(Script(TOOLBAR_JS % json.dumps(meta)))
    path = report.write(OUT)
    sizes = pd.Series([len(layers_for(s)) for s in rows.slide_name])
    print(f"-> {path} ({path.stat().st_size / 1e6:.1f} MB), {cards} cards, "
          f"layers/slide: {sizes.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
