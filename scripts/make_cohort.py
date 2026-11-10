"""Build the BRC grading cohort from the clinical records CSV + the WSI tree.

Reads the semicolon-separated records CSV (one row per case) and the on-disk
slide tree, then emits the two CSVs the pipeline consumes:

  * ``data_mapping.csv`` — feeds ``preprocessing/tiling.py``. Columns:
    ``path`` (WSI path *without* extension; tiling appends ``.mrxs``),
    ``grade``, ``case``.
  * ``labels.csv`` — feeds ``ml.data.datasets.GradingDataset``. Columns:
    ``slide_id``, ``grade``, ``split``.

Grade mapping (per README, class 0 = "no tumour epithelium (N)"):
  * ``…-T`` slides -> the case's Nottingham grade (1/2/3)
  * ``…-N`` slides -> grade 0 (no tumour)

Filtering (BRC-A-5/14/15): drop grade 9 (not assessable) and neoadjuvant
cases (``neoajuvant y/n == y``).

Splitting (BRC-A-16): stratified by grade, grouped by case — every slide of a
case lands in the same split. Fractions default to 70/15/15 (same as the demo).

Usage::

    uv run python scripts/make_cohort.py \
        --records /path/to/records.csv \
        --tree /mnt/bioptic_tree \
        --out /path/to/cohort
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


# Column names in the records CSV (semicolon-separated).
RECORD_COL = "record"
GRADE_COL = "grade"
NEO_COL = "neoajuvant y/n"
MORPHOLOGY_COL = "morphology"

# BRC-A-11/15: the grading task is restricted to morphology code 8500.
MORPHOLOGY_8500 = "8500"

# BRC-A-20: rare/unsuitable slides excluded entirely before dataset creation.
# Case-level entries drop the whole record; slide-level entries drop one slide.
EXCLUDED_CASES = {
    "2020_02716", "2020_04882", "2020_00857", "2020_01125", "2023_02823",
    "2022_09552", "2024_00115", "2020_00781", "2020_02500",
}
EXCLUDED_SLIDES = {
    "2020_00172-03-N", "2021_06248-10-T2", "2022_05492-1-N", "2022_09069-1-N",
    "2021_08684-01-N", "2021_06115-02-N", "2022_07918-2-N", "2023_02864-2-T",
    "2023_02503-3-T", "2021_03825-02-T",
}

# Slide-name suffixes. Tumour slides end in ``-T`` / ``-T<n>`` and carry the
# case grade; normal slides end in ``-N`` / ``-N<n>`` and are class 0 (no
# tumour). IHC-marker slides (``-M``, ``-Ki67``, ``-ER``, ``-PR``, ...) are a
# different stain, not morphology, and are excluded from the grading cohort.
import re


TUMOUR_RE = re.compile(r"-T\d?$")
NORMAL_RE = re.compile(r"-N\d?$")

# Split fractions (train/val/test), matching configs/data/demo.yaml.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

VALID_GRADES = {1, 2, 3}  # 0 is assigned to N slides; 9 is filtered out.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True, help="Path to the records CSV.")
    p.add_argument("--tree", required=True, help="Root of the WSI tree (e.g. /mnt/bioptic_tree).")
    p.add_argument("--out", required=True, help="Output directory for the two CSVs.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the split.")
    p.add_argument(
        "--ext", default=".mrxs",
        help="Slide file extension present on disk (default .mrxs).",
    )
    p.add_argument(
        "--require-embeddings", default=None, metavar="DIR",
        help="Drop slides with no <slide_id>.parquet in DIR (slides excluded from "
        "embedding, e.g. for missing QC masks, cannot be trained on).",
    )
    return p.parse_args()


def read_records(path: Path) -> dict[str, int]:
    """Return {record: grade} for cases that pass the BRC filters.

    Opens with ``utf-8-sig`` so an Excel-written BOM doesn't turn the first
    header into ``\ufeffrecord``. Prints skip reasons so an empty result is
    diagnosable.
    """
    cases: dict[str, int] = {}
    skips = {
        "no_record": 0, "excluded_case": 0, "morphology": 0,
        "neoadjuvant": 0, "bad_grade": 0, "grade_9": 0,
    }
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        fields = reader.fieldnames or []
        for row in reader:
            record = (row.get(RECORD_COL) or "").strip()
            if not record:
                skips["no_record"] += 1
                continue
            # BRC-A-20: rare/unsuitable cases excluded entirely.
            if record in EXCLUDED_CASES:
                skips["excluded_case"] += 1
                continue
            # BRC-A-11/15: morphology must be 8500 (normalize "8 500" -> "8500").
            morphology = (row.get(MORPHOLOGY_COL) or "").replace(" ", "").strip()
            if morphology != MORPHOLOGY_8500:
                skips["morphology"] += 1
                continue
            # Neoadjuvant cases are excluded (BRC-A-14/15).
            if (row.get(NEO_COL) or "").strip().lower() == "y":
                skips["neoadjuvant"] += 1
                continue
            try:
                grade = int((row.get(GRADE_COL) or "").strip())
            except ValueError:
                skips["bad_grade"] += 1
                continue
            # Grade 9 = not assessable (BRC-A-5).
            if grade not in VALID_GRADES:
                skips["grade_9"] += 1
                continue
            cases[record] = grade

    if not cases:
        print(f"  columns found: {fields}", file=sys.stderr)
        print(f"  skips: {skips}", file=sys.stderr)
        if RECORD_COL not in fields:
            print(f"  ERROR: column {RECORD_COL!r} not found — check the header.", file=sys.stderr)
    return cases


def find_slides(tree: Path, record: str, ext: str) -> list[Path]:
    """Locate the case's slide files under the tree.

    Layout: ``<tree>/<year>/<month>/<serial>/<slide_id>/<slide_id>.mrxs`` where
    the record ``2019_08728`` (year + 5-digit serial) maps to ``2019/08/728``
    (serial splits as first-2-digits month + last-3-digits serial), and each
    slide is a subdirectory named ``<record>-NN-T`` / ``<record>-NN-N`` holding
    its ``.mrxs`` file (plus a ``.mrxs.tiff`` and ``label.png``).
    """
    year, serial = record.split("_")
    month, serial_part = serial[:2], serial[2:]
    case_dir = tree / year / month / serial_part
    if not case_dir.is_dir():
        return []
    slides: list[Path] = []
    for sub in sorted(case_dir.iterdir()):
        if not sub.is_dir():
            continue
        mrxs = sub / (sub.name + ext)
        if mrxs.is_file():
            slides.append(mrxs)
    return slides


def main() -> None:
    args = parse_args()
    tree = Path(args.tree)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = read_records(Path(args.records))
    print(f"Cases passing filters: {len(cases)}")

    rng = random.Random(args.seed)

    # 1. Discover slides per case and assign grades (T -> case grade, N -> 0).
    slides: list[dict] = []  # {case, grade, slide_id, path}
    missing: list[str] = []
    skipped: dict[str, int] = defaultdict(int)  # suffix -> count
    for record, grade in sorted(cases.items()):
        found = find_slides(tree, record, args.ext)
        if not found:
            missing.append(record)
            continue
        for slide in found:
            name = slide.stem  # e.g. 2019_08728-01-T (matches <slide_id>.parquet)
            # BRC-A-20: drop specific rare/unsuitable slides.
            if name in EXCLUDED_SLIDES:
                continue
            if TUMOUR_RE.search(name):
                slide_grade = grade
            elif NORMAL_RE.search(name):
                slide_grade = 0
            else:
                # Record the trailing token so the breakdown is diagnosable.
                tail = name.rsplit("-", 1)[-1] if "-" in name else "<none>"
                skipped[tail] += 1
                continue
            slides.append(
                {
                    "case": record,
                    "grade": slide_grade,
                    "slide_id": name,
                    "path": str(slide),  # full path with .mrxs (register_dataset verifies existence)
                }
            )

    if missing:
        print(f"WARNING: {len(missing)} cases have no slides on disk, e.g. {missing[:5]}", file=sys.stderr)
    if skipped:
        top = sorted(skipped.items(), key=lambda kv: -kv[1])
        print(f"Skipped {sum(skipped.values())} non-T/N slides; suffix breakdown (top 15):", file=sys.stderr)
        for tail, count in top[:15]:
            print(f"  {tail}: {count}", file=sys.stderr)

    # Slides excluded from embedding (missing QC masks) cannot be trained on —
    # dropping them here beats failing in GradingDataset with a missing-file error.
    if args.require_embeddings:
        emb_dir = Path(args.require_embeddings)
        before = len(slides)
        slides = [s for s in slides if (emb_dir / f"{s['slide_id']}.parquet").is_file()]
        print(f"Dropped {before - len(slides)} slides without embeddings in {emb_dir}")

    # 2. Split: assign per CASE, stratified by the case's Nottingham grade.
    # A case's T slide carries that grade and its N slide is grade 0, so a case
    # spans two slide-grades — stratifying by slide grade would let one case land
    # in two splits (BRC-A-16). Stratify by case grade instead; every slide of a
    # case inherits the case's split.
    by_grade: dict[int, list[str]] = defaultdict(list)
    for case, grade in cases.items():
        by_grade[grade].append(case)
    case_split: dict[str, str] = {}
    for case_list in by_grade.values():
        rng.shuffle(case_list)
        n = len(case_list)
        n_train = round(n * SPLIT_FRACTIONS["train"])
        n_val = round(n * SPLIT_FRACTIONS["val"])
        # Last case(s) go to test so the three splits sum to n.
        for i, case in enumerate(case_list):
            if i < n_train:
                case_split[case] = "train"
            elif i < n_train + n_val:
                case_split[case] = "val"
            else:
                case_split[case] = "test"

    # 3. Write data_mapping.csv (path, grade, case) for tiling.py.
    mapping_path = out_dir / "data_mapping.csv"
    with mapping_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "grade", "case"])
        for s in slides:
            writer.writerow([s["path"], s["grade"], s["case"]])

    # 4. Write labels.csv (slide_id, grade, split) for GradingDataset.
    labels_path = out_dir / "labels.csv"
    with labels_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["slide_id", "grade", "split"])
        for s in slides:
            writer.writerow([s["slide_id"], s["grade"], case_split[s["case"]]])

    # 5. Self-check + summary.
    assert len(slides) > 0, "No slides found — check --tree and --ext."
    for split in SPLIT_FRACTIONS:
        grades_in_split = {s["grade"] for s in slides if case_split[s["case"]] == split}
        assert grades_in_split <= VALID_GRADES | {0}, (split, grades_in_split)

    print(f"\nWrote {len(slides)} slides ({len(cases)} cases) to {out_dir}/")
    print(f"  {mapping_path}")
    print(f"  {labels_path}")
    print("\nSlide count by grade x split:")
    table: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in slides:
        table[s["grade"]][case_split[s["case"]]] += 1
    for grade in sorted(table):
        row = table[grade]
        print(f"  grade {grade}: " + "  ".join(f"{k}={row[k]}" for k in SPLIT_FRACTIONS))


if __name__ == "__main__":
    main()
