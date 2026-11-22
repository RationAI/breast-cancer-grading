"""Do kissmi's 308 MammaPrint explanation slides touch our TRAINING cohort?

Tile-level attention-agreement against his MammaPrint model (MLflow exp 3, run
``6f27144cbb574e7794b997003069b95f``) is only honest on slides our grader never
trained on — the MMCI set taught us that the hard way (36/100 "unseen" negatives
were in train). This joins his ``explanations/slide_keys.csv`` against the
cohort on the PVC at every key that could plausibly mean "same case":

  * his ``record_num``/``slide_id`` (``2023/00835``, ``P2023_0835``, and the
    two-slide ``P2023_02395-1`` variants) and our ``slide_id``/``case``
    (``2020_00782-01-T``) all normalize to one accession key ``(year, serial)``,
  * plus the raw WSI file stem against ``data_mapping.csv`` paths,
  * plus the BRC-A-20 exclusion lists in ``scripts/make_cohort.py`` (those cases
    left the cohort but may still explain a hit).

Read-only: downloads 2 small artifacts, reads 2 CSVs + 5 fold CSVs, prints a
table, writes nothing.

    uv run --no-sync python scripts/check_mammaprint_overlap.py   # in-cluster
    uv run --no-sync python scripts/check_mammaprint_overlap.py --self-test
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd


BASE = Path("/mnt/projects/breast_cancer/brc_grading")
EXPLAIN_RUN = "6f27144cbb574e7794b997003069b95f"
KFOLDS = 5
_ACC = re.compile(r"(\d{4})[_/](\d+)")


def acc_key(s: str) -> tuple[str, int] | None:
    """Accession key shared by both naming styles, or None."""
    m = _ACC.search(str(s))
    return (m.group(1), int(m.group(2))) if m else None


def load_excluded() -> tuple[set[str], set[str]]:
    spec = importlib.util.spec_from_file_location(
        "make_cohort", Path(__file__).resolve().parent / "make_cohort.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module level is constants only
    return mod.EXCLUDED_CASES, mod.EXCLUDED_SLIDES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=EXPLAIN_RUN)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        assert acc_key("2023/00835") == acc_key("P2023_0835") == ("2023", 835)
        assert acc_key("2023_00835-01-T") == ("2023", 835)
        assert acc_key("P2023_02395-1") == acc_key("2023_02395-2-T") == ("2023", 2395)
        assert acc_key("2021_06248-10-T2") == ("2021", 6248)
        assert acc_key("nope") is None
        print("self-test ok")
        return 0

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    keys = pd.read_csv(client.download_artifacts(args.run, "explanations/slide_keys.csv"))
    labels = pd.read_csv(args.base / "labels.csv")
    mapping = pd.read_csv(args.base / "data_mapping.csv")
    folds = {
        k: pd.read_csv(args.base / "kfold" / f"labels-kf{k}.csv") for k in range(KFOLDS)
    }
    excl_cases, excl_slides = load_excluded()

    his = keys[keys.status == "ok"].copy()
    his["key"] = his.record_num.map(acc_key)
    assert his.key.notna().all(), "a record_num failed to normalize -- inspect, do not trust"
    his["stem"] = his.slide_id.str.removeprefix("P")

    ours = labels.copy()
    ours["key"] = ours.slide_id.map(acc_key)
    ours["stem"] = ours.slide_id  # our slide_id IS the WSI file stem
    mapping["key"] = mapping.case.map(acc_key)
    mapping["stem"] = mapping.path.str.rsplit("/", n=1).str[-1]
    excl_keys = {acc_key(c) for c in excl_cases}

    print(f"his slides: {len(his)} | our labeled slides: {len(ours)} "
          f"| data_mapping rows: {len(mapping)}")

    # 1. same case (accession), the level our splits are disjoint on
    hit = ours.merge(his, on="key", suffixes=("_ours", "_his"))
    print(f"\n== case-level overlap with labels.csv: {hit.slide_id_his.nunique()} of "
          f"{his.slide_id.nunique()} his slides (accessions: {hit.key.nunique()})")
    if len(hit):
        cols = ["slide_id_his", "record_num", "slide_id_ours", "grade", "split"]
        for k, f in folds.items():
            fold_slides = set(f.slide_id)
            sub = hit[hit.slide_id_ours.isin(fold_slides)]
            print(f"   kf{k}: {sub.slide_id_his.nunique()} his slides would sit in "
                  f"that fold's TRAIN/VAL/TEST")
        print(hit[cols].sort_values(["split", "slide_id_his"]).to_string(index=False))

    # 2. same case via data_mapping (catches cases excluded from labels.csv)
    hitm = mapping.merge(his, on="key", suffixes=("_ours", "_his"))
    print(f"\n== case-level overlap with data_mapping.csv (incl. cases outside labels): "
          f"{hitm.slide_id_his.nunique()} of {his.slide_id.nunique()}")
    if len(hitm):
        print(hitm[["slide_id_his", "record_num", "case", "grade", "path"]]
              .sort_values("slide_id_his").to_string(index=False))

    # 3. literally the same WSI file (P2023_0835 stem in our mapping paths)
    stem_hit = mapping[mapping.stem.isin(set(his.stem))]
    print(f"\n== identical WSI file stem in data_mapping.csv: {len(stem_hit)} rows")
    if len(stem_hit):
        print(stem_hit[["stem", "grade", "case", "path"]].to_string(index=False))

    # 4. BRC-A-20 excluded cases -- not trained, but same physical case
    ex = his[his.key.isin(excl_keys) | his.slide_id.isin(excl_slides)]
    print(f"\n== his slides in our EXCLUDED lists (never trained, still same case): {len(ex)}")
    if len(ex):
        print(ex[["slide_id", "record_num", "mammaprint_index"]].to_string(index=False))

    verdict = "CONTAMINATED" if len(hit) or len(stem_hit) else "no overlap found"
    print(f"\nVERDICT: {verdict} "
          f"({hit.slide_id_his.nunique()}/{len(his)} of his slides share a labeled case)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
