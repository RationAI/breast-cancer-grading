"""Assemble the xOpat overlay report from the mask TIFFs that are on the PVC.

Runs the internal ``report`` app (it turns mask + slide paths into an HTML
gallery of xOpat deep-link URLs and logs it to MLflow). Submit it as a job --
MLflow and the PVC are only reachable in the cluster::

    REPORT_JOB_NAME=brc-grading-xopat-report5 REPORT_SCRIPT=xopat_report.py \
      REPORT_ARGS='--reporter attention_test --user jiribuchta' \
      uv run --no-sync python scripts/run_report.py

Then::

    kubectl logs -n rationai-jobs-ns job/brc-grading-xopat-report5

The HTML also lands on the PVC under ``reports/`` and in MLflow, experiment
``MIL attention masks``. ``--reporter`` is the file name (without .yaml) in
``report_conf/reporter/`` written by ``attention_masks.py --stage report``:
``attention_test_conf`` = the 5-slide pilot, ``attention_test`` = the whole
split once the mask job is done (regenerate it with ``--stage report``).

``report`` comes from an internal GitLab repo over SSH, which the pods have no
key for, so the copy Tile-Morph already had installed is vendored on the PVC and
put on PYTHONPATH (MIT-licensed, license file included, 207 KB).
shortcut: hand-copied snapshot of report 1.1.5; re-copy from a notebook venv if
their masks.py/xopat_helpers.py change.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/mnt/projects/breast_cancer/brc_grading")
VENDOR = ROOT / "vendor" / "report"
CONF_DIR = ROOT / "report_conf"
EXPERIMENT = "Breast Cancer Grading"

# orjson is a dependency of `report` that our locked venv does not carry; uv's
# configured index reaches nexus through the pod proxy, so no --index-url here.


def compose_forms(conf_dir: Path) -> list[list[str]]:
    """The two ways to point Hydra at an external ``reporter/`` config dir.

    Verified locally against the package: ``--config-dir`` acts as an *extra*
    group dir -- the app's primary config (and its ``mlflow``/``hydra`` groups)
    still come from the installed package, so ``report_conf`` only has to hold
    ``reporter/*.yaml``. The reporter override itself has to be ``+reporter=``
    because upstream ``default.yaml`` keeps ``reporter`` a plain node, so a bare
    ``reporter=name`` has no defaults-list match ("Could not override
    'reporter'").
    """
    return [
        ["--config-dir", str(conf_dir)],
        [f"++hydra.searchpath=[{conf_dir}]"],
    ]


def run_report(
    form: list[str],
    reporter: str,
    user: str,
    out_dir: Path,
    resolve_only: bool,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """One invocation of the report app in its own hydra run dir."""
    argv = [
        sys.executable,
        "-m",
        "report",
        *form,
        f"+reporter={reporter}",
        f"user={user}",
        f"hydra.run.dir={out_dir}",
    ]
    if resolve_only:
        argv += ["--cfg", "job", "--resolve"]
    return subprocess.run(
        argv, check=False, capture_output=True, text=True, timeout=3600, env=env
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reporter", default="attention_test")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--regen",
        action="store_true",
        help="Rewrite the reporter YAML for the masks actually on disk first.",
    )
    ap.add_argument("--user", default=os.getenv("JOB_USER", "jiribuchta"))
    ap.add_argument("--conf-dir", type=Path, default=CONF_DIR)
    ap.add_argument("--vendor", type=Path, default=VENDOR)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/xopat-report"))
    ap.add_argument(
        "--save-to",
        type=Path,
        default=ROOT / "reports",
        help="Where to copy the HTML (PVC path so it survives the pod).",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        for form in compose_forms(Path("CONF")):
            print(" ".join(form))
        return 0

    if not (args.vendor / "report" / "__main__.py").is_file():
        print(f"FATAL: no vendored report package at {args.vendor}")
        return 1
    if args.regen:
        # Cheap stage: reads the bag index + summary.csv, no model, no rendering.
        gen = subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/attention_masks.py",
                "--split",
                args.split,
                "--stage",
                "report",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        print((gen.stdout + gen.stderr)[-800:])
        if gen.returncode:
            print("FATAL: could not regenerate the reporter YAML")
            return 1
    reporter_yaml = args.conf_dir / "reporter" / f"{args.reporter}.yaml"
    if not reporter_yaml.is_file():
        print(f"FATAL: no reporter config {reporter_yaml}")
        return 1
    print(f"reporter config: {reporter_yaml}")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.vendor)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    if subprocess.run(
        [sys.executable, "-c", "import orjson"], capture_output=True, check=False
    ).returncode:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "orjson",
            ],
            check=False,
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    working: list[str] | None = None
    for form in compose_forms(args.conf_dir):
        probe = run_report(
            form, args.reporter, args.user, out_dir, resolve_only=True, env=env
        )
        ok = "mask_retrievers" in probe.stdout
        print(f"form {form}: {'composes' if ok else 'FAILED'}")
        if not ok:
            print((probe.stdout + probe.stderr)[-1200:])
        else:
            print(probe.stdout[-600:])
            working = form
            break
    if working is None:
        print("FATAL: neither hydra invocation composes the config")
        return 1

    result = run_report(
        working, args.reporter, args.user, out_dir, resolve_only=False, env=env
    )
    print((result.stdout + result.stderr)[-2500:])

    produced = sorted(out_dir.rglob("report.html"))
    if not produced:
        print("FATAL: report run left no report.html")
        return 1
    args.save_to.mkdir(parents=True, exist_ok=True)
    dest = args.save_to / f"{args.reporter}.html"
    shutil.copyfile(produced[-1], dest)
    print(f"saved {dest} ({dest.stat().st_size // 1024} KB)")
    print(f"run {result.returncode}; also in MLflow experiment '{EXPERIMENT}'")
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
