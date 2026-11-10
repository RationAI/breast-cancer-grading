"""Submit a CPU-only job that prints the grading results table.

MLflow and the shared PVC holding the checkpoints are only reachable from inside
the cluster, and this laptop's kubeconfig cannot even list services outside
``rationai-jobs-ns``, so the report has to run as a job::

    uv run --no-sync python scripts/run_report.py
    REPORT_VARIANTS=full_s456,full_ord_s456 uv run --no-sync python scripts/run_report.py

Then read it from the pod's logs::

    kubectl logs -n rationai-jobs-ns job/brc-grading-report-jiri-buchta

No GPU: it only reads CSVs and asks MLflow for scalars. The clone/sync steps are
duplicated from ``scripts/run_train.py`` rather than shared, to keep the tested
training submission path untouched.
"""

import os

from kube_jobs import storage, submit_job


submit_job(
    job_name=os.getenv("REPORT_JOB_NAME", "brc-grading-report"),
    image="cerit.io/rationai/base:2.0.6",
    cpu=int(os.getenv("REPORT_CPU", "4")),
    memory=os.getenv("REPORT_MEMORY", "8Gi"),
    public=False,
    script=[
        (
            "for i in 1 2 3; do GIT_TERMINAL_PROMPT=0 git clone "
            "https://github.com/RationAI/breast-cancer-grading.git workdir "
            "&& break; echo clone_attempt_$i_failed; sleep 15; done"
        ),
        "cd workdir",
        "[ -f pyproject.toml ] || { echo 'FATAL: clone failed, no pyproject.toml'; exit 1; }",
        "uv sync --frozen",
        ".venv/bin/python scripts/report_variants.py"
        + (
            f" --variants {os.environ['REPORT_VARIANTS']}"
            if os.getenv("REPORT_VARIANTS")
            else ""
        ),
        # Green-but-did-nothing pods are useless; make the status honest.
        "[ $? -eq 0 ] || { echo FATAL: report failed; exit 1; }",
    ],
    storage=[storage.secure.PROJECTS],
)
