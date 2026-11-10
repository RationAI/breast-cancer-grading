"""Submit the MIL grading training job.

Data it reads (all on the PROJECTS mount, paths from ``configs/data/grading.yaml``):

* ``<project_path>/embeddings/bc20-24/<slide_id>.parquet`` — the 1735 slides from
  the embeddings job, archived as artifacts of MLflow run
  ``96532a9ff29241e4b65d57dbc379e944`` (that run's id is pinned in
  ``configs/ml.yaml`` as the provenance pointer);
* ``<project_path>/labels.csv`` — ``slide_id, grade, split``; must list exactly the
  embedded slides (``make_cohort.py --require-embeddings`` enforces that).

Resumable: checkpoints go to ``<project_path>/checkpoints/ml_attention`` on the
PVC, and resubmitting picks up ``resume.ckpt`` (saved unconditionally every epoch)
automatically, so a preempted pod continues instead of restarting. To deliberately
retrain from scratch, delete that directory.

kube-jobs names the job ``<job_name>-<user>``, so one resubmit per attempt:
``kubectl delete job brc-grading-train-mil-attention-<user> -n rationai-jobs-ns``
(the failed attempt keeps its record in MLflow, so nothing is lost).

Variant runs, so several configs can train side by side without sharing a
checkpoint directory (job name, checkpoint dir and MLflow run name all follow the
variant)::

    TRAIN_VARIANT=tiles2048-s123 TRAIN_OVERRIDES="seed=123" python scripts/run_train.py

Host memory: bags are big (median 61 MB, p90 144 MB, max 336 MB) and validation /
test always read whole slides, so most of the RAM budget is dataloader workers
holding bags in flight. Raise ``memory`` before ``datamodule.num_workers``.
"""

import os

from kube_jobs import storage, submit_job


VARIANT = os.getenv("TRAIN_VARIANT", "")
OVERRIDES = os.getenv("TRAIN_OVERRIDES", "")
SUFFIX = f"-{VARIANT}" if VARIANT else ""
CKPT_DIR = f"/mnt/projects/breast_cancer/brc_grading/checkpoints/ml_attention{SUFFIX}"
RUN_NAME = "Attention MIL — Nottingham grading (Virchow2 embeddings)" + (
    f" [{VARIANT}]" if VARIANT else ""
)


submit_job(
    job_name=f"brc-grading-train-mil-attention{SUFFIX}",
    image="cerit.io/rationai/base:2.0.6",
    cpu=12,
    memory="48Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export PYTHONUNBUFFERED=1",
        # Resume in place if a previous attempt left a checkpoint behind. ``resume``
        # is written every epoch; ``last`` only on improvement (older attempts).
        (
            f'ck="{CKPT_DIR}"; f="$ck/resume.ckpt"; [ -f "$f" ] || f="$ck/last.ckpt"; '
            'resume=""; if [ -f "$f" ]; then resume="checkpoint=$f"; echo "Resuming from $f"; fi'
        ),
        (
            f'uv run -m ml.train +experiment=ml/mil_attention metadata.run_name="{RUN_NAME}" '
            f"checkpoints_dir={CKPT_DIR} {OVERRIDES} $resume"
        ),
    ],
    storage=[storage.secure.PROJECTS],
)
