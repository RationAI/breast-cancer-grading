"""Submit the MIL grading training job.

Data it reads (all on the PROJECTS mount, paths from ``configs/data/grading.yaml``):

* ``<project_path>/embeddings/bc20-24/<slide_id>.parquet`` — the 1735 slides from
  the embeddings job, archived as artifacts of MLflow run
  ``96532a9ff29241e4b65d57dbc379e944`` (that run's id is pinned in
  ``configs/ml.yaml`` as the provenance pointer);
* ``<project_path>/labels.csv`` — ``slide_id, grade, split``; must list exactly the
  embedded slides (``make_cohort.py --require-embeddings`` enforces that).

Resumable: checkpoints go to ``<project_path>/checkpoints/ml_attention`` on the
PVC, and resubmitting picks up ``last.ckpt`` automatically, so a preempted pod
continues instead of restarting. To deliberately retrain from scratch, delete that
directory (or pass ``checkpoint=null`` after removing it).

kube-jobs names the job ``<job_name>-<user>``, so one resubmit per attempt:
``kubectl delete job brc-grading-train-mil-attention-<user> -n rationai-jobs-ns``
(the failed attempt keeps its record in MLflow, so nothing is lost).

Host memory: bags are big (median 61 MB, p90 144 MB, max 336 MB) and ~8 slide
parquets are decoded per second, so most of the RAM budget is dataloader workers
holding bags in flight. Raise ``memory`` before ``datamodule.num_workers``.
"""

from kube_jobs import storage, submit_job


CKPT_DIR = "/mnt/projects/breast_cancer/brc_grading/checkpoints/ml_attention"


submit_job(
    job_name="brc-grading-train-mil-attention",
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
        # Resume in place if a previous attempt left a checkpoint behind.
        (
            f'last="{CKPT_DIR}/last.ckpt"; resume=""; '
            'if [ -f "$last" ]; then resume="checkpoint=$last"; echo "Resuming from $last"; fi'
        ),
        "uv run -m ml.train +experiment=ml/mil_attention $resume",
    ],
    storage=[storage.secure.PROJECTS],
)
