"""Submit the MIL grading training job.

Data it reads (all on the PROJECTS mount, paths from ``configs/data/grading.yaml``):

* ``<project_path>/embeddings/bc20-24/<slide_id>.parquet`` — the 1735 slides from
  the embeddings job, archived as artifacts of MLflow run
  ``96532a9ff29241e4b65d57dbc379e944`` (that run's id is pinned in
  ``configs/ml.yaml`` as the provenance pointer);
* ``<project_path>/labels.csv`` — ``slide_id, grade, split``; regenerate with
  ``scripts/make_cohort.py`` if the cohort changed.

Host memory: bags are big (median 61 MB, p90 144 MB, max 336 MB), so a batch of
8 read by 4 workers needs ~32 GiB of RAM in flight; raise ``memory`` before
raising ``datamodule.batch_size`` / ``num_workers``.
"""

from kube_jobs import storage, submit_job


submit_job(
    job_name="brc-grading-train-mil-attention",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export PYTHONUNBUFFERED=1",
        "uv run -m ml.train +experiment=ml/mil_attention",
    ],
    storage=[storage.secure.PROJECTS],
)
