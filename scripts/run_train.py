"""Submit the MIL grading training job.

Prerequisites: the embeddings job is Complete and registered
(``scripts/register_embeddings.py``), with its printed run_id pasted into
``configs/ml.yaml`` (``dataset.run_id``). Labels live in
``<project_path>/labels.csv`` (slide_id, grade, split) — regenerate with
``scripts/make_cohort.py`` if the cohort changed.

Host memory: bags are big (median 61 MB, p90 144 MB, max 336 MB), so a batch of
8 read by 4 workers needs ~32 GiB of RAM in flight; raise ``memory`` before
raising ``datamodule.batch_size`` / ``num_workers``.
"""

from kube_jobs import storage, submit_job


MANIFEST = "/mnt/projects/breast_cancer/brc_grading/embeddings/manifest.csv"


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
        # ProvenanceCallback looks for manifest.csv under ./data — without it,
        # dataset verification is silently skipped.
        f"mkdir -p data && cp {MANIFEST} data/",
        "uv run -m ml.train +experiment=ml/mil_attention",
    ],
    storage=[storage.secure.PROJECTS],
)
