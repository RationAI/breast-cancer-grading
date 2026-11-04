from kube_jobs import storage, submit_job


submit_job(
    job_name="brc-grading-preprocessing-qc",
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.quality_control",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
