from kube_jobs import storage, submit_job


submit_job(
    job_name="brc-grading-tiling-bc20-24-224",
    image="cerit.io/rationai/base:2.0.6",
    cpu=32,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.tiling +experiment=preprocessing/tiling/bc20_24_224",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
