from kube_jobs import storage, submit_job


submit_job(
    job_name="brc-grading-tiling-mou-224",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.tiling +experiment=preprocessing/tiling/mou_224",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
