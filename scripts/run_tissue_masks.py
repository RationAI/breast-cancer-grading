from kube_jobs import storage, submit_job


submit_job(
    job_name="brc-grading-preprocessing-tissue-masks",
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="8Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.tissue_masks",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
