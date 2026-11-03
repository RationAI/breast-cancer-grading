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
        "uv run -m ml.train +experiment=ml/mil_attention",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
