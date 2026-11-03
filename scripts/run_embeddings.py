from kube_jobs import storage, submit_job


submit_job(
    job_name="brc-grading-embeddings-virchow2",
    image="cerit.io/rationai/base:2.0.6",
    cpu=24,
    memory="64Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "export HF_TOKEN=<YOUR_HF_TOKEN>",
        "uv sync --frozen",
        "uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings/virchow2",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
