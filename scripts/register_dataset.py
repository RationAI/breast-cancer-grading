from kube_jobs import storage, submit_job


submit_job(
    job_name="brc-grading-register-dataset",
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -c \"from rationai.provenance import register_dataset; print(register_dataset(csv='/mnt/projects/breast_cancer/brc_grading/data_mapping.csv', path_column='path', dataset_name='bc20-24', experiment_name='BRC Grading'))\"",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
