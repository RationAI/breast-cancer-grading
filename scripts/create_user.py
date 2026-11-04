from kube_jobs import storage, submit_job


submit_job(
    job_name="brc-grading-create-user",
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -c \"import os; from rationai.provenance import register_new_user; print(register_new_user(username=os.environ['JOB_USER'], real_name='TODO', email='TODO', organization='TODO', lead_name='TODO', lead_email='TODO'))\"",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
