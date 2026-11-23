"""Submit the overnight MammaPrint job (embed 308 WSIs + our attention, in series).

One pod, GPU first, both stages resumable.

    uv run python scripts/run_mammaprint_job.py               # A40, proven config
    MM_GPU=mig-2g.20gb uv run python scripts/run_mammaprint_job.py   # if A40 stays Pending

kube-jobs rejects an existing job name: delete the old one first
(kubectl --context kubas-cluster delete job -n rationai-jobs-ns brc-grading-mmattn) or
run alongside it under a second name: MM_JOB=brc-grading-mmattn-tiff.
"""

import os

from kube_jobs import storage
from run_embeddings import submit_with_hf_token  # HF_TOKEN secret injection


JOB = os.environ.get("MM_JOB", "brc-grading-mmattn")
# proven batch sizes: 512 A40 (cohort job), 512 mig-2g.20gb (configs/preprocessing/
# embeddings.yaml comment), 128 mig-1g.10gb is a guess at ~1/4 of the 10GB slice
BATCH = {"A40": 512, "A100": 1024, "H100": 2048, "mig-2g.20gb": 512, "mig-2g.24gb": 512,
         "mig-1g.10gb": 128}

gpu = os.environ.get("MM_GPU", "A40")

submit_with_hf_token(
    job_name=JOB,
    image="cerit.io/rationai/base:2.0.6",
    cpu=16,
    memory="64Gi",
    gpu=gpu,
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export HF_HOME=/mnt/projects/breast_cancer/brc_grading/huggingface_cache",
        "export PYTHONUNBUFFERED=1",
        f".venv/bin/python scripts/embed_mammaprint.py --batch {BATCH.get(gpu, 512)}",
        ".venv/bin/python scripts/mammaprint_attention_stage.py",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
