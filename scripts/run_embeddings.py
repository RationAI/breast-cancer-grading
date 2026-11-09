from typing import Any
from unittest.mock import patch

from kube_jobs import storage, submit_job
from kubernetes import client


def submit_with_hf_token(**kwargs: Any) -> None:
    original_create = client.BatchV1Api.create_namespaced_job

    def create_with_hf_token(
        api: Any, body: Any, namespace: str, **call_kwargs: Any
    ) -> Any:
        body.spec.template.spec.containers[0].env.append(
            client.V1EnvVar(
                name="HF_TOKEN",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="huggingface-secret", key="token"
                    )
                ),
            )
        )
        return original_create(api, body=body, namespace=namespace, **call_kwargs)

    # ponytail: kube-jobs lacks secret-env support; add it there when upgraded.
    with patch.object(client.BatchV1Api, "create_namespaced_job", create_with_hf_token):
        submit_job(**kwargs)


if __name__ == "__main__":
    submit_with_hf_token(
        job_name="brc-grading-embeddings-virchow2",
        image="cerit.io/rationai/base:2.0.6",
        cpu=12,
        memory="64Gi",
        gpu="A40",
        public=False,
        script=[
            "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
            "cd workdir",
            "uv sync --frozen",
            "export HF_HOME=/mnt/projects/breast_cancer/brc_grading/huggingface_cache",
            "export PYTHONUNBUFFERED=1",
            "uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings/virchow2"
            " precision=bf16-mixed dataloader.batch_size=512",
        ],
        storage=[
            storage.secure.DATA,
            storage.secure.PROJECTS,
            storage.secure.BIOPTIC_TREE,
        ],
    )
