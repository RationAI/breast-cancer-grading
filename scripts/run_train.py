"""Submit the MIL grading training job(s).

Data it reads (all on the PROJECTS mount, paths from ``configs/data/grading.yaml``):

* ``<project_path>/embeddings/bc20-24/<slide_id>.parquet`` — the 1735 slides from
  the embeddings job, archived as artifacts of MLflow run
  ``96532a9ff29241e4b65d57dbc379e944`` (that run's id is pinned in
  ``configs/ml.yaml`` as the provenance pointer);
* ``<project_path>/labels.csv`` — ``slide_id, grade, split``; must list exactly the
  embedded slides (``make_cohort.py --require-embeddings`` enforces that).

Resumable: each variant owns ``<project_path>/checkpoints/ml_attention-<variant>`` on
the PVC and resubmitting picks up ``resume.ckpt`` (written unconditionally every
epoch) automatically, so a preempted pod continues instead of restarting. To
deliberately retrain from scratch, delete that variant's directory.

kube-jobs names the job ``<job_name>-<user>``, so one resubmit per attempt:
``kubectl delete job brc-grading-train-mil-attention-<variant>-<user>
-n rationai-jobs-ns`` (the failed attempt keeps its record in MLflow).

One variant per run::

    TRAIN_VARIANT=tiles2048-s123 TRAIN_OVERRIDES="seed=123" python scripts/run_train.py

Several variants packed onto the one A40 the pod gets (``TRAIN_PLAN`` is
``tag:overrides;tag:overrides``; ``TRAIN_VARIANT`` then only names the job, while the
plan tags name the checkpoint dirs and the MLflow ``variant`` hyperparam). A training
run is data-bound and small -- ~10 GB of the 48 GB on the card -- so packing three of
them is what actually uses the GPU. Keep both budgets inside the pod:

* VRAM: 3 x ~10-12 GB fits an A40, 4 does not;
* RAM: ask for what the variants really use, not for the whole node. Each variant's
  workers hold their own ``datamodule.*.cache_mb`` bag cache (default 3072 MB, falling
  back to ``EMB_CACHE_GB``), so a pack multiplies that by variants x (workers + 1);
  ``TRAIN_EMB_CACHE_GB`` caps it at 5 GB per variant -- still far more than one epoch's
  bags. Request 24 Gi: a 48 Gi request does not schedule on a GPU node that already
  carries other jobs, which is how pack3 got OOM-killed with ``resource is out of
  usage: memory`` before it trained anything.
* CPU: ~3 per variant at ``num_workers=2``; keep the default 12, which is what the solo
  runs scheduled with.

    TRAIN_VARIANT=pack3 TRAIN_PLAN="att-s123:seed=123 datamodule.num_workers=2;\
    att-s456:seed=456 datamodule.num_workers=2;\
    mean-s123:seed=123 ml/aggregator=mean datamodule.num_workers=2" \
    python scripts/run_train.py

Packed runs interleave in the pod log, each line prefixed ``[tag]``; grep that to read
one variant. Each prints ``exit <code>`` at the end, because the pod itself exits 0 as
soon as ``wait`` returns -- check those lines rather than the job status. The run also
samples ``nvidia-smi`` once at ~5 min so the packing is measured rather than assumed.
"""

import os

from kube_jobs import storage, submit_job


BASE_CKPT_DIR = "/mnt/projects/breast_cancer/brc_grading/checkpoints/ml_attention"


def variant_tag(variant: str) -> str:
    """ASCII identifier for the ``variant`` hyperparam (see ``train_command``)."""
    return variant.replace("-", "_").replace(".", "_") or "baseline"


def train_command(variant: str, overrides: str = "") -> str:
    """Resume in place if this variant's directory has a checkpoint, then train.

    Deliberately quote-free where Hydra is concerned: the pod's shell strips quotes
    before Hydra's grammar sees a value, so an em dash or bracket in a run name kills
    config parsing. Variant identity therefore goes in as an ASCII ``variant`` tag.

    Deliberately the venv's own python rather than ``uv run``: three variants share one
    pod, and going through the synced venv directly skips uv's project resolution and
    lock on the critical path (uv is still what builds the venv, one line earlier).
    """
    ckpt = f"{BASE_CKPT_DIR}-{variant}" if variant else BASE_CKPT_DIR
    return (
        f'ck="{ckpt}"; f="$ck/resume.ckpt"; [ -f "$f" ] || f="$ck/last.ckpt"; resume=""; '
        'if [ -f "$f" ]; then resume="checkpoint=$f"; echo "resuming from $f"; fi; '
        ".venv/bin/python -m ml.train +experiment=ml/mil_attention "
        f"checkpoints_dir={ckpt} +metadata.hyperparams.variant={variant_tag(variant)} "
        f"{overrides} $resume"
    )


def packed_plan(plan: str) -> list[str]:
    """Turn ``tag:overrides;tag:overrides`` into backgrounded, log-prefixed lines.

    ``sed -u`` prefixes every line with the tag so the variants stay separable in
    ``kubectl logs``; the trailing ``echo`` reports the training exit code, which the
    pod's own status would otherwise hide. ``true`` closes each line because kube-jobs
    joins the script with ``;`` and a line ending in ``&`` would become ``&;``.
    """
    lines = []
    for spec in filter(None, (s.strip() for s in plan.split(";"))):
        tag, _, overrides = spec.partition(":")
        lines.append(
            f'( {train_command(tag, overrides)}; echo "exit $?" ) 2>&1 '
            f'| sed -u "s|^|[{tag}] |" & true'
        )
    if not lines:
        raise ValueError(f"Empty TRAIN_PLAN: {plan!r}")
    return lines


def main() -> None:
    variant = os.getenv("TRAIN_VARIANT", "")
    plan = os.getenv("TRAIN_PLAN", "")
    training = (
        packed_plan(plan)
        if plan
        else [train_command(variant, os.getenv("TRAIN_OVERRIDES", ""))]
    )
    submit_job(
        job_name=f"brc-grading-train-mil-attention-{variant}"
        if variant
        else "brc-grading-train-mil-attention",
        image="cerit.io/rationai/base:2.0.6",
        cpu=int(os.getenv("TRAIN_CPU", "12")),
        memory=os.getenv("TRAIN_MEMORY", "48Gi"),
        gpu="A40",
        public=False,
        script=[
            # GIT_TERMINAL_PROMPT=0 so a node without GitHub egress fails at once instead
            # of blocking on a username prompt; retry because the pod can land anywhere.
            (
                "for i in 1 2 3; do GIT_TERMINAL_PROMPT=0 git clone "
                "https://github.com/RationAI/breast-cancer-grading.git workdir "
                "&& break; echo clone_attempt_$i_failed; sleep 15; done"
            ),
            "cd workdir",
            # Fail loudly: without this the pod trains nothing, sleeps, and exits 0 with a
            # green job status (git's credential failure looked like a success downstream).
            "[ -f pyproject.toml ] || { echo 'FATAL: clone failed, no pyproject.toml'; exit 1; }",
            "uv sync --frozen",
            (
                # The bag cache budget is per process, so a pack has to divide it out.
                "export PYTHONUNBUFFERED=1 EMB_CACHE_GB="
                + os.getenv("TRAIN_EMB_CACHE_GB", "18")
            ),
            *training,
            "sleep 300; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader",
            "wait",
        ],
        storage=[storage.secure.PROJECTS],
    )


if __name__ == "__main__":
    main()
