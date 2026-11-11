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
import re

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

    Deliberately the venv's own python rather than ``uv run``: several variants can share
    one pod, and going through the synced venv directly skips uv's project resolution and
    lock on the critical path (uv is still what builds the venv, one line earlier).

    The exit code goes into ``/tmp/train_rc`` because the pod's script continues after
    training (the ``sleep`` + ``wait`` tail exists to sample the GPU), so without it a
    pod whose training died immediately still reported ``Succeeded`` -- four did, today.
    """
    ckpt = f"{BASE_CKPT_DIR}-{variant}" if variant else BASE_CKPT_DIR
    return (
        f'ck="{ckpt}"; f="$ck/resume.ckpt"; [ -f "$f" ] || f="$ck/last.ckpt"; resume=""; '
        'if [ -f "$f" ]; then resume="checkpoint=$f"; echo "resuming from $f"; fi; '
        'export PREDICTIONS_DIR="$ck/predictions"; '
        ".venv/bin/python -m ml.train +experiment=ml/mil_attention "
        f"checkpoints_dir={ckpt} +metadata.hyperparams.variant={variant_tag(variant)} "
        f'{overrides} $resume; code=$?; echo "$code" >> /tmp/train_rc; echo "training exit $code"'
    )


def packed_plan(plan: str) -> list[str]:
    """Turn ``tag:overrides;tag:overrides`` into backgrounded, log-prefixed lines.

    ``sed -u`` prefixes every line with the tag so the variants stay separable in
    ``kubectl logs``. ``true`` closes each line because kube-jobs joins the script with
    ``;`` and a line ending in ``&`` would become ``&;``. Exit codes land in
    ``/tmp/train_rc`` (see ``train_command``) for the script's final check.
    """
    lines = []
    for spec in filter(None, (s.strip() for s in plan.split(";"))):
        tag, _, overrides = spec.partition(":")
        lines.append(
            f'( {train_command(tag, overrides)} ) 2>&1 | sed -u "s|^|[{tag}] |" & true'
        )
    if not lines:
        raise ValueError(f"Empty TRAIN_PLAN: {plan!r}")
    return lines


def cache_budget_gib(overrides: str, emb_cache_gb: str) -> float:
    """GiB of bag cache the pod will want, for one training process.

    ``EMB_CACHE_GB`` and ``datamodule.*.cache_mb`` are budgets **per dataloader worker,
    per split**, and a decoded bag is float32 where the parquet on disk is float16, so
    the real footprint is about twice the nominal number. This is the arithmetic that
    OOMKilled three 24 Gi pods in epoch 0 (exit 137, no traceback) on 2026-11-10 at
    ``EMB_CACHE_GB=0.75`` with 4 workers. A ``TRAIN_PLAN`` pack runs several processes in
    one pod, so treat the result as a floor.
    """
    splits = [
        int(m) for m in re.findall(r"(?:train|val|test)\.cache_mb=(\d+)", overrides)
    ]
    # Every match counts, not one trio: a ``TRAIN_PLAN`` pack carries its own cache_mb set per
    # run, and summing only when exactly three showed up scored a two-run pack as the 18 GB
    # config default and refused a pod that fits twice over.
    nominal_gib = sum(splits) / 1024 if splits else float(emb_cache_gb) * 3
    workers = re.search(r"num_workers=(\d+)", overrides)
    return (
        nominal_gib * (int(workers.group(1)) if workers else 8) * 2
    )  # 8 = config default


def check_cache_budget(gib: float, memory: str) -> None:
    """Refuse a pod whose cache cannot fit alongside the batches in flight.

    Half the pod is the line: the other half carries prefetched bags, torch, and the
    CUDA host side, which is how the 24 Gi / 0.75 GB shape failed.
    """
    pod = float(re.sub(r"[^0-9.]", "", memory) or 0)
    if gib > pod / 2:
        raise SystemExit(
            f"FATAL: bag cache would want ~{gib:.0f} GiB of a {memory} pod. Pass fewer "
            "workers, a smaller datamodule.{train,val,test}.cache_mb, or more memory."
        )


def main() -> None:
    variant = os.getenv("TRAIN_VARIANT", "")
    plan = os.getenv("TRAIN_PLAN", "")
    overrides_text = f"{os.getenv('TRAIN_OVERRIDES', '')} {plan}"
    emb_gb = os.getenv("TRAIN_EMB_CACHE_GB", "")
    if emb_gb or "cache_mb=" in overrides_text:
        # Only when the budget was set on purpose: the config default (3 GB x 3 splits x
        # 8 workers) is far over any pod and is somebody else's problem to redesign.
        check_cache_budget(
            cache_budget_gib(overrides_text, emb_gb or "18"),
            os.getenv("TRAIN_MEMORY", "48Gi"),
        )
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
        # A40 or an MIG slice of one (``mig-1g.10gb``, ``mig-2g.20gb`` -- what the other
        # RationAI jobs ask for). A run touches ~10 GB of the card, so a slice is honest
        # capacity-wise and easier to place than a whole card; the cost is fewer SMs, so
        # epochs take longer.
        gpu=os.getenv("TRAIN_GPU", "A40"),
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
            "rm -f /tmp/train_rc",
            *training,
            "sleep 300; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader",
            # Green-but-did-nothing pods cost an hour each; make the job status honest.
            "wait; [ -s /tmp/train_rc ] || { echo FATAL: no training exit code recorded; exit 1; }",
            "awk '{s+=$1} END{exit s>0}' /tmp/train_rc || { echo FATAL: training failed; exit 1; }",
        ],
        storage=[storage.secure.PROJECTS],
    )


if __name__ == "__main__":
    main()
