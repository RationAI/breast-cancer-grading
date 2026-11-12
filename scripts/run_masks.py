"""Submit the xOpat overlay job: attention masks for one split.

Same shape as Tile-Morph-Clustering/scripts/run_masks.py (same image, same
storage), with two additions this project needs: pyvips is not in the locked
venv (libvips itself ships in the image), and libvips encodes one TIFF tile per
thread, so let it use the whole pod.

    uv run --no-sync python scripts/run_masks.py                  # test split
    MASKS_SPLIT=train uv run --no-sync python scripts/run_masks.py
    # another value scale, or a few slides, into their own directory (a new scale
    # must not share a directory: stage 2 skips TIFFs that already exist):
    #   MASKS_ARGS='--scale sqrt --mask-dir .../masks_sqrt/test --slides a,b,c' \

Progress:  kubectl logs -n rationai-jobs-ns job/brc-grading-masks-test

It is resumable -- masks already on the PVC are skipped -- so a preemption or a
resubmit continues where it stopped. Delete a same-named job before resubmitting
(kube-jobs refuses a name collision).
"""
import os

from kube_jobs import storage, submit_job


split = os.getenv("MASKS_SPLIT", "test")

submit_job(
    job_name=os.getenv("MASKS_JOB_NAME", f"brc-grading-masks-{split}"),
    image="cerit.io/rationai/base:2.0.6",
    cpu=int(os.getenv("MASKS_CPU", "8")),
    memory=os.getenv("MASKS_MEMORY", "32Gi"),
    public=False,
    script=[
        (
            "for i in 1 2 3; do GIT_TERMINAL_PROMPT=0 git clone "
            "https://github.com/RationAI/breast-cancer-grading.git workdir "
            "&& break; echo clone_attempt_$i_failed; sleep 15; done"
        ),
        "cd workdir",
        "[ -f pyproject.toml ] || { echo 'FATAL: clone failed, no pyproject.toml'; exit 1; }",
        "uv sync --frozen",
        "uv pip install --python .venv/bin/python pyvips",
        "export VIPS_CONCURRENCY=$(nproc)",
        (
            f".venv/bin/python -u scripts/attention_masks.py --split {split} --stage all"
            f" {os.getenv('MASKS_ARGS', '')}"   # --scale / --mask-dir / --slides / --limit
        ),
        # Green-but-did-nothing pods are useless; make the status honest.
        "[ $? -eq 0 ] || { echo FATAL: masks failed; exit 1; }",
    ],
    storage=[storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
