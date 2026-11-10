"""Packed multi-variant job script generation (scripts/run_train.py)."""

import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_train


PLAN = (
    "att-s123:seed=123 datamodule.num_workers=2;"
    "att-s456:seed=456 datamodule.num_workers=2;"
    "mean-s123:ml/aggregator=mean datamodule.num_workers=2"
)
lines = run_train.packed_plan(PLAN)

assert len(lines) == 3, "one backgrounded line per variant"
assert all(line.endswith("& true") for line in lines), (
    "kube-jobs joins with ';': no bare & line end"
)

expected_dirs = {
    f"{run_train.BASE_CKPT_DIR}-{tag}" for tag in ("att-s123", "att-s456", "mean-s123")
}
dirs = {line.split("checkpoints_dir=")[1].split()[0] for line in lines}
assert dirs == expected_dirs, (
    "each variant needs its own checkpoint dir to stay resumable"
)

for line, tag in zip(lines, ("att-s123", "att-s456", "mean-s123"), strict=True):
    assert f"[{tag}]" in line, "pod log must stay separable per variant"
    assert f"variant={tag.replace('-', '_')}" in line, "ASCII-only Hydra hyperparam"
    assert "resume.ckpt" in line and "$resume" in line, "packed runs must resume too"
    assert 'echo "$code" >> /tmp/train_rc' in line, (
        "pod exits 0 after its sleep+wait tail, so training status must be recorded"
    )

assert "seed=123 datamodule.num_workers=2" in lines[0]
assert "ml/aggregator=mean" in lines[2], "plan overrides reach the Hydra command"
assert all(".venv/bin/python -m ml.train" in line for line in lines), (
    "concurrent 'uv run' calls in one pod drop to the image's system python"
)
assert all("uv run" not in line for line in lines)
assert all("metadata.run_name" not in line for line in lines), (
    "quoted run names break parsing"
)

single = run_train.train_command("", "seed=1")
assert f"checkpoints_dir={run_train.BASE_CKPT_DIR} " in single
assert "variant=baseline" in single
assert "/tmp/train_rc" in single, "a solo pod must report its exit code too"

try:
    run_train.packed_plan("   ")
except ValueError:
    pass
else:
    raise AssertionError("an empty plan must fail before a job is submitted")

# kube-jobs concatenates the whole script with ';' into one bash -c, which is what
# turned a trailing '&' into a syntax error and killed the pod before it trained.
JOINED = ";".join(
    [
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export PYTHONUNBUFFERED=1",
        *lines,
        "sleep 300; nvidia-smi --format=csv,noheader",
        "wait; [ -s /tmp/train_rc ] || { echo FATAL: no training exit code recorded; exit 1; }",
        "awk '{s+=$1} END{exit s>0}' /tmp/train_rc || { echo FATAL: training failed; exit 1; }",
    ]
)
CHECK = subprocess.run(["bash", "-n"], input=JOINED, text=True, check=False)
assert CHECK.returncode == 0, f"pod script does not parse:\n{JOINED}"

# The status check itself, run against recorded codes.
AWK = "awk '{s+=$1} END{exit s>0}'"
for codes, want in (("0\n0\n", 0), ("0\n1\n", 1)):
    got = subprocess.run(
        [
            "bash",
            "-c",
            f"printf '{codes}' > /tmp/train_rc.test; {AWK} /tmp/train_rc.test",
        ],
        check=False,
    ).returncode
    assert got == want, f"status check must exit {want} for codes {codes!r}"

print("run_train plan-packing test passed")
