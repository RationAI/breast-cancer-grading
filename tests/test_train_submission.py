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

# The cache shortcut is the expensive trap: EMB_CACHE_GB applies per split per worker and
# a decoded bag is twice the on-disk fp16 size, so 0.75 GB x 4 workers OOMKilled three
# 24 Gi pods in epoch 0. The submitter has to refuse that shape, and accept the one that
# runs (explicit per-split trio, 4 workers, 32 Gi).
assert run_train.cache_budget_gib("datamodule.num_workers=4", "0.75") == 18.0
assert (
    run_train.cache_budget_gib(
        "datamodule.num_workers=4 datamodule.train.cache_mb=768"
        " datamodule.val.cache_mb=512 datamodule.test.cache_mb=512",
        "18",
    )
    == 14.0
)
try:
    run_train.check_cache_budget(18.0, "24Gi")
except SystemExit as error:
    assert "bag cache" in str(error), error
else:
    raise AssertionError("the shape that OOMKilled three pods must be refused")
run_train.check_cache_budget(14.0, "32Gi")  # the shape the finished runs used

# A two-run pack: both runs' cache_mb sets count, so 2 x (512+256+256) MB x 2 workers x 2
# (fp16 on disk, float32 decoded). Scoring this as the 18 GB default refused a 32 Gi pod that
# only needs 8.
_pair = ";".join(
    f"kf{i}-tf1024:datamodule.num_workers=2 datamodule.train.cache_mb=512"
    " datamodule.val.cache_mb=256 datamodule.test.cache_mb=256"
    for i in (0, 1)
)
assert run_train.cache_budget_gib(_pair, "18") == 8.0, run_train.cache_budget_gib(_pair, "18")
run_train.check_cache_budget(8.0, "32Gi")

print("run_train plan-packing test passed")
