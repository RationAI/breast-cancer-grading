"""GradingDataset bag serving and tile subsampling."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.datasets.grading import GradingDataset


tmp = Path(tempfile.mkdtemp())
emb = tmp / "embeddings"
emb.mkdir()
BIG, SMALL = 9, 3
for slide, n in (("2020_00001-01-T", BIG), ("2020_00002-01-T", SMALL)):
    frame = pd.DataFrame(
        {
            "x": range(n),
            "y": range(n),
            "embedding": [np.full(4, i, dtype="float32") for i in range(n)],
        }
    )
    frame.to_parquet(emb / f"{slide}.parquet")
pd.DataFrame(
    {
        "slide_id": ["2020_00001-01-T", "2020_00002-01-T"],
        "grade": [2, 3],
        "split": ["train", "train"],
    }
).to_csv(tmp / "labels.csv", index=False)

full = GradingDataset(emb, tmp / "labels.csv", "train")
bag, label, meta = full[0]
assert (
    bag.shape == (BIG, 4)
    and label.item() == 2
    and meta["slide_id"] == "2020_00001-01-T"
)

sub = GradingDataset(emb, tmp / "labels.csv", "train", max_tiles=4)
capped, label, _ = sub[0]
assert capped.shape == (4, 4) and label.item() == 2
assert {tuple(row) for row in capped.numpy()} <= {tuple(row) for row in bag.numpy()}
assert sub[1][0].shape == (SMALL, 4), "slides under the cap must stay intact"

draws = {tuple(map(tuple, sub[0][0].tolist())) for _ in range(8)}
assert len(draws) > 1, "tile subsampling must vary between draws"

# --- fp16 bag cache: one read per slide, fp32 out, LRU eviction ---
import torch

from ml.data.datasets import grading


assert grading._BAGS_LIMIT == 0 and not grading._BAGS, "cache stays off unless sized"


class CountingPandas:
    """Stand-in for the pandas module, counting parquet reads."""

    def __init__(self, inner) -> None:
        self.inner, self.reads = inner, 0

    def read_parquet(self, *args, **kwargs):
        self.reads += 1
        return self.inner.read_parquet(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


cached = GradingDataset(emb, tmp / "labels.csv", "train", cache_mb=1)
shim = CountingPandas(grading.pd)
grading.pd = shim
first, second = cached[0][0], cached[0][0]
capped_cached = GradingDataset(
    emb, tmp / "labels.csv", "train", max_tiles=4, cache_mb=1
)[0][0]
grading.pd = shim.inner

assert shim.reads == 1, f"second access must be served from RAM, got {shim.reads} reads"
assert first.dtype == torch.float32 and torch.equal(first, bag), (
    "cache must not change values"
)
assert capped_cached.shape == (4, 4) and capped_cached.dtype == torch.float32
assert all(b.dtype == torch.float16 for b in grading._BAGS.values()), (
    "cache stores fp16"
)

# Two bags (9 and 3 tiles of 4 fp16 floats) do not fit in 80 bytes: the older one goes.
grading._BAGS.clear()
grading._BAGS_BYTES = 0
grading._BAGS_LIMIT = 80
cached[0]
cached[1]
assert len(grading._BAGS) == 1 and grading._BAGS_BYTES <= 80, (
    "LRU must stay under budget"
)
assert next(iter(grading._BAGS)).endswith("2020_00002-01-T.parquet"), (
    "oldest bag must be evicted"
)

print("GradingDataset bag + max_tiles + cache test passed")
