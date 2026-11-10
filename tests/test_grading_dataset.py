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

print("GradingDataset bag + max_tiles test passed")
