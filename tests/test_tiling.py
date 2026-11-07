import io
import math
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd
from openslide import OpenSlideError

from preprocessing.tiling import compute_overlaps_batch


def fake_tile_overlay(_roi, paths, *_args):
    if "bad.tiff" in paths.to_pylist():
        raise OpenSlideError("unreadable test mask")
    if "" in paths.to_pylist():
        raise OpenSlideError("empty test mask path")
    return [np.ma.ones((1, 1, 1), dtype=np.uint8) for _ in paths]


batch = pd.DataFrame(
    {
        "tile_x": [0, 1, 2],
        "tile_y": [0, 1, 2],
        "mpp_x": [0.5, 0.5, 0.5],
        "mpp_y": [0.5, 0.5, 0.5],
        **{
            column: ["good.tiff", "bad.tiff", ""]
            for column in [
                "tissue_mask_path",
                "blur_mask_path",
                "folding_mask_path",
                "residual_mask_path",
            ]
        },
    }
)

stdout = io.StringIO()
with (
    patch("preprocessing.tiling._tile_overlay", side_effect=fake_tile_overlay),
    redirect_stdout(stdout),
):
    result = compute_overlaps_batch(batch, object(), object())

for column in [
    "tissue_overlap",
    "blur_overlap",
    "folding_overlap",
    "residual_overlap",
]:
    assert result.loc[0, column] == 1.0
    assert math.isnan(result.loc[1, column])
    assert result.loc[2, column] == 0.0

assert stdout.getvalue().count("bad.tiff") == 4
print("Mask error handling test passed")
