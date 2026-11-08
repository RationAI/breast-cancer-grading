from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from datasets import Dataset

from ml.data.datasets.tiles import OpenSlideTilesDataset, SlideDataset


slides = Dataset.from_dict(
    {
        "id": ["slide-id"],
        "path": ["/mnt/data/2020_00001-01-T.mrxs"],
        "level": [0],
        "tile_extent_x": [224],
        "tile_extent_y": [224],
    }
)
tiles = Dataset.from_dict({"slide_id": ["slide-id"], "tile_x": [10], "tile_y": [20]})
inner = MagicMock()
inner.__getitem__.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
with patch(
    "ml.data.datasets.tiles.OpenSlideTilesDataset", return_value=inner
) as tiles_reader:
    dataset = SlideDataset(slides_and_tiles=(slides, tiles))
    tile_dataset = next(iter(dataset.generate_datasets()))
    tiles_for_images = tiles_reader.call_args.kwargs["tiles"]
    assert "x" in tiles_for_images.column_names
    assert "y" in tiles_for_images.column_names
    _, metadata = tile_dataset[0]
    assert metadata["x"] == 10
    assert metadata["y"] == 20
    assert metadata["slide_id"] == "2020_00001-01-T"

print("Tile coordinate compatibility test passed")

with patch("ml.data.datasets.tiles.OpenSlide") as open_slide:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    open_slide.return_value.read_tile.return_value = image
    reader = OpenSlideTilesDataset(
        slide_path="slide.mrxs",
        level=0,
        tile_extent_x=224,
        tile_extent_y=224,
        tiles=tiles_for_images,
    )
    assert reader._slide is None
    assert reader[0] is image
    assert reader[0] is image
    open_slide.assert_called_once_with(Path("slide.mrxs"))
    assert open_slide.return_value.read_tile.call_count == 2
    reader.close()
    assert reader._slide is None
    open_slide.return_value.close.assert_called_once()

print("Slide-handle reuse and cleanup test passed")
