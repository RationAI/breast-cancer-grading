import io
from contextlib import redirect_stderr, redirect_stdout
from inspect import unwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import torch

from preprocessing.embeddings import (
    embeddings_are_complete,
    main,
    save_embeddings,
)


with TemporaryDirectory() as directory:
    path = Path(directory) / "slide.parquet"
    assert not embeddings_are_complete(path, 2)

    save_embeddings(
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.tensor([10, 20]),
        torch.tensor([30, 40]),
        path,
    )
    assert embeddings_are_complete(path, 2)
    assert not embeddings_are_complete(path, 3)
    assert pa.parquet.read_table(path).num_rows == 2

    # An interrupted write must be detected and replaced on resume.
    path.write_bytes(b"incomplete")
    assert not embeddings_are_complete(path, 2)
    save_embeddings(
        torch.zeros((2, 3)),
        torch.tensor([10, 20]),
        torch.tensor([30, 40]),
        path,
    )
    assert embeddings_are_complete(path, 2)
    assert not path.with_suffix(".parquet.tmp").exists()

print("Embedding atomic-write and resume tests passed")


class StubTiles:
    def __init__(self, slide_name: str) -> None:
        self.slide_path = Path(slide_name)

    def __len__(self) -> int:
        return 2


class FailingDataLoader:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("simulated decode error")


with TemporaryDirectory() as directory:
    excluded = [
        "2020_01089-02-T",
        "2020_01166-01-N",
        "2020_02743-02-T",
        "2020_06097-01-N",
    ]
    config = SimpleNamespace(
        dataset=SimpleNamespace(path="unused", exclude=excluded),
        output_dir=directory,
        tile_encoder=MagicMock(),
        precision="32",
        dataloader=SimpleNamespace(
            batch_size=1024, num_workers=4, persistent_workers=False
        ),
    )
    tiles = [StubTiles(name) for name in [*excluded, "good-slide"]]
    stdout = io.StringIO()
    with (
        patch(
            "preprocessing.embeddings.SlideDataset",
            return_value=SimpleNamespace(
                slides=range(len(tiles)), generate_datasets=lambda: tiles
            ),
        ),
        patch("preprocessing.embeddings.DataLoader", FailingDataLoader),
        patch("preprocessing.embeddings.hydra.utils.instantiate"),
        redirect_stdout(stdout),
        redirect_stderr(io.StringIO()),
    ):
        logger = MagicMock()
        try:
            unwrap(main)(config, logger)
            raise AssertionError("Expected a slide failure to fail the run")
        except RuntimeError as error:
            assert "good-slide" in str(error)

    assert stdout.getvalue().count("incomplete QC masks") == len(excluded)
    assert "simulated decode error" in stdout.getvalue()
    logger.log_artifacts.assert_not_called()

    save_embeddings(
        torch.zeros((2, 3)),
        torch.tensor([10, 20]),
        torch.tensor([30, 40]),
        Path(directory) / "good-slide.parquet",
    )
    progress = io.StringIO()
    with (
        patch(
            "preprocessing.embeddings.SlideDataset",
            return_value=SimpleNamespace(generate_datasets=lambda: tiles),
        ),
        patch("preprocessing.embeddings.DataLoader") as dataloader,
        patch("preprocessing.embeddings.hydra.utils.instantiate"),
        patch("preprocessing.embeddings.log_provenance"),
        redirect_stdout(stdout),
        redirect_stderr(progress),
    ):
        unwrap(main)(config, MagicMock())
        dataloader.assert_not_called()

    assert "1/1" in progress.getvalue()

    bad_config = SimpleNamespace(
        dataset=SimpleNamespace(path="unused", exclude=["unknown-slide"]),
        output_dir=directory,
        tile_encoder=MagicMock(),
        precision="32",
    )
    with (
        patch(
            "preprocessing.embeddings.SlideDataset",
            return_value=SimpleNamespace(generate_datasets=lambda: tiles),
        ),
        patch("preprocessing.embeddings.hydra.utils.instantiate") as instantiate,
    ):
        try:
            unwrap(main)(bad_config, MagicMock())
            raise AssertionError("Expected an unknown exclusion to fail the run")
        except ValueError as error:
            assert "unknown-slide" in str(error)
        instantiate.assert_not_called()

print("Embedding exclusion, resume-progress, and failure-propagation tests passed")
