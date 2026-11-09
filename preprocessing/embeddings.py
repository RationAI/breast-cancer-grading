import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import albumentations as A
import hydra
import pandas as pd
import pyarrow as pa
import timm
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader

from ml.data.datasets import SlideDataset


class Virchow2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # For this, you need to setup HF_TOKEN=<X> env.variable.
        self.module = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval()
        self.embed_dim: int = cast("int", self.module.embed_dim) * 2
        # class token + mean patch tokens

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = cast("torch.Tensor", self.module(x))  # size: B x 261 x 1280

        class_token = output[:, 0]  # size: B x 1280
        patch_tokens = output[
            :, 5:
        ]  # size: B x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: B x 2560


def save_embeddings(
    slide_tiles_embeddings: torch.Tensor,
    slide_tiles_x: torch.Tensor,
    slide_tiles_y: torch.Tensor,
    embeddings_path: Path,
) -> None:
    """Save the slide embeddings to the specified path.

    Args:
        slide_tiles_embeddings: The embeddings to save.
        slide_tiles_x: The x-coordinates of the tiles.
        slide_tiles_y: The y-coordinates of the tiles.
        embeddings_path: The path to save the embeddings to.
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "x": slide_tiles_x.numpy(),
            "y": slide_tiles_y.numpy(),
            "embedding": [emb.numpy() for emb in slide_tiles_embeddings],
        }
    )

    temporary_path = embeddings_path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(temporary_path, index=False, engine="pyarrow")
        temporary_path.replace(embeddings_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def embeddings_are_complete(embeddings_path: Path, expected_rows: int) -> bool:
    try:
        metadata = pa.parquet.ParquetFile(embeddings_path).metadata
        return metadata is not None and metadata.num_rows == expected_rows
    except (OSError, pa.ArrowException):
        return False


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog(log_stream=False)  # Keep slide progress visible in Kubernetes logs.
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dest = Path(config.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(config.precision) not in {"32", "bf16-mixed"}:
        raise ValueError("precision must be '32' or 'bf16-mixed'")
    mixed_precision = device.type == "cuda" and str(config.precision) == "bf16-mixed"

    dataset = SlideDataset(
        uris=[config.dataset.path],
        transforms=A.Compose(
            [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        ),
    )

    excluded_slides = set(config.dataset.exclude)
    all_tile_datasets = list(dataset.generate_datasets())
    missing_exclusions = excluded_slides - {
        tile_dataset.slide_path.stem for tile_dataset in all_tile_datasets
    }
    if missing_exclusions:
        raise ValueError(f"Excluded slides not found: {sorted(missing_exclusions)}")
    for slide_name in sorted(excluded_slides):
        print(f"Skipping {slide_name}: incomplete QC masks")
    tile_datasets = [
        tile_dataset
        for tile_dataset in all_tile_datasets
        if tile_dataset.slide_path.stem not in excluded_slides
    ]
    # Completed slides count toward progress; the rate estimate below only
    # counts slides embedded by this run.
    completed_slides = sum(
        embeddings_are_complete(
            (dest / tile_dataset.slide_path.stem).with_suffix(".parquet"),
            len(tile_dataset),
        )
        for tile_dataset in tile_datasets
    )
    failed_slides = []

    # Process each slide's tiles separately so embeddings can be saved per-slide.
    print(f"Slides: {completed_slides}/{len(tile_datasets)} already embedded")
    tile_encoder: Virchow2 = hydra.utils.instantiate(config.tile_encoder)
    tile_encoder = tile_encoder.to(device)
    done_slides = completed_slides
    run_started = time.monotonic()
    # Process each slide's tiles separately so embeddings can be saved per-slide.
    for tile_dataset in tile_datasets:
        slide_name = tile_dataset.slide_path.stem
        embeddings_path = (dest / slide_name).with_suffix(".parquet")

        if embeddings_are_complete(embeddings_path, len(tile_dataset)):
            continue
        try:
            dataloader = DataLoader(
                tile_dataset,
                batch_size=config.dataloader.batch_size,
                num_workers=config.dataloader.num_workers,
                persistent_workers=config.dataloader.persistent_workers,
                pin_memory=device.type == "cuda",
            )

            slide_tiles_embeddings = torch.zeros(
                (len(tile_dataset), tile_encoder.embed_dim), dtype=torch.float32
            )
            slide_tiles_x = torch.zeros((len(tile_dataset),), dtype=torch.int32)
            slide_tiles_y = torch.zeros((len(tile_dataset),), dtype=torch.int32)

            for i, (x, metadata) in enumerate(dataloader):
                x = x.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=mixed_precision,
                ):
                    embeddings = cast("torch.Tensor", tile_encoder(x))

                start = i * config.dataloader.batch_size
                end = start + embeddings.size(0)

                # Keep the output schema float32 in both precision modes.
                slide_tiles_embeddings[start:end] = embeddings.float().cpu()
                slide_tiles_x[start:end] = metadata["x"].to("cpu")
                slide_tiles_y[start:end] = metadata["y"].to("cpu")

            save_embeddings(
                slide_tiles_embeddings,
                slide_tiles_x,
                slide_tiles_y,
                embeddings_path,
            )

        except Exception as e:
            failed_slides.append(slide_name)
            print(f"Error processing slide {slide_name}: {e}")
        finally:
            done_slides += 1
            progress = f"{done_slides}/{len(tile_datasets)} slides done"
            if done_slides - completed_slides > 1:  # a rate from one slide is noise
                rate = (done_slides - completed_slides) / (
                    (time.monotonic() - run_started) / 3600
                )
                eta = datetime.now(UTC) + timedelta(
                    hours=(len(tile_datasets) - done_slides) / rate
                )
                progress += f", {rate:.1f} slides/h, ETA {eta:%Y-%m-%d %H:%M} UTC"
            print(progress, flush=True)

    if failed_slides:
        raise RuntimeError(f"Embedding failed for slides: {failed_slides}")

    logger.log_artifacts(str(dest), artifact_path="embeddings")

    log_provenance(
        logger=logger, config=config
    )  # links this run to user, git, image, inputs


if __name__ == "__main__":
    main()
