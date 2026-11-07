"""Script for creating slides and tiles datasets for grading using Ray Data."""

from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import ray
import ray.data as rd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles
from ratiopath.tiling.overlays import _tile_overlay
from ratiopath.tiling.utils import row_hash
from ray.data import Dataset
from ray.data.block import DataBatch
from ray.data.expressions import col
from shapely.geometry import box


BATCH_SIZE = 256


def _overlap_fraction(masked: np.ma.MaskedArray, field: str) -> float:
    """Compute overlap = 1 - fraction of pixels matching the field value."""
    valid = masked.count()
    if valid == 0:
        return 0.0
    field_val = int(field)
    count = (masked.compressed() == field_val).sum()
    return 1.0 - (count / valid)


def compute_overlaps_batch(
    batch: DataBatch,
    tissue_roi: Any,
    full_roi: Any,
) -> DataBatch:
    """Compute ALL mask overlaps for a batch of tiles in one pass.

    Instead of chaining 4 sequential Ray Dataset stages (tissue, blur, folding, residual),
    this single UDF reads all masks per tile and computes overlap scores in one shot.
    """
    # Normalize to pandas (Ray may yield pyarrow Table even with batch_format="pandas")
    if isinstance(batch, pa.Table):
        batch = batch.to_pandas()

    tile_x = pa.array(batch["tile_x"])
    tile_y = pa.array(batch["tile_y"])
    mpp_x = pa.array(batch["mpp_x"])
    mpp_y = pa.array(batch["mpp_y"])

    for path_col, roi, out_col, field in [
        ("tissue_mask_path", tissue_roi, "tissue_overlap", "0"),
        ("blur_mask_path", full_roi, "blur_overlap", "255"),
        ("folding_mask_path", full_roi, "folding_overlap", "255"),
        ("residual_mask_path", full_roi, "residual_overlap", "0"),
    ]:
        paths = pa.array(batch[path_col])
        # Skip empty paths
        if all(p == "" or p is None for p in paths.to_pylist()):
            batch[out_col] = np.zeros(len(batch["tile_x"]), dtype=np.float32)
            continue

        with_overlay = _tile_overlay(roi, paths, tile_x, tile_y, mpp_x, mpp_y)
        overlaps = np.array(
            [_overlap_fraction(m, field) for m in with_overlay],
            dtype=np.float32,
        )
        batch[out_col] = overlaps

    return batch


def tiling(
    row: dict[str, Any],
    tissue_masks_path: str | None,
    qc_masks_path: str | None,
    epithelium_masks_path: str | None,
) -> list[dict[str, Any]]:
    """Generate tile coordinates for a single slide."""
    slide_path = Path(row["path"])
    tiff_slide_name = slide_path.with_suffix(".tiff").name

    tissue_mask = (
        str(Path(tissue_masks_path) / tiff_slide_name) if tissue_masks_path else ""
    )

    epithelium_mask = (
        str(Path(epithelium_masks_path) / tiff_slide_name)
        if epithelium_masks_path
        else ""
    )

    if qc_masks_path:
        qc_path = Path(qc_masks_path)
        blur_mask = str(qc_path / "Piqe/blur_score_per_pixel" / tiff_slide_name)
        folding_mask = str(qc_path / "FoldingFunction/folding_per_pixel" / tiff_slide_name)
        residual_mask = str(qc_path / "ResidualArtifactsAndCoverage/artifacts_coverage" / tiff_slide_name)
    else:
        blur_mask = ""
        folding_mask = ""
        residual_mask = ""

    return [
        {
            "tile_x": x,
            "tile_y": y,
            "slide_id": row["id"],
            "mpp_x": row["mpp_x"],
            "mpp_y": row["mpp_y"],
            "tissue_mask_path": tissue_mask,
            "epithelium_mask_path": epithelium_mask,
            "blur_mask_path": blur_mask,
            "folding_mask_path": folding_mask,
            "residual_mask_path": residual_mask,
            "grade": row["grade"],
            "case": row["case"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if not ray.is_initialized():
        ray.init()

    tissue_masks_path = (
        None
        if config.dataset.mlflow_uris.tissue_masks is None
        else download_artifacts(config.dataset.mlflow_uris.tissue_masks)
    )
    epithelium_masks_path = (
        None
        if config.dataset.mlflow_uris.epithelium_masks is None
        else download_artifacts(config.dataset.mlflow_uris.epithelium_masks)
    )
    qc_masks_path = config.dataset.paths.qc_masks or None

    # Read slide paths from data mapping
    slide_labels_df = pd.read_csv(config.dataset.paths.data_mapping)
    slide_paths = slide_labels_df["path"].tolist()
    metadata_ds = rd.from_pandas(slide_labels_df[["path", "grade", "case"]])

    print(f"\n[INFO] Processing {len(slide_paths)} slides using Ray Data pipeline\n")

    # Read slides
    slides = read_slides(
        slide_paths,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    ).join(
        metadata_ds,
        on=("path",),
        right_on=("path",),
        num_partitions=16,
        join_type="left_outer",
    )

    # Create unique slide IDs and save slide metadata
    slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)

    # Expand slides into tile coordinates (no repartition — map_batches handles batching)
    tiles = slides.flat_map(
        partial(
            tiling,
            tissue_masks_path=tissue_masks_path,
            qc_masks_path=qc_masks_path,
            epithelium_masks_path=epithelium_masks_path,
        ),
        num_cpus=0.2,
        memory=768 * 1024**2,
    )

    # ROI definitions
    offset = config.tile_extent // 4
    size = config.tile_extent // 2
    tissue_roi = box(offset, offset, offset + size, offset + size)
    full_roi = box(0, 0, config.tile_extent, config.tile_extent)

    # Single-pass: compute ALL overlaps in one UDF per batch
    tiles = tiles.map_batches(
        partial(compute_overlaps_batch, tissue_roi=tissue_roi, full_roi=full_roi),
        batch_format="pandas",
        batch_size=BATCH_SIZE,
        num_cpus=1,
        memory=BATCH_SIZE * 8 * 512**2,  # ~512MB per worker
    )

    # Filter tiles with no tissue
    tiles = tiles.filter(expr=col("tissue_overlap") > 0.0)

    # Add epithelium overlap if configured
    if config.dataset.mlflow_uris.epithelium_masks is not None:
        epi_roi = full_roi
        tiles = tiles.map_batches(
            partial(compute_overlaps_batch, tissue_roi=epi_roi, full_roi=epi_roi),
            batch_format="pandas",
            batch_size=BATCH_SIZE,
            num_cpus=1,
            memory=BATCH_SIZE * 8 * 512**2,
        )

    # Drop mask path columns no longer needed
    drop_cols = [
        "tissue_mask_path",
        "epithelium_mask_path",
        "blur_mask_path",
        "folding_mask_path",
        "residual_mask_path",
        "mpp_x",
        "mpp_y",
    ]
    tiles = tiles.drop_columns(drop_cols)

    # Convert Ray Datasets to pandas DataFrames
    slides_df = slides.to_pandas()
    tiles_df = tiles.to_pandas()

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.data_name,
    )

    print(
        f"\n[INFO] Generated {len(slides_df)} slide records and {len(tiles_df)} tile records\n"
    )

    ray.shutdown()

    log_provenance(
        logger=logger, config=config
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
