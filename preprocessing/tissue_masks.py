from pathlib import Path
from typing import cast

import hydra
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, tissue_mask, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance


@ray.remote(memory=3 * 1024**3)
def process_slide(slide_path: Path, level: int, output_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=level)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))
    mask = tissue_mask(slide, mpp=(mpp_x + mpp_y) / 2)
    mask_path = output_path / (slide_path.stem + ".tiff")

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


@with_cli_args(["+preprocessing=tissue_masks"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = pd.read_csv(config.dataset.paths.data_mapping)
    slides = [Path(path) for path in df["path"]]  # paths already include .mrxs

    output_dir = Path(config.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Skip already-processed slides
    unprocessed = [s for s in slides if not (output_dir / (s.stem + ".tiff")).exists()]
    if not unprocessed:
        print("All tissue masks already exist. Nothing to do.")
        return
    print(f"Processing {len(unprocessed)}/{len(slides)} slides")

    process_items(
        unprocessed,
        process_item=process_slide,
        fn_kwargs={
            "level": config.level,
            "output_path": output_dir,
        },
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(
        local_dir=str(output_dir), artifact_path=config.mlflow_artifact_path
    )

    log_provenance(
        logger=logger, config=config
    )  # links this run to user, git, image, inputs


if __name__ == "__main__":
    main()
