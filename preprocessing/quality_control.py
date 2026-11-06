"""QC mask generation using rationai-sdk AsyncClient.

Uses the new QC service (v0.2.3) with pre-computed tissue masks.
Skips slides that already have output masks.
"""

import asyncio
import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai import AsyncClient
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance
from rationai.types import SlideCheckConfig
from tqdm import tqdm

# One representative output file per slide — if it exists, the slide is done.
MARKER_DIR = "ResidualArtifactsAndCoverage/artifacts_coverage"
MARKER_FILE = "{stem}.tiff"


def download_tissue_masks(artifact_uri: str, local_dir: Path) -> None:
    """Download tissue masks from MLflow if not already present locally."""
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Tissue masks already present in {local_dir}, skipping download.")
        return
    # Download to parent so the artifact's directory name becomes local_dir
    local_dir.parent.mkdir(exist_ok=True, parents=True)
    print(f"Downloading tissue masks from {artifact_uri}...")
    mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=str(local_dir.parent))
    print(f"Downloaded {len(list(local_dir.iterdir()))} tissue masks.")


def filter_unprocessed(slides: list[Path], output_path: Path) -> list[Path]:
    """Return only slides that don't already have output masks."""
    marker_dir = output_path / MARKER_DIR
    unprocessed = []
    for slide in slides:
        marker = marker_dir / MARKER_FILE.format(stem=slide.stem)
        if not marker.exists():
            unprocessed.append(slide)
    return unprocessed


async def qc_main(
    config: DictConfig,
    slides: list[Path],
    report_path: str,
    logger: MLFlowLogger,
) -> None:
    """Run QC checks on all slides and generate a report."""
    output_path = Path(config.output_path)
    url = config.url

    # Download tissue masks from MLflow if available
    tissue_mask_uri = config.get("tissue_mask_artifact_uri")
    if tissue_mask_uri:
        download_tissue_masks(tissue_mask_uri, Path(config.tissue_mask_dir))

    # Skip already-processed slides
    slides = filter_unprocessed(slides, output_path)
    if not slides:
        print("All slides already processed. Nothing to do.")
        return

    print(f"Processing {len(slides)} slides")

    check_config = SlideCheckConfig(
        check_residual=config.check_residual,
        check_folding=config.check_folding,
        check_blur=config.check_blur,
        tissue_mask_dir=config.get("tissue_mask_dir"),
    )

    async with AsyncClient(qc_base_url=url, timeout=config.request_timeout) as client:
        pbar = tqdm(total=len(slides), desc="QC slides", unit="slide")
        results = []
        async for result in client.qc.check_slides(
            wsi_paths=slides,
            output_path=output_path,
            config=check_config,
            timeout=config.request_timeout,
            max_concurrent=config.request_limit,
        ):
            pbar.update(1)
            results.append(result)
        pbar.close()

        # Log failures
        failed = [r for r in results if not r.success]
        if failed:
            print(f"\n{len(failed)}/{len(results)} slides failed:")
            for r in failed:
                print(f"  {r.wsi_path}: {r.error}")

        # Generate report
        await client.qc.generate_report(
            backgrounds=slides,
            mask_dir=output_path,
            save_location=report_path,
            compute_metrics=True,
        )

        try:
            logger.experiment.log_artifacts(
                run_id=logger.run_id, local_dir=Path(report_path).parent.as_posix()
            )
        except Exception:
            print(f"Could not upload report artifact from {report_path}")


@with_cli_args(["+preprocessing=qc_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    """Entry point for quality control preprocessing."""
    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(config.dataset.paths.data_mapping)
    slides = [Path(path) for path in df["path"]]

    with tempfile.TemporaryDirectory(
        prefix="qc_masks_report", dir=Path(config.project_path).as_posix()
    ) as tmp_dir:
        report_path = Path(tmp_dir) / "report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        asyncio.run(
            qc_main(
                config=config,
                slides=slides,
                report_path=report_path.absolute().as_posix(),
                logger=logger,
            )
        )

    log_provenance(logger=logger, config=config)


if __name__ == "__main__":
    main()
