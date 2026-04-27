import logging
from pathlib import Path

import pycolmap

logger = logging.getLogger(__name__)


def run_incremental_mapping(
    database_path: Path,
    image_dir: Path,
    output_path: Path,
    options: dict,
    device: pycolmap.Device = pycolmap.Device.auto,
) -> dict[int, pycolmap.Reconstruction]:
    if not database_path.exists():
        raise FileNotFoundError(f"COLMAP database not found: {database_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    output_path.mkdir(parents=True, exist_ok=True)

    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.min_num_matches = options["min_num_matches"]
    mapping_options.max_num_models = options["max_num_models"]

    logger.info(
        "Running incremental mapping (min_num_matches=%d, max_num_models=%d)",
        options["min_num_matches"],
        options["max_num_models"],
    )

    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=output_path,
        options=mapping_options,
    )

    logger.info(
        "Incremental mapping complete: %d model(s) reconstructed, saved to '%s'",
        len(reconstructions),
        output_path,
    )

    return reconstructions
