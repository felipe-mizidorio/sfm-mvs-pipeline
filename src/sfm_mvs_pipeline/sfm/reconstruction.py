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
    mapping_options.ba_use_gpu = device != pycolmap.Device.cpu

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


def load_best_reconstruction(sparse_dir: Path) -> tuple[pycolmap.Reconstruction, Path]:
    if not sparse_dir.exists():
        raise FileNotFoundError(f"sparse_dir does not exist: {sparse_dir}")

    model_paths = sorted(sparse_dir.iterdir(), key=lambda p: int(p.name))
    if not model_paths:
        raise FileNotFoundError(f"No sparse models found under '{sparse_dir}'")

    best_path = model_paths[0]
    best_recon = pycolmap.Reconstruction(str(best_path))
    best_count = best_recon.num_reg_images()

    for path in model_paths[1:]:
        recon = pycolmap.Reconstruction(str(path))
        count = recon.num_reg_images()
        if count > best_count:
            best_recon, best_path, best_count = recon, path, count

    logger.info(
        "Selected sparse model '%s' with %d registered images (of %d candidate model(s))",
        best_path,
        best_count,
        len(model_paths),
    )
    return best_recon, best_path
