"""
Dense reconstruction via image undistortion and PatchMatch Stereo.

Note: ``patch_match_stereo`` requires a CUDA-enabled build of pycolmap.
Install the ``gpu`` dependency group to get ``pycolmap-cuda12`` instead of
the CPU-only ``pycolmap`` package.
"""

import logging
from pathlib import Path

import pycolmap

logger = logging.getLogger(__name__)


def run_dense_reconstruction(
    sparse_path: Path,
    image_dir: Path,
    mvs_path: Path,
    options: dict,
) -> None:
    if not sparse_path.exists():
        raise FileNotFoundError(f"Sparse reconstruction not found: {sparse_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    mvs_path.mkdir(parents=True, exist_ok=True)

    logger.info("Undistorting images into MVS workspace '%s'", mvs_path)
    pycolmap.undistort_images(
        output_path=mvs_path,
        input_path=sparse_path,
        image_path=image_dir,
    )

    patch_match_options = pycolmap.PatchMatchOptions()
    patch_match_options.max_image_size = options["max_image_size"]
    patch_match_options.window_radius = options["window_radius"]
    patch_match_options.num_samples = options["num_samples"]

    logger.info(
        "Running PatchMatch Stereo (max_image_size=%d, window_radius=%d, num_samples=%d)",
        options["max_image_size"],
        options["window_radius"],
        options["num_samples"],
    )
    pycolmap.patch_match_stereo(
        workspace_path=mvs_path,
        options=patch_match_options,
    )

    logger.info("Dense reconstruction complete: depth maps written to '%s'", mvs_path)
