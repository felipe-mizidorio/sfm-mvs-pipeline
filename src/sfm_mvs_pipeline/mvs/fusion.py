import logging
from pathlib import Path

import pycolmap

logger = logging.getLogger(__name__)

# Depth maps written by patch_match_stereo live here inside the MVS workspace.
_DEPTH_MAPS_DIR = "stereo/depth_maps"


def fuse_depth_maps(
    mvs_path: Path,
    output_path: Path,
    options: dict,
) -> pycolmap.Reconstruction:
    depth_maps_dir = mvs_path / _DEPTH_MAPS_DIR
    if not depth_maps_dir.exists() or not any(depth_maps_dir.iterdir()):
        raise FileNotFoundError(
            f"No depth map outputs found in '{depth_maps_dir}'. "
            "Run dense_reconstruction.run_dense_reconstruction() first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fusion_options = pycolmap.StereoFusionOptions()
    fusion_options.min_num_pixels = options["min_num_pixels"]
    fusion_options.max_reproj_error = options["max_reproj_error"]

    logger.info(
        "Fusing depth maps from '%s' into '%s' "
        "(min_num_pixels=%d, max_reproj_error=%.2f)",
        mvs_path,
        output_path,
        options["min_num_pixels"],
        options["max_reproj_error"],
    )

    reconstruction = pycolmap.stereo_fusion(
        output_path=output_path,
        workspace_path=mvs_path,
        options=fusion_options,
        output_type="PLY",
    )

    if not output_path.exists():
        raise RuntimeError(
            f"stereo_fusion ran but output file was not created: {output_path}"
        )

    logger.info(
        "Stereo fusion complete: '%s' (%.2f MB)",
        output_path,
        output_path.stat().st_size / 1_048_576,
    )

    return reconstruction
