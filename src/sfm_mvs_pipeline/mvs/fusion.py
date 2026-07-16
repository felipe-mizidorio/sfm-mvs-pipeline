import logging
from pathlib import Path

import numpy as np
import pycolmap

logger = logging.getLogger(__name__)

# Depth maps written by patch_match_stereo live here inside the MVS workspace.
_DEPTH_MAPS_DIR = "stereo/depth_maps"


def fuse_depth_maps(
    mvs_path: Path,
    output_path: Path,
    options: dict,
    bbox_min: list[float] | None = None,
    bbox_max: list[float] | None = None,
    mask_path: Path | None = None,
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

    if mask_path is not None:
        # Masks must be aligned to the UNDISTORTED workspace images (see
        # mvs/mask_undistortion.py). Only pixels with mask value > 0 may
        # contribute fused points.
        fusion_options.mask_path = str(mask_path)
        logger.info("Fusion mask directory: '%s'", mask_path)

    if bbox_min is not None and bbox_max is not None:
        # pycolmap expects float32 arrays of shape (3, 1); pyright cannot prove
        # the reshape produces that static shape.
        fusion_options.bounding_box = (  # pyright: ignore[reportAttributeAccessIssue]
            np.array(bbox_min, dtype=np.float32).reshape(3, 1),
            np.array(bbox_max, dtype=np.float32).reshape(3, 1),
        )
        logger.info("Bounding box clipping: min=%s, max=%s", bbox_min, bbox_max)

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
