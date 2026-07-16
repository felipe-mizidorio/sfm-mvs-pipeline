"""Warp per-image masks from original-frame space into the undistorted MVS workspace.

Stereo fusion (``StereoFusionOptions.mask_path``) reads masks aligned to the
*undistorted* workspace images (``mvs/images/<name>``), while preprocessing
generates masks on the original frames. A mask misaligned at the head contour
would defeat its purpose, so each undistorted mask pixel is mapped through the
original camera's distortion model and sampled with nearest-neighbour (masks
stay strictly binary).

COLMAP mask convention throughout: ``<image filename>.png``, white (255) keeps,
black (0) discards. Pixels that fall outside the original frame (blank border
regions introduced by undistortion) are set to 0 — they carry no image content.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pycolmap

logger = logging.getLogger(__name__)

# Camera models with params (f, cx, cy) or (fx, fy, cx, cy) and radial k terms
# that this module knows how to invert. Extend deliberately, not implicitly.
_SUPPORTED_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
}


def _unpack_intrinsics(camera: pycolmap.Camera) -> tuple[float, float, float, float, list[float]]:
    """Return (fx, fy, cx, cy, radial_ks) for a supported camera model."""
    model = camera.model.name
    p = [float(v) for v in camera.params]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return f, f, cx, cy, []
    if model == "PINHOLE":
        fx, fy, cx, cy = p
        return fx, fy, cx, cy, []
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        return f, f, cx, cy, [k]
    if model == "RADIAL":
        f, cx, cy, k1, k2 = p
        return f, f, cx, cy, [k1, k2]
    raise ValueError(
        f"Unsupported camera model for mask undistortion: {model} "
        f"(supported: {sorted(_SUPPORTED_MODELS)})"
    )


def undistortion_maps(
    original_camera: pycolmap.Camera,
    undistorted_camera: pycolmap.Camera,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel maps (map_x, map_y) from undistorted image coords into original coords.

    For every pixel of the undistorted image: normalize through the (pinhole)
    undistorted intrinsics, apply the original model's radial distortion, then
    project through the original intrinsics. Suitable for ``cv2.remap``.
    """
    fx_u, fy_u, cx_u, cy_u, ks_u = _unpack_intrinsics(undistorted_camera)
    if ks_u and any(k != 0.0 for k in ks_u):
        raise ValueError("undistorted camera must be distortion-free")
    fx_o, fy_o, cx_o, cy_o, ks_o = _unpack_intrinsics(original_camera)

    w, h = int(undistorted_camera.width), int(undistorted_camera.height)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - cx_u) / fx_u
    y = (v - cy_u) / fy_u

    r2 = x * x + y * y
    radial = np.ones_like(r2)
    r_pow = r2
    for k in ks_o:
        radial += k * r_pow
        r_pow = r_pow * r2

    map_x = (x * radial * fx_o + cx_o).astype(np.float32)
    map_y = (y * radial * fy_o + cy_o).astype(np.float32)
    return map_x, map_y


def undistort_masks(
    mask_path: Path,
    original_sparse_path: Path,
    mvs_path: Path,
    output_dir_name: str = "fusion_masks",
) -> tuple[Path, dict]:
    """Warp all masks for registered images into the MVS workspace.

    Args:
        mask_path: Directory of original-frame masks (COLMAP ``<name>.png``).
        original_sparse_path: Sparse model with the ORIGINAL (distorted) cameras.
        mvs_path: MVS workspace (contains ``sparse/`` with undistorted cameras).

    Returns:
        (output directory, stats dict).
    """
    original_rec = pycolmap.Reconstruction(str(original_sparse_path))
    mvs_rec = pycolmap.Reconstruction(str(mvs_path / "sparse"))
    out_dir = mvs_path / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    original_by_name = {im.name: im for im in original_rec.images.values()}

    maps_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    written = 0
    missing = 0
    for image in mvs_rec.images.values():
        src = mask_path / f"{image.name}.png"
        mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning("Mask not found for '%s' — skipping.", image.name)
            missing += 1
            continue
        original_image = original_by_name.get(image.name)
        if original_image is None:
            logger.warning("No original camera for '%s' — skipping.", image.name)
            missing += 1
            continue

        key = (original_image.camera_id, image.camera_id)
        if key not in maps_cache:
            maps_cache[key] = undistortion_maps(
                original_rec.cameras[original_image.camera_id],
                mvs_rec.cameras[image.camera_id],
            )
        map_x, map_y = maps_cache[key]

        warped = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cv2.imwrite(str(out_dir / f"{image.name}.png"), warped)
        written += 1

    stats = {"masks_written": written, "masks_missing": missing}
    logger.info(
        "Fusion masks: %d written, %d missing -> '%s'", written, missing, out_dir
    )
    return out_dir, stats
