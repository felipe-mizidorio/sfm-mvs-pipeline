import logging
from pathlib import Path

import pycolmap

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def camera_prior_from_manifest(manifest_data: dict) -> tuple[str, str] | None:
    """Derive an intrinsics prior from the frames manifest's camera block.

    Uses the 35mm-equivalent focal length recorded by preprocessing:
    fx ≈ f35 × image_width_px / 36. Returns (camera_model, camera_params)
    suitable for extract_features, or None when the manifest carries no
    usable focal metadata (the common case: messaging apps strip it).

    The params act as a bundle-adjustment initialization, not a fixed
    calibration — COLMAP still refines focal length during mapping.
    """
    camera = manifest_data.get("camera") or {}
    f35 = camera.get("focal_length_35mm")
    width = camera.get("width_px")
    height = camera.get("height_px")
    if not f35 or not width or not height:
        return None
    fx = float(f35) * float(width) / 36.0
    params = f"{fx},{float(width) / 2.0},{float(height) / 2.0},0"
    return "SIMPLE_RADIAL", params


def extract_features(
    database_path: Path,
    image_dir: Path,
    options: dict,
    device: pycolmap.Device = pycolmap.Device.auto,
    camera_model: str | None = None,
    camera_params: str | None = None,
    image_names: list[str] | None = None,
    mask_path: Path | None = None,
    shared_camera: bool = False,
) -> None:
    if not image_dir.exists():
        raise ValueError(f"image_dir does not exist: {image_dir}")

    if image_names is not None:
        images = image_names
        if not images:
            raise ValueError(f"No images in manifest for {image_dir}")
    else:
        images = [
            p
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        if not images:
            raise ValueError(f"No images found in {image_dir}")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = options["max_num_features"]
    extraction_options.sift.first_octave = options["first_octave"]

    reader_options = pycolmap.ImageReaderOptions()
    if camera_model is not None:
        reader_options.camera_model = camera_model
    if camera_params is not None:
        reader_options.camera_params = camera_params
    if mask_path is not None:
        reader_options.mask_path = mask_path
        logger.info("Using mask directory: '%s'", mask_path)
    # One shared camera for same-device video frames (explicit calibration
    # always implies it); AUTO's per-image self-calibration is the legacy path.
    camera_mode = (
        pycolmap.CameraMode.SINGLE
        if camera_model is not None or shared_camera
        else pycolmap.CameraMode.AUTO
    )

    logger.info(
        "Extracting features from %d images in '%s' into '%s'%s",
        len(images),
        image_dir,
        database_path,
        f" (camera_model={camera_model})" if camera_model else "",
    )

    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        image_names=image_names or [],
        camera_mode=camera_mode,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=device,
    )

    logger.info("Feature extraction complete: %s", database_path)
