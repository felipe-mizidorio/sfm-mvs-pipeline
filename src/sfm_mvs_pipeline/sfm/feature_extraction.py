import logging
from pathlib import Path

import pycolmap

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def extract_features(
    database_path: Path,
    image_dir: Path,
    options: dict,
    device: pycolmap.Device = pycolmap.Device.auto,
    camera_model: str | None = None,
    camera_params: str | None = None,
    image_names: list[str] | None = None,
    mask_path: Path | None = None,
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
    camera_mode = (
        pycolmap.CameraMode.SINGLE if camera_model is not None else pycolmap.CameraMode.AUTO
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
