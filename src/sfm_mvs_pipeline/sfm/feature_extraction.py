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
) -> None:
    if not image_dir.exists():
        raise ValueError(f"image_dir does not exist: {image_dir}")

    images = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = options["max_num_features"]
    extraction_options.sift.first_octave = options["first_octave"]

    logger.info(
        "Extracting features from %d images in '%s' into '%s'",
        len(images),
        image_dir,
        database_path,
    )

    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        extraction_options=extraction_options,
        device=device,
    )

    logger.info("Feature extraction complete: %s", database_path)
