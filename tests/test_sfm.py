from pathlib import Path
from unittest.mock import patch

import pytest

from sfm_mvs_pipeline.sfm.feature_extraction import extract_features
from sfm_mvs_pipeline.sfm.feature_matching import match_features

_EXTRACTION_OPTIONS = {
    "max_num_features": 1024,
    "first_octave": -1,
}

_MATCHING_OPTIONS = {
    "method": "exhaustive",
    "vocab_tree": {
        "vocab_tree_path": "",
        "num_nearest_neighbors": 100,
    },
}


def _place_images(directory: Path, n: int = 3) -> None:
    """Create n empty files with .png extension to satisfy the image-discovery step."""
    for i in range(n):
        (directory / f"frame_{i:03d}.png").touch()


class TestFeatureExtraction:
    def test_feature_extraction_runs(self, tmp_path):
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        _place_images(image_dir)
        database_path = tmp_path / "colmap.db"

        def _fake_extract(*args, **kwargs):
            database_path.touch()

        with patch("pycolmap.extract_features", side_effect=_fake_extract) as mock_fn:
            extract_features(database_path, image_dir, _EXTRACTION_OPTIONS)

        assert database_path.exists()
        mock_fn.assert_called_once()

    def test_feature_extraction_missing_image_dir(self, tmp_path):
        with pytest.raises(ValueError, match="image_dir does not exist"):
            extract_features(
                database_path=tmp_path / "colmap.db",
                image_dir=tmp_path / "nonexistent",
                options=_EXTRACTION_OPTIONS,
            )

    def test_feature_extraction_empty_image_dir(self, tmp_path):
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        with pytest.raises(ValueError, match="No images found"):
            extract_features(
                database_path=tmp_path / "colmap.db",
                image_dir=image_dir,
                options=_EXTRACTION_OPTIONS,
            )


class TestFeatureMatching:
    def test_feature_matching_missing_database(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="COLMAP database not found"):
            match_features(
                database_path=tmp_path / "nonexistent.db",
                options=_MATCHING_OPTIONS,
            )

    def test_feature_matching_exhaustive_runs(self, tmp_path):
        database_path = tmp_path / "colmap.db"
        database_path.touch()

        with patch("pycolmap.match_exhaustive") as mock_fn:
            match_features(database_path, _MATCHING_OPTIONS)

        mock_fn.assert_called_once_with(database_path=database_path)

    def test_feature_matching_unknown_method_raises(self, tmp_path):
        database_path = tmp_path / "colmap.db"
        database_path.touch()

        options = {**_MATCHING_OPTIONS, "method": "bogus"}
        with pytest.raises(ValueError, match="Unknown matching method"):
            match_features(database_path, options)
