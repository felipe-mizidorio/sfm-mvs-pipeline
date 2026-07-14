from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sfm_mvs_pipeline.sfm.reconstruction import load_best_reconstruction


def _mock_reconstruction_for(path_str: str, counts: dict[str, int]) -> MagicMock:
    name = Path(path_str).name
    mock = MagicMock()
    mock.num_reg_images.return_value = counts[name]
    return mock


def test_load_best_reconstruction_returns_most_registered_images(tmp_path):
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    counts = {"0": 5, "1": 12, "2": 8}
    for name in counts:
        (sparse_dir / name).mkdir()

    def _side_effect(path_str):
        return _mock_reconstruction_for(path_str, counts)

    with patch(
        "sfm_mvs_pipeline.sfm.reconstruction.pycolmap.Reconstruction",
        side_effect=_side_effect,
    ) as mock_ctor:
        reconstruction, best_path = load_best_reconstruction(sparse_dir)

    assert best_path == sparse_dir / "1"
    assert reconstruction.num_reg_images() == 12
    assert mock_ctor.call_count == 3


def test_load_best_reconstruction_single_model(tmp_path):
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    (sparse_dir / "0").mkdir()
    counts = {"0": 7}

    def _side_effect(path_str):
        return _mock_reconstruction_for(path_str, counts)

    with patch(
        "sfm_mvs_pipeline.sfm.reconstruction.pycolmap.Reconstruction",
        side_effect=_side_effect,
    ):
        reconstruction, best_path = load_best_reconstruction(sparse_dir)

    assert best_path == sparse_dir / "0"
    assert reconstruction.num_reg_images() == 7


def test_load_best_reconstruction_raises_on_empty_sparse_dir(tmp_path):
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No sparse models found"):
        load_best_reconstruction(sparse_dir)


def test_load_best_reconstruction_raises_on_missing_sparse_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_best_reconstruction(tmp_path / "nonexistent")
