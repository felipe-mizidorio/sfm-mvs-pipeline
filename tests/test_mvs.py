from unittest.mock import MagicMock, patch

import pytest

from sfm_mvs_pipeline.mvs.dense_reconstruction import run_dense_reconstruction
from sfm_mvs_pipeline.mvs.fusion import fuse_depth_maps

_DENSE_OPTIONS = {
    "max_image_size": 2000,
    "window_radius": 5,
    "num_samples": 15,
}

_FUSION_OPTIONS = {
    "min_num_pixels": 5,
    "max_reproj_error": 2.0,
}


@patch("sfm_mvs_pipeline.mvs.dense_reconstruction.pycolmap.patch_match_stereo")
@patch("sfm_mvs_pipeline.mvs.dense_reconstruction.pycolmap.PatchMatchOptions")
@patch("sfm_mvs_pipeline.mvs.dense_reconstruction.pycolmap.undistort_images")
def test_dense_reconstruction_runs(mock_undistort, mock_patch_match_opts, mock_pms, tmp_path):
    sparse_path = tmp_path / "sparse"
    sparse_path.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    run_dense_reconstruction(
        sparse_path=sparse_path,
        image_dir=image_dir,
        mvs_path=tmp_path / "mvs",
        options=_DENSE_OPTIONS,
    )

    mock_undistort.assert_called_once()
    mock_pms.assert_called_once()


def test_dense_reconstruction_missing_sparse_path(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Sparse reconstruction not found"):
        run_dense_reconstruction(
            sparse_path=tmp_path / "nonexistent_sparse",
            image_dir=image_dir,
            mvs_path=tmp_path / "mvs",
            options=_DENSE_OPTIONS,
        )


@patch("sfm_mvs_pipeline.mvs.fusion.pycolmap.StereoFusionOptions")
@patch("sfm_mvs_pipeline.mvs.fusion.pycolmap.stereo_fusion")
def test_stereo_fusion_runs(mock_stereo_fusion, mock_fusion_opts, tmp_path):
    mvs_path = tmp_path / "mvs"
    depth_maps_dir = mvs_path / "stereo" / "depth_maps"
    depth_maps_dir.mkdir(parents=True)
    (depth_maps_dir / "image0.photometric.bin").touch()

    output_path = tmp_path / "dense.ply"

    def _create_ply(*args, **kwargs):
        output_path.touch()
        return MagicMock()

    mock_stereo_fusion.side_effect = _create_ply

    fuse_depth_maps(mvs_path=mvs_path, output_path=output_path, options=_FUSION_OPTIONS)

    mock_stereo_fusion.assert_called_once()
    _, call_kwargs = mock_stereo_fusion.call_args
    assert call_kwargs["output_path"] == output_path


def test_stereo_fusion_missing_mvs_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        fuse_depth_maps(
            mvs_path=tmp_path / "nonexistent_mvs",
            output_path=tmp_path / "dense.ply",
            options=_FUSION_OPTIONS,
        )
