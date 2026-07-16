from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sfm_mvs_pipeline.mvs.fusion import fuse_depth_maps
from sfm_mvs_pipeline.mvs.mask_undistortion import undistortion_maps

_FUSION_OPTIONS = {
    "min_num_pixels": 5,
    "max_reproj_error": 2.0,
}


def _camera(model_name: str, width: int, height: int, params: list[float]) -> MagicMock:
    camera = MagicMock()
    camera.model.name = model_name
    camera.width = width
    camera.height = height
    camera.params = params
    return camera


# --- undistortion_maps ---


def test_identity_when_no_distortion_and_same_intrinsics():
    cam = _camera("PINHOLE", 64, 48, [50.0, 50.0, 32.0, 24.0])
    map_x, map_y = undistortion_maps(cam, cam)

    u, v = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(48, dtype=np.float32))
    np.testing.assert_allclose(map_x, u, atol=1e-4)
    np.testing.assert_allclose(map_y, v, atol=1e-4)


def test_principal_point_is_fixed_point_of_radial_distortion():
    # At the principal point r=0, so distortion has no effect regardless of k.
    original = _camera("SIMPLE_RADIAL", 64, 48, [50.0, 32.0, 24.0, -0.2])
    undistorted = _camera("PINHOLE", 64, 48, [50.0, 50.0, 32.0, 24.0])
    map_x, map_y = undistortion_maps(original, undistorted)

    assert map_x[24, 32] == pytest.approx(32.0, abs=1e-4)
    assert map_y[24, 32] == pytest.approx(24.0, abs=1e-4)


def test_negative_k_pulls_border_samples_inward():
    # Barrel distortion (k < 0): the original image content is compressed
    # toward the centre, so undistorted border pixels must sample INSIDE
    # the original frame (map value < pixel coordinate at the right edge).
    original = _camera("SIMPLE_RADIAL", 64, 48, [50.0, 32.0, 24.0, -0.2])
    undistorted = _camera("PINHOLE", 64, 48, [50.0, 50.0, 32.0, 24.0])
    map_x, _ = undistortion_maps(original, undistorted)

    assert map_x[24, 63] < 63.0


def test_unsupported_model_raises():
    original = _camera("OPENCV_FISHEYE", 64, 48, [50.0, 50.0, 32.0, 24.0, 0, 0, 0, 0])
    undistorted = _camera("PINHOLE", 64, 48, [50.0, 50.0, 32.0, 24.0])
    with pytest.raises(ValueError, match="Unsupported camera model"):
        undistortion_maps(original, undistorted)


def test_distorted_target_camera_rejected():
    original = _camera("SIMPLE_RADIAL", 64, 48, [50.0, 32.0, 24.0, -0.2])
    not_pinhole = _camera("SIMPLE_RADIAL", 64, 48, [50.0, 32.0, 24.0, 0.1])
    with pytest.raises(ValueError, match="distortion-free"):
        undistortion_maps(original, not_pinhole)


# --- fusion mask wiring ---


@patch("sfm_mvs_pipeline.mvs.fusion.pycolmap.StereoFusionOptions")
@patch("sfm_mvs_pipeline.mvs.fusion.pycolmap.stereo_fusion")
def test_fusion_receives_mask_path(mock_stereo_fusion, mock_fusion_opts, tmp_path):
    mvs_path = tmp_path / "mvs"
    depth_maps_dir = mvs_path / "stereo" / "depth_maps"
    depth_maps_dir.mkdir(parents=True)
    (depth_maps_dir / "image0.photometric.bin").touch()
    mask_dir = mvs_path / "fusion_masks"
    mask_dir.mkdir()
    output_path = tmp_path / "dense.ply"

    def _create_ply(*args, **kwargs):
        output_path.touch()
        return MagicMock()

    mock_stereo_fusion.side_effect = _create_ply

    fuse_depth_maps(
        mvs_path=mvs_path,
        output_path=output_path,
        options=_FUSION_OPTIONS,
        mask_path=mask_dir,
    )

    assert mock_fusion_opts.return_value.mask_path == str(mask_dir)


@patch("sfm_mvs_pipeline.mvs.fusion.pycolmap.StereoFusionOptions")
@patch("sfm_mvs_pipeline.mvs.fusion.pycolmap.stereo_fusion")
def test_fusion_mask_path_not_set_by_default(mock_stereo_fusion, mock_fusion_opts, tmp_path):
    mvs_path = tmp_path / "mvs"
    depth_maps_dir = mvs_path / "stereo" / "depth_maps"
    depth_maps_dir.mkdir(parents=True)
    (depth_maps_dir / "image0.photometric.bin").touch()
    output_path = tmp_path / "dense.ply"

    def _create_ply(*args, **kwargs):
        output_path.touch()
        return MagicMock()

    mock_stereo_fusion.side_effect = _create_ply
    options_obj = mock_fusion_opts.return_value
    # Simulate the real attribute default so we can detect accidental writes.
    options_obj.mask_path = ""

    fuse_depth_maps(mvs_path=mvs_path, output_path=output_path, options=_FUSION_OPTIONS)

    assert options_obj.mask_path == ""
