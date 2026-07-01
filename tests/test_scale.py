"""Tests for metric scale recovery from ArUco markers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest

from sfm_mvs_pipeline.scale.aruco_scale import (
    apply_scale_to_mesh,
    apply_scale_to_ply,
    recover_scale,
    recover_scale_safe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_cube_pcd(path: Path) -> np.ndarray:
    """Write an 8-point unit-cube point cloud to path. Returns the points array."""
    pts = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(path), pcd)
    return pts


def _write_cube_mesh(path: Path) -> np.ndarray:
    """Write a unit-sphere mesh (vertices near unit scale) to path."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
    verts = np.asarray(mesh.vertices).copy()
    o3d.io.write_triangle_mesh(str(path), mesh)
    return verts


# ---------------------------------------------------------------------------
# apply_scale_to_ply
# ---------------------------------------------------------------------------

def test_apply_scale_to_ply_doubles_coordinates(tmp_path):
    ply = tmp_path / "cloud.ply"
    original = _write_cube_pcd(ply)

    apply_scale_to_ply(ply, 2.0)

    result = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
    np.testing.assert_allclose(result, original * 2.0, rtol=1e-5)


def test_apply_scale_to_ply_preserves_shape(tmp_path):
    ply = tmp_path / "cloud.ply"
    original = _write_cube_pcd(ply)
    apply_scale_to_ply(ply, 0.5)
    result = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
    assert result.shape == original.shape


# ---------------------------------------------------------------------------
# apply_scale_to_mesh
# ---------------------------------------------------------------------------

def test_apply_scale_to_mesh_doubles_vertices(tmp_path):
    ply = tmp_path / "mesh.ply"
    original = _write_cube_mesh(ply)

    apply_scale_to_mesh(ply, 2.0)

    result = np.asarray(o3d.io.read_triangle_mesh(str(ply)).vertices)
    np.testing.assert_allclose(result, original * 2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# recover_scale — synthetic reconstruction
# ---------------------------------------------------------------------------

def _make_mock_reconstruction(
    marker_world: np.ndarray,
    scale_factor: float,
) -> tuple[MagicMock, dict[str, list[dict]]]:
    """Build a mock Reconstruction with two cameras observing a marker.

    Places two cameras looking at the origin along -Z, 1 unit apart on X.
    The marker corners are specified in world coordinates. Returns
    (mock_reconstruction, detections_dict) where detections contain the
    projected 2D coordinates.

    Args:
        marker_world: (4, 3) array of marker corner world positions in
            *reconstruction units* (not mm). The known physical size is
            derived from scale_factor.
        scale_factor: mm per reconstruction unit (ground-truth scale we
            want recover_scale to return).
    """
    # Two camera positions in world space (reconstruction units).
    cam_positions = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])

    # Simple pinhole intrinsics (fx=fy=500, cx=cy=320).
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    detections: dict[str, list[dict]] = {}
    mock_images: dict[int, MagicMock] = {}
    mock_cameras: dict[int, MagicMock] = {}

    for img_id, cam_pos in enumerate(cam_positions):
        # cam_from_world: R = I, t = -R @ cam_pos = -cam_pos (since R=I)
        R = np.eye(3)
        t = -R @ cam_pos

        # Project marker corners.
        proj_corners = []
        for pt3d in marker_world:
            pc = R @ pt3d + t
            u = fx * pc[0] / pc[2] + cx
            v = fy * pc[1] / pc[2] + cy
            proj_corners.append([float(u), float(v)])

        image_name = f"frame_{img_id:03d}.jpg"
        detections[image_name] = [{"id": 0, "corners": proj_corners}]

        # Mock camera.
        mock_cam = MagicMock()
        mock_cam.calibration_matrix.return_value = K.tolist()
        mock_cameras[img_id] = mock_cam

        # Mock image.
        mock_img = MagicMock()
        mock_img.image_id = img_id
        mock_img.camera_id = img_id
        mock_img.name = image_name
        mock_img.has_pose = True
        mock_img.cam_from_world.rotation.matrix.return_value = R
        mock_img.cam_from_world.translation = t
        mock_images[img_id] = mock_img

    mock_recon = MagicMock()
    mock_recon.images = mock_images
    mock_recon.cameras = mock_cameras

    return mock_recon, detections


def test_recover_scale_synthetic():
    """recover_scale should return the correct mm/unit factor within 1%."""
    # Marker corners in reconstruction units: 10-unit square at z=0.
    side_recon = 10.0
    marker_world = np.array(
        [[0.0, 0.0, 0.0], [side_recon, 0.0, 0.0], [side_recon, side_recon, 0.0], [0.0, side_recon, 0.0]]
    )
    # Physical side = 50 mm → scale = 50/10 = 5 mm/unit.
    marker_length_mm = 50.0
    expected_scale = marker_length_mm / side_recon

    mock_recon, detections = _make_mock_reconstruction(marker_world, expected_scale)

    result = recover_scale(
        reconstruction=mock_recon,
        image_dir=Path("/nonexistent"),  # not used when detections provided
        marker_length_mm=marker_length_mm,
        detections=detections,
        min_views=2,
    )

    assert abs(result - expected_scale) / expected_scale < 0.01, (
        f"Expected scale {expected_scale:.4f}, got {result:.4f}"
    )


def test_recover_scale_no_markers_raises(tmp_path):
    """recover_scale raises RuntimeError when no markers can be triangulated."""
    mock_recon = MagicMock()
    mock_recon.images = {}

    with pytest.raises(RuntimeError, match="No ArUco markers"):
        recover_scale(
            reconstruction=mock_recon,
            image_dir=tmp_path,
            marker_length_mm=50.0,
            detections={},
            min_views=2,
        )


# ---------------------------------------------------------------------------
# recover_scale_safe
# ---------------------------------------------------------------------------

def test_recover_scale_safe_returns_none_when_marker_length_falsy():
    mock_recon = MagicMock()
    with patch("sfm_mvs_pipeline.scale.aruco_scale.recover_scale") as mock_recover:
        result = recover_scale_safe(
            reconstruction=mock_recon,
            image_dir=Path("/nonexistent"),
            marker_length_mm=None,
            aruco_dict_id=0,
            detections=None,
            min_views=2,
        )

    assert result is None
    mock_recover.assert_not_called()


def test_recover_scale_safe_returns_factor_on_success():
    mock_recon = MagicMock()
    with patch(
        "sfm_mvs_pipeline.scale.aruco_scale.recover_scale", return_value=5.0
    ) as mock_recover:
        result = recover_scale_safe(
            reconstruction=mock_recon,
            image_dir=Path("/nonexistent"),
            marker_length_mm=50.0,
            aruco_dict_id=0,
            detections={"frame.jpg": []},
            min_views=2,
        )

    assert result == 5.0
    mock_recover.assert_called_once_with(
        reconstruction=mock_recon,
        image_dir=Path("/nonexistent"),
        marker_length_mm=50.0,
        aruco_dict_id=0,
        detections={"frame.jpg": []},
        min_views=2,
    )


def test_recover_scale_safe_returns_none_on_runtime_error(caplog):
    mock_recon = MagicMock()
    with patch(
        "sfm_mvs_pipeline.scale.aruco_scale.recover_scale",
        side_effect=RuntimeError("No ArUco markers could be triangulated"),
    ):
        with caplog.at_level("WARNING"):
            result = recover_scale_safe(
                reconstruction=mock_recon,
                image_dir=Path("/nonexistent"),
                marker_length_mm=50.0,
                aruco_dict_id=0,
                detections=None,
                min_views=2,
            )

    assert result is None
    assert "Scale recovery failed" in caplog.text
