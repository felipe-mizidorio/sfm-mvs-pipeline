import numpy as np
import open3d as o3d

from sfm_mvs_pipeline.postprocess.membrane_filter import (
    filter_membrane_points,
    marker_protection_radii,
)


def _cloud(points, greys):
    """Cloud whose per-point colour is a uniform grey in 0-255."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cols = np.repeat(np.asarray(greys, dtype=np.float64)[:, None], 3, axis=1) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def _square_marker(centre, half=10.0):
    """Axis-aligned square of 4 corners, half-diagonal ~= half*sqrt(2)."""
    c = np.asarray(centre, dtype=np.float64)
    return {
        0: c + [-half, -half, 0.0],
        1: c + [half, -half, 0.0],
        2: c + [half, half, 0.0],
        3: c + [-half, half, 0.0],
    }


def test_removes_pale_points_away_from_markers():
    pcd = _cloud([[100.0, 0, 0], [101.0, 0, 0]], [200, 20])
    markers = {0: _square_marker([0, 0, 0])}

    filtered, stats = filter_membrane_points(pcd, markers, marker_margin=5.0)

    assert stats["applied"] is True
    assert stats["points_removed"] == 1
    assert len(filtered.points) == 1
    # The surviving point is the dark one.
    assert np.allclose(np.asarray(filtered.points)[0], [101.0, 0, 0])


def test_pale_points_on_a_marker_are_protected():
    """The markers are white — deleting them would destroy the scale anchors."""
    marker = _square_marker([0, 0, 0])
    on_marker = [list(marker[0]), [0.0, 0.0, 0.0]]
    pcd = _cloud(on_marker, [255, 240])

    filtered, stats = filter_membrane_points(pcd, {0: marker}, marker_margin=5.0)

    assert stats["points_removed"] == 0
    assert stats["points_pale"] == 2
    assert stats["points_pale_protected"] == 2
    assert len(filtered.points) == 2


def test_dark_points_are_never_removed():
    pcd = _cloud([[500.0, 500.0, 500.0], [0.0, 0.0, 0.0]], [10, 90])
    filtered, stats = filter_membrane_points(pcd, {0: _square_marker([0, 0, 0])})

    assert stats["points_removed"] == 0
    assert len(filtered.points) == 2


def test_returns_unfiltered_when_no_markers_are_available():
    """Failing safe: without marker protection, pale removal would eat the markers."""
    pcd = _cloud([[100.0, 0, 0]], [250])

    filtered, stats = filter_membrane_points(pcd, {})

    assert stats["applied"] is False
    assert stats["skipped_reason"] == "no triangulated markers to protect"
    assert len(filtered.points) == 1


def test_returns_unfiltered_when_cloud_has_no_colours():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[100.0, 0.0, 0.0]]))

    filtered, stats = filter_membrane_points(pcd, {0: _square_marker([0, 0, 0])})

    assert stats["applied"] is False
    assert stats["skipped_reason"] == "point cloud has no colours"
    assert len(filtered.points) == 1


def test_empty_cloud_is_handled():
    filtered, stats = filter_membrane_points(
        o3d.geometry.PointCloud(), {0: _square_marker([0, 0, 0])}
    )
    assert stats["applied"] is False
    assert stats["skipped_reason"] == "empty point cloud"
    assert len(filtered.points) == 0


def test_protection_radius_is_derived_per_marker():
    """A larger marker must earn a larger protection sphere."""
    small = _square_marker([0, 0, 0], half=10.0)
    large = _square_marker([200, 0, 0], half=20.0)

    centroids, radii = marker_protection_radii({0: small, 1: large}, margin=5.0)

    assert len(centroids) == 2
    assert radii[0] < radii[1]
    # half-diagonal of the 10-half square is sqrt(200) ~= 14.14, plus 5 margin
    assert np.isclose(radii[0], np.sqrt(200.0) + 5.0)
    assert np.isclose(radii[1], np.sqrt(800.0) + 5.0)


def test_threshold_is_respected():
    pcd = _cloud([[100.0, 0, 0], [102.0, 0, 0]], [160, 140])
    markers = {0: _square_marker([0, 0, 0])}

    _, strict = filter_membrane_points(pcd, markers, pale_threshold=150.0)
    _, loose = filter_membrane_points(pcd, markers, pale_threshold=130.0)

    assert strict["points_removed"] == 1
    assert loose["points_removed"] == 2
