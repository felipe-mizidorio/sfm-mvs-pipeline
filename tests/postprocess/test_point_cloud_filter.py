import numpy as np
import open3d as o3d
import pytest
from pathlib import Path

from sfm_mvs_pipeline.postprocess.point_cloud_filter import filter_point_cloud


def _write_pcd_with_outlier(path: Path) -> None:
    rng = np.random.default_rng(0)
    cluster = rng.normal(0.0, 0.01, (200, 3))
    outlier = np.array([[999.0, 999.0, 999.0]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([cluster, outlier]))
    o3d.io.write_point_cloud(str(path), pcd)


def test_filter_removes_extreme_outlier(tmp_path):
    in_ply = tmp_path / "in.ply"
    out_ply = tmp_path / "out.ply"
    _write_pcd_with_outlier(in_ply)

    result = filter_point_cloud(in_ply, out_ply, nb_neighbors=20, std_ratio=2.0)

    pts = np.asarray(result.points)
    assert not np.any(np.all(pts > 900, axis=1)), "extreme outlier was not removed"
    assert len(pts) < 201


def test_filter_writes_output_ply(tmp_path):
    in_ply = tmp_path / "in.ply"
    out_ply = tmp_path / "out.ply"
    _write_pcd_with_outlier(in_ply)

    filter_point_cloud(in_ply, out_ply, nb_neighbors=20, std_ratio=2.0)

    assert out_ply.exists() and out_ply.stat().st_size > 0


def test_filter_returns_point_cloud(tmp_path):
    in_ply = tmp_path / "in.ply"
    out_ply = tmp_path / "out.ply"
    _write_pcd_with_outlier(in_ply)

    result = filter_point_cloud(in_ply, out_ply, nb_neighbors=20, std_ratio=2.0)

    assert isinstance(result, o3d.geometry.PointCloud)


def test_filter_raises_on_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        filter_point_cloud(
            tmp_path / "ghost.ply", tmp_path / "out.ply", nb_neighbors=20, std_ratio=2.0
        )
