import numpy as np
import open3d as o3d
import pytest

from sfm_mvs_pipeline.visualization.plotly_viz import save_mesh_html, save_point_cloud_html


def _pcd(n: int = 50) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(n, 3))
    return pcd


def _mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    )
    return mesh


def test_save_point_cloud_html_creates_file(tmp_path):
    out = tmp_path / "pcd.html"
    save_point_cloud_html(_pcd(), out, "Test PCD")
    assert out.exists() and out.stat().st_size > 0


def test_save_mesh_html_creates_file(tmp_path):
    out = tmp_path / "mesh.html"
    save_mesh_html(_mesh(), out, "Test Mesh")
    assert out.exists() and out.stat().st_size > 0


def test_save_point_cloud_html_creates_parent_dirs(tmp_path):
    out = tmp_path / "a" / "b" / "pcd.html"
    save_point_cloud_html(_pcd(), out, "Test")
    assert out.exists()


def test_save_point_cloud_html_downsamples_without_error(tmp_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(300_000, 3))
    out = tmp_path / "big.html"
    save_point_cloud_html(pcd, out, "Big PCD", max_points=200_000)
    assert out.exists()
