import open3d as o3d
import pytest

from sfm_mvs_pipeline.mesh.surface_reconstruction import reconstruct_surface

_MESH_OPTIONS = {
    "depth": 6,
    "scale": 1.1,
    "linear_fit": False,
}


@pytest.fixture()
def sphere_ply(tmp_path):
    """Write a uniformly sampled unit-sphere point cloud to disk."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    pcd = mesh.sample_points_uniformly(number_of_points=500)
    path = tmp_path / "sphere.ply"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def test_surface_reconstruction_creates_output(sphere_ply, tmp_path):
    output = tmp_path / "mesh.ply"
    reconstruct_surface(
        input_ply=sphere_ply,
        output_ply=output,
        options=_MESH_OPTIONS,
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_surface_reconstruction_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        reconstruct_surface(
            input_ply=tmp_path / "nonexistent.ply",
            output_ply=tmp_path / "mesh.ply",
            options=_MESH_OPTIONS,
        )


def test_surface_reconstruction_creates_parent_dirs(sphere_ply, tmp_path):
    nested_output = tmp_path / "a" / "b" / "mesh.ply"
    reconstruct_surface(
        input_ply=sphere_ply,
        output_ply=nested_output,
        options=_MESH_OPTIONS,
    )
    assert nested_output.exists()
