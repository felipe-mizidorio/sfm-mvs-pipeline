import numpy as np
import open3d as o3d
import pytest

from sfm_mvs_pipeline.evaluation.metrics import (
    chamfer_distance,
    hausdorff_distance,
    evaluate,
)


def _make_cloud(n: int = 200, seed: int = 0) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rng.random((n, 3)))
    return pcd


@pytest.fixture()
def cloud_a():
    return _make_cloud(seed=0)


@pytest.fixture()
def cloud_b():
    return _make_cloud(seed=1)


@pytest.fixture()
def cloud_a_ply(tmp_path, cloud_a):
    path = tmp_path / "cloud_a.ply"
    o3d.io.write_point_cloud(str(path), cloud_a)
    return path


@pytest.fixture()
def cloud_b_ply(tmp_path, cloud_b):
    path = tmp_path / "cloud_b.ply"
    o3d.io.write_point_cloud(str(path), cloud_b)
    return path


class TestChamferDistance:
    def test_identical_clouds_is_zero(self, cloud_a):
        result = chamfer_distance(cloud_a, cloud_a)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self, cloud_a, cloud_b):
        assert chamfer_distance(cloud_a, cloud_b) == pytest.approx(
            chamfer_distance(cloud_b, cloud_a)
        )

    def test_non_negative(self, cloud_a, cloud_b):
        assert chamfer_distance(cloud_a, cloud_b) >= 0.0


class TestHausdorffDistance:
    def test_identical_clouds_is_zero(self, cloud_a):
        result = hausdorff_distance(cloud_a, cloud_a)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self, cloud_a, cloud_b):
        assert hausdorff_distance(cloud_a, cloud_b) >= 0.0

    def test_greater_than_chamfer(self, cloud_a, cloud_b):
        # Hausdorff is max; Chamfer is mean — Hausdorff >= Chamfer * 2 >= Chamfer
        hd = hausdorff_distance(cloud_a, cloud_b)
        cd = chamfer_distance(cloud_a, cloud_b)
        assert hd >= cd


class TestEvaluate:
    def test_returns_requested_metrics_only(self, cloud_a_ply, cloud_b_ply):
        options = {"metrics": {"chamfer": True, "hausdorff": False, "rms": False}}
        result = evaluate(cloud_a_ply, cloud_b_ply, options)
        assert set(result.keys()) == {"chamfer"}

    def test_all_metrics_returned_when_all_enabled(self, cloud_a_ply, cloud_b_ply):
        options = {"metrics": {"chamfer": True, "hausdorff": True, "rms": True}}
        result = evaluate(cloud_a_ply, cloud_b_ply, options)
        assert set(result.keys()) == {"chamfer", "hausdorff", "rms"}

    def test_values_are_floats(self, cloud_a_ply, cloud_b_ply):
        options = {"metrics": {"chamfer": True, "hausdorff": True, "rms": True}}
        result = evaluate(cloud_a_ply, cloud_b_ply, options)
        assert all(isinstance(v, float) for v in result.values())

    def test_missing_predicted_raises(self, tmp_path, cloud_b_ply):
        options = {"metrics": {"chamfer": True}}
        with pytest.raises(FileNotFoundError):
            evaluate(tmp_path / "ghost.ply", cloud_b_ply, options)
