import logging
from pathlib import Path

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def chamfer_distance(
    predicted: o3d.geometry.PointCloud,
    ground_truth: o3d.geometry.PointCloud,
) -> float:
    """Mean of squared nearest-neighbour distances, averaged over both directions."""
    d_pred_to_gt = np.asarray(predicted.compute_point_cloud_distance(ground_truth))
    d_gt_to_pred = np.asarray(ground_truth.compute_point_cloud_distance(predicted))
    return float((np.mean(d_pred_to_gt**2) + np.mean(d_gt_to_pred**2)) / 2.0)


def hausdorff_distance(
    predicted: o3d.geometry.PointCloud,
    ground_truth: o3d.geometry.PointCloud,
) -> float:
    """Max over both directed Hausdorff distances."""
    d_pred_to_gt = np.asarray(predicted.compute_point_cloud_distance(ground_truth))
    d_gt_to_pred = np.asarray(ground_truth.compute_point_cloud_distance(predicted))
    return float(max(d_pred_to_gt.max(), d_gt_to_pred.max()))


def rms_distance(
    predicted: o3d.geometry.PointCloud,
    ground_truth: o3d.geometry.PointCloud,
) -> float:
    """Root mean square of nearest-neighbour distances from predicted to ground truth."""
    d = np.asarray(predicted.compute_point_cloud_distance(ground_truth))
    return float(np.sqrt(np.mean(d**2)))


_METRIC_FNS = {
    "chamfer": chamfer_distance,
    "hausdorff": hausdorff_distance,
    "rms": rms_distance,
}


def evaluate(
    predicted_ply: Path,
    ground_truth_ply: Path,
    options: dict,
) -> dict[str, float]:
    if not predicted_ply.exists():
        raise FileNotFoundError(f"Predicted point cloud not found: {predicted_ply}")
    if not ground_truth_ply.exists():
        raise FileNotFoundError(
            f"Ground truth point cloud not found: {ground_truth_ply}"
        )

    predicted = o3d.io.read_point_cloud(str(predicted_ply))
    ground_truth = o3d.io.read_point_cloud(str(ground_truth_ply))

    if len(predicted.points) == 0:
        raise ValueError(f"Predicted point cloud '{predicted_ply}' contains no points.")
    if len(ground_truth.points) == 0:
        raise ValueError(
            f"Ground truth point cloud '{ground_truth_ply}' contains no points."
        )

    requested = options["metrics"]
    unknown = [m for m in requested if m not in _METRIC_FNS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Valid: {list(_METRIC_FNS)}")

    results: dict[str, float] = {}
    for name in requested:
        if not requested[name]:
            continue
        value = _METRIC_FNS[name](predicted, ground_truth)
        logger.info("Metric %s = %.6f", name, value)
        results[name] = value

    return results
