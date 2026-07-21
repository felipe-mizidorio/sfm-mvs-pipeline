"""Colour-based removal of pale 'membrane' contamination from the dense cloud.

The membranes in this capture are occlusion-boundary background bleed: bright
points from the pale support surface and desk, fused at the head silhouette and
left inside the spherical head crop. `docs/membrane_filter_report.md` establishes
that the membrane mesh vertices are backed by REAL points (94% have a dense point
within 2 mm) whose colour is overwhelmingly pale (median 93.5% of supporting
points above the pale threshold, against 0% for genuine head surface). They are
contamination, not Poisson hallucination, so they can only be removed upstream of
Poisson — which is what this module does.

**The ArUco markers are white**, and they are load-bearing: they define the metric
scale and the head-crop geometry. A naive "delete pale points" rule would erase
the marker faces and open holes exactly where Poisson would then bridge new
surface. So removal is suppressed inside a protection sphere around every
triangulated marker.

The protection radius is derived **per marker** from that marker's own
triangulated corners (max corner-to-centroid distance + a margin) rather than
from a global constant. This adapts to the real marker size in the reconstruction
instead of hard-coding one, and it automatically widens the zone for a marker
whose triangulation is inflated — failing safe, toward protecting more.

SCENE-DEPENDENT. This filter assumes a dark subject against pale contamination.
It does not generalize to a capture where subject and background have similar
brightness, or where the subject is pale. See the report's limitations.
"""

import logging

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

# Mean-RGB (0-255) at or above which a point is treated as pale. Shared with the
# analysis tooling (arm_metrics.py, membrane_support.py) — keep in sync.
DEFAULT_PALE_THRESHOLD = 150.0
# Added to each marker's own max corner-to-centroid distance to absorb
# triangulation error and any white border extending past the black square.
DEFAULT_MARKER_MARGIN_MM = 5.0


def marker_protection_radii(
    marker_corners: dict[int, dict[int, np.ndarray]],
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-marker centroid and protection radius, in the input coordinate units.

    Args:
        marker_corners: {marker_id: {corner_idx: 3D point}}, as returned by
            scale.aruco_scale.triangulate_marker_corners. Same frame/units as the
            point cloud passed to filter_membrane_points.
        margin: added to each marker's max corner-to-centroid distance, in the
            same units.

    Returns:
        (centroids (M,3), radii (M,)). Both empty when no markers are supplied.
    """
    centroids = []
    radii = []
    for corners in marker_corners.values():
        pts = np.asarray(list(corners.values()), dtype=np.float64)
        if len(pts) == 0:
            continue
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        radii.append(float(np.linalg.norm(pts - centroid, axis=1).max()) + margin)
    if not centroids:
        return np.empty((0, 3)), np.empty((0,))
    return np.asarray(centroids), np.asarray(radii)


def filter_membrane_points(
    pcd: o3d.geometry.PointCloud,
    marker_corners: dict[int, dict[int, np.ndarray]],
    pale_threshold: float = DEFAULT_PALE_THRESHOLD,
    marker_margin: float = DEFAULT_MARKER_MARGIN_MM,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Remove pale points that lie outside every marker's protection sphere.

    A point is removed iff BOTH hold:
      1. mean RGB >= pale_threshold, and
      2. it is outside the protection sphere of every triangulated marker.

    Warn-don't-abort: a cloud without colours, or with no triangulated markers,
    returns unchanged with the reason recorded in the stats dict. Returning the
    cloud unfiltered is the safe failure — deleting pale points with no marker
    protection available would destroy the marker faces.

    Args:
        pcd: dense cloud. Must carry colours to be filtered.
        marker_corners: triangulated marker corners in the SAME frame and units
            as pcd.
        pale_threshold: mean RGB (0-255) at or above which a point is pale.
        marker_margin: protection margin, in pcd's units.

    Returns:
        (filtered_pcd, stats) — stats is manifest-ready and always records
        whether the filter actually ran.
    """
    n_before = len(pcd.points)
    stats: dict = {
        "enabled": True,
        "applied": False,
        "pale_threshold": float(pale_threshold),
        "marker_margin": float(marker_margin),
        "points_before": int(n_before),
        "points_after": int(n_before),
        "points_removed": 0,
    }

    if n_before == 0:
        stats["skipped_reason"] = "empty point cloud"
        logger.warning("Membrane filter: point cloud is empty — skipping.")
        return pcd, stats

    if not pcd.has_colors():
        stats["skipped_reason"] = "point cloud has no colours"
        logger.warning(
            "Membrane filter: cloud has no colours, cannot classify pale points — "
            "returning it unfiltered."
        )
        return pcd, stats

    centroids, radii = marker_protection_radii(marker_corners, marker_margin)
    stats["n_markers_protected"] = int(len(centroids))
    if len(centroids) == 0:
        stats["skipped_reason"] = "no triangulated markers to protect"
        logger.warning(
            "Membrane filter: no triangulated markers available, so marker faces "
            "cannot be protected — returning the cloud unfiltered rather than "
            "risking deletion of the scale anchors."
        )
        return pcd, stats

    points = np.asarray(pcd.points, dtype=np.float64)
    mean_rgb = np.asarray(pcd.colors, dtype=np.float64).mean(axis=1) * 255.0
    is_pale = mean_rgb >= pale_threshold

    # Protected = inside ANY marker's own protection sphere.
    protected = np.zeros(len(points), dtype=bool)
    for centroid, radius in zip(centroids, radii):
        np.logical_or(
            protected,
            np.linalg.norm(points - centroid, axis=1) <= radius,
            out=protected,
        )

    to_remove = is_pale & ~protected
    keep_idx = np.flatnonzero(~to_remove)
    filtered = pcd.select_by_index(keep_idx)

    stats.update({
        "applied": True,
        "points_after": int(len(filtered.points)),
        "points_removed": int(to_remove.sum()),
        "points_pale": int(is_pale.sum()),
        "points_protected": int(protected.sum()),
        "points_pale_protected": int((is_pale & protected).sum()),
        "protection_radius_min": float(radii.min()),
        "protection_radius_max": float(radii.max()),
        "protection_radius_median": float(np.median(radii)),
    })
    logger.info(
        "Membrane filter: removed %d of %d points (%.2f%%) — %d pale, of which "
        "%d were protected by %d marker sphere(s) (radius %.3f-%.3f).",
        stats["points_removed"],
        n_before,
        100.0 * stats["points_removed"] / n_before,
        stats["points_pale"],
        stats["points_pale_protected"],
        stats["n_markers_protected"],
        stats["protection_radius_min"],
        stats["protection_radius_max"],
    )
    return filtered, stats
