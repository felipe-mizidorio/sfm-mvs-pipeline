"""Metric scale recovery from ArUco markers for neonatal cranial morphometry.

The pipeline produces geometry in arbitrary SfM units.  ArUco markers printed
on the neonatal cap have a known physical side length, so their 3D corner
positions triangulated from the reconstruction can be used to derive a scale
factor that converts reconstruction units to millimetres.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import cv2.aruco
import numpy as np
import open3d as o3d
import pycolmap

logger = logging.getLogger(__name__)


def recover_scale(
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    marker_length_mm: float,
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50,
    detections: dict[str, list[dict]] | None = None,
    min_views: int = 2,
) -> float:
    """Return a scale factor (mm / reconstruction-unit) derived from ArUco markers.

    Args:
        reconstruction: Completed sparse reconstruction from incremental mapping.
        image_dir: Directory containing the registered images (supports subdirs).
        marker_length_mm: Physical side length of the ArUco square in millimetres.
        aruco_dict_id: OpenCV ArUco dictionary ID (default DICT_4X4_50).
        detections: Pre-computed detections from the preprocessing manifest.
            Format: {image_name: [{"id": int, "corners": [[x,y],…]×4}, …]}.
            If provided, images are not re-read from disk.
        min_views: Minimum number of views required to triangulate a marker corner.

    Returns:
        Median scale factor across all triangulated markers.

    Raises:
        RuntimeError: If no markers can be triangulated from the reconstruction.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # marker_obs[marker_id] = list of (image_id, corners_2d[4,2])
    marker_obs: dict[int, list[tuple[int, np.ndarray]]] = {}

    for image in reconstruction.images.values():
        if not image.has_pose:
            continue

        image_name = image.name

        if detections is not None:
            if image_name not in detections:
                continue
            raw_markers = detections[image_name]
            for m in raw_markers:
                mid = int(m["id"])
                corners = np.array(m["corners"], dtype=np.float32).reshape(4, 2)
                marker_obs.setdefault(mid, []).append((image.image_id, corners))
        else:
            img_path = _find_image(image_dir, image_name)
            if img_path is None:
                logger.debug("Image not found on disk: %s", image_name)
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                logger.debug("Could not read image: %s", img_path)
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            corners_list, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            for corners_raw, mid in zip(corners_list, ids.flatten()):
                corners = corners_raw.reshape(4, 2).astype(np.float32)
                marker_obs.setdefault(int(mid), []).append((image.image_id, corners))

    logger.info(
        "ArUco detections: %d distinct marker IDs across registered images",
        len(marker_obs),
    )

    scale_estimates: list[float] = []

    for mid, obs in marker_obs.items():
        if len(obs) < min_views:
            logger.debug("Marker %d: only %d views, need %d — skipping", mid, len(obs), min_views)
            continue

        # Triangulate each of the 4 corners from all observation pairs.
        corners_3d: list[np.ndarray] = []  # one entry per successfully triangulated corner
        for corner_idx in range(4):
            pts_3d: list[np.ndarray] = []
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    img_id_a, corners_a = obs[i]
                    img_id_b, corners_b = obs[j]
                    p3d = _triangulate_point(
                        reconstruction,
                        img_id_a,
                        corners_a[corner_idx],
                        img_id_b,
                        corners_b[corner_idx],
                    )
                    if p3d is not None:
                        pts_3d.append(p3d)
            if pts_3d:
                corners_3d.append(np.mean(pts_3d, axis=0))

        if len(corners_3d) < 2:
            logger.debug("Marker %d: could not triangulate enough corners", mid)
            continue

        # Use side length (corner 0 → corner 1) as the reference distance.
        side_3d = float(np.linalg.norm(corners_3d[0] - corners_3d[1]))
        if side_3d < 1e-9:
            continue
        scale = marker_length_mm / side_3d
        logger.debug("Marker %d: side_3d=%.6f → scale=%.4f mm/unit", mid, side_3d, scale)
        scale_estimates.append(scale)

    if not scale_estimates:
        raise RuntimeError(
            "No ArUco markers could be triangulated from the reconstruction. "
            "Check that markers are visible in multiple registered frames and that "
            "marker_length_mm / dict_id in aruco.yaml match the physical markers."
        )

    scale_factor = float(np.median(scale_estimates))
    logger.info(
        "Scale recovery: %d marker(s) used, scale=%.4f mm/unit (median)",
        len(scale_estimates),
        scale_factor,
    )
    return scale_factor


def apply_scale_to_ply(ply_path: Path, scale: float) -> None:
    """Multiply all XYZ coordinates in a point-cloud PLY by scale in-place."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * scale)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    logger.info("Applied scale %.4f to point cloud '%s'", scale, ply_path)


def apply_scale_to_mesh(ply_path: Path, scale: float) -> None:
    """Multiply all vertex coordinates in a mesh PLY by scale in-place."""
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
    o3d.io.write_triangle_mesh(str(ply_path), mesh)
    logger.info("Applied scale %.4f to mesh '%s'", scale, ply_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_image(image_dir: Path, name: str) -> Path | None:
    """Locate an image by name under image_dir (handles nested paths)."""
    candidate = image_dir / name
    if candidate.exists():
        return candidate
    # Fall back to searching by filename only (in case of path prefix differences).
    filename = Path(name).name
    matches = list(image_dir.rglob(filename))
    return matches[0] if matches else None


def _projection_matrix(
    reconstruction: pycolmap.Reconstruction,
    image_id: int,
) -> np.ndarray:
    """Return the 3×4 projection matrix P = K [R | t] for a registered image."""
    image = reconstruction.images[image_id]
    camera = reconstruction.cameras[image.camera_id]
    K = np.array(camera.calibration_matrix())
    cfw = image.cam_from_world
    R = cfw.rotation.matrix()
    t = cfw.translation.reshape(3, 1)
    return K @ np.hstack([R, t])


def _triangulate_point(
    reconstruction: pycolmap.Reconstruction,
    img_id_a: int,
    pt2d_a: np.ndarray,
    img_id_b: int,
    pt2d_b: np.ndarray,
) -> np.ndarray | None:
    """Triangulate a single 3D point from two corresponding 2D observations."""
    try:
        P1 = _projection_matrix(reconstruction, img_id_a)
        P2 = _projection_matrix(reconstruction, img_id_b)
    except KeyError:
        return None

    pts1 = pt2d_a.reshape(2, 1).astype(np.float64)
    pts2 = pt2d_b.reshape(2, 1).astype(np.float64)
    hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
    w = hom[3, 0]
    if abs(w) < 1e-9:
        return None
    return (hom[:3, 0] / w).astype(np.float64)
