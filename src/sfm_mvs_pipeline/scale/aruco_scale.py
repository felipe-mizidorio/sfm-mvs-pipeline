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


def triangulate_marker_corners(
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50,
    detections: dict[str, list[dict]] | None = None,
    min_views: int = 2,
) -> dict[int, dict[int, np.ndarray]]:
    """Triangulate ArUco marker corners into reconstruction (SfM-unit) space.

    Args:
        reconstruction: Completed sparse reconstruction from incremental mapping.
        image_dir: Directory containing the registered images (supports subdirs).
        aruco_dict_id: OpenCV ArUco dictionary ID (default DICT_4X4_50).
        detections: Pre-computed detections from the preprocessing manifest.
            Format: {image_name: [{"id": int, "corners": [[x,y],…]×4}, …]}.
            If provided, images are not re-read from disk.
        min_views: Minimum number of views required to triangulate a marker corner.

    Returns:
        {marker_id: {corner_index: xyz}} with one mean 3D position per
        successfully triangulated corner. Markers observed in fewer than
        min_views images, or with no triangulable corner, are omitted.
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

    corners_by_marker: dict[int, dict[int, np.ndarray]] = {}

    for mid, obs in marker_obs.items():
        if len(obs) < min_views:
            logger.debug("Marker %d: only %d views, need %d — skipping", mid, len(obs), min_views)
            continue

        # Triangulate each of the 4 corners from all observation pairs.
        corners_3d: dict[int, np.ndarray] = {}  # corner index → triangulated 3D point
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
                corners_3d[corner_idx] = np.mean(pts_3d, axis=0)

        if corners_3d:
            corners_by_marker[mid] = corners_3d

    return corners_by_marker


def marker_corner_points(
    corners_by_marker: dict[int, dict[int, np.ndarray]],
) -> np.ndarray:
    """Flatten triangulated corners into an (N, 3) array of SfM-unit points."""
    pts = [p for corners in corners_by_marker.values() for p in corners.values()]
    return np.array(pts) if pts else np.empty((0, 3))


def _scale_from_marker_corners(
    corners_by_marker: dict[int, dict[int, np.ndarray]],
    marker_length_mm: float,
) -> float:
    """Median mm/SfM-unit factor across markers with a measurable side.

    Raises:
        RuntimeError: If no marker yields a usable side length.
    """
    scale_estimates: list[float] = []

    for mid, corners_3d in corners_by_marker.items():
        # Only adjacent corner pairs measure the marker side; (0,2) and (1,3)
        # are diagonals (side × √2) and must not be used as the reference.
        side_lengths = [
            float(np.linalg.norm(corners_3d[a] - corners_3d[b]))
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
            if a in corners_3d and b in corners_3d
        ]
        side_lengths = [s for s in side_lengths if s > 1e-9]
        if not side_lengths:
            logger.debug("Marker %d: no adjacent corner pair triangulated", mid)
            continue

        side_3d = float(np.median(side_lengths))
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


def recover_scale(
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    marker_length_mm: float,
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50,
    detections: dict[str, list[dict]] | None = None,
    min_views: int = 2,
) -> float:
    """Return a scale factor (mm / reconstruction-unit) derived from ArUco markers.

    See triangulate_marker_corners for argument semantics.

    Returns:
        Median scale factor across all triangulated markers.

    Raises:
        RuntimeError: If no markers can be triangulated from the reconstruction.
    """
    corners_by_marker = triangulate_marker_corners(
        reconstruction=reconstruction,
        image_dir=image_dir,
        aruco_dict_id=aruco_dict_id,
        detections=detections,
        min_views=min_views,
    )
    return _scale_from_marker_corners(corners_by_marker, marker_length_mm)


def recover_scale_details_safe(
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    marker_length_mm: float | None,
    aruco_dict_id: int,
    detections: dict[str, list[dict]] | None,
    min_views: int,
) -> tuple[float | None, np.ndarray | None, dict[int, dict[int, np.ndarray]] | None]:
    """Recover scale, flattened corner points, and per-marker corners; never raises.

    Returns (scale_factor, marker_points, corners_by_marker):
    - marker_points is an (N, 3) array of triangulated ArUco corner positions
      in SfM units — the input for automatic head-crop sizing.
    - corners_by_marker is {marker_id: {corner_index: xyz}} — the input for
      the independent layout-based scale sanity check.
    All are None if marker_length_mm is falsy (scale recovery disabled) or if
    recovery fails (logged as a warning): without a valid scale the marker
    positions cannot size a metric crop.
    """
    if not marker_length_mm:
        return None, None, None

    logger.info("=== Scale recovery: detecting ArUco markers ===")
    try:
        corners_by_marker = triangulate_marker_corners(
            reconstruction=reconstruction,
            image_dir=image_dir,
            aruco_dict_id=aruco_dict_id,
            detections=detections,
            min_views=min_views,
        )
        scale_factor = _scale_from_marker_corners(corners_by_marker, marker_length_mm)
        logger.info("Scale factor: %.6f mm/unit", scale_factor)
        return scale_factor, marker_corner_points(corners_by_marker), corners_by_marker
    except RuntimeError as exc:
        logger.warning("Scale recovery failed: %s — outputs remain in SfM units.", exc)
        return None, None, None
    except Exception:
        logger.exception(
            "Scale recovery crashed unexpectedly — outputs remain in SfM units."
        )
        return None, None, None


def recover_scale_and_markers_safe(
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    marker_length_mm: float | None,
    aruco_dict_id: int,
    detections: dict[str, list[dict]] | None,
    min_views: int,
) -> tuple[float | None, np.ndarray | None]:
    """recover_scale_details_safe for callers that don't need per-marker corners."""
    scale_factor, marker_points, _ = recover_scale_details_safe(
        reconstruction=reconstruction,
        image_dir=image_dir,
        marker_length_mm=marker_length_mm,
        aruco_dict_id=aruco_dict_id,
        detections=detections,
        min_views=min_views,
    )
    return scale_factor, marker_points


def recover_scale_safe(
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    marker_length_mm: float | None,
    aruco_dict_id: int,
    detections: dict[str, list[dict]] | None,
    min_views: int,
) -> float | None:
    """recover_scale_and_markers_safe for callers that only need the factor."""
    scale_factor, _ = recover_scale_and_markers_safe(
        reconstruction=reconstruction,
        image_dir=image_dir,
        marker_length_mm=marker_length_mm,
        aruco_dict_id=aruco_dict_id,
        detections=detections,
        min_views=min_views,
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
    cfw = image.cam_from_world()
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
