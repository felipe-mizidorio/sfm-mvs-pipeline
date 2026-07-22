"""Shared post-processing helpers used by all pipeline entry-point scripts.

Orchestrates SOR → visualize → Poisson → LCC → Taubin → visualize and writes
pipeline_manifest.json. Each script calls these functions after stereo fusion,
inserting scale recovery in between (which is script-specific).
"""

import datetime
import hashlib
import json
import logging
import platform
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap

from sfm_mvs_pipeline.mesh.surface_reconstruction import _apply_lcc, _apply_taubin, _run_poisson
from sfm_mvs_pipeline.postprocess.membrane_filter import filter_membrane_points
from sfm_mvs_pipeline.postprocess.point_cloud_filter import filter_point_cloud
from sfm_mvs_pipeline.visualization.plotly_viz import save_mesh_html, save_point_cloud_html

logger = logging.getLogger(__name__)

# --- Automatic head-crop sizing -------------------------------------------
# Domestic-use constraint: the parent films at home and never tunes crop
# parameters, so the radius is derived from what capture already provides —
# ArUco corner positions (SfM units) and the recovered mm/unit scale.
# Anatomy: a term neonate's occipitofrontal circumference is ~33–37 cm
# (head diameter ~105–118 mm). Markers sit on a cap on the crown, so the
# crop must reach from the marker cloud past the chin/occiput.
HEAD_CROP_MARGIN_MM = 100.0
# Never crop tighter than a full neonatal head around the centre.
HEAD_CROP_MIN_RADIUS_MM = 140.0
# Cap against wild marker triangulations: beyond this the crop stops
# excluding background and the radius estimate itself is suspect.
HEAD_CROP_MAX_RADIUS_MM = 250.0
# Legacy fallback (SfM units, ≈230 mm at typical capture scale) used when
# scale or marker positions are unavailable.
DEFAULT_HEAD_RADIUS_SFM = 1.5
# Minimum triangulated ArUco corners for the corner centroid to be trusted as
# the crop centre. One marker (4 corners) has no redundancy: a single bad
# triangulation drags the centroid arbitrarily far with nothing to counter it,
# and 4 coplanar corners sit ON the scalp shell rather than sampling its
# curvature. Two markers (8 corners) is the minimum at which corners
# cross-check each other and the centroid starts pulling toward the cranial
# interior. Even a poor domestic capture that recovered scale at all
# triangulates several of the cap's ~19 markers, so 8 is conservative for
# real captures while still catching the degenerate single-marker case.
HEAD_CROP_MIN_MARKER_CORNERS = 8


def estimate_head_center(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """Least-squares intersection of camera optical axes → approximate head center.

    Raises:
        np.linalg.LinAlgError: If poses do not constrain a unique intersection
            (e.g. no posed images, or all optical axes parallel).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for img in reconstruction.images.values():
        if not img.has_pose:
            continue
        cfw = img.cam_from_world()
        R = cfw.rotation.matrix()
        t = cfw.translation
        C = -R.T @ t
        d = R[2]
        d /= np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ C
    return np.linalg.solve(A, b)


def crop_to_sphere(
    pcd: o3d.geometry.PointCloud, center: np.ndarray, radius: float
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    dists = np.linalg.norm(pts - center, axis=1)
    mask = dists <= radius
    logger.info(
        "Spherical crop: keeping %d / %d points within radius %.4f SfM units of center %s",
        mask.sum(),
        len(pts),
        radius,
        np.round(center, 4),
    )
    return pcd.select_by_index(np.where(mask)[0])


def auto_head_radius(
    center: np.ndarray,
    marker_points: np.ndarray | None,
    scale_factor: float | None,
) -> tuple[float, dict] | None:
    """Head-crop radius (SfM units) derived from triangulated ArUco corners.

    Median marker-corner distance from the crop centre (robust to a few bad
    triangulations) plus HEAD_CROP_MARGIN_MM, clamped to
    [HEAD_CROP_MIN_RADIUS_MM, HEAD_CROP_MAX_RADIUS_MM]. Returns None when the
    scale factor or marker positions are unavailable — callers fall back to
    DEFAULT_HEAD_RADIUS_SFM.

    Returns:
        (radius, clamp_info) where clamp_info records whether the clamp fired
        ("min" | "max" | False) and the pre-clamp value in millimetres. With a
        well-placed centre the clamp should never fire, so tripping it is a
        sentinel for an upstream problem (bad triangulations or wrong scale).
    """
    if scale_factor is None or marker_points is None or len(marker_points) == 0:
        return None
    dists = np.linalg.norm(np.asarray(marker_points) - np.asarray(center), axis=1)
    unclamped = float(np.median(dists)) + HEAD_CROP_MARGIN_MM / scale_factor
    radius = max(unclamped, HEAD_CROP_MIN_RADIUS_MM / scale_factor)
    radius = min(radius, HEAD_CROP_MAX_RADIUS_MM / scale_factor)
    clamped: str | bool = False
    if radius > unclamped:
        clamped = "min"
    elif radius < unclamped:
        clamped = "max"
    if clamped:
        logger.warning(
            "Auto head-crop radius clamped to %s bound: computed %.1f mm from "
            "markers, using %.1f mm. A correct centre should not hit the clamp "
            "— check marker triangulations and metric scale.",
            clamped,
            unclamped * scale_factor,
            radius * scale_factor,
        )
    clamp_info = {
        "radius_clamped": clamped,
        "radius_unclamped_mm": unclamped * scale_factor,
    }
    logger.info(
        "Auto head-crop radius: %.4f SfM units (%.1f mm) from %d marker corner(s)",
        radius,
        radius * scale_factor,
        len(marker_points),
    )
    return radius, clamp_info


def run_head_crop(
    dense_filtered_ply: Path,
    output_dir: Path,
    reconstruction: pycolmap.Reconstruction,
    head_radius_override: float | None,
    scale_factor: float | None,
    marker_points: np.ndarray | None,
) -> tuple[Path, dict]:
    """Spherical crop of the SOR-filtered cloud to the head region.

    Centre selection: centroid of the triangulated ArUco corners when at least
    HEAD_CROP_MIN_MARKER_CORNERS are available, else the least-squares
    optical-axis intersection. On a one-sided capture arc the optical-axis
    point converges on the most-observed region (face/sides), not the cranial
    centre; the corner centroid sits on the cap's curved shell and is
    intrinsically biased toward the cranial interior, with no anatomical
    constant needed.

    Radius selection: explicit override (debug) > auto-derived from ArUco
    markers > DEFAULT_HEAD_RADIUS_SFM fallback. An override of 0 (or negative)
    disables the crop entirely.

    Returns:
        (input_for_poisson, crop_stats): the PLY the mesh stage should consume
        (the cropped file, or dense_filtered_ply when the crop is skipped or
        removes every point) and stats for pipeline_manifest.json.
    """
    if head_radius_override is not None and head_radius_override <= 0:
        logger.info("Head crop disabled (--head-radius %s).", head_radius_override)
        return dense_filtered_ply, {}

    if marker_points is not None and len(marker_points) >= HEAD_CROP_MIN_MARKER_CORNERS:
        head_center = np.asarray(marker_points).mean(axis=0)
        center_source = "aruco_centroid"
        logger.info(
            "Head centre from ArUco corner centroid (%d corners).", len(marker_points)
        )
    else:
        try:
            head_center = estimate_head_center(reconstruction)
        except np.linalg.LinAlgError:
            logger.warning(
                "Could not estimate head centre from camera poses — skipping head crop."
            )
            return dense_filtered_ply, {}
        center_source = "optical_axis_fallback"
        logger.warning(
            "Only %d triangulated ArUco corner(s) (< %d) — falling back to the "
            "optical-axis head centre, which is biased on one-sided captures.",
            0 if marker_points is None else len(marker_points),
            HEAD_CROP_MIN_MARKER_CORNERS,
        )
    logger.info("Head center (SfM, %s): %s", center_source, np.round(head_center, 4))

    clamp_info: dict = {}
    if head_radius_override is not None:
        radius, radius_source = float(head_radius_override), "override"
    else:
        auto_result = auto_head_radius(head_center, marker_points, scale_factor)
        if auto_result is None:
            logger.warning(
                "No usable ArUco scale/markers for auto crop radius — "
                "falling back to %.2f SfM units.",
                DEFAULT_HEAD_RADIUS_SFM,
            )
            radius, radius_source = DEFAULT_HEAD_RADIUS_SFM, "default_fallback"
        else:
            (radius, clamp_info), radius_source = auto_result, "aruco_auto"

    logger.info(
        "=== Post-fusion spherical crop (radius=%.3f SfM units, %s) ===",
        radius,
        radius_source,
    )
    pcd = o3d.io.read_point_cloud(str(dense_filtered_ply))
    logger.info("Dense filtered cloud before crop: %d points", len(pcd.points))
    pcd_crop = crop_to_sphere(pcd, head_center, radius)

    if len(pcd_crop.points) == 0:
        logger.warning(
            "Crop removed all points! Falling back to SOR-filtered dense cloud. "
            "Check camera poses / marker triangulation, or override --head-radius."
        )
        return dense_filtered_ply, {}

    cropped_ply = output_dir / "dense_filtered_cropped.ply"
    o3d.io.write_point_cloud(str(cropped_ply), pcd_crop)
    logger.info(
        "Cropped dense cloud: %d points, saved to '%s'", len(pcd_crop.points), cropped_ply
    )

    crop_stats = {
        "head_crop": {
            "radius_sfm_units": radius,
            "radius_source": radius_source,
            **clamp_info,
            "center_sfm": [float(c) for c in head_center],
            "center_source": center_source,
            "points_before": len(pcd.points),
            "points_after": len(pcd_crop.points),
        }
    }
    return cropped_ply, crop_stats


def run_sor_and_visualize(
    input_ply: Path,
    output_dir: Path,
    filter_opts: dict,
) -> tuple[Path, dict]:
    """Run SOR on input_ply, save filtered PLY to output_dir, write two HTML checkpoints.

    Returns:
        (dense_filtered_ply, sor_stats) where sor_stats contains point counts
        for inclusion in pipeline_manifest.json.
    """
    viz_dir = output_dir / "visualizations"

    pcd_raw = o3d.io.read_point_cloud(str(input_ply))
    save_point_cloud_html(pcd_raw, viz_dir / "dense_raw.html", "Dense cloud (raw)")

    dense_filtered_ply = output_dir / "dense_filtered.ply"
    pcd_filtered = filter_point_cloud(
        input_ply,
        dense_filtered_ply,
        filter_opts["nb_neighbors"],
        filter_opts["std_ratio"],
    )
    save_point_cloud_html(
        pcd_filtered, viz_dir / "dense_after_sor.html", "Dense cloud (after SOR)"
    )

    sor_stats = {
        "point_cloud_filtering": {
            "nb_neighbors": filter_opts["nb_neighbors"],
            "std_ratio": filter_opts["std_ratio"],
            "points_before": len(pcd_raw.points),
            "points_after": len(pcd_filtered.points),
            "points_removed": len(pcd_raw.points) - len(pcd_filtered.points),
        }
    }
    return dense_filtered_ply, sor_stats


def run_poisson_lcc_and_visualize(
    input_ply: Path,
    output_ply: Path,
    output_dir: Path,
    mesh_opts: dict,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Poisson + density trim + LCC + Taubin smoothing, with HTML checkpoints.

    Saves visualizations: mesh_before_lcc.html, mesh_after_lcc.html,
    mesh_after_taubin.html (skipped if taubin_smoothing not configured).

    Returns:
        (final_mesh, lcc_stats) where lcc_stats contains triangle counts
        for inclusion in pipeline_manifest.json.
    """
    viz_dir = output_dir / "visualizations"

    pcd = o3d.io.read_point_cloud(str(input_ply))
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud '{input_ply}' is empty.")

    pre_lcc_mesh = _run_poisson(pcd, mesh_opts)
    save_mesh_html(pre_lcc_mesh, viz_dir / "mesh_before_lcc.html", "Mesh (before LCC)")

    mesh, lcc_stats = _apply_lcc(pre_lcc_mesh, mesh_opts)
    save_mesh_html(mesh, viz_dir / "mesh_after_lcc.html", "Mesh (after LCC)")

    mesh = _apply_taubin(mesh, mesh_opts)
    smoothing_cfg = mesh_opts.get("taubin_smoothing", {})
    if smoothing_cfg and int(smoothing_cfg.get("iterations", 10)) > 0:
        save_mesh_html(mesh, viz_dir / "mesh_after_taubin.html", "Mesh (after Taubin smoothing)")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_ply), mesh)
    logger.info(
        "Mesh saved: %d vertices, %d triangles → '%s'",
        len(mesh.vertices),
        len(mesh.triangles),
        output_ply,
    )
    return mesh, lcc_stats


# Known sources of run-to-run output variation, recorded in every manifest so
# a reader comparing two runs of the same input knows what may legitimately
# differ (reproducibility requirement, plan task A2).
NON_DETERMINISM_NOTES = [
    "GPU PatchMatch stereo (dense reconstruction) is non-deterministic; "
    "depth maps and fused point counts vary between runs.",
    "Feature matching and incremental mapping are multi-threaded; "
    "bundle-adjusted poses (and undistorted image sizes derived from them) "
    "vary slightly between runs, as can sparse model selection.",
    "Poisson reconstruction, SOR, cropping and scale recovery are "
    "deterministic given identical inputs.",
]


def build_provenance(
    frames_manifest: Path | None,
    resolved_configs: dict,
) -> dict:
    """Reproducibility block for pipeline_manifest.json.

    Records the library versions, the SHA-256 of the input frames manifest
    (None when the run had no manifest), the fully-resolved config values the
    run actually used, and the known non-determinism sources.
    """
    manifest_sha256 = None
    if frames_manifest is not None and Path(frames_manifest).exists():
        manifest_sha256 = hashlib.sha256(Path(frames_manifest).read_bytes()).hexdigest()
    return {
        "environment": {
            "python": platform.python_version(),
            "pycolmap": pycolmap.__version__,
            "open3d": o3d.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "frames_manifest_sha256": manifest_sha256,
        "resolved_configs": resolved_configs,
        "non_determinism_notes": NON_DETERMINISM_NOTES,
    }


def with_fusion_mask_provenance(
    provenance: dict,
    enabled: bool,
    source_mask_dir: Path | None = None,
    workspace_mask_dir: Path | None = None,
    stats: dict | None = None,
) -> dict:
    """Record whether stereo fusion was mask-restricted.

    Always writes the ``fusion_masks`` block, including ``enabled: false``. A run
    that simply omits the key is indistinguishable from one produced before
    fusion masking existed, which makes past clouds unattributable.
    """
    block: dict = {"enabled": enabled}
    if enabled:
        block["source_mask_dir"] = str(source_mask_dir)
        block["workspace_mask_dir"] = str(workspace_mask_dir)
        block.update(stats or {})
    provenance["fusion_masks"] = block
    return provenance


def with_membrane_filter_provenance(
    provenance: dict,
    enabled: bool,
    stats: dict | None = None,
) -> dict:
    """Record whether the dense cloud was membrane-filtered before Poisson.

    Always writes the ``membrane_filter`` block, including ``enabled: false``,
    for the same reason as the fusion-mask block: a run that omits the key is
    indistinguishable from one produced before the filter existed, which makes
    past meshes unattributable.
    """
    block: dict = {"enabled": enabled}
    if enabled:
        block.update(stats or {})
    provenance["membrane_filter"] = block
    return provenance


def run_membrane_filter(
    input_ply: Path,
    output_dir: Path,
    marker_corners: dict | None,
    pale_threshold: float,
    marker_margin_mm: float,
    scale_factor: float | None,
) -> tuple[Path, dict]:
    """Filter pale membrane points out of the cropped cloud before Poisson.

    The cloud is still in SfM units at this point in the pipeline, so the
    millimetre margin is converted with the recovered scale. Without a scale
    factor the conversion is impossible and the filter is skipped rather than
    guessed at.

    Returns:
        (input_for_poisson, stats) — the filtered PLY when the filter ran, the
        untouched input otherwise.
    """
    if not marker_corners:
        logger.warning(
            "Membrane filter requested but no triangulated ArUco markers are "
            "available to protect — skipping (marker faces are the scale anchor)."
        )
        return input_ply, {"enabled": True, "applied": False,
                          "skipped_reason": "no triangulated markers to protect"}
    if not scale_factor:
        logger.warning(
            "Membrane filter requested but no metric scale was recovered, so the "
            "millimetre marker margin cannot be converted to SfM units — skipping."
        )
        return input_ply, {"enabled": True, "applied": False,
                           "skipped_reason": "no scale factor for margin conversion"}

    logger.info("=== Membrane filter (colour-based, marker-protected) ===")
    pcd = o3d.io.read_point_cloud(str(input_ply))
    filtered, stats = filter_membrane_points(
        pcd,
        marker_corners,
        pale_threshold=pale_threshold,
        marker_margin=marker_margin_mm / scale_factor,
    )
    stats["marker_margin_mm"] = float(marker_margin_mm)
    if not stats.get("applied"):
        return input_ply, stats

    out_ply = output_dir / "dense_filtered_cropped_membrane.ply"
    o3d.io.write_point_cloud(str(out_ply), filtered)
    logger.info("Membrane-filtered cloud saved to '%s'", out_ply)
    return out_ply, stats


def write_pipeline_manifest(
    output_dir: Path,
    run_script: str,
    sor_stats: dict,
    lcc_stats: dict,
    mesh_opts: dict,
    scale_factor: float | None,
    scale_sanity: dict | None = None,
    scale_self_consistency: dict | None = None,
    scale_status: dict | None = None,
    provenance: dict | None = None,
) -> None:
    """Write pipeline_manifest.json to output_dir."""
    manifest = {
        "run_script": run_script,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        **sor_stats,
        "poisson_surface_reconstruction": {
            "depth": mesh_opts["depth"],
            "scale": mesh_opts["scale"],
            "linear_fit": mesh_opts["linear_fit"],
            "density_threshold": mesh_opts.get("density_threshold", 0.01),
        },
        "taubin_smoothing": mesh_opts.get("taubin_smoothing"),
        "scale_factor_mm_per_unit": scale_factor,
        **lcc_stats,
    }
    # Explicit, unmissable scale state. `scale_factor_mm_per_unit` above is kept
    # for backwards compatibility, but a bare null there is easy to miss and
    # cannot distinguish "recovered but never validated" from "validated".
    if scale_status is not None:
        manifest["scale"] = scale_status
    if scale_sanity is not None:
        manifest["scale_sanity_check"] = scale_sanity
    if scale_self_consistency is not None:
        manifest["scale_self_consistency"] = scale_self_consistency
    if provenance is not None:
        manifest.update(provenance)
    out = output_dir / "pipeline_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    logger.info("Pipeline manifest written to '%s'", out)
