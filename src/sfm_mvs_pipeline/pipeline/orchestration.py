"""Shared post-processing helpers used by all pipeline entry-point scripts.

Orchestrates SOR → visualize → Poisson → LCC → Taubin → visualize and writes
pipeline_manifest.json. Each script calls these functions after stereo fusion,
inserting scale recovery in between (which is script-specific).
"""

import datetime
import json
import logging
from pathlib import Path

import open3d as o3d

from sfm_mvs_pipeline.mesh.surface_reconstruction import _apply_lcc, _apply_taubin, _run_poisson
from sfm_mvs_pipeline.postprocess.point_cloud_filter import filter_point_cloud
from sfm_mvs_pipeline.visualization.plotly_viz import save_mesh_html, save_point_cloud_html

logger = logging.getLogger(__name__)


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


def write_pipeline_manifest(
    output_dir: Path,
    run_script: str,
    sor_stats: dict,
    lcc_stats: dict,
    mesh_opts: dict,
    scale_factor: float | None,
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
    out = output_dir / "pipeline_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    logger.info("Pipeline manifest written to '%s'", out)
