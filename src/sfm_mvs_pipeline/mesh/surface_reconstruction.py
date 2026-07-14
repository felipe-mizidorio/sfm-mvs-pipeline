import logging
from pathlib import Path

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def _run_poisson(
    pcd: o3d.geometry.PointCloud,
    options: dict,
) -> o3d.geometry.TriangleMesh:
    """Poisson reconstruction + density trim. Returns the pre-LCC mesh."""
    if not pcd.has_normals():
        logger.info("Point cloud has no normals — estimating normals")
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=30)

    logger.info(
        "Running Poisson Surface Reconstruction (depth=%d, scale=%.2f, linear_fit=%s)",
        options["depth"],
        options["scale"],
        options["linear_fit"],
    )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd=pcd,
        depth=options["depth"],
        scale=options["scale"],
        linear_fit=options["linear_fit"],
    )

    density_threshold = options.get("density_threshold", 0.01)
    densities_np = np.asarray(densities)
    cutoff = np.quantile(densities_np, density_threshold)
    vertices_to_remove = densities_np < cutoff
    mesh.remove_vertices_by_mask(vertices_to_remove)
    logger.info(
        "Density trimming: removed %d low-density vertices (quantile=%.3f)",
        int(vertices_to_remove.sum()),
        density_threshold,
    )
    return mesh


def _apply_lcc(
    mesh: o3d.geometry.TriangleMesh,
    options: dict,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Keep only the largest connected component. Returns (mesh, stats_dict)."""
    if not options.get("keep_largest_component", True):
        return mesh, {}

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    largest_idx = int(cluster_n_triangles.argmax())
    triangles_to_remove = triangle_clusters != largest_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    stats = {
        "lcc": {
            "triangles_kept": int(cluster_n_triangles[largest_idx]),
            "triangles_removed": int(triangles_to_remove.sum()),
            "components_removed": len(cluster_n_triangles) - 1,
        }
    }
    logger.info(
        "LCC filter: kept %d triangles, removed %d across %d smaller component(s)",
        stats["lcc"]["triangles_kept"],
        stats["lcc"]["triangles_removed"],
        stats["lcc"]["components_removed"],
    )
    return mesh, stats


def _apply_taubin(
    mesh: o3d.geometry.TriangleMesh,
    options: dict,
) -> o3d.geometry.TriangleMesh:
    """Apply Taubin smoothing to reduce high-frequency wrinkles from MVS noise.

    No-op when options has no 'taubin_smoothing' key or iterations == 0.
    Returns a new TriangleMesh (Open3D filter_smooth_taubin does not modify in-place).
    """
    smoothing = options.get("taubin_smoothing", {})
    if not smoothing:
        return mesh
    n_iter = int(smoothing.get("iterations", 10))
    if n_iter == 0:
        return mesh
    lambda_filter = float(smoothing.get("lambda_filter", 0.5))
    mu = float(smoothing.get("mu", -0.53))
    logger.info(
        "Taubin smoothing: %d iterations (lambda=%.2f, mu=%.2f)",
        n_iter,
        lambda_filter,
        mu,
    )
    return mesh.filter_smooth_taubin(
        number_of_iterations=n_iter,
        lambda_filter=lambda_filter,
        mu=mu,
    )


def reconstruct_surface(
    input_ply: Path,
    output_ply: Path,
    options: dict,
) -> o3d.geometry.TriangleMesh:
    if not input_ply.exists():
        raise FileNotFoundError(f"Input point cloud not found: {input_ply}")

    pcd = o3d.io.read_point_cloud(str(input_ply))
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud '{input_ply}' contains no points.")

    mesh = _run_poisson(pcd, options)
    mesh, _ = _apply_lcc(mesh, options)
    mesh = _apply_taubin(mesh, options)

    logger.info(
        "Mesh reconstructed: %d vertices, %d triangles",
        len(mesh.vertices),
        len(mesh.triangles),
    )

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_ply), mesh)
    logger.info("Mesh saved to '%s'", output_ply)
    return mesh
