import logging
from pathlib import Path

import open3d as o3d

logger = logging.getLogger(__name__)


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

    if not pcd.has_normals():
        logger.info("Point cloud has no normals — estimating normals")
        pcd.estimate_normals()

    logger.info(
        "Running Poisson Surface Reconstruction (depth=%d, scale=%.2f, linear_fit=%s)",
        options["depth"],
        options["scale"],
        options["linear_fit"],
    )

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd=pcd,
        depth=options["depth"],
        scale=options["scale"],
        linear_fit=options["linear_fit"],
    )

    logger.info(
        "Mesh reconstructed: %d vertices, %d triangles",
        len(mesh.vertices),
        len(mesh.triangles),
    )

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_ply), mesh)
    logger.info("Mesh saved to '%s'", output_ply)

    return mesh
