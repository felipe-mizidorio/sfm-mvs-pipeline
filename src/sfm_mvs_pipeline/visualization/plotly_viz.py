"""Plotly-based visualizations of point clouds and meshes, saved as standalone HTML."""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def save_point_cloud_html(
    pcd: o3d.geometry.PointCloud,
    path: Path,
    title: str,
    max_points: int = 200_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(pcd.points)
    idx = None
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        logger.info(
            "Downsampled point cloud to %d points for '%s'", max_points, path.name
        )

    colors = None
    if pcd.has_colors():
        clrs = np.asarray(pcd.colors)
        if idx is not None:
            clrs = clrs[idx]
        colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in clrs]

    fig = go.Figure(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=1, color=colors or pts[:, 2], colorscale="Viridis"),
        )
    )
    fig.update_layout(title=title, scene=dict(aspectmode="data"))
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Saved point cloud HTML to '%s'", path)


def save_mesh_html(
    mesh: o3d.geometry.TriangleMesh,
    path: Path,
    title: str,
    max_vertices: int = 200_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    if len(verts) > max_vertices:
        target = int(len(tris) * max_vertices / len(verts))
        mesh = mesh.simplify_quadric_decimation(target)
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        logger.info("Decimated mesh to %d vertices for '%s'", len(verts), path.name)

    colors = None
    if mesh.has_vertex_colors():
        clrs = np.asarray(mesh.vertex_colors)
        colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in clrs]

    fig = go.Figure(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=tris[:, 0],
            j=tris[:, 1],
            k=tris[:, 2],
            vertexcolor=colors,
            colorscale="Viridis" if colors is None else None,
            intensity=verts[:, 2] if colors is None else None,
        )
    )
    fig.update_layout(title=title, scene=dict(aspectmode="data"))
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Saved mesh HTML to '%s'", path)
