"""Re-run stereo fusion (optionally with bbox), SOR, crop to head region,
then scale recovery and Poisson + LCC mesh.

Typical usage (no bbox — full fusion, post-fusion SOR + spatial crop):
    uv run python scripts/resume_from_mvs.py \\
        --output-dir data/processed/<session> \\
        --image-dir path/to/filtered/frames \\
        --frames-manifest path/to/manifest.json \\
        --head-radius 1.5

Optionally add --bbox-min / --bbox-max to also clip at the fusion step.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
import yaml

from sfm_mvs_pipeline.mvs.fusion import fuse_depth_maps
from sfm_mvs_pipeline.pipeline.orchestration import (
    run_poisson_lcc_and_visualize,
    run_sor_and_visualize,
    write_pipeline_manifest,
)
from sfm_mvs_pipeline.scale.aruco_scale import (
    apply_scale_to_mesh,
    apply_scale_to_ply,
    recover_scale_safe,
)
from sfm_mvs_pipeline.sfm.reconstruction import load_best_reconstruction

_REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _estimate_head_center(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """Least-squares intersection of camera optical axes → approximate head center."""
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


def _crop_to_sphere(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-fuse depth maps, SOR, crop to head sphere, scale recovery, Poisson + LCC."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--frames-manifest", default=None, type=Path)
    parser.add_argument("--aruco-config", default=_REPO_ROOT / "configs/aruco.yaml", type=Path)
    parser.add_argument("--mesh-config", default=_REPO_ROOT / "configs/mesh.yaml", type=Path)
    parser.add_argument("--colmap-config", default=_REPO_ROOT / "configs/colmap.yaml", type=Path)
    parser.add_argument(
        "--bbox-min",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional fusion-time bbox min (SfM units) — coarse background cut.",
    )
    parser.add_argument(
        "--bbox-max",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional fusion-time bbox max (SfM units) — coarse background cut.",
    )
    parser.add_argument(
        "--head-radius",
        type=float,
        default=1.5,
        help="Post-fusion spherical crop radius in SfM units (default 1.5 ≈ 230 mm). "
        "0 = no crop.",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip stereo fusion and reuse the existing dense.ply (already in SfM units).",
    )
    args = parser.parse_args()

    with args.aruco_config.open() as f:
        aruco_cfg = yaml.safe_load(f).get("aruco", {})
    with args.mesh_config.open() as f:
        mesh_cfg = yaml.safe_load(f)
    with args.colmap_config.open() as f:
        colmap_cfg = yaml.safe_load(f)
    filter_cfg = mesh_cfg["point_cloud_filtering"]

    output_dir: Path = args.output_dir
    mvs_dir = output_dir / "mvs"
    sparse_dir = output_dir / "sparse"
    dense_ply = output_dir / "dense.ply"
    mesh_ply = output_dir / "mesh.ply"
    cropped_ply = output_dir / "dense_filtered_cropped.ply"

    manifest_detections = None
    if args.frames_manifest is not None:
        manifest_data = json.loads(args.frames_manifest.read_text())
        manifest_detections = manifest_data.get("marker_detections")

    reconstruction, best_sparse = load_best_reconstruction(sparse_dir)
    logger.info(
        "Loaded sparse model from '%s': %d registered images",
        best_sparse,
        reconstruction.num_reg_images(),
    )

    # --- Step 1: Stereo fusion ---
    if args.skip_fusion:
        logger.info("Skipping stereo fusion (--skip-fusion). Using existing '%s'.", dense_ply)
    else:
        logger.info("=== Stereo fusion ===")
        fuse_depth_maps(
            mvs_path=mvs_dir,
            output_path=dense_ply,
            options=colmap_cfg["stereo_fusion"],
            bbox_min=args.bbox_min,
            bbox_max=args.bbox_max,
        )

    # --- Step 2: SOR on raw dense cloud ---
    logger.info("=== Point cloud filtering (SOR) ===")
    dense_filtered_ply, sor_stats = run_sor_and_visualize(dense_ply, output_dir, filter_cfg)

    # --- Step 3: Post-fusion spherical crop (on SOR-filtered cloud) ---
    input_for_poisson = dense_filtered_ply
    if args.head_radius > 0:
        logger.info("=== Post-fusion spherical crop (radius=%.3f SfM units) ===", args.head_radius)
        head_center = _estimate_head_center(reconstruction)
        logger.info("Head center (SfM): %s", np.round(head_center, 4))
        pcd = o3d.io.read_point_cloud(str(dense_filtered_ply))
        logger.info("Dense filtered cloud before crop: %d points", len(pcd.points))
        pcd_crop = _crop_to_sphere(pcd, head_center, args.head_radius)
        if len(pcd_crop.points) == 0:
            logger.warning(
                "Crop removed all points! Falling back to SOR-filtered dense cloud. "
                "Try increasing --head-radius."
            )
            pcd_crop = pcd
        o3d.io.write_point_cloud(str(cropped_ply), pcd_crop)
        logger.info(
            "Cropped dense cloud: %d points, saved to '%s'",
            len(pcd_crop.points),
            cropped_ply,
        )
        input_for_poisson = cropped_ply

    # --- Step 4: Scale recovery ---
    marker_length_mm = aruco_cfg.get("marker_length_mm")
    scale_factor = recover_scale_safe(
        reconstruction=reconstruction,
        image_dir=args.image_dir,
        marker_length_mm=float(marker_length_mm) if marker_length_mm else None,
        aruco_dict_id=int(aruco_cfg.get("dict_id", 0)),
        detections=manifest_detections,
        min_views=int(aruco_cfg.get("min_views", 2)),
    )

    # --- Step 5: Poisson + LCC + visualization ---
    logger.info("=== Poisson surface reconstruction + LCC ===")
    _, lcc_stats = run_poisson_lcc_and_visualize(
        input_for_poisson, mesh_ply, output_dir, mesh_cfg["poisson_surface_reconstruction"]
    )

    # --- Step 6: Apply metric scale once, after meshing ---
    if scale_factor is not None:
        apply_scale_to_ply(dense_ply, scale_factor)
        apply_scale_to_ply(dense_filtered_ply, scale_factor)
        if input_for_poisson == cropped_ply:
            apply_scale_to_ply(cropped_ply, scale_factor)
        apply_scale_to_mesh(mesh_ply, scale_factor)
        logger.info("Applied scale %.6f mm/unit to outputs.", scale_factor)

    write_pipeline_manifest(
        output_dir,
        "resume_from_mvs.py",
        sor_stats,
        lcc_stats,
        mesh_cfg["poisson_surface_reconstruction"],
        scale_factor,
    )

    logger.info("Done. Outputs in '%s'", output_dir)


if __name__ == "__main__":
    main()
