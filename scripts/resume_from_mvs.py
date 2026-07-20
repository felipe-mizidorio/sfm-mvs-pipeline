"""Re-run stereo fusion (optionally with bbox), SOR, crop to head region,
then scale recovery and Poisson + LCC mesh.

Typical usage (no bbox — full fusion, SOR + automatic ArUco-derived head crop):
    uv run python scripts/resume_from_mvs.py \\
        --output-dir data/processed/<session> \\
        --image-dir path/to/filtered/frames \\
        --frames-manifest path/to/manifest.json

The head-crop radius is derived automatically from the triangulated ArUco
markers; --head-radius is a debug-only override (0 disables the crop).
Optionally add --bbox-min / --bbox-max to also clip at the fusion step.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from sfm_mvs_pipeline.mvs.fusion import fuse_depth_maps
from sfm_mvs_pipeline.mvs.mask_undistortion import undistort_masks_safe
from sfm_mvs_pipeline.pipeline.orchestration import (
    build_provenance,
    run_head_crop,
    run_poisson_lcc_and_visualize,
    run_sor_and_visualize,
    with_fusion_mask_provenance,
    write_pipeline_manifest,
)
from sfm_mvs_pipeline.scale.aruco_scale import (
    apply_scale_to_mesh,
    apply_scale_to_ply,
    recover_scale_details_safe,
)
from sfm_mvs_pipeline.scale.layout_check import check_marker_layout
from sfm_mvs_pipeline.scale.self_consistency import check_scale_self_consistency
from sfm_mvs_pipeline.sfm.reconstruction import load_best_reconstruction

_REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _guard_against_double_scale(output_dir: Path) -> None:
    """Refuse --skip-fusion when a previous resume run already scaled dense.ply.

    resume_from_mvs.py scales dense.ply in place after meshing. Re-running with
    --skip-fusion would re-derive the scale from the (unscaled) sparse model and
    apply it again to the already-scaled cloud — silently producing geometry
    scale² times too large. Fail loudly instead.
    """
    prev_manifest_path = output_dir / "pipeline_manifest.json"
    if not prev_manifest_path.exists():
        return
    prev = json.loads(prev_manifest_path.read_text())
    if prev.get("run_script") == "resume_from_mvs.py" and prev.get(
        "scale_factor_mm_per_unit"
    ):
        logger.error(
            "dense.ply in '%s' was already scaled to millimetres by a previous "
            "resume_from_mvs.py run (scale %.6f mm/unit, see pipeline_manifest.json). "
            "Running with --skip-fusion would double-scale it. Re-run without "
            "--skip-fusion to regenerate dense.ply from mvs/, or delete "
            "pipeline_manifest.json if dense.ply was replaced manually.",
            output_dir,
            prev["scale_factor_mm_per_unit"],
        )
        sys.exit(1)


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
        default=None,
        help="DEBUG override for the spherical head-crop radius, in SfM units. "
        "Not needed in normal use: the radius is auto-derived from the "
        "triangulated ArUco markers and marker_length_mm. 0 disables the crop.",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip stereo fusion and reuse the existing dense.ply (already in SfM units).",
    )
    parser.add_argument(
        "--fusion-masks",
        action="store_true",
        help="Warp the frames-manifest masks into the undistorted MVS workspace and "
        "restrict stereo fusion to them. Requires --frames-manifest with a 'mask_dir'.",
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

    if args.skip_fusion:
        _guard_against_double_scale(output_dir)

    manifest_detections = None
    manifest_data = None
    if args.frames_manifest is not None:
        manifest_data = json.loads(args.frames_manifest.read_text())
        manifest_detections = manifest_data.get("marker_detections")

    # Masks live next to the frames the manifest describes, exactly as in
    # run_pipeline.py: <image-dir>/<manifest mask_dir>.
    mask_path: Path | None = None
    if args.fusion_masks:
        if manifest_data is None:
            logger.error("--fusion-masks requires --frames-manifest. Aborting.")
            sys.exit(1)
        if not manifest_data.get("mask_dir"):
            logger.error(
                "--fusion-masks requested but frames manifest '%s' has no 'mask_dir'. Aborting.",
                args.frames_manifest,
            )
            sys.exit(1)
        candidate = args.image_dir / manifest_data["mask_dir"]
        if not candidate.is_dir():
            logger.error("Mask directory '%s' does not exist. Aborting.", candidate)
            sys.exit(1)
        mask_path = candidate
        logger.info("Using mask directory: '%s'", mask_path)

    reconstruction, best_sparse = load_best_reconstruction(sparse_dir)
    logger.info(
        "Loaded sparse model from '%s': %d registered images",
        best_sparse,
        reconstruction.num_reg_images(),
    )

    # --- Step 1: Stereo fusion ---
    fusion_mask_dir: Path | None = None
    fusion_mask_stats: dict | None = None
    if args.skip_fusion:
        logger.info("Skipping stereo fusion (--skip-fusion). Using existing '%s'.", dense_ply)
    else:
        if mask_path is not None:
            logger.info("=== Undistorting masks for stereo fusion ===")
            fusion_mask_dir, fusion_mask_stats = undistort_masks_safe(
                mask_path=mask_path,
                original_sparse_path=best_sparse,
                mvs_path=mvs_dir,
            )
        logger.info("=== Stereo fusion ===")
        fusion_start = time.perf_counter()
        fuse_depth_maps(
            mvs_path=mvs_dir,
            output_path=dense_ply,
            options=colmap_cfg["stereo_fusion"],
            bbox_min=args.bbox_min,
            bbox_max=args.bbox_max,
            mask_path=fusion_mask_dir,
        )
        logger.info("Stereo fusion took %.1f s", time.perf_counter() - fusion_start)

    # --- Step 2: SOR on raw dense cloud ---
    logger.info("=== Point cloud filtering (SOR) ===")
    dense_filtered_ply, sor_stats = run_sor_and_visualize(dense_ply, output_dir, filter_cfg)

    # --- Step 3: Scale recovery (before the crop: the auto crop radius is
    # derived in millimetres and converted to SfM units via the scale) ---
    marker_length_mm = aruco_cfg.get("marker_length_mm")
    scale_factor, marker_points, corners_by_marker = recover_scale_details_safe(
        reconstruction=reconstruction,
        image_dir=args.image_dir,
        marker_length_mm=float(marker_length_mm) if marker_length_mm else None,
        aruco_dict_id=int(aruco_cfg.get("dict_id", 0)),
        detections=manifest_detections,
        min_views=int(aruco_cfg.get("min_views", 2)),
    )
    scale_sanity = check_marker_layout(
        corners_by_marker or {}, scale_factor, aruco_cfg.get("layout_check")
    )
    scale_self_consistency = check_scale_self_consistency(
        corners_by_marker or {}, float(marker_length_mm) if marker_length_mm else None
    )

    # --- Step 4: Post-fusion spherical crop (on SOR-filtered cloud) ---
    input_for_poisson, crop_stats = run_head_crop(
        dense_filtered_ply,
        output_dir,
        reconstruction,
        head_radius_override=args.head_radius,
        scale_factor=scale_factor,
        marker_points=marker_points,
    )
    sor_stats.update(crop_stats)

    # --- Step 5: Poisson + LCC + visualization ---
    logger.info("=== Poisson surface reconstruction + LCC ===")
    _, lcc_stats = run_poisson_lcc_and_visualize(
        input_for_poisson, mesh_ply, output_dir, mesh_cfg["poisson_surface_reconstruction"]
    )

    # --- Step 6: Apply metric scale once, after meshing ---
    if scale_factor is not None:
        apply_scale_to_ply(dense_ply, scale_factor)
        apply_scale_to_ply(dense_filtered_ply, scale_factor)
        if input_for_poisson != dense_filtered_ply:
            apply_scale_to_ply(input_for_poisson, scale_factor)
        apply_scale_to_mesh(mesh_ply, scale_factor)
        logger.info("Applied scale %.6f mm/unit to outputs.", scale_factor)

    write_pipeline_manifest(
        output_dir,
        "resume_from_mvs.py",
        sor_stats,
        lcc_stats,
        mesh_cfg["poisson_surface_reconstruction"],
        scale_factor,
        scale_sanity=scale_sanity,
        scale_self_consistency=scale_self_consistency,
        provenance=with_fusion_mask_provenance(
            build_provenance(
                args.frames_manifest,
                {"aruco": aruco_cfg, "colmap": colmap_cfg, "mesh": mesh_cfg},
            ),
            enabled=fusion_mask_dir is not None,
            source_mask_dir=mask_path,
            workspace_mask_dir=fusion_mask_dir,
            stats=fusion_mask_stats,
        ),
    )

    logger.info("Done. Outputs in '%s'", output_dir)


if __name__ == "__main__":
    main()
