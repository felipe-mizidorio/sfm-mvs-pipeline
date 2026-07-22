"""End-to-end SfM + MVS + mesh + evaluation pipeline."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pycolmap
import yaml

from sfm_mvs_pipeline.evaluation.metrics import evaluate
from sfm_mvs_pipeline.mvs.dense_reconstruction import run_dense_reconstruction
from sfm_mvs_pipeline.mvs.fusion import fuse_depth_maps
from sfm_mvs_pipeline.mvs.mask_undistortion import undistort_masks_safe
from sfm_mvs_pipeline.pipeline.orchestration import (
    build_provenance,
    run_head_crop,
    run_membrane_filter,
    run_poisson_lcc_and_visualize,
    run_sor_and_visualize,
    with_fusion_mask_provenance,
    with_membrane_filter_provenance,
    write_pipeline_manifest,
)
from sfm_mvs_pipeline.postprocess.membrane_filter import (
    DEFAULT_MARKER_MARGIN_MM,
    DEFAULT_PALE_THRESHOLD,
)
from sfm_mvs_pipeline.scale.aruco_scale import (
    apply_scale_to_mesh,
    apply_scale_to_ply,
    recover_scale_details_safe,
)
from sfm_mvs_pipeline.scale.layout_check import check_marker_layout
from sfm_mvs_pipeline.scale.policy import (
    UnscaledOutputError,
    enforce_scale_policy,
    resolve_scale_status,
    unscaled_artifact_path,
)
from sfm_mvs_pipeline.scale.self_consistency import check_scale_self_consistency
from sfm_mvs_pipeline.sfm.feature_extraction import (
    camera_prior_from_manifest,
    extract_features,
)
from sfm_mvs_pipeline.sfm.feature_matching import match_features
from sfm_mvs_pipeline.sfm.reconstruction import run_incremental_mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SfM-MVS-mesh pipeline.")
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory for all pipeline outputs.",
    )
    parser.add_argument(
        "--colmap-config",
        default=_REPO_ROOT / "configs/colmap.yaml",
        type=Path,
        help="Path to colmap.yaml config file.",
    )
    parser.add_argument(
        "--mesh-config",
        default=_REPO_ROOT / "configs/mesh.yaml",
        type=Path,
        help="Path to mesh.yaml config file.",
    )
    parser.add_argument(
        "--evaluation-config",
        default=_REPO_ROOT / "configs/evaluation.yaml",
        type=Path,
        help="Path to evaluation.yaml config file.",
    )
    parser.add_argument(
        "--aruco-config",
        default=_REPO_ROOT / "configs/aruco.yaml",
        type=Path,
        help="Path to aruco.yaml config file for metric scale recovery.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground truth .ply for evaluation (optional).",
    )
    parser.add_argument(
        "--skip-mvs",
        action="store_true",
        help="Stop after sparse reconstruction (useful on CPU-only machines).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="Device for pycolmap feature extraction, matching, and mapping.",
    )
    # --- Camera calibration (G2) ---
    parser.add_argument(
        "--camera-model",
        default=None,
        type=str,
        help=(
            "COLMAP camera model name (e.g. OPENCV, PINHOLE, SIMPLE_RADIAL). "
            "When provided, --camera-params must also be given and a single "
            "shared camera is used for all images."
        ),
    )
    parser.add_argument(
        "--camera-params",
        default=None,
        type=str,
        help=(
            "Space-separated camera intrinsics matching the chosen model. "
            "For OPENCV: 'fx fy cx cy k1 k2 p1 p2'. "
            "For PINHOLE: 'fx fy cx cy'."
        ),
    )
    # --- Preprocessing manifest (G3) ---
    parser.add_argument(
        "--frames-manifest",
        default=None,
        type=Path,
        help=(
            "Path to a JSON manifest produced by the ArUco preprocessing pipeline. "
            'Expected keys: "frames" (list of image filenames to use) and optionally '
            '"marker_detections" ({frame: [{id, corners}]}) for scale recovery.'
        ),
    )
    # --- Head crop (debug override only) ---
    parser.add_argument(
        "--head-radius",
        type=float,
        default=None,
        help="DEBUG override for the spherical head-crop radius, in SfM units. "
        "Not needed in normal use: the radius is auto-derived from the "
        "triangulated ArUco markers and marker_length_mm. 0 disables the crop.",
    )
    # --- Bounding-box clipping (G4) ---
    parser.add_argument(
        "--bbox-min",
        default=None,
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Minimum corner of axis-aligned bounding box for stereo fusion clipping.",
    )
    parser.add_argument(
        "--bbox-max",
        default=None,
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Maximum corner of axis-aligned bounding box for stereo fusion clipping.",
    )
    parser.add_argument(
        "--fusion-masks",
        action="store_true",
        help="EXPERIMENTAL. Also restrict stereo fusion to the frames-manifest masks "
        "(warped into the undistorted MVS workspace). Off by default: with the "
        "current ArUco convex-hull masks this deletes genuine head surface away "
        "from the markers without reducing silhouette bleed — see "
        "docs/fusion_masks_report.md. Masks always apply to feature extraction "
        "regardless of this flag.",
    )
    parser.add_argument(
        "--membrane-filter",
        action="store_true",
        help="Remove pale 'membrane' contamination from the cropped cloud before "
        "Poisson, protecting the white ArUco marker faces. Off by default because "
        "it is SCENE-DEPENDENT: it assumes a dark subject against pale "
        "contamination and would delete the subject in a capture where the "
        "subject is pale. See docs/membrane_filter_report.md.",
    )
    parser.add_argument(
        "--allow-unscaled",
        action="store_true",
        help="Continue even if metric scale recovery fails, writing output in "
        "arbitrary SfM units. OFF by default: without this flag a failed scale "
        "recovery is a hard stop, because the alternative is a complete, "
        "plausible-looking mesh whose numbers are not millimetres. Artefacts "
        "written under this flag are renamed to *.UNSCALED_sfm_units.* and the "
        "manifest records scale.status 'unscaled'.",
    )
    parser.add_argument(
        "--membrane-pale-threshold",
        type=float,
        default=DEFAULT_PALE_THRESHOLD,
        help="Mean RGB (0-255) at or above which a point counts as pale.",
    )
    parser.add_argument(
        "--membrane-marker-margin-mm",
        type=float,
        default=DEFAULT_MARKER_MARGIN_MM,
        help="Protection margin added to each marker's own corner extent, in mm.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    colmap_cfg = _load_yaml(args.colmap_config)
    mesh_cfg = _load_yaml(args.mesh_config)
    eval_cfg = _load_yaml(args.evaluation_config)
    aruco_cfg = _load_yaml(args.aruco_config).get("aruco", {})
    filter_cfg = mesh_cfg["point_cloud_filtering"]

    device = pycolmap.Device.cpu if args.device == "cpu" else pycolmap.Device.auto

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / "database.db"
    if database_path.exists():
        logger.warning("Removing stale database '%s'", database_path)
        database_path.unlink()
    sparse_dir = output_dir / "sparse"
    mvs_dir = output_dir / "mvs"
    dense_ply = output_dir / "dense.ply"
    mesh_ply = output_dir / "mesh.ply"
    results_dir = output_dir / "results"
    metrics_json = results_dir / "metrics.json"

    # --- Parse preprocessing manifest (G3) ---
    manifest_frames: list[str] | None = None
    manifest_detections: dict | None = None
    manifest_data: dict = {}
    mask_path: Path | None = None
    if args.frames_manifest is not None:
        manifest_data = json.loads(args.frames_manifest.read_text())
        manifest_frames = manifest_data.get("frames")
        manifest_detections = manifest_data.get("marker_detections")
        logger.info(
            "Frames manifest loaded: %d frames, %d pre-detected markers",
            len(manifest_frames or []),
            len(manifest_detections or {}),
        )
        if manifest_data.get("mask_dir"):
            candidate_mask_path = Path(args.image_dir) / manifest_data["mask_dir"]
            if candidate_mask_path.is_dir():
                mask_path = candidate_mask_path
                logger.info("Using mask directory: '%s'", mask_path)
            else:
                logger.warning(
                    "Manifest mask_dir '%s' does not exist — ignoring masks.",
                    candidate_mask_path,
                )

    # --- Camera intrinsics: explicit flags > EXIF-derived prior > shared
    # self-calibration. Same-device video always shares one camera model.
    camera_model, camera_params = args.camera_model, args.camera_params
    if camera_model or camera_params:
        intrinsics_source = "explicit"
    else:
        prior = camera_prior_from_manifest(manifest_data)
        if prior is not None:
            camera_model, camera_params = prior
            intrinsics_source = "exif_prior"
            logger.info(
                "Camera prior from manifest focal metadata: %s (%s)",
                camera_model,
                camera_params,
            )
        else:
            intrinsics_source = "self_calibration_shared"
    logger.info("Intrinsics source: %s", intrinsics_source)

    # --- Step 1/7: Feature extraction ---
    logger.info("=== Step 1/7: Feature extraction ===")
    extract_features(
        database_path=database_path,
        image_dir=args.image_dir,
        options=colmap_cfg["feature_extraction"],
        device=device,
        camera_model=camera_model,
        camera_params=camera_params,
        image_names=manifest_frames,
        mask_path=mask_path,
        shared_camera=True,
    )

    # --- Step 2/7: Feature matching ---
    logger.info("=== Step 2/7: Feature matching ===")
    match_features(
        database_path=database_path,
        options=colmap_cfg["feature_matching"],
        device=device,
    )

    # --- Step 3/7: Sparse reconstruction ---
    logger.info("=== Step 3/7: Sparse (incremental) reconstruction ===")
    reconstructions = run_incremental_mapping(
        database_path=database_path,
        image_dir=args.image_dir,
        output_path=sparse_dir,
        options=colmap_cfg["incremental_mapping"],
        device=device,
    )

    if not reconstructions:
        logger.error("No models reconstructed. Check features and matches.")
        sys.exit(1)

    # Use the model with the most registered images.
    best_model_idx = max(
        reconstructions, key=lambda k: reconstructions[k].num_reg_images()
    )
    sparse_model_path = sparse_dir / str(best_model_idx)
    logger.info(
        "Using sparse model %d (%d images)",
        best_model_idx,
        reconstructions[best_model_idx].num_reg_images(),
    )

    if args.skip_mvs:
        logger.info("--skip-mvs set: stopping after sparse reconstruction.")
        return

    # --- Step 4/7: Dense reconstruction (undistortion + PatchMatch Stereo) ---
    logger.info("=== Step 4/7: Dense reconstruction ===")
    run_dense_reconstruction(
        sparse_path=sparse_model_path,
        image_dir=args.image_dir,
        mvs_path=mvs_dir,
        options=colmap_cfg["patch_match_stereo"],
    )

    # --- Step 4b/7: Warp masks into the undistorted MVS workspace ---
    # Feature-extraction masks never reach MVS, so they are re-projected here for
    # StereoFusionOptions.mask_path. Opt-in: measured on video_test_20260716_115516,
    # ArUco hull masks at fusion cut 9.3% of mesh area inside the head sphere while
    # leaving silhouette contamination flat (docs/fusion_masks_report.md).
    fusion_mask_dir: Path | None = None
    fusion_mask_stats: dict | None = None
    if args.fusion_masks and mask_path is None:
        logger.warning(
            "--fusion-masks requested but no mask directory is available; "
            "fusing unmasked."
        )
    if args.fusion_masks and mask_path is not None:
        logger.info("=== Step 4b/7: Undistorting masks for stereo fusion ===")
        fusion_mask_dir, fusion_mask_stats = undistort_masks_safe(
            mask_path=mask_path,
            original_sparse_path=sparse_model_path,
            mvs_path=mvs_dir,
        )

    # --- Step 5/7: Stereo fusion ---
    logger.info("=== Step 5/7: Stereo fusion ===")
    fuse_depth_maps(
        mvs_path=mvs_dir,
        output_path=dense_ply,
        options=colmap_cfg["stereo_fusion"],
        bbox_min=args.bbox_min,
        bbox_max=args.bbox_max,
        mask_path=fusion_mask_dir,
    )

    # --- Step 5b/7: Point cloud filtering (SOR) + visualization ---
    logger.info("=== Step 5b/7: Point cloud filtering (SOR) ===")
    dense_filtered_ply, sor_stats = run_sor_and_visualize(dense_ply, output_dir, filter_cfg)

    # --- Metric scale recovery (before the crop: the auto crop radius is
    # derived in millimetres and converted to SfM units via the scale) ---
    marker_length_mm = aruco_cfg.get("marker_length_mm")
    scale_factor, marker_points, corners_by_marker = recover_scale_details_safe(
        reconstruction=reconstructions[best_model_idx],
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

    # Gate before the crop and the mesh: a failed scale recovery must not be
    # able to produce a finished, metric-looking mesh by default.
    scale_status = resolve_scale_status(scale_factor, scale_sanity)
    try:
        enforce_scale_policy(scale_status, allow_unscaled=args.allow_unscaled)
    except UnscaledOutputError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # --- Step 5c/7: Spherical crop to head region (auto-sized from ArUco) ---
    cropped_ply, crop_stats = run_head_crop(
        dense_filtered_ply,
        output_dir,
        reconstructions[best_model_idx],
        head_radius_override=args.head_radius,
        scale_factor=scale_factor,
        marker_points=marker_points,
    )
    sor_stats.update(crop_stats)

    # --- Step 5d/7: Optional membrane filter (opt-in, off by default) ---
    input_for_poisson = cropped_ply
    membrane_stats: dict | None = None
    if args.membrane_filter:
        input_for_poisson, membrane_stats = run_membrane_filter(
            cropped_ply,
            output_dir,
            marker_corners=corners_by_marker,
            pale_threshold=args.membrane_pale_threshold,
            marker_margin_mm=args.membrane_marker_margin_mm,
            scale_factor=scale_factor,
        )

    # --- Step 6/7: Poisson reconstruction + LCC + visualization ---
    logger.info("=== Step 6/7: Surface (Poisson) reconstruction + LCC ===")
    _, lcc_stats = run_poisson_lcc_and_visualize(
        input_for_poisson, mesh_ply, output_dir, mesh_cfg["poisson_surface_reconstruction"]
    )

    # Scale is applied once per file, after meshing: scaling dense_filtered_ply
    # before Poisson would double-scale the mesh derived from it.
    if scale_factor is not None:
        # Each written cloud scaled exactly once. dict.fromkeys de-duplicates
        # while preserving order, so a run where the crop or the membrane filter
        # was skipped does not scale the same file twice.
        for ply in dict.fromkeys([dense_filtered_ply, cropped_ply, input_for_poisson]):
            apply_scale_to_ply(ply, scale_factor)
        apply_scale_to_mesh(mesh_ply, scale_factor)
    else:
        # Reached only under --allow-unscaled. Rename every artefact so a stray
        # file cannot later be mistaken for metric output.
        for ply in dict.fromkeys([dense_filtered_ply, cropped_ply, input_for_poisson]):
            ply.rename(unscaled_artifact_path(ply))
        mesh_ply = mesh_ply.rename(unscaled_artifact_path(mesh_ply))

    # --- Step 7/7: Pipeline manifest ---
    provenance = build_provenance(
        args.frames_manifest,
        {"aruco": aruco_cfg, "colmap": colmap_cfg, "mesh": mesh_cfg},
    )
    provenance["intrinsics_source"] = intrinsics_source
    with_fusion_mask_provenance(
        provenance,
        enabled=fusion_mask_dir is not None,
        source_mask_dir=mask_path,
        workspace_mask_dir=fusion_mask_dir,
        stats=fusion_mask_stats,
    )
    with_membrane_filter_provenance(
        provenance, enabled=args.membrane_filter, stats=membrane_stats
    )
    write_pipeline_manifest(
        output_dir,
        "run_pipeline.py",
        sor_stats,
        lcc_stats,
        mesh_cfg["poisson_surface_reconstruction"],
        scale_factor,
        scale_sanity=scale_sanity,
        scale_self_consistency=scale_self_consistency,
        scale_status=scale_status,
        provenance=provenance,
    )

    # --- Optional: Evaluation ---
    if args.ground_truth is not None:
        logger.info("=== Evaluation: computing metrics ===")
        results = evaluate(
            predicted_ply=mesh_ply,
            ground_truth_ply=args.ground_truth,
            options=eval_cfg["evaluation"],
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        with metrics_json.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results saved to '%s'", metrics_json)

    logger.info("Pipeline complete. Outputs in '%s'", output_dir)


if __name__ == "__main__":
    main()
