"""End-to-end SfM + MVS + mesh + evaluation pipeline."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pycolmap
import yaml

from sfm_mvs_pipeline.evaluation.metrics import evaluate
from sfm_mvs_pipeline.mesh.surface_reconstruction import reconstruct_surface
from sfm_mvs_pipeline.mvs.dense_reconstruction import run_dense_reconstruction
from sfm_mvs_pipeline.mvs.fusion import fuse_depth_maps
from sfm_mvs_pipeline.scale.aruco_scale import (
    apply_scale_to_mesh,
    apply_scale_to_ply,
    recover_scale,
)
from sfm_mvs_pipeline.sfm.feature_extraction import extract_features
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    colmap_cfg = _load_yaml(args.colmap_config)
    mesh_cfg = _load_yaml(args.mesh_config)
    eval_cfg = _load_yaml(args.evaluation_config)
    aruco_cfg = _load_yaml(args.aruco_config).get("aruco", {})

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
    if args.frames_manifest is not None:
        manifest_data = json.loads(args.frames_manifest.read_text())
        manifest_frames = manifest_data.get("frames")
        manifest_detections = manifest_data.get("marker_detections")
        logger.info(
            "Frames manifest loaded: %d frames, %d pre-detected markers",
            len(manifest_frames or []),
            len(manifest_detections or {}),
        )

    # --- Step 1: Feature extraction ---
    logger.info("=== Step 1/6: Feature extraction ===")
    extract_features(
        database_path=database_path,
        image_dir=args.image_dir,
        options=colmap_cfg["feature_extraction"],
        device=device,
        camera_model=args.camera_model,
        camera_params=args.camera_params,
        image_names=manifest_frames,
    )

    # --- Step 2: Feature matching ---
    logger.info("=== Step 2/6: Feature matching ===")
    match_features(
        database_path=database_path,
        options=colmap_cfg["feature_matching"],
        device=device,
    )

    # --- Step 3: Sparse reconstruction ---
    logger.info("=== Step 3/6: Sparse (incremental) reconstruction ===")
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

    # --- Step 4: Dense reconstruction (undistortion + PatchMatch Stereo) ---
    logger.info("=== Step 4/6: Dense reconstruction ===")
    run_dense_reconstruction(
        sparse_path=sparse_model_path,
        image_dir=args.image_dir,
        mvs_path=mvs_dir,
        options=colmap_cfg["patch_match_stereo"],
    )

    # --- Step 5: Stereo fusion ---
    logger.info("=== Step 5/6: Stereo fusion ===")
    fuse_depth_maps(
        mvs_path=mvs_dir,
        output_path=dense_ply,
        options=colmap_cfg["stereo_fusion"],
        bbox_min=args.bbox_min,
        bbox_max=args.bbox_max,
    )

    # --- Metric scale recovery (G1 + G6) ---
    scale_factor: float | None = None
    marker_length_mm = aruco_cfg.get("marker_length_mm")
    if marker_length_mm:
        logger.info("=== Scale recovery: detecting ArUco markers ===")
        try:
            scale_factor = recover_scale(
                reconstruction=reconstructions[best_model_idx],
                image_dir=args.image_dir,
                marker_length_mm=float(marker_length_mm),
                aruco_dict_id=int(aruco_cfg.get("dict_id", 0)),
                detections=manifest_detections,
                min_views=int(aruco_cfg.get("min_views", 2)),
            )
            logger.info("Scale factor: %.6f mm/unit", scale_factor)
            apply_scale_to_ply(dense_ply, scale_factor)
        except RuntimeError as exc:
            logger.warning("Scale recovery failed: %s — outputs remain in SfM units.", exc)

    # --- Step 6: Surface reconstruction ---
    logger.info("=== Step 6/6: Surface (Poisson) reconstruction ===")
    reconstruct_surface(
        input_ply=dense_ply,
        output_ply=mesh_ply,
        options=mesh_cfg["poisson_surface_reconstruction"],
    )

    if scale_factor is not None:
        apply_scale_to_mesh(mesh_ply, scale_factor)

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
