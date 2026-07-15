"""Resume pipeline from an existing dense.ply — runs SOR, scale recovery, Poisson + LCC."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from sfm_mvs_pipeline.pipeline.orchestration import (
    build_provenance,
    run_poisson_lcc_and_visualize,
    run_sor_and_visualize,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume pipeline from existing dense.ply: SOR + scale recovery + Poisson + LCC."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--frames-manifest", default=None, type=Path)
    parser.add_argument("--aruco-config", default=_REPO_ROOT / "configs/aruco.yaml", type=Path)
    parser.add_argument("--mesh-config", default=_REPO_ROOT / "configs/mesh.yaml", type=Path)
    args = parser.parse_args()

    with args.aruco_config.open() as f:
        aruco_cfg = yaml.safe_load(f).get("aruco", {})
    with args.mesh_config.open() as f:
        mesh_cfg = yaml.safe_load(f)
    filter_cfg = mesh_cfg["point_cloud_filtering"]

    output_dir: Path = args.output_dir
    dense_ply = output_dir / "dense.ply"
    mesh_ply = output_dir / "mesh.ply"
    sparse_dir = output_dir / "sparse"

    if not dense_ply.exists():
        logger.error("dense.ply not found at '%s' — cannot resume.", dense_ply)
        sys.exit(1)

    # Load best sparse model from disk.
    reconstruction, best_sparse = load_best_reconstruction(sparse_dir)
    logger.info(
        "Loaded sparse model from '%s': %d registered images",
        best_sparse,
        reconstruction.num_reg_images(),
    )

    # Load manifest detections if provided.
    manifest_detections = None
    if args.frames_manifest is not None:
        manifest_data = json.loads(args.frames_manifest.read_text())
        manifest_detections = manifest_data.get("marker_detections")
        logger.info("Manifest loaded: %d pre-detected marker entries", len(manifest_detections or {}))

    # SOR + visualization.
    logger.info("=== Point cloud filtering (SOR) ===")
    dense_filtered_ply, sor_stats = run_sor_and_visualize(dense_ply, output_dir, filter_cfg)

    # Scale recovery.
    marker_length_mm = aruco_cfg.get("marker_length_mm")
    scale_factor, _, corners_by_marker = recover_scale_details_safe(
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
    # Poisson + LCC + visualization.
    logger.info("=== Poisson surface reconstruction + LCC ===")
    _, lcc_stats = run_poisson_lcc_and_visualize(
        dense_filtered_ply, mesh_ply, output_dir, mesh_cfg["poisson_surface_reconstruction"]
    )

    # Scale is applied once per file, after meshing: scaling dense_filtered_ply
    # before Poisson would double-scale the mesh derived from it.
    if scale_factor is not None:
        apply_scale_to_ply(dense_filtered_ply, scale_factor)
        apply_scale_to_mesh(mesh_ply, scale_factor)

    write_pipeline_manifest(
        output_dir,
        "resume_from_dense.py",
        sor_stats,
        lcc_stats,
        mesh_cfg["poisson_surface_reconstruction"],
        scale_factor,
        scale_sanity=scale_sanity,
        scale_self_consistency=scale_self_consistency,
        provenance=build_provenance(
            args.frames_manifest, {"aruco": aruco_cfg, "mesh": mesh_cfg}
        ),
    )

    logger.info("Resume complete. Outputs in '%s'", output_dir)


if __name__ == "__main__":
    main()
