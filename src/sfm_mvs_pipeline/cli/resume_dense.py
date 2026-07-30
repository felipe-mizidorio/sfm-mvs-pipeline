"""Resume pipeline from an existing dense.ply — runs SOR, scale recovery, Poisson + LCC."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from sfm_mvs_pipeline.pipeline.orchestration import (
    build_provenance,
    run_post_fusion,
    run_sor,
    with_fusion_mask_provenance,
)
from sfm_mvs_pipeline.sfm.reconstruction import load_best_reconstruction

_REPO_ROOT = Path(__file__).resolve().parents[3]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume pipeline from existing dense.ply: SOR + scale recovery + Poisson + LCC."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--frames-manifest", default=None, type=Path)
    parser.add_argument(
        "--aruco-config", default=_REPO_ROOT / "configs/aruco.yaml", type=Path
    )
    parser.add_argument(
        "--mesh-config", default=_REPO_ROOT / "configs/mesh.yaml", type=Path
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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
        logger.info(
            "Manifest loaded: %d pre-detected marker entries",
            len(manifest_detections or {}),
        )

    # SOR.
    logger.info("=== Point cloud filtering (SOR) ===")
    dense_filtered_ply, sor_stats = run_sor(dense_ply, output_dir, filter_cfg)

    # Provenance: no stereo fusion happens on this path, so the fusion-mask
    # block is always recorded as disabled; run_post_fusion adds the membrane
    # block. The input dense.ply is left untouched (it is the operator-supplied
    # input, and scaling it in place would double-scale on a re-run).
    provenance = build_provenance(
        args.frames_manifest, {"aruco": aruco_cfg, "mesh": mesh_cfg}
    )
    with_fusion_mask_provenance(provenance, enabled=False)

    run_post_fusion(
        output_dir=output_dir,
        run_script="sfm-mvs-resume-dense",
        reconstruction=reconstruction,
        image_dir=args.image_dir,
        aruco_cfg=aruco_cfg,
        mesh_cfg=mesh_cfg,
        dense_filtered_ply=dense_filtered_ply,
        sor_stats=sor_stats,
        mesh_ply=mesh_ply,
        provenance=provenance,
        manifest_detections=manifest_detections,
        allow_unscaled=args.allow_unscaled,
    )

    logger.info("Resume complete. Outputs in '%s'", output_dir)


if __name__ == "__main__":
    main()
