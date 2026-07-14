"""Tests for reproducibility fields in pipeline_manifest.json (A2)."""

import hashlib
import json
from pathlib import Path

from sfm_mvs_pipeline.pipeline.orchestration import (
    build_provenance,
    write_pipeline_manifest,
)

SOR_STATS = {"point_cloud_filtering": {"points_before": 10, "points_after": 9}}
LCC_STATS = {"lcc": {"triangles_kept": 100, "triangles_removed": 1}}
MESH_OPTS = {"depth": 9, "scale": 1.1, "linear_fit": False}


def test_build_provenance_environment_versions(tmp_path: Path):
    provenance = build_provenance(None, {})

    env = provenance["environment"]
    for key in ("python", "pycolmap", "open3d", "opencv", "numpy"):
        assert env[key], f"missing version for {key}"
    assert provenance["frames_manifest_sha256"] is None
    assert isinstance(provenance["non_determinism_notes"], list)
    assert provenance["non_determinism_notes"]


def test_build_provenance_hashes_frames_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"frames": []}')
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()

    provenance = build_provenance(manifest, {"aruco": {"marker_length_mm": 50.0}})

    assert provenance["frames_manifest_sha256"] == expected
    assert provenance["resolved_configs"]["aruco"]["marker_length_mm"] == 50.0


def test_write_pipeline_manifest_includes_provenance_and_sanity(tmp_path: Path):
    provenance = build_provenance(None, {"mesh": MESH_OPTS})
    sanity = {"status": "passed", "num_pairs_checked": 1}

    write_pipeline_manifest(
        tmp_path,
        "run_pipeline.py",
        SOR_STATS,
        LCC_STATS,
        MESH_OPTS,
        146.08,
        scale_sanity=sanity,
        provenance=provenance,
    )

    data = json.loads((tmp_path / "pipeline_manifest.json").read_text())
    assert data["scale_factor_mm_per_unit"] == 146.08
    assert data["scale_sanity_check"]["status"] == "passed"
    assert data["environment"]["pycolmap"]
    assert data["resolved_configs"]["mesh"]["depth"] == 9
    assert data["non_determinism_notes"]


def test_write_pipeline_manifest_backward_compatible(tmp_path: Path):
    """Old call signature (no new kwargs) still works and keeps old keys."""
    write_pipeline_manifest(
        tmp_path, "resume_from_dense.py", SOR_STATS, LCC_STATS, MESH_OPTS, None
    )

    data = json.loads((tmp_path / "pipeline_manifest.json").read_text())
    assert data["run_script"] == "resume_from_dense.py"
    assert data["scale_factor_mm_per_unit"] is None
    assert "scale_sanity_check" not in data
