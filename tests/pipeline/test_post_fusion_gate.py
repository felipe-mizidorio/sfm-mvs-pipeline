"""The shared post-fusion backbone must honour the scale policy.

`run_post_fusion` is where all three CLI entrypoints converge, so the scale
gate is tested here once rather than per script. A failed scale recovery must
hard-stop by default, and only produce UNSCALED-marked artefacts when the
operator opts in — the invariant `scale/policy.py` exists to protect.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sfm_mvs_pipeline.pipeline import orchestration
from sfm_mvs_pipeline.pipeline.orchestration import build_provenance, run_post_fusion
from sfm_mvs_pipeline.scale.policy import STATUS_UNSCALED, UNSCALED_MARKER

MESH_CFG = {
    "poisson_surface_reconstruction": {"depth": 9, "scale": 1.1, "linear_fit": False}
}


def _run_post_fusion(tmp_path: Path, *, allow_unscaled: bool) -> Path:
    """Drive run_post_fusion with an empty aruco config (scale recovery disabled,
    i.e. the unscaled path) while stubbing the o3d-heavy crop and mesh steps."""
    dense_filtered = tmp_path / "dense_filtered.ply"
    dense_filtered.write_text("ply")
    mesh_ply = tmp_path / "mesh.ply"
    mesh_ply.write_text("ply")

    with (
        patch.object(orchestration, "run_head_crop", return_value=(dense_filtered, {})),
        patch.object(
            orchestration, "run_poisson_lcc", return_value=(None, {"lcc": {}})
        ),
    ):
        return run_post_fusion(
            output_dir=tmp_path,
            run_script="sfm-mvs-resume-dense",
            reconstruction=MagicMock(),
            image_dir=tmp_path,
            aruco_cfg={},  # no marker_length_mm → scale disabled → unscaled status
            mesh_cfg=MESH_CFG,
            dense_filtered_ply=dense_filtered,
            sor_stats={"point_cloud_filtering": {}},
            mesh_ply=mesh_ply,
            provenance=build_provenance(None, {}),
            allow_unscaled=allow_unscaled,
        )


def test_post_fusion_hard_stops_when_unscaled_and_not_opted_in(tmp_path: Path):
    with pytest.raises(SystemExit):
        _run_post_fusion(tmp_path, allow_unscaled=False)

    # No mesh is written on the hard stop.
    assert not (tmp_path / "pipeline_manifest.json").exists()


def test_post_fusion_marks_artefacts_and_manifest_when_opted_in(tmp_path: Path):
    mesh_ply = _run_post_fusion(tmp_path, allow_unscaled=True)

    # Artefacts are renamed so they cannot be mistaken for metric output.
    assert UNSCALED_MARKER in mesh_ply.name
    assert mesh_ply.exists()
    assert (tmp_path / f"dense_filtered.{UNSCALED_MARKER}.ply").exists()

    # The manifest records the unscaled status explicitly.
    data = json.loads((tmp_path / "pipeline_manifest.json").read_text())
    assert data["scale"]["status"] == STATUS_UNSCALED
    assert data["scale_factor_mm_per_unit"] is None
