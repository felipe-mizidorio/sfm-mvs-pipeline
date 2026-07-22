"""The pipeline manifest must express the scale state unambiguously."""

import json

from sfm_mvs_pipeline.pipeline.orchestration import write_pipeline_manifest
from sfm_mvs_pipeline.scale.policy import (
    STATUS_UNSCALED,
    STATUS_UNVALIDATED,
    STATUS_VALIDATED,
    resolve_scale_status,
    unscaled_artifact_path,
)

_MESH_OPTS = {"depth": 9, "scale": 1.1, "linear_fit": False, "density_threshold": 0.01}


def _write(tmp_path, scale_factor, scale_sanity):
    status = resolve_scale_status(scale_factor, scale_sanity)
    write_pipeline_manifest(
        tmp_path,
        "run_pipeline.py",
        {},
        {},
        _MESH_OPTS,
        scale_factor,
        scale_sanity=scale_sanity,
        scale_status=status,
    )
    return json.loads((tmp_path / "pipeline_manifest.json").read_text())


def test_manifest_marks_an_unscaled_run(tmp_path) -> None:
    manifest = _write(tmp_path, None, None)

    assert manifest["scale"]["status"] == STATUS_UNSCALED
    assert manifest["scale"]["units"] == "sfm_units"
    assert manifest["scale_factor_mm_per_unit"] is None


def test_manifest_distinguishes_unvalidated_from_validated(tmp_path) -> None:
    """The middle state every current run is in must be visible as its own thing."""
    unvalidated = _write(tmp_path, 61.71, None)
    validated = _write(tmp_path, 61.71, {"status": "passed", "num_pairs_checked": 2})

    assert unvalidated["scale"]["status"] == STATUS_UNVALIDATED
    assert unvalidated["scale"]["validated_against_known_distances"] is False

    assert validated["scale"]["status"] == STATUS_VALIDATED
    assert validated["scale"]["validated_against_known_distances"] is True

    # Both carry the same scale factor, so the factor alone cannot tell them apart.
    assert unvalidated["scale_factor_mm_per_unit"] == validated["scale_factor_mm_per_unit"]
    assert unvalidated["scale"]["status"] != validated["scale"]["status"]


def test_manifest_without_scale_status_is_unchanged(tmp_path) -> None:
    """Back-compat: callers that do not pass a status get the old manifest."""
    write_pipeline_manifest(tmp_path, "run_pipeline.py", {}, {}, _MESH_OPTS, 61.71)
    manifest = json.loads((tmp_path / "pipeline_manifest.json").read_text())

    assert "scale" not in manifest
    assert manifest["scale_factor_mm_per_unit"] == 61.71


def test_renaming_an_unscaled_artifact_moves_the_file(tmp_path) -> None:
    mesh = tmp_path / "mesh.ply"
    mesh.write_text("ply")

    renamed = mesh.rename(unscaled_artifact_path(mesh))

    assert not mesh.exists()
    assert renamed.name == "mesh.UNSCALED_sfm_units.ply"
    assert renamed.read_text() == "ply"
