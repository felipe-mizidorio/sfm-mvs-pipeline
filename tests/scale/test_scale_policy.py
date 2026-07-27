"""Tests for the metric-scale status classification and output policy."""

from pathlib import Path

import pytest

from sfm_mvs_pipeline.scale.policy import (
    STATUS_FAILED_VALIDATION,
    STATUS_UNSCALED,
    STATUS_UNVALIDATED,
    STATUS_VALIDATED,
    UnscaledOutputError,
    enforce_scale_policy,
    resolve_scale_status,
    unscaled_artifact_path,
)

# --- resolve_scale_status: the three-way (plus failure) distinction ---------


def test_absent_scale_factor_is_unscaled() -> None:
    status = resolve_scale_status(None, None)

    assert status["status"] == STATUS_UNSCALED
    assert status["units"] == "sfm_units"
    assert status["scale_factor_mm_per_unit"] is None
    assert status["validated_against_known_distances"] is False


def test_scale_without_configured_layout_is_unvalidated() -> None:
    """The state every current run is in: recovered, never checked."""
    status = resolve_scale_status(61.71, None)

    assert status["status"] == STATUS_UNVALIDATED
    assert status["units"] == "mm"
    assert status["scale_factor_mm_per_unit"] == 61.71
    assert status["validated_against_known_distances"] is False


def test_scale_with_passing_layout_check_is_validated() -> None:
    status = resolve_scale_status(61.71, {"status": "passed", "num_pairs_checked": 2})

    assert status["status"] == STATUS_VALIDATED
    assert status["units"] == "mm"
    assert status["validated_against_known_distances"] is True


def test_scale_with_failing_layout_check_is_failed_validation() -> None:
    status = resolve_scale_status(61.71, {"status": "warning", "max_abs_residual_pct": 9.0})

    assert status["status"] == STATUS_FAILED_VALIDATION
    assert status["validated_against_known_distances"] is False


def test_insufficient_markers_is_unvalidated_not_validated() -> None:
    """The check ran but measured nothing, so it validated nothing."""
    status = resolve_scale_status(61.71, {"status": "insufficient_markers"})

    assert status["status"] == STATUS_UNVALIDATED
    assert status["validated_against_known_distances"] is False


def test_status_carries_a_human_readable_reason() -> None:
    assert resolve_scale_status(None, None)["detail"]
    assert resolve_scale_status(61.71, None)["detail"]


# --- enforce_scale_policy: strict by default --------------------------------


def test_strict_policy_rejects_unscaled_output() -> None:
    status = resolve_scale_status(None, None)

    with pytest.raises(UnscaledOutputError, match="--allow-unscaled"):
        enforce_scale_policy(status, allow_unscaled=False)


def test_permissive_policy_allows_unscaled_output() -> None:
    status = resolve_scale_status(None, None)

    enforce_scale_policy(status, allow_unscaled=True)  # must not raise


def test_strict_policy_allows_unvalidated_scale() -> None:
    """Unvalidated is the current normal state; gating it would block every run."""
    status = resolve_scale_status(61.71, None)

    enforce_scale_policy(status, allow_unscaled=False)  # must not raise


def test_strict_policy_allows_failed_validation() -> None:
    """A suspect scale is loud in the manifest and logs, but is still metric."""
    status = resolve_scale_status(61.71, {"status": "warning", "max_abs_residual_pct": 9.0})

    enforce_scale_policy(status, allow_unscaled=False)  # must not raise


# --- unscaled_artifact_path: artefacts identify themselves ------------------


def test_unscaled_artifact_path_marks_the_filename() -> None:
    marked = unscaled_artifact_path(Path("out/mesh.ply"))

    assert marked.name == "mesh.UNSCALED_sfm_units.ply"
    assert marked.parent == Path("out")


def test_unscaled_artifact_path_is_idempotent() -> None:
    once = unscaled_artifact_path(Path("out/mesh.ply"))
    twice = unscaled_artifact_path(once)

    assert once == twice
