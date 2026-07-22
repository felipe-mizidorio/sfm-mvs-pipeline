"""Metric-scale status classification and the unscaled-output policy.

Scale recovery is deliberately non-fatal: `recover_scale_details_safe` logs and
returns `None` rather than raising, so a scale hiccup cannot waste hours of GPU
time. The cost of that choice was a silent failure path — a run whose scale
recovery returned `None` skipped scaling and still wrote a complete mesh and
manifest in raw SfM units. The numbers looked like millimetres. They were not.

This module makes that state explicit rather than forbidden:

* `resolve_scale_status` classifies the run so the manifest can distinguish a
  scale factor that is *absent*, *present but never checked against ground
  truth*, or *present and validated* — a distinction the manifest previously
  could not express, despite every current run sitting in the middle state.
* `enforce_scale_policy` turns an absent scale factor into a hard stop unless
  the operator explicitly opted in, so unscaled output cannot happen silently.
* `unscaled_artifact_path` renames artefacts written in SfM units, so a stray
  `.ply` cannot later be mistaken for metric output.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: A scale factor was recovered and confirmed against known cap distances.
STATUS_VALIDATED = "validated"
#: A scale factor was recovered but never checked against ground truth --
#: either no `known_distances_mm` are configured, or none could be measured.
STATUS_UNVALIDATED = "recovered_unvalidated"
#: A scale factor was recovered and the layout check exceeded its tolerance.
STATUS_FAILED_VALIDATION = "recovered_failed_validation"
#: No scale factor could be recovered. Output is in arbitrary SfM units.
STATUS_UNSCALED = "unscaled"

#: Inserted into the filename of any artefact written in raw SfM units.
UNSCALED_MARKER = "UNSCALED_sfm_units"


class UnscaledOutputError(RuntimeError):
    """Raised when a run would write non-metric output without opting in."""


def resolve_scale_status(
    scale_factor: float | None,
    scale_sanity: dict | None,
) -> dict:
    """Classify the metric-scale state of a run for the manifest.

    Args:
        scale_factor: Recovered mm/SfM-unit factor, or None if recovery failed.
        scale_sanity: The `check_marker_layout` result, or None when the check
            did not run (no `known_distances_mm` configured, or no scale).

    Returns:
        A manifest-ready dict with an explicit `status`, the coordinate `units`
        the artefacts are actually in, and a human-readable `detail`.
    """
    if scale_factor is None:
        return {
            "status": STATUS_UNSCALED,
            "units": "sfm_units",
            "scale_factor_mm_per_unit": None,
            "validated_against_known_distances": False,
            "detail": (
                "Scale recovery failed or was skipped. Coordinates are in "
                "arbitrary SfM units, NOT millimetres. Do not measure this output."
            ),
        }

    sanity = scale_sanity or {}
    sanity_status = sanity.get("status")

    if sanity_status == "passed":
        status = STATUS_VALIDATED
        detail = (
            "Scale recovered and confirmed against the configured cap layout "
            f"({sanity.get('num_pairs_checked')} pair(s) checked)."
        )
        validated = True
    elif sanity_status == "warning":
        status = STATUS_FAILED_VALIDATION
        detail = (
            "Scale recovered but inter-marker distances deviate from the "
            f"configured cap layout by up to "
            f"{sanity.get('max_abs_residual_pct')}%. The scale factor is "
            "suspect -- review before measuring."
        )
        validated = False
    else:
        status = STATUS_UNVALIDATED
        detail = (
            "Scale recovered but never checked against ground truth: no usable "
            "entries in aruco.yaml -> layout_check.known_distances_mm. "
            "Millimetre figures are internally consistent but of unverified "
            "accuracy -- precision, not accuracy."
        )
        validated = False

    return {
        "status": status,
        "units": "mm",
        "scale_factor_mm_per_unit": float(scale_factor),
        "validated_against_known_distances": validated,
        "detail": detail,
    }


def enforce_scale_policy(scale_status: dict, allow_unscaled: bool) -> None:
    """Stop the run if it would write non-metric output without consent.

    Only an *absent* scale factor is a hard stop. A recovered-but-unvalidated
    factor is deliberately allowed: it is the state of every run until
    `known_distances_mm` can be populated, which is blocked on physical access
    to the cap. Gating on it would block all current work while doing nothing
    about the silent path this guard exists to close.

    Args:
        scale_status: The result of `resolve_scale_status`.
        allow_unscaled: True when the operator explicitly opted in to
            non-metric output.

    Raises:
        UnscaledOutputError: If the scale factor is absent and the operator did
            not opt in.
    """
    if scale_status["status"] != STATUS_UNSCALED:
        return

    if not allow_unscaled:
        raise UnscaledOutputError(
            "Metric scale recovery failed, so this run would write a mesh in "
            "arbitrary SfM units that looks metric but is not. Stopping before "
            "any mesh is written.\n"
            "Fix the scale recovery (check aruco.yaml -> marker_length_mm, "
            "marker visibility and min_views), or re-run with --allow-unscaled "
            "to accept non-metric output. Unscaled artefacts are renamed to "
            f"*.{UNSCALED_MARKER}.* and the manifest records status "
            f"'{STATUS_UNSCALED}'."
        )

    logger.warning(
        "PROCEEDING WITHOUT METRIC SCALE (--allow-unscaled). All coordinates "
        "are in arbitrary SfM units, not millimetres. Artefacts are renamed to "
        "*.%s.* and pipeline_manifest.json records status '%s'.",
        UNSCALED_MARKER,
        STATUS_UNSCALED,
    )


def unscaled_artifact_path(path: Path) -> Path:
    """Return `path` renamed to identify it as raw SfM units.

    `out/mesh.ply` becomes `out/mesh.UNSCALED_sfm_units.ply`. The marker rides
    in the filename rather than a sidecar so that it survives the file being
    copied or moved somewhere else. Idempotent, so re-marking is safe.
    """
    if UNSCALED_MARKER in path.name:
        return path
    return path.with_name(f"{path.stem}.{UNSCALED_MARKER}{path.suffix}")
