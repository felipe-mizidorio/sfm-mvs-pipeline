"""Independent sanity check of the recovered metric scale (plan task A3).

The primary scale factor is derived from marker *side* lengths. This check
uses a different observable — center-to-center distances between markers whose
physical layout is known (rigid cap) — so a systematic error in the primary
computation (e.g. the historical diagonal-corner-pairing bug) produces a
visible residual here instead of propagating silently into clinical
measurements.

Design decision: the check never aborts a run. Scale recovery itself is
deliberately non-fatal (a scale hiccup must not waste hours of GPU time), and
a false abort on a QC heuristic would contradict that. Exceeding the tolerance
logs a prominent warning and is recorded in pipeline_manifest.json, where
downstream consumers must gate on it.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_marker_layout(
    corners_by_marker: dict[int, dict[int, np.ndarray]],
    scale_factor: float | None,
    layout_cfg: dict | None,
) -> dict | None:
    """Compare triangulated inter-marker distances against the known layout.

    Args:
        corners_by_marker: {marker_id: {corner_index: xyz}} in SfM units, as
            returned by triangulate_marker_corners.
        scale_factor: Recovered mm/SfM-unit factor; None disables the check.
        layout_cfg: The `layout_check` block from aruco.yaml. Expected shape:
            {"known_distances_mm": [{"ids": [a, b], "distance_mm": d}, ...],
             "warn_tolerance_pct": 5.0}

    Returns:
        A manifest-ready dict with per-pair results and an aggregate status
        ("passed" | "warning" | "insufficient_markers"), or None when the
        check is disabled (no layout configured, or no scale recovered).
    """
    if not layout_cfg or not layout_cfg.get("known_distances_mm"):
        logger.debug("Scale sanity check skipped: no marker layout configured.")
        return None
    if scale_factor is None:
        logger.info("Scale sanity check skipped: no scale factor recovered.")
        return None

    tolerance_pct = float(layout_cfg.get("warn_tolerance_pct", 5.0))
    centers = {
        mid: np.mean(list(corners.values()), axis=0)
        for mid, corners in corners_by_marker.items()
        if corners
    }

    pairs: list[dict] = []
    residuals: list[float] = []
    for entry in layout_cfg["known_distances_mm"]:
        ids = list(entry.get("ids", []))
        expected_mm = entry.get("distance_mm")
        if len(ids) != 2 or not expected_mm or ids[0] not in centers or ids[1] not in centers:
            pairs.append({"ids": ids, "status": "unavailable"})
            continue
        measured_mm = float(
            np.linalg.norm(centers[ids[0]] - centers[ids[1]]) * scale_factor
        )
        residual_pct = (measured_mm - float(expected_mm)) / float(expected_mm) * 100.0
        residuals.append(residual_pct)
        pairs.append(
            {
                "ids": ids,
                "expected_mm": float(expected_mm),
                "measured_mm": measured_mm,
                "residual_pct": residual_pct,
            }
        )

    if not residuals:
        logger.warning(
            "Scale sanity check: none of the %d configured marker pair(s) "
            "could be measured from the triangulation.",
            len(pairs),
        )
        status = "insufficient_markers"
        max_abs_residual = None
    else:
        max_abs_residual = max(abs(r) for r in residuals)
        status = "passed" if max_abs_residual <= tolerance_pct else "warning"
        if status == "warning":
            logger.warning(
                "SCALE SANITY CHECK FAILED: inter-marker distances deviate up "
                "to %.1f%% from the configured cap layout (tolerance %.1f%%). "
                "The recovered scale factor is suspect — do not use this "
                "mesh for clinical measurements without review. Details in "
                "pipeline_manifest.json → scale_sanity_check.",
                max_abs_residual,
                tolerance_pct,
            )
        else:
            logger.info(
                "Scale sanity check passed: %d pair(s), max residual %.2f%% "
                "(tolerance %.1f%%).",
                len(residuals),
                max_abs_residual,
                tolerance_pct,
            )

    return {
        "status": status,
        "num_pairs_checked": len(residuals),
        "max_abs_residual_pct": max_abs_residual,
        "warn_tolerance_pct": tolerance_pct,
        "pairs": pairs,
    }
