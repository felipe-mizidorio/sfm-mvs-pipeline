"""Marker self-consistency checks for the recovered metric scale.

Two metrics computed from the same triangulated ArUco corners the primary
scale factor uses (no re-detection, no new capture):

1. Per-marker scale dispersion — each fully-triangulated marker implies its
   own mm/SfM-unit scale from its four side lengths; the spread across markers
   (coefficient of variation) measures how consistently the markers were
   triangulated.
2. Diagonal/side ratio — for a planar square the corner-to-corner diagonal is
   side x sqrt(2). The primary scale is derived from the sides only, so the
   diagonal is an independent geometric relation on the same corners and a
   sentinel for perspective / non-planarity error (the inverse of the
   historical diagonal-corner-pairing bug).

IMPORTANT: these checks measure PRECISION (internal consistency), NOT
ACCURACY. They verify the markers agree with each other, not that the
millimetres are correct — marker_length_mm is still the only metric anchor,
so anything built on it is self-referential. Independent accuracy validation
(a measured inter-marker distance, plan task A3, or an in-frame ruler)
remains an open debt.

Design decision: warn-only, never abort — same rationale as layout_check.py.
We are still characterizing the baseline, so the thresholds below are
documented soft guesses, not validated abort criteria.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# Soft warn threshold on the coefficient of variation of per-marker scales.
# 5% mirrors the layout-check tolerance (warn_tolerance_pct); we do not yet
# know the normal range for this capture setup, so this is a characterization
# aid, not a validated limit.
CV_WARN_THRESHOLD = 0.05

# A marker whose scale deviates from the mean by more than k*std is flagged.
# k=2 flags the ~5% tail under normality; with the ~19 markers of a typical
# cap, k=3 would almost never flag anything. Caveat: with a single gross
# outlier in a small sample the outlier inflates the std and can mask itself,
# so an empty outlier list does not prove all markers are good.
SCALE_OUTLIER_K = 2.0

# Tolerance on the diagonal/side ratio's relative deviation from sqrt(2).
# 5% corresponds to a marker folded out of plane by roughly 0.7x its side —
# a gross defect; smaller perspective errors stay below it. Warn-only.
DIAGONAL_RATIO_TOLERANCE_PCT = 5.0

_EPS = 1e-9

# Adjacent corner pairs measure the marker side; (0,2) and (1,3) are the
# diagonals. OpenCV corner order guarantees this pairing.
_SIDE_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))
_DIAGONAL_PAIRS = ((0, 2), (1, 3))

CAVEAT = (
    "Precision/internal-consistency metrics only: they verify the markers "
    "agree with each other, NOT that the millimetres are accurate. "
    "marker_length_mm remains the only metric anchor; independent accuracy "
    "validation (measured inter-marker distances, A3) is still required."
)


def check_scale_self_consistency(
    corners_by_marker: dict[int, dict[int, np.ndarray]],
    marker_length_mm: float | None,
) -> dict | None:
    """Compute both self-consistency metrics from triangulated corners.

    Args:
        corners_by_marker: {marker_id: {corner_index: xyz}} in SfM units, as
            returned by triangulate_marker_corners.
        marker_length_mm: Physical marker side length; falsy disables the
            check (scale recovery itself is disabled in that case).

    Returns:
        A manifest-ready dict (pure JSON types) with per_marker_scale and
        diagonal_ratio blocks, or None when disabled. Markers without all
        four corners, or with degenerate (near-zero) sides or diagonals, are
        skipped and counted in n_skipped. Never raises on marker geometry.
    """
    if not marker_length_mm:
        return None

    scales: dict[int, float] = {}
    ratios: dict[int, dict[str, float]] = {}
    n_skipped = 0

    for mid, corners in corners_by_marker.items():
        if any(idx not in corners for idx in range(4)):
            n_skipped += 1
            continue
        sides = [
            float(np.linalg.norm(corners[a] - corners[b])) for a, b in _SIDE_PAIRS
        ]
        diagonals = [
            float(np.linalg.norm(corners[a] - corners[b])) for a, b in _DIAGONAL_PAIRS
        ]
        if min(sides) < _EPS or min(diagonals) < _EPS:
            n_skipped += 1
            continue

        mean_side = float(np.mean(sides))
        scales[mid] = float(marker_length_mm) / mean_side
        ratio = float(np.mean(diagonals)) / mean_side
        ratios[mid] = {
            "ratio": ratio,
            "deviation_pct": (ratio / math.sqrt(2.0) - 1.0) * 100.0,
        }

    if n_skipped:
        logger.info(
            "Scale self-consistency: %d marker(s) skipped "
            "(incomplete or degenerate corners).",
            n_skipped,
        )

    return {
        "caveat": CAVEAT,
        "n_markers_used": len(scales),
        "n_skipped": n_skipped,
        "per_marker_scale": _per_marker_scale_block(scales),
        "diagonal_ratio": _diagonal_ratio_block(ratios),
    }


def _per_marker_scale_block(scales: dict[int, float]) -> dict:
    """Dispersion statistics over the per-marker scale estimates."""
    block: dict = {
        "scales_mm_per_unit": scales,
        "mean_mm_per_unit": None,
        "std_mm_per_unit": None,
        "cv": None,
        "cv_warn_threshold": CV_WARN_THRESHOLD,
        "outlier_k": SCALE_OUTLIER_K,
        "outliers": [],
        "status": "insufficient_markers",
    }
    if len(scales) < 2:
        return block

    values = np.array(list(scales.values()))
    mean = float(values.mean())
    std = float(values.std(ddof=1))  # sample std: markers are a sample
    cv = std / mean
    block.update(
        mean_mm_per_unit=mean,
        std_mm_per_unit=std,
        cv=cv,
        outliers=sorted(
            mid for mid, s in scales.items() if abs(s - mean) > SCALE_OUTLIER_K * std
        ),
        status="ok" if cv <= CV_WARN_THRESHOLD else "warning",
    )
    if block["status"] == "warning":
        logger.warning(
            "Scale self-consistency: per-marker scales disagree (CV %.1f%% > "
            "%.1f%%, %d marker(s), worst outliers: %s). Possible motion, poor "
            "calibration, or bad triangulations — precision check only, run "
            "continues. Details in pipeline_manifest.json -> "
            "scale_self_consistency.",
            cv * 100.0,
            CV_WARN_THRESHOLD * 100.0,
            len(scales),
            block["outliers"] or "none beyond k*std",
        )
    else:
        logger.info(
            "Scale self-consistency: per-marker scale CV %.2f%% over %d "
            "marker(s) (threshold %.1f%%).",
            cv * 100.0,
            len(scales),
            CV_WARN_THRESHOLD * 100.0,
        )
    return block


def _diagonal_ratio_block(ratios: dict[int, dict[str, float]]) -> dict:
    """Per-marker diagonal/side ratio vs the planar-square sqrt(2)."""
    block: dict = {
        "expected_ratio": math.sqrt(2.0),
        "tolerance_pct": DIAGONAL_RATIO_TOLERANCE_PCT,
        "per_marker": ratios,
        "flagged": [],
        "status": "insufficient_markers",
    }
    if not ratios:
        return block

    flagged = sorted(
        mid
        for mid, entry in ratios.items()
        if abs(entry["deviation_pct"]) > DIAGONAL_RATIO_TOLERANCE_PCT
    )
    block.update(flagged=flagged, status="warning" if flagged else "ok")
    if flagged:
        logger.warning(
            "Scale self-consistency: %d marker(s) deviate more than %.1f%% "
            "from the planar diagonal/side ratio sqrt(2): %s. Suggests "
            "non-planar or badly triangulated corners — precision check "
            "only, run continues.",
            len(flagged),
            DIAGONAL_RATIO_TOLERANCE_PCT,
            flagged,
        )
    else:
        logger.info(
            "Scale self-consistency: diagonal/side ratio within %.1f%% of "
            "sqrt(2) for all %d marker(s).",
            DIAGONAL_RATIO_TOLERANCE_PCT,
            len(ratios),
        )
    return block
