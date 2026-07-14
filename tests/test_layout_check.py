"""Tests for the independent scale sanity check (A3).

The check compares triangulated inter-marker distances (converted to mm via
the recovered scale factor) against known physical distances from the cap
layout config. It shares no failure mode with the primary marker-side scale
computation, so a systematic scale error shows up as a residual here.
"""

import numpy as np

from sfm_mvs_pipeline.scale.layout_check import check_marker_layout


def square_corners(center: np.ndarray, side: float) -> dict[int, np.ndarray]:
    """Axis-aligned square marker corners around a center, in SfM units."""
    h = side / 2.0
    offsets = [(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)]
    return {i: center + np.array(o) for i, o in enumerate(offsets)}


def make_two_marker_scene(
    distance_units: float, side_units: float = 0.34
) -> dict[int, dict[int, np.ndarray]]:
    return {
        0: square_corners(np.zeros(3), side_units),
        1: square_corners(np.array([distance_units, 0.0, 0.0]), side_units),
    }


LAYOUT_CFG = {
    "known_distances_mm": [{"ids": [0, 1], "distance_mm": 80.0}],
    "warn_tolerance_pct": 5.0,
}


def test_correct_scale_passes():
    # Markers 0.8 units apart, scale 100 mm/unit -> measured 80 mm == expected.
    corners = make_two_marker_scene(0.8)

    result = check_marker_layout(corners, 100.0, LAYOUT_CFG)

    assert result is not None
    assert result["status"] == "passed"
    assert result["num_pairs_checked"] == 1
    assert abs(result["max_abs_residual_pct"]) < 1e-6
    pair = result["pairs"][0]
    assert pair["measured_mm"] == 80.0
    assert pair["residual_pct"] == 0.0


def test_injected_scale_error_flagged():
    # Same geometry, but a scale off by +30% -> measured 104 mm vs 80 mm.
    corners = make_two_marker_scene(0.8)

    result = check_marker_layout(corners, 130.0, LAYOUT_CFG)

    assert result is not None
    assert result["status"] == "warning"
    assert result["max_abs_residual_pct"] > 5.0


def test_no_layout_config_returns_none():
    corners = make_two_marker_scene(0.8)
    assert check_marker_layout(corners, 100.0, None) is None
    assert check_marker_layout(corners, 100.0, {}) is None
    assert (
        check_marker_layout(corners, 100.0, {"known_distances_mm": []}) is None
    )


def test_no_scale_returns_none():
    corners = make_two_marker_scene(0.8)
    assert check_marker_layout(corners, None, LAYOUT_CFG) is None


def test_missing_marker_recorded_as_unavailable():
    # Only marker 0 triangulated; the configured pair (0, 1) cannot be measured.
    corners = {0: square_corners(np.zeros(3), 0.34)}

    result = check_marker_layout(corners, 100.0, LAYOUT_CFG)

    assert result is not None
    assert result["status"] == "insufficient_markers"
    assert result["num_pairs_checked"] == 0
    assert result["pairs"][0]["status"] == "unavailable"


def test_mixed_pairs_use_only_available_ones():
    corners = make_two_marker_scene(0.8)
    cfg = {
        "known_distances_mm": [
            {"ids": [0, 1], "distance_mm": 80.0},
            {"ids": [0, 7], "distance_mm": 120.0},
        ],
        "warn_tolerance_pct": 5.0,
    }

    result = check_marker_layout(corners, 100.0, cfg)

    assert result is not None
    assert result["status"] == "passed"
    assert result["num_pairs_checked"] == 1
    statuses = {tuple(p["ids"]): p.get("status", "measured") for p in result["pairs"]}
    assert statuses[(0, 7)] == "unavailable"


def test_never_raises_on_malformed_pair_entry():
    corners = make_two_marker_scene(0.8)
    cfg = {
        "known_distances_mm": [{"ids": [0], "distance_mm": 80.0}],
        "warn_tolerance_pct": 5.0,
    }

    result = check_marker_layout(corners, 100.0, cfg)

    assert result is not None
    assert result["status"] == "insufficient_markers"
