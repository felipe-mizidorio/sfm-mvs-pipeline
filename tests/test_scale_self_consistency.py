"""Tests for the marker self-consistency checks in the scale stage.

These checks measure PRECISION (do the markers agree with each other?), not
ACCURACY (are the millimetres right?). Both metrics are computed from the same
triangulated corners the primary scale uses, so they can flag inconsistent
triangulation but cannot detect a wrong marker_length_mm.
"""

import json
import math

import numpy as np

from sfm_mvs_pipeline.pipeline.orchestration import write_pipeline_manifest
from sfm_mvs_pipeline.scale.self_consistency import check_scale_self_consistency


def square_corners(
    center: np.ndarray, side: float
) -> dict[int, np.ndarray]:
    """Axis-aligned square marker corners around a center, in SfM units."""
    h = side / 2.0
    offsets = [(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)]
    return {i: center + np.array(o) for i, o in enumerate(offsets)}


MARKER_LENGTH_MM = 20.0


def test_uniform_markers_report_zero_dispersion_and_sqrt2_diagonals():
    corners = {
        mid: square_corners(np.array([mid * 1.0, 0.0, 0.0]), 0.4)
        for mid in range(3)
    }

    result = check_scale_self_consistency(corners, MARKER_LENGTH_MM)

    assert result is not None
    assert result["n_markers_used"] == 3
    assert result["n_skipped"] == 0

    scale_block = result["per_marker_scale"]
    assert scale_block["status"] == "ok"
    # side 0.4 units at 20 mm -> 50 mm/unit for every marker
    assert abs(scale_block["mean_mm_per_unit"] - 50.0) < 1e-9
    assert scale_block["std_mm_per_unit"] < 1e-9
    assert scale_block["cv"] < 1e-9
    assert scale_block["outliers"] == []

    diag_block = result["diagonal_ratio"]
    assert diag_block["status"] == "ok"
    assert diag_block["flagged"] == []
    for entry in diag_block["per_marker"].values():
        assert abs(entry["ratio"] - math.sqrt(2.0)) < 1e-9
        assert abs(entry["deviation_pct"]) < 1e-9


def test_dispersed_scales_warn_and_flag_the_outlier_marker():
    # Nine consistent markers plus one whose side triangulated 47% too long:
    # the implied per-marker scale (40 mm/unit vs ~58.8) is a >2-sigma outlier
    # and pushes the CV over the warn threshold.
    corners = {
        mid: square_corners(np.array([mid * 1.0, 0.0, 0.0]), 0.34)
        for mid in range(9)
    }
    corners[9] = square_corners(np.array([9.0, 0.0, 0.0]), 0.5)

    result = check_scale_self_consistency(corners, MARKER_LENGTH_MM)

    assert result is not None
    scale_block = result["per_marker_scale"]
    assert scale_block["status"] == "warning"
    assert scale_block["cv"] > scale_block["cv_warn_threshold"]
    assert scale_block["outliers"] == [9]
    # The result must be manifest-ready (pure JSON types, no numpy scalars).
    json.dumps(result)


def test_folded_marker_flagged_by_diagonal_ratio():
    # Non-planar marker: corner 3 lifted out of plane by 0.8x the side.
    # Sides and diagonals no longer satisfy diagonal = side * sqrt(2)
    # (deviation ~ -5.7%), while flat markers stay unflagged.
    side = 0.4
    folded = square_corners(np.zeros(3), side)
    folded[3] = folded[3] + np.array([0.0, 0.0, 0.8 * side])
    corners = {
        0: square_corners(np.array([1.0, 0.0, 0.0]), side),
        1: square_corners(np.array([2.0, 0.0, 0.0]), side),
        7: folded,
    }

    result = check_scale_self_consistency(corners, MARKER_LENGTH_MM)

    assert result is not None
    diag_block = result["diagonal_ratio"]
    assert diag_block["status"] == "warning"
    assert diag_block["flagged"] == [7]
    assert abs(diag_block["per_marker"][7]["deviation_pct"]) > diag_block["tolerance_pct"]


def test_incomplete_and_degenerate_markers_are_skipped():
    incomplete = square_corners(np.zeros(3), 0.4)
    del incomplete[2]  # only 3 corners triangulated
    degenerate = {i: np.zeros(3) for i in range(4)}  # all corners collapsed
    corners = {
        0: square_corners(np.array([1.0, 0.0, 0.0]), 0.4),
        1: incomplete,
        2: degenerate,
    }

    result = check_scale_self_consistency(corners, MARKER_LENGTH_MM)

    assert result is not None
    assert result["n_markers_used"] == 1
    assert result["n_skipped"] == 2
    # Dispersion needs at least two per-marker scales.
    assert result["per_marker_scale"]["status"] == "insufficient_markers"
    assert result["per_marker_scale"]["cv"] is None
    # The diagonal check is per-marker and still runs on the one good marker.
    assert result["diagonal_ratio"]["status"] == "ok"
    assert 0 in result["diagonal_ratio"]["per_marker"]
    assert 1 not in result["diagonal_ratio"]["per_marker"]


def test_no_usable_markers_reports_insufficient():
    result = check_scale_self_consistency({}, MARKER_LENGTH_MM)

    assert result is not None
    assert result["n_markers_used"] == 0
    assert result["per_marker_scale"]["status"] == "insufficient_markers"
    assert result["diagonal_ratio"]["status"] == "insufficient_markers"
    json.dumps(result)


def test_disabled_without_marker_length():
    corners = {0: square_corners(np.zeros(3), 0.4)}
    assert check_scale_self_consistency(corners, None) is None
    assert check_scale_self_consistency(corners, 0.0) is None


def test_manifest_records_self_consistency_block(tmp_path):
    sor_stats = {"point_cloud_filtering": {"points_before": 10, "points_after": 9}}
    lcc_stats = {"lcc": {"triangles_kept": 100, "triangles_removed": 1}}
    mesh_opts = {"depth": 9, "scale": 1.1, "linear_fit": False}
    corners = {
        mid: square_corners(np.array([mid * 1.0, 0.0, 0.0]), 0.4)
        for mid in range(2)
    }
    block = check_scale_self_consistency(corners, MARKER_LENGTH_MM)

    write_pipeline_manifest(
        tmp_path,
        "run_pipeline.py",
        sor_stats,
        lcc_stats,
        mesh_opts,
        50.0,
        scale_self_consistency=block,
    )

    data = json.loads((tmp_path / "pipeline_manifest.json").read_text())
    assert data["scale_self_consistency"]["n_markers_used"] == 2
    assert data["scale_self_consistency"]["per_marker_scale"]["status"] == "ok"


def test_manifest_omits_block_when_none(tmp_path):
    sor_stats = {"point_cloud_filtering": {"points_before": 10, "points_after": 9}}
    lcc_stats = {"lcc": {"triangles_kept": 100, "triangles_removed": 1}}
    mesh_opts = {"depth": 9, "scale": 1.1, "linear_fit": False}

    write_pipeline_manifest(
        tmp_path, "run_pipeline.py", sor_stats, lcc_stats, mesh_opts, None
    )

    data = json.loads((tmp_path / "pipeline_manifest.json").read_text())
    assert "scale_self_consistency" not in data
