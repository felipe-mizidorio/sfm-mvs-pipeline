"""Head-crop tests for the domestic-use scenario.

The end user is a parent filming at home: backgrounds are unpredictable and
nobody tunes crop parameters. These tests build synthetic "room" point clouds
(head-sized blob + ArUco corners on its surface + background clutter) and run
the real CPU chain SOR → auto-crop → Poisson → LCC → Taubin, asserting the
final mesh is the head, not the background. This is core product behaviour,
not an edge case — keep these green.

Geometry convention: SfM units with a ground-truth scale of 100 mm/unit.
Head = sphere of radius 0.6 units (60 mm ≈ neonatal head) at the origin.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import open3d as o3d
import pytest
import yaml

from sfm_mvs_pipeline.pipeline.orchestration import (
    DEFAULT_HEAD_RADIUS_SFM,
    HEAD_CROP_MARGIN_MM,
    HEAD_CROP_MAX_RADIUS_MM,
    HEAD_CROP_MIN_RADIUS_MM,
    auto_head_radius,
    crop_to_sphere,
    estimate_head_center,
    run_head_crop,
    run_poisson_lcc_and_visualize,
    run_sor_and_visualize,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

SCALE_MM_PER_UNIT = 100.0
HEAD_RADIUS_UNITS = 0.6  # 60 mm
HEAD_CENTER = np.zeros(3)


# ---------------------------------------------------------------------------
# Synthetic scene builders
# ---------------------------------------------------------------------------


def _sphere_surface_points(
    n: int, radius: float, center: np.ndarray, rng: np.random.Generator, noise: float
) -> np.ndarray:
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius + rng.normal(scale=noise, size=(n, 1))
    return center + directions * radii


def _make_head_points(rng: np.random.Generator, n: int = 8000) -> np.ndarray:
    return _sphere_surface_points(n, HEAD_RADIUS_UNITS, HEAD_CENTER, rng, noise=0.004)


def _make_marker_corner_points(rng: np.random.Generator) -> np.ndarray:
    """32 ArUco corner points (8 markers × 4 corners) on the crown of the head."""
    corners = []
    for k in range(8):
        azimuth = 2 * np.pi * k / 8
        polar = np.deg2rad(25 + 15 * (k % 3))  # crown region only
        center = HEAD_RADIUS_UNITS * np.array(
            [
                np.sin(polar) * np.cos(azimuth),
                np.sin(polar) * np.sin(azimuth),
                np.cos(polar),
            ]
        )
        # 4 corners offset tangentially (50 mm marker → 0.5 units side, half = 0.25).
        tangent_a = np.cross(center, [0.0, 0.0, 1.0])
        if np.linalg.norm(tangent_a) < 1e-9:
            tangent_a = np.array([1.0, 0.0, 0.0])
        tangent_a /= np.linalg.norm(tangent_a)
        tangent_b = np.cross(center / np.linalg.norm(center), tangent_a)
        half = 0.25
        for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(center + sa * half * tangent_a + sb * half * tangent_b)
    return np.array(corners) + rng.normal(scale=0.005, size=(32, 3))


def _make_clutter(scenario: str, rng: np.random.Generator) -> np.ndarray:
    """Background clutter for three domestic 'room' scenarios."""
    if scenario == "sparse_far":
        # A few hundred stray points scattered around the room, 0.3–0.8 m away.
        directions = rng.normal(size=(400, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = rng.uniform(3.0, 8.0, size=(400, 1))
        return directions * radii

    if scenario == "dense_near_wall":
        # A dense wall 220 mm behind the head plus a toy 190 mm from centre.
        wall = np.column_stack(
            [
                np.full(6000, 2.2) + rng.normal(scale=0.01, size=6000),
                rng.uniform(-2.0, 2.0, size=6000),
                rng.uniform(-2.0, 2.0, size=6000),
            ]
        )
        toy = _sphere_surface_points(
            1500, 0.3, np.array([1.9, 0.0, 0.0]), rng, noise=0.005
        )
        return np.vstack([wall, toy])

    if scenario == "clutter_overlapping_crop":
        # A crib bar passing through the crop sphere (closest point ~141 mm from
        # centre, inside the ~160 mm auto radius) plus a wall behind it.
        ys = rng.uniform(-2.0, 2.0, size=3000)
        angles = rng.uniform(0, 2 * np.pi, size=3000)
        bar = np.column_stack(
            [
                1.4 + 0.08 * np.cos(angles),
                ys,
                0.2 + 0.08 * np.sin(angles),
            ]
        )
        wall = np.column_stack(
            [
                np.full(4000, 3.0) + rng.normal(scale=0.01, size=4000),
                rng.uniform(-2.0, 2.0, size=4000),
                rng.uniform(-2.0, 2.0, size=4000),
            ]
        )
        return np.vstack([bar, wall])

    raise ValueError(f"unknown scenario: {scenario}")


def _make_camera_ring_reconstruction(
    n_cameras: int = 16, ring_radius: float = 2.5
) -> MagicMock:
    """Mock pycolmap Reconstruction: cameras orbiting the head, aimed at origin."""
    images = {}
    for i in range(n_cameras):
        angle = 2 * np.pi * i / n_cameras
        height = 0.3 + 0.9 * (i % 4) / 3
        cam_pos = np.array(
            [ring_radius * np.cos(angle), ring_radius * np.sin(angle), height]
        )
        forward = (HEAD_CENTER - cam_pos).astype(float)
        forward /= np.linalg.norm(forward)
        right = np.cross([0.0, 0.0, 1.0], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.vstack([right, down, forward])
        t = -R @ cam_pos

        cfw = MagicMock()
        cfw.rotation.matrix.return_value = R
        cfw.translation = t
        img = MagicMock()
        img.has_pose = True
        img.cam_from_world.return_value = cfw
        images[i] = img

    recon = MagicMock()
    recon.images = images
    return recon


def _write_cloud(points: np.ndarray, path: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path), pcd)


# ---------------------------------------------------------------------------
# Unit tests: auto radius derivation and head-centre estimation
# ---------------------------------------------------------------------------


def test_auto_head_radius_from_markers():
    rng = np.random.default_rng(0)
    markers = _make_marker_corner_points(rng)
    radius = auto_head_radius(HEAD_CENTER, markers, SCALE_MM_PER_UNIT)
    # median corner distance (~0.65 units, corners stick out tangentially)
    # + 100 mm margin, within clamps.
    expected = float(
        np.median(np.linalg.norm(markers - HEAD_CENTER, axis=1))
    ) + HEAD_CROP_MARGIN_MM / SCALE_MM_PER_UNIT
    assert radius == pytest.approx(expected, rel=1e-6)
    assert HEAD_CROP_MIN_RADIUS_MM / SCALE_MM_PER_UNIT <= radius
    assert radius <= HEAD_CROP_MAX_RADIUS_MM / SCALE_MM_PER_UNIT


def test_auto_head_radius_min_clamp():
    markers = np.tile(np.array([[0.05, 0.0, 0.0]]), (8, 1))
    radius = auto_head_radius(HEAD_CENTER, markers, SCALE_MM_PER_UNIT)
    assert radius == pytest.approx(HEAD_CROP_MIN_RADIUS_MM / SCALE_MM_PER_UNIT)


def test_auto_head_radius_max_clamp():
    markers = np.tile(np.array([[3.0, 0.0, 0.0]]), (8, 1))
    radius = auto_head_radius(HEAD_CENTER, markers, SCALE_MM_PER_UNIT)
    assert radius == pytest.approx(HEAD_CROP_MAX_RADIUS_MM / SCALE_MM_PER_UNIT)


def test_auto_head_radius_unavailable_inputs():
    markers = np.array([[0.6, 0.0, 0.0]])
    assert auto_head_radius(HEAD_CENTER, markers, None) is None
    assert auto_head_radius(HEAD_CENTER, None, SCALE_MM_PER_UNIT) is None
    assert auto_head_radius(HEAD_CENTER, np.empty((0, 3)), SCALE_MM_PER_UNIT) is None


def test_estimate_head_center_from_camera_ring():
    recon = _make_camera_ring_reconstruction()
    center = estimate_head_center(recon)
    np.testing.assert_allclose(center, HEAD_CENTER, atol=1e-9)


def test_crop_to_sphere_keeps_inside_points():
    pcd = o3d.geometry.PointCloud()
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pcd.points = o3d.utility.Vector3dVector(pts)
    cropped = crop_to_sphere(pcd, np.zeros(3), 1.0)
    assert len(cropped.points) == 2


# ---------------------------------------------------------------------------
# run_head_crop behaviour
# ---------------------------------------------------------------------------


def test_run_head_crop_zero_radius_disables_crop(tmp_path):
    rng = np.random.default_rng(1)
    ply = tmp_path / "dense_filtered.ply"
    _write_cloud(_make_head_points(rng, n=500), ply)
    result_ply, stats = run_head_crop(
        ply,
        tmp_path,
        _make_camera_ring_reconstruction(),
        head_radius_override=0.0,
        scale_factor=SCALE_MM_PER_UNIT,
        marker_points=_make_marker_corner_points(rng),
    )
    assert result_ply == ply
    assert stats == {}


def test_run_head_crop_falls_back_without_scale(tmp_path):
    rng = np.random.default_rng(2)
    ply = tmp_path / "dense_filtered.ply"
    _write_cloud(_make_head_points(rng, n=500), ply)
    result_ply, stats = run_head_crop(
        ply,
        tmp_path,
        _make_camera_ring_reconstruction(),
        head_radius_override=None,
        scale_factor=None,
        marker_points=None,
    )
    assert result_ply == tmp_path / "dense_filtered_cropped.ply"
    assert stats["head_crop"]["radius_source"] == "default_fallback"
    assert stats["head_crop"]["radius_sfm_units"] == pytest.approx(
        DEFAULT_HEAD_RADIUS_SFM
    )


def test_run_head_crop_override_wins_over_auto(tmp_path):
    rng = np.random.default_rng(3)
    ply = tmp_path / "dense_filtered.ply"
    _write_cloud(_make_head_points(rng, n=500), ply)
    _, stats = run_head_crop(
        ply,
        tmp_path,
        _make_camera_ring_reconstruction(),
        head_radius_override=2.0,
        scale_factor=SCALE_MM_PER_UNIT,
        marker_points=_make_marker_corner_points(rng),
    )
    assert stats["head_crop"]["radius_source"] == "override"
    assert stats["head_crop"]["radius_sfm_units"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# End-to-end domestic scenarios: SOR → auto-crop → Poisson → LCC → Taubin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario", ["sparse_far", "dense_near_wall", "clutter_overlapping_crop"]
)
def test_domestic_background_removed_end_to_end(scenario, tmp_path):
    rng = np.random.default_rng(42)
    head = _make_head_points(rng)
    clutter = _make_clutter(scenario, rng)
    dense_ply = tmp_path / "dense.ply"
    _write_cloud(np.vstack([head, clutter]), dense_ply)

    mesh_cfg = yaml.safe_load((_REPO_ROOT / "configs/mesh.yaml").read_text())

    # SOR (real config values)
    dense_filtered_ply, sor_stats = run_sor_and_visualize(
        dense_ply, tmp_path, mesh_cfg["point_cloud_filtering"]
    )

    # Auto crop from ArUco-derived scale + marker positions
    markers = _make_marker_corner_points(rng)
    input_for_poisson, crop_stats = run_head_crop(
        dense_filtered_ply,
        tmp_path,
        _make_camera_ring_reconstruction(),
        head_radius_override=None,
        scale_factor=SCALE_MM_PER_UNIT,
        marker_points=markers,
    )
    assert input_for_poisson == tmp_path / "dense_filtered_cropped.ply"
    assert crop_stats["head_crop"]["radius_source"] == "aruco_auto"
    # Crop keeps the head (allow some loss to SOR) …
    assert crop_stats["head_crop"]["points_after"] >= 0.8 * len(head)
    # … and never keeps the full clutter volume.
    assert crop_stats["head_crop"]["points_after"] < len(head) + 0.5 * len(clutter)

    # Poisson → density trim → LCC → Taubin (real config values)
    mesh, lcc_stats = run_poisson_lcc_and_visualize(
        input_for_poisson,
        tmp_path / "mesh.ply",
        tmp_path,
        mesh_cfg["poisson_surface_reconstruction"],
    )

    assert len(mesh.vertices) > 1000, "final mesh suspiciously small — LCC red flag"

    extent = mesh.get_max_bound() - mesh.get_min_bound()
    max_allowed = 2 * HEAD_RADIUS_UNITS * 1.35  # head diameter + Poisson/smoothing slack
    assert np.all(extent <= max_allowed), (
        f"[{scenario}] mesh bbox {extent} exceeds head-sized bound {max_allowed}: "
        "background survived into the final mesh"
    )
    mesh_center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    assert np.linalg.norm(mesh_center - HEAD_CENTER) < 0.15, (
        f"[{scenario}] mesh centre {mesh_center} far from head centre"
    )

    # Numbers for the CPU-validation report.
    print(
        f"\n[{scenario}] head={len(head)} clutter={len(clutter)} "
        f"sor_after={sor_stats['point_cloud_filtering']['points_after']} "
        f"crop_radius={crop_stats['head_crop']['radius_sfm_units']:.3f}u "
        f"crop_after={crop_stats['head_crop']['points_after']} "
        f"lcc_kept={lcc_stats['lcc']['triangles_kept']} "
        f"lcc_removed={lcc_stats['lcc']['triangles_removed']} "
        f"mesh_vertices={len(mesh.vertices)} bbox_extent={np.round(extent, 3)}"
    )


# ---------------------------------------------------------------------------
# KNOWN LIMITATION: background contiguous with the head
# ---------------------------------------------------------------------------


def _make_contiguous_blanket(rng: np.random.Generator, r_outer: float = 2.0) -> np.ndarray:
    """Blanket plane the head physically rests on (z = -0.58 cuts the sphere).

    Sampled from the head-contact ring (r ≈ 0.153) outward at head-like point
    density, so Poisson fuses blanket and head into one connected surface.
    """
    blanket_z = -0.58
    contact_r = np.sqrt(HEAD_RADIUS_UNITS**2 - blanket_z**2)
    spacing = 0.024  # ≈ head surface point spacing (8000 pts on r=0.6 sphere)
    n = int(np.pi * (r_outer**2 - contact_r**2) / spacing**2)
    rr = np.sqrt(rng.uniform(contact_r**2, r_outer**2, size=n))
    th = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.column_stack(
        [rr * np.cos(th), rr * np.sin(th), np.full(n, blanket_z)]
    )
    return pts + rng.normal(scale=0.004, size=(n, 3))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN LIMITATION (documented, not yet addressed): background that is "
        "physically contiguous with the head — blanket/pillow/mattress the head "
        "rests on, a caregiver's hand — fuses with the head into ONE Poisson "
        "surface component, and LCC-by-largest-component cannot separate it by "
        "definition. Measured current behaviour (2026-07-08): LCC removes only "
        "~40 of ~142k triangles, final mesh bbox ≈ 3.5×3.5×1.2 units vs the "
        "head's 1.2, ~55% of final vertices belong to the blanket. There is no "
        "point-distance threshold to tune: LCC runs on shared-edge triangle "
        "adjacency of the Poisson mesh, and at depth 9 the Poisson voxel here "
        "is ~0.007 units (~0.7 mm), so any touching surface merges. Mitigation "
        "candidates (support-plane removal before Poisson, capture-protocol "
        "guidance) are tracked in project docs; when one lands and this test "
        "XPASSes, remove this marker."
    ),
)
def test_contiguous_blanket_limitation(tmp_path):
    """Asserts the DESIRED behaviour (head-sized mesh) — currently fails."""
    rng = np.random.default_rng(7)
    head = _make_head_points(rng)
    blanket = _make_contiguous_blanket(rng)
    dense_ply = tmp_path / "dense.ply"
    _write_cloud(np.vstack([head, blanket]), dense_ply)

    mesh_cfg = yaml.safe_load((_REPO_ROOT / "configs/mesh.yaml").read_text())
    dense_filtered_ply, _ = run_sor_and_visualize(
        dense_ply, tmp_path, mesh_cfg["point_cloud_filtering"]
    )
    input_for_poisson, crop_stats = run_head_crop(
        dense_filtered_ply,
        tmp_path,
        _make_camera_ring_reconstruction(),
        head_radius_override=None,
        scale_factor=SCALE_MM_PER_UNIT,
        marker_points=_make_marker_corner_points(rng),
    )
    # The crop itself works (auto radius, most of the blanket removed) …
    assert crop_stats["head_crop"]["radius_source"] == "aruco_auto"

    mesh, _ = run_poisson_lcc_and_visualize(
        input_for_poisson,
        tmp_path / "mesh.ply",
        tmp_path,
        mesh_cfg["poisson_surface_reconstruction"],
    )

    # … but the blanket ring inside the crop stays connected to the head, so
    # this head-sized bound is what a fix must achieve — and currently isn't.
    extent = mesh.get_max_bound() - mesh.get_min_bound()
    assert np.all(extent <= 2 * HEAD_RADIUS_UNITS * 1.35), (
        f"contiguous blanket survived into final mesh: bbox extent {extent}"
    )


# ---------------------------------------------------------------------------
# --skip-fusion double-scale guard (resume_from_mvs.py)
# ---------------------------------------------------------------------------

_RESUME_SCRIPT = _REPO_ROOT / "scripts" / "resume_from_mvs.py"


def _run_resume_skip_fusion(output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(_RESUME_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--image-dir",
            str(output_dir),
            "--skip-fusion",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


def test_skip_fusion_refuses_already_scaled_dense(tmp_path):
    (tmp_path / "pipeline_manifest.json").write_text(
        json.dumps(
            {"run_script": "resume_from_mvs.py", "scale_factor_mm_per_unit": 123.4}
        )
    )
    result = _run_resume_skip_fusion(tmp_path)
    assert result.returncode == 1
    assert "double-scale" in result.stdout + result.stderr


def test_skip_fusion_allowed_after_run_pipeline(tmp_path):
    # run_pipeline.py never scales dense.ply itself, so --skip-fusion is safe;
    # the guard must not trigger (the script then fails later on the missing
    # sparse model, which is expected in this bare tmp dir).
    (tmp_path / "pipeline_manifest.json").write_text(
        json.dumps({"run_script": "run_pipeline.py", "scale_factor_mm_per_unit": 123.4})
    )
    result = _run_resume_skip_fusion(tmp_path)
    assert "double-scale" not in result.stdout + result.stderr
