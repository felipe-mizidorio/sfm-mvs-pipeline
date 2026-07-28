# sfm-mvs-pipeline

A Python pipeline for 3D reconstruction using Structure-from-Motion (SfM) and Multi-View Stereo (MVS), with support for metric scale recovery from ArUco markers and known camera intrinsics injection for neonatal cranial morphometry.

---

## Overview

The pipeline covers the full reconstruction workflow from raw images to a dense 3D mesh in millimetres:

1. **Feature Extraction** — SIFT-based keypoint detection and description
2. **Feature Matching** — Sequential, exhaustive, or vocabulary tree-based matching
3. **Sparse Reconstruction (SfM)** — Incremental bundle adjustment via COLMAP
4. **Dense Reconstruction (MVS)** — PatchMatch Stereo depth map estimation
5. **Depth Map Fusion** — Stereo fusion into a dense point cloud
6. **Point Cloud Filtering** — Statistical Outlier Removal (SOR)
7. **Metric Scale Recovery** — ArUco marker triangulation to convert SfM units to millimetres, with layout and self-consistency checks
8. **Head Crop** — Spherical crop to the head region, auto-sized from the triangulated markers
9. **Surface Reconstruction** — Poisson reconstruction, density trim, largest-connected-component filtering, optional Taubin smoothing
10. **Evaluation** — Chamfer Distance, Hausdorff Distance, and RMS metrics (optional)

Every run writes a `pipeline_manifest.json` recording the resolved configuration, library versions, per-stage point/triangle counts, and the metric-scale status.

---

## Project Structure

```
sfm-mvs-pipeline/
├── src/
│   └── sfm_mvs_pipeline/
│       ├── cli/          # Console entrypoints (run, resume_mvs, resume_dense)
│       ├── sfm/          # Feature extraction, matching, bundle adjustment
│       ├── mvs/          # Dense reconstruction, fusion, mask undistortion
│       ├── postprocess/  # Dense-cloud filtering (SOR, membrane filter)
│       ├── pipeline/     # Shared post-fusion orchestration + manifest
│       ├── mesh/         # Surface reconstruction (Poisson)
│       ├── scale/        # ArUco-based metric scale recovery + scale policy
│       ├── evaluation/   # 3D evaluation metrics
│       └── visualization/ # Plotly HTML checkpoints
├── tests/                # Mirrors src/ layout (one directory per package)
├── configs/              # YAML configuration files for COLMAP and pipeline stages
├── data/                 # Folder skeleton tracked via .gitkeep; contents managed with DVC
│   ├── raw/              # Original input images (never modified)
│   ├── processed/        # Intermediate outputs per pipeline stage
│   └── results/          # Final metrics and reports
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python 3.12
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) (required for GPU acceleration on Linux)
- An NVIDIA GPU is required for the MVS stage (`patch_match_stereo`)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/sfm-mvs-pipeline.git
cd sfm-mvs-pipeline
```

Install core dependencies (CPU, e.g. development machine without GPU):

```bash
uv sync
```

Install with GPU support (NVIDIA GPU + CUDA 12):

```bash
uv sync --group gpu
```

> **Note:** The `gpu` group replaces the CPU-only `pycolmap` with `pycolmap-cuda12`, which enables GPU-accelerated feature extraction, matching, and MVS. The CUDA Toolkit must be installed separately on Linux.

Install dev tools (linting, type checking, tests):

```bash
uv sync --group dev
```

---

## Running the Pipeline

```bash
uv run sfm-mvs-run \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene
```

Two resume entrypoints skip the expensive early stages when re-running the
post-fusion pipeline on an existing output directory:

- `sfm-mvs-resume-mvs` — re-runs stereo fusion onward from an existing MVS workspace (depth maps).
- `sfm-mvs-resume-dense` — resumes from an existing `dense.ply` (SOR → scale → Poisson).

Each command is also runnable as a module, e.g. `uv run python -m sfm_mvs_pipeline.cli.run`.

### `sfm-mvs-run` reference

#### Core

| Argument | Default | Description |
|---|---|---|
| `--image-dir` | *(required)* | Directory of input images; supports subdirectories |
| `--output-dir` | *(required)* | Root directory for all pipeline outputs |
| `--colmap-config` | `configs/colmap.yaml` | Path to COLMAP config file |
| `--mesh-config` | `configs/mesh.yaml` | Path to mesh reconstruction config |
| `--evaluation-config` | `configs/evaluation.yaml` | Path to evaluation config |
| `--aruco-config` | `configs/aruco.yaml` | Path to ArUco scale recovery config |
| `--ground-truth` | `None` | Path to ground truth `.ply` for evaluation (optional) |
| `--skip-mvs` | `False` | Stop after sparse reconstruction (useful on CPU-only machines) |
| `--device` | `auto` | Device for pycolmap: `auto` or `cpu` |

#### Camera calibration

| Argument | Default | Description |
|---|---|---|
| `--camera-model` | `None` | COLMAP camera model name (e.g. `OPENCV`, `PINHOLE`, `SIMPLE_RADIAL`). When set, a single shared camera is used for all images and self-calibration is skipped. |
| `--camera-params` | `None` | Space-separated intrinsics matching the model. For `OPENCV`: `"fx fy cx cy k1 k2 p1 p2"`. For `PINHOLE`: `"fx fy cx cy"`. |

Intrinsics are resolved in this order: explicit flags above → focal metadata in the frames manifest (EXIF-derived prior) → shared self-calibration. All images always share one camera model, since captures come from a single device.

#### Preprocessing pipeline integration

| Argument | Default | Description |
|---|---|---|
| `--frames-manifest` | `None` | Path to a JSON manifest from the ArUco preprocessing stage. See [Frames manifest](#frames-manifest) below. |

#### Metric scale

| Argument | Default | Description |
|---|---|---|
| `--allow-unscaled` | `False` | Continue when metric scale recovery fails, writing output in arbitrary SfM units. Without it, a failed recovery is a hard stop. See [Metric scale policy](#metric-scale-policy). |

#### Head crop

The dense cloud is automatically cropped to a sphere around the head before meshing. The centre is the centroid of the triangulated ArUco corners (falling back to the intersection of camera optical axes when fewer than 8 corners are available), and the radius is derived from the marker positions plus a neonatal-anatomy margin — no user input needed, by design, for at-home captures with arbitrary backgrounds. Requires `marker_length_mm` in `configs/aruco.yaml`; without a recovered scale the crop falls back to a fixed 1.5 SfM-unit radius.

| Argument | Default | Description |
|---|---|---|
| `--head-radius` | `None` (auto) | **Debug-only** override of the crop radius, in SfM units. `0` disables the crop. Not needed in normal use. |

#### Fusion clipping and masking

| Argument | Default | Description |
|---|---|---|
| `--bbox-min X Y Z` | `None` | Minimum corner of an axis-aligned bounding box applied during stereo fusion. Useful to discard background and retain only the head volume. |
| `--bbox-max X Y Z` | `None` | Maximum corner of the bounding box. Both `--bbox-min` and `--bbox-max` must be provided together. |
| `--fusion-masks` | `False` | **Experimental.** Warp the frames-manifest masks into the undistorted MVS workspace and restrict stereo fusion to them. Off by default: with the current ArUco convex-hull masks this removes genuine head surface away from the markers without reducing silhouette bleed. Masks always apply to feature extraction regardless of this flag. |

#### Membrane filter

| Argument | Default | Description |
|---|---|---|
| `--membrane-filter` | `False` | Remove pale "membrane" contamination from the cropped cloud before Poisson, protecting the white ArUco marker faces. Off by default because it is **scene-dependent**: it assumes a dark subject against pale contamination and would delete the subject in a capture where the subject is pale. Requires a recovered scale and triangulated markers; skipped with a warning otherwise. |
| `--membrane-pale-threshold` | `150.0` | Mean RGB (0–255) at or above which a point counts as pale. |
| `--membrane-marker-margin-mm` | `5.0` | Protection margin added to each marker's own corner extent, in mm. |

### `sfm-mvs-resume-mvs` reference

Re-fuses depth maps from an existing `<output-dir>/mvs/` workspace and runs everything downstream. Accepts `--output-dir`, `--image-dir`, `--frames-manifest`, `--aruco-config`, `--mesh-config`, `--colmap-config`, `--bbox-min`, `--bbox-max`, `--head-radius`, `--fusion-masks`, `--membrane-filter`, `--membrane-pale-threshold`, `--membrane-marker-margin-mm`, and `--allow-unscaled` (same semantics as above), plus:

| Argument | Default | Description |
|---|---|---|
| `--skip-fusion` | `False` | Skip stereo fusion and reuse the existing `dense.ply`. Refuses to run if a previous `sfm-mvs-resume-mvs` run already scaled `dense.ply` to millimetres, which would double-scale the geometry. |

Unlike `sfm-mvs-run`, `--fusion-masks` here requires `--frames-manifest` with a `mask_dir` and aborts if either is missing.

```bash
uv run sfm-mvs-resume-mvs \
  --output-dir data/processed/session_01 \
  --image-dir data/raw/session_01/frames \
  --frames-manifest data/raw/session_01/manifest.json
```

### `sfm-mvs-resume-dense` reference

Resumes from an existing `<output-dir>/dense.ply`: SOR → scale recovery → Poisson + LCC. Accepts `--output-dir`, `--image-dir`, `--frames-manifest`, `--aruco-config`, and `--mesh-config`. It performs no head crop and does not enforce the scale policy — an unrecoverable scale leaves the output in SfM units with `scale_factor_mm_per_unit: null` in the manifest.

```bash
uv run sfm-mvs-resume-dense \
  --output-dir data/processed/session_01 \
  --image-dir data/raw/session_01/frames
```

### Example: sparse-only run on CPU

```bash
uv run sfm-mvs-run \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene \
  --device cpu \
  --skip-mvs
```

### Example: full run with evaluation

```bash
uv run sfm-mvs-run \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene \
  --ground-truth data/raw/my_scene_gt.ply
```

### Example: neonatal capture with known intrinsics and metric scale

```bash
uv run sfm-mvs-run \
  --image-dir data/raw/session_01/frames \
  --output-dir data/processed/session_01 \
  --camera-model OPENCV \
  --camera-params "3024 3024 2016 1512 0.12 -0.05 0.0 0.0" \
  --frames-manifest data/raw/session_01/manifest.json \
  --bbox-min -0.15 -0.15 -0.05 \
  --bbox-max  0.15  0.15  0.25
```

Metric scale is applied automatically when `marker_length_mm` is set in `configs/aruco.yaml`. The output `mesh.ply` will have coordinates in millimetres.

### Frames manifest

The manifest JSON produced by the ArUco preprocessing pipeline:

```json
{
  "frames": ["frame_042.jpg", "frame_043.jpg", "frame_101.jpg"],
  "mask_dir": "masks",
  "marker_detections": {
    "frame_042.jpg": [{"id": 0, "corners": [[120,80],[160,80],[160,120],[120,120]]}],
    "frame_101.jpg": [{"id": 0, "corners": [[200,95],[240,95],[240,135],[200,135]]}]
  }
}
```

Pass it with `--frames-manifest path/to/manifest.json`.

| Key | Effect |
|---|---|
| `frames` | Only the listed frames are passed to COLMAP |
| `marker_detections` | Reused for scale recovery so images are not re-read |
| `mask_dir` | Directory relative to `--image-dir` holding per-image masks; applied to feature extraction, and to stereo fusion under `--fusion-masks` |

Focal-length metadata in the manifest, when present, is used as the camera intrinsics prior (see [Camera calibration](#camera-calibration)).

---

## Metric scale policy

Scale recovery is deliberately non-fatal — a marker hiccup must not waste hours of GPU time — so it logs and returns no factor rather than raising. To prevent that from silently producing a complete, metric-looking mesh in arbitrary units, every run classifies its scale state in `pipeline_manifest.json` under `scale.status`:

| Status | Meaning |
|---|---|
| `validated` | Scale recovered and confirmed against the configured cap layout |
| `recovered_unvalidated` | Scale recovered but never checked against ground truth (no usable `layout_check.known_distances_mm`). Internally consistent, of unverified accuracy — precision, not accuracy. |
| `recovered_failed_validation` | Scale recovered but inter-marker distances deviate beyond the tolerance. Suspect; review before measuring. |
| `unscaled` | No scale recovered. Coordinates are in arbitrary SfM units, **not** millimetres. |

Only `unscaled` stops a run, and only without `--allow-unscaled`. Artefacts written under that flag are renamed to `*.UNSCALED_sfm_units.*` so a stray `.ply` cannot later be mistaken for metric output.

Two warn-only quality checks are recorded alongside it: `scale_sanity_check` (triangulated inter-marker distances vs. the known cap layout) and `scale_self_consistency` (per-marker scale dispersion, and the diagonal/side ratio of each square). Neither ever aborts a run.

---

## Outputs

| Path | Description |
|---|---|
| `<output-dir>/database.db` | COLMAP feature database |
| `<output-dir>/sparse/` | Sparse reconstruction models |
| `<output-dir>/mvs/` | Undistorted images and depth maps |
| `<output-dir>/dense.ply` | Fused dense point cloud |
| `<output-dir>/dense_filtered.ply` | Dense cloud after SOR |
| `<output-dir>/dense_filtered_cropped.ply` | After the spherical head crop |
| `<output-dir>/dense_filtered_cropped_membrane.ply` | After the membrane filter (only with `--membrane-filter`) |
| `<output-dir>/mesh.ply` | Final reconstructed mesh |
| `<output-dir>/visualizations/*.html` | Plotly checkpoints (dense raw/after SOR, mesh before/after LCC, after Taubin) |
| `<output-dir>/pipeline_manifest.json` | Resolved configs, library versions, per-stage counts, scale status, non-determinism notes |
| `<output-dir>/results/metrics.json` | Evaluation metrics (if `--ground-truth` provided) |

`pipeline_manifest.json` records which stages legitimately vary between runs of the same input: GPU PatchMatch stereo and multi-threaded matching/mapping are non-deterministic; SOR, cropping, scale recovery and Poisson are not.

---

## Configuration

All pipeline parameters are controlled by YAML files in `configs/`. Config comments carry the measured evidence behind each value — update the comment alongside the value.

### `configs/colmap.yaml`

| Section | Key | Description |
|---|---|---|
| `feature_extraction` | `max_num_features` | Max SIFT features per image (default: 8192) |
| `feature_extraction` | `first_octave` | Starting octave; `-1` = half image size |
| `feature_matching` | `method` | `sequential` (default; video captures, O(N)), `exhaustive` (small or weakly-overlapping sets), or `vocab_tree` (large) |
| `feature_matching` | `sequential.overlap` | Subsequent frames matched against each frame (default: 5) |
| `feature_matching` | `vocab_tree.vocab_tree_path` | Path to vocabulary tree file (required for `vocab_tree`) |
| `feature_matching` | `vocab_tree.num_nearest_neighbors` | Nearest-neighbor images to match per query |
| `incremental_mapping` | `min_num_matches` | Minimum inlier matches to extend reconstruction |
| `incremental_mapping` | `max_num_models` | Maximum number of reconstructed models |
| `patch_match_stereo` | `max_image_size` | Max image side for downsampling before MVS |
| `patch_match_stereo` | `window_radius` | Patch window half-size for cost aggregation |
| `patch_match_stereo` | `num_samples` | Random hypothesis samples per pixel per iteration |
| `stereo_fusion` | `min_num_pixels` | Minimum consistent views to fuse a depth sample |
| `stereo_fusion` | `max_reproj_error` | Max reprojection error (px) for fusion |

`configs/colmap_exhaustive.yaml` is a copy differing only in `feature_matching.method: exhaustive`, for captures where the sequential match graph is too sparse and incremental mapping splits into multiple models. Pass it with `--colmap-config`; changes to `colmap.yaml` usually need mirroring there.

### `configs/mesh.yaml`

| Section | Key | Description |
|---|---|---|
| `poisson_surface_reconstruction` | `depth` | Octree depth; higher = finer detail, slower (default: 9) |
| `poisson_surface_reconstruction` | `scale` | Bounding box scale factor; increase to avoid boundary artifacts |
| `poisson_surface_reconstruction` | `linear_fit` | Use conjugate-gradient solver (`true`) or direct solver (`false`) |
| `poisson_surface_reconstruction` | `density_threshold` | Poisson density quantile below which vertices are removed (default: 0.01). This is a **global** quantile, so raising it shaves the whole surface uniformly. |
| `poisson_surface_reconstruction` | `keep_largest_component` | Keep only the largest connected component after Poisson (removes background noise) |
| `poisson_surface_reconstruction` | `taubin_smoothing` | `iterations` / `lambda_filter` / `mu`. Disabled by default (`iterations: 0`); set > 0 to re-enable. |
| `point_cloud_filtering` | `nb_neighbors` | Neighbourhood size for the SOR mean-distance computation (default: 20) |
| `point_cloud_filtering` | `std_ratio` | Remove points whose mean neighbour distance exceeds mean + `std_ratio` × std dev (default: 2.0) |

### `configs/aruco.yaml`

| Key | Description |
|---|---|
| `marker_length_mm` | Physical side length of the ArUco square in millimetres. Set to `0` or omit to disable scale recovery. |
| `dict_id` | OpenCV ArUco dictionary ID (`0` = `DICT_4X4_50`, `1` = `DICT_4X4_100`, `2` = `DICT_4X4_250`) |
| `min_views` | Minimum number of registered views required to triangulate a marker corner (default: `2`) |
| `layout_check.known_distances_mm` | Known physical centre-to-centre distances between markers on the rigid cap, as `{ids: [a, b], distance_mm: d}` entries. Empty list disables the check. |
| `layout_check.warn_tolerance_pct` | Residual (percent of the expected distance) above which the check warns (default: `5.0`) |

### `configs/evaluation.yaml`

| Key | Description |
|---|---|
| `metrics.chamfer` | Enable Chamfer Distance (symmetric mean NN distance) |
| `metrics.hausdorff` | Enable Hausdorff Distance (max NN distance; outlier-sensitive) |
| `metrics.rms` | Enable RMS of nearest-neighbour distances |
| `output_format` | Output format for results (`json`) |

---

## Development

### Linting and type checking

```bash
uv run ruff check src/
uv run ruff format src/
uv run pyright src/
```

`ruff check --fix` and `ruff format` also run as pre-commit hooks:

```bash
uv run pre-commit install
```

### Tests

The suite mocks COLMAP and runs on synthetic geometry, so it needs no GPU and no image data.

```bash
uv run pytest                                     # full suite
uv run pytest tests/scale/test_scale.py            # one file
uv run pytest tests/scale/test_scale.py::test_recover_scale_synthetic  # one test
uv run pytest -k head_crop                         # by name
```
