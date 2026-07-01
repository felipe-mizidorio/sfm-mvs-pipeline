# sfm-mvs-pipeline

A Python pipeline for 3D reconstruction using Structure-from-Motion (SfM) and Multi-View Stereo (MVS), with support for metric scale recovery from ArUco markers and known camera intrinsics injection for neonatal cranial morphometry.

---

## Overview

The pipeline covers the full reconstruction workflow from raw images to a dense 3D mesh:

1. **Feature Extraction** — SIFT-based keypoint detection and description
2. **Feature Matching** — Exhaustive or vocabulary tree-based matching
3. **Sparse Reconstruction (SfM)** — Incremental bundle adjustment via COLMAP
4. **Dense Reconstruction (MVS)** — PatchMatch Stereo depth map estimation
5. **Depth Map Fusion** — Stereo fusion into a dense point cloud
6. **Metric Scale Recovery** — ArUco marker triangulation to convert SfM units to millimetres
7. **Surface Reconstruction** — Poisson Surface Reconstruction
8. **Evaluation** — Chamfer Distance, Hausdorff Distance, and RMS metrics

---

## Project Structure

```
sfm-mvs-pipeline/
├── src/
│   └── sfm_mvs_pipeline/
│       ├── sfm/          # Feature extraction, matching, bundle adjustment
│       ├── mvs/          # Dense reconstruction (PatchMatch Stereo, fusion)
│       ├── mesh/         # Surface reconstruction (Poisson)
│       ├── scale/        # ArUco-based metric scale recovery
│       └── evaluation/   # 3D evaluation metrics
├── scripts/              # Entrypoints to run the full pipeline
├── notebooks/            # Exploratory analysis and result visualisation
├── configs/              # YAML configuration files for COLMAP and pipeline stages
├── data/                 # Not tracked by Git — managed with DVC
│   ├── raw/              # Original input images (never modified)
│   ├── processed/        # Intermediate outputs per pipeline stage
│   └── results/          # Final metrics and reports
├── tests/
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

Install notebook dependencies:

```bash
uv sync --group notebook
```

---

## Running the Pipeline

```bash
uv run python scripts/run_pipeline.py \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene
```

### CLI Reference

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

#### Preprocessing pipeline integration

| Argument | Default | Description |
|---|---|---|
| `--frames-manifest` | `None` | Path to a JSON manifest from the ArUco preprocessing stage. Keys: `"frames"` (list of filenames to use) and optionally `"marker_detections"` (`{frame: [{id, corners}]}`) to skip re-detection during scale recovery. |

#### Fusion clipping

| Argument | Default | Description |
|---|---|---|
| `--bbox-min X Y Z` | `None` | Minimum corner of an axis-aligned bounding box applied during stereo fusion. Useful to discard background and retain only the head volume. |
| `--bbox-max X Y Z` | `None` | Maximum corner of the bounding box. Both `--bbox-min` and `--bbox-max` must be provided together. |

### Example: sparse-only run on CPU

```bash
uv run python scripts/run_pipeline.py \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene \
  --device cpu \
  --skip-mvs
```

### Example: full run with evaluation

```bash
uv run python scripts/run_pipeline.py \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene \
  --ground-truth data/raw/my_scene_gt.ply
```

### Example: neonatal capture with known intrinsics and metric scale

```bash
uv run python scripts/run_pipeline.py \
  --image-dir data/raw/session_01/frames \
  --output-dir data/processed/session_01 \
  --camera-model OPENCV \
  --camera-params "3024 3024 2016 1512 0.12 -0.05 0.0 0.0" \
  --frames-manifest data/raw/session_01/manifest.json \
  --bbox-min -0.15 -0.15 -0.05 \
  --bbox-max  0.15  0.15  0.25
```

Metric scale is applied automatically when `marker_length_mm` is set in `configs/aruco.yaml`. The output `mesh.ply` will have coordinates in millimetres.

### Example: using a preprocessing manifest

The manifest JSON produced by the ArUco preprocessing pipeline:

```json
{
  "frames": ["frame_042.jpg", "frame_043.jpg", "frame_101.jpg"],
  "marker_detections": {
    "frame_042.jpg": [{"id": 0, "corners": [[120,80],[160,80],[160,120],[120,120]]}],
    "frame_101.jpg": [{"id": 0, "corners": [[200,95],[240,95],[240,135],[200,135]]}]
  }
}
```

Pass it with `--frames-manifest path/to/manifest.json`. Only the listed frames are passed to COLMAP; stored marker detections are reused for scale recovery so images do not need to be re-read.

### Outputs

| Path | Description |
|---|---|
| `<output-dir>/database.db` | COLMAP feature database |
| `<output-dir>/sparse/` | Sparse reconstruction models |
| `<output-dir>/mvs/` | Undistorted images and depth maps |
| `<output-dir>/dense.ply` | Fused dense point cloud |
| `<output-dir>/mesh.ply` | Final reconstructed mesh |
| `<output-dir>/results/metrics.json` | Evaluation metrics (if `--ground-truth` provided) |

---

## Configuration

All pipeline parameters are controlled by YAML files in `configs/`.

### `configs/colmap.yaml`

| Section | Key | Description |
|---|---|---|
| `feature_extraction` | `max_num_features` | Max SIFT features per image (default: 8192) |
| `feature_extraction` | `first_octave` | Starting octave; `-1` = half image size |
| `feature_matching` | `method` | `exhaustive` (small datasets) or `vocab_tree` (large) |
| `feature_matching` | `vocab_tree.vocab_tree_path` | Path to vocabulary tree file (required for `vocab_tree`) |
| `feature_matching` | `vocab_tree.num_nearest_neighbors` | Nearest-neighbor images to match per query |
| `incremental_mapping` | `min_num_matches` | Minimum inlier matches to extend reconstruction |
| `incremental_mapping` | `max_num_models` | Maximum number of reconstructed models |
| `patch_match_stereo` | `max_image_size` | Max image side for downsampling before MVS |
| `patch_match_stereo` | `window_radius` | Patch window half-size for cost aggregation |
| `patch_match_stereo` | `num_samples` | Random hypothesis samples per pixel per iteration |
| `stereo_fusion` | `min_num_pixels` | Minimum consistent views to fuse a depth sample |
| `stereo_fusion` | `max_reproj_error` | Max reprojection error (px) for fusion |

### `configs/mesh.yaml`

| Key | Description |
|---|---|
| `depth` | Octree depth; higher = finer detail, slower (default: 9) |
| `scale` | Bounding box scale factor; increase to avoid boundary artifacts |
| `linear_fit` | Use conjugate-gradient solver (`true`) or direct solver (`false`) |
| `density_threshold` | Poisson density quantile below which vertices are removed (default: 0.01); reduces floating geometry at mesh boundaries |

### `configs/aruco.yaml`

| Key | Description |
|---|---|
| `marker_length_mm` | Physical side length of the ArUco square in millimetres. Set to `0` or omit to disable scale recovery. |
| `dict_id` | OpenCV ArUco dictionary ID (default: `0` = `DICT_4X4_50`) |
| `min_views` | Minimum number of registered views required to triangulate a marker corner (default: `2`) |

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
uv run pyright src/
```

### Tests

```bash
uv run pytest
```

### Notebooks

Launch JupyterLab after installing the `notebook` group:

```bash
uv run jupyter lab
```
