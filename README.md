# sfm-mvs-pipeline

A Python pipeline for 3D reconstruction using Structure-from-Motion (SfM) and Multi-View Stereo (MVS).

---

## Overview

The pipeline covers the full reconstruction workflow from raw images to a dense 3D mesh:

1. **Feature Extraction** — SIFT-based keypoint detection and description
2. **Feature Matching** — Exhaustive or vocabulary tree-based matching
3. **Sparse Reconstruction (SfM)** — Incremental bundle adjustment via COLMAP
4. **Dense Reconstruction (MVS)** — PatchMatch Stereo depth map estimation
5. **Depth Map Fusion** — Stereo fusion into a dense point cloud
6. **Surface Reconstruction** — Poisson Surface Reconstruction
7. **Evaluation** — Chamfer Distance, Hausdorff Distance, and RMS metrics

---

## Project Structure

```
sfm-mvs-pipeline/
├── src/
│   └── sfm_mvs_pipeline/
│       ├── sfm/          # Feature extraction, matching, bundle adjustment
│       ├── mvs/          # Dense reconstruction (PatchMatch Stereo, fusion)
│       ├── mesh/         # Surface reconstruction (Poisson)
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
poetry install
```

Install with GPU support (NVIDIA GPU + CUDA 12):

```bash
poetry install --with gpu
```

> **Note:** The `gpu` group replaces the CPU-only `pycolmap` with `pycolmap-cuda12`, which enables GPU-accelerated feature extraction, matching, and MVS. The CUDA Toolkit must be installed separately on Linux.

Install dev tools (linting, type checking, tests):

```bash
poetry install --with dev
```

Install notebook dependencies:

```bash
poetry install --with notebook
```

---

## Running the Pipeline

```bash
poetry run python scripts/run_pipeline.py \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene
```

### CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--image-dir` | *(required)* | Directory containing input images |
| `--output-dir` | *(required)* | Root directory for all pipeline outputs |
| `--colmap-config` | `configs/colmap.yaml` | Path to COLMAP config file |
| `--mesh-config` | `configs/mesh.yaml` | Path to mesh reconstruction config |
| `--evaluation-config` | `configs/evaluation.yaml` | Path to evaluation config |
| `--ground-truth` | `None` | Path to ground truth `.ply` for evaluation (optional) |
| `--skip-mvs` | `False` | Stop after sparse reconstruction (useful on CPU-only machines) |
| `--device` | `auto` | Device for pycolmap: `auto` or `cpu` |

### Example: sparse-only run on CPU

```bash
poetry run python scripts/run_pipeline.py \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene \
  --device cpu \
  --skip-mvs
```

### Example: full run with evaluation

```bash
poetry run python scripts/run_pipeline.py \
  --image-dir data/raw/my_scene \
  --output-dir data/processed/my_scene \
  --ground-truth data/raw/my_scene_gt.ply
```

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
poetry run ruff check src/
poetry run pyright src/
```

### Tests

```bash
poetry run pytest
```

### Notebooks

Launch JupyterLab after installing the `notebook` group:

```bash
poetry run jupyter lab
```
