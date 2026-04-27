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

Install dependencies (CPU environment, e.g. development machine without GPU):

```bash
poetry install
```

Install dependencies with GPU support (e.g. machine with NVIDIA GPU and CUDA 12):

```bash
poetry install --with gpu
```

> **Note:** The `gpu` group replaces the CPU-only `pycolmap` with `pycolmap-cuda12`, which enables GPU-accelerated feature extraction, matching, and MVS. The CUDA Toolkit must be installed separately on Linux.