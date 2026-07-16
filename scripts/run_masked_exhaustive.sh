#!/bin/bash
# Phase-2 rear-coverage fix run (2026-07-15): identical to the "masked" benchmark
# variant except feature_matching.method=exhaustive (configs/colmap_exhaustive.yaml).
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/sfm-mvs-pipeline"
cd /mnt/c/Users/Felip/Documentos/heAICare/sfm-mvs-pipeline || exit 1

SESSION=video_test_20260714_213712
ARUCO=/mnt/c/Users/Felip/Documentos/heAICare/aruco-frame-preprocessing/data/frames/$SESSION
OUT=data/processed/${SESSION}_masked_exhaustive
LOG=data/processed/pipeline_${SESSION}_masked_exhaustive.log

mkdir -p data/processed
uv run python scripts/run_pipeline.py \
  --image-dir "$ARUCO/filtered" \
  --output-dir "$OUT" \
  --frames-manifest "$ARUCO/manifest.json" \
  --colmap-config configs/colmap_exhaustive.yaml \
  >> "$LOG" 2>&1
echo "PIPELINE_EXIT_CODE=$?" >> "$LOG"
