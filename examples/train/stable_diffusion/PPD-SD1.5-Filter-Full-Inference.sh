#!/usr/bin/env bash
set -euo pipefail

INPUT_IMAGE="${1:-dog.jpg}"
DEFAULT_OUTPUT_ROOT="./models/train/SD1.5-PPD-filter-full"
CHECKPOINT_PATH="${2:-}"
OUTPUT_IMAGE="${3:-output_sd15_filter_full.png}"
RADIUS="${4:-1200}"
DEVICE="${5:-cuda}"
INFERENCE_STEPS="${6:-50}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  latest_version_dir="$(find "$DEFAULT_OUTPUT_ROOT/lightning_logs" -maxdepth 1 -type d -name 'version_*' 2>/dev/null | sort -V | tail -n 1)"
  if [[ -n "$latest_version_dir" && -f "$latest_version_dir/checkpoints/last.ckpt" ]]; then
    CHECKPOINT_PATH="$latest_version_dir/checkpoints/last.ckpt"
  else
    CHECKPOINT_PATH="$DEFAULT_OUTPUT_ROOT/lightning_logs/version_0/checkpoints/last.ckpt"
  fi
fi

PYTHONPATH=. python examples/image_synthesis/sd_text_to_image_ppd.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --input_image "$INPUT_IMAGE" \
  --prompt "make this image photorealistic" \
  --radius "$RADIUS" \
  --device "$DEVICE" \
  --num_inference_steps "$INFERENCE_STEPS" \
  --output_name "$OUTPUT_IMAGE"