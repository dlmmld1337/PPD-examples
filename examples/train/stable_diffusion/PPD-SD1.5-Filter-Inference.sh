#!/usr/bin/env bash
set -euo pipefail

INPUT_IMAGE="${1:-dog.jpg}"
LORA_CHECKPOINT_PATH="${2:-./models/train/SD1.5-PPD-filter-lora/last.ckpt}"
OUTPUT_IMAGE="${3:-output_sd15_filter.png}"
RADIUS="${4:-1200}"
DEVICE="${5:-cuda}"
INFERENCE_STEPS="${6:-50}"

PYTHONPATH=. python examples/image_synthesis/sd_text_to_image_ppd.py \
  --lora_checkpoint_path "$LORA_CHECKPOINT_PATH" \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_target_modules "to_q,to_k,to_v,to_out" \
  --input_image "$INPUT_IMAGE" \
  --prompt "make this image photorealistic" \
  --radius "$RADIUS" \
  --device "$DEVICE" \
  --num_inference_steps "$INFERENCE_STEPS" \
  --output_name "$OUTPUT_IMAGE"