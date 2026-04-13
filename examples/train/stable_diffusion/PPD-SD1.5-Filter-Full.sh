#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 PYTHONPATH=. "$PYTHON_BIN" examples/train/stable_diffusion/train_sd15_lora_ppd_hf_img2img.py \
  --pretrained_path "models/stable_diffusion/v1-5-pruned-emaonly.safetensors" \
  --dataset_name "dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora" \
  --cache_dir "/code/dataset/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora" \
  --image_column "edited_image" \
  --cond_image_column "input_image" \
  --instance_prompt "make this image photorealistic" \
  --height 512 \
  --width 512 \
  --batch_size 1 \
  --accumulate_grad_batches 8 \
  --learning_rate 1e-5 \
  --training_mode full \
  --max_epochs 100 \
  --max_steps -1 \
  --checkpointing_steps 0 \
  --checkpointing_epochs 10 \
  --sample_radius_exponential \
  --min_structural_noise_radius 100 \
  --structural_noise_lambda 0.1 \
  --validation_steps 0 \
  --validation_epochs 1 \
  --num_validation_images 30 \
  --validation_inference_steps 30 \
  --progress_log_steps 10 \
  --precision 32 \
  --use_gradient_checkpointing \
  --output_path "./models/train/SD1.5-PPD-filter-full"