#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"
PYTHONPATH=. "$PYTHON_BIN" examples/train/stable_diffusion/train_sd15_lora_ppd_hf_img2img.py \
  --pretrained_path "models/stable_diffusion/v1-5-pruned-emaonly.safetensors" \
  --dataset_name "dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora" \
  --cache_dir "/code/dataset/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora" \
  --image_column "edited_image" \
  --cond_image_column "input_image" \
  --instance_prompt "make this image photorealistic" \
  --height 512 \
  --width 512 \
  --batch_size 4 \
  --accumulate_grad_batches 1 \
  --learning_rate 1e-4 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --max_steps 10000 \
  --checkpointing_steps 1000 \
  --sample_radius_exponential \
  --min_structural_noise_radius 100 \
  --structural_noise_lambda 0.1 \
  --validation_steps 250 \
  --num_validation_images 4 \
  --validation_inference_steps 30 \
  --precision 16-mixed \
  --output_path "./models/train/SD1.5-PPD-filter-lora"