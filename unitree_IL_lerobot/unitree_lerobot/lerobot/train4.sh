#!/bin/bash
set -e

REPO_IDS=(
  "Dali424/sidepickcube_img_head_wrist_world"
)

for REPO in "${REPO_IDS[@]}"; do
  echo "============================================"
  echo "Training ACT policy on dataset (GPU3): ${REPO}"
  echo "============================================"

  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --policy.type=act \
    --output_dir=outputs/train/wandb_head_wrist_world \
    --steps=500000
done
