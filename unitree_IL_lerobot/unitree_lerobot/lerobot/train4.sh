#!/bin/bash
set -e

REPO_IDS=(
  "Dali424/sidepickcube_img_head_wrist"
)

for REPO in "${REPO_IDS[@]}"; do
  echo "============================================"
  echo "Training CACT policy on dataset (GPU3): ${REPO}"
  echo "============================================"

  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --policy.type=cascaded_act \
    --output_dir=outputs/train/cact \
    --steps=500000
done
