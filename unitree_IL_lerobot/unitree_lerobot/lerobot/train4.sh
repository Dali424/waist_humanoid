#!/bin/bash
set -euo pipefail

REPO_IDS=("Dali424/sidepickcube_img_head_wrist")

# Train act_hier_hand
for REPO in "${REPO_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --policy.type=act_hier_hand \
    --output_dir="outputs/act_hier_hand" \
    --steps=300000
done
