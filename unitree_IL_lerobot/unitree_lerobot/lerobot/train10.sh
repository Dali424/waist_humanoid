#!/bin/bash
set -euo pipefail

REPO_IDS=("Dali424/sidepickcube_img_head_wrist")

# Sequentially train act_hier and act_hier_dual

for REPO in "${REPO_IDS[@]}"; do
  # act_hier
  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --policy.type=act_hier \
    --output_dir="outputs/act_hier" \
    --steps=300000

  # act_hier_dual
  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --policy.type=act_hier_dual \
    --output_dir="outputs/act_hier_dual" \
    --steps=300000
done
