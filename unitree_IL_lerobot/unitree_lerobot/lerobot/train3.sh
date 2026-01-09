#!/bin/bash
set -euo pipefail

REPO_IDS=("Dali424/sidepickcube_img_head_wrist")

# Train act_hier_dual_hand on fixed GPUs 4,5,6,7 (one run per device).
CUDA_VISIBLE_DEVICES=4 python src/lerobot/scripts/train.py \
  --dataset.repo_id="${REPO_IDS[0]}" \
  --policy.push_to_hub=false \
  --policy.type=act_hier_dual_hand \
  --output_dir="outputs/act_hier_dual_hand_gpu4" \
  --steps=300000

CUDA_VISIBLE_DEVICES=5 python src/lerobot/scripts/train.py \
  --dataset.repo_id="${REPO_IDS[0]}" \
  --policy.push_to_hub=false \
  --policy.type=act_hier_dual_hand \
  --output_dir="outputs/act_hier_dual_hand_gpu5" \
  --steps=300000

CUDA_VISIBLE_DEVICES=6 python src/lerobot/scripts/train.py \
  --dataset.repo_id="${REPO_IDS[0]}" \
  --policy.push_to_hub=false \
  --policy.type=act_hier_dual_hand \
  --output_dir="outputs/act_hier_dual_hand_gpu6" \
  --steps=300000

CUDA_VISIBLE_DEVICES=7 python src/lerobot/scripts/train.py \
  --dataset.repo_id="${REPO_IDS[0]}" \
  --policy.push_to_hub=false \
  --policy.type=act_hier_dual_hand \
  --output_dir="outputs/act_hier_dual_hand_gpu7" \
  --steps=300000
