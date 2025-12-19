#!/bin/bash
set -euo pipefail

REPO_IDS=("Dali424/sidepickcube_img_head_wrist")

for REPO in "${REPO_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --policy.type=act_hier_dual \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.n_action_steps=1 \
    --output_dir="outputs/act_hier_dual_ensemble" \
    --steps=300000
done
