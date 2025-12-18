#!/bin/bash
set -euo pipefail

REPO_IDS=("Dali424/sidepickcube_img_head_wrist")
OUT_DIR="outputs/cascaded_act_arm_first_norm_waist"

for REPO in "${REPO_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --policy.type=cascaded_act \
    --policy.arm_first=true \
    --policy.norm_waist_tokens=true \
    --output_dir="${OUT_DIR}" \
    --steps=300000
done
