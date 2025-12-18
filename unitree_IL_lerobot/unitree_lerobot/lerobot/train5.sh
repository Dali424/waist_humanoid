#!/bin/bash
set -euo pipefail

REPO_IDS=("Dali424/sidepickcube_img_head_wrist")
OUT_DIR="outputs/cascaded_act_interleave_encoder_first"

for REPO in "${REPO_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --policy.type=cascaded_act_interleave \
    --output_dir="${OUT_DIR}" \
    --steps=300000
done
