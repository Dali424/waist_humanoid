#!/bin/bash
set -e

REPO_IDS=(
  "Dali424/sidepickcube_img_head_wrist"
)

for REPO in "${REPO_IDS[@]}"; do

  CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/train.py \
    --dataset.repo_id="${REPO}" \
    --policy.push_to_hub=false \
    --policy.type=cascaded_act \
    --output_dir=outputs/decoder_layer_7/cact \
    --steps=300000 \
    --dataset.video_backend=pyav
done
