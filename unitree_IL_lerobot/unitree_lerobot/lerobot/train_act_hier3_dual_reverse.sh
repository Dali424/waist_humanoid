#!/bin/bash
set -euo pipefail

REPO_IDS=("Dali424/sidepickcube_img_head_wrist")

# Train act_hier3_dual_reverse (dual cross-attn, hand -> arm -> waist) once on GPU 7.
CUDA_VISIBLE_DEVICES=7 python src/lerobot/scripts/train.py \
  --dataset.repo_id="${REPO_IDS[0]}" \
  --policy.push_to_hub=false \
  --policy.type=act_hier3_dual_reverse \
  --output_dir="outputs/act_hier3_dual_reverse" \
  --steps=300000
