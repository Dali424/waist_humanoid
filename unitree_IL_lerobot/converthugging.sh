#!/bin/bash
set -e

RAW_DIR=./datasets
ROBOT_TYPE=Unitree_G1_Inspire_Waist



python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id Dali424/sidepickcube_img_head_rightwrist_world \
  --robot_type "${ROBOT_TYPE}" \
  --camera-names head right_wrist world \
  --raw-camera-keys color_0 color_2 color_3 \
  --push-to-hub


python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id Dali424/sidepickcube_img_head_rightwrist \
  --robot_type "${ROBOT_TYPE}" \
  --camera-names head right_wrist \
  --raw-camera-keys color_0 color_2 \
  --push-to-hub


python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id Dali424/sidepickcube_img_rightwrist_world \
  --robot_type "${ROBOT_TYPE}" \
  --camera-names right_wrist world \
  --raw-camera-keys color_2 color_3 \
  --push-to-hub


python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id Dali424/sidepickcube_img_rightwrist \
  --robot_type "${ROBOT_TYPE}" \
  --camera-names right_wrist \
  --raw-camera-keys color_2 \
  --push-to-hub

