"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import os
import torch
import logging

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.policies.factory import make_policy
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from multiprocessing.sharedctypes import SynchronizedArray

from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
)
from unitree_lerobot.eval_robot.utils.sim_savedata_utils import (
    EvalRealConfig,
    process_data_add,
    is_success,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def eval_policy(
    cfg: EvalRealConfig,
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    policy.reset()  # Set policy to evaluation mode
    # Attention hook setup
    attn_cache = {}
    attention_dir = None
    cam_keys: list[str] = []
    tokens_per_cam = None
    non_image_tokens = 1  # latent token
    cam_grids: dict[str, tuple[int, int]] = {}
    cam_dirs: dict[str, str] = {}
    if cfg.save_attention:
        attention_dir = os.path.join(cfg.task_dir, getattr(cfg, "attention_dir", "attention"))
        os.makedirs(attention_dir, exist_ok=True)

        def make_hook(name):
            def hook(module, inp, output):
                # MultiheadAttention returns (attn_out, attn_weights)
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_cache[name] = output[1].detach().cpu()
            return hook

        for name, module in policy.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(make_hook(name))
        # Derive camera token layout (all cameras assumed same shape/stride).
        try:
            cam_keys = list(policy.config.image_features.keys())
            stride = 16 if getattr(policy.config, "replace_final_stride_with_dilation", False) else 32
            if cam_keys:
                for cam in cam_keys:
                    c, h, w = policy.config.input_features[cam].shape
                    gh, gw = h // stride, w // stride
                    cam_grids[cam] = (gh, gw)
                    cam_dir = os.path.join(attention_dir, cam.replace(".", "-"))
                    os.makedirs(cam_dir, exist_ok=True)
                    cam_dirs[cam] = cam_dir
                tokens_per_cam = next(iter(cam_grids.values()))
                tokens_per_cam = tokens_per_cam[0] * tokens_per_cam[1]
                if policy.config.robot_state_feature:
                    non_image_tokens += 1
                if policy.config.env_state_feature:
                    non_image_tokens += 1
        except Exception as e:
            logger_mp.warning(f"Failed to derive camera token layout for attention viz: {e}")

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        (
            arm_ctrl,
            arm_ik,
            ee_shared_mem,
            arm_dof,
            ee_dof,
            sim_state_subscriber,
            sim_reward_subscriber,
            episode_writer,
            reset_pose_publisher,
            waist_dof,
        ) = (
            robot_interface[key]
            for key in [
                "arm_ctrl",
                "arm_ik",
                "ee_shared_mem",
                "arm_dof",
                "ee_dof",
                "sim_state_subscriber",
                "sim_reward_subscriber",
                "episode_writer",
                "reset_pose_publisher",
                "waist_dof",
            ]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        # Get initial pose from the first step of the dataset
        from_idx = dataset.episode_data_index["from"][0].item()
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        # Auto-detect waist usage from dataset shape vs controller DOF
        try:
            ds_state_dim = int(step["observation.state"].shape[0])
            ee_total = 2 * ee_dof if cfg.ee else 0
            expected_no_waist = int(arm_dof + ee_total)
            expected_with_waist = int(expected_no_waist + waist_dof)
            if not cfg.use_waist and waist_dof > 0 and ds_state_dim == expected_with_waist:
                cfg.use_waist = True
                if hasattr(arm_ctrl, "enable_waist_control"):
                    arm_ctrl.enable_waist_control(True)
                logger_mp.info(
                    f"Auto-enabled waist: dataset_dim={ds_state_dim}, expected_with_waist={expected_with_waist}"
                )
        except Exception:
            pass

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None

        reward_stats = {
            "reward_sum": 0.0,
            "episode_num": 0.0,
        }

        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move

            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                if cfg.save_data:
                    if reward_stats["episode_num"] == 0:
                        episode_writer.create_episode()
                loop_start_time = time.perf_counter()

                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array,
                    wrist_img_array,
                    tv_img_shape,
                    wrist_img_shape,
                    is_binocular,
                    has_wrist_cam,
                    arm_ctrl,
                )
                # Ensure all image features expected by the policy are present.
                # If some camera keys are missing (e.g., cam_right_high in mono-head sim),
                # backfill them from the corresponding color_i keys to match training.
                try:
                    image_features = getattr(policy.config, "image_features", [])
                except Exception:
                    image_features = []
                if image_features:
                    cam_fallback_map = {
                        # cams -> color_i mapping consistent with dataset recordings:
                        # color_0 = head, color_1 = left wrist, color_2 = right wrist, color_3 = world
                        "observation.images.head": "observation.images.color_0",
                        "observation.images.wrist_left": "observation.images.color_1",
                        "observation.images.wrist_right": "observation.images.color_2",
                        "observation.images.left_wrist": "observation.images.color_1",
                        "observation.images.right_wrist": "observation.images.color_2",
                        "observation.images.world": "observation.images.color_3",
                        # legacy cam_* names (older configs)
                        "observation.images.cam_left_high": "observation.images.color_0",
                        "observation.images.cam_right_high": "observation.images.color_3",
                        "observation.images.cam_left_wrist": "observation.images.color_1",
                        "observation.images.cam_right_wrist": "observation.images.color_2",
                    }
                    for feat_key in image_features:
                        if feat_key in observation:
                            continue
                        src_key = cam_fallback_map.get(feat_key)
                        if src_key and src_key in observation:
                            observation[feat_key] = observation[src_key]
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                left_ee_state = full_state[:ee_dof]
                right_ee_state = full_state[ee_dof:]
                parts = [current_arm_q, left_ee_state, right_ee_state]
                if cfg.use_waist and hasattr(arm_ctrl, "get_current_waist_q") and waist_dof > 0:
                    current_waist_q_raw = arm_ctrl.get_current_waist_q()
                    # Map controller order [yaw, roll, pitch] -> policy order [yaw, pitch, roll]
                    if waist_dof == 3 and len(current_waist_q_raw) == 3:
                        current_waist_q = np.array([
                            current_waist_q_raw[0],  # yaw stays
                            current_waist_q_raw[2],  # pitch <- from index 2
                            current_waist_q_raw[1],  # roll  <- from index 1
                        ])
                    else:
                        current_waist_q = current_waist_q_raw
                    parts.append(current_waist_q)
                state_tensor = torch.from_numpy(np.concatenate(parts, axis=0)).float()
                observation["observation.state"] = state_tensor
                # 2. Get Action from Policy
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                )
                action_np = action.cpu().numpy()
                # 3. Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                # Optional waist action at the tail
                if cfg.use_waist and waist_dof > 0:
                    ee_action_total = 2 * ee_dof if cfg.ee else 0
                    waist_start = arm_dof + ee_action_total
                    waist_action_raw = action_np[waist_start : waist_start + waist_dof]
                    # Map policy waist order [yaw, pitch, roll] -> controller order [yaw, roll, pitch]
                    if waist_dof == 3 and len(waist_action_raw) == 3:
                        waist_action = np.array([
                            waist_action_raw[0],  # yaw stays
                            waist_action_raw[2],  # pitch -> roll index
                            waist_action_raw[1],  # roll  -> pitch index
                        ])
                    else:
                        waist_action = waist_action_raw
                    if hasattr(arm_ctrl, "ctrl_arm_and_waist"):
                        arm_ctrl.ctrl_arm_and_waist(arm_action, tau, waist_action)
                    else:
                        arm_ctrl.ctrl_dual_arm(arm_action, tau)
                else:
                    arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)
                # save data
                if cfg.save_data:
                    # Save data (extend with waist if present)
                    current_waist_q = (
                        arm_ctrl.get_current_waist_q() if cfg.use_waist and waist_dof > 0 and hasattr(arm_ctrl, "get_current_waist_q") else None
                    )
                    process_data_add(
                        episode_writer,
                        observation,
                        current_arm_q,
                        full_state,
                        action,
                        arm_dof,
                        ee_dof,
                        waist_dof if cfg.use_waist else 0,
                        current_waist_q,
                    )

                    is_success(
                        sim_reward_subscriber,
                        episode_writer,
                        reset_pose_publisher,
                        policy,
                        cfg,
                        reward_stats,
                        init_arm_pose,
                        robot_interface,
                    )

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                if (
                    cfg.save_attention
                    and attn_cache
                    and cam_keys
                    and tokens_per_cam
                    and idx % max(1, cfg.attention_interval) == 0
                ):
                    # Prefer decoder->encoder cross-attn map
                    key = next((k for k in attn_cache.keys() if "multihead_attn" in k), next(iter(attn_cache)))
                    attn = attn_cache[key]
                    if attn.dim() == 3:
                        # PyTorch MultiheadAttention returns (tgt, batch, src)
                        attn_map = attn[:, 0, :]
                    else:
                        attn_map = attn  # assume (tgt, src)
                    attn_vec = attn_map.mean(dim=0)  # average over decoder queries -> (src,)
                    for cam_idx, cam in enumerate(cam_keys):
                        start = non_image_tokens + cam_idx * tokens_per_cam
                        end = start + tokens_per_cam
                        if end > attn_vec.numel():
                            continue
                        gh, gw = cam_grids.get(cam, (None, None))
                        if gh is None or gw is None:
                            continue
                        cam_attn = attn_vec[start:end].reshape(gh, gw)
                        cam_attn = cam_attn - cam_attn.min()
                        if cam_attn.max() > 0:
                            cam_attn = cam_attn / cam_attn.max()
                        img_t = observation.get(cam)
                        if img_t is None:
                            continue
                        if img_t.ndim == 3 and img_t.shape[0] in (1, 3, 4):
                            img_np = img_t.permute(1, 2, 0).cpu().numpy()
                        else:
                            img_np = img_t.cpu().numpy()
                        if img_np.dtype != np.uint8:
                            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                        scale_y = max(1, img_np.shape[0] // gh)
                        scale_x = max(1, img_np.shape[1] // gw)
                        up = np.kron(cam_attn, np.ones((scale_y, scale_x)))
                        up = up[: img_np.shape[0], : img_np.shape[1]]
                        plt.figure(figsize=(4, 3))
                        plt.imshow(img_np)
                        plt.imshow(up, cmap="jet", alpha=0.4)
                        plt.axis("off")
                        ep_num_raw = reward_stats.get("episode_num", 0)
                        if isinstance(ep_num_raw, (int, float)):
                            ep_num_str = f"{int(ep_num_raw):04d}"
                        else:
                            ep_num_str = str(ep_num_raw)
                        fname = os.path.join(
                            cam_dirs.get(cam, attention_dir),
                            f"ep{ep_num_str}_step{idx}.png",
                        )
                        plt.tight_layout()
                        plt.savefig(fname, dpi=150)
                        plt.close()
                    attn_cache.clear()
                idx += 1
                try:
                    reward_stats["episode_num"] = reward_stats["episode_num"] + 1
                except Exception:
                    reward_stats["episode_num"] = 0
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if image_info:
            cleanup_resources(image_info)
        # Clean up sim state subscriber if it exists
        if "sim_state_subscriber" in locals() and sim_state_subscriber:
            sim_state_subscriber.stop_subscribe()
            logger_mp.info("SimStateSubscriber cleaned up")
        if "sim_reward_subscriber" in locals() and sim_reward_subscriber:
            sim_reward_subscriber.stop_subscribe()
            logger_mp.info("SimRewardSubscriber cleaned up")


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg=cfg, policy=policy, dataset=dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
