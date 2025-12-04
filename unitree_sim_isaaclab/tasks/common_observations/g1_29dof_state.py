# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
g1_29dof state
"""     
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import os

import sys
from multiprocessing import shared_memory

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch

def get_robot_boy_joint_names() -> list[str]:
    return [
        # leg joints (12)
        # left leg (6)
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        # right leg (6)
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        # waist joints (3)
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",

        # arm joints (14)
        # left arm (7)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        # right arm (7)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

def get_robot_arm_joint_names() -> list[str]:
    return [
        # arm joints (14)
        # left arm (7)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        # right arm (7)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

# global variable to cache the DDS instance
from dds.dds_master import dds_manager
_g1_robot_dds = None
_dds_initialized = False

# 观测缓存：索引张量与DDS限速（50FPS）+ 预分配缓冲
_obs_cache = {
    "device": None,
    "batch": None,
    "boy_idx_t": None,
    "boy_idx_batch": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "combined_buf": None,
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}

# IMU 加速度缓存：用于通过速度差分计算加速度
# IMU acceleration cache: for computing acceleration via velocity differentiation
_imu_acc_cache = {
    "prev_vel": None,
    "dt": 0.01,
}

def _get_g1_robot_dds_instance():
    """get the DDS instance, delay initialization"""
    global _g1_robot_dds, _dds_initialized
    
    if not _dds_initialized or _g1_robot_dds is None:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            print(f"dds_manager: {dds_manager}")
            _g1_robot_dds = dds_manager.get_object("g129")
            print("[g1_state] G1 robot DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _g1_robot_dds:
                        dds_manager.unregister_object("g129")
                        print("[g1_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[g1_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[g1_state] Failed to get G1 robot DDS instance: {e}")
            _g1_robot_dds = None
        
        _dds_initialized = True
    
    return _g1_robot_dds

def get_robot_boy_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """get the robot body joint states, positions and velocities
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function
    
    Returns:
        torch.Tensor
        - the first 29 elements are joint positions
        - the middle 29 elements are joint velocities
        - the last 29 elements are joint torques
    """
    # get all joint states
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque  # use applied_torque to get joint torques
    device = joint_pos.device
    batch = joint_pos.shape[0]

    # 预计算并缓存索引张量（列索引）
    global _obs_cache
    if _obs_cache["device"] != device or _obs_cache["boy_idx_t"] is None:
        boy_joint_indices = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
        _obs_cache["boy_idx_t"] = torch.tensor(boy_joint_indices, dtype=torch.long, device=device)
        _obs_cache["device"] = device
        _obs_cache["batch"] = None  # force re-init batch-shaped buffers

    idx_t = _obs_cache["boy_idx_t"]
    n = idx_t.numel()

    # 预分配/复用 batch 形状索引与输出缓冲
    if _obs_cache["batch"] != batch or _obs_cache["boy_idx_batch"] is None:
        _obs_cache["boy_idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["combined_buf"] = torch.empty(batch, n * 3, device=device, dtype=joint_pos.dtype)
        _obs_cache["batch"] = batch

    idx_batch = _obs_cache["boy_idx_batch"]
    pos_buf = _obs_cache["pos_buf"]
    vel_buf = _obs_cache["vel_buf"]
    torque_buf = _obs_cache["torque_buf"]
    combined_buf = _obs_cache["combined_buf"]

    # 使用 gather(out=...) 填充，避免新张量分配
    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))

    # 组合为一个缓冲，避免 cat 分配
    combined_buf[:, 0:n].copy_(pos_buf)
    combined_buf[:, n:2*n].copy_(vel_buf)
    combined_buf[:, 2*n:3*n].copy_(torque_buf)

    # write to DDS（限速发布，避免高频CPU拷贝）
    if enable_dds and combined_buf.shape[0] > 0:
        try:
            import time
            now_ms = int(time.time() * 1000)
            if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
                g1_robot_dds = _get_g1_robot_dds_instance()
                if g1_robot_dds:
                    imu_data = get_robot_imu_data(env)
                    if imu_data.shape[0] > 0:
                        g1_robot_dds.write_robot_state(
                            pos_buf[0].contiguous().cpu().numpy(),
                            vel_buf[0].contiguous().cpu().numpy(),
                            torque_buf[0].contiguous().cpu().numpy(),
                            imu_data[0].contiguous().cpu().numpy(),
                        )
                        _obs_cache["dds_last_ms"] = now_ms
        except Exception as e:
            print(f"[g1_state] Error writing robot state to DDS: {e}")
    
    return combined_buf


def quat_conjugate(q):
    # q: [batch, 4] with order (w, x, y, z)
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    return torch.cat([w, -xyz], dim=1)

def quat_rotate_vec(q, v):
    # Rotate vector v (batch,3) by quaternion q (w,x,y,z)
    # returns rotated vector (batch,3)
    # Using: v' = q * (0, v) * q_conj
    # q * r: quaternion multiplication
    w = q[:, 0:1]
    x = q[:, 1:2]
    y = q[:, 2:3]
    z = q[:, 3:4]

    # compute q * (0, v)
    vx = v[:, 0:1]
    vy = v[:, 1:2]
    vz = v[:, 2:3]

    # quaternion multiply q * r where r = (0, v)
    rw_w = - (x*vx + y*vy + z*vz)
    rw_x =   w*vx + y*vz - z*vy
    rw_y =   w*vy + z*vx - x*vz
    rw_z =   w*vz + x*vy - y*vx

    # multiply result by q_conj: (rw) * q_conj
    # q_conj = (w, -x, -y, -z)
    out_x = rw_w * (-x) + rw_x * w + rw_y * (-z) - rw_z * (-y)
    out_y = rw_w * (-y) + rw_y * w + rw_z * (-x) - rw_x * (-z)
    out_z = rw_w * (-z) + rw_z * w + rw_x * (-y) - rw_y * (-x)

    return torch.cat([out_x, out_y, out_z], dim=1)

def get_robot_imu_data(env, use_torso_imu: bool = True) -> torch.Tensor:
    """Corrected IMU: returns position (world), quaternion (w,x,y,z), accelerometer (in IMU/body frame), gyro (in body frame)
    """
    data = env.scene["robot"].data
    global _imu_acc_cache

    # --- dt ---
    dt = _imu_acc_cache["dt"]
    try:
        if hasattr(env, 'physics_dt'):
            dt = float(env.physics_dt)
        elif hasattr(env, 'step_dt'):
            dt = float(env.step_dt)
        elif hasattr(env, 'dt'):
            dt = float(env.dt)
    except Exception:
        pass
    if dt <= 0:
        dt = _imu_acc_cache["dt"]

    # --- extract pose & vel ---
    if use_torso_imu:
        try:
            body_names = data.body_names
            imu_torso_idx = body_names.index("imu_in_torso")
            body_pose = data.body_link_pose_w  # [batch, num_links, 7]
            body_vel = data.body_link_vel_w    # [batch, num_links, 6]
            pos = body_pose[:, imu_torso_idx, :3]
            quat = body_pose[:, imu_torso_idx, 3:7]  
            lin_vel = body_vel[:, imu_torso_idx, :3]
            ang_vel = body_vel[:, imu_torso_idx, 3:6]
        except ValueError:
            use_torso_imu = False

    if not use_torso_imu:
        root_state = data.root_state_w  # [batch, 13]
        pos = root_state[:, :3]
        quat = root_state[:, 3:7]      
        lin_vel = root_state[:, 7:10]
        ang_vel = root_state[:, 10:13]

    # ensure tensors on same device
    device = lin_vel.device if isinstance(lin_vel, torch.Tensor) else torch.device('cpu')
    if _imu_acc_cache["prev_vel"] is None:
        _imu_acc_cache["prev_vel"] = torch.zeros_like(lin_vel).to(device)
    else:
        if _imu_acc_cache["prev_vel"].device != device:
            _imu_acc_cache["prev_vel"] = _imu_acc_cache["prev_vel"].to(device)

    # --- compute a_world = dv/dt ---
    a_world = (lin_vel - _imu_acc_cache["prev_vel"]) / dt

    # gravity in world frame; assume world z-up and gravity vector points down
    g_world = torch.zeros_like(a_world)
    g_world[:, 2] = -9.81  # [0,0,-9.81]

    # proper acceleration in world frame (subtract gravity)
    a_world_corrected = a_world - g_world  # = a_world - g

    # rotate to IMU/body frame: a_body = R^T * a_world_corrected
    # which is equivalent to rotating the vector by quaternion conjugate
    # assumes quat format (w,x,y,z)
    a_body = quat_rotate_vec(quat, a_world_corrected)  # since quat_rotate_vec(q, v) computes q * (0,v) * q_conj
    # Note: the above rotates v from world->body only if quat represents rotation from body->world.
    # If quat is world->body, skip conjugation - check API's quaternion convention.

    # On first frame (no meaningful dv/dt), _imu_acc_cache["prev_vel"] was zeros so a_world large; optionally handle first-frame:
    # You may prefer to set a_body to -R^T(g_world) when prev_vel uninitialized:
    # But here we already used zeros prev_vel; if you want first-frame stable:
    # if torch.allclose(_imu_acc_cache["prev_vel"], torch.zeros_like(_imu_acc_cache["prev_vel"])):
    #     a_body = quat_rotate_vec(quat, -g_world)

    # Update cache
    _imu_acc_cache["prev_vel"] = lin_vel.clone().detach()
    _imu_acc_cache["dt"] = dt

    imu_data = torch.cat([pos, quat, a_body, ang_vel], dim=1)
    return imu_data
