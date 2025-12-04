# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import sys
import os
import threading
import queue

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from image_server.shared_memory_utils import MultiImageWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
multi_image_writer = MultiImageWriter()

def set_writer_options(enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
    try:
        multi_image_writer.set_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=skip_cvtcolor)
        print(f"[camera_state] writer options: jpeg={enable_jpeg}, quality={jpeg_quality}, skip_cvtcolor={skip_cvtcolor}")
    except Exception as e:
        print(f"[camera_state] failed to set writer options: {e}")


_camera_cache = {
    'available_cameras': None,
    'camera_keys': None,
    'last_scene_id': None,
    'frame_step': 0,
    'write_interval_steps': 2,
}


_return_placeholder = None
_async_queue = None
_async_thread = None
_async_started = False

def _async_writer_loop(q: "queue.Queue", writer: MultiImageWriter):
    while True:
        try:
            item = q.get()
            if item is None:
                break
            writer.write_images(item)
        except Exception as e:
            print(f"[camera_state] Async writer error: {e}")

def _ensure_async_started():
    global _async_started, _async_queue, _async_thread
    if not _async_started:
        _async_queue = queue.Queue(maxsize=1)
        _async_thread = threading.Thread(target=_async_writer_loop, args=(_async_queue, multi_image_writer), daemon=True)
        _async_thread.start()
        _async_started = True


def get_camera_image(
    env: ManagerBasedRLEnv,
) -> dict:
    # pass
    """get multiple camera images and write them to shared memory
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        dict: dictionary containing multiple camera images
    """
    global _return_placeholder
    if _return_placeholder is None:
        _return_placeholder = torch.zeros((1, 480, 640, 3))


    _camera_cache['frame_step'] = (_camera_cache['frame_step'] + 1) % max(1, _camera_cache['write_interval_steps'])


    scene_id = id(env.scene)
    if _camera_cache['last_scene_id'] != scene_id:
        _camera_cache['camera_keys'] = list(env.scene.keys())
        _camera_cache['available_cameras'] = [name for name in _camera_cache['camera_keys'] if "camera" in name.lower()]
        _camera_cache['last_scene_id'] = scene_id


    if _camera_cache['frame_step'] == 0:
        try:
            dt = getattr(env, 'physics_dt', 0.02)
            if hasattr(env.scene, 'sensors') and env.scene.sensors:
                for sensor in env.scene.sensors.values():
                    try:
                        sensor.update(dt, force_recompute=False)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # get the camera images
    images = {}
    # env.sim.render()
    

    camera_keys = _camera_cache['camera_keys']
    # Head image: keep robot viewpoint for teleop (front camera).
    # Prefer front camera; fall back to world camera if needed.
    head_image = None
    if "front_camera" in camera_keys:
        head_image = env.scene["front_camera"].data.output["rgb"][0]
    elif "world_camera" in camera_keys:
        head_image = env.scene["world_camera"].data.output["rgb"][0]

    # Wrist cameras: left and right.
    left_wrist_image = None
    right_wrist_image = None
    if "left_wrist_camera" in camera_keys:
        left_wrist_image = env.scene["left_wrist_camera"].data.output["rgb"][0]
    if "right_wrist_camera" in camera_keys:
        right_wrist_image = env.scene["right_wrist_camera"].data.output["rgb"][0]

    # Optional world (perspective) view.
    persp_image = None
    if "world_camera" in camera_keys:
        persp_image = env.scene["world_camera"].data.output["rgb"][0]

    def _to_numpy(tensor):
        return tensor.numpy() if tensor.device.type == "cpu" else tensor.cpu().numpy()

    # Head slot.
    if head_image is not None:
        images["head"] = _to_numpy(head_image)

    # Left wrist slot: prefer real left wrist, fall back to perspective if needed.
    if left_wrist_image is not None:
        images["left"] = _to_numpy(left_wrist_image)
    elif persp_image is not None:
        images["left"] = _to_numpy(persp_image)

    # Right wrist slot: prefer real right wrist, fall back to left/perspective.
    if right_wrist_image is not None:
        images["right"] = _to_numpy(right_wrist_image)
    elif "left" in images:
        images["right"] = images["left"]
    elif persp_image is not None:
        images["right"] = _to_numpy(persp_image)
    # World camera slot: if available, expose explicitly for downstream consumers.
    if persp_image is not None:
        images["world"] = _to_numpy(persp_image)
    
    # if no camera with the specified name is found, try other common camera names
    if not images:

        available_cameras = _camera_cache['available_cameras']
        if available_cameras:
            print(f"[camera_state] No standard cameras found. Available cameras: {available_cameras}")
            
            # if there are available cameras, use the first three as head, left, right
            for i, camera_name in enumerate(available_cameras[:3]):
                camera_image = env.scene[camera_name].data.output["rgb"][0]
                
               
                if camera_image.device.type == 'cpu':
                    numpy_image = camera_image.numpy()
                else:
                    numpy_image = camera_image.cpu().numpy()
                
                if i == 0:
                    images["head"] = numpy_image
                elif i == 1:
                    images["left"] = numpy_image
                elif i == 2:
                    images["right"] = numpy_image
    

    if images and _camera_cache['frame_step'] == 0:
        _ensure_async_started()
        try:
            
            if _async_queue.full():
                _async_queue.get_nowait()
            _async_queue.put_nowait(images)
        except Exception:
            pass
    elif not images:
        print("[camera_state] No camera images found in the environment")
    
    return _return_placeholder
