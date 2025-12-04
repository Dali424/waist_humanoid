# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym

from . import pickplace_redblock_g1_29dof_inspire_farlean_env_cfg


gym.register(
    id="Isaac-PickPlace-RedBlock-G129-Inspire-Joint-FarSpawn",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_redblock_g1_29dof_inspire_farlean_env_cfg.PickPlaceG129InspireFarSpawnEnvCfg,
    },
    disable_env_checker=True,
)

