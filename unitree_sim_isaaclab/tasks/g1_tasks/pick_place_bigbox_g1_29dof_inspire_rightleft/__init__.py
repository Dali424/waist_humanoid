# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import pickplace_bigbox_g1_29dof_inspire_rightleft_env_cfg


gym.register(
    id="Isaac-PickPlace-BigBox-G129-Inspire-Joint-RightLeft",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            pickplace_bigbox_g1_29dof_inspire_rightleft_env_cfg.PickPlaceG129InspireRightLeftBigBoxEnvCfg
        ),
    },
    disable_env_checker=True,
)
