
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module
contains various task implementations for the G1 robot, such as pick and place, motion control, etc.
"""

# use relative import
from . import pick_place_cylinder_g1_29dof_inspire

from . import stack_rgyblock_g1_29dof_inspire
from . import pick_place_redblock_g1_29dof_inspire
from . import pick_place_redblock_g1_29dof_inspire_farlean
from . import pick_place_redblock_g1_29dof_inspire_side
from . import pick_place_bigbox_g1_29dof_inspire_rightleft
from . import move_cylinder_g1_29dof_inspire_wholebody

# export all modules
__all__ = [
        "stack_rgyblock_g1_29dof_inspire",
        "pick_place_redblock_g1_29dof_inspire",
        "pick_place_redblock_g1_29dof_inspire_farlean",
        "pick_place_redblock_g1_29dof_inspire_side",
        "pick_place_bigbox_g1_29dof_inspire_rightleft",
        "pick_place_cylinder_g1_29dof_inspire",
        "move_cylinder_g1_29dof_inspire_wholebody"
]
