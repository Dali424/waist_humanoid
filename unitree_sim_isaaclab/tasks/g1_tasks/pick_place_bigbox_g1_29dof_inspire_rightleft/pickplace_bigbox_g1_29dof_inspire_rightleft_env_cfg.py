# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
import os
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils

from . import mdp
from tasks.common_config import G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_redblock import TableRedBlockSceneCfg

project_root = os.environ.get("PROJECT_ROOT")

# Big box settings (body-sized so both arms are needed)
BOX_SIZE = (0.21, 0.154, 0.175)
BOX_MASS = 0.30
TABLE_TOP_Z = 0.81

# Base object's default position in the base scene
OBJ_BASE_X, OBJ_BASE_Y = (-4.25, -4.03)
OBJ_BASE_Z = TABLE_TOP_Z + 0.5 * BOX_SIZE[2]

# Spawn range (relative offset from object's default pose)
RIGHT_SPAWN_CENTER_OFFSET_X = -0.35
RIGHT_SPAWN_CENTER_OFFSET_Y = -0.15
SPAWN_HALF_RANGE = 0.05  # about 10 cm total randomization
SPAWN_X_RANGE = (
    RIGHT_SPAWN_CENTER_OFFSET_X - SPAWN_HALF_RANGE,
    RIGHT_SPAWN_CENTER_OFFSET_X + SPAWN_HALF_RANGE,
)
SPAWN_Y_RANGE = (
    RIGHT_SPAWN_CENTER_OFFSET_Y - SPAWN_HALF_RANGE,
    RIGHT_SPAWN_CENTER_OFFSET_Y + SPAWN_HALF_RANGE,
)

# Precompute spawn area visualization transform and size (module-level constants)
SPAWN_CENTER_X = OBJ_BASE_X + 0.5 * (SPAWN_X_RANGE[0] + SPAWN_X_RANGE[1])
SPAWN_CENTER_Y = OBJ_BASE_Y + 0.5 * (SPAWN_Y_RANGE[0] + SPAWN_Y_RANGE[1])
SPAWN_CENTER_Z = OBJ_BASE_Z - 0.02  # slightly above table top
# Ensure strictly positive spawn area extents to avoid degenerate transforms.
SPAWN_SIZE_X = max(1e-3, SPAWN_X_RANGE[1] - SPAWN_X_RANGE[0])
SPAWN_SIZE_Y = max(1e-3, SPAWN_Y_RANGE[1] - SPAWN_Y_RANGE[0])
SPAWN_SIZE_Z = 0.01

# Toggle for visualizing the spawn range on the table.
SHOW_SPAWN_AREA = False

# Target (success) area visualization for left table area
LEFT_TARGET_CENTER_OFFSET_X = 0.35
LEFT_TARGET_CENTER_OFFSET_Y = -0.15
TARGET_HALF_X = 0.12
TARGET_HALF_Y = 0.12
TARGET_MIN_X = OBJ_BASE_X + LEFT_TARGET_CENTER_OFFSET_X - TARGET_HALF_X
TARGET_MAX_X = OBJ_BASE_X + LEFT_TARGET_CENTER_OFFSET_X + TARGET_HALF_X
TARGET_MIN_Y = OBJ_BASE_Y + LEFT_TARGET_CENTER_OFFSET_Y - TARGET_HALF_Y
TARGET_MAX_Y = OBJ_BASE_Y + LEFT_TARGET_CENTER_OFFSET_Y + TARGET_HALF_Y
TARGET_MIN_Z = OBJ_BASE_Z - 0.05
TARGET_MAX_Z = OBJ_BASE_Z + 0.05
TARGET_CENTER = (
    0.5 * (TARGET_MIN_X + TARGET_MAX_X),
    0.5 * (TARGET_MIN_Y + TARGET_MAX_Y),
    0.5 * (TARGET_MIN_Z + TARGET_MAX_Z),
)
TARGET_SIZE = (
    max(1e-3, TARGET_MAX_X - TARGET_MIN_X),
    max(1e-3, TARGET_MAX_Y - TARGET_MIN_Y),
    max(1e-3, TARGET_MAX_Z - TARGET_MIN_Z),
)
SHOW_TARGET_AREA = False


@configclass
class ObjectTableSceneCfg(TableRedBlockSceneCfg):
    """Scene: G1 + big red box + table with spawn/target visualization."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix(
        init_pos=(-4.2, -3.7, 0.76), init_rot=(0.7071, 0, 0, -0.7071)
    )

    # Cameras
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera()

    # Override table USD for this task
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.3, -4.2, -0.2], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_leftyellowbox.usd",
        ),
    )

    # Override object with a larger box
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[OBJ_BASE_X, OBJ_BASE_Y, OBJ_BASE_Z],
            rot=[0.9659, 0.0, 0.0, -0.2588],
        ),
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.69, 0.55, 0.34),
                metallic=0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )

    # Optional spawn-area visualization (only created when SHOW_SPAWN_AREA is True)
    if SHOW_SPAWN_AREA:
        spawn_area: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/envs/env_.*/SpawnArea",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[SPAWN_CENTER_X, SPAWN_CENTER_Y, SPAWN_CENTER_Z],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=sim_utils.CuboidCfg(
                size=(SPAWN_SIZE_X, SPAWN_SIZE_Y, SPAWN_SIZE_Z),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), opacity=1.0),
            ),
        )
    # Optional target-area visualization (helps to see success zone)
    if SHOW_TARGET_AREA:
        target_area: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/envs/env_.*/TargetArea",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=list(TARGET_CENTER),
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=sim_utils.CuboidCfg(
                size=TARGET_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.6, 1.0), opacity=0.4),
            ),
        )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.reset_object_estimate)


@configclass
class RewardsCfg:
    reward = RewTerm(
        func=mdp.compute_reward,
        weight=1.0,
        params={
            "post_min_x": TARGET_MIN_X,
            "post_max_x": TARGET_MAX_X,
            "post_min_y": TARGET_MIN_Y,
            "post_max_y": TARGET_MAX_Y,
            "post_min_height": TARGET_MIN_Z,
            "post_max_height": TARGET_MAX_Z,
        },
    )


@configclass
class EventCfg:
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": list(SPAWN_X_RANGE), "y": list(SPAWN_Y_RANGE)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceG129InspireRightLeftBigBoxEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 2
        self.sim.physx.contact_offset = 0.015
        self.sim.physx.rest_offset = 0.001
        self.sim.physx.num_position_iterations = 12
        self.sim.physx.num_velocity_iterations = 4

        self.event_manager = SimpleEventManager()
        # local reset with right-spawn ranges
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_root_state_uniform(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                    pose_range={"x": list(SPAWN_X_RANGE), "y": list(SPAWN_Y_RANGE)},
                    velocity_range={},
                    asset_cfg=SceneEntityCfg("object"),
                )
            ),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_scene_to_default(env, torch.arange(env.num_envs, device=env.device))
            ),
        )
