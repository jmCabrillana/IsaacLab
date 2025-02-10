# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaac.lab_assets import HUMANOID_CFG

import isaac.lab.sim as sim_utils
from isaac.lab.assets import ArticulationCfg
from isaac.lab.envs import DirectRLEnvCfg, ViewerCfg
from isaac.lab.scene import InteractiveSceneCfg
from isaac.lab.sim import SimulationCfg, PhysxCfg
from isaac.lab.terrains import TerrainImporterCfg
from isaac.lab.utils import configclass
from isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file


from isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class HumanoidCamEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    num_actions = 21
    num_observations = 75
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, 
                                       physx=PhysxCfg(gpu_collision_stack_size=2**26)) # 2**26
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8 # -0.1 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # camera
    tiled_camera: TiledCameraCfg =  TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/head/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.3, 0.0, 2.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=448,
        height=256,
    )


class HumanoidCamEnv(LocomotionEnv):
    cfg: HumanoidCamEnvCfg

    def __init__(self, cfg: HumanoidCamEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
