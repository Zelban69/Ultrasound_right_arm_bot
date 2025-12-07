# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg  # Import for terrain
from isaaclab.utils import configclass

# --- 1. IMPORT YOUR ROBOT ---
# Import your custom robot configuration using its absolute path
from right_arm_bot.robots.right_arm_bot import RIGHT_ARM_BOT_CFG


@configclass
class RightArmBotEnvCfg(DirectRLEnvCfg):
    """Configuration for the RightArmBot environment."""

    # --- 2. ENVIRONMENT-SPECIFIC PARAMETERS ---
    # env
    episode_length_s = 20.0  # Length of each episode in seconds
    decimation = 4           # Number of sim steps per env step
    action_scale = 1.0       # (Optional) Scale actions from agent
    
    # --- 3. ACTION & OBSERVATION SPACES (CRITICAL) ---
    # Your robot has 10 joints
    action_space = 10
    # A common observation space is (joint_pos (10), joint_vel (10))
    # This will depend on what you define in your env.py file.
    observation_space = 23
    state_space = 20

    # --- 4. SIMULATION SETTINGS ---
    # (Applied from your config)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- 5. SCENE DEFINITION ---
    # (Applied from your config)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  
        env_spacing=2.5,
        replicate_physics=True
    )

    # --- 6. ROBOT ASSIGNMENT ---
    robot: ArticulationCfg = RIGHT_ARM_BOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # --- 7. REWARD SCALES ---
    # These are placeholders. You must implement the logic for these
    # rewards in your 'right_arm_bot_env.py' file.
    
    # rewarding proximity to a goal
    reaching_goal_reward_scale: float = 10.0
    
    # ACTION RATE: Small penalty
    action_rate_reward_scale: float = -0.1
    
    # JOINT EFFORT: 
    # Since we normalized the torque to 0-1, we can use a normal number here.
    # -0.5 means "Using 100% power on all joints costs -0.5 points per step"
    # This is much safer than -1e-9.
    joint_effort_reward_scale: float = -0.01
    
    # TERMINATION: Strong penalty for crashing
    terminated_reward_scale: float = -10.0
    
    # ORIENTATION: Reward for pointing straight down
    # Start with 1.0. If it still tilts, increase to 2.0 or 5.0.
    orientation_reward_scale: float = 1.0