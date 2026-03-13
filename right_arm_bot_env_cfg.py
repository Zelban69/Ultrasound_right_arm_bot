from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from right_arm_bot.robots.right_arm_bot import RIGHT_ARM_BOT_CFG


@configclass
class RightArmBotEnvCfg(DirectRLEnvCfg):
    """Reaching config with targets spawned below the probe."""

    # Timing
    decimation = 4
    episode_length_s = 20.0

    # Spaces
    action_space = 10
    observation_space = 23
    state_space = 23

    # Debug
    debug_print_asset_names = True

    # Control
    hold_position_steps = 0
    success_after_n_steps = 10
    terminate_on_success = True

    # Keep this False unless the parsed soft limits are known to be sane.
    use_soft_joint_limits = False

    # Action processing
    action_scale = 0.12
    reset_joint_noise = 0.0

    # Robot references
    base_joint_name = "Joint_00_02"
    probe_joint_name = "Joint_09_10"
    ee_body_name = "Link_11"

    # Manual clamps
    base_joint_min = -0.5
    base_joint_max = 0.5
    probe_joint_min = -1.74533
    probe_joint_max = -1.0472

    # Target offsets relative to the default end-effector local position.
    # IMPORTANT: Z is always negative, so the target always spawns below the probe.
    target_offset_min = (-0.04, -0.04, -0.18)
    target_offset_max = (0.04, 0.04, -0.10)

    # Prevent targets from spawning too close to the floor.
    target_min_z_local = 0.02

    # Success
    success_tolerance = 0.01

    # Rewards / penalties
    reaching_goal_reward_scale = 15.0
    reaching_sigma = 0.06
    success_reward_scale = 25.0

    action_rate_reward_scale = -0.05
    action_rate_penalty_cap = -1.0

    joint_effort_reward_scale = -0.005
    joint_effort_penalty_cap = -2.0
    effort_normalization = 200.0

    joint_deviation_reward_scale = -0.10
    terminated_reward_scale = -10.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

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

    goal_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetMarker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=False,
    )

    robot: ArticulationCfg = RIGHT_ARM_BOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
