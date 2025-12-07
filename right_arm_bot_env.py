# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply
from .right_arm_bot_env_cfg import RightArmBotEnvCfg


class RightArmBotEnv(DirectRLEnv):
    """Environment for the RightArmBot manipulation task."""
    
    cfg: RightArmBotEnvCfg

    def __init__(self, cfg: RightArmBotEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        # initialize the base class
        super().__init__(cfg, render_mode, **kwargs)
        #print("MY JOINT ORDER:", self._robot.data.joint_names)
        # --- FIX: Auto-update Config Dimensions to match Robot ---
        action_dim = gym.spaces.flatdim(self.single_action_space)
        if self.cfg.action_space != action_dim:
            #print(f"[INFO] Action space mismatch. Updating config from {self.cfg.action_space} to {action_dim}")
            self.cfg.action_space = action_dim

        obs_dim = gym.spaces.flatdim(self.single_observation_space)
        if self.cfg.observation_space != obs_dim:
            print(f"[INFO] Observation space mismatch. Updating config from {self.cfg.observation_space} to {obs_dim}")
            self.cfg.observation_space = obs_dim
            # Also update state_space to match observation_space if it was set to something else
            self.cfg.state_space = obs_dim 

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        # Initialize targets
        self.targets = torch.zeros((self.num_envs, 3), device=self.device)

        # create reward logging dictionary
        self._episode_sums = {
            "reaching_goal_reward_scale": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "action_rate_reward_scale": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "joint_effort_reward_scale": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "terminated_reward_scale": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

    """
    Implementation of abstract methods
    """

    def _setup_scene(self):
        """Setup the scene."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        
        # --- 1. Calculate Standard Targets ---
        targets = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        # --- 2. CLAMP JOINT_09_10 ---
        PROBE_JOINT_IDX = 8
        
        # Define your allowed range in Radians
        # Example: Allow it to wiggle slightly between 80 and 100 degrees
        MIN_ANGLE = -1.0472  # ~50 degrees
        MAX_ANGLE = -1.74533  # ~100 degrees
        
        # Clamp the value so it never exceeds these limits
        targets[:, PROBE_JOINT_IDX] = torch.clamp(targets[:, PROBE_JOINT_IDX], min=MIN_ANGLE, max=MAX_ANGLE)

        # --- 3. APPLY ---
        self._processed_actions = targets

    def _apply_action(self):
        """Apply actions to the robot."""
        self._robot.set_joint_position_target(self._processed_actions)
        self._robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        self._previous_actions = self._actions.clone()

        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        
        ee_pos = self._robot.data.body_pos_w[:, -1, :]
        # Calculate vector to target (Goal Position - Current Position)
        target_vec = self.targets - ee_pos
        # Concatenate observations
        obs = torch.cat([joint_pos_rel, joint_vel, target_vec], dim=-1)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards from the environment."""
        
        # --- 1. REACHING REWARD ---
        ee_pos = self._robot.data.body_pos_w[:, -1, :] 
        distance = torch.norm(self.targets - ee_pos, dim=-1)
        
        # Sharper tolerance for scanning (0.15)
        reward_reaching = torch.exp(-distance / 0.15) * self.cfg.reaching_goal_reward_scale
        
        # --- 2. ORIENTATION REWARD ---
        # Get the quaternion orientation of the end-effector
        ee_quat = self._robot.data.body_quat_w[:, -1, :]
        
        # Define the "Forward" direction of your probe in LOCAL space (X-axis)
        probe_vec_local = torch.zeros((self.num_envs, 3), device=self.device)
        probe_vec_local[:, 1] = -1.0 
        
        probe_vec_world = quat_apply(ee_quat, probe_vec_local)
        
        # Define the target "Down" vector in WORLD space (0, 0, -1)
        target_vec_world = torch.zeros((self.num_envs, 3), device=self.device)
        target_vec_world[:, 2] = -1.0 
        
        # Calculate Cosine Similarity (Dot Product)
        dot_prod = torch.sum(probe_vec_world * target_vec_world, dim=-1)
        
        reward_orientation = dot_prod * self.cfg.orientation_reward_scale

        # --- 3. PENALTIES ---
        reward_action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1) * self.cfg.action_rate_reward_scale

        torques = self._robot.data.applied_torque
        max_torque = 1000.0 
        norm_torques = torques / max_torque
        reward_effort = torch.sum(torch.square(norm_torques), dim=1) * self.cfg.joint_effort_reward_scale
        
        reward_terminated = self.reset_terminated.float() * self.cfg.terminated_reward_scale

        # --- LOGGING ---
        rewards = {
            "reaching_goal_reward_scale": reward_reaching,
            "orientation_reward_scale": reward_orientation,
            "action_rate_reward_scale": reward_action_rate,
            "joint_effort_reward_scale": reward_effort,
            "terminated_reward_scale": reward_terminated,
        }
        
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Log episode sums
        for key, value in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[key] += value
            
        return total_reward

    def _get_dones(self) -> tuple:
        """Get dones from the environment."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the environment for the given env_ids."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # Reset buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # --- MODIFIED: ABDOMINAL SCAN TASK ---
        # We define a "Patient Bed" relative to the robot base.
        # X axis: Depth (how far in front of the robot)
        # Y axis: Width (The horizontal scan direction)
        # Z axis: Height (The height of the patient's belly)

        num_resets = len(env_ids)
        
        # Fixed Depth (X): The patient is 0.2 meters in front of the robot
        # We add tiny noise (+- 5cm) so it doesn't overfit to exactly 0.5000m
        self.targets[env_ids, 0] = 0.2 + (torch.rand(num_resets, device=self.device) * 0.1 - 0.05)
        
        # Variable Width (Y): The Scan Line. 
        # Randomly sample anywhere from -0.2m (left) to +0.2m (right)
        self.targets[env_ids, 1] = (torch.rand(num_resets, device=self.device) * 0.6) - 0.4
        
        # Fixed Height (Z): The patient is lying down at 0.15m height
        self.targets[env_ids, 2] = 0.15
        # -------------------------------------

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)