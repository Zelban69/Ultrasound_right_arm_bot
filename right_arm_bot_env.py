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

from .right_arm_bot_env_cfg import RightArmBotEnvCfg


class RightArmBotEnv(DirectRLEnv):
    """Environment for the RightArmBot manipulation task."""
    
    cfg: RightArmBotEnvCfg

    def __init__(self, cfg: RightArmBotEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        # initialize the base class
        super().__init__(cfg, render_mode, **kwargs)

        # --- FIX: Auto-update Config Dimensions to match Robot ---
        # This prevents the 'ValueError: mismatch' crash by accepting whatever the robot provides.
        action_dim = gym.spaces.flatdim(self.single_action_space)
        if self.cfg.action_space != action_dim:
            print(f"[INFO] Action space mismatch. Updating config from {self.cfg.action_space} to {action_dim}")
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
        """Actions applied to the robot before physics step."""
        self._actions = actions.clone()
        # process actions: apply action scale and add to default joint positions
        # Ensure we broadcast correctly if robot has more joints than actions
        targets = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
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
        
        # Concatenate observations
        obs = torch.cat([joint_pos_rel, joint_vel], dim=-1)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards from the environment."""
        ee_pos = self._robot.data.body_pos_w[:, -1, :] 
        
        distance = torch.norm(self.targets - ee_pos, dim=-1)
        
        reward_reaching = -1.0 * distance * self.cfg.reaching_goal_reward_scale
        reward_action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1) * self.cfg.action_rate_reward_scale
        reward_effort = torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * self.cfg.joint_effort_reward_scale
        reward_terminated = self.reset_terminated.float() * self.cfg.terminated_reward_scale

        rewards = {
            "reaching_goal_reward_scale": reward_reaching,
            "action_rate_reward_scale": reward_action_rate,
            "joint_effort_reward_scale": reward_effort,
            "terminated_reward_scale": reward_terminated,
        }
        
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
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
        
        # Randomize targets
        self.targets[env_ids] = torch.rand((len(env_ids), 3), device=self.device) * 1.0 - 0.5
        self.targets[env_ids, 2] = torch.rand((len(env_ids)), device=self.device) * 0.5 + 0.1

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