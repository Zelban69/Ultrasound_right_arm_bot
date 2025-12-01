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

from.right_arm_bot_env_cfg import RightArmBotEnvCfg


class RightArmBotEnv(DirectRLEnv):
    """Environment for the RightArmBot manipulation task."""
    
    cfg: RightArmBotEnvCfg

    def __init__(self, cfg: RightArmBotEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        # initialize the base class
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        if self.cfg.action_space!= action_dim:
            raise ValueError(
                f"Action space mismatch: cfg has {self.cfg.action_space}, but env has {action_dim}"
            )

        obs_dim = gym.spaces.flatdim(self.single_observation_space)
        if self.cfg.observation_space!= obs_dim:
            raise ValueError(
                f"Observation space mismatch: cfg has {self.cfg.observation_space}, but env has {obs_dim}"
            )

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
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
        # spawn the robot
        self._robot = Articulation(self.cfg.robot)
        # add robot to the scene
        self.scene.articulations["robot"] = self._robot

        # spawn the terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Actions applied to the robot before physics step."""
        self._actions = actions.clone()
        # process actions: apply action scale and add to default joint positions
        # this is for position control, which matches the ImplicitActuatorCfg
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        """Apply actions to the robot."""
        # set joint position targets
        self._robot.set_joint_position_target(self._processed_actions)
        # write data to sim
        self._robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        # store previous actions
        self._previous_actions = self._actions.clone()

        # compute observations:
        # 1. relative joint positions (current - default)
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        # 2. joint velocities
        joint_vel = self._robot.data.joint_vel
        
        # concatenate observations
        # This matches the observation_space = 20 (10 pos + 10 vel)
        obs = torch.cat([joint_pos_rel, joint_vel], dim=-1)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards from the environment."""
        # Calculate distance from robot end-effector to target
        # Assuming the end-effector is the last body in the articulation
        ee_pos = self._robot.data.body_pos_w[:, -1, :] # [num_envs, 3]
        
        # Distance to target
        distance = torch.norm(self.targets - ee_pos, dim=-1)
        
        # Reward: Negative distance (closer is better)
        reward_reaching = -1.0 * distance * self.cfg.reaching_goal_reward_scale
        # (Placeholder for a reaching task)        
        # Penalize large action rates (to encourage smoothness)
        reward_action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1) * self.cfg.action_rate_reward_scale

        # Penalize joint effort
        reward_effort = torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * self.cfg.joint_effort_reward_scale
        
        # Penalty for termination
        reward_terminated = self.reset_terminated.float() * self.cfg.terminated_reward_scale

        # ---
        # Log rewards
        # ---
        rewards = {
            "reaching_goal_reward_scale": reward_reaching,
            "action_rate_reward_scale": reward_action_rate,
            "joint_effort_reward_scale": reward_effort,
            "terminated_reward_scale": reward_terminated,
        }
        
        # sum all rewards
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # log episode sums
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return total_reward

    def _get_dones(self) -> tuple:
        """Get dones from the environment."""
        # time-out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # (Placeholder for other terminations)
        # e.g., check for collisions if you add a contact sensor
        terminated = torch.zeros_like(time_out)

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the environment for the given env_ids."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        # reset the robot
        self._robot.reset(env_ids)
        # reset the base class
        super()._reset_idx(env_ids)
        
        # spread out resets
        #if len(env_ids) == self.num_envs:
        #    self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # reset action buffers
        #self._actions[env_ids] = 0.0
        #self._previous_actions[env_ids] = 0.0
        
        #Generate random targets within reach of the arm (e.g., 0.5m radius)
        if env_ids is None: env_ids = self._robot._ALL_INDICES
    
    # Create a buffer for targets if it doesn't exist
        if not hasattr(self, "targets"):
            self.targets = torch.zeros((self.num_envs, 3), device=self.device)
            
        # Randomize targets for resetting envs
        self.targets[env_ids] = torch.rand((len(env_ids), 3), device=self.device) * 1.0 - 0.5
        self.targets[env_ids, 2] = torch.rand((len(env_ids)), device=self.device) * 0.5 + 0.1 # Z height
        # ---
        # reset robot state
        # ---
        # get default joint states
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # get default root state
        default_root_state = self._robot.data.default_root_state[env_ids]
        # add terrain origin
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # write all default states to sim
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ---
        # logging
        # ---
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])

            extras[key] = episodic_sum_avg / self.max_episode_length_s

            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras_term = dict() 
        extras_term["time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras_term)
