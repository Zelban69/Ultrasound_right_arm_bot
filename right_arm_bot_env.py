from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers

from .right_arm_bot_env_cfg import RightArmBotEnvCfg


class RightArmBotEnv(DirectRLEnv):
    """Position-reaching environment for RightArmBot.

    Changes from your current version:
    - keeps the current no-patient / no-ellipsoid setup
    - removes duplicated EE-body/default-EE-local setup
    - samples targets relative to the current default EE pose
    - forces target Z-offset to stay below the probe
    - clamps target local Z above the ground plane
    """

    cfg: RightArmBotEnvCfg

    def __init__(self, cfg: RightArmBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._goal_marker = VisualizationMarkers(self.cfg.goal_marker)

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = self._robot.data.default_joint_pos.clone()
        self.targets = torch.zeros((self.num_envs, 3), device=self.device)

        self._base_joint_idx = self._resolve_single_joint(self.cfg.base_joint_name)
        self._probe_joint_idx = self._resolve_single_joint(self.cfg.probe_joint_name)
        self._ee_body_idx = self._resolve_single_body(self.cfg.ee_body_name)

        self._target_offset_min = torch.tensor(self.cfg.target_offset_min, dtype=torch.float32, device=self.device)
        self._target_offset_max = torch.tensor(self.cfg.target_offset_max, dtype=torch.float32, device=self.device)
        self._target_min_z_local = float(self.cfg.target_min_z_local)

        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Cache the arm's default EE pose in each env's local frame.
        self._default_ee_local = (
            self._robot.data.body_pos_w[:, self._ee_body_idx, :] - self.scene.env_origins
        ).clone()

        if self.cfg.debug_print_asset_names:
            print("[RightArmBotEnv] num_envs:", self.num_envs)
            print("[RightArmBotEnv] joint_names:", self._robot.data.joint_names)
            print("[RightArmBotEnv] body_names:", self._robot.data.body_names)
            print(
                "[RightArmBotEnv] selected EE body:",
                self._robot.data.body_names[self._ee_body_idx],
                f"(index={self._ee_body_idx})",
            )
            print("[RightArmBotEnv] is_fixed_base:", self._robot.is_fixed_base)
            print(
                "[RightArmBotEnv] default_joint_pos[0]:",
                self._robot.data.default_joint_pos[0].detach().cpu().tolist(),
            )
            print(
                "[RightArmBotEnv] default_ee_local[0]:",
                self._default_ee_local[0].detach().cpu().tolist(),
            )

        self._episode_sums = {
            "reward_reaching": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "reward_success": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "penalty_action_rate": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "penalty_joint_effort": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "penalty_joint_deviation": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "penalty_terminated": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "success_count": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "invalid_count": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

    def _resolve_single_joint(self, joint_name: str) -> int:
        joint_ids, joint_names = self._robot.find_joints(joint_name)
        if len(joint_ids) != 1:
            raise ValueError(
                f"Expected exactly one joint for pattern {joint_name!r}, got {len(joint_ids)}: {joint_names}. "
                f"Available joints: {self._robot.data.joint_names}"
            )
        return int(joint_ids[0])

    def _resolve_single_body(self, body_name: str) -> int:
        body_ids, body_names = self._robot.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body for pattern {body_name!r}, got {len(body_ids)}: {body_names}. "
                f"Available bodies: {self._robot.data.body_names}"
            )
        return int(body_ids[0])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions.copy_(self._actions)

        if self.cfg.hold_position_steps > 0:
            hold_mask = (self.episode_length_buf < self.cfg.hold_position_steps).unsqueeze(-1)
            actions = torch.where(hold_mask, torch.zeros_like(actions), actions)

        self._actions = torch.clamp(actions.clone(), -1.0, 1.0)

        # Absolute position targets around default joint pose.
        targets = self._robot.data.default_joint_pos + self.cfg.action_scale * self._actions

        if self.cfg.use_soft_joint_limits:
            soft_limits = self._robot.data.soft_joint_pos_limits
            valid_limits = torch.isfinite(soft_limits).all(dim=-1) & (soft_limits[..., 1] >= soft_limits[..., 0])

            safe_lower = torch.where(valid_limits, soft_limits[..., 0], torch.full_like(soft_limits[..., 0], -1.0e6))
            safe_upper = torch.where(valid_limits, soft_limits[..., 1], torch.full_like(soft_limits[..., 1], 1.0e6))
            targets = torch.clamp(targets, safe_lower, safe_upper)

        targets[:, self._base_joint_idx] = torch.clamp(
            targets[:, self._base_joint_idx],
            min=self.cfg.base_joint_min,
            max=self.cfg.base_joint_max,
        )
        targets[:, self._probe_joint_idx] = torch.clamp(
            targets[:, self._probe_joint_idx],
            min=self.cfg.probe_joint_min,
            max=self.cfg.probe_joint_max,
        )

        self._processed_actions = targets

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        self._goal_marker.visualize(self.targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx, :]
        target_vec = self.targets - ee_pos

        obs = torch.cat((joint_pos_rel, joint_vel, target_vec), dim=-1)
        return {"policy": obs, "critic": obs}

    def _compute_success_and_distance(self, ee_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.linalg.norm(self.targets - ee_pos, dim=-1)
        success = (distance <= self.cfg.success_tolerance) & (
            self.episode_length_buf >= self.cfg.success_after_n_steps
        )
        return success, distance

    def _compute_invalid_state(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        body_pos = self._robot.data.body_pos_w

        joint_pos_bad = torch.isnan(joint_pos).any(dim=1) | torch.isinf(joint_pos).any(dim=1)
        joint_vel_bad = torch.isnan(joint_vel).any(dim=1) | torch.isinf(joint_vel).any(dim=1)
        body_pos_bad = torch.isnan(body_pos).any(dim=(1, 2)) | torch.isinf(body_pos).any(dim=(1, 2))

        return joint_pos_bad | joint_vel_bad | body_pos_bad

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx, :]
        success, distance = self._compute_success_and_distance(ee_pos)
        invalid_state = self._compute_invalid_state()

        reward_reaching = torch.exp(-distance / self.cfg.reaching_sigma) * self.cfg.reaching_goal_reward_scale
        reward_success = success.float() * self.cfg.success_reward_scale

        action_delta = self._actions - self._previous_actions
        penalty_action_rate = torch.sum(action_delta.square(), dim=1) * self.cfg.action_rate_reward_scale
        penalty_action_rate = torch.clamp(penalty_action_rate, min=self.cfg.action_rate_penalty_cap)

        applied_torque = self._robot.data.applied_torque
        normalized_torque = applied_torque / self.cfg.effort_normalization
        penalty_joint_effort = torch.sum(normalized_torque.square(), dim=1) * self.cfg.joint_effort_reward_scale
        penalty_joint_effort = torch.clamp(penalty_joint_effort, min=self.cfg.joint_effort_penalty_cap)

        joint_deviation = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        penalty_joint_deviation = torch.sum(joint_deviation.square(), dim=1) * self.cfg.joint_deviation_reward_scale

        penalty_terminated = invalid_state.float() * self.cfg.terminated_reward_scale

        total_reward = (
            reward_reaching
            + reward_success
            + penalty_action_rate
            + penalty_joint_effort
            + penalty_joint_deviation
            + penalty_terminated
        )

        self._episode_sums["reward_reaching"] += reward_reaching
        self._episode_sums["reward_success"] += reward_success
        self._episode_sums["penalty_action_rate"] += penalty_action_rate
        self._episode_sums["penalty_joint_effort"] += penalty_joint_effort
        self._episode_sums["penalty_joint_deviation"] += penalty_joint_deviation
        self._episode_sums["penalty_terminated"] += penalty_terminated
        self._episode_sums["success_count"] += success.float()
        self._episode_sums["invalid_count"] += invalid_state.float()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx, :]
        success, _ = self._compute_success_and_distance(ee_pos)
        invalid_state = self._compute_invalid_state()

        self._last_success.copy_(success)
        self._last_invalid.copy_(invalid_state)

        terminated = invalid_state | (success if self.cfg.terminate_on_success else torch.zeros_like(success))
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        if self.cfg.reset_joint_noise > 0.0:
            joint_pos += (2.0 * torch.rand_like(joint_pos) - 1.0) * self.cfg.reset_joint_noise

        if not self._robot.is_fixed_base:
            default_root_state = self._robot.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] += self.scene.env_origins[env_ids]
            self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._processed_actions[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        self._sample_targets(env_ids)
        self._goal_marker.visualize(self.targets)

        extras = {}
        reward_keys = (
            "reward_reaching",
            "reward_success",
            "penalty_action_rate",
            "penalty_joint_effort",
            "penalty_joint_deviation",
            "penalty_terminated",
        )
        for key in reward_keys:
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        extras["success_rate"] = torch.mean(self._episode_sums["success_count"][env_ids])
        extras["invalid_rate"] = torch.mean(self._episode_sums["invalid_count"][env_ids])
        self._episode_sums["success_count"][env_ids] = 0.0
        self._episode_sums["invalid_count"][env_ids] = 0.0

        self.extras["log"] = extras

    def _sample_targets(self, env_ids: Sequence[int]):
        num_resets = len(env_ids)

        # Sample around the default EE pose, but always BELOW the probe.
        rand = torch.rand((num_resets, 3), device=self.device)
        offsets = self._target_offset_min + rand * (self._target_offset_max - self._target_offset_min)

        # Safety guard: never allow the sampled offset to go above the probe.
        offsets[:, 2] = torch.minimum(offsets[:, 2], torch.full_like(offsets[:, 2], -1.0e-4))

        local_targets = self._default_ee_local[env_ids] + offsets

        # Keep the target above the ground plane a bit.
        local_targets[:, 2] = torch.clamp(local_targets[:, 2], min=self._target_min_z_local)

        self.targets[env_ids] = local_targets + self.scene.env_origins[env_ids]
