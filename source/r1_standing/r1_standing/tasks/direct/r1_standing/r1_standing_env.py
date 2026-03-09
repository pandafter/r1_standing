# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Environment for the Unitree R1 humanoid robot - Push Recovery with Stepping."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .r1_standing_env_cfg import R1StandingEnvCfg


class R1StandingEnv(DirectRLEnv):
    cfg: R1StandingEnvCfg

    def __init__(self, cfg: R1StandingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices
        self._joint_ids, _ = self.robot.find_joints(self.cfg.joint_names)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Previous actions
        self._previous_actions = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device
        )

        # Push system
        self.push_interval = int(
            self.cfg.push_interval_s / (self.cfg.sim.dt * self.cfg.decimation)
        )
        self._push_timer = torch.zeros(self.num_envs, device=self.device)

        # Root body
        self._root_body_id = 0

        # Feet bodies
        self._left_foot_id, _ = self.robot.find_bodies(".*left.*ankle.*")
        self._right_foot_id, _ = self.robot.find_bodies(".*right.*ankle.*")

        # ---- Dynamic joint detection (robust to URDF joint ordering) ---- #
        self._knee_joint_ids, _ = self.robot.find_joints(".*knee.*")
        self._hip_left_ids, _ = self.robot.find_joints(".*left.*hip.*")
        self._hip_right_ids, _ = self.robot.find_joints(".*right.*hip.*")
        self._left_leg_ids, _ = self.robot.find_joints(
            [".*left.*hip.*", ".*left.*knee.*", ".*left.*ankle.*"]
        )
        self._right_leg_ids, _ = self.robot.find_joints(
            [".*right.*hip.*", ".*right.*knee.*", ".*right.*ankle.*"]
        )

    # --------------------------------------------------------------------- #
    # Scene
    # --------------------------------------------------------------------- #

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/GroundPlane", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------------------- #
    # Push Logic
    # --------------------------------------------------------------------- #

    def _apply_random_pushes(self):
        env_ids = (self._push_timer >= self.push_interval).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        # Fuerzas aleatorias
        forces = torch.zeros((len(env_ids), 1, 3), device=self.device)  # 1 body: root
        torques = torch.zeros((len(env_ids), 1, 3), device=self.device)

        # Magnitud de fuerza aleatoria
        forces[:, 0, 0] = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.push_force_max - self.cfg.push_force_min
        ) + self.cfg.push_force_min
        forces[:, 0, 1] = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.push_force_max - self.cfg.push_force_min
        ) + self.cfg.push_force_min

        # Direcciones aleatorias (+/-)
        signs = torch.randint(0, 2, (len(env_ids), 2), device=self.device) * 2 - 1
        forces[:, 0, :2] *= signs

        # Aplicar al root
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=[self._root_body_id],
            env_ids=env_ids
        )

        # Escribir datos a simulador
        self.robot.write_data_to_sim()

        # Reiniciar timer
        self._push_timer[env_ids] = 0

    # --------------------------------------------------------------------- #
    # RL Pipeline
    # --------------------------------------------------------------------- #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._push_timer += 1
        self._apply_random_pushes()

    def _apply_action(self) -> None:
        target_positions = (
            self.robot.data.default_joint_pos
            + self.actions * self.cfg.action_scale
        )
        # Clamping de posiciones articulares a soft limits
        target_positions = torch.clamp(
            target_positions,
            self.robot.data.soft_joint_pos_limits[:, :, 0],
            self.robot.data.soft_joint_pos_limits[:, :, 1],
        )
        self.robot.set_joint_position_target(
            target_positions, joint_ids=self._joint_ids
        )
        self._previous_actions = self.actions.clone()

    # --------------------------------------------------------------------- #
    # Observations
    # --------------------------------------------------------------------- #

    def _get_observations(self) -> dict:
        # Distancia entre pies
        left_foot_pos = self.robot.data.body_pos_w[:, self._left_foot_id[0]]
        right_foot_pos = self.robot.data.body_pos_w[:, self._right_foot_id[0]]
        feet_distance = torch.norm(left_foot_pos[:, :2] - right_foot_pos[:, :2], dim=-1, keepdim=True)

        # Observaciones (88 dims — NO CAMBIAR)
        obs = torch.cat(
            (
                self.robot.data.projected_gravity_b,       # 3: orientacion/gravedad
                self.robot.data.root_ang_vel_b,           # 3: velocidad angular del root
                self.robot.data.root_lin_vel_b,           # 3: velocidad lineal del root
                self.joint_pos - self.robot.data.default_joint_pos,  # 26: desviacion de joints
                self.joint_vel,                            # 26: velocidad de joints
                self._previous_actions,                    # 26: acciones previas
                feet_distance                              # 1: distancia entre pies
            ),
            dim=-1,
        )

        return {"policy": obs}

    # --------------------------------------------------------------------- #
    # Rewards
    # --------------------------------------------------------------------- #

    def _get_rewards(self) -> torch.Tensor:

        left_foot_pos = self.robot.data.body_pos_w[:, self._left_foot_id[0]]
        right_foot_pos = self.robot.data.body_pos_w[:, self._right_foot_id[0]]
        com_xy = self.robot.data.root_pos_w[:, :2]
        joint_pos_dev = self.joint_pos - self.robot.data.default_joint_pos

        # Pre-index tensors using dynamic joint IDs (outside TorchScript)
        hip_left_pos_dev = joint_pos_dev[:, self._hip_left_ids]
        hip_right_pos_dev = joint_pos_dev[:, self._hip_right_ids]
        left_leg_actions = self.actions[:, self._left_leg_ids]
        right_leg_actions = self.actions[:, self._right_leg_ids]
        knee_pos_dev = joint_pos_dev[:, self._knee_joint_ids]

        return compute_rewards(
            # --- Reward scales (floats) ---
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_base_height,
            self.cfg.rew_scale_orientation,
            self.cfg.rew_scale_joint_pos,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_recovery,
            self.cfg.rew_scale_feet_separation,
            self.cfg.rew_scale_com_support,
            self.cfg.rew_scale_co_activation,
            self.cfg.rew_scale_sync,
            self.cfg.rew_scale_rigidity,
            self.cfg.enable_com_reward,
            self.cfg.target_base_height,
            self.cfg.rew_scale_feet_together,
            self.cfg.target_feet_distance,
            self.cfg.rew_scale_foot_lateral_symmetry,
            self.cfg.rew_scale_com_lateral_balance,
            self.cfg.rew_scale_yaw_rate,
            self.cfg.rew_scale_knee_extension,
            self.cfg.rew_scale_return_to_default,
            self.cfg.rew_scale_bilateral_balance,
            # --- Tensors ---
            self.robot.data.root_pos_w[:, 2],
            self.robot.data.projected_gravity_b,
            joint_pos_dev,
            self.joint_vel,
            self.actions,
            self._previous_actions,
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            left_foot_pos,
            right_foot_pos,
            com_xy,
            self.reset_terminated,
            self.joint_pos,
            # --- Pre-indexed tensors (dynamic joint detection) ---
            hip_left_pos_dev,
            hip_right_pos_dev,
            left_leg_actions,
            right_leg_actions,
            knee_pos_dev,
        )

    # --------------------------------------------------------------------- #
    # Done
    # --------------------------------------------------------------------- #

    def _get_dones(self):

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        base_height = self.robot.data.root_pos_w[:, 2]
        fallen_height = base_height < 0.3

        gravity_proj = self.robot.data.projected_gravity_b
        orientation_error = torch.sum(torch.square(gravity_proj[:, :2]), dim=-1)
        fallen_tilt = orientation_error > 0.5

        fallen = fallen_height | fallen_tilt

        return fallen, time_out

    # --------------------------------------------------------------------- #
    # Reset
    # --------------------------------------------------------------------- #

    def _reset_idx(self, env_ids: Sequence[int] | None):

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 2] = self.cfg.initial_base_height

        self._push_timer[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# --------------------------------------------------------------------- #
# Reward Function
# --------------------------------------------------------------------- #

@torch.jit.script
def compute_rewards(
    # --- Reward scales (floats) ---
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_base_height: float,
    rew_scale_orientation: float,
    rew_scale_joint_pos: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_recovery: float,
    rew_scale_feet_separation: float,
    rew_scale_com_support: float,
    rew_scale_co_activation: float,
    rew_scale_sync: float,
    rew_scale_rigidity: float,
    enable_com_reward: float,  # 1.0 = activo, 0.0 = inactivo
    target_height: float,
    rew_scale_feet_together: float,
    target_feet_distance: float,
    rew_scale_foot_lateral_symmetry: float,
    rew_scale_com_lateral_balance: float,
    rew_scale_yaw_rate: float,
    rew_scale_knee_extension: float,
    rew_scale_return_to_default: float,
    rew_scale_bilateral_balance: float,
    # --- Tensors ---
    base_height: torch.Tensor,
    gravity_proj: torch.Tensor,
    joint_pos_dev: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    left_foot_pos: torch.Tensor,
    right_foot_pos: torch.Tensor,
    com_xy: torch.Tensor,
    reset_terminated: torch.Tensor,
    joint_pos: torch.Tensor,
    # --- Pre-indexed tensors (dynamic joint detection) ---
    hip_left_pos_dev: torch.Tensor,
    hip_right_pos_dev: torch.Tensor,
    left_leg_actions: torch.Tensor,
    right_leg_actions: torch.Tensor,
    knee_pos_dev: torch.Tensor,
):
    # ================================================================
    # BASE REWARDS
    # ================================================================

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()

    height_error = torch.square(base_height - target_height)
    rew_height = rew_scale_base_height * torch.exp(-height_error / 0.05)

    orientation_error = torch.sum(torch.square(gravity_proj[:, :2]), dim=-1)
    rew_orientation = rew_scale_orientation * torch.exp(-orientation_error / 0.01)

    rew_joint_pos = rew_scale_joint_pos * torch.sum(torch.square(joint_pos_dev), dim=-1)
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=-1)

    action_diff = torch.sum(torch.square(actions - previous_actions), dim=-1)
    rew_action_rate = rew_scale_action_rate * action_diff

    rew_lin_vel = rew_scale_lin_vel * torch.sum(torch.square(lin_vel), dim=-1)
    rew_ang_vel = rew_scale_ang_vel * torch.sum(torch.square(ang_vel), dim=-1)

    # ================================================================
    # FEET SEPARATION (quadratic penalty for excess opening)
    # ================================================================

    foot_distance = torch.norm(left_foot_pos[:, :2] - right_foot_pos[:, :2], dim=-1)
    support_center = (left_foot_pos[:, :2] + right_foot_pos[:, :2]) / 2.0
    max_allowed = 0.40
    excess = torch.clamp(foot_distance - max_allowed, min=0.0)
    rew_feet_sep = -rew_scale_feet_separation * torch.square(excess)

    # ================================================================
    # COM SUPPORT (gated by enable_com_reward — ETAPA 2+)
    # ================================================================

    com_error = torch.norm(com_xy - support_center, dim=-1)
    rew_com = rew_scale_com_support * torch.exp(-com_error / 0.15)
    rew_com = rew_com * float(enable_com_reward)

    # ================================================================
    # RECOVERY (orientation-based)
    # ================================================================

    rew_recovery = rew_scale_recovery * (1.0 - orientation_error)

    # ================================================================
    # HIP SYMMETRY (dynamic joint IDs — sum across hip joints per side)
    # ================================================================

    hip_left = torch.sum(hip_left_pos_dev, dim=-1)
    hip_right = torch.sum(hip_right_pos_dev, dim=-1)
    symmetry_penalty = torch.square(hip_left + hip_right)
    rew_symmetry = -0.5 * symmetry_penalty

    # ================================================================
    # CO-ACTIVATION & SYNC (dynamic leg joint IDs)
    # ================================================================

    co_activation = torch.sum(left_leg_actions * right_leg_actions, dim=-1)
    rew_co_activation = -rew_scale_co_activation * torch.square(co_activation)

    sync = torch.sum(left_leg_actions * right_leg_actions, dim=-1)
    rew_sync = -rew_scale_sync * torch.square(sync)

    # ================================================================
    # RIGIDITY PENALTY
    # ================================================================

    eps = 0.001
    is_stationary = torch.abs(joint_vel) < eps
    rew_rigidity = rew_scale_rigidity * torch.sum(is_stationary.float(), dim=-1)

    # ================================================================
    # FEET TOGETHER (distance toward target)
    # ================================================================

    feet_distance_error = torch.square(foot_distance - target_feet_distance)
    rew_feet_together = rew_scale_feet_together * torch.exp(-feet_distance_error / 0.05)

    # ================================================================
    # FOOT LATERAL SYMMETRY
    # ================================================================

    left_foot_x = left_foot_pos[:, 0] - com_xy[:, 0]
    right_foot_x = right_foot_pos[:, 0] - com_xy[:, 0]
    foot_lateral_error = torch.square(left_foot_x + right_foot_x)
    rew_foot_lateral_symmetry = rew_scale_foot_lateral_symmetry * torch.exp(-foot_lateral_error / 0.15)

    # ================================================================
    # COM LATERAL BALANCE
    # FIX: Positive sign — rewards centered COM (was inverted: penalized centering)
    # ================================================================

    com_lateral_error = torch.abs(com_xy[:, 0] - support_center[:, 0])
    rew_com_lateral_balance = rew_scale_com_lateral_balance * torch.exp(-com_lateral_error / 0.15)

    # ================================================================
    # YAW RATE PENALTY
    # ================================================================

    yaw_rate = ang_vel[:, 2]
    rew_yaw_rate = -rew_scale_yaw_rate * torch.square(yaw_rate) * 0.1

    # ================================================================
    # KNEE EXTENSION PENALTY
    # FIX: With negative scale (-5.0), penalizes bent knees (deviation from default)
    # Uses dynamic knee joint IDs via pre-indexed knee_pos_dev
    # ================================================================

    rew_knee_extension = rew_scale_knee_extension * torch.sum(torch.square(knee_pos_dev), dim=-1)

    # ================================================================
    # NEW: RETURN TO DEFAULT POSE (gated by orientation quality)
    # Only activates when robot is upright — rewards being near default pose
    # ================================================================

    orientation_gate = torch.exp(-orientation_error / 0.02)
    total_joint_deviation = torch.sum(torch.square(joint_pos_dev), dim=-1)
    rew_return_to_default = rew_scale_return_to_default * orientation_gate * torch.exp(-total_joint_deviation / 0.5)

    # ================================================================
    # NEW: BILATERAL BALANCE (symmetric foot heights — weight distribution proxy)
    # Penalizes asymmetric ankle heights (one foot lifting = uneven load)
    # ================================================================

    left_foot_z = left_foot_pos[:, 2]
    right_foot_z = right_foot_pos[:, 2]
    foot_height_asymmetry = torch.square(left_foot_z - right_foot_z)
    rew_bilateral_balance = -rew_scale_bilateral_balance * foot_height_asymmetry

    # ================================================================
    # TOTAL REWARD
    # FIX: Removed rew_scale_recovery scalar leak (was adding constant +2.0)
    # ================================================================

    total_reward = (
        rew_alive
        + rew_termination
        + rew_height
        + rew_orientation
        + rew_joint_pos
        + rew_joint_vel
        + rew_action_rate
        + rew_lin_vel
        + rew_ang_vel
        + rew_feet_sep
        + rew_com
        + rew_recovery
        + rew_symmetry
        + rew_co_activation
        + rew_sync
        + rew_rigidity
        + rew_feet_together
        + rew_foot_lateral_symmetry
        + rew_com_lateral_balance
        + rew_yaw_rate
        + rew_knee_extension
        + rew_return_to_default
        + rew_bilateral_balance
    )

    return total_reward * 0.1
