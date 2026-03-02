# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.r1 import R1_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class R1StandingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    # Spaces
    action_space = 26
    observation_space = 26 * 2 + 3 + 3 + 3 + 26
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = R1_CFG.replace(prim_path="/World/envs/env_.*/R1")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8000,
        env_spacing=3.0,
        replicate_physics=True
    )

    # Joint control
    joint_names = [".*joint"]
    action_scale = 0.02  # Acciones pequeñas para micro-ajustes
    
    # ===== RECOMPENSAS PARA STANDING CON ANKLE PITCH =====
    # Reward: mantener posición vertical
    rew_scale_alive = 1.0
    rew_scale_terminated = -500.0

    #POSTURA
    rew_scale_orientation = 15.0
    rew_scale_base_height = 8.0
    
    # Penalty: minimizar movimiento overall
    rew_scale_joint_pos = -0.02
    rew_scale_joint_vel = -0.02
    rew_scale_lin_vel = -0.02
    rew_scale_ang_vel = -0.02
    rew_scale_action_rate = -0.05
    
    
    # Reset
    initial_base_height = 0.75
    target_base_height = 0.75
    max_tilt_angle = 0.6  # Ángulo pequeño - detectar caída temprano


    # -------------------------------------------------
    # PUSH RECOVERY CONFIG
    # -------------------------------------------------

    push_interval_s = 2.0          # cada cuántos segundos aplica empujón
    push_force_min = 100.0         # fuerza mínima (N)
    push_force_max = 250.0         # fuerza máxima (N)

    # -------------------------------------------------
    # NUEVOS REWARD SCALES
    # -------------------------------------------------

    rew_scale_recovery = 5.0
    rew_scale_feet_separation = 2.0
    rew_scale_com_support = 4.0

