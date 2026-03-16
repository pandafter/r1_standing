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
    observation_space = 26 * 2 + 3 + 3 + 3 + 26 + 1  # 88 = gravity(3) + ang_vel(3) + lin_vel(3) + joint_pos(26) + joint_vel(26) + prev_actions(26) + feet_distance(1)
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
    action_scale = 0.025  # Acciones moderadas para micro-ajustes
    
    # ===== RECOMPENSAS PARA STANDING CON ANKLE PITCH =====
    # Reward: mantener posicion vertical
    rew_scale_alive = 1.0
    rew_scale_terminated = -500.0

    #POSTURA
    rew_scale_orientation = 20.0
    rew_scale_base_height = 10.0
    
    # Penalty: minimizar movimiento overall (mantener como antes para estabilidad dinamica)
    rew_scale_joint_pos = -0.02
    rew_scale_joint_vel = -0.02
    rew_scale_lin_vel = -0.03
    rew_scale_ang_vel = -0.05
    rew_scale_action_rate = -0.03
    
    # Penalty por desviacion de rodillas del default (mantener piernas rectas)
    rew_scale_knee_extension = -5.0  # Penaliza rodillas desviadas del default
    
    
    # Reset
    initial_base_height = 0.75
    target_base_height = 0.75
    max_tilt_angle = 0.6  # Angulo pequeno - detectar caida temprano


    # -------------------------------------------------
    # PUSH RECOVERY CONFIG
    # -------------------------------------------------

    # ETAPA 1: Pushes OFF (postura estatica pura)
    push_interval_s = 1.0     # Pushes desactivados
    push_force_min = 0.02
    push_force_max = 0.03

    # Flags para controlar que rewards estan activos
    enable_com_reward = 1.0      # ETAPA 1: OFF (1.0 = activo, 0.0 = inactivo)
    enable_feet_penalty_quad = True  # Penalizacion cuadratica activa

    # -------------------------------------------------
    # NUEVOS REWARD SCALES - ETAPA 1: Postura Estatica Pura
    # -------------------------------------------------

    rew_scale_recovery = 2.0
    rew_scale_feet_separation = 8.0  # Penalizacion CUADRATICA por pies muy abiertos
    rew_scale_com_support = 1.0       # ETAPA 1: OFF

    # ===== REWARDS PARA PIES JUNTOS Y EQUILIBRIO SIMETRICO =====
    # Balance entre pies juntos y capacidad de correccion
    rew_scale_feet_together = 10.0   # Moderate - permite correccion
    target_feet_distance = 0.28       # Distancia objetivo entre pies (~28cm)
    
    # Reward por simetria entre pies (evitar pivote extremo)
    rew_scale_foot_lateral_symmetry = 15.0  # Moderate
    
    # Recompensa por COM centrado sobre soporte
    rew_scale_com_lateral_balance = 12.0  # Recompensa COM centrado sobre soporte
    
    # Penalizacion por velocidad angular en Z (giro) - permitir correcciones pequenas
    rew_scale_yaw_rate = 15.0  # Moderate - permite pequenos ajustes

    # Penalizaciones estructurales
    rew_scale_co_activation = 0.5    # Penalizacion co-activacion bilateral
    rew_scale_sync = 0.3             # Penalizacion sincronizacion
    rew_scale_rigidity = 0.05         # Penalizacion rigidez

    # Nuevos rewards: retorno a pose y balance bilateral
    rew_scale_return_to_default = 6.0    # Reward por retornar a pose default cuando erguido
    rew_scale_bilateral_balance = 8.0    # Penalizacion asimetria de altura de pies
    
    # ===== REWARDS PARA ESTABILIDAD TORSO Y BRAZOS (corregir giro hacia izquierda) =====
    # Penalizacion fuerte por rotacion del torso (yaw) - evitar que gire hacia un lado
    rew_scale_torso_yaw = 25.0
    
    # Penalizacion por brazo izquierdo cruzando hacia la derecha
    # El brazo izquierdo no debe pasar de la linea central del cuerpo
    rew_scale_left_arm_crossing = 20.0
    
    # Reward por mantener brazos en posicion neutral/simetrica
    rew_scale_arm_symmetry = 15.0
    
    # Penalizacion por movimiento excesivo de brazos
    rew_scale_arm_movement = 3.0
