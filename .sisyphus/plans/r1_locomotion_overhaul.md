# Plan: R1 Locomotion — Mid-Level Navigation Policy Overhaul

## TL;DR

> **Quick Summary**: Complete rewrite of the R1 locomotion environment from a broken velocity-tracking prototype to a proper navigation-based mid-level policy with bilateral symmetry augmentation.
> 
> **Deliverables**:
> - Rewritten `r1_locomotion_env.py` with navigation formulation (target in robot frame, 26-joint action space)
> - Rewritten `r1_locomotion_env_cfg.py` with locomotion-appropriate reward scales
> - New bilateral symmetry module (`mdp/symmetry/r1_bilateral.py`)
> - Fixed RSL-RL v4 agent config with symmetry integration
> - Fixed task registration (remove "Template-" prefix)
> 
> **Estimated Effort**: Large (32 tasks)
> **Parallel Execution**: YES — 5 waves
> **Critical Path**: Task 1 → Task 5 → Task 11 → Task 20 → Task 30

---

## Context

### Original Request
User wants a mid-level locomotion policy for the Unitree R1 humanoid that:
- Uses navigation formulation (target position as command, not velocity tracking)
- Implements bilateral symmetry augmentation (RSL-RL v4 native)
- Follows research conclusions: WASD updates target_pos, policy decides how to walk
- No YOLO/LIDAR (deferred to future policy layer)

### Interview Summary
**Key Decisions**:
- Action space = 26 joints only (target_pos as observation/command, not action output)
- Navigation in robot frame (relative target, not world coordinates)
- Implement symmetry NOW using RSL-RL v4's `RslRlSymmetryCfg`
- Two separate plans: standing fixes first, then locomotion overhaul

**Research Findings**:
- Isaac Lab H1 velocity tracking as reference for biped rewards (`feet_air_time_positive_biped`)
- Isaac Lab direct/humanoid uses potential-based navigation with heading reward
- RSL-RL v4 symmetry: `compute_symmetric_states(env, obs, actions)` with TensorDict
- ANYmal example at `isaaclab_tasks/.../locomotion/velocity/mdp/symmetry/anymal.py` as template
- R1 has 11 bilateral joint pairs for symmetry mapping

### Current State (BROKEN — needs major rework)
The existing `r1_locomotion_env.py` (513 lines) has fundamental design issues:
1. **Action space = 29** (26 joints + 2 target_delta + 1 speed_factor) — policy outputs target commands, contradicting research
2. **Reward conflicts**: `rew_rigidity` rewards stiff joints, `rew_co_activation` penalizes walking gait, `rew_return_to_default` penalizes walking deviation
3. **Agent config**: OLD RSL-RL format (missing `actor`, `critic`, `obs_groups`) — will crash
4. **Task ID**: Still `"Template-R1-Locomotion-Direct-v0"` — template name
5. **No symmetry**: No bilateral symmetry implementation
6. **No gait rewards**: No feet_air_time, no contact pattern, no stride rewards

---

## Work Objectives

### Core Objective
Rewrite the R1 locomotion environment to implement a navigation-based mid-level policy where the robot receives target positions as commands and learns to walk toward them using a natural bipedal gait, stabilized by bilateral symmetry augmentation.

### Concrete Deliverables
- `r1_locomotion_env.py` — complete rewrite (~500 lines)
- `r1_locomotion_env_cfg.py` — complete rewrite (~120 lines)
- `agents/rsl_rl_ppo_cfg.py` — RSL-RL v4 format + symmetry config (~60 lines)
- `mdp/symmetry/r1_bilateral.py` — bilateral symmetry function (~150 lines)
- `mdp/__init__.py` + `mdp/symmetry/__init__.py` — package inits
- `__init__.py` — fixed task registration

### Definition of Done
- [ ] `python scripts/rsl_rl/train.py --task=R1-Locomotion-Direct-v0` starts without errors
- [ ] Training runs for 100+ iterations with rewards trending upward
- [ ] No `KeyError` from RSL-RL v4 config
- [ ] Symmetry augmentation active (logged in training output)

### Must Have
- Navigation formulation with robot-frame target commands
- 26-joint action space (no target/speed actions)
- Bilateral symmetry (11 pairs, data augmentation + mirror loss)
- Proper bipedal gait rewards (feet air time, contact alternation, feet slide penalty)
- Stability rewards from standing policy (orientation, height, COM balance)
- Arm-return reward (arms at natural pose while walking)
- RSL-RL v4 compatible agent config

### Must NOT Have (Guardrails)
- ❌ Network architecture changes (keep [256, 128, 64])
- ❌ YOLO or LIDAR integration (deferred)
- ❌ RND or additional networks
- ❌ Velocity tracking formulation (use navigation instead)
- ❌ Hardcoded joint indices (use find_joints with regex)
- ❌ Python lists or dynamic indexing inside @torch.jit.script
- ❌ `rew_rigidity` — kills walking
- ❌ `rew_co_activation` / `rew_sync` — kills bipedal gait
- ❌ Strong `rew_return_to_default` for legs — conflicts with walking pose
- ❌ `velocity_learnable` action outputs — removed

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (smoke test pattern from r1_standing)
- **Automated tests**: YES (tests-after for reward function, smoke test for param count)
- **Framework**: Python script + assertions (same pattern as r1_standing smoke test)

### QA Policy
- Every task includes a verification step
- Symmetry module gets dedicated unit test (swap → swap = identity)
- Final verification: training launch + 100 iterations without crash
- Evidence saved to `.sisyphus/evidence/`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — independent scaffolding):
├── Task 1: Fix __init__.py task registration [quick]
├── Task 2: Create mdp/ package structure [quick]
├── Task 3: Fix agent config for RSL-RL v4 [quick]
└── Task 4: Design observation space constants [quick]

Wave 2 (Core modules — after Wave 1):
├── Task 5: Rewrite env cfg (rewards, spaces, commands) [unspecified-high]
├── Task 6: Implement bilateral symmetry module [deep]
├── Task 7: Design reward function signature [unspecified-high]
└── Task 8: Integrate symmetry into agent config [quick]

Wave 3 (Environment core — after Wave 2):
├── Task 9: Rewrite env __init__ + scene setup [unspecified-high]
├── Task 10: Rewrite _pre_physics_step + _apply_action [unspecified-high]
├── Task 11: Rewrite _get_observations (robot-frame target) [deep]
├── Task 12: Rewrite _get_dones + _reset_idx [unspecified-high]
└── Task 13: Implement gait tracking state (contact, phase) [deep]

Wave 4 (Rewards — after Wave 3):
├── Task 14: Implement _get_rewards (pre-indexing) [unspecified-high]
├── Task 15: Implement compute_rewards — stability block [deep]
├── Task 16: Implement compute_rewards — locomotion block [deep]
├── Task 17: Implement compute_rewards — gait block [deep]
├── Task 18: Implement compute_rewards — efficiency block [unspecified-high]
└── Task 19: Implement compute_rewards — total + assembly [unspecified-high]

Wave 5 (Verification — after Wave 4):
├── Task 20: Smoke test for reward function [unspecified-high]
├── Task 21: Symmetry unit test [deep]
├── Task 22: Training launch test (100 iterations) [deep]
└── Task 23: Final checklist verification [unspecified-high]
```

### Dependency Matrix
- **1-4**: None → 5-8
- **5**: 4 → 9-13
- **6**: 2 → 8, 21
- **7**: 5 → 14-19
- **8**: 3, 6 → 22
- **9-13**: 5 → 14
- **14-19**: 7, 9-13 → 20
- **20-23**: 14-19 → done

---

## R1 Joint Structure Reference

### Complete Joint Map (26 joints)
```
HEAD (2):   head_pitch_joint, head_yaw_joint
WAIST (2):  waist_yaw_joint, waist_roll_joint
L_LEG (6):  left_hip_pitch/roll/yaw, left_knee, left_ankle_pitch/roll
R_LEG (6):  right_hip_pitch/roll/yaw, right_knee, right_ankle_pitch/roll
L_ARM (5):  left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll
R_ARM (5):  right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll
```

### Symmetry Pairs (11 pairs)
```
left_hip_pitch      ↔  right_hip_pitch      (same sign)
left_hip_roll       ↔  right_hip_roll       (NEGATE)
left_hip_yaw        ↔  right_hip_yaw        (NEGATE)
left_knee           ↔  right_knee           (same sign)
left_ankle_pitch    ↔  right_ankle_pitch    (same sign)
left_ankle_roll     ↔  right_ankle_roll     (NEGATE)
left_shoulder_pitch ↔  right_shoulder_pitch (same sign)
left_shoulder_roll  ↔  right_shoulder_roll  (NEGATE)
left_shoulder_yaw   ↔  right_shoulder_yaw   (NEGATE)
left_elbow          ↔  right_elbow          (same sign)
left_wrist_roll     ↔  right_wrist_roll     (NEGATE)

Non-swapped (4): head_pitch, head_yaw (same), waist_yaw (NEGATE), waist_roll (NEGATE)
```

### Observation Space Design (97 dims)
```
projected_gravity_b         3   
root_ang_vel_b              3   
root_lin_vel_b              3   
joint_pos - default         26  
joint_vel                   26  
previous_actions            26  
command_target_xy_robot     2   (target in robot frame: forward, lateral)
command_target_distance     1   
command_target_heading      1   
feet_contact_binary         2   (left, right — 1 if grounded)
foot_height_left            1   
foot_height_right           1   
gait_phase                  2   (sin/cos of gait clock)
                          ----
Total:                     97
```

### Reward Structure
```
STABILITY BLOCK:
  alive               +1.0    1 - terminated
  terminated          -500.0  terminated
  base_height         +10.0   exp(-height_err²/0.05)
  orientation         +15.0   exp(-tilt/0.01)
  com_lateral_balance  +8.0   exp(-lateral_err/0.15)

LOCOMOTION BLOCK:
  target_progress     +15.0   dot(vel_body, dir_to_target_robot)
  target_reached      +20.0   exp(-dist²/0.5) gated by dist < 0.3
  heading_alignment    +8.0   exp(-heading_err²/0.5)

GAIT BLOCK:
  feet_air_time        +5.0   (air_time - 0.3) per foot, biped alternating
  feet_slide           -3.0   foot_vel_xy * contact_force
  feet_separation      -5.0   clamp(dist - 0.5, 0)²

EFFICIENCY BLOCK:
  joint_vel            -0.01  sum(vel²)
  action_rate          -0.02  sum((a - a_prev)²)
  energy               -0.005 sum(|torque * vel|)
  knee_extension       -2.0   sum(knee_dev²)
  yaw_rate             -5.0   yaw_rate² * 0.1
  arm_return           +6.0   exp(-arm_dev²/0.3)
  bilateral_balance    -3.0   (left_z - right_z)²
```

---

## TODOs

### Wave 1: Foundation (all independent, start immediately)

- [ ] 1. Fix Task Registration in __init__.py

  **What to do**:
  - Change gym registration ID from `"Template-R1-Locomotion-Direct-v0"` to `"R1-Locomotion-Direct-v0"`
  - Update entry_point and cfg references if class names change

  **File**: `C:\space_r1\r1_locomotion\source\r1_locomotion\r1_locomotion\tasks\direct\r1_locomotion\__init__.py`

  **References**:
  - Current registration at line 16: `id="Template-R1-Locomotion-Direct-v0"`
  - r1_standing pattern: `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\__init__.py`

  **Acceptance Criteria**:
  - [ ] Task ID is `"R1-Locomotion-Direct-v0"` (no "Template-" prefix)

---

- [ ] 2. Create mdp/ Package Structure

  **What to do**:
  - Create directory: `r1_locomotion/tasks/direct/r1_locomotion/mdp/`
  - Create directory: `r1_locomotion/tasks/direct/r1_locomotion/mdp/symmetry/`
  - Create `mdp/__init__.py` with empty content
  - Create `mdp/symmetry/__init__.py` that exports the symmetry function

  **File paths to create**:
  - `C:\space_r1\r1_locomotion\source\r1_locomotion\r1_locomotion\tasks\direct\r1_locomotion\mdp\__init__.py`
  - `C:\space_r1\r1_locomotion\source\r1_locomotion\r1_locomotion\tasks\direct\r1_locomotion\mdp\symmetry\__init__.py`

  **References**:
  - ANYmal symmetry structure: `C:\space_r1\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\mdp\symmetry\anymal.py`

  **Acceptance Criteria**:
  - [ ] `mdp/` directory exists with `__init__.py`
  - [ ] `mdp/symmetry/` directory exists with `__init__.py`

---

- [ ] 3. Fix Agent Config for RSL-RL v4

  **What to do**:
  - Rewrite `rsl_rl_ppo_cfg.py` to RSL-RL v4 format
  - Add `obs_groups = {"actor": ["policy"], "critic": ["policy"]}`
  - Add `actor` dict with `class_name`, `hidden_dims=[256,128,64]`, `activation="elu"`, `obs_normalization=False`, `stochastic=True`, `init_noise_std=1.0`
  - Add `critic` dict with `class_name`, `hidden_dims=[256,128,64]`, `activation="elu"`, `obs_normalization=False`
  - Keep `policy = RslRlPpoActorCriticCfg(...)` for Isaac Lab compatibility
  - Update `policy` hidden dims from `[256,128,64]` to match actor/critic
  - Placeholder for symmetry config (will be filled in Task 8)

  **File**: `C:\space_r1\r1_locomotion\source\r1_locomotion\r1_locomotion\tasks\direct\r1_locomotion\agents\rsl_rl_ppo_cfg.py`

  **References**:
  - Working r1_standing config: `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\agents\rsl_rl_ppo_cfg.py` (already fixed for v4)
  - RSL-RL v4 PPO.construct_algorithm: `C:\Users\veter\AppData\Local\Programs\Python\Python311\Lib\site-packages\rsl_rl\algorithms\ppo.py` lines 482-515

  **Acceptance Criteria**:
  - [ ] Config has `actor`, `critic`, `obs_groups` dicts
  - [ ] `algorithm` section has `symmetry_cfg=None` placeholder

---

- [ ] 4. Design Observation Space Constants

  **What to do**:
  - Document and define the observation space dimensions as constants
  - Calculate total: gravity(3) + ang_vel(3) + lin_vel(3) + joint_pos(26) + joint_vel(26) + prev_actions(26) + target_xy(2) + target_dist(1) + target_heading(1) + feet_contact(2) + foot_heights(2) + gait_phase(2) = **97**
  - This is just a documentation/reference task — the actual observation construction happens in Task 11

  **Acceptance Criteria**:
  - [ ] Observation space = 97 documented and consistent across env_cfg and this plan

---

### Wave 2: Core Modules (after Wave 1)

- [ ] 5. Rewrite Environment Config

  **What to do**:
  - Complete rewrite of `r1_locomotion_env_cfg.py`
  - Change `action_space` from 29 to 26
  - Change `observation_space` from 113 to 97
  - Remove `target_pos_scale`, `max_target_distance`, `speed_scale`, `min_speed`, `max_speed` (no longer actions)
  - Add command config: `command_target_distance_range = [0.5, 3.0]`, `command_target_angle_range = [-pi, pi]`, `command_resample_time = 10.0`
  - Replace all reward scales with the new structure (see Reward Structure in plan header)
  - Remove: `rew_scale_position_error`, `rew_scale_velocity_learnable`, `rew_scale_progress`, `rew_scale_co_activation`, `rew_scale_sync`, `rew_scale_rigidity`, `rew_scale_return_to_default`, `rew_scale_foot_lateral_symmetry`
  - Add: `rew_scale_target_progress`, `rew_scale_target_reached`, `rew_scale_heading_alignment`, `rew_scale_feet_air_time`, `rew_scale_feet_slide`, `rew_scale_energy`, `rew_scale_arm_return`
  - Add gait config: `target_air_time = 0.3`, `gait_frequency = 1.5` (Hz)

  **File**: `C:\space_r1\r1_locomotion\source\r1_locomotion\r1_locomotion\tasks\direct\r1_locomotion\r1_locomotion_env_cfg.py`

  **References**:
  - Current cfg: read lines 1-100 (already read, all content known)
  - H1 humanoid cfg: `C:\space_r1\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\h1\rough_env_cfg.py`
  - Anymal C velocity cfg: `C:\space_r1\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\anymal_c\anymal_c_env_cfg.py`
  - Reward scales from plan header "Reward Structure" section

  **Acceptance Criteria**:
  - [ ] `action_space = 26`
  - [ ] `observation_space = 97`
  - [ ] No references to `velocity_learnable`, `speed_factor`, `target_pos_scale`
  - [ ] All reward scales from the plan's Reward Structure are present
  - [ ] `episode_length_s = 30.0` (keep existing)

---

- [ ] 6. Implement Bilateral Symmetry Module

  **What to do**:
  - Create `r1_bilateral.py` implementing `compute_symmetric_states(env, obs, actions)`
  - Use TensorDict for observations (RSL-RL v4 format)
  - Resolve joint indices DYNAMICALLY at first call using `env.unwrapped.robot.find_joints()`
  - Cache resolved indices for subsequent calls
  - Implement 2-way symmetry (left↔right swap only, no front-back for biped)
  - Handle observation symmetry: swap joint pos/vel for L↔R pairs, negate lateral velocity, negate yaw rate, negate lateral gravity component, swap foot data, negate lateral target command
  - Handle action symmetry: swap joint actions for L↔R pairs, negate roll/yaw joints
  - Return `(obs_aug, actions_aug)` where batch is doubled (original + mirrored)

  **File**: `C:\space_r1\r1_locomotion\source\r1_locomotion\r1_locomotion\tasks\direct\r1_locomotion\mdp\symmetry\r1_bilateral.py`

  **References**:
  - ANYmal symmetry (template): `C:\space_r1\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\mdp\symmetry\anymal.py`
  - RSL-RL symmetry extension: `C:\Users\veter\AppData\Local\Programs\Python\Python311\Lib\site-packages\rsl_rl\extensions\symmetry.py`
  - RSL-RL PPO symmetry integration: `C:\Users\veter\AppData\Local\Programs\Python\Python311\Lib\site-packages\rsl_rl\algorithms\ppo.py` lines 82-103, 240-355
  - R1 joint pairs from plan header "Symmetry Pairs" section
  - Isaac Lab symmetry config: `C:\space_r1\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\symmetry_cfg.py`

  **Key implementation details**:
  - Function signature: `def compute_symmetric_states(env, obs, actions) -> tuple`
  - `obs` is a TensorDict with key `"policy"` containing the flat observation
  - Must map observation indices to joint groups (obs layout from plan header)
  - Sign flips: roll joints and yaw joints get negated on L↔R swap
  - Lateral components: `root_lin_vel_b[:, 1]` (lateral vel), `root_ang_vel_b[:, 2]` (yaw), `projected_gravity_b[:, 1]` (lateral gravity), `command_target_xy_robot[:, 1]` (lateral target)

  **Acceptance Criteria**:
  - [ ] Function returns `(obs_aug, actions_aug)` with doubled batch size
  - [ ] swap(swap(x)) == x (involution property)
  - [ ] Sign flips applied to all roll/yaw joints
  - [ ] Joint indices resolved dynamically via find_joints (no hardcoded indices)
  - [ ] Caching mechanism for resolved indices

---

- [ ] 7. Design Reward Function Signature

  **What to do**:
  - Define the exact parameter list for the `@torch.jit.script` `compute_rewards` function
  - Must include all reward scales as float params, then all tensor data
  - Order matters — must match `_get_rewards` call order
  - Document which pre-indexed tensors are needed outside TorchScript

  **Acceptance Criteria**:
  - [ ] Parameter list documented with exact order
  - [ ] All 18 rewards from the plan's Reward Structure have corresponding scale params
  - [ ] All tensor params identified

---

- [ ] 8. Integrate Symmetry into Agent Config

  **What to do**:
  - Update `rsl_rl_ppo_cfg.py` from Task 3 to add symmetry config
  - Add import for `RslRlSymmetryCfg`
  - Add `symmetry_cfg = RslRlSymmetryCfg(use_data_augmentation=True, use_mirror_loss=True, mirror_loss_coeff=1.0, data_augmentation_func="r1_locomotion.tasks.direct.r1_locomotion.mdp.symmetry.r1_bilateral:compute_symmetric_states")`
  - The `data_augmentation_func` can be a string (resolved by `resolve_callable` in RSL-RL)

  **File**: `C:\space_r1\r1_locomotion\source\r1_locomotion\r1_locomotion\tasks\direct\r1_locomotion\agents\rsl_rl_ppo_cfg.py`

  **References**:
  - RSL-RL resolve_callable: resolves dotted string to callable
  - Kuka Allegro example: no symmetry but shows algorithm config structure
  - RSL-RL PPO line 90: `resolve_callable(symmetry_cfg["data_augmentation_func"])`

  **Acceptance Criteria**:
  - [ ] `algorithm` has `symmetry_cfg` field set (not None)
  - [ ] `data_augmentation_func` points to our r1_bilateral module
  - [ ] `use_data_augmentation=True`, `use_mirror_loss=True`

---

### Wave 3: Environment Core (after Wave 2)

- [ ] 9. Rewrite env __init__ + _setup_scene

  **What to do**:
  - Keep `_setup_scene` (identical to current — robot + ground + light)
  - Rewrite `__init__`:
    - Joint detection (keep existing pattern for legs, add arms like standing fixes)
    - Remove `target_pos`, `current_pos`, `velocity_learnable` tensors
    - Add: `self._command_target_world = zeros(N, 2)` — target in world frame
    - Add: `self._command_target_robot = zeros(N, 2)` — target in robot frame (updated each step)
    - Add: `self._command_resample_timer = zeros(N)` — timer for command resampling
    - Add: `self._feet_air_time = zeros(N, 2)` — per-foot air time tracker
    - Add: `self._last_feet_contact = zeros(N, 2)` — previous contact state
    - Add: `self._gait_phase = zeros(N)` — gait clock
    - Add arm detection: `self._arm_ids` via `find_joints([".*shoulder.*", ".*elbow.*", ".*wrist.*"])`

  **File**: `C:\space_r1\r1_locomotion\source\r1_locomotion\r1_locomotion\tasks\direct\r1_locomotion\r1_locomotion_env.py`

  **References**:
  - Current env __init__: lines 35-74 (already read)
  - r1_standing env __init__: pattern for find_joints usage
  - Isaac Lab anymal_c env: `feet_air_time` tracking pattern

  **Acceptance Criteria**:
  - [ ] No `target_pos`, `velocity_learnable`, `current_pos` attributes
  - [ ] Has `_command_target_world`, `_command_target_robot`, `_command_resample_timer`
  - [ ] Has `_feet_air_time`, `_last_feet_contact`, `_gait_phase`
  - [ ] Arm joints detected dynamically

---

- [ ] 10. Rewrite _pre_physics_step + _apply_action

  **What to do**:
  - `_pre_physics_step`:
    - Remove target_pos update from actions (actions are now 26 joints only)
    - Remove velocity_learnable extraction
    - Add command resampling: when timer expires, sample new random target
    - Add world-to-robot-frame target conversion (rotate by inverse heading)
    - Update gait phase clock: `self._gait_phase += dt * cfg.gait_frequency * 2 * pi`
  - `_apply_action`:
    - Same as current (position-based control with soft limit clamping)
    - Actions are `actions[:, :26]` → simplifies to just `self.actions`

  **Key formula for world-to-robot-frame conversion**:
  ```python
  # Get robot heading from quaternion
  heading = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))
  cos_h, sin_h = cos(-heading), sin(-heading)
  # Rotate target delta into robot frame
  delta_world = self._command_target_world - root_pos_xy
  target_robot_x = delta_world[:, 0] * cos_h - delta_world[:, 1] * sin_h
  target_robot_y = delta_world[:, 0] * sin_h + delta_world[:, 1] * cos_h
  ```

  **References**:
  - Isaac Lab direct/humanoid: heading computation from quaternion
  - Current _apply_action: lines 119-138 (keep position control pattern)

  **Acceptance Criteria**:
  - [ ] No action parsing beyond index 26
  - [ ] Command resampling on timer
  - [ ] Target converted to robot frame each step
  - [ ] Gait phase clock advances

---

- [ ] 11. Rewrite _get_observations (robot-frame target)

  **What to do**:
  - Build observation tensor with 97 dimensions (see Observation Space Design in plan header)
  - Remove `target_pos`, `current_pos`, `velocity_learnable` from obs
  - Add `command_target_xy_robot` (2D: forward, lateral in robot frame)
  - Add `command_target_distance` (scalar: distance to target)
  - Add `command_target_heading` (scalar: heading error in radians)
  - Add `feet_contact_binary` (2D: left/right foot contact)
  - Add `foot_height_left`, `foot_height_right` (1D each)
  - Add `gait_phase` as sin/cos encoding (2D)

  **Computing foot contact**:
  ```python
  # Use foot body positions — if z < threshold, consider grounded
  left_contact = (left_foot_pos[:, 2] < 0.05).float()
  right_contact = (right_foot_pos[:, 2] < 0.05).float()
  ```

  **References**:
  - Current _get_observations: lines 144-171 (113 dims)
  - Isaac Lab anymal_c obs structure (contact + height scan)

  **Acceptance Criteria**:
  - [ ] Output dict `{"policy": obs}` with obs shape `(N, 97)`
  - [ ] No `target_pos`, `current_pos`, `velocity_learnable` in obs
  - [ ] Robot-frame target commands present
  - [ ] Foot contact and height present
  - [ ] Gait phase as sin/cos

---

- [ ] 12. Rewrite _get_dones + _reset_idx

  **What to do**:
  - `_get_dones`: Keep same logic (height < 0.3 OR tilt > 0.5 OR timeout)
  - `_reset_idx`:
    - Remove target_pos reset and velocity_learnable reset
    - Add command resampling on reset: sample random target in world frame
    - Reset feet_air_time, last_feet_contact, gait_phase
    - Reset command_resample_timer

  **Command sampling on reset**:
  ```python
  dist = uniform(cfg.command_target_distance_range[0], cfg.command_target_distance_range[1])
  angle = uniform(cfg.command_target_angle_range[0], cfg.command_target_angle_range[1])
  self._command_target_world[env_ids, 0] = spawn_x + dist * cos(angle)
  self._command_target_world[env_ids, 1] = spawn_y + dist * sin(angle)
  ```

  **References**:
  - Current _reset_idx: lines 281-319

  **Acceptance Criteria**:
  - [ ] No `target_pos` or `velocity_learnable` in reset
  - [ ] Random target sampled in world frame on reset
  - [ ] All locomotion state variables reset

---

- [ ] 13. Implement Gait Tracking State (contact, phase, air time)

  **What to do**:
  - In `_pre_physics_step` or a helper method, update gait tracking:
    - Detect foot contact from foot body z-position
    - Track air time per foot: increment when airborne, reset on contact
    - Track contact transitions (touchdown, liftoff) for reward computation
  - Store `self._feet_contact = [left_contact, right_contact]` as boolean tensor
  - Store `self._feet_air_time` as float tensor (seconds each foot has been in air)

  **Air time tracking logic**:
  ```python
  contact = stack([left_z < 0.05, right_z < 0.05], dim=-1).float()
  # Time in air: increment if not contacting, reset on contact
  self._feet_air_time += dt
  self._feet_air_time *= (1.0 - contact)  # reset to 0 on contact
  # Detect first contact (was airborne, now grounded)
  first_contact = (contact > 0.5) & (self._last_feet_contact < 0.5)
  self._last_feet_contact = contact
  ```

  **References**:
  - Isaac Lab anymal_c: `feet_air_time` reward computation
  - H1 velocity tracking: `feet_air_time_positive_biped` reward

  **Acceptance Criteria**:
  - [ ] `_feet_air_time` tracks seconds each foot is airborne
  - [ ] Resets to 0 on contact
  - [ ] `_last_feet_contact` tracks previous state for transition detection
  - [ ] `first_contact` boolean for gait reward

---

### Wave 4: Rewards (after Wave 3)

- [ ] 14. Implement _get_rewards (pre-indexing + function call)

  **What to do**:
  - Pre-index all tensors outside TorchScript (same pattern as r1_standing)
  - Compute derived quantities: target_distance, target_heading, progress
  - Call compute_rewards with all params in correct order (from Task 7)

  **Pre-indexed tensors needed**:
  ```python
  joint_pos_dev = self.joint_pos - self.robot.data.default_joint_pos
  knee_pos_dev = joint_pos_dev[:, self._knee_joint_ids]
  arm_pos_dev = joint_pos_dev[:, self._arm_ids]
  left_foot_pos = self.robot.data.body_pos_w[:, self._left_foot_id[0]]
  right_foot_pos = self.robot.data.body_pos_w[:, self._right_foot_id[0]]
  com_xy = self.robot.data.root_pos_w[:, :2]
  ```

  **Derived quantities**:
  ```python
  target_distance = torch.norm(self._command_target_robot, dim=-1)
  target_heading = torch.atan2(self._command_target_robot[:, 1], self._command_target_robot[:, 0])
  # Progress: velocity in direction of target (robot frame)
  vel_robot = self.robot.data.root_lin_vel_b[:, :2]
  dir_to_target = self._command_target_robot / (target_distance.unsqueeze(-1) + 1e-6)
  progress = torch.sum(vel_robot * dir_to_target, dim=-1)
  ```

  **References**:
  - r1_standing _get_rewards: `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\r1_standing_env.py` lines 172-233

  **Acceptance Criteria**:
  - [ ] All tensor indexing done outside @torch.jit.script
  - [ ] Parameter order matches compute_rewards signature
  - [ ] No Python lists in function call

---

- [ ] 15. Implement compute_rewards — Stability Block

  **What to do**:
  - Inside `@torch.jit.script def compute_rewards(...)`:
  - `rew_alive = scale * (1 - terminated)`
  - `rew_termination = scale * terminated`
  - `rew_height = scale * exp(-height_err²/0.05)`
  - `rew_orientation = scale * exp(-tilt_err/0.01)`
  - `rew_com_lateral_balance = scale * exp(-lateral_err/0.15)` (same as standing, fixed sign)

  **Acceptance Criteria**:
  - [ ] 5 stability rewards computed correctly
  - [ ] All use positive-reward formulation (exp(-error)) not negative

---

- [ ] 16. Implement compute_rewards — Locomotion Block

  **What to do**:
  - `rew_target_progress = scale * progress` (dot product of velocity toward target in robot frame)
  - `rew_target_reached = scale * exp(-dist²/0.5) * (dist < 0.3).float()` (gate: only when close)
  - `rew_heading_alignment = scale * exp(-heading²/0.5)` (face target direction)

  **Acceptance Criteria**:
  - [ ] Progress reward positive when moving toward target, negative when moving away
  - [ ] Target reached only triggers within 0.3m
  - [ ] Heading alignment maximal when facing target

---

- [ ] 17. Implement compute_rewards — Gait Block

  **What to do**:
  - `rew_feet_air_time`: For each foot, reward `(air_time - target_air_time)` at first_contact. Only count if alternating (left then right, not both).
    ```python
    rew_feet_air_time = scale * torch.sum((feet_air_time - target_air_time) * first_contact, dim=-1)
    ```
  - `rew_feet_slide`: Penalize foot sliding (velocity while in contact)
    ```python
    foot_vel_left = torch.norm(left_foot_vel[:, :2], dim=-1)
    foot_vel_right = torch.norm(right_foot_vel[:, :2], dim=-1)
    slide_left = foot_vel_left * left_contact
    slide_right = foot_vel_right * right_contact
    rew_feet_slide = -scale * (slide_left + slide_right)
    ```
  - `rew_feet_separation`: Same as standing — penalize excess opening

  **Note**: `feet_air_time` and `first_contact` are tensors passed from `_get_rewards` (pre-computed in Task 13).

  **References**:
  - Isaac Lab `feet_air_time_positive_biped` reward function
  - Isaac Lab `feet_slide` reward function

  **Acceptance Criteria**:
  - [ ] Air time reward: positive for proper step duration, negative for too short/long
  - [ ] Slide penalty: only penalizes when foot is grounded AND moving
  - [ ] Separation: only penalizes excess (>0.5m), not normal stride width

---

- [ ] 18. Implement compute_rewards — Efficiency Block

  **What to do**:
  - `rew_joint_vel = -scale * sum(vel²)`
  - `rew_action_rate = -scale * sum((a - a_prev)²)`
  - `rew_energy = -scale * sum(|torque * vel|)` (Note: torque may need to be approximated from joint effort)
  - `rew_knee_extension = -scale * sum(knee_dev²)` (keep from standing)
  - `rew_yaw_rate = -scale * yaw_rate² * 0.1` (reduced for locomotion — some yaw needed for turning)
  - `rew_arm_return = scale * exp(-arm_dev²/0.3)` (arms at natural pose)
  - `rew_bilateral_balance = -scale * (left_z - right_z)²` (even foot heights when both grounded)

  **Acceptance Criteria**:
  - [ ] 7 efficiency/regularization rewards computed
  - [ ] Energy reward uses absolute value of torque*vel product
  - [ ] Arm return uses same formula as standing fixes plan

---

- [ ] 19. Implement compute_rewards — Total Assembly

  **What to do**:
  - Sum all 18 reward terms
  - Apply `* 0.1` global scaling (same as current env)
  - Return total_reward tensor

  **Acceptance Criteria**:
  - [ ] Exactly 18 terms summed
  - [ ] No missing or duplicate terms
  - [ ] Global 0.1 scale applied

---

### Wave 5: Verification (after Wave 4)

- [ ] 20. Smoke Test for Reward Function

  **What to do**:
  - Create `C:\space_r1\r1_locomotion\.sisyphus\evidence\reward_smoke_test.py`
  - Test compute_rewards with synthetic tensors
  - Verify: correct sign for each reward, correct dimensions, no NaN
  - Verify parameter count matches signature
  - Test edge cases: zero velocity, zero target distance, max tilt

  **Acceptance Criteria**:
  - [ ] All 18 reward terms tested for correct sign
  - [ ] Edge cases covered (at target, far from target, fallen)
  - [ ] No NaN or Inf in outputs

---

- [ ] 21. Symmetry Unit Test

  **What to do**:
  - Create `C:\space_r1\r1_locomotion\.sisyphus\evidence\symmetry_test.py`
  - Test involution: `sym(sym(obs, act)) == (obs, act)` (up to floating point)
  - Test that swapped joint indices are correct (left knee → right knee position)
  - Test sign flips on roll/yaw joints
  - Test observation symmetry (lateral velocity negated, etc.)

  **Acceptance Criteria**:
  - [ ] Involution property verified (max error < 1e-5)
  - [ ] All 11 joint pairs swap correctly
  - [ ] Sign flips verified for all 7 negated joints
  - [ ] Observation lateral components negated correctly

---

- [ ] 22. Training Launch Test

  **What to do**:
  - Run: `python scripts/rsl_rl/train.py --task=R1-Locomotion-Direct-v0 --max_iterations 100`
  - Verify: no crashes, no KeyError, no dimension mismatches
  - Check logs: symmetry augmentation active (should print obs_groups resolution)
  - Capture first 100 iterations of reward curve
  - Save evidence: terminal output screenshot

  **Acceptance Criteria**:
  - [ ] Training starts without errors
  - [ ] Runs 100 iterations without crash
  - [ ] Symmetry active in logs
  - [ ] Reward values are finite (no NaN/Inf)

---

- [ ] 23. Final Checklist Verification

  **Checklist**:
  - [ ] action_space = 26 in cfg (no target/speed actions)
  - [ ] observation_space = 97 in cfg
  - [ ] All 18 reward terms present in compute_rewards
  - [ ] compute_rewards param count matches _get_rewards call
  - [ ] No @torch.jit.script violations (no Python lists, no dynamic indexing)
  - [ ] Symmetry module resolves joint indices dynamically
  - [ ] Agent config has actor + critic + obs_groups + symmetry_cfg
  - [ ] Task registered as "R1-Locomotion-Direct-v0" (not "Template-")
  - [ ] No references to velocity_learnable, speed_factor, target_pos_scale
  - [ ] Robot-frame target commands in observations
  - [ ] Gait phase clock in observations
  - [ ] Feet contact and height in observations
  - [ ] Arms detected via find_joints, arm_return reward present
  - [ ] No hardcoded joint indices anywhere

---

## Commit Strategy

```
Wave 1: feat(r1-locomotion): scaffold package structure, fix registration and agent config
Wave 2: feat(r1-locomotion): add env config, symmetry module, and reward signature
Wave 3: feat(r1-locomotion): rewrite environment core (obs, actions, reset, gait tracking)
Wave 4: feat(r1-locomotion): implement complete reward function (18 terms)
Wave 5: test(r1-locomotion): add smoke tests, symmetry tests, and training launch verification
```

---

## Success Criteria

### Verification Commands
```bash
# Task registration
python scripts/list_envs.py | grep R1-Locomotion  # Expected: R1-Locomotion-Direct-v0

# Training launch
python scripts/rsl_rl/train.py --task=R1-Locomotion-Direct-v0 --max_iterations 100
# Expected: no crash, reward values finite, symmetry active in logs

# Smoke tests
python .sisyphus/evidence/reward_smoke_test.py   # Expected: all pass
python .sisyphus/evidence/symmetry_test.py       # Expected: all pass
```

### Final Checklist
- [ ] All "Must Have" present (navigation, symmetry, gait rewards, 26-action, robot-frame)
- [ ] All "Must NOT Have" absent (no rigidity, no co-activation, no velocity_learnable, no hardcoded indices)
- [ ] All tests pass
- [ ] Training runs 100+ iterations without crash
