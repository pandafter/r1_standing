# Plan: R1 Standing — Behavior Fixes (Anti-Circling + Natural Arms)

## Metadata
- **Created**: 2026-03-04
- **Scope**: r1_standing project only
- **Estimated tasks**: 7
- **Prerequisites**: Reward finetune plan completed (all 5 waves done)
- **Constraints**: NO network architecture changes, NO observation space changes (88 dims), dynamic joint detection, TorchScript-compatible rewards

## Context

After the reward finetune, the standing policy has two remaining behavioral issues:
1. **Circling**: Robot walks in small circles adjusting position instead of staying stationary. Root cause: NO penalty for XY displacement from spawn origin — only `rew_scale_lin_vel = -0.03` which allows slow drift.
2. **Arms at 90°**: The policy bends arms to ~90° instead of a natural resting position. Root cause: `default_joint_pos` for arms = 0.0 rad (straight hanging), and `rew_scale_return_to_default = 6.0` is spread across all 26 joints, too weak for arms specifically.

## Files to Modify

| File | Project | Changes |
|------|---------|---------|
| `C:\space_r1\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\r1.py` | IsaacLab (shared) | Change arm default_joint_pos values |
| `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\r1_standing_env.py` | r1_standing | Add origin displacement reward + arm-specific reward |
| `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\r1_standing_env_cfg.py` | r1_standing | Add new reward scales + update existing scales |

## R1 Joint Reference (Arms)

```
LEFT ARM:                          RIGHT ARM:
left_shoulder_pitch_joint (0.0)    right_shoulder_pitch_joint (0.0)
left_shoulder_roll_joint (0.05)    right_shoulder_roll_joint (-0.05)
left_shoulder_yaw_joint (0.0)      right_shoulder_yaw_joint (0.0)
left_elbow_joint (0.0)             right_elbow_joint (0.0)
left_wrist_roll_joint (0.0)        right_wrist_roll_joint (0.0)
```

---

## Tasks

### Task 1: Change R1_CFG Default Arm Positions to Natural Pose

**File**: `C:\space_r1\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\r1.py`

**What**: Change the `init_state.joint_pos` for arm joints from straight (0 rad) to a natural relaxed pose with arms slightly bent at waist height.

**Exact changes** in the `joint_pos` dict inside `InitialStateCfg`:

```python
# FROM:
"left_shoulder_pitch_joint": 0.0,
"left_shoulder_roll_joint": 0.05,
"left_shoulder_yaw_joint": 0.0,
"left_elbow_joint": 0,
"left_wrist_roll_joint": 0.0,
"right_shoulder_pitch_joint": 0.0,
"right_shoulder_roll_joint": -0.05,
"right_shoulder_yaw_joint": 0.0,
"right_elbow_joint": 0,
"right_wrist_roll_joint": 0.0,

# TO:
"left_shoulder_pitch_joint": 0.2,       # arms slightly forward
"left_shoulder_roll_joint": 0.1,        # slightly out from body
"left_shoulder_yaw_joint": 0.0,
"left_elbow_joint": 0.4,               # bent ~23° at elbow (natural waist height)
"left_wrist_roll_joint": 0.0,
"right_shoulder_pitch_joint": 0.2,      # mirror left
"right_shoulder_roll_joint": -0.1,      # mirror left (negative for right side)
"right_shoulder_yaw_joint": 0.0,
"right_elbow_joint": 0.4,              # mirror left
"right_wrist_roll_joint": 0.0,
```

**Rationale**: shoulder_pitch=0.2 rad (~11°) brings arms slightly forward. elbow=0.4 rad (~23°) bends them naturally so hands rest near waist height. shoulder_roll=±0.1 rad holds them slightly away from the body.

**WARNING**: This file is shared between r1_standing and r1_locomotion. Both will use the new defaults.

**QA**: After change, verify `default_joint_pos` dict has 26 entries. Visually confirm the values make anatomical sense (positive pitch = forward, positive elbow = bent).

---

### Task 2: Add Arm Joint Detection to r1_standing_env.py

**File**: `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\r1_standing_env.py`

**What**: In `__init__`, add dynamic detection for arm joints using `find_joints`, following the same pattern as leg joints.

**Add after line 59** (after `self._right_leg_ids` declaration):

```python
# Arm joint detection (for arm-specific return-to-default reward)
self._left_arm_ids, _ = self.robot.find_joints(
    [".*left.*shoulder.*", ".*left.*elbow.*", ".*left.*wrist.*"]
)
self._right_arm_ids, _ = self.robot.find_joints(
    [".*right.*shoulder.*", ".*right.*elbow.*", ".*right.*wrist.*"]
)
self._arm_ids, _ = self.robot.find_joints(
    [".*shoulder.*", ".*elbow.*", ".*wrist.*"]
)
```

**QA**: `self._arm_ids` should resolve to 10 indices (5 left + 5 right). `self._left_arm_ids` should be 5. `self._right_arm_ids` should be 5.

---

### Task 3: Store Initial Base Position for Displacement Penalty

**File**: `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\r1_standing_env.py`

**What**: Track the spawn origin XY for each env so we can penalize displacement.

**Add in `__init__`** (after `self._root_body_id = 0`, around line 44):

```python
# Track spawn position for displacement penalty
self._initial_base_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
```

**Update `_reset_idx`** — after `default_root_state[:, :3] += self.scene.env_origins[env_ids]` (line 272), add:

```python
# Record spawn XY for displacement penalty
self._initial_base_pos_xy[env_ids, 0] = default_root_state[:, 0]
self._initial_base_pos_xy[env_ids, 1] = default_root_state[:, 1]
```

**QA**: After reset, `self._initial_base_pos_xy` should contain the XY world position where each env's robot was spawned.

---

### Task 4: Add Two New Rewards — Origin Displacement + Arm Return

**File**: `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\r1_standing_env.py`

**What**: Add two new reward terms to `_get_rewards` and the `compute_rewards` function.

#### 4a. Pre-index arm tensors in `_get_rewards`

**Add after** `knee_pos_dev = joint_pos_dev[:, self._knee_joint_ids]` (around line 184):

```python
arm_pos_dev = joint_pos_dev[:, self._arm_ids]
```

#### 4b. Pass new data to `compute_rewards`

**Add to the `compute_rewards(...)` call** — the function signature must be extended with:
- Two new float params: `self.cfg.rew_scale_origin_displacement`, `self.cfg.rew_scale_arm_return`
- Two new tensor params: `self._initial_base_pos_xy`, `self.robot.data.root_pos_w[:, :2]` (current XY), `arm_pos_dev`

**NOTE**: The exact insertion point in the parameter list must match the order in the `@torch.jit.script` function. Add the new scales AFTER `rew_scale_bilateral_balance` and the new tensors AFTER `knee_pos_dev`.

#### 4c. Add to `compute_rewards` TorchScript function

**New parameters** (add to signature):

```python
rew_scale_origin_displacement: float,
rew_scale_arm_return: float,
# new tensors:
initial_base_pos_xy: torch.Tensor,
current_base_pos_xy: torch.Tensor,
arm_pos_dev: torch.Tensor,
```

**New reward blocks** (add before TOTAL REWARD section):

```python
# ================================================================
# ORIGIN DISPLACEMENT PENALTY (anti-circling)
# Penalizes XY distance from spawn position
# ================================================================

displacement = torch.norm(current_base_pos_xy - initial_base_pos_xy, dim=-1)
rew_origin_displacement = -rew_scale_origin_displacement * torch.square(displacement)

# ================================================================
# ARM RETURN TO DEFAULT (stronger than global return_to_default)
# Extra reward for arms being near their natural pose
# ================================================================

arm_deviation = torch.sum(torch.square(arm_pos_dev), dim=-1)
rew_arm_return = rew_scale_arm_return * torch.exp(-arm_deviation / 0.3)
```

**Add to total_reward sum**:
```python
+ rew_origin_displacement
+ rew_arm_return
```

**QA**: 
- `rew_origin_displacement` should be 0 when robot is at spawn, increasingly negative as it drifts
- `rew_arm_return` should be positive and maximum when arms are at default pose
- Verify compute_rewards parameter count matches `_get_rewards` call (currently 43 params, will become 48)

---

### Task 5: Add New Config Scales + Adjust Existing

**File**: `C:\space_r1\r1_standing\source\r1_standing\r1_standing\tasks\direct\r1_standing\r1_standing_env_cfg.py`

**Add** at the end of the file (after `rew_scale_bilateral_balance`):

```python
# Anti-circling: penalize displacement from spawn origin
rew_scale_origin_displacement = 10.0   # Strong — must stay at spawn

# Arm-specific return to default (on top of global return_to_default)
rew_scale_arm_return = 8.0             # Strong — arms must be natural
```

**Adjust existing scale**:

```python
# CHANGE: increase lin_vel penalty to further discourage movement
rew_scale_lin_vel = -0.08  # was -0.03 — stronger penalty for ANY linear velocity
```

**QA**: Verify file has no syntax errors. Count total reward scales — should be consistent with env.py.

---

### Task 6: Verification — Smoke Test Update

**File**: `C:\space_r1\r1_standing\.sisyphus\evidence\reward_smoke_test.py`

**What**: Update the existing smoke test to cover the 2 new rewards:
- Add test for `rew_origin_displacement`: when robot at spawn → 0 penalty; when displaced → negative
- Add test for `rew_arm_return`: when arms at default → max reward; when deviated → lower
- Update parameter count check (43 → 48)
- Update total reward summation test

**QA**: All existing 17 tests must still pass. New tests (at least 4: 2 per reward × 2 cases each) must pass.

---

### Task 7: Final Verification Wave

**Checklist**:
- [ ] R1_CFG `joint_pos` dict still has exactly 26 entries
- [ ] r1_standing_env.py `__init__` detects arm joints (10 total)
- [ ] r1_standing_env.py `_reset_idx` records initial XY position
- [ ] r1_standing_env.py `_get_rewards` passes all new params
- [ ] r1_standing_env.py `compute_rewards` has matching param count
- [ ] r1_standing_env_cfg.py has 2 new scales + 1 adjusted
- [ ] Smoke test passes (21+ tests, all green)
- [ ] `total_reward` sum includes both new terms
- [ ] No Python lists or dynamic indexing inside `@torch.jit.script`
- [ ] Arm defaults make anatomical sense: shoulder_pitch=0.2, elbow=0.4 → hands near waist

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| R1_CFG change affects r1_locomotion too | Both projects benefit from natural arm pose — not a conflict |
| origin_displacement scale too strong → robot freezes | Start at 10.0, can reduce if training shows zero exploration |
| Arm return conflicts with global return_to_default | They complement each other: global is weak (6.0 across 26 joints), arm-specific adds 8.0 for 10 arm joints specifically |
| Smoke test param count mismatch | Task 6 explicitly updates the count |

## Execution Order

1. Task 1 (R1_CFG) — no dependencies
2. Tasks 2-3 (env.py setup) — parallel
3. Task 4 (new rewards) — depends on 2, 3
4. Task 5 (cfg scales) — parallel with 4
5. Task 6 (smoke test) — after 4, 5
6. Task 7 (verification) — after 6
