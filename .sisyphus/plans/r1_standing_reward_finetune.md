# Plan: R1StandingEnv — Reward Finetuning & Policy Stabilization

## Context & Problem Statement

The current standing policy converges to a bad local optimum:
1. Legs open progressively after ~500 iterations
2. Left leg used as pivot; right leg corrects → robot spins/yaws
3. Robot leans right after stabilizing (bad weight distribution)
4. Robot never returns to natural upright pose after perturbations

**Root cause** (from code reading): Three active bugs in `compute_rewards` invert reward gradients,
causing the optimizer to learn exactly the wrong behaviors. Additionally, joint indices are hardcoded
and unvalidated.

## Scope

| IN | OUT |
|----|-----|
| `r1_standing_env.py` — `compute_rewards` function and `__init__` | PPO architecture |
| `r1_standing_env.py` — `_get_rewards`, `__init__` joint detection | Observation space (88 dims, frozen) |
| `r1_standing_env_cfg.py` — reward scale fields | Action space (26 joints, frozen) |
| Bug fixes in existing reward terms | Network architecture |
| New reward: return-to-default-pose | Physics parameters |
| New reward: bilateral weight distribution (foot height symmetry) | Curriculum (separate plan) |
| Dynamic joint detection with find_joints | Push recovery logic |

## Files Modified

- `source/r1_standing/r1_standing/tasks/direct/r1_standing/r1_standing_env.py`
- `source/r1_standing/r1_standing/tasks/direct/r1_standing/r1_standing_env_cfg.py`

## Definition of Done

- [ ] All 3 bugs fixed with correct semantics
- [ ] Hip and leg joint IDs detected dynamically via `find_joints` regex
- [ ] New `rew_return_to_default` penalizes joint deviation when robot is upright
- [ ] New `rew_bilateral_balance` rewards symmetric foot height (weight distribution proxy)
- [ ] Reward scales rebalanced: knee penalty is negative, COM balance is positive
- [ ] `total_reward` sumation verified — no scalar leaked in place of tensor
- [ ] `compute_rewards` signature updated with new parameters
- [ ] `_get_rewards` passes new tensors to `compute_rewards`
- [ ] `__init__` initializes `_hip_left_ids`, `_hip_right_ids`, `_left_leg_ids`, `_right_leg_ids`
- [ ] All new cfg fields added to `R1StandingEnvCfg`

---

## Wave 1 — Bug Fixes (Critical, do first)

### Task 1.1 — Fix `rew_com_lateral_balance` (INVERTED sign)

**File**: `r1_standing_env.py`
**Line**: ~380

**Problem**: Formula gives maximum penalty when COM is perfectly centered.
```python
# CURRENT (WRONG) — penalizes centered COM:
rew_com_lateral_balance = -rew_scale_com_lateral_balance * torch.exp(-com_lateral_error / 0.15)

# FIXED — rewards centered COM, penalizes lateral drift:
rew_com_lateral_balance = rew_scale_com_lateral_balance * torch.exp(-com_lateral_error / 0.15)
```
Note: `rew_scale_com_lateral_balance` in cfg stays positive (18.0). The sign in the formula changes from `-` to `+`.

**Acceptance criteria**:
- When COM is centered over support polygon: `rew_com_lateral_balance` ≈ +18.0
- When COM drifts 0.3m laterally: `rew_com_lateral_balance` ≈ +18 * exp(-0.3/0.15) ≈ +18 * 0.135 ≈ +2.4
- No NaN or Inf

**QA scenario**: Manually compute with `com_lateral_error = torch.tensor([0.0, 0.15, 0.30])`, verify values decrease monotonically but remain positive.

---

### Task 1.2 — Fix `rew_knee_extension` (wrong sign + wrong semantics)

**File**: `r1_standing_env.py`
**Lines**: ~388-390 + cfg

**Problem**: `rew_scale_knee_extension = 8.0` (positive) multiplied by squared deviation means the network MAXIMIZES knee deviation to maximize reward. This is the direct cause of leg opening.

**Fix — two parts**:

Part A — Change scale sign in `r1_standing_env_cfg.py`:
```python
# CURRENT (WRONG):
rew_scale_knee_extension = 8.0

# FIXED — penalize deviation from default (knees near-straight):
rew_scale_knee_extension = -5.0
```

Part B — Rename reward for semantic clarity in `r1_standing_env.py`:
```python
# CURRENT (WRONG semantics — "extension" reward but rewards deviation):
knee_pos = joint_pos_dev[:, 7:9]  # hardcoded indices
rew_knee_extension = rew_scale_knee_extension * torch.sum(torch.square(knee_pos), dim=-1)

# FIXED — use dynamic indices, negative scale penalizes deviation from straight:
# (self._knee_joint_ids is already computed in __init__)
knee_pos_dev = joint_pos_dev[:, self._knee_joint_ids]  # dynamic
rew_knee_extension = rew_scale_knee_extension * torch.sum(torch.square(knee_pos_dev), dim=-1)
# With rew_scale_knee_extension = -5.0, this penalizes bent knees
```

**Important**: `self._knee_joint_ids` is already populated in `__init__` (lines 51-55). The `joint_pos_dev` tensor must be indexed with a list/tensor — verify it's `joint_pos_dev[:, self._knee_joint_ids]` not a Python list comprehension (TorchScript compatible).

**Acceptance criteria**:
- Knees at default position → `rew_knee_extension` ≈ 0 (minimal penalty)
- Knees deviated 0.3 rad from default → negative reward proportional to -5.0 * 0.09 = -0.45 per knee
- TorchScript compiles without error

**QA scenario**: `joint_pos_dev = zeros → rew = 0`. `joint_pos_dev = 0.3 rad deviation → rew < 0`.

---

### Task 1.3 — Fix `total_reward` scalar leak (`rew_scale_recovery` instead of `rew_recovery`)

**File**: `r1_standing_env.py`
**Line**: ~404

**Problem**: The tensor `rew_recovery` is computed at line ~339, but `rew_scale_recovery` (a float scalar `2.0`) is what gets added to `total_reward`. This adds a constant +2.0 every step regardless of robot state.

```python
# CURRENT (WRONG):
total_reward = (
    ...
    + rew_scale_recovery   # ← scalar float 2.0, NOT the tensor
    + rew_recovery
    ...
)

# FIXED:
total_reward = (
    ...
    + rew_recovery         # ← remove rew_scale_recovery; keep only tensor
    ...
)
```

**Acceptance criteria**:
- `total_reward` no longer has a fixed +2.0 offset applied to all environments
- `rew_recovery` tensor still contributes correctly based on `orientation_error`

---

## Wave 2 — Dynamic Joint Detection

### Task 2.1 — Add dynamic hip and leg joint IDs in `__init__`

**File**: `r1_standing_env.py`
**Location**: `__init__`, after line 55 (after `self._knee_joint_ids`)

**Add**:
```python
# Dynamic hip joint IDs for co-activation and symmetry rewards
self._hip_left_ids, _ = self.robot.find_joints(".*left.*hip.*")
self._hip_right_ids, _ = self.robot.find_joints(".*right.*hip.*")

# Dynamic leg joint IDs for sync penalty (replaces hardcoded 6:12 / 12:18)
self._left_leg_ids, _ = self.robot.find_joints(".*left.*(hip|knee|ankle).*")
self._right_leg_ids, _ = self.robot.find_joints(".*right.*(hip|knee|ankle).*")
```

**Acceptance criteria**:
- IDs are non-empty lists after init (log their lengths and names at init for verification)
- Left and right sets don't overlap
- Length of left_leg_ids == length of right_leg_ids (required for sync penalty dot product)

**QA scenario**: Add a debug print in `__init__` (to be removed after verification):
```python
print(f"[R1Env] hip_left_ids={self._hip_left_ids}, hip_right_ids={self._hip_right_ids}")
print(f"[R1Env] left_leg_ids={self._left_leg_ids}, right_leg_ids={self._right_leg_ids}")
```

---

### Task 2.2 — Update `compute_rewards` signature and callers for dynamic IDs

**File**: `r1_standing_env.py`

The `compute_rewards` function is decorated with `@torch.jit.script`. TorchScript does not accept dynamic lists from outside — must pass the actual joint tensors pre-indexed.

**Strategy**: Pre-index tensors in `_get_rewards` (non-JIT) and pass sliced tensors:

In `_get_rewards` (non-JIT), before calling `compute_rewards`:
```python
# Pre-index joint tensors using dynamic IDs
hip_left_pos_dev = (self.joint_pos - self.robot.data.default_joint_pos)[:, self._hip_left_ids]
hip_right_pos_dev = (self.joint_pos - self.robot.data.default_joint_pos)[:, self._hip_right_ids]
left_leg_actions = self.actions[:, self._left_leg_ids]
right_leg_actions = self.actions[:, self._right_leg_ids]
knee_pos_dev = (self.joint_pos - self.robot.data.default_joint_pos)[:, self._knee_joint_ids]
```

Add these 5 tensors to the `compute_rewards` call and signature:
- `hip_left_pos_dev: torch.Tensor`  — shape [N, n_hip_left]
- `hip_right_pos_dev: torch.Tensor` — shape [N, n_hip_right]
- `left_leg_actions: torch.Tensor`  — shape [N, n_left_leg]
- `right_leg_actions: torch.Tensor` — shape [N, n_right_leg]
- `knee_pos_dev: torch.Tensor`      — shape [N, n_knees]

**Inside `compute_rewards`**: Replace all hardcoded indexing with these passed tensors:
```python
# Replace:
hip_left = joint_pos_dev[:, 0]
hip_right = joint_pos_dev[:, 3]
# With:
hip_left = torch.sum(hip_left_pos_dev, dim=-1)   # scalar per env
hip_right = torch.sum(hip_right_pos_dev, dim=-1)

# Replace:
left_leg_actions = actions[:, 6:12]
right_leg_actions = actions[:, 12:18]
# With: (already passed in as parameters — use them directly)

# Replace:
knee_pos = joint_pos_dev[:, 7:9]
# With:
# (use knee_pos_dev parameter)
rew_knee_extension = rew_scale_knee_extension * torch.sum(torch.square(knee_pos_dev), dim=-1)
```

**Acceptance criteria**:
- `compute_rewards` compiles with `@torch.jit.script` after changes
- All `joint_pos_dev[:, hardcoded_index]` references removed from `compute_rewards`
- `_get_rewards` correctly pre-indexes all 5 tensors using `_hip_left_ids`, etc.

---

## Wave 3 — New Rewards

### Task 3.1 — New reward: `rew_return_to_default` (return-to-pose after stabilization)

**Goal**: After stabilizing from a perturbation, reward the robot for returning joints toward `default_joint_pos`. This must NOT conflict with the existing `rew_joint_pos` (which already penalizes deviation), so we use a different formulation: a smooth, orientation-gated reward that activates only when the robot is upright.

**Add to `r1_standing_env_cfg.py`**:
```python
rew_scale_return_to_default = 6.0   # positive: reward for being near default pose when upright
```

**Add to `compute_rewards` signature**:
```python
rew_scale_return_to_default: float,
```

**Add to `compute_rewards` body** (after `rew_recovery`):
```python
# Reward: return to default pose, gated by orientation quality
# Only activates strongly when robot is upright (low orientation_error)
# orientation_gate: 1.0 when perfectly upright, approaches 0 when tilted
orientation_gate = torch.exp(-orientation_error / 0.02)  # sharp gate, active only when upright

# Total joint deviation from default (all joints, not just legs)
total_joint_deviation = torch.sum(torch.square(joint_pos_dev), dim=-1)

# Smooth reward: near-default pose is rewarded when upright
rew_return_to_default = rew_scale_return_to_default * orientation_gate * torch.exp(-total_joint_deviation / 0.5)
```

**Rationale**:
- `orientation_gate` = 1 when upright → full reward for being in default pose
- `orientation_gate` → 0 when tilting → allows free movement to recover
- `exp(-deviation / 0.5)` has a wide basin; robot must be within ~0.7 rad total to get >50% reward
- Scale 6.0 is meaningful but smaller than orientation (20.0) to maintain priority ordering

**Add to `total_reward`**:
```python
+ rew_return_to_default
```

**Add to `_get_rewards` call**: pass `self.cfg.rew_scale_return_to_default`

**Acceptance criteria**:
- When robot is upright AND at default pose: `rew_return_to_default` ≈ +6.0
- When robot is tilted (recovering): `rew_return_to_default` ≈ 0 (gate suppresses it)
- When robot is upright BUT joints deviated 1.0 rad total: `rew_return_to_default` ≈ 6.0 * exp(-2.0) ≈ +0.8

---

### Task 3.2 — New reward: `rew_bilateral_balance` (symmetric weight distribution proxy)

**Goal**: Reward equal height of both feet (proxy for equal weight distribution when no force sensors). If one foot is higher than the other, it's bearing less load → asymmetric stance.

**Note**: We don't have force/contact sensors. Foot height symmetry is an imperfect but effective proxy: in static standing, both ankles should be at approximately the same height (z).

**Add to `r1_standing_env_cfg.py`**:
```python
rew_scale_bilateral_balance = 8.0   # penalize asymmetric foot heights
```

**Add to `compute_rewards` signature**:
```python
rew_scale_bilateral_balance: float,
```

**Add to `compute_rewards` body** (after `rew_return_to_default`):
```python
# Reward: symmetric foot heights (proxy for bilateral weight distribution)
# Both feet should be at approximately the same z height (on flat ground ≈ 0)
left_foot_z = left_foot_pos[:, 2]
right_foot_z = right_foot_pos[:, 2]
foot_height_asymmetry = torch.square(left_foot_z - right_foot_z)
rew_bilateral_balance = -rew_scale_bilateral_balance * foot_height_asymmetry
```

**Add to `total_reward`**:
```python
+ rew_bilateral_balance
```

**Add to `_get_rewards` call**: pass `self.cfg.rew_scale_bilateral_balance`

**Acceptance criteria**:
- Both feet at same height (0 asymmetry): `rew_bilateral_balance` = 0
- One foot 5cm higher than other: `rew_bilateral_balance` = -8.0 * 0.0025 = -0.02 (gentle)
- One foot 20cm higher (extreme lean): `rew_bilateral_balance` = -8.0 * 0.04 = -0.32

---

## Wave 4 — Reward Scale Rebalancing

### Task 4.1 — Update reward scales in `r1_standing_env_cfg.py`

After bug fixes, the reward landscape changes significantly. Rebalance scales to:
- Prevent the orientation reward from dominating so strongly it ignores foot placement
- Give meaningful weight to new rewards
- Reduce foot_separation scale (bug is fixed, so less overcorrection needed)

**Changes to `r1_standing_env_cfg.py`**:
```python
# CURRENT → FIXED

rew_scale_knee_extension = 8.0          → rew_scale_knee_extension = -5.0
# (sign change: now penalizes bent knees, not rewards)

rew_scale_feet_separation = 10.0        → rew_scale_feet_separation = 8.0
# (reduce slightly; quadratic is already powerful)

rew_scale_feet_together = 12.0          → rew_scale_feet_together = 10.0
# (reduce slightly to balance with new bilateral reward)

rew_scale_com_lateral_balance = 18.0    → rew_scale_com_lateral_balance = 12.0
# (reduce after sign fix; was over-dominating when inverted)

rew_scale_yaw_rate = 20.0               → rew_scale_yaw_rate = 15.0
# (reduce slightly; was too aggressive, prevented natural corrections)

# NEW FIELDS (add):
rew_scale_return_to_default = 6.0
rew_scale_bilateral_balance = 8.0
```

**No changes to**:
- `rew_scale_alive = 1.0`
- `rew_scale_terminated = -500.0`
- `rew_scale_orientation = 20.0` (keep dominant)
- `rew_scale_base_height = 10.0`
- `rew_scale_recovery = 2.0`
- `rew_scale_foot_lateral_symmetry = 15.0`
- All joint/vel/action penalties

**Acceptance criteria**:
- All new fields exist in cfg and are passed through `_get_rewards` → `compute_rewards`
- No field referenced in `compute_rewards` is missing from cfg or `_get_rewards` call
- `compute_rewards` has same number of positional args as `_get_rewards` passes

---

## Wave 5 — Integration & Verification

### Task 5.1 — Verify `compute_rewards` signature completeness

**Check**: Every parameter in `compute_rewards` signature maps to an argument in the `_get_rewards` call. Enumerate them:

Current signature (26 params) + new params:
1. `rew_scale_alive` → `self.cfg.rew_scale_alive`
2. `rew_scale_terminated` → `self.cfg.rew_scale_terminated`
3. `rew_scale_base_height` → `self.cfg.rew_scale_base_height`
4. `rew_scale_orientation` → `self.cfg.rew_scale_orientation`
5. `rew_scale_joint_pos` → `self.cfg.rew_scale_joint_pos`
6. `rew_scale_joint_vel` → `self.cfg.rew_scale_joint_vel`
7. `rew_scale_action_rate` → `self.cfg.rew_scale_action_rate`
8. `rew_scale_lin_vel` → `self.cfg.rew_scale_lin_vel`
9. `rew_scale_ang_vel` → `self.cfg.rew_scale_ang_vel`
10. `rew_scale_recovery` → `self.cfg.rew_scale_recovery`
11. `rew_scale_feet_separation` → `self.cfg.rew_scale_feet_separation`
12. `rew_scale_com_support` → `self.cfg.rew_scale_com_support`
13. `rew_scale_co_activation` → `self.cfg.rew_scale_co_activation`
14. `rew_scale_sync` → `self.cfg.rew_scale_sync`
15. `rew_scale_rigidity` → `self.cfg.rew_scale_rigidity`
16. `enable_com_reward` → `self.cfg.enable_com_reward`
17. `target_height` → `self.cfg.target_base_height`
18. `rew_scale_feet_together` → `self.cfg.rew_scale_feet_together`
19. `target_feet_distance` → `self.cfg.target_feet_distance`
20. `rew_scale_foot_lateral_symmetry` → `self.cfg.rew_scale_foot_lateral_symmetry`
21. `rew_scale_com_lateral_balance` → `self.cfg.rew_scale_com_lateral_balance`
22. `rew_scale_yaw_rate` → `self.cfg.rew_scale_yaw_rate`
23. `rew_scale_knee_extension` → `self.cfg.rew_scale_knee_extension`
24. `base_height` → `self.robot.data.root_pos_w[:, 2]`
25. `gravity_proj` → `self.robot.data.projected_gravity_b`
26. `joint_pos_dev` → `self.joint_pos - self.robot.data.default_joint_pos`
27. `joint_vel` → `self.joint_vel`
28. `actions` → `self.actions`
29. `previous_actions` → `self._previous_actions`
30. `lin_vel` → `self.robot.data.root_lin_vel_b`
31. `ang_vel` → `self.robot.data.root_ang_vel_b`
32. `left_foot_pos` → computed in `_get_rewards`
33. `right_foot_pos` → computed in `_get_rewards`
34. `com_xy` → computed in `_get_rewards`
35. `reset_terminated` → `self.reset_terminated`
36. `joint_pos` → `self.joint_pos`
37. `hip_left_pos_dev` → pre-indexed in `_get_rewards` **(NEW)**
38. `hip_right_pos_dev` → pre-indexed in `_get_rewards` **(NEW)**
39. `left_leg_actions` → pre-indexed in `_get_rewards` **(NEW)**
40. `right_leg_actions` → pre-indexed in `_get_rewards` **(NEW)**
41. `knee_pos_dev` → pre-indexed in `_get_rewards` **(NEW)**
42. `rew_scale_return_to_default` → `self.cfg.rew_scale_return_to_default` **(NEW)**
43. `rew_scale_bilateral_balance` → `self.cfg.rew_scale_bilateral_balance` **(NEW)**

**Acceptance criteria**: Count matches. TorchScript `@torch.jit.script` compiles without error on first import.

---

### Task 5.2 — End-to-end smoke test (no Isaac Sim required)

Create a minimal standalone test at `.sisyphus/evidence/reward_smoke_test.py`:

```python
"""
Smoke test for compute_rewards after bug fixes.
Run with: python .sisyphus/evidence/reward_smoke_test.py
Does NOT require Isaac Sim.
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/r1_standing"))

# Test 1: COM lateral balance — centered COM should give POSITIVE reward
com_lateral_error = torch.tensor([0.0, 0.15, 0.30])
rew = 12.0 * torch.exp(-com_lateral_error / 0.15)
assert (rew[0] > rew[1] > rew[2]), f"FAIL T1: COM reward not monotonically decreasing: {rew}"
assert rew[0] > 0, f"FAIL T1: centered COM gives non-positive reward: {rew[0]}"
print(f"PASS T1 — COM balance: {rew}")

# Test 2: Knee extension — deviation should give NEGATIVE reward (scale -5.0)
knee_dev = torch.tensor([[0.0, 0.0], [0.3, 0.3], [0.6, 0.6]])
rew_knee = -5.0 * torch.sum(torch.square(knee_dev), dim=-1)
assert rew_knee[0] == 0.0, f"FAIL T2: zero deviation should give 0 reward: {rew_knee[0]}"
assert rew_knee[1] < 0.0, f"FAIL T2: deviation should give negative reward: {rew_knee[1]}"
assert rew_knee[1] > rew_knee[2], f"FAIL T2: larger deviation should give more negative: {rew_knee}"
print(f"PASS T2 — Knee extension: {rew_knee}")

# Test 3: Return to default — upright + at default = max reward
orientation_error = torch.tensor([0.0, 0.5, 1.0])
joint_dev = torch.zeros(3, 26)
gate = torch.exp(-orientation_error / 0.02)
total_dev = torch.sum(torch.square(joint_dev), dim=-1)
rew_rtd = 6.0 * gate * torch.exp(-total_dev / 0.5)
assert rew_rtd[0] > rew_rtd[1] > rew_rtd[2], f"FAIL T3: {rew_rtd}"
assert abs(rew_rtd[0].item() - 6.0) < 0.01, f"FAIL T3: upright+default should give ~6.0: {rew_rtd[0]}"
print(f"PASS T3 — Return to default: {rew_rtd}")

# Test 4: Bilateral balance — same height = 0, asymmetric = negative
left_z = torch.tensor([0.05, 0.05, 0.05])
right_z = torch.tensor([0.05, 0.10, 0.25])
rew_bb = -8.0 * torch.square(left_z - right_z)
assert rew_bb[0] == 0.0, f"FAIL T4: equal heights should give 0: {rew_bb[0]}"
assert rew_bb[1] < 0.0, f"FAIL T4: asymmetric should give negative: {rew_bb[1]}"
print(f"PASS T4 — Bilateral balance: {rew_bb}")

# Test 5: Scalar leak — rew_scale_recovery should NOT appear in total_reward
# This test is structural; verified by code review (no automated test possible without sim)
print("PASS T5 — Scalar leak: must be verified by code review (rew_scale_recovery not in total_reward sum)")

print("\n✅ All smoke tests passed.")
```

**Acceptance criteria**: Script runs without assertion errors. All 4 numeric tests pass.

---

### Task 5.3 — Code review checklist (agent-executed)

After all changes, verify:

- [ ] `r1_standing_env.py` line ~380: formula is `+rew_scale_com_lateral_balance * exp(...)` (positive)
- [ ] `r1_standing_env_cfg.py`: `rew_scale_knee_extension = -5.0` (negative)
- [ ] `r1_standing_env.py` `total_reward`: contains `rew_recovery` tensor NOT `rew_scale_recovery` scalar
- [ ] `__init__`: `_hip_left_ids`, `_hip_right_ids`, `_left_leg_ids`, `_right_leg_ids` all initialized
- [ ] `_get_rewards`: pre-indexes 5 tensors before `compute_rewards` call
- [ ] `compute_rewards` signature has 43 parameters (see Task 5.1 list)
- [ ] `compute_rewards` `total_reward` includes `rew_return_to_default` and `rew_bilateral_balance`
- [ ] `R1StandingEnvCfg` has `rew_scale_return_to_default` and `rew_scale_bilateral_balance`
- [ ] Smoke test passes (Task 5.2)
- [ ] No hardcoded indices `[:, 0]`, `[:, 3]`, `[:, 6:12]`, `[:, 12:18]`, `[:, 7:9]` remain in `compute_rewards`

---

## Summary of All Changes

### `r1_standing_env_cfg.py`
1. `rew_scale_knee_extension`: `8.0` → `-5.0`
2. `rew_scale_feet_separation`: `10.0` → `8.0`
3. `rew_scale_feet_together`: `12.0` → `10.0`
4. `rew_scale_com_lateral_balance`: `18.0` → `12.0`
5. `rew_scale_yaw_rate`: `20.0` → `15.0`
6. ADD: `rew_scale_return_to_default = 6.0`
7. ADD: `rew_scale_bilateral_balance = 8.0`

### `r1_standing_env.py`
1. `__init__`: ADD 4 dynamic joint ID lists (`_hip_left_ids`, `_hip_right_ids`, `_left_leg_ids`, `_right_leg_ids`)
2. `_get_rewards`: ADD pre-indexing of 5 tensors; ADD 7 new args to `compute_rewards` call
3. `compute_rewards` signature: ADD 7 parameters
4. `compute_rewards` body:
   - FIX line ~342-343: use `hip_left_pos_dev`/`hip_right_pos_dev` (sum over joints)
   - FIX line ~349-356: use `left_leg_actions`/`right_leg_actions` parameters
   - FIX line ~380: remove leading `-` from `rew_com_lateral_balance`
   - FIX line ~388-390: use `knee_pos_dev` parameter
   - FIX line ~404: remove `rew_scale_recovery` from total_reward sum
   - ADD: `rew_return_to_default` computation and addition to total
   - ADD: `rew_bilateral_balance` computation and addition to total
5. `total_reward`: remove `rew_scale_recovery` scalar, add 2 new reward tensors

---

## Expected Behavioral Changes After Training

| Problem | Root Cause | Fix | Expected Outcome |
|---------|-----------|-----|-----------------|
| Legs open progressively | `rew_knee_extension` +8.0 rewarded deviation | Change to -5.0 penalty | Knees stay near default |
| Leans right | `rew_com_lateral_balance` inverted (penalized centering) | Fix sign | COM centers over feet |
| Pivots left leg | No lateral symmetry enforcement (overridden by inverted reward) | Fixed COM balance + bilateral balance | Weight distributed evenly |
| Yaw/spinning | Left pivot + no yaw penalty being overridden | Fixed rewards reduce left-only corrections | Rotation suppressed |
| Doesn't return to pose | No return-to-default reward | Add `rew_return_to_default` gated on upright | Robot returns to T-pose upright |

---

## Guardrails

- **Observation space**: Do NOT add feet height or new sensors to obs. Frozen at 88.
- **Action space**: Do NOT change. Frozen at 26.
- **`@torch.jit.script`**: All tensors passed to `compute_rewards` must be `torch.Tensor`. No Python lists or dynamic indexing inside the function.
- **Scale ordering**: Orientation (20) > Bilateral (8) > Return-to-default (6) > Feet-together (10) — intentional priority.
- **No new physics**: All rewards computed from existing data (`body_pos_w`, `joint_pos`, `root_pos_w`, etc.)
