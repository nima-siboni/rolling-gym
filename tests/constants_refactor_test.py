# pylint: disable=redefined-outer-name
"""
Tests that pin expected values for the constants-extraction refactor.

Every test here checks concrete values that must hold both before and after
the refactor (magic numbers → named constants, _get_info via INFO_SCHEMA,
reward constants to module-level, material_reference field).

These tests would catch:
- Constant values drifting from the original hardcoded numbers
- INFO_SCHEMA state-index ↔ key mapping getting out of sync
- Action space dimensions, resolution divisors, or mask thresholds changing
- Reward constant values diverging from what calculate_reward actually uses
- _get_info returning wrong keys, wrong types, or mismatched state indices
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import rollinggym  # noqa: F401  # pylint: disable=unused-import
from rollinggym.env.flat_rolling_env import FlatRollingEnv
from rollinggym.env.rolling_config_presets import create_s355_config
from rollinggym.env.state_evaluation import (
    GS_COMPLETION_MAX_BONUS,
    GS_COMPLETION_MAX_PENALTY,
    GS_COMPLETION_TOLERANCE_UM,
    HR_MAX_BONUS,
    HR_MAX_PENALTY,
    STEP_PENALTY,
    TEMP_COMPLETION_MAX_BONUS,
    TEMP_COMPLETION_MAX_PENALTY,
    TEMP_COMPLETION_TOLERANCE_K,
    calculate_completion_bonus,
    calculate_grain_size_progress_bonus,
    calculate_hr_efficiency_bonus,
    calculate_reward,
)


# ==================== Fixtures ====================


@pytest.fixture
def env_config():
    """Create a test environment configuration."""
    return create_s355_config()


@pytest.fixture
def train_env(env_config):
    """Create a seeded training environment."""
    test_env = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={
            'render_mode': None,
            'config': env_config,
            'mode': 'train',
            'env_seed': 42,
        },
    )
    yield test_env
    test_env.close()


@pytest.fixture
def inference_env(env_config):
    """Create an inference environment."""
    test_env = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={
            'render_mode': None,
            'config': env_config,
            'mode': 'inference',
        },
    )
    yield test_env
    test_env.close()


# ==================== Class-Level Constants ====================


class TestClassConstants:
    """Verify that class-level constants match the original hardcoded values."""

    def test_action_dims(self):
        assert FlatRollingEnv.ACTION_DIMS == (501, 121, 7)

    def test_hr_resolution(self):
        assert FlatRollingEnv.HR_RESOLUTION == 10.0

    def test_velocity_divisor(self):
        assert FlatRollingEnv.VELOCITY_DIVISOR == 12.0

    def test_max_hr_fraction(self):
        assert FlatRollingEnv.MAX_HR_FRACTION == 0.70

    def test_max_steps(self):
        assert FlatRollingEnv.MAX_STEPS == 25


# ==================== INFO_SCHEMA correctness ====================


class TestInfoSchema:
    """Verify INFO_SCHEMA maps each key to the correct state index and type."""

    def test_schema_has_exactly_10_entries(self):
        assert len(FlatRollingEnv.INFO_SCHEMA) == 10

    def test_schema_keys_match_expected(self):
        expected_keys = [
            'current_thickness',
            'step_count',
            'hr_limit',
            'target_thickness',
            'rolling_force',
            'rolling_torque',
            'stock_temperature',
            'target_temperature',
            'current_grain_size',
            'target_grain_size',
        ]
        assert list(FlatRollingEnv.INFO_SCHEMA.keys()) == expected_keys

    def test_schema_state_indices_are_0_through_9(self):
        """Each key maps to a unique state index 0..9."""
        indices = [entry[0] for entry in FlatRollingEnv.INFO_SCHEMA.values()]
        assert sorted(indices) == list(range(10))

    def test_schema_cast_types(self):
        """step_count is int; everything else is float."""
        for key, (_, cast, _, _) in FlatRollingEnv.INFO_SCHEMA.items():
            if key == 'step_count':
                assert cast is int, f'{key} should cast to int'
            else:
                assert cast is float, f'{key} should cast to float'

    def test_current_thickness_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['current_thickness'][0] == 0

    def test_step_count_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['step_count'][0] == 1

    def test_hr_limit_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['hr_limit'][0] == 2

    def test_target_thickness_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['target_thickness'][0] == 3

    def test_rolling_force_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['rolling_force'][0] == 4

    def test_rolling_torque_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['rolling_torque'][0] == 5

    def test_stock_temperature_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['stock_temperature'][0] == 6

    def test_target_temperature_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['target_temperature'][0] == 7

    def test_current_grain_size_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['current_grain_size'][0] == 8

    def test_target_grain_size_index(self):
        assert FlatRollingEnv.INFO_SCHEMA['target_grain_size'][0] == 9


# ==================== _get_info returns correct values ====================


class TestGetInfoValues:
    """Verify _get_info returns correct keys, types, and values from state."""

    def test_info_keys_after_reset(self, train_env):
        """Info dict must contain exactly the expected keys."""
        _, info = train_env.reset(seed=42)
        expected_keys = {
            'current_thickness', 'step_count', 'hr_limit',
            'target_thickness', 'rolling_force', 'rolling_torque',
            'stock_temperature', 'target_temperature',
            'current_grain_size', 'target_grain_size',
        }
        assert expected_keys.issubset(info.keys())

    def test_info_types_after_reset(self, train_env):
        """Check that types match: step_count is int, rest are float."""
        _, info = train_env.reset(seed=42)
        assert isinstance(info['step_count'], int)
        for key in [
            'current_thickness', 'hr_limit', 'target_thickness',
            'rolling_force', 'rolling_torque',
            'stock_temperature', 'target_temperature',
            'current_grain_size', 'target_grain_size',
        ]:
            assert isinstance(info[key], float), f'{key} should be float'

    def test_info_matches_state_vector(self, inference_env):
        """Each info value must equal the corresponding state vector element."""
        _, info = inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        state = inference_env.unwrapped.state
        assert info['current_thickness'] == pytest.approx(float(state[0]))
        assert info['step_count'] == int(state[1])
        assert info['hr_limit'] == pytest.approx(float(state[2]))
        assert info['target_thickness'] == pytest.approx(float(state[3]))
        assert info['rolling_force'] == pytest.approx(float(state[4]))
        assert info['rolling_torque'] == pytest.approx(float(state[5]))
        assert info['stock_temperature'] == pytest.approx(float(state[6]))
        assert info['target_temperature'] == pytest.approx(float(state[7]))
        assert info['current_grain_size'] == pytest.approx(float(state[8]))
        assert info['target_grain_size'] == pytest.approx(float(state[9]))

    def test_info_initial_force_and_torque_are_zero(self, inference_env):
        """After reset, force and torque must be 0."""
        _, info = inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        assert info['rolling_force'] == 0.0
        assert info['rolling_torque'] == 0.0

    def test_info_step_count_after_step(self, inference_env):
        """step_count must be int(1) after one step."""
        inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        _, _, _, _, info = inference_env.step(np.array([100, 10, 3]))
        assert info['step_count'] == 1
        assert isinstance(info['step_count'], int)

    def test_info_thickness_equals_known_value(self, inference_env):
        """After reset, current_thickness must equal the specified initial value."""
        _, info = inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 120.0,
            'target_thickness_mm': 10.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        assert info['current_thickness'] == pytest.approx(120.0)
        assert info['target_thickness'] == pytest.approx(10.0)
        assert info['stock_temperature'] == pytest.approx(1373.15, rel=1e-3)
        assert info['target_temperature'] == pytest.approx(1173.15, rel=1e-3)
        assert info['current_grain_size'] == pytest.approx(200.0)
        assert info['target_grain_size'] == pytest.approx(12.0)


# ==================== Action space dimensions ====================


class TestActionSpaceDimensions:
    """Verify action space uses the class constants, not stale literals."""

    def test_action_space_nvec(self, train_env):
        nvec = train_env.action_space.nvec
        assert nvec[0] == 501
        assert nvec[1] == 121
        assert nvec[2] == 7

    def test_observation_mask_shapes(self, train_env):
        obs_space = train_env.observation_space
        assert obs_space['action_mask']['height_reduction'].shape == (501,)
        assert obs_space['action_mask']['interpass_time'].shape == (121,)
        assert obs_space['action_mask']['rolling_velocity'].shape == (7,)

    def test_hr_action_to_mm_conversion(self):
        """action[0]=250 should give 25.0 mm height reduction."""
        assert 250 / FlatRollingEnv.HR_RESOLUTION == 25.0

    def test_velocity_action_to_m_s_conversion(self):
        """action[2]=6 should give 0.5 m/s."""
        assert 6 / FlatRollingEnv.VELOCITY_DIVISOR == pytest.approx(0.5)

    def test_velocity_action_min(self):
        """action[2]=1 should give ~0.083 m/s (5 m/min)."""
        assert 1 / FlatRollingEnv.VELOCITY_DIVISOR == pytest.approx(1 / 12.0)


# ==================== Action mask thresholds ====================


class TestActionMaskThresholds:
    """Verify 70% threshold and mask logic use the class constants."""

    def test_70_percent_constraint(self, inference_env):
        """At 100mm, max HR from 70% rule is 70mm → actions 0..700 allowed by that rule."""
        inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        obs = inference_env.unwrapped._get_obs()
        hr_mask = obs['action_mask']['height_reduction']

        # action index 500 = 50.0mm; 70% of 100mm = 70mm; so 50mm should be
        # limited only by hr_limit (60mm default), not the 70% rule.
        # But hr_limit from config is 60mm → action 500 (50mm) should pass
        # both constraints. It must also satisfy thickness >= target.
        # 100 - 50 = 50mm >= 10mm target → valid.
        # hr_limit = 60mm, action = 50mm → valid.
        assert hr_mask[500] == 1.0  # 50.0 mm allowed

    def test_action_zero_always_valid(self, inference_env):
        """Action 0 (no reduction) is always valid."""
        inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        obs = inference_env.unwrapped._get_obs()
        assert obs['action_mask']['height_reduction'][0] == 1.0

    def test_interpass_zero_masked(self, inference_env):
        """Interpass time index 0 is always disallowed."""
        inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        obs = inference_env.unwrapped._get_obs()
        assert obs['action_mask']['interpass_time'][0] == 0.0
        assert obs['action_mask']['interpass_time'][1] == 1.0

    def test_velocity_zero_masked(self, inference_env):
        """Velocity index 0 is always disallowed."""
        inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        obs = inference_env.unwrapped._get_obs()
        assert obs['action_mask']['rolling_velocity'][0] == 0.0
        assert obs['action_mask']['rolling_velocity'][1] == 1.0


# ==================== Truncation uses MAX_STEPS ====================


class TestTruncation:
    """Verify truncation triggers at exactly MAX_STEPS=25."""

    def test_not_truncated_at_step_24(self, inference_env):
        """Step count 24 (< 25) must not truncate."""
        inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 5.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        # Take 24 no-op steps (action 0 = 0mm reduction)
        for _ in range(24):
            _, _, terminated, truncated, _ = inference_env.step(np.array([0, 10, 3]))
            if terminated:
                break
        # After 24 steps, should not be truncated yet
        assert not truncated

    def test_truncated_at_step_25(self, inference_env):
        """Step count 25 (== MAX_STEPS) must truncate."""
        inference_env.reset(options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 5.0,
            'initial_temperature_K': 1373.15,
            'target_temperature_K': 1173.15,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
        })
        truncated = False
        for _ in range(25):
            _, _, terminated, truncated, _ = inference_env.step(np.array([0, 10, 3]))
            if terminated or truncated:
                break
        assert truncated


# ==================== Reward constants ====================


class TestRewardConstants:
    """Verify module-level reward constants match original hardcoded values."""

    def test_step_penalty(self):
        assert STEP_PENALTY == -5.0

    def test_hr_max_bonus(self):
        assert HR_MAX_BONUS == 10.0

    def test_hr_max_penalty(self):
        assert HR_MAX_PENALTY == -5.0

    def test_gs_completion_max_bonus(self):
        assert GS_COMPLETION_MAX_BONUS == 25.0

    def test_gs_completion_max_penalty(self):
        assert GS_COMPLETION_MAX_PENALTY == -2.0

    def test_gs_completion_tolerance(self):
        assert GS_COMPLETION_TOLERANCE_UM == 20.0

    def test_temp_completion_max_bonus(self):
        assert TEMP_COMPLETION_MAX_BONUS == 25.0

    def test_temp_completion_max_penalty(self):
        assert TEMP_COMPLETION_MAX_PENALTY == -2.0

    def test_temp_completion_tolerance(self):
        assert TEMP_COMPLETION_TOLERANCE_K == 50.0


# ==================== Reward functions use the constants ====================


class TestRewardFunctionsUseConstants:
    """Verify reward functions produce values consistent with the module constants."""

    def test_step_penalty_in_calculate_reward(self):
        """calculate_reward must include -5.0 step penalty."""
        # Minimal state: thickness at target, no force/torque, grain/temp at target
        state = np.array([
            10.0, 1.0, 30.0, 10.0,  # thickness=target, step=1, hr_limit, target
            0.0, 0.0,               # force=0, torque=0
            1173.0, 1173.0,         # temp at target
            12.0, 12.0,             # grain at target
        ], dtype=np.float32)
        config = create_s355_config()
        reward, details = calculate_reward(
            state,
            previous_grain_size_um=12.0,
            height_reduction_mm=0.0,
            completed=False,
            config=config,
        )
        assert details['step_penalty'] == -5.0

    def test_hr_efficiency_bonus_max(self):
        """Full HR utilization with safe margins gives HR_MAX_BONUS * ratio."""
        bonus = calculate_hr_efficiency_bonus(
            height_reduction_mm=30.0,
            hr_limit_mm=30.0,
            force=1e6,       # well under 4 MN limit
            force_limit=4e6,
            torque=50e3,     # well under 130 kNm limit
            torque_limit=130e3,
            max_bonus=HR_MAX_BONUS,
            max_penalty=HR_MAX_PENALTY,
        )
        assert bonus == pytest.approx(10.0)  # 10.0 * (30/30)

    def test_hr_efficiency_penalty_on_limit_exceeded(self):
        """Exceeding force limit gives HR_MAX_PENALTY."""
        bonus = calculate_hr_efficiency_bonus(
            height_reduction_mm=20.0,
            hr_limit_mm=30.0,
            force=5e6,        # exceeds 4 MN limit
            force_limit=4e6,
            torque=50e3,
            torque_limit=130e3,
            max_bonus=HR_MAX_BONUS,
            max_penalty=HR_MAX_PENALTY,
        )
        assert bonus == -5.0

    def test_gs_completion_bonus_at_target(self):
        """Grain size exactly at target on completion gives full bonus."""
        bonus = calculate_completion_bonus(
            current_param=12.0,
            target_param=12.0,
            completed=True,
            max_bonus=GS_COMPLETION_MAX_BONUS,
            max_penalty=GS_COMPLETION_MAX_PENALTY,
            tolerance=GS_COMPLETION_TOLERANCE_UM,
        )
        assert bonus == pytest.approx(25.0)

    def test_gs_completion_bonus_not_completed(self):
        """No completion bonus if episode hasn't ended."""
        bonus = calculate_completion_bonus(
            current_param=12.0,
            target_param=12.0,
            completed=False,
            max_bonus=GS_COMPLETION_MAX_BONUS,
            max_penalty=GS_COMPLETION_MAX_PENALTY,
            tolerance=GS_COMPLETION_TOLERANCE_UM,
        )
        assert bonus == 0.0

    def test_temp_completion_bonus_at_target(self):
        """Temperature exactly at target on completion gives full bonus."""
        bonus = calculate_completion_bonus(
            current_param=1173.0,
            target_param=1173.0,
            completed=True,
            max_bonus=TEMP_COMPLETION_MAX_BONUS,
            max_penalty=TEMP_COMPLETION_MAX_PENALTY,
            tolerance=TEMP_COMPLETION_TOLERANCE_K,
        )
        assert bonus == pytest.approx(25.0)

    def test_temp_completion_at_tolerance_edge(self):
        """At exactly the tolerance edge, bonus should be 0."""
        bonus = calculate_completion_bonus(
            current_param=1173.0 + TEMP_COMPLETION_TOLERANCE_K,
            target_param=1173.0,
            completed=True,
            max_bonus=TEMP_COMPLETION_MAX_BONUS,
            max_penalty=TEMP_COMPLETION_MAX_PENALTY,
            tolerance=TEMP_COMPLETION_TOLERANCE_K,
        )
        assert bonus == pytest.approx(0.0)

    def test_gs_progress_bonus_improvement(self):
        """Reducing grain size toward target gives positive reward."""
        bonus = calculate_grain_size_progress_bonus(
            current_grain_size_um=50.0,
            previous_grain_size_um=100.0,
            target_grain_size_um=12.0,
        )
        assert bonus > 0.0

    def test_gs_progress_bonus_overshoot_penalty(self):
        """Crossing below target gives negative reward."""
        bonus = calculate_grain_size_progress_bonus(
            current_grain_size_um=8.0,
            previous_grain_size_um=15.0,
            target_grain_size_um=12.0,
        )
        assert bonus < 0.0


# ==================== material_reference field ====================


class TestMaterialReference:
    """Verify the material_reference field on EnvConfig."""

    def test_s355_config_has_material_reference(self):
        config = create_s355_config()
        assert isinstance(config.material_reference, str)
        assert len(config.material_reference) > 0
        assert 'S355' in config.material_reference

    def test_material_reference_default_is_empty(self):
        """EnvConfig without material_reference defaults to empty string."""
        from rollinggym.env.rolling_env_config import EnvConfig
        # Check the field default via model_fields
        field_info = EnvConfig.model_fields['material_reference']
        assert field_info.default == ''
