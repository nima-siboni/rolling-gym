"""
Pytest tests for action masking functionality.

Tests that the action mask correctly enforces three constraints:
1. No action above hr_lim
2. No height reduction beyond 70% of current thickness
3. No action that brings thickness below target
"""
# pylint: disable=redefined-outer-name
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import rollinggym  # noqa: F401  # pylint: disable=unused-import
from rollinggym.env.rolling_config_presets import create_s355_config


@pytest.fixture
def env():
  """Create a test environment."""
  test_env = gym.make(
      'rollinggym/FlatRolling-v0.0',
      env_config={
          'render_mode': None,
          'config': create_s355_config(),
          'mode': 'inference',
      },
  )
  yield test_env
  test_env.close()


class TestActionMasking:
  """Test suite for action masking functionality."""

  def test_standard_scenario(self, env): # pylint: disable=too-many-locals
    """Test standard scenario: 100mm -> 10mm, hr_limit=30mm.

    Note: observations are normalized, so we use the info dict for raw values.
    """
    obs, info = env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'hr_limit_mm': 30.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    action_mask = obs['action_mask']

    # Find max valid action (using height_reduction mask)
    hr_mask = action_mask['height_reduction']
    valid_actions = np.where(hr_mask > 0.5)[0]
    max_valid_action = valid_actions.max() if len(valid_actions) > 0 else 0
    max_valid_hr = max_valid_action / 10.0

    # Use raw values from info dict (observations are normalized)
    current_thickness = info['current_thickness']
    hr_limit = info['hr_limit']
    target_thickness = info['target_thickness']

    # Constraint 1: hr <= hr_limit (30mm)
    constraint_1_max = hr_limit
    # Constraint 2: hr <= 70% of thickness (70mm)
    constraint_2_max = current_thickness * 0.70
    # Constraint 3: thickness - hr >= target (90mm)
    constraint_3_max = current_thickness - target_thickness

    expected_max = min(constraint_1_max, constraint_2_max, constraint_3_max)

    assert abs(max_valid_hr - expected_max) < 0.2, \
        f'Action mask incorrect! Expected {expected_max}, got {max_valid_hr}'

  def test_near_target_scenario(self, env):
    """Test near target: 12mm -> 10mm, hr_limit=30mm (limited actions)."""
    obs, _ = env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 12.0,
            'target_thickness_mm': 10.0,
            'hr_limit_mm': 30.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    action_mask = obs['action_mask']
    hr_mask = action_mask['height_reduction']
    valid_actions = np.where(hr_mask > 0.5)[0]
    max_valid_action = valid_actions.max() if len(valid_actions) > 0 else 0
    max_valid_hr = max_valid_action / 10.0

    # Should be limited by target constraint: 12 - 10 = 2mm
    expected_max = min(30.0, 12.0 * 0.70, 2.0)  # Should be 2.0mm

    assert abs(max_valid_hr - expected_max) < 0.2, \
        f'Action mask incorrect! Expected {expected_max}, got {max_valid_hr}'

  def test_small_hr_limit(self, env):
    """Test with small hr_limit: 100mm -> 10mm, hr_limit=10mm."""
    obs, _ = env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'hr_limit_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    action_mask = obs['action_mask']
    hr_mask = action_mask['height_reduction']
    valid_actions = np.where(hr_mask > 0.5)[0]
    max_valid_action = valid_actions.max() if len(valid_actions) > 0 else 0
    max_valid_hr = max_valid_action / 10.0

    # Should be limited by hr_limit constraint: 10mm
    expected_max = 10.0

    assert abs(max_valid_hr - expected_max) < 0.2, \
        f'Action mask incorrect! Expected {expected_max}, got {max_valid_hr}'

  def test_dynamic_mask_update_after_step(self, env):
    """Test that action mask updates correctly after taking a step."""
    obs, _ = env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'hr_limit_mm': 30.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    initial_thickness = obs['observations'][0]

    # Take action: reduce by 30mm (action[0] = 300), interpass time = 1s (action[1] = 1),
    # velocity = 2
    obs, _, _, _, _ = env.step([300, 1, 2])

    after_thickness = obs['observations'][0]
    after_valid_count = obs['action_mask']['height_reduction'].sum()

    # Verify thickness decreased
    assert after_thickness < initial_thickness, \
        f'Thickness should decrease: {initial_thickness} -> {after_thickness}'

    # Verify mask was recalculated (valid actions may have changed)
    assert after_valid_count > 0, 'Should have at least some valid actions'

  def test_action_zero_always_valid(self, env):
    """Test that action 0 (no reduction) is always valid."""
    base_params = {
        'initial_grain_size_um': 200.0,
        'target_grain_size_um': 12.0,
        'initial_temperature_K': 1373.0,
        'target_temperature_K': 1173.0,
    }
    test_cases = [
        {'initial_thickness_mm': 100.0, 'target_thickness_mm': 10.0, 'hr_limit_mm': 30},
        {'initial_thickness_mm': 12.0, 'target_thickness_mm': 10.0, 'hr_limit_mm': 30},
        {'initial_thickness_mm': 50.0, 'target_thickness_mm': 5.0, 'hr_limit_mm': 5.0},
    ]

    for case in test_cases:
      obs, _ = env.reset(options={'mode': 'inference', **case, **base_params})
      action_mask = obs['action_mask']
      hr_mask = action_mask['height_reduction']
      assert hr_mask[0] == 1.0, \
          f'Action 0 should always be valid, case: {case}'

  def test_70_percent_constraint_binding(self, env):
    """Test scenario where 70% constraint is the binding one.

    With thickness=20mm, hr_limit=30mm, target=5mm:
    - Constraint 1 (hr_limit): 30mm
    - Constraint 2 (70%): 20 * 0.70 = 14mm  <-- binding
    - Constraint 3 (target): 20 - 5 = 15mm
    """
    obs, info = env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 20.0,
            'target_thickness_mm': 5.0,
            'hr_limit_mm': 30.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    action_mask = obs['action_mask']
    hr_mask = action_mask['height_reduction']
    valid_actions = np.where(hr_mask > 0.5)[0]
    max_valid_action = valid_actions.max() if len(valid_actions) > 0 else 0
    max_valid_hr = max_valid_action / 10.0

    current_thickness = info['current_thickness']
    expected_max = current_thickness * 0.70  # 14.0mm, this is the binding constraint

    assert abs(max_valid_hr - expected_max) < 0.2, \
        f'70% constraint should be binding! Expected {expected_max}, got {max_valid_hr}'

  def test_valid_action_count(self, env):
    """Test that valid action count is reasonable."""
    obs, _ = env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'hr_limit_mm': 30.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    valid_count = obs['action_mask']['height_reduction'].sum()

    # With hr_limit=30mm and 0.1mm resolution, should have ~300 valid actions
    # (0 to 30mm in 0.1mm steps = 301 actions, constrained by 70% rule)
    assert 1 <= valid_count <= 501, \
        f'Valid action count {valid_count} out of expected range'
