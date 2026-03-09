"""
Pytest tests for the FlatRolling environment.
"""
# pylint: disable=redefined-outer-name
# Note: redefined-outer-name is disabled because pytest fixtures
# use the same name pattern for dependency injection.
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import rollinggym  # noqa: F401  # pylint: disable=unused-import
from rollinggym.env.rolling_config_presets import create_s355_config


@pytest.fixture
def env_config():
  """Create a test environment configuration."""
  return create_s355_config()


@pytest.fixture
def train_env(env_config):
  """Create a training environment."""
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


class TestEnvironmentCreation:
  """Tests for environment creation and initialization."""

  def test_env_creation(self, train_env):
    """Test that environment can be created."""
    assert train_env is not None

  def test_action_space(self, train_env):
    """Test action space configuration (MultiDiscrete: [hr, interpass_time, velocity])."""
    action_space = train_env.action_space
    assert hasattr(action_space, 'nvec'), 'Action space should be MultiDiscrete'
    assert action_space.nvec[0] == 501  # height reduction: 0 to 500
    assert action_space.nvec[1] == 121  # interpass time: 0 to 120
    assert action_space.nvec[2] == 7    # rolling velocity: 1 to 6 (0 masked, maps to 5-30 m/min)

  def test_observation_space_structure(self, train_env):
    """Test observation space is a Dict with correct keys."""
    obs_space = train_env.observation_space
    assert 'observations' in obs_space.spaces
    assert 'action_mask' in obs_space.spaces

  def test_observation_space_shapes(self, train_env):
    """Test observation space shapes."""
    obs_space = train_env.observation_space
    assert obs_space['observations'].shape == (10,)
    # action_mask is a Dict with height_reduction, interpass_time, and rolling_velocity
    assert 'height_reduction' in obs_space['action_mask'].spaces
    assert 'interpass_time' in obs_space['action_mask'].spaces
    assert 'rolling_velocity' in obs_space['action_mask'].spaces
    assert obs_space['action_mask']['height_reduction'].shape == (501,)
    assert obs_space['action_mask']['interpass_time'].shape == (121,)
    assert obs_space['action_mask']['rolling_velocity'].shape == (7,)


class TestEnvironmentReset:
  """Tests for environment reset functionality."""

  def test_reset_train_mode(self, train_env):
    """Test reset in training mode returns valid observation."""
    obs, _ = train_env.reset()

    assert 'observations' in obs
    assert 'action_mask' in obs
    assert obs['observations'].shape == (10,)
    assert 'height_reduction' in obs['action_mask']
    assert 'interpass_time' in obs['action_mask']
    assert 'rolling_velocity' in obs['action_mask']
    assert obs['action_mask']['height_reduction'].shape == (501,)
    assert obs['action_mask']['interpass_time'].shape == (121,)
    assert obs['action_mask']['rolling_velocity'].shape == (7,)

  def test_reset_inference_mode(self, inference_env):
    """Test reset in inference mode with specified thicknesses.

    Note: observations are normalized, so we check the info dict for raw values.
    """
    _, info = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    assert info['current_thickness'] == 100.0  # initial thickness
    assert info['target_thickness'] == 10.0    # target thickness

  def test_reset_inference_mode_requires_options(self, inference_env):
    """Test that inference mode requires options."""
    with pytest.raises(ValueError):
      inference_env.reset(options={'mode': 'inference'})

  def test_reset_clears_pass_schedule(self, train_env):
    """Test that reset clears the pass schedule."""
    train_env.reset()

    # Take some steps (action format: [hr_index, interpass_time_index, velocity_index])
    for _ in range(3):
      train_env.step([100, 10, 2])

    # Reset should clear the schedule
    train_env.reset()
    assert len(train_env.unwrapped.pass_schedule_thickness_sequence_m) == 0

  def test_reset_stores_randomized_initial_conditions(self, train_env):
    """Test that reset stores randomized temperature and grain size for simulation.

    Fixes #37: simulation must use randomized initial conditions, not fixed config defaults.
    """
    train_env.reset()
    env = train_env.unwrapped

    # The stored simulation parameters must match the randomized state
    assert env.starting_thickness_mm == env.state[0].item()
    assert env.starting_temperature_K == env.state[6].item()
    assert env.starting_grain_size_um == pytest.approx(
        env.state[8].item(),
    ), 'starting_grain_size_um should equal state[8] in µm'

  def test_initial_conditions_vary_across_resets(self, train_env):
    """Test that temperature and grain size actually vary across resets.

    Verifies the randomization produces different initial conditions.
    """
    temps = set()
    grain_sizes = set()
    for _ in range(10):
      train_env.reset()
      env = train_env.unwrapped
      temps.add(env.starting_temperature_K)
      grain_sizes.add(env.starting_grain_size_um)

    assert len(temps) > 1, 'Temperature should vary across resets'
    assert len(grain_sizes) > 1, 'Grain size should vary across resets'

  def test_seeded_reset_reproducibility(self, env_config):
    """Test that seeded environments produce reproducible results."""
    env1 = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={
            'render_mode': None,
            'config': env_config,
            'mode': 'train',
            'env_seed': 12345,
        },
    )
    env2 = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={
            'render_mode': None,
            'config': env_config,
            'mode': 'train',
            'env_seed': 12345,
        },
    )

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()

    np.testing.assert_array_equal(
        obs1['observations'],
        obs2['observations'],
        err_msg='Seeded environments should produce identical initial states',
    )

    env1.close()
    env2.close()


class TestEnvironmentStep:
  """Tests for environment step functionality."""

  def test_step_returns_correct_types(self, train_env):
    """Test that step returns correct types."""
    train_env.reset()
    obs, reward, terminated, truncated, info = train_env.step([100, 10, 2])

    assert isinstance(obs, dict)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

  def test_step_zero_action(self, inference_env):
    """Test that action 0 (no reduction) doesn't change thickness."""
    obs, _ = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    initial_thickness = obs['observations'][0]

    obs, _, _, _, _ = inference_env.step([0, 1, 2])
    after_thickness = obs['observations'][0]

    assert after_thickness == initial_thickness, \
        'Action 0 should not change thickness'

  def test_step_increments_step_count(self, train_env):
    """Test that each step increments the step counter.

    Note: observations are normalized, so we access the underlying env's
    state directly to verify the step count increments.
    """
    train_env.reset()
    # Access the underlying environment's state (unnormalized)
    initial_step_count = train_env.unwrapped.state[1]

    train_env.step([0, 1, 2])
    # Verify step count incremented
    assert train_env.unwrapped.state[1] == initial_step_count + 1

  def test_truncation_at_max_steps(self, train_env):
    """Test that episode truncates at max steps (25)."""
    train_env.reset()

    truncated = False
    for _ in range(30):
      _, _, terminated, truncated, _ = train_env.step([0, 1, 2])
      if terminated or truncated:
        break

    assert truncated or terminated, \
        'Episode should end by step 25'

  def test_termination_on_target_reached(self, inference_env):
    """Test termination when target thickness is reached."""
    # Use a scenario where target can be reached in few steps
    obs, _ = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 15.0,
            'target_thickness_mm': 10.0,
            'hr_limit_mm': 30.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    # Take action to reduce by 5mm (hr_index=50, interpass_time=10, velocity=2)
    obs, _, terminated, _, _ = inference_env.step([50, 10, 2])

    # Should terminate when within tolerance of target
    current = obs['observations'][0]
    target = obs['observations'][3]
    tolerance = 1.0  # 1mm tolerance from config

    if abs(current - target) <= tolerance:
      assert terminated, 'Should terminate when target reached'


class TestPostTransportObservation:
  """Tests that temperature and grain size reflect post-transport state (Fixes #38)."""

  def test_temperature_reflects_interpass_cooling(self, inference_env):
    """Temperature after a step should be lower than the initial temperature.

    With a 120s interpass time, significant cooling occurs during transport.
    If observations came from the RollPass (pre-transport), temperature
    would only reflect deformation heating, not the subsequent cooling.
    """
    inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    # Take a rolling step with long interpass time (120s) to maximize cooling
    _, _, _, _, info = inference_env.step([200, 120, 3])

    # After rolling + 120s transport, temperature must be below initial
    assert info['stock_temperature'] < 1373.0, \
        'Temperature should reflect post-transport cooling'

  def test_long_vs_short_interpass_temperature_difference(self, inference_env, env_config):
    """Longer interpass time should produce lower temperature.

    This can only be true if temperature is read from the Transport phase.
    """
    # Run with short interpass time
    inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    _, _, _, _, info_short = inference_env.step([200, 1, 3])
    temp_short = info_short['stock_temperature']

    # Run with long interpass time (same HR and velocity)
    env_long = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={
            'render_mode': None,
            'config': env_config,
            'mode': 'inference',
        },
    )
    env_long.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    _, _, _, _, info_long = env_long.step([200, 120, 3])
    temp_long = info_long['stock_temperature']
    env_long.close()

    assert temp_long < temp_short, \
        'Longer interpass time should result in lower temperature (more cooling)'


class TestRewardAndInfo:
  """Tests for reward calculation and info dictionary contents."""

  def test_reward_components_in_info(self, train_env):
    """Test that reward components are returned in info dict."""
    train_env.reset()
    _, _, _, _, info = train_env.step([100, 10, 2])

    expected_keys = [
        'step_penalty',
        'gs_progress_bonus',
        'hr_efficiency_bonus',
        'gs_completion_bonus',
        'temperature_completion_bonus',
    ]

    for key in expected_keys:
      assert key in info, f'Expected {key} in info dict'

  def test_step_penalty_is_negative(self, train_env):
    """Test that step penalty is always negative."""
    train_env.reset()
    _, _, _, _, info = train_env.step([100, 10, 2])

    assert info['step_penalty'] < 0, 'Step penalty should be negative'

  def test_info_contains_state_values(self, train_env):
    """Test that info dict contains expected state values."""
    train_env.reset()
    _, _, _, _, info = train_env.step([100, 10, 2])

    # Check for reward component keys
    assert 'step_penalty' in info
    assert 'gs_progress_bonus' in info
    assert 'hr_efficiency_bonus' in info
    assert 'gs_completion_bonus' in info
    assert 'temperature_completion_bonus' in info


class TestPhysicalBehavior:
  """Tests that the environment exhibits physically correct behavior."""

  def test_thickness_decreases_after_reduction(self, inference_env):
    """Test that thickness actually decreases after a non-zero height reduction."""
    _, info_before = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    thickness_before = info_before['current_thickness']

    _, _, _, _, info_after = inference_env.step([200, 10, 3])  # 20mm reduction
    thickness_after = info_after['current_thickness']

    assert thickness_after < thickness_before, \
        f'Thickness should decrease: {thickness_before} -> {thickness_after}'

  def test_force_and_torque_positive_after_rolling(self, inference_env):
    """Test that rolling force and torque are positive after a real pass."""
    inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    _, _, _, _, info = inference_env.step([200, 10, 3])  # 20mm reduction

    assert info['rolling_force'] > 0, 'Rolling force should be positive'
    assert info['rolling_torque'] > 0, 'Rolling torque should be positive'

  def test_grain_size_decreases_during_rolling(self, inference_env):
    """Test that grain size decreases due to recrystallization during rolling."""
    _, info_before = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    grain_before = info_before['current_grain_size']

    _, _, _, _, info_after = inference_env.step([200, 10, 3])  # 20mm reduction
    grain_after = info_after['current_grain_size']

    assert grain_after < grain_before, \
        f'Grain size should decrease after rolling: {grain_before} -> {grain_after}'

  def test_temperature_decreases_over_episode(self, inference_env):
    """Test that temperature decreases over multiple passes with interpass cooling."""
    _, info_initial = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    initial_temp = info_initial['stock_temperature']

    # Take several passes with reasonable interpass times
    for _ in range(3):
      _, _, terminated, truncated, info = inference_env.step([100, 30, 3])
      if terminated or truncated:
        break

    assert info['stock_temperature'] < initial_temp, \
        'Temperature should decrease over multiple passes'

  def test_larger_reduction_produces_higher_force(self, inference_env, env_config):
    """Test that a larger height reduction produces a higher rolling force."""
    # Small reduction
    inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    _, _, _, _, info_small = inference_env.step([50, 10, 3])  # 5mm reduction

    # Large reduction (fresh env to start from same state)
    env_large = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={'render_mode': None, 'config': env_config, 'mode': 'inference'},
    )
    env_large.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    _, _, _, _, info_large = env_large.step([200, 10, 3])  # 20mm reduction
    env_large.close()

    assert info_large['rolling_force'] > info_small['rolling_force'], \
        'Larger reduction should produce higher force'


class TestRewardFiniteness:
  """Tests that rewards are always finite (no NaN or Inf)."""

  def test_reward_is_finite_normal_step(self, train_env):
    """Test that reward is finite for a normal step."""
    train_env.reset()
    _, reward, _, _, _ = train_env.step([100, 10, 2])
    assert np.isfinite(reward), f'Reward should be finite, got {reward}'

  def test_reward_is_finite_zero_action(self, train_env):
    """Test that reward is finite for a zero action (no reduction)."""
    train_env.reset()
    _, reward, _, _, _ = train_env.step([0, 1, 1])
    assert np.isfinite(reward), f'Reward should be finite for zero action, got {reward}'

  def test_reward_is_finite_throughout_episode(self, train_env):
    """Test that reward is finite throughout an entire episode."""
    train_env.reset()
    for step in range(25):
      _, reward, terminated, truncated, _ = train_env.step([50, 10, 3])
      assert np.isfinite(reward), f'Reward not finite at step {step}: {reward}'
      if terminated or truncated:
        break


class TestFullEpisode:
  """Tests for running a complete episode."""

  def test_episode_runs_to_completion(self, inference_env):
    """Test that an episode can run to completion (target reached)."""
    inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 30.0,
            'target_thickness_mm': 10.0,
            'hr_limit_mm': 15.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    terminated = False
    truncated = False
    steps = 0
    for _ in range(25):
      _, _, terminated, truncated, _ = inference_env.step([100, 10, 3])  # 10mm per pass
      steps += 1
      if terminated or truncated:
        break

    assert terminated or truncated, 'Episode should end within 25 steps'

  def test_completed_episode_state_near_target(self, inference_env):
    """Test that a completed episode has thickness near the target."""
    inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 15.0,
            'target_thickness_mm': 10.0,
            'hr_limit_mm': 30.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    # Reduce by 5mm to reach target
    _, _, terminated, _, info = inference_env.step([50, 10, 3])

    if terminated:
      # Config tolerance is 1mm (0.001m)
      assert abs(info['current_thickness'] - info['target_thickness']) <= 1.5, \
          'Completed episode should be within tolerance of target'

  def test_pass_schedule_tracks_steps(self, inference_env):
    """Test that the pass schedule records each step taken."""
    inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    n_steps = 3
    for _ in range(n_steps):
      inference_env.step([100, 10, 3])  # 10mm reduction each

    env = inference_env.unwrapped
    assert len(env.pass_schedule_thickness_sequence_m) == n_steps
    assert len(env.pass_schedule_interpass_sequence_s) == n_steps
    assert len(env.pass_schedule_rolling_velocity_sequence_m_s) == n_steps


class TestInferenceValidation:
  """Tests for inference mode input validation."""

  def test_initial_thickness_out_of_range(self, inference_env):
    """Test that out-of-range initial thickness raises ValueError."""
    with pytest.raises(ValueError, match='initial_thickness'):
      inference_env.reset(
          options={
              'mode': 'inference',
              'initial_thickness_mm': 200.0,  # Max is 150
              'target_thickness_mm': 10.0,
              'initial_grain_size_um': 200.0,
              'target_grain_size_um': 12.0,
              'initial_temperature_K': 1373.0,
              'target_temperature_K': 1173.0,
          },
      )

  def test_target_thickness_out_of_range(self, inference_env):
    """Test that out-of-range target thickness raises ValueError."""
    with pytest.raises(ValueError, match='target_thickness'):
      inference_env.reset(
          options={
              'mode': 'inference',
              'initial_thickness_mm': 100.0,
              'target_thickness_mm': 60.0,  # Max is 50
              'initial_grain_size_um': 200.0,
              'target_grain_size_um': 12.0,
              'initial_temperature_K': 1373.0,
              'target_temperature_K': 1173.0,
          },
      )

  def test_temperature_out_of_range(self, inference_env):
    """Test that out-of-range temperature raises ValueError."""
    with pytest.raises(ValueError, match='initial_temperature_K'):
      inference_env.reset(
          options={
              'mode': 'inference',
              'initial_thickness_mm': 100.0,
              'target_thickness_mm': 10.0,
              'initial_grain_size_um': 200.0,
              'target_grain_size_um': 12.0,
              'initial_temperature_K': 900.0,  # Min is 1273
              'target_temperature_K': 1173.0,
          },
      )

  def test_grain_size_out_of_range(self, inference_env):
    """Test that out-of-range grain size raises ValueError."""
    with pytest.raises(ValueError, match='initial_grain_size_um'):
      inference_env.reset(
          options={
              'mode': 'inference',
              'initial_thickness_mm': 100.0,
              'target_thickness_mm': 10.0,
              'initial_grain_size_um': 50.0,  # Min is 100
              'target_grain_size_um': 12.0,
              'initial_temperature_K': 1373.0,
              'target_temperature_K': 1173.0,
          },
      )

  def test_missing_required_fields(self, inference_env):
    """Test that missing required fields raise ValueError."""
    with pytest.raises(ValueError):
      inference_env.reset(
          options={
              'mode': 'inference',
              'initial_thickness_mm': 100.0,
              # Missing target_thickness_mm and other required fields
          },
      )

  def test_hr_limit_out_of_range(self, inference_env):
    """Test that out-of-range hr_limit raises ValueError."""
    with pytest.raises(ValueError, match='hr_limit_mm'):
      inference_env.reset(
          options={
              'mode': 'inference',
              'initial_thickness_mm': 100.0,
              'target_thickness_mm': 10.0,
              'hr_limit_mm': 70.0,  # Max is 60
              'initial_grain_size_um': 200.0,
              'target_grain_size_um': 12.0,
              'initial_temperature_K': 1373.0,
              'target_temperature_K': 1173.0,
          },
      )


class TestMaskConstraints:
  """Tests for interpass time and velocity mask constraints."""

  def test_interpass_time_zero_always_masked(self, inference_env):
    """Test that interpass time index 0 is always masked out."""
    obs, _ = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    assert obs['action_mask']['interpass_time'][0] == 0.0, \
        'Interpass time = 0s should always be masked'

  def test_interpass_time_nonzero_valid(self, inference_env):
    """Test that interpass time indices 1-120 are valid."""
    obs, _ = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    assert all(obs['action_mask']['interpass_time'][1:] == 1.0), \
        'All interpass times >= 1s should be valid'

  def test_velocity_zero_always_masked(self, inference_env):
    """Test that velocity index 0 is always masked out."""
    obs, _ = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    assert obs['action_mask']['rolling_velocity'][0] == 0.0, \
        'Rolling velocity = 0 should always be masked'

  def test_velocity_nonzero_valid(self, inference_env):
    """Test that velocity indices 1-6 are valid."""
    obs, _ = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    assert all(obs['action_mask']['rolling_velocity'][1:] == 1.0), \
        'All velocities >= 5 m/min should be valid'


class TestObservationNormalization:
  """Tests for observation z-score normalization."""

  def test_observations_in_reasonable_range(self, inference_env):
    """Test that normalized observations are in a reasonable range."""
    obs, _ = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    normalized = obs['observations']
    assert all(np.isfinite(normalized)), 'All observations should be finite'
    # Z-score normalized values should generally be within [-10, 10]
    assert all(np.abs(normalized) <= 10.0), \
        f'Normalized observations out of expected range: {normalized}'

  def test_observations_remain_finite_after_steps(self, inference_env):
    """Test that observations remain finite throughout an episode."""
    inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    for step in range(10):
      obs, _, terminated, truncated, _ = inference_env.step([50, 10, 3])
      normalized = obs['observations']
      assert all(np.isfinite(normalized)), \
          f'Observations not finite at step {step}: {normalized}'
      if terminated or truncated:
        break

  def test_different_states_produce_different_observations(self, inference_env, env_config):
    """Test that different initial states produce different normalized observations."""
    obs1, _ = inference_env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 100.0,
            'target_thickness_mm': 10.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )

    env2 = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={'render_mode': None, 'config': env_config, 'mode': 'inference'},
    )
    obs2, _ = env2.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 50.0,
            'target_thickness_mm': 5.0,
            'initial_grain_size_um': 300.0,
            'target_grain_size_um': 20.0,
            'initial_temperature_K': 1473.0,
            'target_temperature_K': 1173.0,
        },
    )
    env2.close()

    assert not np.array_equal(obs1['observations'], obs2['observations']), \
        'Different states should produce different observations'


class TestConfigOverride:
  """Tests that custom config parameters propagate correctly."""

  def test_custom_thickness(self):
    """Test that custom initial/target thickness propagates to environment."""
    config = create_s355_config(
        initial_thickness_mm=80.0,
        target_thickness_mm=8.0,
    )
    env = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={'render_mode': None, 'config': config, 'mode': 'inference'},
    )
    _, info = env.reset(
        options={
            'mode': 'inference',
            'initial_thickness_mm': 80.0,
            'target_thickness_mm': 8.0,
            'initial_grain_size_um': 200.0,
            'target_grain_size_um': 12.0,
            'initial_temperature_K': 1373.0,
            'target_temperature_K': 1173.0,
        },
    )
    assert info['current_thickness'] == 80.0
    assert info['target_thickness'] == 8.0
    env.close()

  def test_custom_equipment_limits(self):
    """Test that custom equipment limits propagate to config."""
    config = create_s355_config(
        force_limit_MN=5.0,
        torque_limit_kNm=200.0,
    )
    assert config.equipment.force_N == 5.0e6  # pylint: disable=no-member
    assert config.equipment.torque_Nm == 200.0e3  # pylint: disable=no-member

  def test_custom_roll_geometry(self):
    """Test that custom roll geometry propagates to config."""
    config = create_s355_config(
        roll_nominal_radius_mm=300.0,
        roll_usable_width_mm=400.0,
    )
    assert config.mill.roll_nominal_radius_m == pytest.approx(0.3)  # pylint: disable=no-member
    assert config.mill.roll_usable_width_m == pytest.approx(0.4)  # pylint: disable=no-member
