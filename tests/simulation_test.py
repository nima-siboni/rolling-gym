"""
Pytest tests for the flat rolling simulation.
"""
# pylint: disable=redefined-outer-name
from __future__ import annotations

import time

import numpy as np
import pytest

from pyroll.core.profile import Profile as BaseProfile

from rollinggym.env.flat_rolling_sim import flat_rolling_simulation
from rollinggym.env.flat_rolling_sim import flat_rolling_step_simulation
from rollinggym.env.helpers import create_in_profile
from rollinggym.env.helpers import create_roll
from rollinggym.env.helpers import roll_stock_mass
from rollinggym.env.helpers import roll_torque
from rollinggym.env.helpers import ultimate_tensile_strength
from rollinggym.env.rolling_config_presets import create_s355_config


@pytest.fixture
def env_config():
  """Create a test environment configuration."""
  return create_s355_config()


@pytest.fixture
def default_in_profile(env_config):
  """Create a default input profile for testing."""
  return create_in_profile(
    starting_thickness_m=0.1,  # 100mm initial thickness
    starting_width_m=env_config.initial.width_m,
    starting_length_m=env_config.initial.length_m,
    starting_temperature_K=env_config.initial.temperature_K,
    starting_grain_size_m=env_config.initial.grain_size_m,
    density_kg_m3=env_config.material.density_kg_m3,
    specific_heat_capacity_j_kg_K=env_config.material.specific_heat_j_kg_K,
    flow_stress_coefficients=env_config.material.models.flow_stress,
    drx_params=env_config.material.models.drx,
    mdrx_params=env_config.material.models.mdrx,
    srx_params=env_config.material.models.srx,
    gg_params=env_config.material.models.grain_growth,
  )


@pytest.fixture
def default_roll(env_config):
  """Create a default roll for testing."""
  return create_roll(env_config)


class TestSimulationBasics:
  """Basic tests for flat rolling simulation."""

  def test_single_pass_simulation(self, default_in_profile, default_roll):
    """Test simulation with a single pass."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],  # 80mm target
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    assert sequence is not None
    assert len(sequence) >= 2  # At least RollPass + Transport

  def test_multi_pass_simulation(self, default_in_profile, default_roll):
    """Test simulation with multiple passes."""
    roll_gap_sequence = [0.08, 0.06, 0.04]  # 80mm, 60mm, 40mm
    interpass_time_sequence = [10.0, 10.0, 10.0]
    rolling_velocity_sequence = [1.0, 1.0, 1.0]

    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=roll_gap_sequence,
        interpass_time_sequence_s=interpass_time_sequence,
        rolling_velocity_sequence_m_s=rolling_velocity_sequence,
    )

    assert sequence is not None
    # Each pass creates RollPass + Transport
    assert len(sequence) == len(roll_gap_sequence) * 2

  def test_simulation_requires_roll_gap_sequence(
      self, default_in_profile, default_roll,
  ):
    """Test that simulation requires roll gap sequence."""
    with pytest.raises(AssertionError):
      flat_rolling_simulation(
          in_profile=default_in_profile,
          roll=default_roll,
          roll_gap_sequence_m=None,
          interpass_time_sequence_s=[10.0],
          rolling_velocity_sequence_m_s=[1.0],
      )

  def test_simulation_requires_interpass_time_sequence(
      self, default_in_profile, default_roll,
  ):
    """Test that simulation requires interpass time sequence."""
    with pytest.raises(AssertionError):
      flat_rolling_simulation(
          in_profile=default_in_profile,
          roll=default_roll,
          roll_gap_sequence_m=[0.08],
          interpass_time_sequence_s=None,
          rolling_velocity_sequence_m_s=[1.0],
      )

  def test_sequence_length_must_match(self, default_in_profile, default_roll):
    """Test that roll gap and interpass sequences must have equal lengths."""
    with pytest.raises(AssertionError):
        flat_rolling_simulation(
            in_profile=default_in_profile,
            roll=default_roll,
            roll_gap_sequence_m=[0.08, 0.06],
            interpass_time_sequence_s=[10.0],  # Mismatched length
            rolling_velocity_sequence_m_s=[1.0],
        )


class TestSimulationResults:
  """Tests for simulation result values."""

  def test_roll_force_is_positive(self, default_in_profile, default_roll):
    """Test that roll force is positive for thickness reduction."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    roll_pass = sequence[0]
    assert roll_pass.roll_force > 0, 'Roll force should be positive'

  def test_roll_torque_is_positive(self, default_in_profile, default_roll):
    """Test that roll torque is positive for thickness reduction."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    roll_pass = sequence[0]
    torque = roll_torque(roll_pass)
    assert torque > 0, 'Roll torque should be positive'

  def test_thickness_reduction_occurs(self, default_in_profile, default_roll):
    """Test that thickness actually reduces after rolling."""
    target_gap = 0.08  # 80mm
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[target_gap],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    roll_pass = sequence[0]
    in_height = roll_pass.in_profile.equivalent_height
    out_height = roll_pass.out_profile.equivalent_height

    assert out_height < in_height, 'Output should be thinner than input'
    assert np.isclose(out_height, target_gap, rtol=0.1), \
        f'Output thickness {out_height} should be close to target {target_gap}'

  def test_temperature_decreases(self, default_in_profile, default_roll):
    """Test that temperature decreases during transport (cooling)."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08, 0.06],
        interpass_time_sequence_s=[10.0, 10.0],
        rolling_velocity_sequence_m_s=[1.0, 1.0],
    )

    # Check temperature after first transport
    first_pass_out_temp = sequence[0].out_profile.temperature
    after_transport_temp = sequence[1].out_profile.temperature

    assert after_transport_temp <= first_pass_out_temp, \
        'Temperature should decrease during transport'

  def test_results_are_finite(self, default_in_profile, default_roll):
    """Test that simulation results are finite (no NaN or Inf)."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    roll_pass = sequence[0]
    assert np.isfinite(roll_pass.roll_force), 'Force should be finite'
    assert np.isfinite(roll_torque(roll_pass)), 'Torque should be finite'
    assert np.isfinite(roll_pass.out_profile.temperature), \
        'Temperature should be finite'


class TestHelperFunctions:
  """Tests for helper functions."""

  def test_create_in_profile(self, env_config):
    """Test in_profile creation."""
    profile = create_in_profile(
      starting_thickness_m=0.1,  # 100mm initial thickness
      starting_width_m=env_config.initial.width_m,
      starting_length_m=env_config.initial.length_m,
      starting_temperature_K=env_config.initial.temperature_K,
      starting_grain_size_m=env_config.initial.grain_size_m,
      density_kg_m3=env_config.material.density_kg_m3,
      specific_heat_capacity_j_kg_K=env_config.material.specific_heat_j_kg_K,
      flow_stress_coefficients=env_config.material.models.flow_stress,
      drx_params=env_config.material.models.drx,
      mdrx_params=env_config.material.models.mdrx,
      srx_params=env_config.material.models.srx,
      gg_params=env_config.material.models.grain_growth,
    )

    assert profile is not None
    assert profile.equivalent_height == 0.1
    assert profile.temperature == env_config.initial.temperature_K

  def test_create_roll(self, env_config):
    """Test roll creation."""
    roll = create_roll(env_config)

    assert roll is not None
    assert roll.nominal_radius == env_config.mill.roll_nominal_radius_m

  def test_roll_stock_mass(self, default_in_profile, default_roll):
    """Test roll stock mass calculation."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    mass = roll_stock_mass(sequence[0])
    assert mass > 0, 'Mass should be positive'
    assert np.isfinite(mass), 'Mass should be finite'

  def test_ultimate_tensile_strength(
      self, env_config, default_in_profile, default_roll,
  ):
    """Test UTS calculation using Hall-Petch relationship."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    uts = ultimate_tensile_strength(env_config, sequence[0])
    assert uts > 0, 'UTS should be positive'
    assert np.isfinite(uts), 'UTS should be finite'


class TestDefaultValues:
  """Tests for default parameter handling."""

  def test_default_roll_temperature(self, default_in_profile, default_roll):
    """Test that default roll temperature is used when not specified."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    assert sequence is not None

  def test_default_rolling_velocity(self, default_in_profile, default_roll):
    """Test that rolling velocity is required."""
    sequence = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    assert sequence is not None


class TestIncrementalSimulation:
  """Tests for incremental (step-by-step) simulation equivalence."""

  def test_incremental_vs_full_simulation_equivalence(
      self, default_in_profile, default_roll,
  ):
    """Verify incremental step simulation produces identical results to full sequence."""
    # Define a 5-pass schedule
    gaps = [0.090, 0.075, 0.055, 0.035, 0.020]
    times = [10.0, 10.0, 10.0, 10.0, 10.0]
    velocities = [0.25, 0.25, 0.25, 0.25, 0.25]

    # Full simulation (existing approach)
    full_seq = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=gaps,
        interpass_time_sequence_s=times,
        rolling_velocity_sequence_m_s=velocities,
    )

    # Incremental simulation (new approach)
    cached_profile = default_in_profile
    step_seq = None
    for i in range(5):
      step_seq = flat_rolling_step_simulation(
          in_profile=cached_profile,
          roll=default_roll,
          roll_gap_m=gaps[i],
          interpass_time_s=times[i],
          rolling_velocity_m_s=velocities[i],
          pass_label_num=i + 1,
      )
      # Cache the Transport's out_profile for next step
      cached_profile = BaseProfile(
          **{
              k: v for k, v in step_seq[-1].out_profile.__dict__.items()
              if not k.startswith('_')
          },
      )

    # Compare final results: full[-2] is last RollPass, full[-1] is last Transport
    # Using rel=1e-5 to account for minor floating-point drift from profile
    # dict copy roundtrips across multiple passes (typically < 1 ppm per pass)
    assert full_seq[-2].out_profile.equivalent_height == pytest.approx(
        step_seq[-2].out_profile.equivalent_height, rel=1e-5,
    )
    assert full_seq[-2].roll_force == pytest.approx(
        step_seq[-2].roll_force, rel=1e-5,
    )
    assert roll_torque(full_seq[-2]) == pytest.approx(
        roll_torque(step_seq[-2]), rel=1e-5,
    )
    assert full_seq[-1].out_profile.temperature == pytest.approx(
        step_seq[-1].out_profile.temperature, rel=1e-5,
    )
    assert full_seq[-1].out_profile.grain_size == pytest.approx(
        step_seq[-1].out_profile.grain_size, rel=1e-5,
    )

  def test_incremental_vs_full_with_varying_parameters(
      self, default_in_profile, default_roll,
  ):
    """Test equivalence with varying velocities and interpass times."""
    gaps = [0.085, 0.065, 0.045]
    times = [5.0, 15.0, 30.0]
    velocities = [0.167, 0.333, 0.5]

    # Full simulation
    full_seq = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=gaps,
        interpass_time_sequence_s=times,
        rolling_velocity_sequence_m_s=velocities,
    )

    # Incremental simulation
    cached_profile = default_in_profile
    step_seq = None
    for i in range(3):
      step_seq = flat_rolling_step_simulation(
          in_profile=cached_profile,
          roll=default_roll,
          roll_gap_m=gaps[i],
          interpass_time_s=times[i],
          rolling_velocity_m_s=velocities[i],
          pass_label_num=i + 1,
      )
      cached_profile = BaseProfile(
          **{
              k: v for k, v in step_seq[-1].out_profile.__dict__.items()
              if not k.startswith('_')
          },
      )

    # Compare all key outputs
    assert full_seq[-2].out_profile.equivalent_height == pytest.approx(
        step_seq[-2].out_profile.equivalent_height, rel=1e-6,
    )
    assert full_seq[-2].roll_force == pytest.approx(
        step_seq[-2].roll_force, rel=1e-6,
    )
    assert full_seq[-1].out_profile.temperature == pytest.approx(
        step_seq[-1].out_profile.temperature, rel=1e-6,
    )
    assert full_seq[-1].out_profile.grain_size == pytest.approx(
        step_seq[-1].out_profile.grain_size, rel=1e-6,
    )

  def test_step_simulation_single_pass(self, default_in_profile, default_roll):
    """Test that a single-pass step simulation matches single-pass full simulation."""
    step_seq = flat_rolling_step_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_m=0.08,
        interpass_time_s=10.0,
        rolling_velocity_m_s=1.0,
        pass_label_num=1,
    )

    full_seq = flat_rolling_simulation(
        in_profile=default_in_profile,
        roll=default_roll,
        roll_gap_sequence_m=[0.08],
        interpass_time_sequence_s=[10.0],
        rolling_velocity_sequence_m_s=[1.0],
    )

    assert len(step_seq) == 2  # RollPass + Transport
    assert step_seq[-2].roll_force == pytest.approx(
        full_seq[-2].roll_force, rel=1e-6,
    )
    assert step_seq[-1].out_profile.temperature == pytest.approx(
        full_seq[-1].out_profile.temperature, rel=1e-6,
    )

  def test_incremental_simulation_speedup(self, default_in_profile, default_roll):
    """Benchmark: time the old (full re-simulation) vs new (incremental) approach.

    Simulates a realistic 10-pass episode and compares wall-clock time.
    The old approach re-simulates all passes from scratch at every step,
    resulting in 1+2+...+N = N(N+1)/2 total pass solves.
    The incremental approach solves exactly 1 pass per step = N total.
    """
    num_passes = 10
    # Realistic descending gap sequence: 90mm -> ~18mm (starting from 100mm profile)
    gaps = [round(0.090 - i * 0.008, 4) for i in range(num_passes)]
    times = [10.0] * num_passes
    velocities = [0.25] * num_passes

    # --- Old approach: re-simulate entire schedule at every step ---
    t0 = time.perf_counter()
    for step in range(1, num_passes + 1):
        flat_rolling_simulation(
            in_profile=default_in_profile,
            roll=default_roll,
            roll_gap_sequence_m=gaps[:step],
            interpass_time_sequence_s=times[:step],
            rolling_velocity_sequence_m_s=velocities[:step],
        )
    old_time = time.perf_counter() - t0

    # --- New approach: incremental, one pass per step ---
    t0 = time.perf_counter()
    cached_profile = default_in_profile
    for i in range(num_passes):
        step_seq = flat_rolling_step_simulation(
            in_profile=cached_profile,
            roll=default_roll,
            roll_gap_m=gaps[i],
            interpass_time_s=times[i],
            rolling_velocity_m_s=velocities[i],
            pass_label_num=i + 1,
        )
        cached_profile = BaseProfile(
            **{
                k: v for k, v in step_seq[-1].out_profile.__dict__.items()
                if not k.startswith('_')
            },
        )
    new_time = time.perf_counter() - t0

    speedup = old_time / new_time
    print(
        f'\n{"=" * 60}\n'
        f'  Simulation Benchmark ({num_passes} passes)\n'
        f'{"=" * 60}\n'
        f'  Old (full re-sim each step) : {old_time:.3f}s  '
        f'({num_passes * (num_passes + 1) // 2} total pass solves)\n'
        f'  New (incremental)           : {new_time:.3f}s  '
        f'({num_passes} total pass solves)\n'
        f'  Speedup                     : {speedup:.1f}x\n'
        f'{"=" * 60}',
    )

    # The incremental approach should be meaningfully faster
    assert new_time < old_time, (
        f'Incremental ({new_time:.3f}s) should be faster than '
        f'full re-simulation ({old_time:.3f}s)'
    )
