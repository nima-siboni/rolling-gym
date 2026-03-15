# pylint: disable=redefined-outer-name
"""
Regression tests that pin exact numeric outputs from deterministic runs.

These values were captured from the original code (before the constants-extraction
refactor) and must remain identical after the refactor.  Any drift in action-space
encoding, reward calculation, state mapping, or mask logic will break a test here.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import rollinggym  # noqa: F401  # pylint: disable=unused-import
from rollinggym.env.rolling_config_presets import create_s355_config
from rollinggym.env.state_evaluation import (
    calculate_reward,
    evaluate_state,
)


# ==================== Fixtures ====================


@pytest.fixture
def env_config():
    return create_s355_config()


@pytest.fixture
def seeded_train_env(env_config):
    env = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={
            'render_mode': None,
            'config': env_config,
            'mode': 'train',
            'env_seed': 123,
        },
    )
    yield env
    env.close()


@pytest.fixture
def inference_env(env_config):
    env = gym.make(
        'rollinggym/FlatRolling-v0.0',
        env_config={
            'render_mode': None,
            'config': env_config,
            'mode': 'inference',
        },
    )
    yield env
    env.close()


INFERENCE_OPTIONS = {
    'mode': 'inference',
    'initial_thickness_mm': 100.0,
    'target_thickness_mm': 10.0,
    'initial_temperature_K': 1373.15,
    'target_temperature_K': 1173.15,
    'initial_grain_size_um': 200.0,
    'target_grain_size_um': 12.0,
}


# ==================== Training reset with seed 123 ====================


class TestSeededTrainReset:
    """Pin exact values from a seeded training reset (env_seed=123, reset seed=123)."""

    def test_initial_thickness(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['current_thickness'] == 90.0

    def test_initial_step_count(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['step_count'] == 0

    def test_initial_hr_limit(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['hr_limit'] == 47.0

    def test_initial_target_thickness(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['target_thickness'] == 11.0

    def test_initial_force_zero(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['rolling_force'] == 0.0

    def test_initial_torque_zero(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['rolling_torque'] == 0.0

    def test_initial_temperature(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['stock_temperature'] == 1286.0

    def test_initial_target_temperature(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['target_temperature'] == 1255.0

    def test_initial_grain_size(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['current_grain_size'] == 188.0

    def test_initial_target_grain_size(self, seeded_train_env):
        _, info = seeded_train_env.reset(seed=123)
        assert info['target_grain_size'] == 10.0

    def test_normalized_observations(self, seeded_train_env):
        obs, _ = seeded_train_env.reset(seed=123)
        expected = [0.75, -1.5, 1.7000000476837158, 0.20000000298023224, -1.5]
        np.testing.assert_allclose(
            obs['observations'][:5].tolist(), expected, rtol=1e-6,
        )

    def test_hr_mask_sum(self, seeded_train_env):
        obs, _ = seeded_train_env.reset(seed=123)
        assert obs['action_mask']['height_reduction'].sum() == 471.0

    def test_interpass_mask_index_zero(self, seeded_train_env):
        obs, _ = seeded_train_env.reset(seed=123)
        assert obs['action_mask']['interpass_time'][0] == 0.0

    def test_velocity_mask_index_zero(self, seeded_train_env):
        obs, _ = seeded_train_env.reset(seed=123)
        assert obs['action_mask']['rolling_velocity'][0] == 0.0


# ==================== Inference 3-step sequence ====================


class TestInference3StepSequence:
    """Pin exact rewards, info values, and flags from a 3-step inference run."""

    def _run_3_steps(self, inference_env):
        """Helper: reset + 3 deterministic steps, return list of (obs, reward, term, trunc, info)."""
        inference_env.reset(options=INFERENCE_OPTIONS)
        actions = [
            np.array([200, 10, 3]),  # 20mm, 10s, 0.25 m/s
            np.array([150, 15, 4]),  # 15mm, 15s, 0.333 m/s
            np.array([100, 20, 2]),  # 10mm, 20s, 0.167 m/s
        ]
        results = []
        for action in actions:
            results.append(inference_env.step(action))
        return results

    # ---------- Step 1: 20mm reduction ----------

    def test_step1_reward(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][1] == pytest.approx(10.140756607055664, rel=1e-5)

    def test_step1_not_terminated(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][2] is False

    def test_step1_not_truncated(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][3] is False

    def test_step1_thickness(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['current_thickness'] == pytest.approx(
            80.0000991821289, rel=1e-5,
        )

    def test_step1_step_count(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['step_count'] == 1

    def test_step1_force(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['rolling_force'] == pytest.approx(
            1579568.125, rel=1e-4,
        )

    def test_step1_torque(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['rolling_torque'] == pytest.approx(
            50401.8828125, rel=1e-4,
        )

    def test_step1_temperature(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['stock_temperature'] == pytest.approx(
            1358.1688232421875, rel=1e-4,
        )

    def test_step1_grain_size(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['current_grain_size'] == pytest.approx(
            58.31090545654297, rel=1e-4,
        )

    def test_step1_gs_progress_bonus(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['gs_progress_bonus'] == pytest.approx(
            11.807424, rel=1e-4,
        )

    def test_step1_hr_efficiency_bonus(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['hr_efficiency_bonus'] == pytest.approx(
            3.3333335, rel=1e-4,
        )

    def test_step1_step_penalty(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[0][4]['step_penalty'] == -5.0

    # ---------- Step 2: 15mm reduction ----------

    def test_step2_reward(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[1][1] == pytest.approx(-2.4192419052124023, rel=1e-4)

    def test_step2_thickness(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[1][4]['current_thickness'] == pytest.approx(
            65.00016784667969, rel=1e-5,
        )

    def test_step2_step_count(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[1][4]['step_count'] == 2

    def test_step2_hr_efficiency_bonus(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[1][4]['hr_efficiency_bonus'] == pytest.approx(2.5, rel=1e-4)

    # ---------- Step 3: 10mm reduction ----------

    def test_step3_reward(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[2][1] == pytest.approx(-4.053238868713379, rel=1e-4)

    def test_step3_thickness(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[2][4]['current_thickness'] == pytest.approx(
            55.00018310546875, rel=1e-5,
        )

    def test_step3_step_count(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[2][4]['step_count'] == 3

    def test_step3_temperature(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[2][4]['stock_temperature'] == pytest.approx(
            1300.5234375, rel=1e-4,
        )

    def test_step3_grain_size(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[2][4]['current_grain_size'] == pytest.approx(
            65.98067474365234, rel=1e-4,
        )

    def test_step3_gs_progress_negative(self, inference_env):
        """Step 3 grain size increased (grain growth > recrystallization) → negative progress."""
        results = self._run_3_steps(inference_env)
        assert results[2][4]['gs_progress_bonus'] == pytest.approx(
            -0.71990556, rel=1e-3,
        )

    def test_step3_hr_efficiency_bonus(self, inference_env):
        results = self._run_3_steps(inference_env)
        assert results[2][4]['hr_efficiency_bonus'] == pytest.approx(
            1.6666667, rel=1e-4,
        )


# ==================== evaluate_state pinned values ====================


class TestEvaluateStatePinned:
    """Pin evaluate_state outputs for known state vectors."""

    def test_completed_state(self):
        config = create_s355_config()
        state = np.array([
            10.5, 5.0, 60.0, 10.0,
            1.5e6, 50e3,
            1173.0, 1173.0,
            12.0, 12.0,
        ], dtype=np.float32)
        evals = evaluate_state(state, 5.0, config)
        assert evals[0] is False  # hr_lim_exceeded
        assert evals[1] is False  # overshoot
        assert evals[2] is True   # completed
        assert evals[3] is False  # excessive_force
        assert evals[4] is False  # excessive_torque
        assert evals[5] is False  # simulation_failed

    def test_simulation_failed_state(self):
        config = create_s355_config()
        state = np.array([
            50.0, 3.0, 60.0, 10.0,
            -100.0, -100.0,
            -100.0, 1173.0,
            -100.0, 12.0,
        ], dtype=np.float32)
        evals = evaluate_state(state, 10.0, config)
        assert evals[5] is True  # simulation_failed

    def test_excessive_force(self):
        config = create_s355_config()
        # force = 3.8e6, limit = 4e6, threshold = 90% = 3.6e6 → exceeded
        state = np.array([
            50.0, 3.0, 60.0, 10.0,
            3.8e6, 50e3,
            1300.0, 1173.0,
            50.0, 12.0,
        ], dtype=np.float32)
        evals = evaluate_state(state, 10.0, config)
        assert evals[3] is True   # excessive_force
        assert evals[4] is False  # excessive_torque

    def test_excessive_torque(self):
        config = create_s355_config()
        # torque = 120e3, limit = 130e3, threshold = 90% = 117e3 → exceeded
        state = np.array([
            50.0, 3.0, 60.0, 10.0,
            1e6, 120e3,
            1300.0, 1173.0,
            50.0, 12.0,
        ], dtype=np.float32)
        evals = evaluate_state(state, 10.0, config)
        assert evals[3] is False  # excessive_force
        assert evals[4] is True   # excessive_torque

    def test_overshoot(self):
        config = create_s355_config()
        # thickness = 8.0 < target(10.0) - tolerance(1mm) = 9.0 → overshoot
        state = np.array([
            8.0, 5.0, 60.0, 10.0,
            1e6, 50e3,
            1300.0, 1173.0,
            50.0, 12.0,
        ], dtype=np.float32)
        evals = evaluate_state(state, 5.0, config)
        assert evals[1] is True  # overshoot


# ==================== calculate_reward pinned values ====================


class TestCalculateRewardPinned:
    """Pin exact reward values from calculate_reward for known inputs."""

    def test_reward_at_completion_perfect(self):
        """Completed episode at target grain + temp → ~46.08 total."""
        config = create_s355_config()
        state = np.array([
            10.5, 5.0, 60.0, 10.0,
            1.5e6, 50e3,
            1173.0, 1173.0,
            12.0, 12.0,
        ], dtype=np.float32)
        reward, details = calculate_reward(
            state,
            previous_grain_size_um=15.0,
            height_reduction_mm=5.0,
            completed=True,
            config=config,
        )
        assert reward == pytest.approx(46.083335876464844, rel=1e-5)
        assert details['gs_progress_bonus'] == pytest.approx(0.25, rel=1e-4)
        assert details['hr_efficiency_bonus'] == pytest.approx(
            0.8333333730697632, rel=1e-5,
        )
        assert details['gs_completion_bonus'] == pytest.approx(25.0)
        assert details['temperature_completion_bonus'] == pytest.approx(25.0)
        assert details['step_penalty'] == -5.0

    def test_reward_simulation_failed(self):
        """Simulation failure sentinels → total = -15.0."""
        config = create_s355_config()
        state = np.array([
            50.0, 3.0, 60.0, 10.0,
            -100.0, -100.0,
            -100.0, 1173.0,
            -100.0, 12.0,
        ], dtype=np.float32)
        reward, details = calculate_reward(
            state,
            previous_grain_size_um=50.0,
            height_reduction_mm=10.0,
            completed=False,
            config=config,
        )
        assert reward == pytest.approx(-15.0)
        assert details['gs_progress_bonus'] == pytest.approx(-5.0)
        assert details['hr_efficiency_bonus'] == pytest.approx(-5.0)
        assert details['gs_completion_bonus'] == pytest.approx(0.0)
        assert details['temperature_completion_bonus'] == pytest.approx(0.0)
        assert details['step_penalty'] == -5.0


# ==================== Action → physical conversions in step() ====================


class TestActionConversions:
    """Verify that action indices convert to the same physical values as before."""

    def test_hr_action_200_gives_20mm(self, inference_env):
        """action[0]=200 → 20.0mm height reduction."""
        inference_env.reset(options=INFERENCE_OPTIONS)
        _, _, _, _, info = inference_env.step(np.array([200, 10, 3]))
        # Thickness should decrease by ~20mm (simulation may adjust slightly)
        assert info['current_thickness'] == pytest.approx(80.0, abs=0.5)

    def test_hr_action_0_gives_no_reduction(self, inference_env):
        """action[0]=0 → 0mm height reduction, thickness unchanged."""
        _, info_before = inference_env.reset(options=INFERENCE_OPTIONS)
        _, _, _, _, info_after = inference_env.step(np.array([0, 10, 3]))
        assert info_after['current_thickness'] == pytest.approx(
            info_before['current_thickness'], abs=0.01,
        )

    def test_interpass_action_is_direct_seconds(self, inference_env):
        """action[1]=10 means 10 seconds interpass time (direct mapping)."""
        inference_env.reset(options=INFERENCE_OPTIONS)
        # We can only verify indirectly: step_count increments, step completes
        _, _, _, _, info = inference_env.step(np.array([100, 10, 3]))
        assert info['step_count'] == 1

    def test_velocity_action_3_gives_0_25_m_s(self):
        """action[2]=3 → 3/12 = 0.25 m/s."""
        from rollinggym.env.flat_rolling_env import FlatRollingEnv
        assert 3 / FlatRollingEnv.VELOCITY_DIVISOR == pytest.approx(0.25)


# ==================== Mask boundary correctness ====================


class TestMaskBoundaryPinned:
    """Pin mask behavior at specific state boundaries."""

    def test_near_target_limited_actions(self, inference_env):
        """At 12mm with target 10mm, only small reductions are allowed."""
        inference_env.reset(options={
            **INFERENCE_OPTIONS,
            'initial_thickness_mm': 12.0,
        })
        obs = inference_env.unwrapped._get_obs()
        hr_mask = obs['action_mask']['height_reduction']

        # Max allowed: min(hr_limit=60, 70%*12=8.4, 12-10=2) = 2mm → action index 20
        assert hr_mask[0] == 1.0   # 0mm always valid
        assert hr_mask[20] == 1.0  # 2.0mm valid (12-2=10 == target)
        assert hr_mask[21] == 0.0  # 2.1mm would go below target

    def test_all_masks_are_float32(self, inference_env):
        inference_env.reset(options=INFERENCE_OPTIONS)
        obs = inference_env.unwrapped._get_obs()
        assert obs['action_mask']['height_reduction'].dtype == np.float32
        assert obs['action_mask']['interpass_time'].dtype == np.float32
        assert obs['action_mask']['rolling_velocity'].dtype == np.float32

    def test_interpass_mask_shape(self, inference_env):
        inference_env.reset(options=INFERENCE_OPTIONS)
        obs = inference_env.unwrapped._get_obs()
        assert obs['action_mask']['interpass_time'].shape == (121,)
        # All valid except index 0
        assert obs['action_mask']['interpass_time'].sum() == 120.0

    def test_velocity_mask_shape(self, inference_env):
        inference_env.reset(options=INFERENCE_OPTIONS)
        obs = inference_env.unwrapped._get_obs()
        assert obs['action_mask']['rolling_velocity'].shape == (7,)
        # All valid except index 0
        assert obs['action_mask']['rolling_velocity'].sum() == 6.0
