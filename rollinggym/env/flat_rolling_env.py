# pylint: disable=duplicate-code
"""
Flat rolling environment for training a reinforcement learning agent.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pyroll.core as pr
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.spaces import Dict
from gymnasium.spaces import MultiDiscrete
from pyroll.core.profile import Profile as BaseProfile

from rollinggym.env.flat_rolling_sim import flat_rolling_step_simulation
from rollinggym.env.helpers import create_in_profile
from rollinggym.env.helpers import create_roll
from rollinggym.env.helpers import roll_torque
from rollinggym.env.helpers import timeout
from rollinggym.env.state_evaluation import calculate_reward
from rollinggym.env.state_evaluation import evaluate_state


logger = logging.getLogger(__name__)


class FlatRollingEnv(Env):     # pylint: disable=too-many-instance-attributes
    """
    A flat rolling environment for reinforcement learning agents.

    Observation space is a Dict with two keys:
        - 'observations': z-score normalized state vector (10 continuous values):
            thickness, step count, HR limit, target thickness, force, torque,
            temperature, target temperature, grain size, target grain size.
        - 'action_mask': Dict of binary masks for valid actions per dimension.

    Action space is MultiDiscrete([501, 121, 7]):
        height reduction, interpass time, and rolling velocity.

    Uses incremental PyRoll simulation with profile caching to avoid
    re-simulating the entire pass schedule at each step.
    """

    # ==================== Class-Level Constants ====================

    #: Schema for the info dict returned by ``_get_info()``.
    #: Maps key name → (state_index, cast_type, unit, human-readable description).
    INFO_SCHEMA: dict[str, tuple[int, type, str, str]] = {
        'current_thickness': (0, float, 'mm', 'Current slab thickness'),
        'step_count': (1, int, 'int', 'Current pass number (0-indexed)'),
        'hr_limit': (2, float, 'mm', 'Max allowed height reduction per pass'),
        'target_thickness': (3, float, 'mm', 'Desired final thickness'),
        'rolling_force': (4, float, 'N', 'Force from last pass (0 at start)'),
        'rolling_torque': (5, float, 'Nm', 'Torque from last pass (0 at start)'),
        'stock_temperature': (6, float, 'K', 'Current temperature'),
        'target_temperature': (7, float, 'K', 'Desired final temperature'),
        'current_grain_size': (8, float, 'um', 'Current grain size in micrometers'),
        'target_grain_size': (9, float, 'um', 'Desired final grain size in micrometers'),
    }

    #: Action space dimensions: (height_reduction, interpass_time, rolling_velocity).
    ACTION_DIMS: tuple[int, int, int] = (501, 121, 7)

    #: Divisor to convert HR action index to mm (action[0] / HR_RESOLUTION).
    HR_RESOLUTION: float = 10.0

    #: Divisor to convert velocity action index to m/s (action[2] / VELOCITY_DIVISOR).
    VELOCITY_DIVISOR: float = 12.0

    #: Maximum height reduction as a fraction of current thickness.
    MAX_HR_FRACTION: float = 0.70

    #: Maximum number of steps before episode truncation.
    MAX_STEPS: int = 25

    def __init__(
            self,
            env_config,
    ):

        super().__init__()
        self.config = env_config.get('config', None)
        if self.config is None:
            raise ValueError(
                'Config must be provided to initialize FlatRollingEnv.',
            )

        # state comprises thickness, step count (for truncation), height reduction limit,
        # target thickness, rolling force, rolling torque, current temperature,
        # target temperature, current grain size, and target grain size.
        # Actual pass number = len(pass_schedule_thickness_sequence)
        self.state = np.zeros((10,), dtype=np.float32)

        # Observation normalization statistics (mean and std for each state dimension)
        # These values are based on typical ranges observed during training episodes:
        # - state[0]: thickness ~5-150 mm
        # - state[1]: step count 0-25
        # - state[2]: HR limit ~20-40 mm
        # - state[3]: target thickness ~5-15 mm
        # - state[4]: force ~0-4e6 N
        # - state[5]: torque ~0-1e5 Nm
        # - state[6]: current temperature ~700-1350 K
        # - state[7]: target temperature ~1073-1273 K (800-1000 C)
        # - state[8]: current grain size ~1-250 µm
        # - state[9]: target grain size ~5-25 µm
        self.obs_mean = np.array(
            [
                60.0,       # state[0]: thickness [mm]
                12.0,       # state[1]: step count
                30.0,       # state[2]: HR limit [mm]
                10.0,       # state[3]: target thickness [mm]
                1.5e6,      # state[4]: force [N]
                4.0e4,      # state[5]: torque [Nm]
                1025.0,     # state[6]: current temperature [K]
                1173.0,     # state[7]: target temperature [K]
                100.0,      # state[8]: current grain size [µm]
                15.0,       # state[9]: target grain size [µm]
            ], dtype=np.float32,
        )

        self.obs_std = np.array(
            [
                40.0,       # state[0]: thickness [mm]
                8.0,        # state[1]: step count
                10.0,       # state[2]: HR limit [mm]
                5.0,        # state[3]: target thickness [mm]
                1.0e6,      # state[4]: force [N]
                2.5e4,      # state[5]: torque [Nm]
                200.0,      # state[6]: current temperature [K]
                100.0,      # state[7]: target temperature [K]
                60.0,       # state[8]: current grain size [µm]
                8.0,        # state[9]: target grain size [µm]
            ], dtype=np.float32,
        )

        self.default_mode = env_config.get('mode', 'train')

        # Create dedicated RNG for this environment instance
        self.env_seed = env_config.get('env_seed', None)
        if self.env_seed is not None:
            self.rng = np.random.default_rng(self.env_seed)
            print(f'FlatRollingEnv initialized with seed: {self.env_seed}')
        else:
            if self.default_mode != 'inference':
                self.rng = np.random.default_rng()
                print('FlatRollingEnv initialized without seed (using random state)')

        # action[0]: 0.00 to 50.00 mm height reduction with 0.1 mm resolution
        # action[1]: 1 to 120 s inter-pass time (0 is masked as invalid)
        # action[2]: 0.083-0.50 m/s (5-30 m/min) rolling velocity (0 is masked as invalid)
        self.action_space = MultiDiscrete(list(self.ACTION_DIMS))

        # Observation space as Dict for action masking
        # "observations" contains the z-score normalized state vector
        # After normalization, values are approximately in range [-3, 3] for typical data
        # Using [-10, 10] bounds to accommodate outliers
        # "action_mask" contains binary mask for valid actions (1.0 = allowed, 0.0 = disallowed)
        self.observation_space = Dict({
            'observations': Box(
                # All observations are z-score normalized:
                # state[0]: Current thickness (normalized)
                # state[1]: Step count (normalized)
                # state[2]: Height reduction limit (normalized)
                # state[3]: Target thickness (normalized)
                # state[4]: Rolling force (normalized)
                # state[5]: Rolling torque (normalized)
                # state[6]: Current rolling stock temperature (normalized)
                # state[7]: Target temperature (normalized)
                # state[8]: Current material grain size (normalized)
                # state[9]: Target material grain size (normalized)
                low=np.array([-10.0] * 10, dtype=np.float32),
                high=np.array([10.0] * 10, dtype=np.float32),
                dtype=np.float32,
            ),
            'action_mask': Dict({
                'height_reduction': Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.ACTION_DIMS[0],),
                    dtype=np.float32,
                ),
                'interpass_time': Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.ACTION_DIMS[1],),
                    dtype=np.float32,
                ),
                'rolling_velocity': Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.ACTION_DIMS[2],),
                    dtype=np.float32,
                ),
            }),
        })

        # Initialize height reduction and inter-pass time lists
        self.pass_schedule_thickness_sequence_m: list[float] = []
        self.pass_schedule_interpass_sequence_s: list[float] = []
        self.pass_schedule_rolling_velocity_sequence_m_s: list[float] = []
        self.starting_thickness_mm = None
        self.starting_temperature_K = None  # pylint: disable=invalid-name
        self.starting_grain_size_um = None

        # Cached PyRoll objects for incremental simulation (avoids re-simulating
        # the entire pass schedule each step). Updated in reset() and calculate_state().
        self._cached_profile = None
        self._cached_roll = None

        assert self.render_mode is None or self.render_mode in self.metadata['render_modes']

    def _get_action_mask(self) -> np.ndarray:
        """
        Compute the action mask for both height reduction and interpass time
        based on physical constraints.

        Height reduction constraints:
        1. No action above hr_lim (state[2])
        2. No height reduction beyond 70% of current thickness (state[0])
        3. No action that brings thickness below target (state[3])
        Interpass time constraints:
        1. Interpass time must be at least 1 second

        Returns:
            dict: containing 'height_reduction' and 'interpass_time' masks
            in the form of np.ndarrays (Binary mask where 1.0 = action allowed,
            0.0 = action disallowed)
        """
        current_thickness_mm = self.state[0]
        hr_limit_mm = self.state[2]
        target_thickness_mm = self.state[3]

        # Vectorized height reduction mask: actions 0..500 map to 0.0..50.0 mm
        actions_mm = np.arange(self.ACTION_DIMS[0]) / self.HR_RESOLUTION
        hr_mask = (
            (
                actions_mm <= hr_limit_mm
            ) & (
                actions_mm <= current_thickness_mm * self.MAX_HR_FRACTION
            ) & (
                current_thickness_mm - actions_mm >= target_thickness_mm
            )
        ).astype(np.float32)

        # Ensure at least action 0 (no reduction) is always valid as a safety fallback
        # This prevents complete masking in edge cases
        hr_mask[0] = 1.0

        # Inter-pass time mask (typically all valid, therefore all ones)
        int_mask = np.ones(self.ACTION_DIMS[1], dtype=np.float32)
        int_mask[0] = 0.0  # inter-pass time of 0 seconds is disallowed

        # velocity mask (5-30 m/min with 5 m/min steps, 0 is disallowed)
        vel_mask = np.ones(self.ACTION_DIMS[2], dtype=np.float32)
        vel_mask[0] = 0.0  # rolling velocity of 0 m/s is disallowed

        return {
            'height_reduction': hr_mask,
            'interpass_time': int_mask,
            'rolling_velocity': vel_mask,
        }

    def _get_obs(self) -> dict:
        """
        Get the current observation of the environment.

        Observations are z-score normalized using pre-computed mean and std
        to bring all features to similar scales (approximately [-3, 3]).
        This is critical for the neural network to learn from all features,
        especially small-magnitude ones like target grain size.

        Returns:
            dict: Dictionary containing 'observations' (normalized state vector)
                  and 'action_mask'
        """
        # Z-score normalization: (x - mean) / std
        normalized_state = (self.state - self.obs_mean) / (self.obs_std + 1e-8)
        return {
            'observations': normalized_state.astype(np.float32),
            'action_mask': self._get_action_mask(),
        }

    def _get_info(self) -> dict:
        """
        Get additional information about the current state of the environment.

        Keys are defined by :attr:`INFO_SCHEMA` (single source of truth).

        Returns:
            dict: A dictionary containing additional information.
        """
        return {
            k: cast(self.state[idx])
            for k, (idx, cast, _, _) in self.INFO_SCHEMA.items()
        }

    def reset(      # pylint: disable=too-many-branches
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.
        Args:
            seed: Random seed for reproducibility
            options: Dict with optional keys:
                - 'mode': 'train' or 'inference' (defaults to self.default_mode)
                - 'initial_thickness': Starting thickness in mm (for inference mode)
                - 'target_thickness': Target thickness in mm (for inference mode)
                - 'hr_limit': Height reduction limit per pass (optional, defaults to 35.0)

        """
        super().reset(seed=seed)

        mode = 'train'
        if options is not None:
            mode = options.get('mode', self.default_mode)
        else:
            mode = self.default_mode

        if mode == 'inference':
            # Inference mode: use specified values
            if options is None:
                raise ValueError('options dict required for inference mode')

            initial_thickness = options.get('initial_thickness_mm', None)
            target_thickness = options.get('target_thickness_mm', None)
            hr_limit_mm = options.get('hr_limit_mm', None)
            initial_grain_size_um = options.get('initial_grain_size_um', None)
            target_grain_size_um = options.get('target_grain_size_um', None)
            initial_temperature_K = options.get(        # pylint: disable=invalid-name
                'initial_temperature_K', None,
            )
            target_temperature_K = options.get(         # pylint: disable=invalid-name
                'target_temperature_K', None,
            )

            if any([
                initial_thickness is None, target_thickness is None,
                initial_grain_size_um is None, target_grain_size_um is None,
                initial_temperature_K is None, target_temperature_K is None,
            ]):
                raise ValueError(
                    'options must specify all of the following parameters for inference mode: '
                    "'initial_thickness_mm', 'target_thickness_mm', 'initial_grain_size_um', "
                    "'target_grain_size_um', 'initial_temperature_K', 'target_temperature_K'",
                )

            if not 5 <= initial_thickness <= 150:
                raise ValueError(
                    f'initial_thickness {initial_thickness} out of range [5, 150 mm]',
                )
            if not 1 <= target_thickness <= 50:
                raise ValueError(
                    f'target_thickness {target_thickness} out of range [1, 50 mm]',
                )

            if not 100 <= initial_grain_size_um <= 500:
                raise ValueError(
                    f'initial_grain_size_um {initial_grain_size_um} out of range [100, 500 um]',
                )
            if not 5 <= target_grain_size_um <= 25:
                raise ValueError(
                    f'target_grain_size_um {target_grain_size_um} out of range [5, 25 um]',
                )
            if not 1273 <= initial_temperature_K <= 1523:
                raise ValueError(
                    f'initial_temperature_K {initial_temperature_K} out of range '
                    f'[1273, 1523 K] (1000-1250°C)',
                )
            if not 1073 <= target_temperature_K <= 1273:
                raise ValueError(
                    f'target_temperature_K {target_temperature_K} out of range '
                    f'[1073, 1273 K] (800-1000°C)',
                )

            # Validate hr_limit_mm if provided, otherwise use default of 60.0
            if hr_limit_mm is None:
                hr_limit_mm = 60.0
            elif not 1 <= hr_limit_mm <= 60:
                raise ValueError(
                    f'hr_limit_mm {hr_limit_mm} out of range [1, 60 mm]',
                )

            self.state = np.array(
                (
                    initial_thickness,
                    0.0,
                    hr_limit_mm,
                    target_thickness,
                    0.0,
                    0.0,
                    initial_temperature_K,
                    target_temperature_K,
                    initial_grain_size_um,
                    target_grain_size_um,
                ), dtype=np.float32,
            )
        else:
            # Training mode: using random values
            # Use instance RNG instead of global np.random for proper seeding
            self.state = np.array(
                (
                    # randomized starting thickness between 90 and 110 mm
                    self.rng.integers(90, 111),
                    0.0,
                    # randomized height reduction limit between 20 and 60 mm
                    self.rng.integers(20, 60),
                    # randomized target thickness between 5 and 15 mm
                    self.rng.integers(5, 16),
                    0.0,
                    0.0,
                    # randomized initial temperature between 1000 and 1250 C (1273 to 1523 K)
                    self.rng.integers(1273, 1524),
                    # randomized target temperature between 800 and 1000 C (1073 to 1273 K)
                    self.rng.integers(1073, 1274),
                    # randomized initial grain size between 100 and 500 µm
                    self.rng.integers(100, 501),
                    # randomized target grain size between 5 and 25 µm
                    self.rng.integers(5, 26),
                ), dtype=np.float32,
            )

        # reset height reduction and inter-pass time lists
        self.pass_schedule_thickness_sequence_m.clear()
        self.pass_schedule_interpass_sequence_s.clear()
        self.pass_schedule_rolling_velocity_sequence_m_s.clear()
        self.starting_thickness_mm = self.state[0].item()
        self.starting_temperature_K = self.state[6].item()
        self.starting_grain_size_um = self.state[8].item()

        # Create and cache the initial PyRoll profile and roll for incremental simulation
        self._cached_profile = create_in_profile(
            starting_thickness_m=self.starting_thickness_mm * 1e-3,
            starting_width_m=self.config.initial.width_m,
            starting_length_m=self.config.initial.length_m,
            starting_temperature_K=self.starting_temperature_K,
            starting_grain_size_m=self.starting_grain_size_um * 1e-6,
            specific_heat_capacity_j_kg_K=self.config.material.specific_heat_j_kg_K,
            density_kg_m3=self.config.material.density_kg_m3,
            flow_stress_coefficients=self.config.material.models.flow_stress,
            drx_params=self.config.material.models.drx,
            mdrx_params=self.config.material.models.mdrx,
            srx_params=self.config.material.models.srx,
            gg_params=self.config.material.models.grain_growth,
        )
        self._cached_roll = create_roll(self.config)

        return self._get_obs(), self._get_info()

    def calculate_state(  # pylint: disable=too-many-locals, too-many-statements
            self,
            prev_state,
            hr_mm,
            int_time_s,
            roll_vel_m_s,
            pass_schedule_thickness_sequence_m,
            pass_schedule_interpass_sequence_s,
            pass_schedule_rolling_velocity_sequence_m_s,
    ) -> np.ndarray:
        """
        Update the environment state based on the action taken.

        Args:
            prev_state: Previous state vector
            hr: Height reduction in mm
            int_time: Inter-pass time in seconds
            pass_schedule_thic: List of thickness values
            pass_schedule_inter: List of inter-pass times
        """

        new_state = prev_state.copy()
        thicknesses_m = pass_schedule_thickness_sequence_m.copy()
        interpass_times_s = pass_schedule_interpass_sequence_s.copy()
        rolling_velocities_m_s = pass_schedule_rolling_velocity_sequence_m_s.copy()

        # increment step count everytime this function is called
        new_state[1] += 1

        if hr_mm < 1:
            # Small hr breaks the simulation. Effects are negligible anyway. So, skip
            # the simulation. In a discrete action space, only hr of 0.0 triggers this.
            # state[0]: "current thickness" is unchanged since hr = 0.0
            # state[2]: "hr limit" does not change in the episode
            # state[3]: "target thickness" does not change in the episode
            new_state[4] = 0.0
            new_state[5] = 0.0
            # state[6]: "current rolling stock temperature" remains unchanged
            # state[7]: "target temperature" remains unchanged (static)
            # state[8]: "current material grain size" remains unchanged
            # state[9]: "target material grain size" remains unchanged (static)

            # Evolve the cached profile to account for cooling during this no-op.
            # The state vector is NOT updated (matching original behavior), but the
            # cached profile reflects the correct thermal state for subsequent passes.
            if self._cached_profile is not None and int_time_s > 0:
                try:
                    noop_seq = pr.PassSequence([
                        pr.Transport(
                            label=f'noop_transport_{int(new_state[1])}',
                            duration=int_time_s,
                        ),
                    ])
                    noop_seq.solve(self._cached_profile)
                    self._cached_profile = BaseProfile(
                        **{
                            k: v for k, v in noop_seq[-1].out_profile.__dict__.items()
                            if not k.startswith('_')
                        },
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass  # If Transport-only solve fails, keep existing cache

        else:

            # update thickness
            new_state[0] -= hr_mm
            # update pass schedule sequences
            pass_roll_gap_m = new_state[0] * 1e-3
            thicknesses_m.append(pass_roll_gap_m)
            interpass_times_s.append(int_time_s)
            rolling_velocities_m_s.append(roll_vel_m_s)

            # Log simulation start for debugging
            num_passes = len(thicknesses_m)
            start_time = time.time()
            logger.debug(
                'Starting simulation with %d passes, '
                'initial thickness: %.2fmm, target thickness: %.2fmm, '
                'hr: %.2fmm, int_time: %.2fs, current thickness: %.2fmm ',
                num_passes,
                self.starting_thickness_mm,
                new_state[3],
                hr_mm,
                int_time_s,
                new_state[0],
            )

            try:
                # Apply timeout protection to prevent indefinite hangs
                with timeout(65):  # 65 second timeout per simulation
                    # Incremental simulation: only simulate the new pass using
                    # the cached profile from the previous step's output.
                    rolling_sequence = flat_rolling_step_simulation(
                        in_profile=self._cached_profile,
                        roll=self._cached_roll,
                        roll_gap_m=pass_roll_gap_m,
                        interpass_time_s=int_time_s,
                        rolling_velocity_m_s=roll_vel_m_s,
                        pass_label_num=int(new_state[1]),
                    )

                # Log successful simulation
                elapsed = time.time() - start_time
                logger.debug('Simulation completed in %.2fs', elapsed)

                # Extract results from simulation
                # Thickness, force, torque come from the RollPass [-2]
                sim_thickness_mm = rolling_sequence[-2].out_profile.equivalent_height * 1e3
                sim_force_N = rolling_sequence[-2].roll_force                                   # noqa: E501 # pylint: disable=invalid-name, line-too-long
                sim_torque_Nm = roll_torque(rolling_sequence[-2])                               # noqa: E501 # pylint: disable=invalid-name, line-too-long
                # Temperature and grain size come from the Transport [-1]
                # to include interpass cooling and recrystallization effects
                sim_temperature_K = rolling_sequence[-1].out_profile.temperature                # noqa: E501 # pylint: disable=invalid-name, line-too-long
                sim_grain_size_m = rolling_sequence[-1].out_profile.grain_size

                # Validate simulation results for NaN or infinite values
                if (
                    not np.isfinite(sim_thickness_mm) or not np.isfinite(sim_force_N) or not np.isfinite(sim_torque_Nm)  # noqa: E501 # pylint: disable=line-too-long
                ):
                    logger.warning(
                        'Simulation returned invalid values (NaN/Inf) at pass %d. '
                        'thickness_m=%.2f, force_N=%.2f, torque_Nm=%.2f,'
                        ' temperature_K=%.2f, grain_size_um=%.2f. '
                        'Keeping previous state and applying penalty.',
                        num_passes, sim_thickness_mm, sim_force_N, sim_torque_Nm,
                        sim_temperature_K, sim_grain_size_m * 1e6,
                    )
                    # Don't update thickness - keep previous state
                    # Revert thickness change
                    new_state[0] = prev_state[0]
                    new_state[4] = -100.0  # Signal failure - force
                    new_state[5] = -100.0  # Signal failure - torque
                    new_state[6] = -100.0  # Signal failure - temperature
                    # state[7] is target_temperature - static, no update
                    new_state[8] = -100.0  # Signal failure - grain size
                    thicknesses_m.pop()  # Remove invalid thickness from schedule
                    interpass_times_s.pop()
                    rolling_velocities_m_s.pop()
                    # Do NOT update _cached_profile — retain last good state
                else:
                    # Update state with ACTUAL thickness from simulation
                    new_state[0] = sim_thickness_mm
                    new_state[4] = sim_force_N
                    new_state[5] = sim_torque_Nm
                    new_state[6] = sim_temperature_K
                    # state[7] is target_temperature - static, no update
                    new_state[8] = sim_grain_size_m * 1e6  # Convert m to µm
                    # Cache the Transport's out_profile for the next step
                    self._cached_profile = BaseProfile(
                        **{
                            k: v for k, v in rolling_sequence[-1].out_profile.__dict__.items()
                            if not k.startswith('_')
                        },
                    )

            except TimeoutError as e:
                # Simulation timed out - keep previous state and apply penalty
                elapsed = time.time() - start_time
                logger.warning(
                    'Simulation timeout after %.2fs at pass %d. '
                    'Action: [hr=%.2fmm, interpass_time=%.2fs], thickness=%.2fmm. '
                    'Initial thickness=%.2fmm, target thickness=%.2fmm. '
                    'Keeping previous state and applying penalty. Error: %s',
                    elapsed,
                    num_passes,
                    hr_mm,
                    int_time_s,
                    prev_state[0],
                    self.starting_thickness_mm,
                    new_state[3],
                    e,
                )
                # Don't update thickness - keep previous state
                new_state[0] = prev_state[0]  # Revert thickness change
                new_state[4] = -100.0  # Signal failure - force
                new_state[5] = -100.0  # Signal failure - torque
                new_state[6] = -100.0  # Signal failure - temperature
                # state[7] is target_temperature - static, no update
                new_state[8] = -100.0  # Signal failure - grain size

                # Remove the last added pass from schedule since simulation failed
                thicknesses_m.pop()
                interpass_times_s.pop()
                rolling_velocities_m_s.pop()
                # Do NOT update _cached_profile — retain last good state

            except Exception as e:  # pylint: disable=broad-exception-caught
                # Simulation failed for other reasons - keep previous state and penalize
                # Note: We catch broad Exception here intentionally to prevent any
                # simulation failure from hanging the training process
                elapsed = time.time() - start_time
                logger.warning(
                    'Simulation failed after %.2fs at pass %d. '
                    'Initial thickness=%.2fmm, target thickness=%.2fmm. '
                    'Action: [hr=%.2fmm, interpass_time=%.2fs], thickness=%.2fmm. '
                    'Keeping previous state and applying penalty. Error: %s: %s',
                    elapsed,
                    num_passes,
                    self.starting_thickness_mm,
                    new_state[3],
                    hr_mm,
                    int_time_s,
                    prev_state[0],
                    type(
                        e,
                    ).__name__, e,
                )
                # Don't update thickness - keep previous state
                new_state[0] = prev_state[0]  # Revert thickness change
                new_state[4] = -100.0  # Signal failure - force
                new_state[5] = -100.0  # Signal failure - torque
                new_state[6] = -100.0  # Signal failure - temperature
                # state[7] is target_temperature - static, no update
                new_state[8] = -100.0  # Signal failure - grain size
                # Remove the last added pass from schedule since simulation failed
                thicknesses_m.pop()
                interpass_times_s.pop()
                rolling_velocities_m_s.pop()
                # Do NOT update _cached_profile — retain last good state

        return new_state, thicknesses_m, interpass_times_s, rolling_velocities_m_s

    def step(self, action: np.float32) -> tuple[  # pylint: disable=too-many-locals
        dict, np.float32, bool, bool, dict[str, np.float32],
    ]:
        """
        Take a step in the environment based on the action.

        Args:
            action: np.ndarray of shape (2,) with [hr_action, time_action]
        """

        assert self.state is not None, 'Call reset before using step method.'

        previous_state = self.state.copy()
        # action[0] is in [0, 500], height reduction in [0, 50] mm.
        height_reduction_mm = int(action[0]) / self.HR_RESOLUTION
        # action[1] is in [1, 120]  (0 is masked), interpass time in [1, 120] seconds
        interpass_time_s = int(action[1])
        # action[2] is in [1, 6] (0 is masked), rolling velocity in [0.083, 0.50] m/s
        roll_velocity_m_s = int(action[2]) / self.VELOCITY_DIVISOR

        (
            new_state, new_schedule_thick_m, new_schedule_inter_s, new_rolling_velocities_m_s,
        ) = self.calculate_state(
            previous_state,
            height_reduction_mm,
            interpass_time_s,
            roll_velocity_m_s,
            self.pass_schedule_thickness_sequence_m,
            self.pass_schedule_interpass_sequence_s,
            self.pass_schedule_rolling_velocity_sequence_m_s,
        )

        state_checks = evaluate_state(
            new_state, height_reduction_mm, self.config,
        )

        terminated = state_checks[2]            # completed

        reward, reward_details = calculate_reward(
            new_state,
            previous_grain_size_um=self.state[8],
            height_reduction_mm=height_reduction_mm,
            completed=terminated,
            config=self.config,
        )
        truncated = bool(new_state[1] >= self.MAX_STEPS)  # max steps per episode

        # update the environment state and the pass_schedule
        self.state = new_state
        self.pass_schedule_thickness_sequence_m = new_schedule_thick_m
        self.pass_schedule_interpass_sequence_s = new_schedule_inter_s
        self.pass_schedule_rolling_velocity_sequence_m_s = new_rolling_velocities_m_s

        # Combine reward details with raw state values in info dict
        info = reward_details.copy()
        info.update(self._get_info())

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        """
        Think about showing the current agent decisions and state of the environment using render().
        """
        return None

    def close(self):
        """Clean up resources used by the environment."""

        self.pass_schedule_thickness_sequence_m.clear()
        self.pass_schedule_interpass_sequence_s.clear()
        self.pass_schedule_rolling_velocity_sequence_m_s.clear()
        self._cached_profile = None
        self._cached_roll = None
        self.state = None

    # ==================== LLM Prompt Description Methods ====================
    # Each method returns a markdown section derived from class constants and
    # config values — single source of truth, no hardcoded duplicates.

    @classmethod
    def describe_info_dict(cls) -> str:
        """Describe the info dict keys, units, and meanings from :attr:`INFO_SCHEMA`."""
        rows = '\n'.join(
            f'| {key:<19} | {unit:<4} | {desc:<40} |'
            for key, (_, _, unit, desc) in cls.INFO_SCHEMA.items()
        )
        return (
            '### info dict (raw physical units, NOT normalized)\n'
            '\n'
            '| Key                 | Unit | Description                              |\n'
            '|---------------------|------|------------------------------------------|\n'
            f'{rows}\n'
        )

    @classmethod
    def describe_action_space(cls) -> str:
        """Describe action masks, return format, and index-to-physical conversions."""
        hr, ipt, vel = cls.ACTION_DIMS
        vel_max_idx = vel - 1
        return (
            '### action_mask dict (boolean arrays, 1.0 = allowed, 0.0 = blocked)\n'
            '\n'
            '| Key                | Shape    | Meaning                                          |\n'
            '|--------------------|----------|--------------------------------------------------|\n'
            f'| height_reduction   | ({hr},) '
            f'| Index i -> i/{cls.HR_RESOLUTION:.1f} mm reduction. Index 0 = 0mm.   |\n'
            f'| interpass_time     | ({ipt},) '
            f'| Index i -> i seconds. Index 0 is always masked.   |\n'
            f'| rolling_velocity   | ({vel},)   '
            f'| Index i -> i/{cls.VELOCITY_DIVISOR:.1f} m/s. Index 0 is always masked.  |\n'
            '\n'
            '### What your function must return\n'
            '\n'
            'A list of 3 integers: [hr_action, interpass_action, velocity_action]\n'
            '\n'
            '## Action -> Physical Value Conversions\n'
            '\n'
            f'- **Height reduction**: hr_mm = action[0] / {cls.HR_RESOLUTION:.1f}\n'
            f'  To get X mm reduction, return int(X * {cls.HR_RESOLUTION:.0f}). E.g. 25mm -> 250.\n'
            '- **Interpass time**: interpass_s = action[1]\n'
            f'  Direct seconds. Return an int from 1 to {ipt - 1}.\n'
            '- **Rolling velocity**: velocity_m_s = action[2] / {:.1f}\n'.format(cls.VELOCITY_DIVISOR)
            + f'  Index 1 = {1 / cls.VELOCITY_DIVISOR:.3f} m/s (5 m/min), '
            f'index {vel_max_idx} = {vel_max_idx / cls.VELOCITY_DIVISOR:.1f} m/s (30 m/min).\n'
        )

    @classmethod
    def describe_action_mask_constraints(cls) -> str:
        """Describe the physical constraints that shape the action mask."""
        pct = int(cls.MAX_HR_FRACTION * 100)
        return (
            '## Action Mask Constraints (what limits the HR mask)\n'
            '\n'
            '- HR cannot exceed hr_limit (from info dict)\n'
            f'- HR cannot exceed {pct}% of current thickness\n'
            '- HR cannot bring thickness below target\n'
            '- Index 0 (no reduction) is always valid as a safety fallback\n'
            '- Interpass time index 0 is always masked (minimum 1 second)\n'
            f'- Velocity index 0 is always masked (minimum {1 / cls.VELOCITY_DIVISOR:.3f} m/s)\n'
        )

    @classmethod
    def describe_reward_structure(cls) -> str:
        """Describe the 5 reward components from :mod:`state_evaluation` constants."""
        from rollinggym.env.state_evaluation import (  # noqa: C0415
            GS_COMPLETION_MAX_BONUS,
            GS_COMPLETION_TOLERANCE_UM,
            HR_MAX_BONUS,
            HR_MAX_PENALTY,
            STEP_PENALTY,
            TEMP_COMPLETION_MAX_BONUS,
            TEMP_COMPLETION_TOLERANCE_K,
        )
        return (
            '## Reward Structure (exact values)\n'
            '\n'
            'Each step produces 5 reward components:\n'
            '\n'
            f'1. **Step penalty**: {STEP_PENALTY} every step. Fewer passes = better.\n'
            '2. **GS progress bonus**: Rewards reducing grain size toward target.\n'
            '   Severely penalizes overshooting below target (-5.0 to -10.0 penalty).\n'
            f'3. **HR efficiency bonus**: 0 to {HR_MAX_BONUS:.0f} for large reductions within equipment limits.\n'
            f'   {HR_MAX_PENALTY} penalty if force or torque exceeded.\n'
            f'4. **GS completion bonus**: 0 to {GS_COMPLETION_MAX_BONUS:.0f} at episode end, proportional to grain size\n'
            f'   accuracy. Tolerance: +/- {GS_COMPLETION_TOLERANCE_UM:.0f} um.\n'
            f'5. **Temperature completion bonus**: 0 to {TEMP_COMPLETION_MAX_BONUS:.0f} at episode end, proportional to\n'
            f'   temperature accuracy. Tolerance: +/- {TEMP_COMPLETION_TOLERANCE_K:.0f} K.\n'
        )

    @classmethod
    def describe_equipment_limits(cls, config: 'EnvConfig') -> str:
        """Describe equipment limits derived from a live config."""
        force = config.equipment.force_N
        torque = config.equipment.torque_Nm
        return (
            '## Equipment Limits\n'
            '\n'
            f'- Force limit: {force:,.0f} N ({force / 1e6:.0f} MN)\n'
            f'- Torque limit: {torque:,.0f} Nm ({torque / 1e3:.0f} kNm per roll)\n'
        )

    @classmethod
    def describe_termination(cls, config: 'EnvConfig') -> str:
        """Describe episode termination conditions."""
        tolerance_mm = config.target.tolerance_m * 1e3
        return (
            '## Episode Termination\n'
            '\n'
            f'- **Success**: |current_thickness - target_thickness| <= {tolerance_mm:.0f} mm\n'
            f'- **Truncation**: {cls.MAX_STEPS} steps reached (bad outcome)\n'
        )

    @staticmethod
    def describe_domain_knowledge() -> str:
        """Describe metallurgical domain knowledge for hot rolling of S355 steel."""
        return (
            '## Domain Knowledge: Hot Rolling of S355 Steel\n'
            '\n'
            '- Higher temperature -> lower flow stress -> lower forces and torques\n'
            '- Heavier reductions -> more dynamic recrystallization -> finer grain size\n'
            '- Longer interpass time -> more static recrystallization + grain growth + cooling\n'
            '- Temperature drops roughly 20-80 K per pass depending on interpass time and\n'
            '  deformation heating\n'
            '- Grain refinement is most effective with heavy reductions at high temperatures\n'
            '- As thickness decreases, the same mm-reduction represents a larger % strain,\n'
            '  causing higher forces\n'
            '- Near the target: switch to small reductions for precise thickness control\n'
            '- Rolling velocity affects strain rate: higher velocity -> higher strain rate\n'
            '  -> higher flow stress -> higher forces, but also more deformation heating\n'
        )

    @classmethod
    def describe_source_code(cls) -> str:
        """Return source code of key env methods for LLM reference."""
        import inspect  # noqa: C0415
        methods = [
            cls._get_info,
            cls._get_action_mask,
            cls.step,
            cls.calculate_state,
            cls.reset,
        ]
        parts = []
        for method in methods:
            name = method.__name__
            source = inspect.getsource(method)
            parts.append(f'### FlatRollingEnv.{name}\n```python\n{source}```')
        return '\n\n'.join(parts) + '\n'

    @classmethod
    def describe_reward_source_code(cls) -> str:
        """Return source code of reward functions for LLM reference."""
        import inspect  # noqa: C0415
        from rollinggym.env.state_evaluation import (  # noqa: C0415
            calculate_completion_bonus,
            calculate_grain_size_progress_bonus,
            calculate_hr_efficiency_bonus,
            calculate_reward,
        )
        funcs = [
            calculate_reward,
            calculate_completion_bonus,
            calculate_hr_efficiency_bonus,
            calculate_grain_size_progress_bonus,
        ]
        parts = []
        for func in funcs:
            source = inspect.getsource(func)
            parts.append(f'### {func.__name__}\n```python\n{source}```')
        return '\n\n'.join(parts) + '\n'

    @classmethod
    def describe_all(cls, config: 'EnvConfig') -> str:
        """Return all description sections joined together."""
        return '\n'.join([
            cls.describe_info_dict(),
            cls.describe_action_space(),
            cls.describe_action_mask_constraints(),
            cls.describe_reward_structure(),
            cls.describe_equipment_limits(config),
            cls.describe_termination(config),
            cls.describe_domain_knowledge(),
            '## Source Code Reference\n',
            cls.describe_source_code(),
            '## Reward Functions Source Code\n',
            cls.describe_reward_source_code(),
        ])
