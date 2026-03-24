"""
State evaluation and reward calculation for the flat rolling environment.

State evaluations:
    state_evals[0] = hr_lim_exceeded
    state_evals[1] = overshoot
    state_evals[2] = completed
    state_evals[3] = excessive_force
    state_evals[4] = excessive_torque
    state_evals[5] = simulation_failed

State vector:
    state[0]: Current thickness [mm]
    state[1]: Step count [-] (for truncation at 25 steps)
    state[2]: Height reduction limit [mm]
    state[3]: Target thickness [mm]
    state[4]: Rolling force [N]
    state[5]: Rolling torque [Nm]
    state[6]: Current rolling stock temperature [K]
    state[7]: Target temperature [K]
    state[8]: Current material grain size [µm]
    state[9]: Target material grain size [µm]
"""
from __future__ import annotations

import logging

from rollinggym.env.rolling_env_config import EnvConfig

logger = logging.getLogger(__name__)

# ==================== Reward Constants ====================
# Single source of truth for reward values used in calculate_reward()
# and referenced by FlatRollingEnv.describe_reward_structure().

STEP_PENALTY: float = -5.0

HR_MAX_BONUS: float = 10.0
HR_MAX_PENALTY: float = -5.0

GS_COMPLETION_MAX_BONUS: float = 25.0
GS_COMPLETION_MAX_PENALTY: float = -2.0
GS_COMPLETION_TOLERANCE_UM: float = 20.0

TEMP_COMPLETION_MAX_BONUS: float = 25.0
TEMP_COMPLETION_MAX_PENALTY: float = -2.0
TEMP_COMPLETION_TOLERANCE_K: float = 50.0


def calculate_grain_size_progress_bonus(
    current_grain_size_um: float,
    previous_grain_size_um: float,
    target_grain_size_um: float,
    weight: float = 1.0,
    undershoot_penalty_factor: float = 10.0,
) -> float:
    """
    Asymmetric progress reward: Rewards approach, severely penalizes crossing below target.
    This scheme:
    - Rewards progress toward target when above it
    - Gives small reward for staying near target
    - Severely penalizes any move that takes grain size below target
    - Extra penalty for continuing to decrease when already below target
    Args:
        current_grain_size_um: Current grain size after this step
        previous_grain_size_um: Grain size before this step
        target_grain_size_um: Target grain size
        weight: Scaling factor for the reward
        undershoot_penalty_factor: Multiplier for undershoot penalties
    Returns:
        Progress-based reward with asymmetric penalties
    """
    prev_above_target = previous_grain_size_um >= target_grain_size_um
    curr_above_target = current_grain_size_um >= target_grain_size_um
    # Case 1: Both above target - reward for getting closer
    if prev_above_target and curr_above_target:
        improvement = previous_grain_size_um - current_grain_size_um
        return weight * (improvement / target_grain_size_um)
    # Case 2: Was above, now below - crossed the target (overshoot!)
    if prev_above_target and not curr_above_target:
        # Penalize proportional to how far below we went
        overshoot = target_grain_size_um - current_grain_size_um
        penalty = undershoot_penalty_factor * \
            weight * (overshoot / target_grain_size_um)
        return max(-5.0, -penalty)
    # Case 3: Was below, still below
    if not prev_above_target and not curr_above_target:
        if current_grain_size_um < previous_grain_size_um:
            # Getting worse (further below target) - severe penalty
            worsening = previous_grain_size_um - current_grain_size_um
            penalty = undershoot_penalty_factor * \
                weight * (worsening / target_grain_size_um)
            return max(-5.0, -penalty)
        # Recovering toward target - small reward
        recovery = current_grain_size_um - previous_grain_size_um
        return 0.5 * weight * (recovery / target_grain_size_um)
    # Case 4: Was below, now above (recovering) - reward
    if not prev_above_target and curr_above_target:
        return weight * 0.5  # Bonus for recovering from undershoot
    return 0.0


def calculate_hr_efficiency_bonus(
    height_reduction_mm: float,
    hr_limit_mm: float,
    force: float,
    force_limit: float,
    torque: float,
    torque_limit: float,
    max_bonus: float = 10.0,
    max_penalty: float = -5.0,
) -> float:
    """
    Reward high HR achievement within equipment constraints.

    This function directly rewards taking large height reductions rather than
    high force/torque utilization. This prevents gaming via cooling (which
    increases force but doesn't increase HR).

    Args:
        height_reduction_mm: Height reduction taken in this pass [mm]
        hr_limit_mm: Maximum allowed height reduction [mm]
        force: Actual rolling force [N]
        force_limit: Equipment force limit [N]
        torque: Actual rolling torque [Nm]
        torque_limit: Equipment torque limit [Nm]

    Returns:
        Bonus in range [-5, 10] based on HR achievement and safety margin
    """
    assert max_bonus > 0.0, 'max_bonus must be positive'
    assert max_penalty < 0.0, 'max_penalty must be negative'

    if height_reduction_mm < 0.1:
        return 0.0  # No-op pass, no bonus

    # Check for simulation failure sentinel values (-100.0)
    if force < 0 or torque < 0:
        return max_penalty  # Penalize simulation failures

    # HR achievement ratio (how much of the allowed HR did we use?)
    hr_ratio = height_reduction_mm / hr_limit_mm

    # Equipment safety margin (penalize getting too close to limits)
    force_margin = 1.0 - (force / force_limit)
    torque_margin = 1.0 - (torque / torque_limit)
    safety_factor = min(force_margin, torque_margin)

    # Reward high HR, but penalize if approaching limits
    if safety_factor > 0.1:  # At least 10% safety margin
        return max_bonus * hr_ratio  # Max bonus for using full HR limit
    if safety_factor > 0:     # Between 0-10% margin
        return (max_bonus * 0.5) * hr_ratio  # Reduced reward
    # safety_factor <= 0: Exceeded limits
    return max_penalty


def calculate_completion_bonus(
    current_param: float,
    target_param: float,
    completed: bool,
    max_bonus: float,
    max_penalty: float,
    tolerance: float,
) -> float:
    """
    Bonus for reaching target parameter when episode completes.

    Uses a continuous reward function:
    - Within tolerance: linearly scales from max_bonus (at target) to 0 (at tolerance edge)
    - Outside tolerance: linearly scales from 0 (at tolerance edge) to
    max_penalty (at 2x tolerance)

    Args:
        current_param: Final parameter value
        target_param: Target parameter value
        completed: Whether the episode completed (reached target thickness)
        tolerance: Acceptable deviation from target
        max_bonus: Maximum positive bonus when at target
        max_penalty: Maximum negative penalty when far outside tolerance

    Returns:
        Bonus in range [max_penalty, max_bonus] if completed, 0 otherwise
    """
    assert max_bonus > 0.0, 'max_bonus must be positive'
    assert max_penalty < 0.0, 'max_penalty must be negative'

    if not completed:
        return 0.0

    deviation = abs(current_param - target_param)

    if deviation <= tolerance:
        # Within tolerance: reward based on accuracy (25 at target, 0 at edge)
        return max_bonus * (1.0 - deviation / tolerance)

    # Outside tolerance: smooth penalty that continues from 0 at boundary
    overshoot_ratio = (deviation - tolerance) / tolerance
    return max_penalty * min(1.0, overshoot_ratio)


def evaluate_state(current_state, hr_mm, config: EnvConfig) -> list:
    """
    Check the current environment state against operational constraints.

    Args:
        current_state: 10-element state vector:
            [thickness_mm, step_count, hr_limit_mm, target_thickness_mm,
             force_N, torque_Nm, temperature_K, target_temperature_K,
             grain_size_um, target_grain_size_um]
        hr_mm: Height reduction taken in this pass [mm].
        config: Environment configuration (used for equipment limits and tolerances).

    Returns:
        list[bool] of length 6 in fixed order:
            [hr_lim_exceeded, overshoot, completed, excessive_force,
             excessive_torque, simulation_failed]
        Simulation failure is indicated by sentinel value -100.0 in force or torque.
        Excessive force/torque is triggered at 90% of the equipment limit.
    """
    # Simulation failure is indicated by -100 sentinel values
    simulation_failed = bool(
        current_state[4] == -100.0 or current_state[5] == -100.0,
    )

    # Height reduction [mm] cannot exceed what the pass-through condition allows
    hr_lim_exceeded = bool(hr_mm > current_state[2])

    # Height reduction more than the tolerance around the target thickness
    overshoot = bool(
        current_state[0] <= current_state[3] - (config.target.tolerance_m * 1e3),  # noqa: E501 # pylint: disable=line-too-long
    )

    # Force exceeding 90% of the device limit
    excessive_force = bool(
        current_state[4] >= config.equipment.force_N * 0.9,
    )

    # Torque exceeding 90% of the device limit
    excessive_torque = bool(
        current_state[5] >= config.equipment.torque_Nm * 0.9,
    )

    # Termination when the target thickness is reached
    completed = bool(
        abs(
            current_state[0] - current_state[3],
        ) <= config.target.tolerance_m * 1e3,
    )

    eval_flags = [
        hr_lim_exceeded,
        overshoot,
        completed,
        excessive_force,
        excessive_torque,
        simulation_failed,
    ]

    flag_names = [
        'hr_lim_exceeded',
        'overshoot',
        'completed',
        'excessive_force',
        'excessive_torque',
        'simulation_failed',
    ]

    for flag_name, flag_value in zip(flag_names, eval_flags):
        if flag_value and flag_name != 'completed':
            logger.warning('State evaluation flag triggered: %s', flag_name)

    return eval_flags


def calculate_reward(
        current_state,
        previous_grain_size_um: float,
        height_reduction_mm: float,
        completed: bool,
        config: EnvConfig,
) -> tuple[float, dict[str, float | str]]:
    """
    Calculate the multi-component reward for the current step.

    Args:
        current_state: 10-element state vector:
            [thickness_mm, step_count, hr_limit_mm, target_thickness_mm,
             force_N, torque_Nm, temperature_K, target_temperature_K,
             grain_size_um, target_grain_size_um]
        previous_grain_size_um: Grain size before this step [µm].
        height_reduction_mm: Height reduction taken in this pass [mm].
        completed: Whether the episode completed (reached target thickness).
        config: Environment configuration (used for equipment limits).

    Returns:
        tuple[float, dict[str, float]]:
            - total_reward: Sum of all reward components.
            - reward_details: Dict with keys:
                'gs_progress_bonus', 'hr_efficiency_bonus', 'gs_completion_bonus',
                'temperature_completion_bonus', 'step_penalty'
    """
    gs_progress_bonus = calculate_grain_size_progress_bonus(
        current_grain_size_um=current_state[8],
        previous_grain_size_um=previous_grain_size_um,
        target_grain_size_um=current_state[9],
    )

    hr_efficiency_bonus = calculate_hr_efficiency_bonus(
        height_reduction_mm=height_reduction_mm,
        hr_limit_mm=current_state[2],
        force=current_state[4],
        force_limit=config.equipment.force_N,
        torque=current_state[5],
        torque_limit=config.equipment.torque_Nm,
        max_bonus=HR_MAX_BONUS,
        max_penalty=HR_MAX_PENALTY,
    )

    gs_completion_bonus = calculate_completion_bonus(
        current_param=current_state[8],   # Current grain size [µm]
        target_param=current_state[9],    # Target grain size [µm]
        completed=completed,
        max_bonus=GS_COMPLETION_MAX_BONUS,
        max_penalty=GS_COMPLETION_MAX_PENALTY,
        tolerance=GS_COMPLETION_TOLERANCE_UM,
    )

    temperature_completion_bonus = calculate_completion_bonus(
        current_param=current_state[6],   # Current temperature [K]
        target_param=current_state[7],    # Target temperature [K]
        completed=completed,
        max_bonus=TEMP_COMPLETION_MAX_BONUS,
        max_penalty=TEMP_COMPLETION_MAX_PENALTY,
        tolerance=TEMP_COMPLETION_TOLERANCE_K,
    )

    step_penalty = STEP_PENALTY

    total = gs_progress_bonus + hr_efficiency_bonus + \
        gs_completion_bonus + temperature_completion_bonus + step_penalty

    return (
        float(total),
        {
            'gs_progress_bonus': gs_progress_bonus,
            'hr_efficiency_bonus': hr_efficiency_bonus,
            'gs_completion_bonus': gs_completion_bonus,
            'temperature_completion_bonus': temperature_completion_bonus,
            'step_penalty': step_penalty,
        },
    )
