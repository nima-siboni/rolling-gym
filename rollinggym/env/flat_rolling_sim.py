"""
Flat rolling simulation built using PyRoll.

Official plugins can be found at: https://pyroll.readthedocs.io/en/latest/plugins/index.html

Default Physical Parameters
---------------------------
The following default values are used when optional parameters are not specified:

- **Roll temperature**: 20°C (293.15 K)
    Typical ambient temperature for work rolls in hot rolling.

These defaults are suitable for research and laboratory-scale simulations.
For production simulations, provide explicit values matching your mill specifications.
"""
from __future__ import annotations

import logging
import sys

import pyroll.core as pr
import pyroll.integral_thermal                          # pylint: disable=unused-import
import pyroll.lippmann_mahrenholz_force_torque          # pylint: disable=unused-import
import pyroll.report                                    # pylint: disable=unused-import
import pyroll.wusatowski_spreading                      # pylint: disable=unused-import
import pyroll.zouhar_contact                            # pylint: disable=unused-import

import rollinggym.env.pyroll_plugins  # pylint: disable=unused-import

# Default simulation parameters
DEFAULT_ROLL_TEMPERATURE_K = 20 + 273.15


def flat_rolling_simulation(
        in_profile: pr.BoxProfile | None = None,
        roll: pr.Roll | None = None,
        roll_gap_sequence_m: list[float] | None = None,
        interpass_time_sequence_s: list[float] | None = None,
        rolling_velocity_sequence_m_s: list[float] | None = None,
        roll_temperature_sequence_K: list[float] | None = None,  # noqa: E501 # pylint: disable=invalid-name, line-too-long
) -> pr.PassSequence:
    """
    Simulate a flat rolling process using PyRoll.

    This function sets up the pass schedule by taking sequences of roll gaps,
    interpass times, roll temperatures, and rolling velocities.

    Args:
        in_profile: Input material profile (BoxProfile for flat rolling).
        roll: Roll configuration including geometry and material properties.
        roll_gap_sequence_m: Target roll gaps for each pass [m].
        interpass_time_sequence_s: Time between passes [s].
        roll_temperature_sequence_K: Roll surface temperature for each pass [K].
            Defaults to 293.15 K (20°C) if not provided.
        rolling_velocity_sequence_m_s: Rolling velocity for each pass [m/s].

    Returns:
        PassSequence containing all roll passes and transport phases.

    Raises:
        AssertionError: If required parameters are missing or sequences have mismatched lengths.
        RuntimeError: If PyRoll solver fails during simulation.

    Note:
        All sequences must have equal lengths. Any arbitrary number of passes can be defined.
    """

    assert roll_gap_sequence_m is not None, \
        'Roll gap sequence must be provided.'
    assert interpass_time_sequence_s is not None, \
        'Interpass time sequence must be provided.'
    assert rolling_velocity_sequence_m_s is not None, \
        'Rolling velocity sequence must be provided.'
    if not roll_temperature_sequence_K:
        roll_temperature_sequence_K = \
            [
                DEFAULT_ROLL_TEMPERATURE_K for _ in range(
                    len(roll_gap_sequence_m),
                )
            ]
    assert isinstance(in_profile, pr.BoxProfile), \
        'Input profile must be a box profile for flat rolling simulation.'
    assert isinstance(roll, pr.Roll), \
        'Roll must be an instance of pyroll.core.Roll.'
    assert len(roll_gap_sequence_m) \
        == len(interpass_time_sequence_s) \
        == len(roll_temperature_sequence_K) \
        == len(rolling_velocity_sequence_m_s), \
        'Sequences must have equal lengths.'

    logging.basicConfig(
        stream=sys.stdout,
        format='[%(levelname)s] %(name)s: %(message)s',
    )
    # NOTSET = 0, DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50,
    logging.getLogger('pyroll').setLevel(logging.ERROR)

    pass_schedule = []
    for pass_num, roll_gap_m in enumerate(roll_gap_sequence_m):
        pass_schedule.append(
            pr.RollPass(
                label=f'flat_{pass_num + 1}',
                roll=roll,
                gap=roll_gap_m,
                temperature=roll_temperature_sequence_K[pass_num],
                velocity=rolling_velocity_sequence_m_s[pass_num],
            ),
        )
        try:
            pass_schedule.append(
                pr.Transport(
                    label=f'transport_{pass_num + 1}',
                    duration=interpass_time_sequence_s[pass_num],
                ),
            )
        except IndexError:
            pass

    sequence = pr.PassSequence(pass_schedule)

    try:
        sequence.solve(in_profile)
    except Exception as e:
        # Log detailed error information for debugging
        logging.error(
            'PyRoll solver failed with %d passes. '
            'Roll gaps: %s, '
            'Error: %s: %s',
            len(roll_gap_sequence_m), roll_gap_sequence_m, type(e).__name__, e,
        )
        # Re-raise the exception with additional context
        raise RuntimeError(
            f'PyRoll simulation solver failed after {len(roll_gap_sequence_m)} passes: {e}',
        ) from e

    return sequence


def flat_rolling_step_simulation(
        in_profile,
        roll: pr.Roll,
        roll_gap_m: float,
        interpass_time_s: float,
        rolling_velocity_m_s: float,
        roll_temperature_K: float = DEFAULT_ROLL_TEMPERATURE_K,  # noqa: E501 # pylint: disable=invalid-name, line-too-long
        pass_label_num: int = 1,
) -> pr.PassSequence:
    """
    Simulate a single rolling pass followed by a transport phase (incremental).

    This function is used for incremental simulation where only the latest pass
    needs to be simulated, using the cached output profile from the previous step
    as the input profile. This avoids re-simulating the entire pass schedule.

    Args:
        in_profile: Input profile from the previous step's cached output.
        roll: Roll configuration including geometry and material properties.
        roll_gap_m: Target roll gap for this pass [m].
        interpass_time_s: Time after this pass [s].
        rolling_velocity_m_s: Rolling velocity for this pass [m/s].
        roll_temperature_K: Roll surface temperature [K]. Defaults to 293.15 K.
        pass_label_num: Pass number for labeling.

    Returns:
        PassSequence of length 2: index [-2] is the RollPass (force, torque, thickness),
        index [-1] is the Transport (post-cooling temperature and grain size).

    Raises:
        RuntimeError: If PyRoll solver fails during simulation.
    """

    logging.getLogger('pyroll').setLevel(logging.ERROR)

    pass_schedule = [
        pr.RollPass(
            label=f'flat_{pass_label_num}',
            roll=roll,
            gap=roll_gap_m,
            temperature=roll_temperature_K,
            velocity=rolling_velocity_m_s,
        ),
        pr.Transport(
            label=f'transport_{pass_label_num}',
            duration=interpass_time_s,
        ),
    ]

    sequence = pr.PassSequence(pass_schedule)

    try:
        sequence.solve(in_profile)
    except Exception as e:
        logging.error(
            'PyRoll step simulation failed at pass %d. '
            'Roll gap: %s, Error: %s: %s',
            pass_label_num, roll_gap_m, type(e).__name__, e,
        )
        raise RuntimeError(
            f'PyRoll step simulation failed at pass {pass_label_num}: {e}',
        ) from e

    return sequence
