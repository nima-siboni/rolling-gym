"""
Helper functions for flat rolling simulation.

"""
from __future__ import annotations

import signal
from contextlib import contextmanager

import pyroll.core as pr
from pyroll.jmak_recrystallization import JMAKGrainGrowthParameters
from pyroll.jmak_recrystallization import JMAKRecrystallizationParameters

from rollinggym.env.pyroll_plugins.custom_flow_stress import CustomFlowStressCoefficients
from rollinggym.env.rolling_env_config import EnvConfig


@contextmanager
def timeout(seconds):
    """
    Context manager to enforce a timeout on a block of code using signals.

    Note:
        This uses SIGALRM which is only available on Unix systems (Linux, macOS).
        For Windows compatibility, consider using threading-based alternatives.

    Args:
        seconds: Maximum number of seconds to allow the code block to run

    Raises:
        TimeoutError: If the code block exceeds the specified timeout
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f'Simulation exceeded {seconds}s timeout')

    # Save the original handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    # Set the alarm
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Cancel the alarm and restore the original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def roll_torque(pass_sequence: pr.PassSequence) -> float:
    """
    Calculate the roll torque for a given roll pass.

    Replaces the PyRoll built-in torque calculation, which uses incorrect parameters.
    Uses mean flow stress and mean width weighted 1:2 between entry and exit profiles.

    Args:
        pass_sequence: A solved PyRoll PassSequence for a single roll pass.

    Returns:
        Roll torque [Nm].
    """
    mean_flow_stress = (
        pass_sequence.in_profile.flow_stress + 2 * pass_sequence.out_profile.flow_stress
    ) / 3
    mean_width = (
        pass_sequence.in_profile.equivalent_width + 2 * (
            pass_sequence.out_profile.equivalent_width
        )
    ) / 3
    height_change = pass_sequence.in_profile.equivalent_height - \
        pass_sequence.out_profile.equivalent_height

    return (
        mean_width * pass_sequence.roll.working_radius * mean_flow_stress * (
            height_change * pass_sequence.roll_torque_loss_function
        )
    )


def roll_stock_mass(pass_sequence: pr.PassSequence) -> float:
    """
    Calculate the mass of the rolling stock entering a roll pass.

    Args:
        pass_sequence: A solved PyRoll PassSequence for a single roll pass.

    Returns:
        Mass of the rolling stock [kg].
    """
    return pass_sequence.in_profile.density * \
        pass_sequence.in_profile.height * \
        pass_sequence.in_profile.width * \
        pass_sequence.in_profile.length


def ultimate_tensile_strength(
        env_config: EnvConfig,
        pass_sequence: pr.PassSequence,
) -> float:
    """
    Calculate the ultimate tensile strength of the rolled stock using the
    Hall-Petch relationship: UTS = σ_0 + k / √d

    Args:
        env_config: Environment configuration containing Hall-Petch parameters.
        pass_sequence: A solved PyRoll PassSequence; grain size is read from
            the output profile of the last element.

    Returns:
        Ultimate tensile strength [Pa].
    """
    hall_petch = env_config.material.hall_petch
    grain_size_um = pass_sequence.out_profile.grain_size * 1e6

    return (
        hall_petch.solid_solution_uts_Pa + (
            hall_petch.k_uts_Pa_um * grain_size_um ** -0.5
        )
    )


def create_in_profile(
        *,
        starting_thickness_m: float,
        starting_width_m: float,
        starting_length_m: float,
        starting_temperature_K: float,   # pylint: disable=invalid-name
        starting_grain_size_m: float,
        density_kg_m3: float,
        specific_heat_capacity_j_kg_K: float,
        flow_stress_coefficients: CustomFlowStressCoefficients,
        drx_params: JMAKRecrystallizationParameters = None,
        mdrx_params: JMAKRecrystallizationParameters = None,
        srx_params: JMAKRecrystallizationParameters = None,
        gg_params: JMAKGrainGrowthParameters = None,
) -> pr.BoxProfile:
    """
    Create a PyRoll BoxProfile representing the initial workpiece.

    Args:
        starting_thickness_m: Initial workpiece thickness [m].
        starting_width_m: Initial workpiece width [m].
        starting_length_m: Initial workpiece length [m].
        starting_temperature_K: Initial workpiece temperature [K].
        starting_grain_size_m: Initial austenite grain size [m].
        density_kg_m3: Material density [kg/m³].
        specific_heat_capacity_j_kg_K: Specific heat capacity [J/(kg·K)].
        flow_stress_coefficients: Hensel flow stress model coefficients.
        drx_params: Dynamic recrystallization JMAK parameters (optional).
        mdrx_params: Metadynamic recrystallization JMAK parameters (optional).
        srx_params: Static recrystallization JMAK parameters (optional).
        gg_params: Grain growth JMAK parameters (optional).

    Returns:
        pr.BoxProfile configured for hot rolling simulation.
    """

    assert starting_thickness_m > 0.0, 'starting_thickness_m must be > 0.0'
    assert starting_width_m > 0.0, 'starting_width_m must be > 0.0'
    assert starting_length_m > 0.0, 'starting_length_m must be > 0.0'
    assert starting_temperature_K > 0.0, 'starting_temperature_K must be > 0.0'
    assert starting_grain_size_m > 0.0, 'starting_grain_size_m must be > 0.0'
    assert density_kg_m3 > 0.0, 'density_kg_m3 must be > 0.0'
    assert specific_heat_capacity_j_kg_K > 0.0, 'specific_heat_j_kg_K must be > 0.0'

    in_profile = pr.Profile.box(
        height=starting_thickness_m,
        width=starting_width_m,
        length=starting_length_m,
        temperature=starting_temperature_K,
        grain_size=starting_grain_size_m,
        density=density_kg_m3,
        specific_heat_capacity=specific_heat_capacity_j_kg_K,
        custom_flow_stress_coefficients=flow_stress_coefficients,
        jmak_dynamic_recrystallization_parameters=drx_params,
        jmak_metadynamic_recrystallization_parameters=mdrx_params,
        jmak_static_recrystallization_parameters=srx_params,
        jmak_grain_growth_parameters=gg_params,
        strain=0,
        recrystallized_fraction=0,
    )

    return in_profile


def create_roll(env_config: EnvConfig) -> pr.Roll:
    """
    Create a PyRoll Roll object from the mill configuration.

    Sets a flat groove with the configured usable width, and applies the
    nominal radius, Poisson's ratio, and elastic modulus from env_config.mill.

    Args:
        env_config: Environment configuration containing mill parameters.

    Returns:
        pr.Roll configured for flat rolling simulation.
    """

    roll = pr.Roll(
        groove=pr.FlatGroove(
            usable_width=env_config.mill.roll_usable_width_m,
        ),
        nominal_radius=env_config.mill.roll_nominal_radius_m,
        poissons_ratio=env_config.mill.roll_poissons_ratio,
        elastic_modulus=env_config.mill.roll_elastic_modulus_Pa,
    )

    return roll
