"""
Configuration preset factory functions for common rolling scenarios.

This module provides pre-configured EnvConfig instances for standard
materials and mill setups, avoiding the need for hardcoded defaults.

Example:
    >>> from rollinggym.env.rolling_config_presets import create_s355_config
    >>>
    >>> # Standard S355 configuration
    >>> config = create_s355_config()
    >>>
    >>> # Custom thickness range
    >>> config = create_s355_config(
    ...     initial_thickness_mm=100.0,
    ...     target_thickness_mm=10.0
    ... )
    >>>
    >>> # Override height reduction limit (greedy_search.py use case)
    >>> config = create_s355_config(height_reduction_lim_mm=25.0)
"""
from __future__ import annotations

import numpy as np
import pyroll.jmak_recrystallization as prj

from rollinggym.env.pyroll_plugins.custom_flow_stress import CustomFlowStressCoefficients
from rollinggym.env.rolling_env_config import DimensionLimits
from rollinggym.env.rolling_env_config import EnvConfig
from rollinggym.env.rolling_env_config import EquipmentLimits
from rollinggym.env.rolling_env_config import HallPetchParameters
from rollinggym.env.rolling_env_config import InitialConditions
from rollinggym.env.rolling_env_config import MaterialConfig
from rollinggym.env.rolling_env_config import MaterialModels
from rollinggym.env.rolling_env_config import MillConfig
from rollinggym.env.rolling_env_config import ProcessConstraints
from rollinggym.env.rolling_env_config import TargetSpecifications


def create_s355_config(  # pylint: disable=too-many-locals
    *,
    # Geometry overrides (most commonly changed)
    initial_thickness_mm: float = 110.0,
    target_thickness_mm: float = 5.0,
    initial_width_mm: float = 260.0,
    initial_length_mm: float = 330.0,
    # Process overrides
    height_reduction_lim_mm: float = 30.0,
    # Temperature/grain overrides
    initial_temperature_C: float = 1100.0,
    target_temperature_C: float = 900.0,
    initial_grain_size_um: float = 200.0,
    target_grain_size_um: float = 12.0,
    # Mill configuration overrides
    roll_nominal_radius_mm: float = 205.0,
    roll_usable_width_mm: float = 350.0,
    # Equipment limits overrides
    force_limit_MN: float = 4.0,
    torque_limit_kNm: float = 130.0,
    power_limit_kW: float = 400.0,
) -> EnvConfig:
    """
    Create EnvConfig for S355 structural steel on 2-high lab mill.

    This factory function provides sensible defaults for hot rolling
    S355 steel on a laboratory-scale two-high reversing mill.

    Args:
        initial_thickness_mm: Starting slab thickness [mm] (default: 110)
        target_thickness_mm: Target final thickness [mm] (default: 5)
        initial_width_mm: Starting slab width [mm] (default: 260)
        initial_length_mm: Starting slab length [mm] (default: 330)
        height_reduction_lim_mm: Max height reduction per pass [mm] (default: 30)
        initial_temperature_C: Initial temperature [°C] (default: 1100)
        target_temperature_C: Target temperature [°C] (default: 900)
        initial_grain_size_um: Initial grain size [µm] (default: 200)
        target_grain_size_um: Target grain size [µm] (default: 12)
        roll_nominal_radius_mm: Roll radius [mm] (default: 205, Ø410mm)
        roll_usable_width_mm: Usable roll width [mm] (default: 350)
        force_limit_MN: Maximum rolling force [MN] (default: 4.0)
        torque_limit_kNm: Maximum torque per roll [kNm] (default: 130)
        power_limit_kW: Maximum power both rolls [kW] (default: 400)

    Returns:
        Fully configured and validated EnvConfig instance for S355 steel

    Example:
        >>> # Standard S355 config
        >>> config = create_s355_config()
        >>> config.material.grade
        'S355'

        >>> # Custom thickness range
        >>> config = create_s355_config(
        ...     initial_thickness_mm=100.0,
        ...     target_thickness_mm=10.0
        ... )

        >>> # Override height reduction limit (greedy_search.py use case)
        >>> config = create_s355_config(height_reduction_lim_mm=25.0)
    """

    # Convert units to SI (meters, Kelvin, Pascals, etc.)
    initial_thickness_m = initial_thickness_mm * 1e-3
    target_thickness_m = target_thickness_mm * 1e-3
    initial_width_m = initial_width_mm * 1e-3
    initial_length_m = initial_length_mm * 1e-3
    height_reduction_lim_m = height_reduction_lim_mm * 1e-3
    initial_temperature_K = initial_temperature_C + \
        273.15      # pylint: disable=invalid-name
    target_temperature_K = target_temperature_C + \
        273.15      # pylint: disable=invalid-name
    initial_grain_size_m = initial_grain_size_um * 1e-6
    target_grain_size_m = target_grain_size_um * 1e-6
    roll_nominal_radius_m = roll_nominal_radius_mm * 1e-3
    roll_usable_width_m = roll_usable_width_mm * 1e-3
    force_limit_N = force_limit_MN * \
        1e6                        # pylint: disable=invalid-name
    torque_limit_Nm = torque_limit_kNm * \
        1e3                    # pylint: disable=invalid-name
    power_limit_W = power_limit_kW * \
        1e3                        # pylint: disable=invalid-name

    # Create S355 material models (from current rolling_env_config.py defaults)
    s355_flow_stress = CustomFlowStressCoefficients(
        a=3750e6,        # [Pa] Strength coefficient
        m1=-0.11,        # [-] Strain rate sensitivity (base)
        m2=0.00024,      # [1/°C] Temperature-strain rate interaction
        m3=-0.003,       # [1/°C] Temperature sensitivity
        m4=0.28,         # [-] Strain hardening exponent
        m5=-0.41,        # [-] Dynamic recovery coefficient
        base_strain=0.1,  # [-] Strain offset
        base_strain_rate=0.1,  # [1/s] Strain rate offset
    )

    s355_drx = prj.JMAKRecrystallizationParameters(
        k=-1.6503,
        n=1.4409,
        a1=1.2338e-3 * 0.70,
        a4=0.1022,
        qa=291876.66 * 0.2013,
        b1=2.0731e-3,
        b4=0.0724,
        qb=291876.66 * 0.2147,
        c1=3339.98,
        qc=291876.66 * -0.1660,
    )

    s355_mdrx = prj.JMAKRecrystallizationParameters(
        n=2.038,
        b1=6.9235e-2,
        qb=248617.4 - 258435.17 * 0.9245,
        c1=840.57,
        qc=258435.17 * -0.1629,
    )

    s355_srx = prj.JMAKRecrystallizationParameters(
        k=np.log(0.5),
        n=1,
        b1=6.31e-18,
        b2=2,
        b4=-2.383,
        qb=330000,
        c1=343,
        c2=-0.5,
        c4=0.4,
        qc=-45000,
    )

    s355_grain_growth = prj.JMAKGrainGrowthParameters(
        d1=4.5,
        d2=4.1e23,
        qd=-435000,
    )

    # S355 Hall-Petch parameters
    s355_hall_petch = HallPetchParameters(
        solid_solution_uts_Pa=386.11e6,
        solid_solution_ys_Pa=130.592e6,
        k_uts_Pa_um=11e6,
        k_ys_Pa_um=19.7e6,
    )

    # Construct config using keyword arguments
    return EnvConfig(
        mill=MillConfig(
            configuration='2hi',
            roll_usable_width_m=roll_usable_width_m,
            roll_nominal_radius_m=roll_nominal_radius_m,
            # roll_poissons_ratio and roll_elastic_modulus_Pa use universal defaults
            friction_coefficient=0.2,  # Hot rolling steel
        ),
        material=MaterialConfig(
            grade='S355',
            rolling_operation='hot_rolling',
            density_kg_m3=7.5e3,
            specific_heat_j_kg_K=629.64,
            hall_petch=s355_hall_petch,
            models=MaterialModels(
                flow_stress=s355_flow_stress,
                drx=s355_drx,
                mdrx=s355_mdrx,
                srx=s355_srx,
                grain_growth=s355_grain_growth,
            ),
        ),
        initial=InitialConditions(
            thickness_m=initial_thickness_m,
            length_m=initial_length_m,
            width_m=initial_width_m,
            temperature_K=initial_temperature_K,
            grain_size_m=initial_grain_size_m,
        ),
        target=TargetSpecifications(
            thickness_m=target_thickness_m,
            tolerance_m=1e-3,  # ±1mm tolerance
            grain_size_m=target_grain_size_m,
            temperature_K=target_temperature_K,
        ),
        process=ProcessConstraints(
            height_reduction_lim_m=height_reduction_lim_m,
            hr_eff_tolerance_m=5e-3,
            # interpass times use universal defaults
        ),
        dimensions=DimensionLimits(
            thickness_lower_m=0.05,
            thickness_upper_m=0.14,
            length_lim_m=10.0,
            width_lim_m=0.35,
        ),
        equipment=EquipmentLimits(
            temperature_K=1523,  # 1250°C
            force_N=force_limit_N,
            torque_Nm=torque_limit_Nm,
            power_W=power_limit_W,
            speed_m_s=0.5,
            # safety_factor uses universal default (0.8)
        ),
    )
