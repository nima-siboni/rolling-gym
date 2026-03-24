"""
Flow stress model based on the Freiberg model but with a custom definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from typing import Union

import numpy as np


@dataclass
class CustomFlowStressCoefficients:  # pylint: disable=too-many-instance-attributes
    """
    Coefficients for the Hensel flow stress model:

        σ_f = a · ε̇^(m1 + m2·T) · exp(m3·T) · ε^m4 · exp(m5·ε)

    where T is temperature in °C, ε is strain (offset by base_strain),
    and ε̇ is strain rate (offset by base_strain_rate). Coefficients m6–m9
    are reserved for extended model variants and default to 0.

    Attributes:
        a: Strength coefficient [Pa].
        m1: Strain rate sensitivity base [-].
        m2: Temperature–strain rate interaction [1/°C].
        m3: Temperature sensitivity [1/°C].
        m4: Strain hardening exponent [-].
        m5: Dynamic recovery coefficient [-].
        m6–m9: Reserved coefficients, unused in the base model (default 0).
        base_strain: Strain offset to avoid singularity at ε=0 (default 0.1).
        base_strain_rate: Strain rate offset to avoid singularity at ε̇=0 (default 0.1).
    """
    a: Optional[float]
    m1: Optional[float] = 0
    m2: Optional[float] = 0
    m3: Optional[float] = 0
    m4: Optional[float] = 0
    m5: Optional[float] = 0
    m6: Optional[float] = 0
    m7: Optional[float] = 0
    m8: Optional[float] = 0
    m9: Optional[float] = 0

    base_strain: Optional[float] = 0.1
    base_strain_rate: Optional[float] = 0.1


def flow_stress(
    coefficients: CustomFlowStressCoefficients, strain: Union[float, np.ndarray],
    strain_rate: Union[float, np.ndarray], temperature: Union[float, np.ndarray],
):
    """
    Calculate flow stress using the Hensel model.

    Args:
        coefficients: Hensel model coefficients (see CustomFlowStressCoefficients).
        strain: Equivalent strain [-]. Offset by base_strain internally.
        strain_rate: Equivalent strain rate [1/s]. Offset by base_strain_rate internally.
        temperature: Absolute temperature [K]. Converted to °C internally.

    Returns:
        Flow stress [Pa], same shape as the broadcastable inputs.
    """

    strain = strain + coefficients.base_strain                              # type: ignore
    strain_rate = strain_rate + coefficients.base_strain_rate               # type: ignore
    temperature = temperature - 273.15

    return (
        coefficients.a * (
            strain_rate ** (                                                # type: ignore
                coefficients.m1 + (                                         # type: ignore
                    coefficients.m2 * temperature                           # type: ignore
                )
            )
        ) * np.exp(
            temperature * coefficients.m3,                                  # type: ignore
        ) * strain ** coefficients.m4 * np.exp(strain * coefficients.m5)    # type: ignore
    )
