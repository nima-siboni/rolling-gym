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
    Class representing the Hensel flow stress model.
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
    Calculates the flow stress according to the model from the provided
    coefficients, strain, strain rate and temperature.

    :param coefficients: the coefficients set to use
    :param strain: the equivalent strain experienced
    :param strain_rate: the equivalent strain rate experienced
    :param temperature: the absolute temperature of the material (K)
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
