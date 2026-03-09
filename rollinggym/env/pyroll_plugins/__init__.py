"""
PyRoll plugin registrations for the Rolling Gym simulation environment.

Importing this package (or any of its submodules) registers all custom PyRoll hooks:
- Custom flow stress model (Hensel/Freiberg-based)
- Corrected Hitchcock roll flattening
"""
from __future__ import annotations

from pyroll.core import DeformationUnit
from pyroll.core import Hook
from pyroll.core import Profile

from . import corrected_hitchcock_roll_flattening  # noqa: F401 — registers Hitchcock hooks
from .custom_flow_stress import CustomFlowStressCoefficients
from .custom_flow_stress import flow_stress

VERSION = '3.0.1post1'

Profile.custom_flow_stress_coefficients = Hook[CustomFlowStressCoefficients]()


@DeformationUnit.Profile.flow_stress
def custom_flow_stress(self: DeformationUnit.Profile):
    if hasattr(self, 'custom_flow_stress_coefficients'):
        return flow_stress(
            self.custom_flow_stress_coefficients,
            self.strain,
            self.unit.strain_rate,
            self.temperature,
        )
    return None


@DeformationUnit.Profile.flow_stress_function
def custom_flow_stress_function(self: DeformationUnit.Profile):
    """
    Returns a function that calculates the flow stress.
    """
    if hasattr(self, 'custom_flow_stress_coefficients'):
        def f(strain: float, strain_rate: float, temperature: float) -> float:
            return flow_stress(
                self.custom_flow_stress_coefficients,
                strain,
                strain_rate,
                temperature,
            )

        return f
    return None
