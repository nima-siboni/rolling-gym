# pylint: disable=invalid-name, too-many-public-methods, no-member
"""
Environment configuration parameters for PyRoll flat rolling simulation.

This module defines the default configuration for a laboratory-scale two-high
reversing mill for hot rolling of S355 structural steel.

Configuration Categories
------------------------

**Mill Configuration**:
    - Two-high (2hi) reversing mill
    - Roll diameter: 410mm (205mm radius)
    - Usable roll width: 350mm

**Workpiece (S355 Steel)**:
    - Initial dimensions: 110mm × 260mm × 330mm (H × W × L)
    - Initial temperature: 1100°C (1373K)
    - Initial grain size: 200µm (typical for reheated slabs)

**Hall-Petch Strengthening Parameters**:
    The ultimate tensile strength (UTS) and yield strength (YS) are calculated
    using the Hall-Petch relationship:

        σ = σ_0 + k / √d

    Where:
    - σ_0 is the solid solution strengthening contribution
    - k is the Hall-Petch constant
    - d is the grain size [µm]

    For S355 steel:
    - UTS: σ_0 = 386.11 MPa, k = 11 MPa·µm^0.5
    - YS:  σ_0 = 130.59 MPa, k = 19.7 MPa·µm^0.5

**Equipment Limits**:
    These represent typical laboratory mill constraints:
    - Maximum force: 4 MN
    - Maximum torque: 65 kNm (per roll)
    - Maximum power: 400 kW (both rolls)
    - Maximum rolling speed: 0.5 m/s

Note:
    All length dimensions are in meters [m], temperatures in Kelvin [K],
    forces in Newtons [N], and times in seconds [s].

Material Parameters
-------------------
This module also contains hardcoded material parameters for S355 structural steel.

**S355 Steel Flow Stress Coefficients for Hensel Model**:

    σ_f = A · ε̇^(m1 + m2·T) · exp(m3·T) · ε^m4 · exp(m5·ε)

    Where:
    - A = 3750 MPa (strength coefficient)
    - m1 = -0.11 (strain rate sensitivity base)
    - m2 = 0.00024 (temperature-strain rate interaction)
    - m3 = -0.003 (temperature sensitivity)
    - m4 = 0.28 (strain hardening exponent)
    - m5 = -0.41 (dynamic recovery coefficient)
    - base_strain = 0.1 (offset to prevent singularity at ε=0)
    - base_strain_rate = 0.1 (offset to prevent singularity at ε̇=0)

    Temperature T is in °C (converted from K internally).

**Steel Physical Properties**:
    - Density: 7500 kg/m³ (typical for structural steel)
    - Specific heat capacity: 629.64 J/(kg·K) (at hot rolling temperatures)

Note:
    The timeout context manager uses SIGALRM which is only available on Unix systems.
    For Windows compatibility, an alternative timeout mechanism would be needed.


-------------------

This module also contains recrystallization parameters for JMAK modules for S355
structural steel.

"""
from __future__ import annotations

from typing import Literal

import pyroll.jmak_recrystallization as prj
from pydantic import BaseModel
from pydantic import computed_field
from pydantic import Field
from pydantic import model_validator

from rollinggym.env.pyroll_plugins.custom_flow_stress import CustomFlowStressCoefficients


# ==================== Nested Configuration Models ====================


class HallPetchParameters(BaseModel):
    """Hall-Petch strengthening parameters for S355 steel."""

    model_config = {'validate_assignment': True}

    solid_solution_uts_Pa: float = Field(
        gt=0,
        description='σ_0 for UTS Hall-Petch calculation [Pa]',
    )
    solid_solution_ys_Pa: float = Field(
        gt=0,
        description='σ_0 for YS Hall-Petch calculation [Pa]',
    )
    k_uts_Pa_um: float = Field(
        gt=0,
        description='Hall-Petch constant for UTS [Pa·µm^0.5]',
    )
    k_ys_Pa_um: float = Field(
        gt=0,
        description='Hall-Petch constant for YS [Pa·µm^0.5]',
    )


class MillConfig(BaseModel):
    """Mill configuration parameters."""

    model_config = {'validate_assignment': True}

    configuration: Literal['2hi', '4hi'] = Field(
        description="Mill type: '2hi' (two-high) or '4hi' (four-high)",
    )
    roll_usable_width_m: float = Field(
        gt=0,
        description='Usable width of the roll [m]',
    )
    roll_nominal_radius_m: float = Field(
        gt=0,
        description='Nominal roll radius [m] (Ø410mm)',
    )
    roll_poissons_ratio: float = Field(
        default=0.3,
        ge=0,
        le=0.5,
        description="Poisson's ratio for steel roll [-]",
    )
    roll_elastic_modulus_Pa: float = Field(
        default=210e9,
        gt=0,
        description='Elastic modulus for steel roll [Pa]',
    )
    friction_coefficient: float = Field(
        ge=0,
        le=1,
        description='Friction coefficient for hot rolling [-]',
    )


class MaterialModels(BaseModel):
    """Material model parameters for S355 steel using PyRoll parameter objects."""

    model_config = {
        'validate_assignment': True,
        # Allow non-Pydantic types (dataclasses)
        'arbitrary_types_allowed': True,
    }

    flow_stress: CustomFlowStressCoefficients = Field(
        description='Flow stress model parameters (Hensel model)',
    )
    drx: prj.JMAKRecrystallizationParameters = Field(
        description='Dynamic recrystallization (DRX) parameters',
    )
    mdrx: prj.JMAKRecrystallizationParameters = Field(
        description='Metadynamic recrystallization (MDRX) parameters',
    )
    srx: prj.JMAKRecrystallizationParameters = Field(
        description='Static recrystallization (SRX) parameters',
    )
    grain_growth: prj.JMAKGrainGrowthParameters = Field(
        description='Grain growth parameters',
    )


class MaterialConfig(BaseModel):
    """Material properties and parameters."""

    model_config = {'validate_assignment': True}

    grade: str = Field(
        description='Workpiece material grade',
    )
    rolling_operation: Literal['hot_rolling', 'cold_rolling'] = Field(
        description='Type of rolling operation',
    )
    density_kg_m3: float = Field(
        gt=0,
        description='Density of structural steel at hot rolling temps [kg/m³]',
    )
    specific_heat_j_kg_K: float = Field(
        gt=0,
        description='Specific heat capacity at hot rolling temps [J/(kg·K)]',
    )
    hall_petch: HallPetchParameters = Field(
        description='Hall-Petch strengthening parameters',
    )
    models: MaterialModels = Field(
        description='Material model parameters (flow stress, recrystallization, grain growth)',
    )


class InitialConditions(BaseModel):
    """Initial workpiece conditions."""

    model_config = {'validate_assignment': True}

    thickness_m: float = Field(
        gt=0,
        description='Initial slab thickness [m]',
    )
    length_m: float = Field(
        gt=0,
        description='Initial slab length [m]',
    )
    width_m: float = Field(
        gt=0,
        description='Initial slab width [m]',
    )
    temperature_K: float = Field(
        gt=273,  # Above 0°C
        lt=1800,  # Below melting point of steel (~1500°C)
        description='Initial temperature [K]',
    )
    grain_size_m: float = Field(
        gt=0,
        description='Initial grain size [m]',
    )


class TargetSpecifications(BaseModel):
    """Target product specifications."""

    model_config = {'validate_assignment': True}

    thickness_m: float = Field(
        gt=0,
        description='Target final thickness [m]',
    )
    tolerance_m: float = Field(
        gt=0,
        description='Tolerance for target thickness [m]',
    )
    grain_size_m: float = Field(
        gt=0,
        description='Target grain size [m]',
    )
    temperature_K: float = Field(
        gt=273,
        lt=1800,
        description='Target temperature [K]',
    )


class ProcessConstraints(BaseModel):
    """Process operation constraints."""

    model_config = {'validate_assignment': True}

    height_reduction_lim_m: float = Field(
        gt=0,
        description='Max height reduction per pass [m]',
    )
    hr_eff_tolerance_m: float = Field(
        gt=0,
        description='Height reduction tolerance [m]',
    )
    interpass_t_lower_lim: int = Field(
        default=5,
        ge=0,
        description='Minimum inter-pass time [s]',
    )
    interpass_t_upper_lim: int = Field(
        default=30,
        ge=0,
        description='Maximum inter-pass time [s]',
    )

    @model_validator(mode='after')
    def validate_interpass_time_range(self) -> 'ProcessConstraints':
        """Ensure upper limit is greater than lower limit."""
        if self.interpass_t_upper_lim <= self.interpass_t_lower_lim:
            raise ValueError(
                f'Upper limit ({self.interpass_t_upper_lim}s) must be greater than '
                f'lower limit ({self.interpass_t_lower_lim}s)',
            )
        return self


class DimensionLimits(BaseModel):
    """Workpiece dimension limits."""

    model_config = {'validate_assignment': True}

    thickness_lower_m: float = Field(
        gt=0,
        description='Minimum workpiece thickness [m]',
    )
    thickness_upper_m: float = Field(
        gt=0,
        description='Maximum workpiece thickness [m]',
    )
    length_lim_m: float = Field(
        gt=0,
        description='Maximum workpiece length [m]',
    )
    width_lim_m: float = Field(
        gt=0,
        description='Maximum workpiece width [m]',
    )


class EquipmentLimits(BaseModel):
    """Equipment operational limits."""

    model_config = {'validate_assignment': True}

    temperature_K: float = Field(
        gt=273,
        description='Maximum material temperature [K]',
    )
    force_N: float = Field(
        gt=0,
        description='Maximum rolling force [N]',
    )
    torque_Nm: float = Field(
        gt=0,
        description='Maximum torque per roll [Nm]',
    )
    power_W: float = Field(
        gt=0,
        description='Maximum power [W]',
    )
    speed_m_s: float = Field(
        gt=0,
        description='Maximum rolling speed [m/s]',
    )
    safety_factor: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description='Safety factor for equipment limits [-]',
    )


# ==================== Main Configuration Class ====================


class EnvConfig(BaseModel):
    """
    Pydantic-based configuration for the flat rolling environment.

    This class provides validated parameters for a laboratory-scale hot rolling
    simulation using a nested structure for better organization.

    Nested Structure:
        - mill: Mill configuration (rolls, friction)
        - material: Material properties (density, Hall-Petch parameters)
        - initial: Initial workpiece conditions
        - target: Target product specifications
        - process: Process constraints (height reduction, inter-pass time)
        - dimensions: Workpiece dimension limits
        - equipment: Equipment operational limits

    The model supports three access patterns:
        1. Nested access: config.mill.roll_nominal_radius_m (recommended)
        2. Flat attribute: config.roll_nominal_radius_m (backward compatible)
        3. Dict access: config['roll_nominal_radius_m'] (backward compatible)

    Example:
        >>> config = EnvConfig()
        >>> # New nested access (recommended)
        >>> config.equipment.force_N
        4000000.0
    """

    model_config = {
        'validate_assignment': True,
        'extra': 'allow',  # Allow flat field access via properties
        'arbitrary_types_allowed': False,
    }

    # ==================== Nested Configuration Groups ====================
    mill: MillConfig = Field(
        description='Mill configuration parameters',
    )
    material: MaterialConfig = Field(
        description='Material properties and parameters',
    )
    initial: InitialConditions = Field(
        description='Initial workpiece conditions',
    )
    target: TargetSpecifications = Field(
        description='Target product specifications',
    )
    process: ProcessConstraints = Field(
        description='Process operation constraints',
    )
    dimensions: DimensionLimits = Field(
        description='Workpiece dimension limits',
    )
    equipment: EquipmentLimits = Field(
        description='Equipment operational limits',
    )
    material_reference: str = Field(
        default='',
        description='Material-specific reference text for LLM prompt context',
    )

    # ==================== Cross-Group Validation ====================
    @model_validator(mode='after')
    def validate_target_less_than_initial(self) -> 'EnvConfig':
        """Ensure target thickness is less than initial thickness."""
        if self.target.thickness_m >= self.initial.thickness_m:
            raise ValueError(
                f'Target thickness ({self.target.thickness_m*1e3:.1f}mm) must be less than '
                f'initial thickness ({self.initial.thickness_m*1e3:.1f}mm)',
            )
        return self

    # ==================== Computed Fields ====================
    @computed_field  # type: ignore[misc]
    @property
    def uts_Pa(self) -> float:
        """
        Calculate ultimate tensile strength using Hall-Petch relationship.

        UTS = σ_0 + k / √d

        Returns:
            float: Ultimate tensile strength [Pa]
        """
        grain_size_um = self.initial.grain_size_m * 1e6
        return (
            self.material.hall_petch.solid_solution_uts_Pa + self.material.hall_petch.k_uts_Pa_um / (grain_size_um ** 0.5)  # noqa: E501 # pylint: disable=line-too-long
        )

    @computed_field  # type: ignore[misc]
    @property
    def ys_Pa(self) -> float:
        """
        Calculate yield strength using Hall-Petch relationship.

        YS = σ_0 + k / √d

        Returns:
            float: Yield strength [Pa]
        """
        grain_size_um = self.initial.grain_size_m * 1e6
        return (
            self.material.hall_petch.solid_solution_ys_Pa + self.material.hall_petch.k_ys_Pa_um / (grain_size_um ** 0.5)  # noqa: E501 # pylint: disable=line-too-long
        )
