# pylint: disable=no-member
"""Tests for configuration preset factory functions."""

import pytest
from pydantic import ValidationError

from rollinggym.env.rolling_config_presets import create_s355_config


def test_create_s355_config_defaults():
    """Test factory creates valid config with all defaults."""
    config = create_s355_config()

    # Verify S355 material
    assert config.material.grade == 'S355'
    assert config.material.density_kg_m3 == 7.5e3
    assert config.material.rolling_operation == 'hot_rolling'

    # Verify geometry defaults
    assert config.initial.thickness_m == pytest.approx(110e-3)  # 110mm
    assert config.target.thickness_m == pytest.approx(5e-3)     # 5mm
    assert config.initial.width_m == pytest.approx(260e-3)
    assert config.initial.length_m == pytest.approx(330e-3)

    # Verify process defaults
    assert config.process.height_reduction_lim_m == pytest.approx(30e-3)  # 30mm

    # Verify mill defaults
    assert config.mill.configuration == '2hi'
    assert config.mill.roll_nominal_radius_m == pytest.approx(205e-3)  # 205mm, Ø410mm
    assert config.mill.roll_usable_width_m == pytest.approx(350e-3)
    assert config.mill.friction_coefficient == pytest.approx(0.2)

    # Verify temperature default
    assert config.initial.temperature_K == pytest.approx(1373.15)  # 1100°C + 273.15

    # Verify grain size defaults
    assert config.initial.grain_size_m == pytest.approx(200e-6)  # 200µm
    assert config.target.grain_size_m == pytest.approx(12e-6)    # 12µm

    # Verify equipment limits
    assert config.equipment.force_N == pytest.approx(4e6)       # 4 MN
    assert config.equipment.torque_Nm == pytest.approx(130e3)   # 130 kNm
    assert config.equipment.power_W == pytest.approx(400e3)     # 400 kW

    # Verify material models are present
    assert config.material.models.flow_stress is not None
    assert config.material.models.drx is not None
    assert config.material.models.mdrx is not None
    assert config.material.models.srx is not None
    assert config.material.models.grain_growth is not None

    # Verify Hall-Petch parameters
    assert config.material.hall_petch.solid_solution_uts_Pa == pytest.approx(386.11e6)
    assert config.material.hall_petch.k_uts_Pa_um == pytest.approx(11e6)


def test_create_s355_config_geometry_overrides():
    """Test factory accepts geometry parameter overrides."""
    config = create_s355_config(
        initial_thickness_mm=100.0,
        target_thickness_mm=10.0,
        initial_width_mm=250.0,
        initial_length_mm=300.0,
    )

    assert config.initial.thickness_m == 100e-3
    assert config.target.thickness_m == 10e-3
    assert config.initial.width_m == 250e-3
    assert config.initial.length_m == 300e-3


def test_create_s355_config_hr_lim_override():
    """Test factory accepts height reduction limit override."""
    config = create_s355_config(height_reduction_lim_mm=25.0)
    assert config.process.height_reduction_lim_m == 25e-3


def test_create_s355_config_temperature_override():
    """Test factory converts temperature correctly."""
    config = create_s355_config(initial_temperature_C=1150.0)
    assert config.initial.temperature_K == pytest.approx(1423.15)  # 1150 + 273.15


def test_create_s355_config_grain_size_override():
    """Test factory converts grain size correctly."""
    config = create_s355_config(
        initial_grain_size_um=150.0,
        target_grain_size_um=10.0,
    )
    assert config.initial.grain_size_m == pytest.approx(150e-6)
    assert config.target.grain_size_m == pytest.approx(10e-6)


def test_create_s355_config_mill_overrides():
    """Test factory accepts mill configuration overrides."""
    config = create_s355_config(
        roll_nominal_radius_mm=200.0,
        roll_usable_width_mm=300.0,
    )
    assert config.mill.roll_nominal_radius_m == 200e-3
    assert config.mill.roll_usable_width_m == 300e-3


def test_create_s355_config_equipment_overrides():
    """Test factory accepts equipment limit overrides."""
    config = create_s355_config(
        force_limit_MN=5.0,
        torque_limit_kNm=150.0,
        power_limit_kW=500.0,
    )
    assert config.equipment.force_N == 5e6
    assert config.equipment.torque_Nm == 150e3
    assert config.equipment.power_W == 500e3


def test_create_s355_config_validation_target_exceeds_initial():
    """Test factory validates target < initial thickness."""
    with pytest.raises(ValidationError) as exc_info:
        create_s355_config(
            initial_thickness_mm=5.0,
            target_thickness_mm=10.0,  # Invalid: target > initial
        )

    assert 'Target thickness' in str(exc_info.value)


def test_greedy_search_use_case():
    """Test greedy_search.py use case: hr_lim override from meters."""
    # Simulate: args.hr_lim = 0.025 (25mm in meters)
    hr_lim_m = 0.025
    config = create_s355_config(height_reduction_lim_mm=hr_lim_m * 1e3)

    assert config.process.height_reduction_lim_m == 0.025
    assert config.initial.thickness_m == 110e-3  # Other defaults unchanged


def test_unit_conversions():
    """Test all unit conversions work correctly."""
    config = create_s355_config(
        initial_thickness_mm=1.0,
        target_thickness_mm=0.5,
        initial_width_mm=1.0,
        initial_length_mm=1.0,
        height_reduction_lim_mm=0.1,
        initial_temperature_C=0.0,
        initial_grain_size_um=1.0,
        target_grain_size_um=0.5,
        roll_nominal_radius_mm=1.0,
        roll_usable_width_mm=1.0,
        force_limit_MN=1.0,
        torque_limit_kNm=1.0,
        power_limit_kW=1.0,
    )

    assert config.initial.thickness_m == pytest.approx(1e-3)
    assert config.target.thickness_m == pytest.approx(0.5e-3)
    assert config.initial.temperature_K == pytest.approx(273.15)  # 0°C + 273.15
    assert config.initial.grain_size_m == pytest.approx(1e-6)
    assert config.target.grain_size_m == pytest.approx(0.5e-6)
    assert config.equipment.force_N == pytest.approx(1e6)
    assert config.equipment.torque_Nm == pytest.approx(1e3)
    assert config.equipment.power_W == pytest.approx(1e3)


def test_computed_fields():
    """Test that computed fields (uts_Pa, ys_Pa) work correctly."""
    config = create_s355_config()

    # Verify computed fields are calculated
    assert config.uts_Pa > 0
    assert config.ys_Pa > 0

    # Verify they use Hall-Petch relationship: σ = σ_0 + k / √d
    grain_size_um = config.initial.grain_size_m * 1e6  # 200µm
    expected_uts = (
        config.material.hall_petch.solid_solution_uts_Pa
        + config.material.hall_petch.k_uts_Pa_um / (grain_size_um ** 0.5)
    )
    assert abs(config.uts_Pa - expected_uts) < 1e-6


def test_material_models_access():
    """Test that material models are accessible via .models. accessor."""
    config = create_s355_config()

    # Verify models are accessible
    assert hasattr(config.material.models, 'flow_stress')
    assert hasattr(config.material.models, 'drx')
    assert hasattr(config.material.models, 'mdrx')
    assert hasattr(config.material.models, 'srx')
    assert hasattr(config.material.models, 'grain_growth')

    # Verify they are the correct types
    from rollinggym.env.pyroll_plugins.custom_flow_stress import CustomFlowStressCoefficients    # pylint: disable=import-outside-toplevel, line-too-long
    import pyroll.jmak_recrystallization as prj                             # pylint: disable=import-outside-toplevel, line-too-long

    assert isinstance(config.material.models.flow_stress, CustomFlowStressCoefficients)
    assert isinstance(config.material.models.drx, prj.JMAKRecrystallizationParameters)
    assert isinstance(config.material.models.mdrx, prj.JMAKRecrystallizationParameters)
    assert isinstance(config.material.models.srx, prj.JMAKRecrystallizationParameters)
    assert isinstance(config.material.models.grain_growth, prj.JMAKGrainGrowthParameters)


def test_keyword_only_arguments():
    """Test that factory function requires keyword arguments."""
    # This should fail because arguments must be keyword-only
    with pytest.raises(TypeError):
        create_s355_config(         # pylint: disable=too-many-function-args
            110.0, 5.0,
        )  # Positional arguments not allowed


def test_multiple_overrides_combined():
    """Test that multiple parameter overrides work together."""
    config = create_s355_config(
        initial_thickness_mm=100.0,
        target_thickness_mm=8.0,
        height_reduction_lim_mm=20.0,
        initial_temperature_C=1150.0,
        force_limit_MN=5.0,
    )

    assert config.initial.thickness_m == pytest.approx(100e-3)
    assert config.target.thickness_m == pytest.approx(8e-3)
    assert config.process.height_reduction_lim_m == pytest.approx(20e-3)
    assert config.initial.temperature_K == pytest.approx(1423.15)  # 1150 + 273.15
    assert config.equipment.force_N == pytest.approx(5e6)

    # Verify non-overridden defaults are still applied
    assert config.initial.width_m == pytest.approx(260e-3)
    assert config.material.grade == 'S355'


def test_s355_specific_material_properties():
    """Test that S355-specific material properties are hardcoded correctly."""
    config = create_s355_config()

    # Verify S355 flow stress coefficients
    assert config.material.models.flow_stress.a == 3750e6
    assert config.material.models.flow_stress.m1 == -0.11
    assert config.material.models.flow_stress.m2 == 0.00024
    assert config.material.models.flow_stress.m3 == -0.003
    assert config.material.models.flow_stress.m4 == 0.28
    assert config.material.models.flow_stress.m5 == -0.41
    assert config.material.models.flow_stress.base_strain == 0.1
    assert config.material.models.flow_stress.base_strain_rate == 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
