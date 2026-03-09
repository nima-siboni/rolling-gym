#!/usr/bin/env python
# pylint: disable=line-too-long, no-member, broad-exception-caught
"""
Test script for Pydantic-based EnvConfig validation features.

Tests that Pydantic validation works correctly after factory function refactoring.

Run this after activating your conda/virtual environment:
    pytest tests/pydantic_config_migration_test.py
"""
from __future__ import annotations

import pytest

from rollinggym.env.rolling_config_presets import create_s355_config


def test_factory_function_creates_valid_config():
    """Test that factory function creates valid configuration."""
    config = create_s355_config()

    # Verify config is valid
    assert config is not None
    assert config.material.grade == 'S355'
    assert config.mill.configuration == '2hi'


def test_factory_function_with_overrides():
    """Test factory function with parameter overrides."""
    config = create_s355_config(
        initial_thickness_mm=100.0,
        target_thickness_mm=10.0,
        height_reduction_lim_mm=25.0,
    )

    assert config.initial.thickness_m == 100e-3
    assert config.target.thickness_m == 10e-3
    assert config.process.height_reduction_lim_m == 25e-3


def test_computed_fields():
    """Test that UTS and YS are computed from Hall-Petch."""
    config = create_s355_config()

    # Check that uts_Pa and ys_Pa are computed, not None
    uts = config.uts_Pa
    ys = config.ys_Pa

    assert uts is not None, 'UTS should be computed, not None'
    assert ys is not None, 'YS should be computed, not None'

    # Verify Hall-Petch calculation
    grain_size_um = config.initial.grain_size_m * 1e6
    expected_uts = (
        config.material.hall_petch.solid_solution_uts_Pa
        + config.material.hall_petch.k_uts_Pa_um / (grain_size_um ** 0.5)
    )
    assert pytest.approx(uts, rel=1e-6) == expected_uts


def test_validation_catches_errors():
    """Test that Pydantic validation catches errors."""
    # Test 1: Negative force should fail
    with pytest.raises(Exception):  # TypeError from factory validation
        create_s355_config(force_limit_MN=-1.0)

    # Test 2: Target >= initial thickness should fail
    with pytest.raises(Exception):  # ValidationError
        create_s355_config(
            initial_thickness_mm=5.0,
            target_thickness_mm=10.0,
        )


def test_validation_on_assignment():
    """Test that validation happens when modifying config fields."""
    config = create_s355_config()

    # Test: Negative force should fail on assignment
    with pytest.raises(Exception):  # ValidationError
        config.equipment.force_N = -1000

    # Test: Zero thickness should fail
    with pytest.raises(Exception):  # ValidationError
        config.initial.thickness_m = 0


def test_all_nested_groups_accessible():
    """Verify all nested configuration groups are accessible."""
    config = create_s355_config()

    # All 7 nested groups should be accessible
    assert config.mill is not None
    assert config.material is not None
    assert config.initial is not None
    assert config.target is not None
    assert config.process is not None
    assert config.dimensions is not None
    assert config.equipment is not None


def test_material_models_accessible():
    """Test that material models are accessible."""
    config = create_s355_config()

    # Material models via .models. accessor
    assert hasattr(config.material, 'models')
    assert config.material.models.flow_stress is not None
    assert config.material.models.drx is not None
    assert config.material.models.mdrx is not None
    assert config.material.models.srx is not None
    assert config.material.models.grain_growth is not None


def test_config_modification():
    """Test that config fields can be modified."""
    config = create_s355_config()

    # Modify a field
    original_force = config.equipment.force_N
    config.equipment.force_N = 5e6
    assert config.equipment.force_N == 5e6
    assert config.equipment.force_N != original_force

    # Modify nested field
    original_temp = config.initial.temperature_K
    config.initial.temperature_K = 1400
    assert config.initial.temperature_K == 1400
    assert config.initial.temperature_K != original_temp


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
