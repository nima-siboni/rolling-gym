#!/usr/bin/env python
# pylint: disable=line-too-long, no-member, broad-exception-caught
"""
Test script for nested Pydantic EnvConfig structure.

Tests the new nested access pattern after factory function refactoring.

Run this after activating your conda/virtual environment:
    pytest tests/nested_config_test.py
"""
from __future__ import annotations

import pytest

from rollinggym.env.rolling_config_presets import create_s355_config


def test_nested_access():
    """Test new nested access patterns."""
    config = create_s355_config()

    # Access nested fields
    assert config.mill.configuration == '2hi'
    assert pytest.approx(config.mill.roll_nominal_radius_m, rel=1e-9) == 205e-3
    assert config.material.grade == 'S355'
    assert config.material.density_kg_m3 == 7.5e3
    assert config.initial.thickness_m == 110e-3
    assert config.target.thickness_m == 5e-3
    assert config.equipment.force_N == 4e6

    # Modify nested fields
    config.equipment.force_N = 5e6
    assert config.equipment.force_N == 5e6

    # Access deeply nested Hall-Petch parameters
    assert config.material.hall_petch.solid_solution_uts_Pa == 386.11e6


def test_material_models_access():
    """Test accessing material models via .models. accessor."""
    config = create_s355_config()

    # Material models should be accessible via .models.
    assert config.material.models.flow_stress is not None
    assert config.material.models.drx is not None
    assert config.material.models.mdrx is not None
    assert config.material.models.srx is not None
    assert config.material.models.grain_growth is not None

    # Check specific values
    assert config.material.models.flow_stress.a == 3750e6
    assert config.material.models.flow_stress.m1 == -0.11


def test_computed_fields():
    """Test that computed fields work with nested structure."""
    config = create_s355_config()

    # UTS and YS should be computed from nested Hall-Petch + initial grain size
    uts = config.uts_Pa
    ys = config.ys_Pa

    assert uts is not None
    assert ys is not None
    assert uts > 0
    assert ys > 0

    # Changing nested grain size should update computed values
    old_uts = uts
    config.initial.grain_size_m = 100e-6  # Coarser grains
    new_uts = config.uts_Pa
    assert new_uts > old_uts


def test_nested_validation():
    """Test that validation works across nested structures."""
    # Target > initial should fail
    with pytest.raises(Exception):  # ValidationError
        create_s355_config(
            initial_thickness_mm=5.0,
            target_thickness_mm=10.0,
        )

    # Negative values should fail
    with pytest.raises(Exception):  # ValidationError or TypeError
        create_s355_config(force_limit_MN=-1.0)


def test_nested_modification():
    """Test that nested fields can be modified and validated."""
    config = create_s355_config()

    # Modify nested field
    original_force = config.equipment.force_N
    config.equipment.force_N = 5e6
    assert config.equipment.force_N == 5e6
    assert config.equipment.force_N != original_force

    # Invalid modification should fail (validation on assignment)
    with pytest.raises(Exception):  # ValidationError
        config.equipment.force_N = -1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
