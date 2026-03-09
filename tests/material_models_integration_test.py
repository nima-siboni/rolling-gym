# pylint: disable=no-member
"""
Test script to verify material models integration with PyRoll parameter objects.
"""
from pyroll.jmak_recrystallization import JMAKRecrystallizationParameters, JMAKGrainGrowthParameters
from rollinggym.env.rolling_config_presets import create_s355_config
from rollinggym.env.pyroll_plugins.custom_flow_stress import CustomFlowStressCoefficients


def test_material_models_integration():
    """Test that material models are correctly integrated as PyRoll parameter objects."""
    print('Testing material models integration...')

    # Create config instance
    config = create_s355_config()

    # Test flow stress parameters
    print('\n1. Testing flow_stress parameters:')
    print(f'   Type: {type(config.material.models.flow_stress)}')
    print(f'   a = {config.material.models.flow_stress.a} Pa')
    print(f'   m1 = {config.material.models.flow_stress.m1}')
    print(f'   base_strain = {config.material.models.flow_stress.base_strain}')

    # Test DRX parameters
    print('\n2. Testing drx parameters:')
    print(f'   Type: {type(config.material.models.drx)}')
    print(f'   k = {config.material.models.drx.k}')
    print(f'   n = {config.material.models.drx.n}')
    print(f'   a1 = {config.material.models.drx.a1}')

    # Test MDRX parameters
    print('\n3. Testing mdrx parameters:')
    print(f'   Type: {type(config.material.models.mdrx)}')
    print(f'   n = {config.material.models.mdrx.n}')
    print(f'   b1 = {config.material.models.mdrx.b1}')

    # Test SRX parameters
    print('\n4. Testing srx parameters:')
    print(f'   Type: {type(config.material.models.srx)}')
    print(f'   k = {config.material.models.srx.k}')
    print(f'   n = {config.material.models.srx.n}')
    print(f'   b1 = {config.material.models.srx.b1}')

    # Test grain growth parameters
    print('\n5. Testing grain_growth parameters:')
    print(f'   Type: {type(config.material.models.grain_growth)}')
    print(f'   d1 = {config.material.models.grain_growth.d1}')
    print(f'   d2 = {config.material.models.grain_growth.d2}')
    print(f'   qd = {config.material.models.grain_growth.qd}')

    # Test that these are the correct PyRoll types

    assert isinstance(config.material.models.flow_stress, CustomFlowStressCoefficients), \
        'flow_stress should be CustomFlowStressCoefficients'
    assert isinstance(config.material.models.drx, JMAKRecrystallizationParameters), \
        'drx should be JMAKRecrystallizationParameters'
    assert isinstance(config.material.models.mdrx, JMAKRecrystallizationParameters), \
        'mdrx should be JMAKRecrystallizationParameters'
    assert isinstance(config.material.models.srx, JMAKRecrystallizationParameters), \
        'srx should be JMAKRecrystallizationParameters'
    assert isinstance(config.material.models.grain_growth, JMAKGrainGrowthParameters), \
        'grain_growth should be JMAKGrainGrowthParameters'

    print('\n✅ All material models integration tests passed!')
    print('\nYou can now access material models like:')
    print('  - config.material.models.flow_stress')
    print('  - config.material.models.drx')
    print('  - config.material.models.mdrx')
    print('  - config.material.models.srx')
    print('  - config.material.models.grain_growth')


if __name__ == '__main__':
    test_material_models_integration()
