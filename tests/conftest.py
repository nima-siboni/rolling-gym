"""
Pytest configuration and shared fixtures for Rolling Gym tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add parent directory to path to import rollinggym package
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
  """Configure pytest with custom markers."""
  config.addinivalue_line(
      'markers', 'slow: marks tests as slow (deselect with "-m not slow")',
  )
  config.addinivalue_line(
      'markers', 'simulation: marks tests that require PyRoll simulation',
  )


@pytest.fixture(scope='session')
def base_env_config():
  """Create a base environment configuration for the test session."""
  from rollinggym.env.rolling_config_presets import create_s355_config  # pylint: disable=import-outside-toplevel
  return create_s355_config()
