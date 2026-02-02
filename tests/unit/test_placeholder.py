"""Placeholder test to ensure CI passes."""

import pytest
from physics_informed_ml import __version__


def test_version():
    """Test version is defined."""
    assert __version__ == "0.1.0"


def test_import():
    """Test basic import works."""
    import physics_informed_ml

    assert physics_informed_ml is not None