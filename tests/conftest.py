"""Shared pytest configuration for the origins_of_the_fittest test suite."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the stochastic test marker."""
    config.addinivalue_line(
        "markers",
        "stochastic: marks tests as stochastic (deselect with '-m \"not stochastic\"')",
    )
