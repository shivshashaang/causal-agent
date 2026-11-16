"""Smoke test: the package is installed and pytest can import it."""

from importlib.metadata import version

import causal_agent


def test_package_is_installed_and_importable() -> None:
    assert causal_agent.__doc__
    assert version("causal-agent")
