"""Configuration and fixtures for pytest."""
from unittest import mock

import pytest


@pytest.fixture(name="logger")
def logger_fixture(mocker) -> mock.Mock:
    """Mock a logger client."""
    return mocker.Mock()
