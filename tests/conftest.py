"""Separate source tests from checks of an installed distribution."""
import pytest


def pytest_addoption(parser):
    parser.addoption('--installed', action='store_true', help='Run isolated installed-package subprocess checks')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--installed'):
        skip = pytest.mark.skip(reason='Use --installed after installing the candidate wheel')
        for item in items:
            if 'installed' in item.keywords:
                item.add_marker(skip)
