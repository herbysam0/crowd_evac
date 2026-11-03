"""Shared fixtures for the designer tests."""
from __future__ import annotations

import pytest

from crowd_evac.designer.controller import DesignController
from crowd_evac.designer.model import DesignModel


@pytest.fixture
def model() -> DesignModel:
    """Provide an empty 15 x 10 m design."""
    return DesignModel()


@pytest.fixture
def ctl() -> DesignController:
    """Provide a controller over an empty 15 x 10 m design."""
    return DesignController(DesignModel())
