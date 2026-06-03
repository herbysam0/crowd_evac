"""Tests for crowd_evac.domain.errors."""
from __future__ import annotations

import pytest

from crowd_evac.domain.errors import (
    CrowdEvacError,
    MalformedScenarioError,
    ScenarioError,
    ScenarioValidationError,
)


class TestCrowdEvacError:
    """Test suite for CrowdEvacError base exception."""

    def test_base_error_is_exception_subclass(self) -> None:
        """Verify CrowdEvacError is an Exception subclass."""
        assert issubclass(CrowdEvacError, Exception)

    def test_base_error_is_raisable(self) -> None:
        """Verify CrowdEvacError can be raised and caught."""
        with pytest.raises(CrowdEvacError):
            raise CrowdEvacError("test error")

    def test_base_error_is_catchable_by_type(self) -> None:
        """Verify CrowdEvacError can be caught by its type."""
        try:
            raise CrowdEvacError("test message")
        except CrowdEvacError as e:
            assert str(e) == "test message"

    def test_base_error_with_message(self) -> None:
        """Verify CrowdEvacError preserves message text."""
        message = "Something went wrong in the simulation"
        error = CrowdEvacError(message)
        assert str(error) == message


class TestScenarioErrors:
    """Test suite for the scenario error hierarchy."""

    # -- Happy path: hierarchy and catch-all --------------------------------

    def test_scenario_error_is_crowd_evac_error(self) -> None:
        """Verify ScenarioError is a CrowdEvacError subclass."""
        assert issubclass(ScenarioError, CrowdEvacError)

    def test_malformed_is_scenario_error(self) -> None:
        """Verify MalformedScenarioError is a ScenarioError subclass."""
        assert issubclass(MalformedScenarioError, ScenarioError)

    def test_validation_is_scenario_error(self) -> None:
        """Verify ScenarioValidationError is a ScenarioError subclass."""
        assert issubclass(ScenarioValidationError, ScenarioError)

    # -- Edge case: caught by parent type -----------------------------------

    def test_malformed_caught_as_crowd_evac_error(self) -> None:
        """Verify MalformedScenarioError is catchable as CrowdEvacError."""
        with pytest.raises(CrowdEvacError):
            raise MalformedScenarioError("bad json")

    def test_validation_caught_as_scenario_error(self) -> None:
        """Verify ScenarioValidationError is catchable as ScenarioError."""
        with pytest.raises(ScenarioError):
            raise ScenarioValidationError("negative width")

    # -- Failure path: distinct types don't overlap -------------------------

    def test_malformed_not_caught_as_validation_error(self) -> None:
        """Verify MalformedScenarioError is NOT a ScenarioValidationError."""
        assert not issubclass(MalformedScenarioError, ScenarioValidationError)

    def test_validation_not_caught_as_malformed_error(self) -> None:
        """Verify ScenarioValidationError is NOT a MalformedScenarioError."""
        assert not issubclass(ScenarioValidationError, MalformedScenarioError)
