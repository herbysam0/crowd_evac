"""Tests for crowd_evac.domain.errors."""
from __future__ import annotations

import pytest

from crowd_evac.domain.errors import CrowdEvacError


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
