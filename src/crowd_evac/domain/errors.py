"""Exceptions for crowd evacuation simulation domain."""
from __future__ import annotations


class CrowdEvacError(Exception):
    """Base exception for all crowd evacuation simulation errors."""

    pass


class ScenarioError(CrowdEvacError):
    """Base exception for scenario data issues."""

    pass


class MalformedScenarioError(ScenarioError):
    """Raised when a scenario file cannot be parsed or has structural errors.

    Covers: invalid JSON, missing required fields, unsupported schema version.
    """

    pass


class ScenarioValidationError(ScenarioError):
    """Raised when scenario data is structurally valid but semantically wrong.

    Examples: negative room dimensions, exit with zero capacity.
    """

    pass
