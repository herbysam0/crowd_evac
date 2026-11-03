"""ScenarioRepository port — scenario data loading interface."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from crowd_evac.domain.floor_plan import FloorPlan


class ScenarioRepository(Protocol):
    """Load scenario data (room geometry only).

    The population and hazards are run-time input, not part of a scenario.
    """

    def load_scenario(self, scenario_id: str) -> FloorPlan:
        """Load a scenario by ID.

        Args:
            scenario_id: Identifier for the scenario (e.g., 'lecture_hall').

        Returns:
            The FloorPlan with all walls, obstacles, and exits.

        Raises:
            FileNotFoundError: If the scenario data cannot be found.
            ValueError: If the scenario data is malformed.
        """
        ...
