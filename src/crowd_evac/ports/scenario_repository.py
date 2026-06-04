"""ScenarioRepository port — scenario data loading interface."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from crowd_evac.domain.floor_plan import FloorPlan


class ScenarioRepository(Protocol):
    """Load scenario data (floor plans, initial population, parameters)."""

    def load_scenario(self, scenario_id: str) -> tuple[FloorPlan, int]:
        """Load a scenario by ID.

        Args:
            scenario_id: Identifier for the scenario (e.g., 'lecture_hall').

        Returns:
            Tuple of (FloorPlan, initial_agent_count). The FloorPlan
            includes all walls, obstacles, and exits; the count is the
            number of agents to spawn initially.

        Raises:
            FileNotFoundError: If the scenario data cannot be found.
            ValueError: If the scenario data is malformed.
        """
        ...
