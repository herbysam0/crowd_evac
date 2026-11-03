"""Convert a design :class:`Layout` to schema-1.0 ``ScenarioData``.

Follows the conventions of the bundled scenarios: ``north`` is the bottom
wall (``y=0``), ``south`` the top wall (``y=height_m``), and an exit's
centre sits on the inner face of its wall.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np

from crowd_evac.designer.geometry import Side
from crowd_evac.designer.layout import (
    Doorway,
    Layout,
    rect_of_wall,
    side_of_wall,
    validate,
)
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.constants import DT, GRID_CELL_SIZE
from crowd_evac.domain.errors import DesignValidationError
from crowd_evac.domain.floor_plan import (
    Exit,
    ExitSide,
    FloorPlan,
    Obstacle,
    Wall,
)
from crowd_evac.pathfinding.flow_field import FlowField
from crowd_evac.scenarios.schema import (
    ExitData,
    FloorPlanData,
    ObstacleData,
    ScenarioData,
    WallData,
)

SCHEMA_VERSION = "1.0"
# Matches the harness default (optimization.harness.DEFAULT_MAX_TICKS);
# duplicated so the designer does not import the optimisation stack.
DESIGN_MAX_TICKS: int = 10_000


def export_scenario(layout: Layout) -> ScenarioData:
    """Build the scenario dict for a design.

    Args:
        layout: Design snapshot.

    Returns:
        ``ScenarioData`` ready for JSON serialisation.

    Raises:
        DesignValidationError: If the layout itself is invalid.
    """
    validate(layout)
    room = layout.room
    floor_plan = _floor_plan(layout)
    extras = layout.extras
    simulation: dict[str, Any] = {
        "dt": DT,
        "max_ticks": DESIGN_MAX_TICKS,
        **extras.simulation,
    }
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": room.name,
        "floor_plan": floor_plan,
        "agents": {
            "count": extras.agent_count,
            "spawn_seed": extras.spawn_seed,
        },
        "simulation": simulation,
    }
    if extras.events:
        data["events"] = [dict(e) for e in extras.events]
    return cast(ScenarioData, data)


def _floor_plan(layout: Layout) -> FloorPlanData:
    """Return the ``floor_plan`` section for a design."""
    room = layout.room
    return {
        "width_m": room.width_m,
        "height_m": room.height_m,
        "walls": [_wall(layout, w.id) for w in layout.walls.values()],
        "obstacles": [
            ObstacleData(
                x=o.x, y=o.y, width=o.width, height=o.height,
                label=f"obstacle_{o.id}",
            )
            for o in layout.obstacles.values()
        ],
        "exits": [_exit(layout, d) for d in layout.doorways.values()],
    }


def _wall(layout: Layout, wall_id: int) -> WallData:
    """Return the schema entry for one wall."""
    x, y, w, h = rect_of_wall(layout, layout.walls[wall_id])
    return WallData(x=x, y=y, width=w, height=h, label=f"wall_{wall_id}")


def _exit(layout: Layout, door: Doorway) -> ExitData:
    """Return the schema entry for one doorway (centre on inner face)."""
    wall = layout.walls[door.wall_id]
    side = side_of_wall(layout, wall)
    assert side is not None  # guaranteed by validate()
    x, y, w, h = rect_of_wall(layout, wall)
    inner: dict[Side, tuple[float, float]] = {
        "north": (door.center, y + h),
        "south": (door.center, y),
        "west": (x + w, door.center),
        "east": (x, door.center),
    }
    ex, ey = inner[side]
    return ExitData(
        x=ex, y=ey, width_m=door.width_m, side=side,
        capacity_per_second=door.capacity_per_second,
        label=f"door_{door.id}",
    )


def check_headless(data: ScenarioData) -> None:
    """Build floor plan, flow field and agents to prove the design runs.

    Args:
        data: Scenario produced by :func:`export_scenario`.

    Raises:
        ScenarioValidationError: E.g. no doorway.
        PathfindingError: If no exit cell is reachable.
        DesignValidationError: If agents cannot be spawned.
    """
    fp = data["floor_plan"]
    plan = FloorPlan(
        width_m=fp["width_m"],
        height_m=fp["height_m"],
        walls=tuple(Wall(**w) for w in fp["walls"]),
        obstacles=tuple(Obstacle(**o) for o in fp["obstacles"]),
        exits=tuple(
            Exit(
                x=e["x"], y=e["y"], width_m=e["width_m"],
                side=ExitSide(e["side"]),
                capacity_per_second=e["capacity_per_second"],
                label=e.get("label", ""),
            )
            for e in fp["exits"]
        ),
    )
    FlowField.build(plan, GRID_CELL_SIZE)
    agents = data["agents"]
    try:
        spawn(plan, agents["count"], np.random.default_rng(agents["spawn_seed"]))
    except ValueError as exc:
        raise DesignValidationError(f"Cannot spawn agents: {exc}") from exc


def build_validated_scenario(layout: Layout) -> ScenarioData:
    """Export a layout and prove it runs headlessly.

    Args:
        layout: Design snapshot.

    Returns:
        Validated scenario data.

    Raises:
        CrowdEvacError: Any export or headless-check failure; callers show
            its message on the status line and write nothing.
    """
    data = export_scenario(layout)
    check_headless(data)
    return data
