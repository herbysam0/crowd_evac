"""Convert a design :class:`Layout` to schema-2.0 ``ScenarioData``.

Follows the conventions of the bundled scenarios: ``north`` is the bottom
wall (``y=0``), ``south`` the top wall (``y=height_m``), and an exit's
centre sits on the inner face of its wall.
"""
from __future__ import annotations

from crowd_evac.designer.geometry import Side
from crowd_evac.designer.layout import (
    Doorway,
    Layout,
    rect_of_wall,
    side_of_wall,
    validate,
)
from crowd_evac.domain.constants import GRID_CELL_SIZE
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

SCHEMA_VERSION = "2.0"


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
    return ScenarioData(
        schema_version=SCHEMA_VERSION,
        name=layout.room.name,
        floor_plan=_floor_plan(layout),
    )


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
    """Build floor plan and flow field to prove the design runs.

    Args:
        data: Scenario produced by :func:`export_scenario`.

    Raises:
        ScenarioValidationError: E.g. no doorway.
        PathfindingError: If no exit cell is reachable.
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
