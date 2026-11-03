"""Convert a loaded scenario into an editable design :class:`Layout`.

The inverse of :mod:`crowd_evac.designer.export`. Walls become centrelines
with their own thickness, exits attach to the nearest boundary wall (matched
by position, not by the ``side`` label, which some scenarios get wrong), and
settings the designer does not edit (agents, simulation, events) are carried
through unchanged so saving never silently drops them.
"""
from __future__ import annotations

from crowd_evac.designer.geometry import Rect
from crowd_evac.designer.layout import (
    Doorway,
    Layout,
    Obstacle,
    RoomSettings,
    Wall,
    rect_of_wall,
    side_of_wall,
    validate,
    wall_span,
)
from crowd_evac.domain.errors import DesignValidationError
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.floor_plan import Wall as PlanWall
from crowd_evac.scenarios.schema import ScenarioData

# An exit centre may sit up to this far from its wall (bundled scenarios
# place it on the inner face or about one cell inside the room).
MAX_EXIT_WALL_DISTANCE_M: float = 1.0


def layout_from_scenario(plan: FloorPlan, data: ScenarioData) -> Layout:
    """Build a validated design layout from a loaded scenario.

    Args:
        plan: Floor plan returned by the scenario loader.
        data: Scenario data (schema version, name, floor plan).

    Returns:
        A layout equivalent to the scenario's geometry.

    Raises:
        DesignValidationError: If the scenario cannot be represented, e.g. an
            exit with no boundary wall, or geometry the designer forbids.
    """
    room = RoomSettings(data["name"], plan.width_m, plan.height_m)
    walls = {i: _wall(i, w) for i, w in enumerate(plan.walls, start=1)}
    base = len(walls)
    obstacles = {
        base + i: Obstacle(base + i, o.x, o.y, o.width, o.height)
        for i, o in enumerate(plan.obstacles, start=1)
    }
    bare = Layout(room, walls, {}, {})
    base += len(obstacles)
    doorways = {
        base + i: _doorway(bare, base + i, e)
        for i, e in enumerate(plan.exits, start=1)
    }
    layout = Layout(room, walls, obstacles, doorways)
    validate(layout)
    return layout


def _wall(wall_id: int, w: PlanWall) -> Wall:
    """Convert a wall rectangle to a centreline plus thickness."""
    if w.width >= w.height:
        y = w.y + w.height / 2.0
        return Wall(wall_id, w.x, y, w.x + w.width, y, w.height)
    x = w.x + w.width / 2.0
    return Wall(wall_id, x, w.y, x, w.y + w.height, w.width)


def _distance(rect: Rect, x: float, y: float) -> float:
    """Return the distance from a point to a rectangle (0 if inside)."""
    rx, ry, rw, rh = rect
    dx = max(rx - x, 0.0, x - (rx + rw))
    dy = max(ry - y, 0.0, y - (ry + rh))
    return float((dx * dx + dy * dy) ** 0.5)


def _doorway(layout: Layout, door_id: int, exit_: Exit) -> Doorway:
    """Attach an exit to the nearest boundary wall of matching orientation."""
    along_x = exit_.side in (ExitSide.NORTH, ExitSide.SOUTH)
    best: tuple[float, Wall] | None = None
    for wall in layout.walls.values():
        if wall.horizontal != along_x or side_of_wall(layout, wall) is None:
            continue
        lo, hi = wall_span(wall)
        centre = exit_.x if along_x else exit_.y
        if not lo <= centre <= hi:
            continue
        dist = _distance(rect_of_wall(layout, wall), exit_.x, exit_.y)
        if dist <= MAX_EXIT_WALL_DISTANCE_M and (
            best is None or dist < best[0]
        ):
            best = (dist, wall)
    if best is None:
        raise DesignValidationError(
            f"Exit '{exit_.label}' at ({exit_.x:g}, {exit_.y:g}) is not on any "
            "wall touching the room boundary"
        )
    centre = exit_.x if along_x else exit_.y
    return Doorway(
        door_id, best[1].id, centre, exit_.width_m, exit_.capacity_per_second
    )
