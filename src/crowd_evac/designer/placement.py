"""Pure pointer maths for the designer: snapping, hit testing, door centres."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from crowd_evac.designer.geometry import Rect
from crowd_evac.designer.layout import (
    Layout,
    Wall,
    rect_of_doorway,
    rect_of_wall,
    wall_span,
)

SelectionKind: TypeAlias = Literal["wall", "doorway", "obstacle"]

# Pointer snap resolution in metres (finer than the 0.25 m simulation grid).
SNAP_STEP_M: float = 0.1
# Decimals kept so 3 * 0.1 is stored as 0.3, not 0.30000000000000004.
SNAP_DECIMALS: int = 6

# A click within this distance of a wall end places the door at that end.
END_SNAP_M: float = 0.5
# Extra pick margin so thin walls are easy to click.
HIT_TOLERANCE_M: float = 0.15


@dataclass(frozen=True)
class Selection:
    """Reference to one selected entity."""

    kind: SelectionKind
    id: int


def snap(value: float, step: float = SNAP_STEP_M) -> float:
    """Round ``value`` to the nearest multiple of ``step``."""
    return round(round(value / step) * step, SNAP_DECIMALS)


def clamp(value: float, low: float, high: float) -> float:
    """Limit ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


def axis_end(
    sx: float, sy: float, px: float, py: float
) -> tuple[float, float]:
    """Return the wall end for a drag from ``(sx, sy)`` to ``(px, py)``.

    The wall follows the dominant axis of the drag, so it is always
    horizontal or vertical.
    """
    if abs(px - sx) >= abs(py - sy):
        return (px, sy)
    return (sx, py)


def _hit(rect: Rect, x: float, y: float) -> bool:
    """Return True if ``(x, y)`` is inside ``rect`` grown by the tolerance."""
    rx, ry, rw, rh = rect
    t = HIT_TOLERANCE_M
    return rx - t <= x <= rx + rw + t and ry - t <= y <= ry + rh + t


def hit_test(layout: Layout, x: float, y: float) -> Selection | None:
    """Return the entity under a point; doorways beat obstacles beat walls."""
    for door in layout.doorways.values():
        if _hit(rect_of_doorway(layout, door), x, y):
            return Selection("doorway", door.id)
    for obs in layout.obstacles.values():
        if _hit(obs.rect, x, y):
            return Selection("obstacle", obs.id)
    for wall in layout.walls.values():
        if _hit(rect_of_wall(layout, wall), x, y):
            return Selection("wall", wall.id)
    return None


def door_center(
    layout: Layout, wall: Wall, x: float, y: float, width_m: float
) -> float:
    """Return the opening centre for a click on a wall.

    Near a wall end the opening starts at that end and extends outward
    (centre = end +/- width/2); if that would leave the room it is placed
    flush inside the end instead. Elsewhere the centre snaps to the grid
    and is clamped so the opening stays within the wall span.

    Args:
        layout: Current design.
        wall: Wall that was clicked.
        x: Click x in metres.
        y: Click y in metres.
        width_m: Opening width.

    Returns:
        Centre coordinate along the wall axis.
    """
    lo, hi = wall_span(wall)
    along = x if wall.horizontal else y
    room = layout.room
    limit = room.width_m if wall.horizontal else room.height_m
    half = width_m / 2.0
    end, outward = min(
        ((lo, -1.0), (hi, 1.0)), key=lambda e: abs(along - e[0])
    )
    if abs(along - end) <= END_SNAP_M:
        centre = end + outward * half
        if centre - half >= 0.0 and centre + half <= limit:
            return centre
        return end - outward * half
    if hi - lo <= width_m:
        return (lo + hi) / 2.0
    return clamp(snap(along), lo + half, hi - half)
