"""Door-frame geometry: outward normals, host walls, and egress reach.

An exit is authored as a centre point, a width and a ``side`` label. This
module turns that into a :class:`DoorFrame` — the door plane, its outward unit
normal and its lateral span — and answers the geometric questions the grid
rasteriser (:mod:`crowd_evac.domain.floor_grid`) and the egress model
(:mod:`crowd_evac.domain.egress_geometry`) both need:

- Which walls *host* the door (the wall the opening is cut through)?
- How much free space lies behind the door plane (:func:`egress_reach`)?

The outward direction is derived from geometry (the exit lies nearer to one
boundary of the room than the other), not from the ``side`` label alone:
bundled scenarios disagree on whether ``north`` means ``y = 0`` or
``y = height``. The label only picks the axis, and breaks the tie for an exit
exactly at the room's mid-line.

Pure Python — no NumPy, framework or I/O imports; no dependency on
:mod:`crowd_evac.domain.floor_plan` (which imports this module).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

HOST_WALL_MAX_GAP_M: float = 1.0
"""Max distance (m) from the door plane to a wall's near face for it to host.

Matches the designer importer's tolerance: bundled scenarios put an exit on
the wall's inner face, in its middle, or up to ~1 m inside the room.
"""

DOOR_CLEAR_MARGIN_M: float = 0.05
"""Slack (m) for door-line tests (queue join, evacuation depth).

Wall push-out holds a centre exactly one radius off a wall, so a depth equal
to ``reach - radius`` would be reachable only in the limit.
"""

_GEOM_EPS: float = 1e-9
"""Absolute tolerance (m) for touching / containment comparisons."""


class ExitSide(str, Enum):
    """Cardinal wall on which an exit opening is positioned."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


# Tie-break signs for an exit exactly on the room mid-line; follows the
# renderer convention (NORTH drawn at y = 0, SOUTH at y = height).
_SIDE_SIGN: dict[ExitSide, float] = {
    ExitSide.WEST: -1.0,
    ExitSide.EAST: 1.0,
    ExitSide.NORTH: -1.0,
    ExitSide.SOUTH: 1.0,
}


class RectLike(Protocol):
    """Anything with an axis-aligned ``x, y, width, height`` box (metres)."""

    @property
    def x(self) -> float:
        """Left edge."""

    @property
    def y(self) -> float:
        """Bottom edge."""

    @property
    def width(self) -> float:
        """Extent along x."""

    @property
    def height(self) -> float:
        """Extent along y."""


@dataclass(frozen=True)
class DoorFrame:
    """Door plane, outward normal and lateral span of one exit.

    Attributes:
        px: x of the exit centre (a point on the door plane), metres.
        py: y of the exit centre, metres.
        axis: Normal axis — ``0`` for an x-normal (EAST / WEST) door,
            ``1`` for a y-normal (NORTH / SOUTH) door.
        sign: ``+1.0`` or ``-1.0``; outward direction along ``axis``.
        lo: Lower lateral bound of the authored opening, metres.
        hi: Upper lateral bound of the authored opening, metres.
    """

    px: float
    py: float
    axis: int
    sign: float
    lo: float
    hi: float

    @property
    def lateral_axis(self) -> int:
        """Axis running along the opening (``1 - axis``)."""
        return 1 - self.axis

    @property
    def normal(self) -> tuple[float, float]:
        """Outward unit normal as ``(nx, ny)``."""
        return (self.sign, 0.0) if self.axis == 0 else (0.0, self.sign)

    @property
    def plane(self) -> float:
        """Door-plane coordinate along :attr:`axis`."""
        return self.px if self.axis == 0 else self.py

    @property
    def centre_lateral(self) -> float:
        """Lateral coordinate of the exit centre."""
        return self.py if self.axis == 0 else self.px

    def outward_range(self, rect: RectLike) -> tuple[float, float]:
        """Return ``(s_min, s_max)`` signed outward distances spanned by *rect*.

        ``s`` is measured from the door plane along the outward normal, so
        negative values lie inside the room and positive values beyond it.
        """
        a0 = rect.x if self.axis == 0 else rect.y
        a1 = a0 + (rect.width if self.axis == 0 else rect.height)
        s0 = (a0 - self.plane) * self.sign
        s1 = (a1 - self.plane) * self.sign
        return min(s0, s1), max(s0, s1)

    def lateral_range(self, rect: RectLike) -> tuple[float, float]:
        """Return ``(lat_min, lat_max)`` of *rect* along the opening axis."""
        if self.axis == 0:
            return rect.y, rect.y + rect.height
        return rect.x, rect.x + rect.width


def door_frame(
    x: float,
    y: float,
    width_m: float,
    side: ExitSide,
    room_width: float,
    room_height: float,
) -> DoorFrame:
    """Build the :class:`DoorFrame` of an exit inside a room.

    The outward sign points toward the room boundary nearer to the exit
    centre; the ``side`` label selects the axis and breaks exact ties.

    Args:
        x: Exit centre x in metres.
        y: Exit centre y in metres.
        width_m: Opening width in metres. Must be positive.
        side: Authored side label.
        room_width: Floor-plan width in metres.
        room_height: Floor-plan height in metres.

    Returns:
        The exit's door frame.

    Example:
        >>> f = door_frame(1.0, 8.2, 1.6, ExitSide.WEST, 16.2, 10.0)
        >>> f.normal, round(f.lo, 6), round(f.hi, 6)
        ((-1.0, 0.0), 7.4, 9.0)
    """
    axis = 0 if side in (ExitSide.EAST, ExitSide.WEST) else 1
    coord, extent = (x, room_width) if axis == 0 else (y, room_height)
    offset = coord - extent / 2.0
    if abs(offset) <= _GEOM_EPS:
        sign = _SIDE_SIGN[side]
    else:
        sign = 1.0 if offset > 0.0 else -1.0
    lateral = y if axis == 0 else x
    half = width_m / 2.0
    return DoorFrame(x, y, axis, sign, lateral - half, lateral + half)


def is_host(
    frame: DoorFrame,
    rect: RectLike,
    max_gap: float = HOST_WALL_MAX_GAP_M,
) -> bool:
    """Return whether *rect* is a wall the door opening is cut through.

    A host wall runs parallel to the opening, covers the exit centre
    laterally, reaches the door plane or beyond it, and its near face is at
    most *max_gap* past the plane.

    Args:
        frame: Door frame of the exit.
        rect: Candidate wall rectangle.
        max_gap: Max outward distance from the plane to the wall's near face.

    Returns:
        ``True`` when *rect* hosts the door.
    """
    s0, s1 = frame.outward_range(rect)
    lat0, lat1 = frame.lateral_range(rect)
    parallel = (lat1 - lat0) >= (s1 - s0)
    covers = lat0 - _GEOM_EPS <= frame.centre_lateral <= lat1 + _GEOM_EPS
    return (
        parallel
        and covers
        and s1 >= -_GEOM_EPS
        and s0 <= max_gap + _GEOM_EPS
    )


def egress_reach(
    frame: DoorFrame,
    blockers: Iterable[RectLike],
    room_width: float,
    room_height: float,
) -> float:
    """Free outward distance behind the door plane along its centre line.

    Measured from the plane to the room boundary or to the near face of the
    first blocker past the plane, whichever is closer. Host walls are carved
    open across the span, so they must not be passed as blockers.

    Args:
        frame: Door frame of the exit.
        blockers: Non-host walls and obstacles; any covering the exit centre
            line past the plane limits the reach.
        room_width: Floor-plan width in metres.
        room_height: Floor-plan height in metres.

    Returns:
        Reach in metres (``>= 0``); ``0`` for a door on the room boundary.
    """
    extent = room_width if frame.axis == 0 else room_height
    reach = extent - frame.plane if frame.sign > 0 else frame.plane
    reach = max(reach, 0.0)
    for rect in blockers:
        lat0, lat1 = frame.lateral_range(rect)
        if not lat0 < frame.centre_lateral < lat1:
            continue
        s0, s1 = frame.outward_range(rect)
        if s1 > _GEOM_EPS:
            reach = min(reach, max(s0, 0.0))
    return reach
