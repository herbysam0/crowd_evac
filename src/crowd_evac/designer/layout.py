"""Immutable design entities and whole-layout validation.

The :class:`Layout` snapshot is validated in full after every candidate
edit, so the design model can reject a change and keep its previous state.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from crowd_evac.designer.geometry import (
    DEFAULT_WALL_THICKNESS_M,
    EPS,
    Rect,
    Side,
    inside_room,
    overlaps,
    perimeter_side,
    wall_rect,
)
from crowd_evac.domain.errors import DesignValidationError


@dataclass(frozen=True)
class RoomSettings:
    """Room-level scenario settings."""

    name: str
    width_m: float
    height_m: float


@dataclass(frozen=True)
class Wall:
    """Axis-aligned wall defined by its centreline from start to end."""

    id: int
    x0: float
    y0: float
    x1: float
    y1: float
    thickness: float = DEFAULT_WALL_THICKNESS_M

    @property
    def horizontal(self) -> bool:
        """Whether the wall runs along the x axis."""
        return abs(self.y0 - self.y1) <= EPS

    @property
    def length(self) -> float:
        """Centreline length in metres."""
        return abs(self.x1 - self.x0) + abs(self.y1 - self.y0)


@dataclass(frozen=True)
class Obstacle:
    """Interior rectangular obstacle."""

    id: int
    x: float
    y: float
    width: float
    height: float

    @property
    def rect(self) -> Rect:
        """Obstacle rectangle ``(x, y, width, height)``."""
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class Doorway:
    """Exit opening attached to a perimeter wall.

    ``center`` is the opening centre along the wall axis (x for horizontal
    walls, y for vertical ones).
    """

    id: int
    wall_id: int
    center: float
    width_m: float
    capacity_per_second: int


# Used when a design is not loaded from a file; the designer has no UI for
# these (schema 1.0 requires the agents block).
DEFAULT_AGENT_COUNT: int = 50
DEFAULT_SPAWN_SEED: int = 42


@dataclass(frozen=True)
class Extras:
    """Scenario settings the designer does not edit but must preserve."""

    agent_count: int = DEFAULT_AGENT_COUNT
    spawn_seed: int = DEFAULT_SPAWN_SEED
    simulation: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class Layout:
    """Complete design snapshot."""

    room: RoomSettings
    walls: Mapping[int, Wall]
    obstacles: Mapping[int, Obstacle]
    doorways: Mapping[int, Doorway]
    extras: Extras = field(default_factory=Extras)


def rect_of_wall(layout: Layout, wall: Wall) -> Rect:
    """Return the thickened rectangle of ``wall`` inside the room."""
    r = layout.room
    return wall_rect(
        wall.x0, wall.y0, wall.x1, wall.y1, r.width_m, r.height_m,
        wall.thickness,
    )


def side_of_wall(layout: Layout, wall: Wall) -> Side | None:
    """Return the perimeter side ``wall`` is flush with, or None."""
    r = layout.room
    return perimeter_side(
        rect_of_wall(layout, wall), wall.horizontal, r.width_m, r.height_m
    )


def wall_span(wall: Wall) -> tuple[float, float]:
    """Return the wall's extent along its own axis."""
    if wall.horizontal:
        return (min(wall.x0, wall.x1), max(wall.x0, wall.x1))
    return (min(wall.y0, wall.y1), max(wall.y0, wall.y1))


def rect_of_doorway(layout: Layout, door: Doorway) -> Rect:
    """Return the opening footprint: its span across the wall thickness."""
    wall = layout.walls[door.wall_id]
    x, y, w, h = rect_of_wall(layout, wall)
    lo = door.center - door.width_m / 2.0
    if wall.horizontal:
        return (lo, y, door.width_m, h)
    return (x, lo, w, door.width_m)


def validate(layout: Layout) -> None:
    """Validate a whole layout.

    Args:
        layout: Candidate snapshot.

    Raises:
        DesignValidationError: On the first violated rule.
    """
    _check_room(layout.room)
    for wall in layout.walls.values():
        _check_wall(layout, wall)
    for obs in layout.obstacles.values():
        _check_obstacle(layout, obs)
    for door in layout.doorways.values():
        _check_doorway(layout, door)
    for a, b in itertools.combinations(layout.doorways.values(), 2):
        if overlaps(rect_of_doorway(layout, a), rect_of_doorway(layout, b)):
            raise DesignValidationError("Doorways overlap each other")


def _check_room(room: RoomSettings) -> None:
    """Reject non-positive room size."""
    if room.width_m <= 0 or room.height_m <= 0:
        raise DesignValidationError("Room size must be positive")


def _check_wall(layout: Layout, wall: Wall) -> None:
    """Reject degenerate, diagonal or out-of-room walls."""
    r = layout.room
    if wall.length <= EPS:
        raise DesignValidationError("Wall length must be positive")
    if wall.thickness <= EPS:
        raise DesignValidationError("Wall thickness must be positive")
    if abs(wall.x0 - wall.x1) > EPS and not wall.horizontal:
        raise DesignValidationError("Wall must be horizontal or vertical")
    in_line = all(
        -EPS <= x <= r.width_m + EPS and -EPS <= y <= r.height_m + EPS
        for x, y in ((wall.x0, wall.y0), (wall.x1, wall.y1))
    )
    rect = rect_of_wall(layout, wall)
    if not in_line or not inside_room(rect, r.width_m, r.height_m):
        raise DesignValidationError(f"Wall {wall.id} is outside the room")


def _check_obstacle(layout: Layout, obs: Obstacle) -> None:
    """Reject bad size/position; wall and doorway overlaps are checked here."""
    r = layout.room
    if obs.width <= 0 or obs.height <= 0:
        raise DesignValidationError("Obstacle size must be positive")
    if not inside_room(obs.rect, r.width_m, r.height_m):
        raise DesignValidationError(
            f"Obstacle {obs.id} is outside the room"
        )
    if any(
        overlaps(obs.rect, rect_of_wall(layout, w))
        for w in layout.walls.values()
    ):
        raise DesignValidationError("Wall and obstacle overlap")
    if any(
        overlaps(obs.rect, rect_of_doorway(layout, d))
        for d in layout.doorways.values()
    ):
        raise DesignValidationError("Doorway and obstacle overlap")


def _check_doorway(layout: Layout, door: Doorway) -> None:
    """Reject doorways that are mis-attached, mis-sized or misplaced."""
    wall = layout.walls.get(door.wall_id)
    if wall is None:
        raise DesignValidationError("Doorway wall does not exist")
    if side_of_wall(layout, wall) is None:
        raise DesignValidationError(
            f"Doorway {door.id} is only allowed on a wall touching the "
            "room boundary (its wall is no longer on the room edge)"
        )
    if door.width_m <= 0 or door.capacity_per_second <= 0:
        raise DesignValidationError(
            "Doorway width and capacity must be positive"
        )
    lo = door.center - door.width_m / 2.0
    hi = door.center + door.width_m / 2.0
    limit = layout.room.width_m if wall.horizontal else layout.room.height_m
    w_lo, w_hi = wall_span(wall)
    # Touching an end is allowed: end placement extends outward from it.
    if lo < -EPS or hi > limit + EPS or lo > w_hi + EPS or hi < w_lo - EPS:
        raise DesignValidationError(
            f"Doorway {door.id} does not fit on its wall"
        )
