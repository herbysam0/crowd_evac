"""Pure rectangle geometry helpers for the room designer.

Coordinates are metres, origin bottom-left, y-up. ``north`` is the bottom
boundary (``y=0``) and ``south`` the top boundary (``y=height_m``); this
matches the existing renderer and bundled scenarios.
"""
from __future__ import annotations

from typing import Literal, TypeAlias

# Rect is (x, y, width, height) in metres.
Rect: TypeAlias = tuple[float, float, float, float]
Side: TypeAlias = Literal["north", "south", "east", "west"]

DEFAULT_WALL_THICKNESS_M: float = 0.5
# Tolerance so touching edges and float noise never count as overlap.
EPS: float = 1e-9


def overlaps(a: Rect, b: Rect) -> bool:
    """Return True if two rectangles overlap with positive area.

    Args:
        a: First rectangle ``(x, y, width, height)``.
        b: Second rectangle ``(x, y, width, height)``.

    Returns:
        False when the rectangles only touch along an edge or corner.
    """
    return (
        a[0] < b[0] + b[2] - EPS
        and b[0] < a[0] + a[2] - EPS
        and a[1] < b[1] + b[3] - EPS
        and b[1] < a[1] + a[3] - EPS
    )


def inside_room(rect: Rect, width_m: float, height_m: float) -> bool:
    """Return True if ``rect`` lies fully within the room bounds."""
    x, y, w, h = rect
    return (
        x >= -EPS
        and y >= -EPS
        and x + w <= width_m + EPS
        and y + h <= height_m + EPS
    )


def is_horizontal(y0: float, y1: float) -> bool:
    """Return True when a wall centreline runs along the x axis."""
    return abs(y0 - y1) <= EPS


def wall_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width_m: float,
    height_m: float,
    thickness: float = DEFAULT_WALL_THICKNESS_M,
) -> Rect:
    """Return the rectangle of an axis-aligned wall centreline.

    The wall is ``thickness`` metres thick, centred on the line, then
    shifted inward when it would poke through a room boundary so that
    walls drawn on the boundary end up flush with it.

    Args:
        x0: Start x. y0: Start y. x1: End x. y1: End y.
        width_m: Room width. height_m: Room height.
        thickness: Wall thickness in metres.

    Returns:
        Wall rectangle ``(x, y, width, height)``. The caller guarantees
        the line is axis-aligned and non-degenerate.
    """
    half = thickness / 2.0
    if is_horizontal(y0, y1):
        y = min(max(y0 - half, 0.0), max(height_m - thickness, 0.0))
        return (min(x0, x1), y, abs(x1 - x0), thickness)
    x = min(max(x0 - half, 0.0), max(width_m - thickness, 0.0))
    return (x, min(y0, y1), thickness, abs(y1 - y0))


def perimeter_side(
    rect: Rect, horizontal: bool, width_m: float, height_m: float
) -> Side | None:
    """Return the room side a wall rectangle is flush against, if any.

    Args:
        rect: Wall rectangle.
        horizontal: Whether the wall runs along the x axis.
        width_m: Room width.
        height_m: Room height.

    Returns:
        ``"north"`` for ``y=0``, ``"south"`` for ``y=height_m``, ``"west"``
        for ``x=0``, ``"east"`` for ``x=width_m``; ``None`` for interior
        walls.
    """
    x, y, w, h = rect
    if horizontal:
        if y <= EPS:
            return "north"
        if y + h >= height_m - EPS:
            return "south"
        return None
    if x <= EPS:
        return "west"
    if x + w >= width_m - EPS:
        return "east"
    return None
