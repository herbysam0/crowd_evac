"""Faint 1 m grid drawn under geometry and agents.

Shared by the simulation window and the room designer so both show an
identical scale aid. Segment maths is pure (no GL); :func:`draw_grid` issues
a single batched arcade call.
"""
from __future__ import annotations

import arcade
from arcade.types import RGBA255

GRID_STEP_M: float = 1.0
# White at low alpha stays faint on any floor colour or background image.
GRID_LINE_COLOR: RGBA255 = (255, 255, 255, 28)
GRID_LINE_WIDTH_PX: float = 1.0


def grid_segments(
    width_m: float,
    height_m: float,
    ppm: float,
    x_off: float,
    y_off: float,
) -> list[tuple[float, float]]:
    """Return pixel end points of the interior 1 m grid lines.

    Room edges are omitted: they coincide with the floor border. Lines sit
    at whole metres ``1 .. int(extent) - 1`` on each axis.

    Args:
        width_m: Room width in metres.
        height_m: Room height in metres.
        ppm: Pixels per metre. Must be positive.
        x_off: Pixel x of the room origin.
        y_off: Pixel y of the room origin.

    Returns:
        Flat list of points; each consecutive pair is one segment. Empty if
        the room is 1 m or smaller on both axes.

    Raises:
        ValueError: If ``ppm`` is not positive.
    """
    if ppm <= 0:
        raise ValueError(f"ppm must be positive, got {ppm}")
    top = y_off + height_m * ppm
    right = x_off + width_m * ppm
    points: list[tuple[float, float]] = []
    for i in range(1, int(width_m)):
        x = x_off + i * GRID_STEP_M * ppm
        points += [(x, y_off), (x, top)]
    for j in range(1, int(height_m)):
        y = y_off + j * GRID_STEP_M * ppm
        points += [(x_off, y), (right, y)]
    return points


def draw_grid(
    width_m: float,
    height_m: float,
    ppm: float,
    x_off: float,
    y_off: float,
) -> None:
    """Draw the 1 m grid in one batched call.

    Args:
        width_m: Room width in metres.
        height_m: Room height in metres.
        ppm: Pixels per metre. Must be positive.
        x_off: Pixel x of the room origin.
        y_off: Pixel y of the room origin.

    Raises:
        ValueError: If ``ppm`` is not positive.
    """
    points = grid_segments(width_m, height_m, ppm, x_off, y_off)
    if points:
        arcade.draw_lines(points, GRID_LINE_COLOR, GRID_LINE_WIDTH_PX)
