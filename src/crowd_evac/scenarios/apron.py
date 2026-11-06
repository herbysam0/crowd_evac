"""Pure scenario transform: add a walled egress apron behind every door.

Schema ``2.1`` scenarios give each exit a :data:`APRON_DEPTH_M` deep corridor
outside its host wall (issue #17). :func:`add_aprons` converts a ``2.0``
scenario dict into ``2.1``; it is shared by the one-time migration script and
the designer export, so both produce identical geometry.

For every room side that hosts at least one exit the floor plan grows by the
apron depth. Growing the west or south side shifts all geometry by the depth
(coordinates stay non-negative). The new strip is filled with wall rects
except one gap per door span; the wall pieces beside a gap are the apron's
side walls, so agents cannot wander along the outside.
"""
from __future__ import annotations

import copy
from typing import Any

from crowd_evac.domain.constants import APRON_DEPTH_M
from crowd_evac.domain.door_geometry import ExitSide, door_frame
from crowd_evac.scenarios.schema import ScenarioData, WallData

APRON_VERSION = "2.1"
_ROUND_DIGITS = 6
_MIN_PIECE_M = 1e-6
_X = 0
_Y = 1

APRON_LABEL_PREFIX = "apron_wall_"
SIDE_NAMES: dict[tuple[int, float], str] = {
    (0, -1.0): "west",
    (0, 1.0): "east",
    (1, -1.0): "ylow",
    (1, 1.0): "yhigh",
}

Interval = tuple[float, float]
SideKey = tuple[int, float]


def _r(value: float) -> float:
    """Round away float noise introduced by shifting coordinates."""
    return round(value, _ROUND_DIGITS)


def _merge(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping or touching intervals, sorted ascending."""
    merged: list[Interval] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + _MIN_PIECE_M:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _pieces(lo: float, hi: float, gaps: list[Interval]) -> list[Interval]:
    """Return the parts of ``[lo, hi]`` left after removing *gaps*."""
    out: list[Interval] = []
    cursor = lo
    for g0, g1 in _merge(gaps):
        if g0 - cursor > _MIN_PIECE_M:
            out.append((cursor, min(g0, hi)))
        cursor = max(cursor, g1)
    if hi - cursor > _MIN_PIECE_M:
        out.append((cursor, hi))
    return out


def _door_gaps(raw_fp: Any) -> dict[SideKey, list[Interval]]:
    """Map each hosting side ``(axis, sign)`` to its door spans (unshifted)."""
    width, height = float(raw_fp["width_m"]), float(raw_fp["height_m"])
    sides: dict[SideKey, list[Interval]] = {}
    for e in raw_fp["exits"]:
        frame = door_frame(
            e["x"], e["y"], e["width_m"], ExitSide(e["side"]), width, height
        )
        sides.setdefault((frame.axis, frame.sign), []).append(
            (frame.lo, frame.hi)
        )
    return sides


def _shift_geometry(raw_fp: Any, dx: float, dy: float) -> None:
    """Translate walls, obstacles and exits in place."""
    for group in ("walls", "obstacles", "exits"):
        for item in raw_fp.get(group, []):
            item["x"] = _r(item["x"] + dx)
            item["y"] = _r(item["y"] + dy)


def _strip_walls(
    axis: int,
    sign: float,
    gaps: list[Interval],
    width: float,
    height: float,
    shift: tuple[float, float],
    depth: float,
) -> list[WallData]:
    """Wall rects filling one side's apron strip, minus the door gaps.

    x-normal strips (east / west) span the full new height; y-normal strips
    span only the old width so corners are never double-covered.
    """
    dx, dy = shift
    if axis == _X:
        near = dx + width if sign > 0 else 0.0
        lat_lo, lat_hi = 0.0, height
        gaps = [(g0 + dy, g1 + dy) for g0, g1 in gaps]
    else:
        near = dy + height if sign > 0 else 0.0
        lat_lo, lat_hi = dx, dx + width
        gaps = [(g0 + dx, g1 + dx) for g0, g1 in gaps]
    walls: list[WallData] = []
    for a, b in _pieces(lat_lo, lat_hi, gaps):
        x, y, w, h = (
            (near, a, depth, b - a) if axis == _X else (a, near, b - a, depth)
        )
        walls.append(
            WallData(x=_r(x), y=_r(y), width=_r(w), height=_r(h), label="")
        )
    return walls


def add_aprons(
    data: ScenarioData, depth: float = APRON_DEPTH_M
) -> ScenarioData:
    """Return a schema-2.1 copy of *data* with an apron behind every door.

    Args:
        data: Scenario in schema ``2.0`` (or ``2.1``, returned unchanged).
        depth: Apron depth in metres. Must be positive.

    Returns:
        New scenario dict; *data* is never mutated. A scenario already on
        ``2.1`` is returned as an unchanged copy (idempotent).

    Raises:
        ValueError: If *depth* is not positive.
    """
    if depth <= 0.0:
        raise ValueError(f"apron depth must be positive, got {depth}")
    out = copy.deepcopy(data)
    if out["schema_version"] == APRON_VERSION:
        return out
    fp: Any = out["floor_plan"]
    sides = _door_gaps(fp)
    width, height = float(fp["width_m"]), float(fp["height_m"])
    shift = (
        depth if (_X, -1.0) in sides else 0.0,
        depth if (_Y, -1.0) in sides else 0.0,
    )
    _shift_geometry(fp, *shift)
    new_walls: list[WallData] = []
    for (axis, sign), gaps in sorted(sides.items()):
        side = _strip_walls(axis, sign, gaps, width, height, shift, depth)
        for wall in side:
            n = len(new_walls) + 1
            wall["label"] = f"{APRON_LABEL_PREFIX}{SIDE_NAMES[axis, sign]}_{n}"
            new_walls.append(wall)
    fp["walls"] = list(fp.get("walls", [])) + new_walls
    fp["width_m"] = _r(width + shift[0] + (depth if (_X, 1.0) in sides else 0))
    fp["height_m"] = _r(
        height + shift[1] + (depth if (_Y, 1.0) in sides else 0)
    )
    out["schema_version"] = APRON_VERSION
    return out
