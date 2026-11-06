"""Rasterise a floor plan to a walkability grid and exit goal cells.

Backs :meth:`FloorPlan.walkable_mask` and :meth:`FloorPlan.exit_cells`.

Door openings are carved **only through their host walls** (see
:func:`~crowd_evac.domain.door_geometry.is_host`), across the host's full
thickness, over the door span quantised *inward* — so the carved gap never
exceeds the authored width. Every non-host wall and every obstacle is then
re-blocked, so a door bounding box can never re-open an interior wall that
merely touches the door edge.

Goal cells are the outermost walkable cells reached by marching outward from
the door plane through the opening (the apron's outer edge when one exists,
else the outer face of the carved gap), so the flow field drives agents
*through* the door rather than to its inner face.

Grid convention: cell ``[r, c]`` covers
``[c*cs, (c+1)*cs) x [r*cs, (r+1)*cs)``; origin at the room's bottom-left.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.door_geometry import DoorFrame, door_frame, is_host
from crowd_evac.domain.errors import ScenarioValidationError

if TYPE_CHECKING:
    from crowd_evac.domain.door_geometry import RectLike
    from crowd_evac.domain.floor_plan import Exit, FloorPlan

Cell = tuple[int, int]

_CELL_EPS: float = 1e-6
"""Tolerance in *cell units* so 7.4 / 0.05 = 147.999… quantises to 148."""


def exit_frame(floor_plan: FloorPlan, exit_: Exit) -> DoorFrame:
    """Return the door frame of *exit_* within *floor_plan*."""
    return door_frame(
        exit_.x, exit_.y, exit_.width_m, exit_.side,
        floor_plan.width_m, floor_plan.height_m,
    )


def host_flags(floor_plan: FloorPlan) -> list[list[bool]]:
    """Return ``flags[e][w]``: whether wall *w* hosts exit *e*."""
    return [
        [is_host(exit_frame(floor_plan, e), w) for w in floor_plan.walls]
        for e in floor_plan.exits
    ]


def walkable_mask(
    floor_plan: FloorPlan, cell_size: float
) -> npt.NDArray[np.bool_]:
    """Compute the boolean walkability grid of *floor_plan*.

    Args:
        floor_plan: Source geometry.
        cell_size: Cell side length in metres. Must be > 0.

    Returns:
        Bool array ``(ceil(height/cs), ceil(width/cs))``; ``True`` = passable.

    Raises:
        ValueError: If *cell_size* is not positive.
        ScenarioValidationError: If a door cut through a host wall spans no
            whole cell.
    """
    rows, cols = _grid_shape(floor_plan, cell_size)
    mask: npt.NDArray[np.bool_] = np.ones((rows, cols), dtype=np.bool_)
    solids: tuple[RectLike, ...] = (*floor_plan.walls, *floor_plan.obstacles)
    for rect in solids:
        _set_rect(mask, rect, cell_size, False)

    flags = host_flags(floor_plan)
    for e_idx, exit_ in enumerate(floor_plan.exits):
        if not any(flags[e_idx]):
            continue  # door in open space: nothing to carve
        frame = exit_frame(floor_plan, exit_)
        lat = _inward_lateral_cells(frame, exit_.label, cell_size, mask.shape)
        for w_idx, wall in enumerate(floor_plan.walls):
            if flags[e_idx][w_idx]:
                _carve(mask, frame, wall, lat, cell_size)

    # Interior walls always win over door openings: re-block every wall that
    # hosts no door, so a door edge touching an interior wall cannot leak.
    for w_idx, wall in enumerate(floor_plan.walls):
        if not any(row[w_idx] for row in flags):
            _set_rect(mask, wall, cell_size, False)
    for obs in floor_plan.obstacles:
        _set_rect(mask, obs, cell_size, False)
    return mask


def exit_cells(floor_plan: FloorPlan, cell_size: float) -> list[Cell]:
    """Return flow-field goal cells: outermost walkable cells of each door.

    For every lateral cell line of a door's inward span, march outward from
    the room-side cell at the door plane through walkable cells; the last
    one reached is a goal. Lines whose room-side cell is blocked are skipped.

    Args:
        floor_plan: Source geometry.
        cell_size: Cell side length in metres. Must be > 0.

    Returns:
        ``(row, col)`` goal cells, all walkable in :func:`walkable_mask`.
        May contain duplicates if exits overlap.

    Raises:
        ValueError: If *cell_size* is not positive.
        ScenarioValidationError: If an exit's span covers no whole cell.
    """
    mask = walkable_mask(floor_plan, cell_size)
    cells: list[Cell] = []
    for exit_ in floor_plan.exits:
        frame = exit_frame(floor_plan, exit_)
        l0, l1 = _inward_lateral_cells(frame, exit_.label, cell_size, mask.shape)
        for lat in range(l0, l1):
            goal = _march_outward(mask, frame, lat, cell_size)
            if goal is not None:
                cells.append(goal)
    return cells


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _grid_shape(floor_plan: FloorPlan, cell_size: float) -> tuple[int, int]:
    """Return ``(rows, cols)``; raise ValueError for a non-positive size."""
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size!r}")
    return (
        math.ceil(floor_plan.height_m / cell_size - _CELL_EPS),
        math.ceil(floor_plan.width_m / cell_size - _CELL_EPS),
    )


def _outward_cells(lo: float, hi: float, cell_size: float, n: int) -> range:
    """Cells touched by ``[lo, hi)`` (outward rounding), clamped to grid."""
    start = max(0, math.floor(lo / cell_size + _CELL_EPS))
    stop = min(n, math.ceil(hi / cell_size - _CELL_EPS))
    return range(start, stop)


def _set_rect(
    mask: npt.NDArray[np.bool_], rect: RectLike, cell_size: float, value: bool
) -> None:
    """Assign *value* to every cell a rectangle touches (outward rounding)."""
    rows = _outward_cells(rect.y, rect.y + rect.height, cell_size, mask.shape[0])
    cols = _outward_cells(rect.x, rect.x + rect.width, cell_size, mask.shape[1])
    mask[rows.start:rows.stop, cols.start:cols.stop] = value


def _inward_lateral_cells(
    frame: DoorFrame, label: str, cell_size: float, shape: tuple[int, ...]
) -> tuple[int, int]:
    """Cells lying wholly inside the door span (inward rounding).

    Raises:
        ScenarioValidationError: If no whole cell fits in the span.
    """
    n = int(shape[0] if frame.lateral_axis == 1 else shape[1])
    l0 = max(0, math.ceil(frame.lo / cell_size - _CELL_EPS))
    l1 = min(n, math.floor(frame.hi / cell_size + _CELL_EPS))
    if l1 <= l0:
        raise ScenarioValidationError(
            f"Exit '{label}' span [{frame.lo:g}, {frame.hi:g}] m covers no "
            f"whole {cell_size:g} m grid cell; the door is unusable"
        )
    return l0, l1


def _carve(
    mask: npt.NDArray[np.bool_],
    frame: DoorFrame,
    wall: RectLike,
    lat: tuple[int, int],
    cell_size: float,
) -> None:
    """Open a host wall across its full thickness over the lateral span."""
    if frame.axis == 0:
        cols = _outward_cells(wall.x, wall.x + wall.width, cell_size, mask.shape[1])
        mask[lat[0]:lat[1], cols.start:cols.stop] = True
    else:
        rows = _outward_cells(
            wall.y, wall.y + wall.height, cell_size, mask.shape[0]
        )
        mask[rows.start:rows.stop, lat[0]:lat[1]] = True


def _march_outward(
    mask: npt.NDArray[np.bool_], frame: DoorFrame, lat: int, cell_size: float
) -> Cell | None:
    """Walk outward from the plane along one lateral line; return last cell."""
    n = int(mask.shape[1] if frame.axis == 0 else mask.shape[0])
    step = 1 if frame.sign > 0 else -1
    k = math.floor(frame.plane / cell_size - step * _CELL_EPS)
    k = min(max(k, 0), n - 1)

    def cell(i: int) -> Cell:
        """Map a normal-axis index on this line to ``(row, col)``."""
        return (lat, i) if frame.axis == 0 else (i, lat)

    if not mask[cell(k)]:
        return None
    while 0 <= k + step < n and mask[cell(k + step)]:
        k += step
    return cell(k)
