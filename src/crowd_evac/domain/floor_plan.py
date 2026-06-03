"""Floor plan domain model: walls, obstacles, walkable region, and exits.

Provides immutable value types describing the physical layout of a single
floor. All coordinates and dimensions are in metres; origin is the
bottom-left corner of the room bounding box.

No framework or I/O imports — pure Python + NumPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.errors import ScenarioValidationError


class ExitSide(str, Enum):
    """Cardinal wall on which an exit opening is positioned."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass(frozen=True)
class Rect:
    """Axis-aligned bounding box in metres.

    Attributes:
        x: Left edge position in metres.
        y: Bottom edge position in metres.
        width: Rectangle width in metres. Must be positive.
        height: Rectangle height in metres. Must be positive.
    """

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Wall(Rect):
    """Rectangular wall segment that blocks agent movement.

    Attributes:
        label: Optional descriptive name for debugging or rendering.
    """

    label: str = ""


@dataclass(frozen=True)
class Obstacle(Rect):
    """Interior rectangular obstacle (seats, pillars, furniture).

    Attributes:
        label: Optional descriptive name for debugging or rendering.
    """

    label: str = ""


@dataclass(frozen=True)
class Exit:
    """Doorway opening through which agents evacuate.

    The opening is centred at ``(x, y)`` and spans ``width_m`` along the
    wall indicated by ``side``.

    Attributes:
        x: Centre x-coordinate of the opening in metres.
        y: Centre y-coordinate of the opening in metres.
        width_m: Opening width in metres along the wall. Must be positive.
        side: Which boundary wall contains this exit.
        capacity_per_second: Maximum throughput in agents per second.
        label: Optional descriptive name.
    """

    x: float
    y: float
    width_m: float
    side: ExitSide
    capacity_per_second: int
    label: str = ""


@dataclass(frozen=True)
class FloorPlan:
    """Physical layout of a single-level floor.

    The walkable region is the room bounding box minus walls/obstacles,
    with exit openings restored as passable. All coordinates are in metres
    with origin at the bottom-left corner.

    Attributes:
        width_m: Room width in metres (x-axis extent). Must be positive.
        height_m: Room height in metres (y-axis extent). Must be positive.
        walls: Immutable sequence of rectangular wall segments.
        obstacles: Immutable sequence of interior obstacles.
        exits: Immutable sequence of evacuation exits.

    Raises:
        ScenarioValidationError: On construction if any geometry is invalid.
    """

    width_m: float
    height_m: float
    walls: tuple[Wall, ...]
    obstacles: tuple[Obstacle, ...]
    exits: tuple[Exit, ...]

    def __post_init__(self) -> None:
        """Validate all geometry on construction."""
        _validate(self)

    def walkable_mask(self, cell_size: float) -> npt.NDArray[np.bool_]:
        """Compute a boolean walkability grid for this floor plan.

        Returns a 2-D boolean array of shape ``(rows, cols)`` where
        ``True`` means a cell is passable. Grid origin ``[0, 0]`` is at
        the room's bottom-left corner. Cell ``[r, c]`` covers
        ``[c*cell_size, (c+1)*cell_size) × [r*cell_size, (r+1)*cell_size)``.

        Exit openings override walls: cells within an exit bounding box are
        always walkable regardless of wall placement.

        Args:
            cell_size: Side length of each grid cell in metres. Must be > 0.

        Returns:
            Boolean ndarray of shape
            ``(ceil(height_m/cell_size), ceil(width_m/cell_size))``.

        Raises:
            ValueError: If cell_size is not positive.
        """
        if cell_size <= 0:
            raise ValueError(
                f"cell_size must be positive, got {cell_size!r}"
            )
        rows = math.ceil(self.height_m / cell_size)
        cols = math.ceil(self.width_m / cell_size)
        mask: npt.NDArray[np.bool_] = np.ones((rows, cols), dtype=np.bool_)

        for rect in (*self.walls, *self.obstacles):
            r0, r1, c0, c1 = _rect_cell_range(
                rect.x, rect.y, rect.width, rect.height, cell_size, rows, cols
            )
            mask[r0:r1, c0:c1] = False

        for exit_ in self.exits:
            bx, by, bw, bh = _exit_bbox(exit_, cell_size)
            r0, r1, c0, c1 = _rect_cell_range(
                bx, by, bw, bh, cell_size, rows, cols
            )
            mask[r0:r1, c0:c1] = True

        return mask

    def exit_cells(self, cell_size: float) -> list[tuple[int, int]]:
        """Return grid cell indices for all exit openings.

        These are the goal cells for flow-field pathfinding (step 1.5).
        Each returned cell lies within an exit opening bounding box and is
        guaranteed walkable in the grid from :meth:`walkable_mask`.

        Args:
            cell_size: Side length of each grid cell in metres. Must be > 0.

        Returns:
            List of ``(row, col)`` integer tuples; one entry per exit cell.
            May contain duplicates if exits overlap.

        Raises:
            ValueError: If cell_size is not positive.
        """
        if cell_size <= 0:
            raise ValueError(
                f"cell_size must be positive, got {cell_size!r}"
            )
        rows = math.ceil(self.height_m / cell_size)
        cols = math.ceil(self.width_m / cell_size)
        cells: list[tuple[int, int]] = []
        for exit_ in self.exits:
            bx, by, bw, bh = _exit_bbox(exit_, cell_size)
            r0, r1, c0, c1 = _rect_cell_range(
                bx, by, bw, bh, cell_size, rows, cols
            )
            for r in range(r0, r1):
                for c in range(c0, c1):
                    cells.append((r, c))
        return cells


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _rect_cell_range(
    x: float,
    y: float,
    width: float,
    height: float,
    cell_size: float,
    rows: int,
    cols: int,
) -> tuple[int, int, int, int]:
    """Return (r0, r1, c0, c1) cell-index range for a rectangle, clamped."""
    c0 = max(0, math.floor(x / cell_size))
    c1 = min(cols, math.ceil((x + width) / cell_size))
    r0 = max(0, math.floor(y / cell_size))
    r1 = min(rows, math.ceil((y + height) / cell_size))
    return r0, r1, c0, c1


def _exit_bbox(exit_: Exit, cell_size: float) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, width, height) bounding box of an exit opening.

    The opening spans ``exit_.width_m`` along the wall's axis; the depth
    perpendicular to the wall equals ``cell_size`` (one cell), centred on
    the exit position.
    """
    half = exit_.width_m / 2.0
    depth = cell_size
    if exit_.side in (ExitSide.SOUTH, ExitSide.NORTH):
        return (exit_.x - half, exit_.y - depth / 2.0, exit_.width_m, depth)
    # EAST or WEST: opening is along the y-axis
    return (exit_.x - depth / 2.0, exit_.y - half, depth, exit_.width_m)


def _validate(fp: FloorPlan) -> None:
    """Validate all FloorPlan geometry.

    Raises:
        ScenarioValidationError: If any constraint is violated.
    """
    if fp.width_m <= 0:
        raise ScenarioValidationError(
            f"FloorPlan width_m must be positive, got {fp.width_m!r}"
        )
    if fp.height_m <= 0:
        raise ScenarioValidationError(
            f"FloorPlan height_m must be positive, got {fp.height_m!r}"
        )
    if not fp.exits:
        raise ScenarioValidationError(
            "FloorPlan must have at least one exit"
        )
    for wall in fp.walls:
        _validate_rect(wall.width, wall.height, "Wall", wall.label)
    for obs in fp.obstacles:
        _validate_rect(obs.width, obs.height, "Obstacle", obs.label)
    for exit_ in fp.exits:
        if exit_.width_m <= 0:
            raise ScenarioValidationError(
                f"Exit '{exit_.label}' width_m must be positive, "
                f"got {exit_.width_m!r}"
            )
        if exit_.capacity_per_second <= 0:
            raise ScenarioValidationError(
                f"Exit '{exit_.label}' capacity_per_second must be a positive "
                f"integer, got {exit_.capacity_per_second!r}"
            )


def _validate_rect(width: float, height: float, kind: str, label: str) -> None:
    """Validate that a rectangle has positive dimensions."""
    if width <= 0:
        raise ScenarioValidationError(
            f"{kind} '{label}' width must be positive, got {width!r}"
        )
    if height <= 0:
        raise ScenarioValidationError(
            f"{kind} '{label}' height must be positive, got {height!r}"
        )
