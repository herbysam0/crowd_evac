"""Grid flow field for multi-exit, re-routable evacuation navigation (FR-4).

Rasterizes a :class:`~crowd_evac.domain.floor_plan.FloorPlan` to a uniform
grid and solves a multi-source Dijkstra from every exit cell. The result is,
per walkable cell, the metric cost to the nearest reachable exit (the
*integration field*) and a unit direction vector pointing one step down the
cost gradient (the *flow field*). Agents sample the flow field by bilinear
interpolation at their continuous position to obtain an exit-seeking
direction (`f_exit`).

The solve is decoupled from stepping: a field is built once per floor and
cached. :meth:`FlowField.recompute` overlays freshly blocked cells (e.g. a
fire blocking an exit) onto the cached walkable mask and re-solves, so
affected agents re-route toward the next-best exit (R4.3 / R4.4).

Movement is 8-connected; diagonal steps are forbidden when either orthogonal
neighbour is blocked, which prevents a followed path from clipping a wall
corner. Consequently, following the flow field from any walkable cell
strictly decreases cost each step and terminates at an exit without crossing
a wall (R4.1 / R4.2).

Coordinate convention matches :class:`FloorPlan`: grid cell ``[r, c]`` covers
``[c*cell_size, (c+1)*cell_size) x [r*cell_size, (r+1)*cell_size)`` with
origin at the room's bottom-left corner, so column maps to the world x-axis
and row maps to the world y-axis. Direction vectors are ``(dx, dy)`` in that
world frame. Pure NumPy — no engine or I/O imports.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.constants import GRID_CELL_SIZE
from crowd_evac.domain.errors import PathfindingError
from crowd_evac.domain.floor_plan import FloorPlan

# 8-connected neighbour offsets as (delta_row, delta_col).
_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)

# Below this Euclidean norm an interpolated direction is treated as
# undefined (arrived at exit, or opposing cells cancel) and returned as zero.
_SAMPLE_EPS: float = 1e-9

Cell = tuple[int, int]


class FlowField:
    """Cached navigation field directing agents toward the nearest exit.

    A field is normally created via :meth:`build` from a floor plan. The
    public arrays are read-only outputs of the solve; mutate the field only
    through :meth:`recompute`, which returns a fresh instance.

    Attributes:
        cell_size: Grid cell side length in metres (same as the rasterised
            floor plan).
        cost: Float64 array of shape ``(rows, cols)``; metric distance to the
            nearest reachable exit, ``inf`` for blocked or unreachable cells.
        direction: Float64 array of shape ``(rows, cols, 2)``; per-cell unit
            ``(dx, dy)`` vector toward the lowest-cost neighbour. Exit cells
            and unreachable cells hold ``(0, 0)``.
    """

    def __init__(
        self,
        cell_size: float,
        walkable: npt.NDArray[np.bool_],
        exit_cells: Iterable[Cell],
    ) -> None:
        """Solve the integration and flow fields for a walkable grid.

        Args:
            cell_size: Grid cell side length in metres. Must be positive.
            walkable: Boolean grid; ``True`` where a cell is passable.
            exit_cells: Goal cells (sources of the Dijkstra solve). Cells
                outside the grid or on a blocked cell are ignored.

        Raises:
            ValueError: If cell_size is not positive.
            PathfindingError: If no supplied exit cell is walkable.
        """
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size!r}")
        self.cell_size: float = cell_size
        self._walkable: npt.NDArray[np.bool_] = walkable
        self._exit_cells: tuple[Cell, ...] = tuple(exit_cells)
        self.cost, self.direction = _solve(
            walkable, self._exit_cells, cell_size
        )

    @classmethod
    def build(
        cls, floor_plan: FloorPlan, cell_size: float = GRID_CELL_SIZE
    ) -> FlowField:
        """Build a flow field by rasterising a floor plan.

        Args:
            floor_plan: Source geometry; its walkable mask and exit cells are
                computed at the given resolution.
            cell_size: Grid cell side length in metres. Must be positive.

        Returns:
            A solved :class:`FlowField` for the floor plan.

        Raises:
            ValueError: If cell_size is not positive.
            PathfindingError: If the floor plan has no walkable exit cell.
        """
        walkable = floor_plan.walkable_mask(cell_size)
        exit_cells = floor_plan.exit_cells(cell_size)
        return cls(cell_size, walkable, exit_cells)

    def recompute(self, blocked_cells: Iterable[Cell]) -> FlowField:
        """Return a re-solved field with extra cells blocked (re-routing).

        Reuses this field's cached walkable mask and exit list — the floor
        geometry is not re-rasterised — overlays ``blocked_cells`` as
        impassable, drops any exit that falls on a newly blocked cell, and
        re-solves. Cells that already routed to an unaffected exit keep their
        direction; only the region downstream of the block changes (R4.4).

        Args:
            blocked_cells: ``(row, col)`` cells to mark impassable. Out-of-
                bounds cells are ignored.

        Returns:
            A new :class:`FlowField`; this instance is left unchanged.

        Raises:
            PathfindingError: If blocking removes every walkable exit cell.
        """
        rows, cols = self._walkable.shape
        new_walkable = self._walkable.copy()
        for r, c in blocked_cells:
            if 0 <= r < rows and 0 <= c < cols:
                new_walkable[r, c] = False
        remaining = [e for e in self._exit_cells if new_walkable[e[0], e[1]]]
        return FlowField(self.cell_size, new_walkable, remaining)

    def sample(
        self, positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Bilinearly sample exit-seeking unit directions at world positions.

        Interpolates the four cell-direction vectors surrounding each
        position (cell centres at ``(c+0.5, r+0.5)*cell_size``) and
        re-normalises to unit length. Positions are clamped to the grid, so
        out-of-bounds queries return the nearest in-grid direction. A near-
        zero interpolant (an agent on an exit, or cancelling neighbours)
        yields a zero vector.

        Args:
            positions: World coordinates of shape ``(N, 2)`` as ``(x, y)``.

        Returns:
            Float64 array of shape ``(N, 2)``; unit ``(dx, dy)`` directions,
            or ``(0, 0)`` where the direction is undefined.

        Raises:
            ValueError: If positions is not a 2-D array with two columns.
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(
                f"positions must have shape (N, 2), got {pos.shape!r}"
            )
        rows, cols = self.cost.shape
        cs = self.cell_size
        fc = np.clip(pos[:, 0] / cs - 0.5, 0.0, cols - 1)
        fr = np.clip(pos[:, 1] / cs - 0.5, 0.0, rows - 1)
        c0 = np.floor(fc).astype(np.intp)
        r0 = np.floor(fr).astype(np.intp)
        c1 = np.minimum(c0 + 1, cols - 1)
        r1 = np.minimum(r0 + 1, rows - 1)
        tc = (fc - c0)[:, None]
        tr = (fr - r0)[:, None]
        d = self.direction
        top = d[r0, c0] * (1.0 - tc) + d[r0, c1] * tc
        bot = d[r1, c0] * (1.0 - tc) + d[r1, c1] * tc
        out: npt.NDArray[np.float64] = top * (1.0 - tr) + bot * tr
        return _normalise_rows(out)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _solve(
    walkable: npt.NDArray[np.bool_],
    exit_cells: tuple[Cell, ...],
    cell_size: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the integration (cost) and flow (direction) fields.

    Raises:
        PathfindingError: If no exit cell is in-bounds and walkable.
    """
    rows, cols = walkable.shape
    cost: npt.NDArray[np.float64] = np.full((rows, cols), np.inf, np.float64)
    sources = [
        (r, c)
        for r, c in exit_cells
        if 0 <= r < rows and 0 <= c < cols and walkable[r, c]
    ]
    if not sources:
        raise PathfindingError(
            "flow field has no walkable exit cell to route toward"
        )
    _dijkstra(walkable, cost, sources, cell_size)
    direction = _build_directions(walkable, cost)
    return cost, direction


def _dijkstra(
    walkable: npt.NDArray[np.bool_],
    cost: npt.NDArray[np.float64],
    sources: list[Cell],
    cell_size: float,
) -> None:
    """Fill ``cost`` in place with metric distance to the nearest source.

    Multi-source 8-connected Dijkstra. Diagonal moves are skipped when either
    shared orthogonal neighbour is blocked (no corner cutting).
    """
    rows, cols = walkable.shape
    heap: list[tuple[float, int, int]] = []
    for r, c in sources:
        cost[r, c] = 0.0
        heapq.heappush(heap, (0.0, r, c))
    while heap:
        dist, r, c = heapq.heappop(heap)
        if dist > cost[r, c]:
            continue
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not walkable[nr, nc]:
                continue
            if dr != 0 and dc != 0 and not (walkable[r, nc] and walkable[nr, c]):
                continue
            step = math.hypot(dr, dc) * cell_size
            if dist + step < cost[nr, nc]:
                cost[nr, nc] = dist + step
                heapq.heappush(heap, (dist + step, nr, nc))


def _build_directions(
    walkable: npt.NDArray[np.bool_],
    cost: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Derive per-cell unit directions toward each cell's cheapest neighbour.

    Vectorised: for every neighbour offset, compare the (corner-cut-aware)
    neighbour cost against the running best. Cells whose best neighbour is
    strictly cheaper than themselves point toward it; exit cells and
    unreachable cells keep ``(0, 0)``.
    """
    rows, cols = cost.shape
    pad_cost = np.full((rows + 2, cols + 2), np.inf, np.float64)
    pad_cost[1:-1, 1:-1] = cost
    pad_walk = np.zeros((rows + 2, cols + 2), np.bool_)
    pad_walk[1:-1, 1:-1] = walkable

    best = np.full((rows, cols), np.inf, np.float64)
    best_dr = np.zeros((rows, cols), np.float64)
    best_dc = np.zeros((rows, cols), np.float64)
    for dr, dc in _NEIGHBOR_OFFSETS:
        neigh = _neighbor_cost(pad_cost, pad_walk, dr, dc, rows, cols)
        better = neigh < best
        best = np.where(better, neigh, best)
        best_dr = np.where(better, float(dr), best_dr)
        best_dc = np.where(better, float(dc), best_dc)

    improving = best < cost
    norm = np.hypot(best_dc, best_dr)
    safe = improving & (norm > 0.0)
    direction = np.zeros((rows, cols, 2), np.float64)
    direction[..., 0] = np.where(safe, best_dc / np.where(safe, norm, 1.0), 0.0)
    direction[..., 1] = np.where(safe, best_dr / np.where(safe, norm, 1.0), 0.0)
    return direction


def _neighbor_cost(
    pad_cost: npt.NDArray[np.float64],
    pad_walk: npt.NDArray[np.bool_],
    dr: int,
    dc: int,
    rows: int,
    cols: int,
) -> npt.NDArray[np.float64]:
    """Return the cost of neighbour ``(dr, dc)`` for each cell, else ``inf``.

    Reads from inf/False-padded arrays so edge cells see blocked neighbours.
    A diagonal neighbour is invalidated when either shared orthogonal cell is
    blocked, mirroring the corner-cut rule used in the cost solve.
    """
    neigh = pad_cost[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    valid = pad_walk[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    if dr != 0 and dc != 0:
        side_row = pad_walk[1 + dr:1 + dr + rows, 1:1 + cols]
        side_col = pad_walk[1:1 + rows, 1 + dc:1 + dc + cols]
        valid = valid & side_row & side_col
    result: npt.NDArray[np.float64] = np.where(valid, neigh, np.inf)
    return result


def _normalise_rows(
    vectors: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return row-wise unit vectors; rows below ``_SAMPLE_EPS`` become zero."""
    norm = np.linalg.norm(vectors, axis=1)
    nonzero = norm > _SAMPLE_EPS
    out: npt.NDArray[np.float64] = np.zeros_like(vectors)
    scale = np.where(nonzero, norm, 1.0)[:, None]
    out = np.where(nonzero[:, None], vectors / scale, 0.0)
    return out
