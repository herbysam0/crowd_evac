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
        cost_penalty: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Solve the integration and flow fields for a walkable grid.

        Args:
            cell_size: Grid cell side length in metres. Must be positive.
            walkable: Boolean grid; ``True`` where a cell is passable.
            exit_cells: Goal cells (sources of the Dijkstra solve). Cells
                outside the grid or on a blocked cell are ignored.
            cost_penalty: Optional per-cell traversal-cost multiplier (a
                "danger field"), shape ``(rows, cols)``, all ``>= 0``. Entering
                a cell costs ``base_step * (1 + cost_penalty[cell])``, so high
                values steer the solve *around* hazardous cells while leaving
                them walkable (no zero-direction trap). ``None`` means no
                penalty (uniform metric cost).

        Raises:
            ValueError: If cell_size is not positive.
            PathfindingError: If no supplied exit cell is walkable.
        """
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size!r}")
        self.cell_size: float = cell_size
        self._walkable: npt.NDArray[np.bool_] = walkable
        self._exit_cells: tuple[Cell, ...] = tuple(exit_cells)
        # Pristine floor mask/exits this field derives from.  ``build`` and
        # ``_rebuild`` overwrite these so re-solves always start from the
        # unblocked floor, letting cleared hazards restore their cells rather
        # than accumulating blocks forever.
        self._base_walkable: npt.NDArray[np.bool_] = walkable
        self._base_exit_cells: tuple[Cell, ...] = self._exit_cells
        self._blocked: frozenset[Cell] = frozenset()
        self._cost_penalty: npt.NDArray[np.float64] | None = cost_penalty
        self.cost, self.direction = _solve(
            walkable, self._exit_cells, cell_size, cost_penalty
        )

    @property
    def blocked(self) -> frozenset[Cell]:
        """Set of hazard-blocked cells currently overlaid on the floor mask.

        Empty for a freshly built field; populated by :meth:`with_blocks` /
        :meth:`recompute`.  Compared by callers to decide whether a re-solve
        is needed when the active-hazard set changes.
        """
        return self._blocked

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

        Adds ``blocked_cells`` to this field's existing :attr:`blocked` set and
        re-solves from the *pristine* floor mask, so the new field carries the
        union of all blocks. Cells that already routed to an unaffected exit
        keep their direction; only the region downstream of the block changes
        (R4.4). To replace the block set entirely (restoring cleared cells),
        use :meth:`with_blocks`.

        Args:
            blocked_cells: ``(row, col)`` cells to additionally mark
                impassable. Out-of-bounds cells are ignored.

        Returns:
            A new :class:`FlowField`; this instance is left unchanged.

        Raises:
            PathfindingError: If blocking removes every walkable exit cell.
        """
        return self._rebuild(
            self._blocked | frozenset(blocked_cells), self._cost_penalty
        )

    def with_blocks(self, blocked_cells: Iterable[Cell]) -> FlowField:
        """Return a field with *exactly* ``blocked_cells`` blocked (absolute).

        Unlike :meth:`recompute` (which accumulates), this replaces the block
        set: it re-solves from the pristine floor mask with only the supplied
        cells impassable, so any previously-blocked cell not in
        ``blocked_cells`` is **restored**. This is what lets a decayed or
        removed hazard free its footprint and let agents recalculate a route.
        The cost penalty is cleared (use :meth:`with_hazards` to set one).

        Args:
            blocked_cells: ``(row, col)`` cells to mark impassable. Out-of-
                bounds cells are ignored.

        Returns:
            A new :class:`FlowField`; this instance is left unchanged.

        Raises:
            PathfindingError: If blocking removes every walkable exit cell.
        """
        return self._rebuild(frozenset(blocked_cells), None)

    def with_hazards(
        self,
        blocked_cells: Iterable[Cell],
        cost_penalty: npt.NDArray[np.float64] | None,
    ) -> FlowField:
        """Return a field re-solved with both a hard block set and a soft cost.

        Replaces *both* the impassable block set (the hazard's physical core)
        and the graded traversal-cost penalty (the danger field over the
        hazard's wider influence radius), re-solving from the pristine floor
        mask. The combination routes agents *around* danger toward the next-best
        exit while keeping every cell walkable, so no agent loses its descent
        direction. Cleared hazards are restored when called with an empty block
        set and ``None`` penalty.

        Args:
            blocked_cells: ``(row, col)`` cells to mark impassable. Out-of-
                bounds cells are ignored.
            cost_penalty: Per-cell traversal-cost multiplier, shape
                ``(rows, cols)`` and ``>= 0``, or ``None`` for no penalty.

        Returns:
            A new :class:`FlowField`; this instance is left unchanged.

        Raises:
            PathfindingError: If blocking removes every walkable exit cell.
        """
        return self._rebuild(frozenset(blocked_cells), cost_penalty)

    def _rebuild(
        self,
        blocked: frozenset[Cell],
        cost_penalty: npt.NDArray[np.float64] | None,
    ) -> FlowField:
        """Re-solve from the pristine floor mask with ``blocked`` + penalty.

        Args:
            blocked: Absolute set of cells to mark impassable; in-bounds cells
                only are applied (out-of-bounds silently ignored).
            cost_penalty: Per-cell traversal-cost multiplier to apply in the
                solve, or ``None``.

        Returns:
            A new :class:`FlowField` carrying the same pristine base, the given
            ``blocked`` set, and the given penalty.

        Raises:
            PathfindingError: If blocking removes every walkable exit cell.
        """
        rows, cols = self._base_walkable.shape
        new_walkable = self._base_walkable.copy()
        in_bounds: set[Cell] = set()
        for r, c in blocked:
            if 0 <= r < rows and 0 <= c < cols:
                new_walkable[r, c] = False
                in_bounds.add((r, c))
        remaining = [
            e for e in self._base_exit_cells if new_walkable[e[0], e[1]]
        ]
        field = FlowField(self.cell_size, new_walkable, remaining, cost_penalty)
        field._base_walkable = self._base_walkable
        field._base_exit_cells = self._base_exit_cells
        field._blocked = frozenset(in_bounds)
        return field

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


def cells_in_radius(
    pos: tuple[float, float],
    radius: float,
    cell_size: float,
    grid_shape: tuple[int, ...],
) -> list[Cell]:
    """Return ``(row, col)`` cells whose centres lie within *radius* of *pos*.

    Uses the flow-field coordinate convention: column index maps to world x,
    row index maps to world y.  Cell ``(r, c)`` has its centre at world
    position ``((c + 0.5) * cell_size, (r + 0.5) * cell_size)``.  Only cells
    strictly inside the grid are returned; out-of-bounds cells are skipped.

    Args:
        pos: World position ``(x, y)`` in metres.
        radius: Search radius in metres.  Non-negative.
        cell_size: Grid cell side length in metres.  Must be positive.
        grid_shape: ``(rows, cols, ...)`` of the grid; only the first two
            dimensions are used.

    Returns:
        List of in-bounds ``(row, col)`` integer tuples whose cell centres
        fall within *radius* of *pos*.  Empty when none qualify.
    """
    x, y = pos
    rows = int(grid_shape[0])
    cols = int(grid_shape[1])
    c0 = int(x / cell_size)
    r0 = int(y / cell_size)
    r_cells = int(math.ceil(radius / cell_size)) + 1

    result: list[Cell] = []
    for dr in range(-r_cells, r_cells + 1):
        for dc in range(-r_cells, r_cells + 1):
            r = r0 + dr
            c = c0 + dc
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            cx = (c + 0.5) * cell_size
            cy = (r + 0.5) * cell_size
            if math.hypot(cx - x, cy - y) <= radius:
                result.append((r, c))
    return result


def build_danger_field(
    grid_shape: tuple[int, ...],
    cell_size: float,
    hazards: Iterable[tuple[float, float, float]],
) -> npt.NDArray[np.float64]:
    """Build a per-cell danger field in ``[0, 1]`` from radial hazards.

    Each hazard ``(x, y, radius)`` contributes a linearly-decaying bump that is
    ``1`` at its centre and ``0`` at (and beyond) its radius; the field is the
    element-wise maximum across hazards (a cell is as dangerous as its worst
    nearby hazard).  The result is intended to be scaled by an avoidance weight
    and passed as ``cost_penalty`` to the flow-field solve, so the danger
    *shape* (steeper toward each core) shapes the route while the weight sets
    its strength.

    Args:
        grid_shape: ``(rows, cols, ...)`` of the grid; only the first two
            dimensions are used.
        cell_size: Grid cell side length in metres. Must be positive.
        hazards: Iterable of ``(x, y, radius)`` world-space hazards. Hazards
            with non-positive radius are skipped.

    Returns:
        Float64 array of shape ``(rows, cols)`` with values in ``[0, 1]``;
        all-zero when no hazard has a positive radius.
    """
    rows = int(grid_shape[0])
    cols = int(grid_shape[1])
    danger: npt.NDArray[np.float64] = np.zeros((rows, cols), dtype=np.float64)
    cx = (np.arange(cols, dtype=np.float64) + 0.5) * cell_size
    cy = (np.arange(rows, dtype=np.float64) + 0.5) * cell_size
    xx, yy = np.meshgrid(cx, cy)
    for hx, hy, hr in hazards:
        if hr <= 0.0:
            continue
        dist = np.hypot(xx - hx, yy - hy)
        contrib = np.clip(1.0 - dist / hr, 0.0, 1.0)
        danger = np.maximum(danger, contrib)
    return danger


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _solve(
    walkable: npt.NDArray[np.bool_],
    exit_cells: tuple[Cell, ...],
    cell_size: float,
    cost_penalty: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the integration (cost) and flow (direction) fields.

    Args:
        walkable: Boolean passability grid.
        exit_cells: Goal cells (Dijkstra sources).
        cell_size: Cell side length in metres.
        cost_penalty: Optional per-cell traversal-cost multiplier; see
            :meth:`FlowField.__init__`.

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
    _dijkstra(walkable, cost, sources, cell_size, cost_penalty)
    direction = _build_directions(walkable, cost)
    return cost, direction


def _dijkstra(
    walkable: npt.NDArray[np.bool_],
    cost: npt.NDArray[np.float64],
    sources: list[Cell],
    cell_size: float,
    cost_penalty: npt.NDArray[np.float64] | None = None,
) -> None:
    """Fill ``cost`` in place with least-cost distance to the nearest source.

    Multi-source 8-connected Dijkstra. Diagonal moves are skipped when either
    shared orthogonal neighbour is blocked (no corner cutting). When
    ``cost_penalty`` is supplied, entering cell ``n`` costs
    ``base_step * (1 + cost_penalty[n])``, so the solved field routes around
    high-penalty (hazardous) cells while keeping them traversable.
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
            if cost_penalty is not None:
                step *= 1.0 + float(cost_penalty[nr, nc])
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
