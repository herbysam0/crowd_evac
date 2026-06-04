"""Static collision map enforcing that agents never cross walls/obstacles.

The navigation flow field (FR-4) *routes* agents around obstacles, but the
additive force terms (crowd repulsion, herd, panic) plus integrator inertia
can still push an agent's integrated position into a wall or obstacle cell.
This module adds the missing hard constraint: after each integration step the
:class:`CollisionMap` rejects any move whose destination lands in a blocked
cell, so obstacles are *never* crossed.

The blocking grid is the static complement of
:meth:`~crowd_evac.domain.floor_plan.FloorPlan.walkable_mask` — walls and
obstacles are impassable, exit openings remain passable. It is built once per
floor and never changes; the fire-blocked overlay used for re-routing
(:meth:`~crowd_evac.pathfinding.flow_field.FlowField.recompute`) is
deliberately *not* applied here, because a hazard repels agents (via the panic
force) rather than physically walling them in.

Resolution is **axis-separated sliding**, the standard grid-collision rule:
try the full move; if blocked, try moving on the x-axis only, then the y-axis
only; if all are blocked, the agent stays put. The velocity component into a
blocked axis is zeroed so the agent does not accumulate momentum against a
wall. Because the maximum per-tick displacement
(``MAX_SPEED * PANIC_SPEED_MULTIPLIER * DT``) is smaller than one grid cell and
every obstacle rasterises to at least one cell thick, an agent can enter at
most an adjacent cell each tick — so checking the destination cell alone is
sufficient and no obstacle can be tunnelled through.

Coordinate convention matches :class:`FloorPlan`: cell ``[r, c]`` covers
``[c*cell_size, (c+1)*cell_size) x [r*cell_size, (r+1)*cell_size)`` with origin
at the room's bottom-left corner, so column maps to world x and row to world y.
Pure NumPy — no engine or I/O imports.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.agent_state import AgentState, Bool1D, Vec2Array
from crowd_evac.domain.constants import GRID_CELL_SIZE
from crowd_evac.domain.floor_plan import FloorPlan


class CollisionMap:
    """Static blocking grid that prevents agents from entering walls/obstacles.

    A map is normally created via :meth:`from_floor_plan`. The blocking grid is
    immutable after construction; resolution reads it to clamp agent moves but
    never alters it.

    Attributes:
        cell_size: Grid cell side length in metres (same resolution as the
            rasterised floor plan and flow field).
    """

    def __init__(
        self,
        blocked: npt.NDArray[np.bool_],
        cell_size: float,
    ) -> None:
        """Wrap a pre-computed blocking grid.

        Args:
            blocked: Boolean grid of shape ``(rows, cols)``; ``True`` where a
                cell is impassable (wall or obstacle).
            cell_size: Grid cell side length in metres. Must be positive.

        Raises:
            ValueError: If ``cell_size`` is not positive or ``blocked`` is not
                a 2-D array.
        """
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size!r}")
        grid = np.asarray(blocked, dtype=np.bool_)
        if grid.ndim != 2:
            raise ValueError(
                f"blocked must be a 2-D grid, got shape {grid.shape!r}"
            )
        self.cell_size: float = cell_size
        self._blocked: npt.NDArray[np.bool_] = grid
        self._rows: int = int(grid.shape[0])
        self._cols: int = int(grid.shape[1])

    @classmethod
    def from_floor_plan(
        cls, floor_plan: FloorPlan, cell_size: float = GRID_CELL_SIZE
    ) -> CollisionMap:
        """Build a collision map from a floor plan's static geometry.

        The blocking grid is the complement of the floor plan's walkable mask,
        so walls and obstacles are impassable while exit openings stay
        passable.

        Args:
            floor_plan: Source geometry; its walkable mask is rasterised at the
                given resolution.
            cell_size: Grid cell side length in metres. Must be positive. Use
                the same value as the flow field so routing and collision
                agree.

        Returns:
            A :class:`CollisionMap` for the floor plan.

        Raises:
            ValueError: If ``cell_size`` is not positive.
        """
        blocked: npt.NDArray[np.bool_] = ~floor_plan.walkable_mask(cell_size)
        return cls(blocked, cell_size)

    @property
    def blocked(self) -> npt.NDArray[np.bool_]:
        """Read-only view of the blocking grid, shape ``(rows, cols)``."""
        view = self._blocked.view()
        view.flags.writeable = False
        return view

    def is_blocked(self, points: Vec2Array) -> Bool1D:
        """Return whether each world point lies in a blocked cell.

        Points outside the grid bounding box are reported as blocked, so the
        room boundary cannot be crossed even where no explicit wall is present.

        Args:
            points: World coordinates of shape ``(N, 2)`` as ``(x, y)``.

        Returns:
            Boolean array of shape ``(N,)``; ``True`` where the point falls in
            a blocked or out-of-bounds cell.

        Raises:
            ValueError: If ``points`` is not a 2-D array with two columns.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"points must have shape (N, 2), got {pts.shape!r}"
            )
        c = np.floor(pts[:, 0] / self.cell_size).astype(np.intp)
        r = np.floor(pts[:, 1] / self.cell_size).astype(np.intp)
        in_bounds: Bool1D = (
            (r >= 0) & (r < self._rows) & (c >= 0) & (c < self._cols)
        )
        # Out-of-bounds defaults to blocked; in-bounds reads the grid.
        result: Bool1D = np.ones(pts.shape[0], dtype=np.bool_)
        result[in_bounds] = self._blocked[r[in_bounds], c[in_bounds]]
        return result

    def resolve(self, state: AgentState, prev_pos: Vec2Array) -> None:
        """Clamp active agents' moves so no agent enters a blocked cell.

        Applies axis-separated sliding between each active agent's previous
        position and its freshly integrated position, writing the corrected
        position back into ``state.pos`` and zeroing any velocity component
        whose axis was blocked. Dead agents are left untouched.

        Args:
            state: Agent state whose positions and velocities are corrected in
                place. Positions must already be integrated for this tick.
            prev_pos: Per-agent positions *before* this tick's integration,
                shape ``(state.count, 2)``. Every active agent's previous
                position is assumed walkable (the invariant this method
                maintains from spawn onward).

        Raises:
            ValueError: If ``prev_pos`` does not have shape ``(state.count, 2)``.
        """
        prev = np.asarray(prev_pos, dtype=np.float64)
        if prev.shape != (state.count, 2):
            raise ValueError(
                f"prev_pos must have shape ({state.count}, 2), "
                f"got {prev.shape!r}"
            )
        active = state.active_indices
        if active.size == 0:
            return

        new_pos, moved_x, moved_y = self._slide(prev[active], state.pos[active])
        state.pos[active] = new_pos

        vel = state.vel[active]
        vel[~moved_x, 0] = 0.0
        vel[~moved_y, 1] = 0.0
        state.vel[active] = vel

    def _slide(
        self, p0: Vec2Array, p1: Vec2Array
    ) -> tuple[Vec2Array, Bool1D, Bool1D]:
        """Axis-separated slide from ``p0`` to ``p1`` against the blocking grid.

        Args:
            p0: Previous (walkable) positions, shape ``(A, 2)``.
            p1: Proposed integrated positions, shape ``(A, 2)``.

        Returns:
            A ``(new_pos, moved_x, moved_y)`` tuple where ``new_pos`` is the
            corrected ``(A, 2)`` position array and the boolean masks flag,
            per agent, whether the x- and y-axis moves were accepted.
        """
        x0, y0 = p0[:, 0], p0[:, 1]
        x1, y1 = p1[:, 0], p1[:, 1]
        full_ok: Bool1D = ~self.is_blocked(p1)
        x_ok: Bool1D = ~self.is_blocked(np.column_stack((x1, y0)))
        y_ok: Bool1D = ~self.is_blocked(np.column_stack((x0, y1)))

        # Full move wins; else slide on x; else slide on y; else stay put.
        moved_x: Bool1D = full_ok | x_ok
        moved_y: Bool1D = full_ok | (~x_ok & y_ok)

        new_pos: Vec2Array = np.empty_like(p1)
        new_pos[:, 0] = np.where(moved_x, x1, x0)
        new_pos[:, 1] = np.where(moved_y, y1, y0)
        return new_pos, moved_x, moved_y
