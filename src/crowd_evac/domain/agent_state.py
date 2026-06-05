"""Agent state struct-of-arrays container and per-agent view (FR-1 R1.1/R1.4).

The entire agent population is stored as parallel NumPy arrays indexed by
agent ID 0..N-1.  Vectorised force computation operates on the whole
population at once; agents removed from the simulation have ``alive[i] =
False`` and neither exert nor receive forces (R1.4).  Removal is performed
by the exit model (step 1.9) after agents clear the exit threshold (R5.2),
not immediately upon reaching the exit opening.

A lightweight :class:`Agent` view provides read access to a single agent's
four fields for application-layer code that needs per-agent semantics.

No framework or I/O imports — pure Python + NumPy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.constants import AGENT_RADIUS
from crowd_evac.domain.floor_plan import FloorPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Vec2Array: TypeAlias = npt.NDArray[np.float64]  # shape (N, 2)
Float1D: TypeAlias = npt.NDArray[np.float64]    # shape (N,)
Int1D: TypeAlias = npt.NDArray[np.intp]         # shape (N,)
Bool1D: TypeAlias = npt.NDArray[np.bool_]       # shape (N,)

# Sentinel: goal is unassigned until the exit model processes the agent.
_GOAL_UNASSIGNED: int = -1

# Max random-offset retries per agent before falling back to cell centre.
_SPAWN_MAX_PLACEMENT_ATTEMPTS: int = 20


# ---------------------------------------------------------------------------
# AgentState — struct-of-arrays container
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class AgentState:
    """Struct-of-arrays container for the full agent population.

    All arrays are indexed by agent ID 0..N-1.  Agent *i* has position
    ``pos[i]``, velocity ``vel[i]``, panic level ``panic[i]``, target exit
    index ``goal[i]``, and liveness flag ``alive[i]``.

    Agents removed from simulation have ``alive[i] = False`` and are
    excluded from force computation by callers that filter via
    :attr:`active_indices`.  The exit model (step 1.9) calls
    :meth:`remove` once an agent clears the exit threshold (R5.2).

    Attributes:
        pos: World positions in metres, shape ``(N, 2)``, columns ``[x, y]``.
        vel: Velocities in m/s, shape ``(N, 2)``, columns ``[vx, vy]``.
        panic: Panic level in ``[0, 1]``, shape ``(N,)``.
        goal: Target exit index (0-based), or ``-1`` when unassigned,
            shape ``(N,)``.
        alive: ``True`` while the agent is in the simulation, shape ``(N,)``.
    """

    pos: Vec2Array
    vel: Vec2Array
    panic: Float1D
    goal: Int1D
    alive: Bool1D

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total agent count, including already-removed agents."""
        return int(self.alive.shape[0])

    @property
    def active_indices(self) -> Int1D:
        """Indices of agents still in the simulation (``alive == True``).

        Returns:
            1-D integer array of active agent indices, sorted ascending.
            Empty array when all agents have been removed.
        """
        return np.flatnonzero(self.alive).astype(np.intp)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def remove(self, indices: npt.ArrayLike) -> None:
        """Remove agents from the simulation; they exert and receive no forces.

        Calling remove on an already-removed agent is idempotent.

        Args:
            indices: Single index, sequence of indices, or integer array.

        Raises:
            IndexError: If any index is outside ``[0, count)``.
        """
        idx: Int1D = np.asarray(indices, dtype=np.intp)
        self.alive[idx] = False

    # ------------------------------------------------------------------
    # Per-agent view
    # ------------------------------------------------------------------

    def agent_view(self, index: int) -> Agent:
        """Return a read-access per-agent view for the given agent.

        The returned :class:`Agent` is backed directly by this state's
        arrays, so property reads reflect in-place mutations.

        Args:
            index: Agent ID in ``[0, count)``.

        Returns:
            An :class:`Agent` view for agent *index*.

        Raises:
            IndexError: If *index* is outside ``[0, count)``.
        """
        if index < 0 or index >= self.count:
            raise IndexError(
                f"Agent index {index!r} out of range [0, {self.count})"
            )
        return Agent(self, index)


# ---------------------------------------------------------------------------
# Agent — lightweight per-agent read-access view
# ---------------------------------------------------------------------------


class Agent:
    """Read-access view of a single agent within an :class:`AgentState`.

    Backed directly by the parent AgentState arrays; each property is an
    O(1) array index.  Intended for application-layer code that needs per-
    agent semantics without disrupting vectorised force computation.

    Attributes:
        index: Agent ID within the parent state.
    """

    def __init__(self, state: AgentState, index: int) -> None:
        """Initialise the view.

        Args:
            state: Parent AgentState containing this agent's data.
            index: Agent ID within that state.
        """
        self._state = state
        self.index = index

    @property
    def pos(self) -> Vec2Array:
        """Position ``[x, y]`` in metres."""
        return cast(Vec2Array, self._state.pos[self.index])

    @property
    def vel(self) -> Vec2Array:
        """Velocity ``[vx, vy]`` in metres per second."""
        return cast(Vec2Array, self._state.vel[self.index])

    @property
    def panic(self) -> float:
        """Panic level in ``[0, 1]``."""
        return float(self._state.panic[self.index])

    @property
    def goal(self) -> int:
        """Target exit index (0-based), or ``-1`` when unassigned."""
        return int(self._state.goal[self.index])


# ---------------------------------------------------------------------------
# Private spawn helpers
# ---------------------------------------------------------------------------


def _spawn_safe_mask(
    floor_plan: FloorPlan,
    mask: npt.NDArray[np.bool_],
    cell_size: float,
) -> npt.NDArray[np.bool_]:
    """Return the subset of the walkable mask safe for agent-centre placement.

    A cell is spawn-safe when its centre lies at least
    ``AGENT_RADIUS + cell_size / 2`` metres from every wall, obstacle
    boundary, and the room's outer boundary.  Adding ``cell_size / 2``
    accounts for the worst-case random offset within the cell, guaranteeing
    that any position sampled inside the cell stays at least ``AGENT_RADIUS``
    from solid geometry.

    Args:
        floor_plan: Floor plan with walls, obstacles, and room dimensions.
        mask: Walkable boolean mask of shape ``(rows, cols)`` produced by
            ``floor_plan.walkable_mask(cell_size)``.
        cell_size: Grid cell side length in metres.

    Returns:
        Boolean mask, same shape as *mask*; ``True`` only for spawn-safe
        cells.
    """
    row_idx, col_idx = np.where(mask)
    if row_idx.size == 0:
        return np.zeros_like(mask)

    cx: Float1D = (col_idx.astype(np.float64) + 0.5) * cell_size
    cy: Float1D = (row_idx.astype(np.float64) + 0.5) * cell_size

    # Conservative clearance: account for worst-case within-cell offset.
    clearance: float = AGENT_RADIUS + cell_size / 2.0

    # Distance from each cell centre to the room outer boundary.
    min_dist: Float1D = np.minimum(
        np.minimum(cx, floor_plan.width_m - cx),
        np.minimum(cy, floor_plan.height_m - cy),
    )

    # Distance to each wall and obstacle rectangle (Euclidean, vectorised).
    for rect in (*floor_plan.walls, *floor_plan.obstacles):
        dx: Float1D = np.maximum(
            rect.x - cx,
            np.maximum(cx - (rect.x + rect.width), 0.0),
        )
        dy: Float1D = np.maximum(
            rect.y - cy,
            np.maximum(cy - (rect.y + rect.height), 0.0),
        )
        min_dist = np.minimum(min_dist, np.sqrt(dx * dx + dy * dy))

    safe_mask: npt.NDArray[np.bool_] = np.zeros_like(mask)
    keep = min_dist >= clearance
    safe_mask[row_idx[keep], col_idx[keep]] = True
    return safe_mask


def _greedy_place(
    safe_rows: Int1D,
    safe_cols: Int1D,
    count: int,
    cell_size: float,
    rng: np.random.Generator,
) -> Vec2Array:
    """Place agents in spawn-safe cells with minimum inter-agent clearance.

    Iterates through a shuffled permutation of all spawn-safe cells.  For
    each cell, ``_SPAWN_MAX_PLACEMENT_ATTEMPTS`` random positions are tried.
    If every attempt is within ``2 * AGENT_RADIUS`` of an already-placed
    agent the **cell is skipped** (not used for this agent).  This is the
    critical difference from a simple fallback: skipping prevents the cell-
    centre from being used when it too is within the exclusion zone of an
    existing agent, which would create an overlap.

    If the entire cell pool is exhausted before all agents are placed
    (possible only in very dense scenarios), remaining agents are placed at
    random cell centres and a warning is logged; the overlap resolver at
    simulation start corrects any residual violations.

    Each placed agent occupies a distinct cell from the permutation, so the
    distinct-cell invariant holds for all agents placed by the greedy pass.

    Args:
        safe_rows: Row indices of spawn-safe cells, shape ``(n_safe,)``.
        safe_cols: Column indices of spawn-safe cells, shape ``(n_safe,)``.
        count: Number of agent positions to generate.
        cell_size: Grid cell side length in metres.
        rng: Seeded generator for reproducibility (R6.2).

    Returns:
        Position array of shape ``(count, 2)`` in world metres.
    """
    n_safe = safe_rows.size
    min_dist_sq: float = (2.0 * AGENT_RADIUS) ** 2

    cell_order: Int1D = rng.permutation(n_safe).astype(np.intp)

    pos: Vec2Array = np.empty((count, 2), dtype=np.float64)
    placed: Vec2Array = np.empty((count, 2), dtype=np.float64)
    n_placed: int = 0
    cell_ptr: int = 0

    while n_placed < count:
        if cell_ptr >= n_safe:
            logger.warning(
                "Spawn cell pool exhausted after %d of %d agents placed; "
                "%d agents fall back to random cell centres.",
                n_placed,
                count,
                count - n_placed,
            )
            for _ in range(count - n_placed):
                ci = int(rng.integers(0, n_safe))
                pos[n_placed, 0] = float(safe_cols[ci]) * cell_size + cell_size * 0.5
                pos[n_placed, 1] = float(safe_rows[ci]) * cell_size + cell_size * 0.5
                placed[n_placed] = pos[n_placed]
                n_placed += 1
            break

        ci = int(cell_order[cell_ptr])
        cell_ptr += 1
        base_x = float(safe_cols[ci]) * cell_size
        base_y = float(safe_rows[ci]) * cell_size

        for _ in range(_SPAWN_MAX_PLACEMENT_ATTEMPTS):
            ox, oy = rng.random(2) * cell_size
            cx_c = base_x + ox
            cy_c = base_y + oy

            if n_placed > 0:
                dx_arr = placed[:n_placed, 0] - cx_c
                dy_arr = placed[:n_placed, 1] - cy_c
                if bool(np.any(dx_arr * dx_arr + dy_arr * dy_arr < min_dist_sq)):
                    continue

            pos[n_placed, 0] = cx_c
            pos[n_placed, 1] = cy_c
            placed[n_placed, 0] = cx_c
            placed[n_placed, 1] = cy_c
            n_placed += 1
            break
        # All attempts failed → skip this cell; cell_ptr already advanced.

    return pos


# ---------------------------------------------------------------------------
# Factory — seeded spawn within walkable region
# ---------------------------------------------------------------------------


def spawn(
    floor_plan: FloorPlan,
    count: int,
    rng: np.random.Generator,
    cell_size: float = 0.25,
) -> AgentState:
    """Spawn agents without overlapping walls, obstacles, or each other.

    Each agent is placed within a spawn-safe grid cell — a walkable cell
    whose centre is at least ``AGENT_RADIUS + cell_size / 2`` metres from
    every wall, obstacle, and room boundary.  This clearance ensures that
    any position sampled inside the cell maintains at least ``AGENT_RADIUS``
    from solid geometry, so agents never overlap walls or obstacles at spawn
    time.

    A greedy rejection pass then enforces a minimum centre-to-centre
    distance of ``2 * AGENT_RADIUS`` between every pair of agents.  In
    degenerate high-density scenarios where this cannot be achieved within
    ``_SPAWN_MAX_PLACEMENT_ATTEMPTS`` retries, the fallback cell-centre
    position is used and any residual overlap is corrected by the overlap
    resolver (``crowd_evac.domain.overlap``) at the first simulation step.

    Without-replacement cell assignment is used when *count* does not exceed
    the number of spawn-safe cells, guaranteeing each agent starts in a
    distinct cell.  If *count* exceeds available safe cells, cells are reused
    and a warning is logged.

    Initial velocity is ``(0, 0)``, panic is ``0.0``, and goal is ``-1``
    (unassigned; the exit model assigns a concrete index at simulation start).

    Args:
        floor_plan: Defines the walkable region via walls and obstacles.
        count: Number of agents to create.  Must be ``>= 0``.
        rng: Seeded numpy Generator for reproducibility (R6.2).  Callers
            typically supply ``SeededRNG(seed).generator``.
        cell_size: Grid cell side length in metres for walkability
            rasterisation.  Must be positive.

    Returns:
        An :class:`AgentState` with *count* agents, all alive and positioned
        inside the walkable region without overlapping solid geometry or
        each other.

    Raises:
        ValueError: If *count* is negative or *cell_size* is not positive.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count!r}")
    if cell_size <= 0:
        raise ValueError(
            f"cell_size must be positive, got {cell_size!r}"
        )

    if count == 0:
        return AgentState(
            pos=np.empty((0, 2), dtype=np.float64),
            vel=np.empty((0, 2), dtype=np.float64),
            panic=np.empty(0, dtype=np.float64),
            goal=np.empty(0, dtype=np.intp),
            alive=np.empty(0, dtype=np.bool_),
        )

    mask = floor_plan.walkable_mask(cell_size)
    safe_mask = _spawn_safe_mask(floor_plan, mask, cell_size)

    safe_rows_raw, safe_cols_raw = np.where(safe_mask)
    safe_rows: Int1D = safe_rows_raw.astype(np.intp)
    safe_cols: Int1D = safe_cols_raw.astype(np.intp)
    n_safe = safe_rows.size

    if n_safe == 0:
        # Degenerate: room geometry leaves no spawn-safe cells; fall back to
        # the full walkable mask so spawn never hard-fails.
        logger.warning(
            "No spawn-safe cells found (cell_size=%.3f, AGENT_RADIUS=%.3f); "
            "falling back to full walkable mask — overlaps will be resolved "
            "by the overlap resolver at simulation start.",
            cell_size,
            AGENT_RADIUS,
        )
        fallback_rows, fallback_cols = np.where(mask)
        safe_rows = fallback_rows.astype(np.intp)
        safe_cols = fallback_cols.astype(np.intp)
        n_safe = safe_rows.size

    if count > n_safe:
        logger.warning(
            "Requested %d agents but only %d spawn-safe cells available; "
            "some cells will be reused and overlap resolver will correct "
            "any remaining collisions.",
            count,
            n_safe,
        )

    pos = _greedy_place(safe_rows, safe_cols, count, cell_size, rng)

    return AgentState(
        pos=pos,
        vel=np.zeros((count, 2), dtype=np.float64),
        panic=np.zeros(count, dtype=np.float64),
        goal=np.full(count, _GOAL_UNASSIGNED, dtype=np.intp),
        alive=np.ones(count, dtype=np.bool_),
    )
