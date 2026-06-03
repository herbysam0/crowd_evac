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

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.floor_plan import FloorPlan

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Vec2Array: TypeAlias = npt.NDArray[np.float64]  # shape (N, 2)
Float1D: TypeAlias = npt.NDArray[np.float64]    # shape (N,)
Int1D: TypeAlias = npt.NDArray[np.intp]         # shape (N,)
Bool1D: TypeAlias = npt.NDArray[np.bool_]       # shape (N,)

# Sentinel: goal is unassigned until the exit model processes the agent.
_GOAL_UNASSIGNED: int = -1


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
# Factory — seeded spawn within walkable region
# ---------------------------------------------------------------------------


def spawn(
    floor_plan: FloorPlan,
    count: int,
    rng: np.random.Generator,
    cell_size: float = 0.25,
) -> AgentState:
    """Spawn agents at uniformly random positions in the walkable region.

    Each agent is placed at a random continuous position inside a walkable
    grid cell.  Cells are sampled **without replacement** when ``count``
    does not exceed the number of walkable cells, guaranteeing each agent
    starts in a distinct cell.  When ``count`` exceeds available cells,
    sampling falls back to replacement — some agents will share cells and
    may start close together; the crowd repulsion force (FR-2, step 1.8)
    separates them in the first simulation steps.

    Initial velocity is ``(0, 0)``, panic is ``0.0``, and goal is ``-1``
    (unassigned; the exit model assigns a concrete index during simulation).

    Args:
        floor_plan: Defines the walkable region via walls and obstacles.
        count: Number of agents to create. Must be ``>= 0``.
        rng: Seeded numpy Generator for reproducibility (R6.2). Callers
            typically supply ``SeededRNG(seed).generator``.
        cell_size: Grid cell side length in metres for walkability
            rasterisation.  Must be positive.

    Returns:
        An :class:`AgentState` with *count* agents, all alive and positioned
        inside the walkable region.

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
    cell_rows, cell_cols = np.where(mask)
    n_cells = cell_rows.size

    # Without-replacement sampling avoids two agents sharing the same cell;
    # replacement is used only when count exceeds the walkable cell count.
    replace = count > n_cells
    chosen = rng.choice(n_cells, size=count, replace=replace)
    r = cell_rows[chosen]
    c = cell_cols[chosen]

    # Uniform random offset within each chosen cell, in [0, cell_size).
    offsets = rng.random((count, 2)) * cell_size

    pos = np.empty((count, 2), dtype=np.float64)
    pos[:, 0] = c * cell_size + offsets[:, 0]   # x from column
    pos[:, 1] = r * cell_size + offsets[:, 1]   # y from row

    return AgentState(
        pos=pos,
        vel=np.zeros((count, 2), dtype=np.float64),
        panic=np.zeros(count, dtype=np.float64),
        goal=np.full(count, _GOAL_UNASSIGNED, dtype=np.intp),
        alive=np.ones(count, dtype=np.bool_),
    )
