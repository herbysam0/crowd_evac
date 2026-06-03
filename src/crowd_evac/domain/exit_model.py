"""Per-exit capacity queuing and agent egress (FR-5 R5.1 / R5.2).

Agents enter an exit's FIFO queue when they come within
``capture_radius`` of the **nearest point on that exit's opening
segment** (not the centre), so wide exits capture approaching agents
uniformly along their full width.  Each tick the queue drains at most
``floor(tokens)`` agents, where tokens accumulate at
``exit.capacity_per_second * dt`` per tick (capped at one second of
credit to limit burst throughput).  Egressed agents are removed from
:class:`~crowd_evac.domain.agent_state.AgentState` and counted in
:attr:`ExitModel.evacuated_count`.

No framework or I/O imports — pure Python + NumPy.
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.constants import DT, EXIT_CAPTURE_RADIUS
from crowd_evac.domain.floor_plan import ExitSide, FloorPlan

logger = logging.getLogger(__name__)


class ExitModel:
    """Per-exit queuing and egress processing for a floor plan.

    Each exit owns a FIFO queue and a token-bucket accumulator that fills
    at ``exit.capacity_per_second * dt`` per tick (capped at one second
    of credit to bound burst throughput).  Agents within
    ``capture_radius`` of the nearest exit opening segment are added to
    that exit's queue; queues drain FIFO until the token bucket empties.

    Removal delegates to
    :meth:`~crowd_evac.domain.agent_state.AgentState.remove`; egressed
    agents neither exert nor receive forces in subsequent ticks.

    Attributes:
        floor_plan: Floor plan whose exits are managed.
        dt: Fixed simulation timestep in seconds.
        capture_radius: Perpendicular distance from an exit opening
            segment at which an agent enters the queue.
    """

    def __init__(
        self,
        floor_plan: FloorPlan,
        dt: float = DT,
        capture_radius: float = EXIT_CAPTURE_RADIUS,
    ) -> None:
        """Initialise queues and token buckets for every exit.

        Args:
            floor_plan: Defines exit positions, widths, and capacities.
            dt: Simulation timestep in seconds.  Must be positive.
            capture_radius: Distance from the exit segment triggering
                queue entry, in metres.  Must be positive.

        Raises:
            ValueError: If *dt* or *capture_radius* is not positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        if capture_radius <= 0.0:
            raise ValueError(
                f"capture_radius must be positive, got {capture_radius!r}"
            )
        self.floor_plan = floor_plan
        self.dt = dt
        self.capture_radius = capture_radius

        n = len(floor_plan.exits)
        self._queues: list[deque[int]] = [deque() for _ in range(n)]
        # Global set of agent IDs already in any queue (prevents re-entry).
        self._queued_set: set[int] = set()
        # Token bucket per exit; capped at capacity_per_second (1 s of credit).
        self._tokens: list[float] = [0.0] * n
        self._evacuated: int = 0
        self._ticks_since_last_egress: int = 0
        # Precomputed segment geometry for vectorised arrival queries.
        self._centres, self._half_widths, self._is_horiz = (
            _precompute_segments(floor_plan)
        )

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def evacuated_count(self) -> int:
        """Total agents successfully egressed since this model was created."""
        return self._evacuated

    @property
    def ticks_since_last_egress(self) -> int:
        """Consecutive ticks in which no agent has egressed.

        Resets to zero whenever at least one agent egresses in a tick.
        Used by
        :func:`~crowd_evac.application.termination.is_evacuation_complete`
        to detect the 'no further egress possible' terminal state (R5.3).
        """
        return self._ticks_since_last_egress

    # ------------------------------------------------------------------
    # Tick interface
    # ------------------------------------------------------------------

    def step(self, state: AgentState) -> int:
        """Process one simulation tick: enqueue arrivals then drain exits.

        Steps executed in order:

        1. Each active agent within *capture_radius* of an exit opening
           segment (and not yet queued) is added to the nearest exit's
           FIFO queue.  The agent's ``goal`` entry is set to that exit's
           index (0-based).
        2. Each exit's token bucket is refilled by
           ``capacity_per_second * dt``, capped at one second of credit.
        3. Each queue drains FIFO until ``floor(tokens)`` slots have been
           consumed.  Alive agents are removed from *state* via
           :meth:`~crowd_evac.domain.agent_state.AgentState.remove`;
           dead agents already in the queue are discarded without
           consuming a capacity token.
        4. :attr:`evacuated_count` is incremented by the egress count;
           :attr:`ticks_since_last_egress` resets to zero on any egress,
           otherwise increments by one.

        Args:
            state: Agent population to update in place.

        Returns:
            Number of agents egressed this tick (zero or more).
        """
        self._enqueue_arrivals(state)
        egressed = self._drain_queues(state)
        if egressed > 0:
            self._ticks_since_last_egress = 0
        else:
            self._ticks_since_last_egress += 1
        return egressed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enqueue_arrivals(self, state: AgentState) -> None:
        """Add newly-arrived agents to exit queues; nearest segment wins."""
        active = state.active_indices
        if active.size == 0 or len(self.floor_plan.exits) == 0:
            return

        pos = state.pos[active]  # (A, 2)
        dist = _segment_distances(
            pos, self._centres, self._half_widths, self._is_horiz
        )  # (A, n_exits)

        min_dist: npt.NDArray[np.float64] = dist.min(axis=1)  # (A,)
        nearest: npt.NDArray[np.intp] = (
            dist.argmin(axis=1).astype(np.intp)
        )  # (A,)
        in_range: npt.NDArray[np.bool_] = min_dist < self.capture_radius

        for agent_idx, exit_idx, close in zip(active, nearest, in_range):
            ai = int(agent_idx)
            if close and ai not in self._queued_set:
                ei = int(exit_idx)
                self._queues[ei].append(ai)
                self._queued_set.add(ai)
                state.goal[ai] = ei

    def _drain_queues(self, state: AgentState) -> int:
        """Drain exit queues up to token capacity; return total egressed."""
        total_egressed = 0
        to_remove: list[int] = []

        for i, exit_ in enumerate(self.floor_plan.exits):
            cap = exit_.capacity_per_second
            # Refill token bucket, capped at 1 second of credit.
            self._tokens[i] = min(
                self._tokens[i] + cap * self.dt, float(cap)
            )
            allowed = math.floor(self._tokens[i])
            drained = 0
            while self._queues[i] and drained < allowed:
                ai = self._queues[i].popleft()
                self._queued_set.discard(ai)
                if state.alive[ai]:
                    to_remove.append(ai)
                    drained += 1
                # Dead agents are discarded without consuming a capacity token.
            self._tokens[i] -= drained
            total_egressed += drained

        if to_remove:
            state.remove(to_remove)
            self._evacuated += len(to_remove)
            logger.debug("Egressed %d agents this tick.", len(to_remove))

        return total_egressed


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no state)
# ---------------------------------------------------------------------------


def _precompute_segments(
    floor_plan: FloorPlan,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    """Extract per-exit segment data for vectorised distance queries.

    Args:
        floor_plan: Floor plan containing exit definitions.

    Returns:
        A three-tuple of:

        - ``centres``: float64 array of shape ``(n_exits, 2)`` with exit
          ``(x, y)`` centres.
        - ``half_widths``: float64 array of shape ``(n_exits,)`` with
          ``width_m / 2`` for each exit.
        - ``is_horiz``: bool array of shape ``(n_exits,)``; ``True`` for
          NORTH / SOUTH exits whose opening runs along the x-axis.
    """
    exits = floor_plan.exits
    n = len(exits)
    centres = np.empty((n, 2), dtype=np.float64)
    half_widths = np.empty(n, dtype=np.float64)
    is_horiz = np.empty(n, dtype=np.bool_)
    for i, e in enumerate(exits):
        centres[i, 0] = e.x
        centres[i, 1] = e.y
        half_widths[i] = e.width_m / 2.0
        is_horiz[i] = e.side in (ExitSide.NORTH, ExitSide.SOUTH)
    return centres, half_widths, is_horiz


def _segment_distances(
    pos: npt.NDArray[np.float64],
    centres: npt.NDArray[np.float64],
    half_widths: npt.NDArray[np.float64],
    is_horiz: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Distance from each agent to the nearest point on each exit segment.

    For a NORTH / SOUTH exit the opening runs along the x-axis::

        nearest_x = clamp(agent_x,  cx − hw,  cx + hw)
        nearest_y = cy

    For an EAST / WEST exit the opening runs along the y-axis::

        nearest_x = cx
        nearest_y = clamp(agent_y,  cy − hw,  cy + hw)

    Args:
        pos: Agent positions, shape ``(A, 2)``.
        centres: Exit centres, shape ``(n_exits, 2)``.
        half_widths: Exit half-widths, shape ``(n_exits,)``.
        is_horiz: ``True`` for NORTH / SOUTH exits, shape ``(n_exits,)``.

    Returns:
        Float64 distance array of shape ``(A, n_exits)``.
    """
    # Column vectors broadcast against (n_exits,) rows → (A, n_exits).
    ax = pos[:, 0:1]    # (A, 1)
    ay = pos[:, 1:2]    # (A, 1)
    cx = centres[:, 0]  # (n_exits,)
    cy = centres[:, 1]  # (n_exits,)
    hw = half_widths    # (n_exits,)

    # Horizontal exits: clamp agent_x to [cx−hw, cx+hw]; pin agent_y to cy.
    nx: npt.NDArray[np.float64] = np.where(
        is_horiz, np.clip(ax, cx - hw, cx + hw), cx
    )  # (A, n_exits)
    ny: npt.NDArray[np.float64] = np.where(
        is_horiz, cy, np.clip(ay, cy - hw, cy + hw)
    )  # (A, n_exits)

    dist: npt.NDArray[np.float64] = np.sqrt(
        (ax - nx) ** 2 + (ay - ny) ** 2
    )
    return dist
