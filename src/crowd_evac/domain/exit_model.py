"""Gated per-exit door queues and agent egress (FR-5 R5.1 / R5.2, issue #17).

Each door is a gate metered by its ``capacity_per_second``:

1. **Gate** — no agent passes a door line unless the door released it.
   Agents whose body reaches the line are held there (centre one radius
   inside, outward velocity zeroed); the crowd behind them backs up
   physically.
2. **Queue** — agents held at a line join that door's FIFO queue; agents
   that drift away from the line drop out (and rejoin at the back later).
3. **Release** — a token bucket filling at ``capacity_per_second * dt`` per
   tick (capped at one second of credit) releases agents FIFO. A released
   agent is placed one radius past the line, inside the span, and keeps
   walking. Doors with no room behind them for a whole body (a boundary
   door without an apron) evacuate the released agent immediately.
4. **Evacuate** — a released agent is removed from
   :class:`~crowd_evac.domain.agent_state.AgentState` and counted in
   :attr:`ExitModel.evacuated_count` once it reaches the outer end of the
   egress path (gap / apron edge).

Walls collide and the gate holds the line, so nobody is ever evacuated
through a wall or faster than the door's capacity.

No framework or I/O imports — pure Python + NumPy.
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.constants import AGENT_RADIUS, DT
from crowd_evac.domain.door_geometry import DOOR_CLEAR_MARGIN_M
from crowd_evac.domain.egress_geometry import (
    EgressGeometry,
    gate,
    in_band,
    outward_distances,
    placement,
)
from crowd_evac.domain.floor_plan import FloorPlan

logger = logging.getLogger(__name__)


class ExitModel:
    """Gated door queues with capacity-metered release for a floor plan.

    Attributes:
        floor_plan: Floor plan whose exits are managed.
        dt: Fixed simulation timestep in seconds.
    """

    def __init__(
        self,
        floor_plan: FloorPlan,
        dt: float = DT,
        agent_radius: float = AGENT_RADIUS,
    ) -> None:
        """Initialise queues, token buckets and door geometry.

        Args:
            floor_plan: Defines exit positions, widths, and capacities.
            dt: Simulation timestep in seconds.  Must be positive.
            agent_radius: Body radius in metres used for the door line,
                placement and evacuation depth.  Must be positive.

        Raises:
            ValueError: If *dt* or *agent_radius* is not positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        self.floor_plan = floor_plan
        self.dt = dt
        n = len(floor_plan.exits)
        self._geometry = EgressGeometry.from_floor_plan(
            floor_plan, agent_radius
        )
        self._queues: list[deque[int]] = [deque() for _ in range(n)]
        # Agent id -> door that released it (walking out, not yet removed).
        self._released: dict[int, int] = {}
        self._tokens: list[float] = [0.0] * n
        self._evacuated: int = 0
        self._ticks_since_last_egress: int = 0
        self._last_per_exit: list[int] = [0] * n
        self._last_release_per_exit: list[int] = [0] * n

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def evacuated_count(self) -> int:
        """Total agents evacuated (removed) since this model was created."""
        return self._evacuated

    @property
    def last_egress_per_exit(self) -> tuple[int, ...]:
        """Per-exit evacuations (removals) from the most recent :meth:`step`.

        Indices follow
        :attr:`~crowd_evac.domain.floor_plan.FloorPlan.exits`; all zero
        before the first call.  Sums to that tick's :meth:`step` result.
        """
        return tuple(self._last_per_exit)

    @property
    def last_release_per_exit(self) -> tuple[int, ...]:
        """Per-exit door releases (gate throughput) from the last tick."""
        return tuple(self._last_release_per_exit)

    @property
    def queue_lengths(self) -> tuple[int, ...]:
        """Agents currently held in each door queue."""
        return tuple(len(q) for q in self._queues)

    @property
    def ticks_since_last_egress(self) -> int:
        """Consecutive ticks with no door release and no evacuation.

        Used by
        :func:`~crowd_evac.application.termination.is_evacuation_complete`
        to detect the 'no further egress possible' terminal state (R5.3).
        """
        return self._ticks_since_last_egress

    # ------------------------------------------------------------------
    # Tick interface
    # ------------------------------------------------------------------

    def step(self, state: AgentState) -> int:
        """Process one tick: gate, queue, release, evacuate.

        Args:
            state: Agent population, updated in place (positions and
                velocities of gated / released agents, ``goal`` of queued
                agents, removal of evacuated agents).

        Returns:
            Number of agents evacuated (removed) this tick.
        """
        self._last_per_exit = [0] * len(self.floor_plan.exits)
        self._last_release_per_exit = [0] * len(self.floor_plan.exits)
        self._gate_and_queue(state)
        released = self._release(state)
        self._evacuate(state)
        # Count from the per-exit tally: it includes agents removed directly
        # on release (doors with no room behind them) as well as released
        # agents reaching the end of the egress path.
        evacuated = sum(self._last_per_exit)
        if released or evacuated:
            self._ticks_since_last_egress = 0
        else:
            self._ticks_since_last_egress += 1
        return evacuated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def hold_at_doors(self, state: AgentState) -> None:
        """Clamp unreleased agents back to the door lines (the gate alone).

        Idempotent. :meth:`step` applies it itself; the simulation also runs
        it inside each no-overlap projection pass so the gate behaves like a
        wall there and the projection does not push held agents through it.

        Args:
            state: Agent population; positions / velocities of agents
                straddling a door line are corrected in place.
        """
        active = state.active_indices
        if active.size == 0:
            return
        released = np.isin(active, np.fromiter(self._released, np.intp))
        pos = state.pos[active]
        vel = state.vel[active]
        already_past = gate(pos, vel, released, self._geometry)
        state.pos[active] = pos
        state.vel[active] = vel
        for row in np.flatnonzero(already_past >= 0):
            self._released[int(active[row])] = int(already_past[row])

    def _gate_and_queue(self, state: AgentState) -> None:
        """Hold unreleased agents at door lines and refresh the queues."""
        self.hold_at_doors(state)
        active = state.active_indices
        if active.size == 0:
            for q in self._queues:
                q.clear()
            return
        geom = self._geometry
        released = np.isin(active, np.fromiter(self._released, np.intp))
        pos = state.pos[active]
        at_line = (
            in_band(pos, geom)
            & (outward_distances(pos, geom) >= -geom.radius - DOOR_CLEAR_MARGIN_M)
            & ~released[:, np.newaxis]
        )
        door_of = {
            int(active[row]): int(np.argmax(at_line[row]))
            for row in np.flatnonzero(at_line.any(axis=1))
        }
        for e, queue in enumerate(self._queues):
            # Keep FIFO order of agents still at this line; append arrivals.
            kept = [ai for ai in queue if door_of.get(ai) == e]
            queued = set(kept)
            kept.extend(
                ai for ai, d in door_of.items() if d == e and ai not in queued
            )
            self._queues[e] = deque(kept)
            for ai in kept:
                state.goal[ai] = e

    def _release(self, state: AgentState) -> int:
        """Release queue heads up to token capacity; return release count."""
        immediate: list[tuple[int, int]] = []
        total = 0
        for e, exit_ in enumerate(self.floor_plan.exits):
            cap = exit_.capacity_per_second
            self._tokens[e] = min(self._tokens[e] + cap * self.dt, float(cap))
            allowed = math.floor(self._tokens[e])
            ids: list[int] = []
            while self._queues[e] and len(ids) < allowed:
                ai = self._queues[e].popleft()
                if state.alive[ai]:
                    ids.append(ai)
            self._tokens[e] -= len(ids)
            self._last_release_per_exit[e] = len(ids)
            total += len(ids)
            if not ids:
                continue
            if self._geometry.place_after[e]:
                rows = np.asarray(ids, dtype=np.intp)
                state.pos[rows] = placement(state.pos[rows], e, self._geometry)
                self._released.update((ai, e) for ai in ids)
            else:
                immediate.extend((ai, e) for ai in ids)
        self._remove(state, immediate)
        return total

    def _evacuate(self, state: AgentState) -> int:
        """Remove released agents that reached the end of the egress path."""
        if not self._released:
            return 0
        ids = np.fromiter(self._released, dtype=np.intp)
        doors = np.fromiter(self._released.values(), dtype=np.intp)
        s = outward_distances(state.pos[ids], self._geometry)
        depth = s[np.arange(ids.size), doors]
        alive = state.alive[ids]
        done = ~alive | (depth >= self._geometry.remove_depth[doors])
        for ai in ids[done]:
            del self._released[int(ai)]
        out = alive & done
        return self._remove(
            state, [(int(a), int(d)) for a, d in zip(ids[out], doors[out])]
        )

    def _remove(self, state: AgentState, items: list[tuple[int, int]]) -> int:
        """Remove ``(agent, door)`` pairs from *state*; return the count."""
        if not items:
            return 0
        state.remove([ai for ai, _ in items])
        for _, e in items:
            self._last_per_exit[e] += 1
        self._evacuated += len(items)
        logger.debug("Evacuated %d agents this tick.", len(items))
        return len(items)
