"""Fixed-step simulation orchestrator (FR-6 / R6.1 / R6.2 / R6.3 / NFR-R3).

:class:`Simulation` ties together all domain components — agent state,
flow field, panic field, exit model — into a single monotonic fixed-timestep
loop.  Each call to :meth:`Simulation.step` advances the simulation by one
``DT``-second tick following this pipeline:

1. Decay panic sources (``PanicField.decay_all``).
2. Raise each active agent's panic level to at least the panic field value
   at their position (panic propagation).
3. Compose all enabled force terms (``forces.compose``).
4. Integrate velocities and positions (``integrator.step``).
5. Resolve collisions against static geometry so no agent crosses a wall or
   obstacle (``CollisionMap.resolve``); skipped when no map is supplied.
6. Enforce the hard no-overlap invariant so agents never overlap agents or
   walls (``overlap.resolve_overlaps``); skipped when disabled.
7. Resolve exits (``ExitModel.step``).
8. Increment the monotonic tick counter.
9. Append a ``"tick_advanced"`` event to the event log.

A :meth:`~Simulation.snapshot` method returns a copy of the current state for
rendering and metrics — calling it any number of times between ticks produces
identical results and never alters simulation outcome (R6.1 — frame-rate
independence).

The event log accumulates :class:`SimEvent` records stamped with the tick at
which they were created (R6.3 / NFR-R3).  External callers (e.g. the
injection API in step 1.14) may append their own events via
:meth:`~Simulation.log_event`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.termination import is_evacuation_complete
from crowd_evac.domain import forces as _forces
from crowd_evac.domain import integrator as _integrator
from crowd_evac.domain.agent_state import AgentState, Bool1D, Float1D, Int1D, Vec2Array
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.constants import AGENT_PANIC_DECAY_RATE, DT
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.overlap import resolve_overlaps
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.pathfinding.flow_field import FlowField

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimEvent — tick-stamped event record
# ---------------------------------------------------------------------------


@dataclass
class SimEvent:
    """A tick-stamped event for the simulation event log (R6.3 / NFR-R3).

    Instances are immutable by convention; treat all fields as read-only after
    construction.  :attr:`payload` is a plain dict, so ``SimEvent`` is not
    hashable and cannot be used as a dict key or set member.

    Attributes:
        tick: Simulation tick at which the event was recorded.
        kind: Short event-type identifier, e.g. ``"tick_advanced"`` or
            ``"panic_source_added"``.
        payload: Arbitrary key/value data associated with this event type.
            Populated from the ``**payload`` kwargs of :meth:`Simulation.log_event`.
    """

    tick: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SimSnapshot — read-only copy of simulation state at a given tick
# ---------------------------------------------------------------------------


@dataclass
class SimSnapshot:
    """Read-only view of simulation state at a specific tick (R7.3 prep).

    All NumPy arrays are *copies* taken at snapshot time; later simulation
    steps do not alter a previously returned snapshot.  Calling
    :meth:`~Simulation.snapshot` any number of times between ticks always
    returns identical data (R6.1 — frame-rate independence).

    Array rows are indexed 0..(N-1) where N is the total agent count,
    including already-egressed agents.  Use :attr:`alive` to distinguish
    active from egressed agents.

    Attributes:
        tick: Monotonic tick counter at snapshot time.
        sim_time: Elapsed simulation time in seconds (``tick × DT``).
        positions: Agent world positions, shape ``(N, 2)``, metres.
        velocities: Agent velocities, shape ``(N, 2)``, m/s.
        panics: Agent panic levels in ``[0, 1]``, shape ``(N,)``.
        alive: Liveness flags, shape ``(N,)``; ``False`` = egressed.
        goals: Target exit indices (0-based or ``-1`` = unassigned),
            shape ``(N,)``.
        evacuated_count: Cumulative agents successfully egressed so far.
        active_count: Agents currently alive in the simulation.
        panic_sources: Panic sources at snapshot time.  These are
            references, not deep copies; treat as read-only.
        events: All events logged through this tick, oldest first.
    """

    tick: int
    sim_time: float
    positions: Vec2Array
    velocities: Vec2Array
    panics: Float1D
    alive: Bool1D
    goals: Int1D
    evacuated_count: int
    active_count: int
    panic_sources: tuple[PanicSource, ...]
    events: tuple[SimEvent, ...]


# ---------------------------------------------------------------------------
# Simulation — fixed-step orchestrator
# ---------------------------------------------------------------------------


class Simulation:
    """Fixed-step simulation orchestrator for a single evacuation scenario (FR-6).

    Ties domain components into a monotonic fixed-timestep loop.  Each
    :meth:`step` call advances by exactly one ``DT``-second tick.  Given the
    same seed and the same sequence of external events, the same tick sequence
    is always produced (R6.2).  Calling :meth:`snapshot` between steps does
    not alter simulation outcome (R6.1).

    Attributes:
        state: Live agent state (SoA).  Mutated in-place each tick.
        flow_field: Navigation flow field.  Rebuilt externally on re-route
            (step 1.14) and re-assigned via the public attribute.
        panic_field: Aggregate panic field.  Sources decay each tick.
        exit_model: Per-exit queue/token manager.  Drives egress.
        rng: Seeded generator threaded through for reproducibility.
        collision_map: Static blocking grid enforcing that agents never cross
            walls or obstacles, or ``None`` to disable the constraint.
    """

    def __init__(
        self,
        state: AgentState,
        flow_field: FlowField,
        panic_field: PanicField,
        exit_model: ExitModel,
        rng: SeededRNG,
        *,
        collision_map: CollisionMap | None = None,
        enable_exit: bool = True,
        enable_crowd: bool = True,
        enable_density: bool = True,
        enable_herd: bool = True,
        enable_panic_repulsion: bool = True,
        enable_no_overlap: bool = True,
    ) -> None:
        """Initialise the simulation with pre-built domain components.

        Args:
            state: Initial agent population.  Ownership is transferred;
                callers must not mutate the state directly after passing it in.
            flow_field: Pre-built navigation flow field for exit-seeking.
            panic_field: Panic field, possibly empty at construction time.
            exit_model: Exit capacity and queue manager bound to the same
                floor plan as ``flow_field``.
            rng: Seeded random generator for any stochastic components.
            collision_map: Static blocking grid built from the same floor plan;
                when supplied, integrated positions are clamped each tick so
                agents never enter a wall or obstacle cell.  ``None`` disables
                collision resolution (force-only headless tests).
            enable_exit: Toggle exit-seeking force term (default ``True``).
            enable_crowd: Toggle agent-agent repulsion (default ``True``).
            enable_density: Toggle density-pressure deceleration
                (default ``True``).
            enable_herd: Toggle panic-scaled herd alignment (default ``True``).
            enable_panic_repulsion: Toggle panic-gradient repulsion
                (default ``True``).
            enable_no_overlap: Toggle the hard no-overlap projection that
                guarantees agents never overlap agents or walls
                (default ``True``).
        """
        self.state = state
        self.flow_field = flow_field
        self.panic_field = panic_field
        self.exit_model = exit_model
        self.rng = rng
        self._collision_map = collision_map

        self._enable_exit = enable_exit
        self._enable_crowd = enable_crowd
        self._enable_density = enable_density
        self._enable_herd = enable_herd
        self._enable_panic_repulsion = enable_panic_repulsion
        self._enable_no_overlap = enable_no_overlap

        self._tick: int = 0
        self._events: list[SimEvent] = []

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        """Monotonic tick counter; increments by 1 on each :meth:`step` call."""
        return self._tick

    @property
    def sim_time(self) -> float:
        """Elapsed simulation time in seconds (``tick × DT``)."""
        return float(self._tick) * DT

    @property
    def is_complete(self) -> bool:
        """``True`` when evacuation has terminated (all evacuated or stalled).

        Delegates to
        :func:`~crowd_evac.application.termination.is_evacuation_complete`.
        """
        return is_evacuation_complete(self.state, self.exit_model)

    # ------------------------------------------------------------------
    # Tick interface
    # ------------------------------------------------------------------

    def step(self) -> int:
        """Advance the simulation by one fixed ``DT``-second timestep.

        Pipeline per tick:

        1. Decay all panic sources by ``DT`` seconds.
        2. Propagate panic field values onto agent panic levels (raise to
           at least the field value at each agent's position).
        3. Compose all enabled force terms into per-agent acceleration.
        4. Integrate velocities and positions (semi-implicit Euler).
        5. Resolve collisions: clamp integrated positions so no agent crosses
           a wall or obstacle (skipped when no collision map is set).
        6. Enforce the hard no-overlap invariant: project agent positions so
           no agent overlaps another agent or a wall (skipped when disabled).
        7. Resolve exits: enqueue arrivals, drain queues to token capacity.
        8. Increment the monotonic tick counter.
        9. Append a ``"tick_advanced"`` event to the event log, stamped
           at the new tick value.

        Returns:
            Number of agents egressed this tick (zero or more).

        Raises:
            RuntimeError: If called when :attr:`is_complete` is ``True``.
        """
        if self.is_complete:
            raise RuntimeError(
                f"step() called on a completed simulation "
                f"(tick={self._tick}); check is_complete before stepping."
            )

        # 1. Decay panic sources (intensity decreases over time)
        self.panic_field.decay_all(DT)

        # 2. Raise agent panic to at least local field value
        self._propagate_panic()

        # 3. Compose per-agent acceleration from all enabled force terms
        accel = _forces.compose(
            self.state,
            self.flow_field,
            self.panic_field,
            enable_exit=self._enable_exit,
            enable_crowd=self._enable_crowd,
            enable_density=self._enable_density,
            enable_herd=self._enable_herd,
            enable_panic_repulsion=self._enable_panic_repulsion,
        )

        # 4. Semi-implicit Euler: update velocities then positions
        prev_pos = self.state.pos.copy()
        _integrator.step(self.state, accel)

        # 5. Reject any move that would cross a wall or obstacle (R1.4 / FR-3)
        if self._collision_map is not None:
            self._collision_map.resolve(self.state, prev_pos)

        # 6. Enforce the hard no-overlap invariant: agents never overlap agents
        #    or walls (step 1.19a item 12 / FR-2 R2.1 / FR-3 R3.2).
        if self._enable_no_overlap:
            resolve_overlaps(self.state, self._collision_map, prev_pos)

        # 7. Resolve exits: enqueue arrivals, drain to capacity
        egressed = self.exit_model.step(self.state)

        # 8. Advance tick (now represents "this step has completed")
        self._tick += 1

        # 9. Stamp tick event at the new tick value
        self._append_event(
            "tick_advanced",
            egressed=egressed,
            active=int(self.state.active_indices.size),
            evacuated=self.exit_model.evacuated_count,
            per_exit_egress=list(self.exit_model.last_egress_per_exit),
        )

        if egressed > 0:
            logger.debug(
                "Tick %d: %d egressed (total evacuated=%d).",
                self._tick,
                egressed,
                self.exit_model.evacuated_count,
            )

        return egressed

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def log_event(self, kind: str, **payload: Any) -> None:
        """Append a tick-stamped event to the event log.

        External callers (e.g. the injection API in step 1.14) record domain
        events with their causal tick (R6.3 / NFR-R3).  Events logged between
        steps carry the *current* :attr:`tick` value.

        Args:
            kind: Short event-type identifier.
            **payload: Arbitrary key/value data for this event.
        """
        self._append_event(kind, **payload)

    def _append_event(self, kind: str, **payload: Any) -> None:
        """Stamp an event at the current tick and append it to the log."""
        self._events.append(
            SimEvent(tick=self._tick, kind=kind, payload=dict(payload))
        )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> SimSnapshot:
        """Return a read-only copy of simulation state at the current tick.

        All NumPy arrays are copied so the snapshot is independent of any
        subsequent simulation steps.  Calling this method any number of times
        between ticks always produces identical results (R6.1 — frame-rate
        independence).

        Returns:
            A :class:`SimSnapshot` capturing state at the current tick.
        """
        return SimSnapshot(
            tick=self._tick,
            sim_time=self.sim_time,
            positions=self.state.pos.copy(),
            velocities=self.state.vel.copy(),
            panics=self.state.panic.copy(),
            alive=self.state.alive.copy(),
            goals=self.state.goal.copy(),
            evacuated_count=self.exit_model.evacuated_count,
            active_count=int(self.state.active_indices.size),
            panic_sources=tuple(self.panic_field.sources),
            events=tuple(self._events),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _propagate_panic(self) -> None:
        """Update active agents' panic toward the local panic field value.

        Panic rises immediately to the field value when an agent is inside
        an influence radius.  When the field value at an agent's position
        drops below their current panic — because they moved out of range or
        the source decayed — their panic decays toward the field value at
        ``AGENT_PANIC_DECAY_RATE`` per second, never falling below the field
        value itself.

        ``np.maximum(field_vals, panic - rate * DT)`` handles both cases in
        one expression: when field_vals >= panic the max selects field_vals
        (panic rises); when field_vals < panic the max decays panic toward
        field_vals but never below it.
        """
        active = self.state.active_indices
        if active.size == 0:
            return
        field_vals = self.panic_field.value_at(self.state.pos[active])
        self.state.panic[active] = np.maximum(
            field_vals,
            self.state.panic[active] - AGENT_PANIC_DECAY_RATE * DT,
        )
