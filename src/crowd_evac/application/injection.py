"""Runtime injection API: add panic sources mid-simulation (FR-12.1 / R4.3).

:func:`add_panic_source` is the single public entry point used by UI
adapters, scripted replays, and tests to inject hazards at arbitrary world
positions during a live run.  Each call:

1. Appends a :class:`~crowd_evac.domain.panic_source.PanicSource` to
   ``sim.panic_field``, making the panic gradient active from the *next*
   tick's decay and propagation phase (R12.1).
2. Optionally blocks the hazard's *physical* footprint (``block_radius``,
   not the larger panic ``radius``) in the navigation flow field and re-solves
   so the crowd routes around the hazard (R4.3 / R12.2).  The re-solve runs
   through :meth:`~crowd_evac.application.simulation.Simulation.refresh_hazard_blocks`,
   which rebuilds from the pristine floor mask for the full set of active
   hazards — so a block is never permanent: it is restored as soon as the
   source decays or is removed.  A
   :exc:`~crowd_evac.domain.errors.PathfindingError` — raised when blocking
   would disconnect all exits — is caught and logged rather than propagated;
   the panic field is always updated regardless.
3. Appends a tick-stamped ``"panic_source_added"`` event to the simulation
   event log (R6.3 / NFR-R3).

:func:`remove_panic_source` withdraws a source, re-solves the flow field so
the source's footprint is restored, and records the removal.

All mutations are routed through the
:class:`~crowd_evac.application.simulation.Simulation` instance so the
"domain state is owned by the loop" invariant is preserved.
"""
from __future__ import annotations

import logging

from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.constants import (
    HAZARD_BLOCK_RADIUS,
    PANIC_DECAY_RATE,
    PANIC_RANGE,
)
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.ports.input_source import (
    InputEvent,
    MovePanicSourceEvent,
    PlacePanicSourceEvent,
)

logger = logging.getLogger(__name__)


def add_panic_source(
    sim: Simulation,
    source_type: str,
    pos: tuple[float, float],
    intensity: float = 1.0,
    radius: float = PANIC_RANGE,
    *,
    block_radius: float = HAZARD_BLOCK_RADIUS,
    decay_rate: float = PANIC_DECAY_RATE,
    block_cells: bool = True,
) -> PanicSource:
    """Add a panic source to a running simulation (FR-12.1 / R4.3).

    Mutates ``sim.panic_field`` immediately and, when *block_cells* is
    ``True``, replaces ``sim.flow_field`` with a re-solved field that routes
    around the cells covered by the new source (R4.3 / R12.2).  A
    tick-stamped ``"panic_source_added"`` event is appended to the event log.

    The new source is effective from the **next** tick: the panic field
    evaluates it during the decay + propagation phase that opens each
    :meth:`~Simulation.step` call; the flow-field replacement is visible
    immediately to any subsequent ``sample`` call.

    Args:
        sim: Running simulation to inject into.  ``panic_field`` is mutated
            in-place; ``flow_field`` may be replaced by a re-solved instance.
        source_type: Human-readable type tag, e.g. ``"fire"``.  Stored in
            the event payload for replay and logging; unused by domain logic.
        pos: World position ``(x, y)`` in metres.
        intensity: Initial source intensity in ``[0, 1]``.  Defaults to
            ``1.0`` (fully active).
        radius: Panic-gradient influence radius in metres.  Defaults to
            :data:`~crowd_evac.domain.constants.PANIC_RANGE`.
        block_radius: Navigation-block footprint radius in metres, decoupled
            from the (larger) panic *radius*.  Defaults to
            :data:`~crowd_evac.domain.constants.HAZARD_BLOCK_RADIUS`; blocking
            the full panic radius would engulf and strand the surrounding
            crowd.
        decay_rate: Intensity reduction per simulated second.  Defaults to
            :data:`~crowd_evac.domain.constants.PANIC_DECAY_RATE`.
        block_cells: When ``True`` (default), grid cells whose centres fall
            within *block_radius* of *pos* are blocked in the navigation flow
            field and a bounded re-route solve is triggered (R4.3 / R12.2).
            The block is restored automatically when the source decays or is
            removed (:meth:`~crowd_evac.application.simulation.Simulation.refresh_hazard_blocks`).
            A :exc:`~crowd_evac.domain.errors.PathfindingError` is caught
            silently if blocking would disconnect every exit.  Set to
            ``False`` to update the panic gradient only.

    Returns:
        The :class:`~crowd_evac.domain.panic_source.PanicSource` appended to
        ``sim.panic_field``.  Retain a reference to pass to
        :func:`remove_panic_source` later.

    Raises:
        ValueError: If *intensity*, *radius*, or *decay_rate* fall outside
            valid ranges (propagated from
            :class:`~crowd_evac.domain.panic_source.PanicSource`).
    """
    x, y = float(pos[0]), float(pos[1])
    source = PanicSource(
        x=x,
        y=y,
        intensity=intensity,
        radius=radius,
        decay_rate=decay_rate,
        source_type=source_type,
        block_radius=block_radius,
        blocks_navigation=block_cells,
    )
    sim.panic_field.add_source(source)

    # Re-solve the flow field for the full active-hazard set (this source plus
    # any pre-existing ones).  A no-op when block_cells is False.
    blocked_count = sim.refresh_hazard_blocks() if block_cells else 0

    sim.log_event(
        "panic_source_added",
        source_type=source_type,
        pos=[x, y],
        intensity=intensity,
        radius=radius,
        blocked_cells=blocked_count,
    )
    logger.info(
        "Injected %s at (%.2f, %.2f) intensity=%.2f radius=%.2f "
        "blocked_cells=%d (tick=%d).",
        source_type,
        x,
        y,
        intensity,
        radius,
        blocked_count,
        sim.tick,
    )
    return source


def remove_panic_source(sim: Simulation, source: PanicSource) -> None:
    """Remove a panic source from a running simulation.

    Withdraws *source* from ``sim.panic_field`` and records the removal in
    the event log.

    The navigation flow field is re-solved from the pristine floor mask so
    the removed source's footprint is restored (its cells become walkable
    again), unless another active hazard still blocks them
    (:meth:`~crowd_evac.application.simulation.Simulation.refresh_hazard_blocks`).

    Args:
        sim: Running simulation.
        source: The :class:`~crowd_evac.domain.panic_source.PanicSource`
            returned by a prior :func:`add_panic_source` call.

    Raises:
        ValueError: If *source* is not present in ``sim.panic_field``
            (propagated from
            :meth:`~crowd_evac.domain.panic_field.PanicField.remove_source`).
    """
    x, y = source.x, source.y
    sim.panic_field.remove_source(source)
    sim.refresh_hazard_blocks()
    sim.log_event(
        "panic_source_removed",
        pos=[x, y],
        intensity_at_removal=source.intensity,
    )
    logger.info(
        "Removed panic source at (%.2f, %.2f) (tick=%d).",
        x,
        y,
        sim.tick,
    )


def process_input_events(
    sim: Simulation,
    events: list[InputEvent],
    current_source: PanicSource | None = None,
    *,
    source_type: str = "fire",
    intensity: float = 1.0,
    radius: float = PANIC_RANGE,
    decay_rate: float = PANIC_DECAY_RATE,
) -> PanicSource | None:
    """Route input events through the injection API (FR-7 R7.2 / FR-15 subset).

    Processes :class:`~crowd_evac.ports.input_source.PlacePanicSourceEvent`
    and :class:`~crowd_evac.ports.input_source.MovePanicSourceEvent` objects
    from an :class:`~crowd_evac.ports.input_source.InputSource`, calling
    :func:`add_panic_source` / :func:`remove_panic_source` on the simulation.
    This is the single bridge where raw UI events become domain mutations —
    no adapter may bypass this function to mutate domain state directly (R7.2).

    Each command is INFO-logged with CLI-equivalent syntax as a seed for the
    replay scripting feature (R15.4).

    Args:
        sim: Running simulation to inject into.
        events: Event list returned by ``InputSource.poll()``.
        current_source: Most recently placed :class:`PanicSource`, or ``None``.
            On a :class:`MovePanicSourceEvent`, this source is removed before
            the new one is placed.
        source_type: Hazard type tag passed to :func:`add_panic_source`.
        intensity: Initial source intensity in ``[0, 1]``.
        radius: Source influence radius in metres.
        decay_rate: Intensity reduction per simulated second.

    Returns:
        The updated current :class:`PanicSource` (replaced on place or move),
        or ``None`` if no events were processed or no source is active.
    """
    for event in events:
        if isinstance(event, PlacePanicSourceEvent):
            logger.info(
                "cmd add_panic_source %s %.4f %.4f"
                " intensity=%.2f radius=%.2f",
                source_type,
                event.pos_m[0],
                event.pos_m[1],
                intensity,
                radius,
            )
            current_source = add_panic_source(
                sim,
                source_type,
                event.pos_m,
                intensity=intensity,
                radius=radius,
                decay_rate=decay_rate,
            )
        elif isinstance(event, MovePanicSourceEvent):
            if current_source is not None:
                remove_panic_source(sim, current_source)
            logger.info(
                "cmd move_panic_source %s %.4f %.4f"
                " intensity=%.2f radius=%.2f",
                source_type,
                event.pos_m[0],
                event.pos_m[1],
                intensity,
                radius,
            )
            current_source = add_panic_source(
                sim,
                source_type,
                event.pos_m,
                intensity=intensity,
                radius=radius,
                decay_rate=decay_rate,
            )
    return current_source
