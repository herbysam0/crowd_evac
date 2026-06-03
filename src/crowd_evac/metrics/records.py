"""Per-tick simulation metric records (FR-8 / R8.2 / R8.3).

Defines :class:`MetricRecord`, a :class:`~typing.TypedDict` capturing the
four required FR-8 metric fields per simulation tick, and the
:func:`make_record` factory that populates it from a read-only
:class:`~crowd_evac.application.simulation.SimSnapshot`.

All computation is pure Python + NumPy; no arcade or I/O imports are
needed (R8.2 — headless availability).
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np

from crowd_evac.application.simulation import SimSnapshot

# Default cell side length for the local-density grid (metres).
_DEFAULT_CELL_SIZE: float = 1.0


class MetricRecord(TypedDict):
    """Per-tick simulation metric record (FR-8 R8.3).

    Attributes:
        tick: Monotonic simulation tick counter.
        sim_time: Elapsed simulation time in seconds.
        evacuated_count: Cumulative agents successfully egressed.
        evac_progress: Fraction of initial agents evacuated, in [0.0, 1.0].
        active_count: Agents still alive in the simulation this tick.
        egressed_this_tick: Total agents egressed during this tick
            (total throughput).
        per_exit_egress: Agents egressed per exit this tick, indexed by
            exit index (0-based).  Empty list when no per-exit data was
            logged (e.g. tick 0).
        peak_density_m2: Peak local agent density in agents per square
            metre within a ``cell_size × cell_size`` grid cell.
    """

    tick: int
    sim_time: float
    evacuated_count: int
    evac_progress: float
    active_count: int
    egressed_this_tick: int
    per_exit_egress: list[int]
    peak_density_m2: float


def make_record(
    snapshot: SimSnapshot,
    initial_count: int,
    *,
    cell_size: float = _DEFAULT_CELL_SIZE,
) -> MetricRecord:
    """Build a :class:`MetricRecord` from a read-only simulation snapshot.

    Extracts egress throughput from the ``tick_advanced`` event logged at
    ``snapshot.tick`` and computes peak local density from live agent
    positions using a 2-D histogram with ``cell_size × cell_size`` cells.

    Args:
        snapshot: Read-only simulation state obtained from
            :meth:`~crowd_evac.application.simulation.Simulation.snapshot`.
        initial_count: Total agents at simulation start.  Used to compute
            :attr:`MetricRecord.evac_progress`.  Must be non-negative.
        cell_size: Side length (metres) of each density grid cell.
            Must be positive.  Defaults to 1.0 m.

    Returns:
        A fully-populated :class:`MetricRecord` for this tick.

    Raises:
        ValueError: If *initial_count* is negative or *cell_size* is not
            positive.
    """
    if initial_count < 0:
        raise ValueError(
            f"initial_count must be non-negative, got {initial_count!r}"
        )
    if cell_size <= 0.0:
        raise ValueError(
            f"cell_size must be positive, got {cell_size!r}"
        )

    egressed_this_tick, per_exit_egress = _extract_throughput(snapshot)

    evac_progress = _evac_progress(snapshot.evacuated_count, initial_count)

    peak_density = _peak_density_m2(snapshot, cell_size)

    return MetricRecord(
        tick=snapshot.tick,
        sim_time=snapshot.sim_time,
        evacuated_count=snapshot.evacuated_count,
        evac_progress=evac_progress,
        active_count=snapshot.active_count,
        egressed_this_tick=egressed_this_tick,
        per_exit_egress=per_exit_egress,
        peak_density_m2=peak_density,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _evac_progress(evacuated_count: int, initial_count: int) -> float:
    """Return fraction of agents evacuated, clamped to [0.0, 1.0].

    Args:
        evacuated_count: Cumulative evacuated agents.
        initial_count: Total agents at simulation start.

    Returns:
        Evacuation fraction in [0.0, 1.0].  Returns 1.0 when
        *initial_count* is zero (vacuously complete).
    """
    if initial_count == 0:
        return 1.0
    return min(float(evacuated_count) / float(initial_count), 1.0)


def _extract_throughput(snapshot: SimSnapshot) -> tuple[int, list[int]]:
    """Extract per-tick egress counts from the snapshot event log.

    Searches for the most recent ``tick_advanced`` event whose ``tick``
    field matches ``snapshot.tick``.  Falls back to zero throughput when
    no such event is found (e.g. at tick 0 before any steps).

    Args:
        snapshot: Simulation snapshot whose event log is searched.

    Returns:
        A 2-tuple ``(egressed_this_tick, per_exit_egress)`` where
        ``egressed_this_tick`` is total agents egressed this tick and
        ``per_exit_egress`` is a list of per-exit counts (possibly empty).
    """
    for event in reversed(snapshot.events):
        if event.kind == "tick_advanced" and event.tick == snapshot.tick:
            egressed = int(event.payload.get("egressed", 0))
            per_exit_raw = event.payload.get("per_exit_egress", [])
            per_exit = [int(v) for v in per_exit_raw]
            return egressed, per_exit
    return 0, []


def _peak_density_m2(snapshot: SimSnapshot, cell_size: float) -> float:
    """Compute peak local agent density using a 2-D histogram.

    Bins all live agent positions into a regular ``cell_size × cell_size``
    grid and returns the count in the densest cell divided by the cell area.

    Args:
        snapshot: Simulation snapshot providing positions and alive flags.
        cell_size: Grid cell side length in metres.

    Returns:
        Peak density in agents per m².  Returns 0.0 if no live agents.
    """
    active_mask = snapshot.alive
    if not np.any(active_mask):
        return 0.0

    active_pos = snapshot.positions[active_mask]  # (M, 2)
    x = active_pos[:, 0]
    y = active_pos[:, 1]

    half = 0.5 * cell_size
    x_min = float(x.min()) - half
    x_max = float(x.max()) + half
    y_min = float(y.min()) - half
    y_max = float(y.max()) + half

    nx = max(1, int(np.ceil((x_max - x_min) / cell_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell_size)))

    hist, _, _ = np.histogram2d(
        x,
        y,
        bins=[nx, ny],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    return float(hist.max()) / (cell_size * cell_size)
