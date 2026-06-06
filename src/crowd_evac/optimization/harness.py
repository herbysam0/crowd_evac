"""Headless evaluation harness for ForceParams weight-set optimisation.

Provides :func:`evaluate` — a single headless simulation run that collects
every signal the downstream metrics modules need — and :func:`evaluate_batch`,
which distributes a Cartesian (candidates × seeds) job across all available
CPU cores using :class:`~concurrent.futures.ProcessPoolExecutor`.

No arcade or rendering imports; the module is safe to run in any headless
environment (CI, background job, subprocess worker).

Parallelisation notes
---------------------
``evaluate_batch`` uses separate processes (not threads) to avoid GIL
constraints on CPU-bound NumPy work.  On Windows the default start method is
``spawn``, so :func:`_eval_task` is a module-level function — lambdas and
nested functions are not picklable under ``spawn``.

GPU acceleration is intentionally deferred: the simulation is NumPy-based and
per-simulation GPU overhead (kernel launch latency) outweighs benefit at
typical agent counts.  GPU-parallel fitness aggregation is addressed in
Step 2.6 where the NSGA-II population loop is the primary bottleneck.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crowd_evac.adapters.io.scenario_loader import (
    load_bundled_scenario,
    load_scenario_file,
)
from crowd_evac.application.injection import add_panic_source
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.constants import (
    HAZARD_BLOCK_RADIUS,
    PANIC_DECAY_RATE,
    PANIC_RANGE,
)
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.params import ForceParams
from crowd_evac.metrics.records import make_record
from crowd_evac.pathfinding.flow_field import FlowField

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from crowd_evac.application.simulation import SimSnapshot
    from crowd_evac.domain.floor_plan import FloorPlan
    from crowd_evac.scenarios.schema import EmergencyEventData, ScenarioData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public defaults (may be overridden per call)
# ---------------------------------------------------------------------------

DEFAULT_MAX_TICKS: int = 10_000
"""Tick cap per evaluation.  10 000 ticks × 0.05 s = 500 s simulated time."""

DEFAULT_WALL_CLOCK_CAP_S: float = 120.0
"""Wall-clock safety cap per evaluation in seconds."""

DEFAULT_HISTORY_INTERVAL: int = 10
"""State snapshot collected every this many ticks (tick 0 is always included)."""


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunResult:
    """All signals produced by one headless simulation run.

    Scalar fields compare cleanly with ``==``.  NumPy array fields must be
    compared element-wise via :func:`numpy.array_equal`.

    Attributes:
        evac_time: Elapsed simulation time (s) at the last egress event, or
            ``sim_time`` at the terminal tick when ``is_terminal`` is True.
        evacuated_fraction: Fraction of initial agents that evacuated,
            in ``[0.0, 1.0]``.
        is_terminal: True when the run was stopped by the tick or wall-clock
            cap rather than natural simulation completion.
        total_ticks: Number of :meth:`~Simulation.step` calls executed.
        initial_count: Number of agents present at simulation start.
        throughput_series: Per-tick egress counts; ``len == total_ticks``.
        density_series: Per-tick peak density (agents/m²); ``len == total_ticks``.
        sample_ticks: Tick indices at which state was captured.  Always
            includes tick 0; subsequent samples at multiples of
            ``history_interval``.
        positions_history: Agent positions at each sample tick.
            Shape ``(len(sample_ticks), initial_count, 2)``.
        velocities_history: Agent velocities at each sample tick.
            Shape ``(len(sample_ticks), initial_count, 2)``.
        panics_history: Per-agent panic levels at each sample tick.
            Shape ``(len(sample_ticks), initial_count)``.
        alive_history: Per-agent liveness flags at each sample tick.
            Shape ``(len(sample_ticks), initial_count)``.
    """

    evac_time: float
    evacuated_fraction: float
    is_terminal: bool
    total_ticks: int
    initial_count: int
    throughput_series: tuple[int, ...]
    density_series: tuple[float, ...]
    sample_ticks: tuple[int, ...]
    positions_history: NDArray[np.float64]
    velocities_history: NDArray[np.float64]
    panics_history: NDArray[np.float64]
    alive_history: NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    params: ForceParams,
    scenario: str | Path,
    seed: int,
    max_ticks: int = DEFAULT_MAX_TICKS,
    *,
    wall_clock_cap_s: float = DEFAULT_WALL_CLOCK_CAP_S,
    history_interval: int = DEFAULT_HISTORY_INTERVAL,
) -> RunResult:
    """Run one headless simulation and return all evaluation signals.

    Builds a :class:`~crowd_evac.application.simulation.Simulation` from
    *scenario* + *params* + *seed*, steps it until
    :attr:`~Simulation.is_complete` is ``True`` or a hard cap is reached, and
    returns a :class:`RunResult` containing the per-tick metric series and a
    compact sampled state history.

    The same (*params*, *scenario*, *seed*) triple always produces a
    bit-identical :class:`RunResult` — the run is fully deterministic.  Any
    ``events`` declared in the scenario file (scripted emergencies) are part of
    that determinism: each is injected via
    :func:`~crowd_evac.application.injection.add_panic_source` at its scheduled
    tick, driving panic and (when it blocks navigation) re-routing the crowd —
    this is how ``hazard_avoidance_cost`` is exercised in headless validation.

    Args:
        params: Force-weight configuration to evaluate.
        scenario: Bundled scenario name (``str``) or filesystem path
            (:class:`~pathlib.Path`).  A ``str`` resolves via
            :func:`~crowd_evac.adapters.io.scenario_loader.load_bundled_scenario`;
            a :class:`~pathlib.Path` resolves via
            :func:`~crowd_evac.adapters.io.scenario_loader.load_scenario_file`.
        seed: RNG seed for agent spawning.  Identical seeds reproduce
            identical tick sequences for the same *params*.
        max_ticks: Hard tick cap.  When the loop reaches this count without
            natural completion ``is_terminal`` is set and the run terminates.
        wall_clock_cap_s: Wall-clock safety cap in seconds, checked before
            each tick.  Prevents pathological weight sets from blocking the
            process indefinitely.
        history_interval: Capture a state snapshot every this many ticks.
            Tick 0 (initial state) is always captured regardless of this value.

    Returns:
        A fully-populated :class:`RunResult`.

    Raises:
        TypeError: If *scenario* is neither ``str`` nor :class:`~pathlib.Path`.
        MalformedScenarioError: If the scenario file cannot be loaded.
    """
    floor_plan, scenario_data = _load_scenario(scenario)
    sim, initial_count = _build_sim(floor_plan, scenario_data, params, seed)
    events_by_tick = _events_by_tick(scenario_data.get("events", []))

    throughput: list[int] = []
    density: list[float] = []
    pos_hist: list[NDArray[np.float64]] = []
    vel_hist: list[NDArray[np.float64]] = []
    pan_hist: list[NDArray[np.float64]] = []
    alive_hist: list[NDArray[np.bool_]] = []
    sample_ticks_list: list[int] = []

    last_egress_time: float = 0.0
    is_terminal = False
    wall_start = time.monotonic()

    # Always capture the initial state at tick 0 before any steps.
    _append_sample(
        sim.snapshot(),
        pos_hist, vel_hist, pan_hist, alive_hist, sample_ticks_list,
    )

    tick_count = 0
    while not sim.is_complete:
        elapsed = time.monotonic() - wall_start
        if tick_count >= max_ticks or elapsed > wall_clock_cap_s:
            is_terminal = True
            logger.debug(
                "evaluate: cap reached ticks=%d wall_s=%.1f scenario=%s seed=%d",
                tick_count,
                elapsed,
                scenario,
                seed,
            )
            break

        # Fire any scripted emergency events scheduled for the current tick
        # before stepping, so the injected source is active for this step's
        # panic-decay/propagation phase (matching live injection semantics).
        if events_by_tick:
            _apply_due_events(sim, events_by_tick)

        sim.step()
        tick_count += 1

        snap = sim.snapshot()
        rec = make_record(snap, initial_count)

        throughput.append(rec["egressed_this_tick"])
        density.append(rec["peak_density_m2"])

        if rec["egressed_this_tick"] > 0:
            last_egress_time = rec["sim_time"]

        if snap.tick % history_interval == 0:
            _append_sample(
                snap,
                pos_hist, vel_hist, pan_hist, alive_hist, sample_ticks_list,
            )

    # evac_time: time of last egress; fall back to current sim_time if no egress.
    evac_time = last_egress_time if last_egress_time > 0.0 else sim.sim_time

    final_snap = sim.snapshot()
    if initial_count > 0:
        evacuated_fraction = min(
            float(final_snap.evacuated_count) / float(initial_count), 1.0
        )
    else:
        evacuated_fraction = 1.0

    if pos_hist:
        positions_history: NDArray[np.float64] = np.stack(pos_hist, axis=0)
        velocities_history: NDArray[np.float64] = np.stack(vel_hist, axis=0)
        panics_history: NDArray[np.float64] = np.stack(pan_hist, axis=0)
        alive_history: NDArray[np.bool_] = np.stack(alive_hist, axis=0)
    else:
        positions_history = np.empty((0, initial_count, 2), dtype=np.float64)
        velocities_history = np.empty((0, initial_count, 2), dtype=np.float64)
        panics_history = np.empty((0, initial_count), dtype=np.float64)
        alive_history = np.empty((0, initial_count), dtype=np.bool_)

    logger.debug(
        "evaluate: done ticks=%d evac_frac=%.3f terminal=%s scenario=%s seed=%d",
        tick_count,
        evacuated_fraction,
        is_terminal,
        scenario,
        seed,
    )

    return RunResult(
        evac_time=evac_time,
        evacuated_fraction=evacuated_fraction,
        is_terminal=is_terminal,
        total_ticks=tick_count,
        initial_count=initial_count,
        throughput_series=tuple(throughput),
        density_series=tuple(density),
        sample_ticks=tuple(sample_ticks_list),
        positions_history=positions_history,
        velocities_history=velocities_history,
        panics_history=panics_history,
        alive_history=alive_history,
    )


def evaluate_batch(
    candidates: Sequence[ForceParams],
    scenario: str | Path,
    seeds: Sequence[int],
    max_ticks: int = DEFAULT_MAX_TICKS,
    *,
    wall_clock_cap_s: float = DEFAULT_WALL_CLOCK_CAP_S,
    history_interval: int = DEFAULT_HISTORY_INTERVAL,
    max_workers: int | None = None,
) -> list[RunResult]:
    """Evaluate a Cartesian (candidates × seeds) batch in parallel.

    Distributes all ``len(candidates) × len(seeds)`` evaluations across a
    :class:`~concurrent.futures.ProcessPoolExecutor` (separate processes; no
    GIL constraints on NumPy work).

    Results are in row-major order: index ``i * len(seeds) + j`` holds the
    result for ``candidates[i]`` evaluated with ``seeds[j]``.

    Args:
        candidates: ForceParams instances to evaluate.  Must be non-empty.
        scenario: Bundled scenario name or filesystem path (see
            :func:`evaluate`).
        seeds: RNG seeds; each candidate is evaluated once per seed.
            Must be non-empty.
        max_ticks: Tick cap forwarded to each :func:`evaluate` call.
        wall_clock_cap_s: Wall-clock cap forwarded to each :func:`evaluate`.
        history_interval: History-sampling interval forwarded to each
            :func:`evaluate`.
        max_workers: Worker process count.  ``None`` uses
            :func:`os.cpu_count` (all logical cores).

    Returns:
        List of :class:`RunResult`, length ``len(candidates) × len(seeds)``.

    Raises:
        ValueError: If *candidates* or *seeds* is empty.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if not seeds:
        raise ValueError("seeds must be non-empty")

    n_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)

    tasks: list[tuple[ForceParams, str | Path, int, int, float, int]] = [
        (p, scenario, s, max_ticks, wall_clock_cap_s, history_interval)
        for p in candidates
        for s in seeds
    ]

    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_eval_task, tasks))
    wall_s = time.monotonic() - t0

    n_tasks = len(tasks)
    logger.info(
        "evaluate_batch: %d tasks / %d workers → %.1f s wall (%.2f s/task avg)",
        n_tasks,
        n_workers,
        wall_s,
        wall_s / n_tasks,
    )
    return results


def rerouted_flow_field(
    scenario: str | Path,
    params: ForceParams,
    seed: int = 0,
) -> FlowField:
    """Return the flow field after applying every scenario hazard block.

    Builds the scenario's simulation and applies *all* of its scripted events
    immediately (their navigation blocks and the graded danger cost scaled by
    ``params.hazard_avoidance_cost``), then returns the resulting re-routed
    field.  This is the field agents navigate while the hazard is active, so it
    is the correct reference for stuck-detection on a hazard run — unlike the
    base, hazard-free field used by the composite fitness (Step 2.6).

    The block geometry depends only on the floor plan, the hazards, and
    ``params``; it is independent of *seed* (which only affects agent spawning),
    so any seed yields the same field.

    Args:
        scenario: Bundled scenario name or filesystem path (see :func:`evaluate`).
        params: Force-weight configuration (supplies ``hazard_avoidance_cost``).
        seed: Spawn seed; irrelevant to the returned field.

    Returns:
        The re-routed :class:`~crowd_evac.pathfinding.flow_field.FlowField`.
        Equal to the base field when the scenario declares no events (or when a
        block would disconnect all exits, which the injector declines).
    """
    floor_plan, scenario_data = _load_scenario(scenario)
    sim, _ = _build_sim(floor_plan, scenario_data, params, seed)
    for event in scenario_data.get("events", []):
        _apply_event(sim, event)
    return sim.flow_field


# ---------------------------------------------------------------------------
# Module-level process-pool worker
# ---------------------------------------------------------------------------


def _eval_task(
    args: tuple[ForceParams, str | Path, int, int, float, int],
) -> RunResult:
    """Unpack args and delegate to :func:`evaluate`.

    Must be a top-level function so it is picklable under the Windows
    ``spawn`` start method used by
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    Args:
        args: ``(params, scenario, seed, max_ticks, wall_clock_cap_s,
            history_interval)``.

    Returns:
        RunResult from the evaluation.
    """
    params, scenario, seed, max_ticks, wall_clock_cap_s, history_interval = args
    return evaluate(
        params,
        scenario,
        seed,
        max_ticks,
        wall_clock_cap_s=wall_clock_cap_s,
        history_interval=history_interval,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_scenario(scenario: str | Path) -> tuple[FloorPlan, ScenarioData]:
    """Dispatch scenario loading by argument type.

    Args:
        scenario: Bundled name (``str``) or filesystem path (:class:`Path`).

    Returns:
        Tuple of ``(FloorPlan, ScenarioData)``.

    Raises:
        TypeError: If scenario is neither ``str`` nor :class:`~pathlib.Path`.
    """
    if isinstance(scenario, str):
        return load_bundled_scenario(scenario)
    if isinstance(scenario, Path):
        return load_scenario_file(scenario)
    raise TypeError(
        f"scenario must be str (bundled name) or Path (file path), "
        f"got {type(scenario).__name__!r}"
    )


def _events_by_tick(
    events: list[EmergencyEventData],
) -> dict[int, list[EmergencyEventData]]:
    """Group scenario events by their firing tick, preserving listed order.

    Args:
        events: The scenario's validated event list (possibly empty).

    Returns:
        Mapping of firing tick to the events scheduled at that tick, in the
        order they appear in the scenario file.  Empty when ``events`` is empty.
    """
    by_tick: dict[int, list[EmergencyEventData]] = {}
    for event in events:
        by_tick.setdefault(event["tick"], []).append(event)
    return by_tick


def _apply_due_events(
    sim: Simulation,
    events_by_tick: dict[int, list[EmergencyEventData]],
) -> None:
    """Inject every event scheduled for the simulation's current tick.

    Args:
        sim: Running simulation; mutated in-place by each injection.
        events_by_tick: Tick-grouped event schedule from
            :func:`_events_by_tick`.
    """
    for event in events_by_tick.get(sim.tick, ()):
        _apply_event(sim, event)


def _apply_event(sim: Simulation, event: EmergencyEventData) -> None:
    """Translate one scenario event into a panic-source injection.

    Optional event fields fall back to the same domain-constant defaults used
    by :func:`~crowd_evac.application.injection.add_panic_source`, so an event
    that specifies only ``tick``/``type``/``pos`` reproduces a default-strength
    hazard.

    Args:
        sim: Running simulation to inject into.
        event: A single validated emergency event.
    """
    pos = (event["pos"][0], event["pos"][1])
    add_panic_source(
        sim,
        event.get("source_type", "fire"),
        pos,
        intensity=event.get("intensity", 1.0),
        radius=event.get("radius", PANIC_RANGE),
        block_radius=event.get("block_radius", HAZARD_BLOCK_RADIUS),
        decay_rate=event.get("decay_rate", PANIC_DECAY_RATE),
        block_cells=event.get("blocks_navigation", True),
    )


def _build_sim(
    floor_plan: FloorPlan,
    scenario_data: ScenarioData,
    params: ForceParams,
    seed: int,
) -> tuple[Simulation, int]:
    """Construct a fully-wired Simulation from loaded scenario data.

    The scenario's own ``spawn_seed`` is ignored; *seed* is used instead so
    the harness controls reproducibility independently of the scenario file.

    Args:
        floor_plan: Domain floor plan from the scenario loader.
        scenario_data: Raw scenario dict; provides agent count.
        params: Force-weight configuration for this evaluation.
        seed: RNG seed for agent spawning.

    Returns:
        ``(Simulation, initial_agent_count)``.
    """
    count: int = scenario_data["agents"]["count"]
    rng = SeededRNG(seed)
    state = spawn(floor_plan, count, rng.generator)
    flow_field = FlowField.build(floor_plan)
    panic_field = PanicField()
    exit_model = ExitModel(floor_plan)
    collision_map = CollisionMap.from_floor_plan(floor_plan)
    sim = Simulation(
        state=state,
        flow_field=flow_field,
        panic_field=panic_field,
        exit_model=exit_model,
        rng=rng,
        params=params,
        collision_map=collision_map,
    )
    return sim, count


def _append_sample(
    snap: SimSnapshot,
    pos_hist: list[NDArray[np.float64]],
    vel_hist: list[NDArray[np.float64]],
    pan_hist: list[NDArray[np.float64]],
    alive_hist: list[NDArray[np.bool_]],
    sample_ticks: list[int],
) -> None:
    """Append one snapshot's arrays to the per-field accumulator lists.

    SimSnapshot arrays are already independent copies (Simulation contract),
    so no additional copying is required.

    Args:
        snap: Read-only simulation snapshot.
        pos_hist: Accumulator for position arrays.
        vel_hist: Accumulator for velocity arrays.
        pan_hist: Accumulator for panic-level arrays.
        alive_hist: Accumulator for liveness-flag arrays.
        sample_ticks: Accumulator for tick indices.
    """
    pos_hist.append(snap.positions)
    vel_hist.append(snap.velocities)
    pan_hist.append(snap.panics)
    alive_hist.append(snap.alive)
    sample_ticks.append(snap.tick)
