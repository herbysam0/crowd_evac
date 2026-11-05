"""Run a Monte Carlo batch: seeds, hazards, worker pool, cancellation.

Headless runs go through :func:`crowd_evac.optimization.harness.evaluate`
via the module-level :func:`run_one` (picklable under Windows ``spawn``).
``evaluate_batch`` is not used: it gives every run the same events, while
hazards here differ per run.

This module never imports arcade; ``--visual`` loads
:mod:`crowd_evac.montecarlo.visual` on demand.
"""
from __future__ import annotations

import math
import os
import secrets
import time
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path
from typing import NamedTuple

import numpy as np

from crowd_evac.adapters.io.scenario_loader import load_bundled_scenario
from crowd_evac.domain.constants import GRID_CELL_SIZE
from crowd_evac.domain.params import ForceParams
from crowd_evac.montecarlo.config import MonteCarloConfig
from crowd_evac.montecarlo.hazards import (
    check_specs_in_room,
    sample_events,
    validate_spec,
)
from crowd_evac.montecarlo.results import (
    MonteCarloResult,
    RunOutcome,
    build_curves,
    make_outcome,
    summarize,
)
from crowd_evac.optimization.harness import DEFAULT_MAX_TICKS, evaluate
from crowd_evac.ports.monte_carlo_listener import MonteCarloListener
from crowd_evac.scenarios.schema import EmergencyEventData

SEED_LIMIT: int = 2**32
"""Exclusive upper bound of every seed (uint32)."""

_CANCEL_POLL_S: float = 0.1
"""How often a pool batch re-checks ``should_cancel`` while runs work."""


class RunTask(NamedTuple):
    """Everything one run needs; plain data so it pickles to a worker.

    Attributes:
        scenario: Bundled room name (or a scenario file path in tests).
        agents: Persons to spawn.
        run_index: Zero-based run number.
        seed: Spawn seed.
        events: Concrete hazards for this run.
    """

    scenario: str | Path
    agents: int
    run_index: int
    seed: int
    events: tuple[EmergencyEventData, ...]


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def resolve_base_seed(seed: int | None) -> tuple[int, bool]:
    """Return the base seed and whether it was generated.

    Args:
        seed: Seed from the config, or ``None`` for a random one.

    Returns:
        ``(base_seed, was_random)``.
    """
    if seed is None:
        return secrets.randbelow(SEED_LIMIT), True
    return seed, False


def derive_seeds(base_seed: int, runs: int) -> tuple[int, ...]:
    """Derive one spawn seed per run from the base seed.

    Args:
        base_seed: Batch base seed.
        runs: Number of runs.

    Returns:
        Distinct plain ints in ``[0, 2**32)``, usable as the game's
        ``--seed``.  The first ``k`` do not change when ``runs`` grows.
    """
    state = np.random.SeedSequence(base_seed).generate_state(runs, np.uint32)
    return tuple(int(s) for s in state)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_one(task: RunTask) -> RunOutcome:
    """Run one headless simulation and summarize it.

    No wall-clock cap is set (it would depend on machine load and break
    reproducibility); ``DEFAULT_MAX_TICKS`` is the only cap.  History
    sampling is pushed past that cap so only tick 0 is stored.

    Args:
        task: The run to execute.

    Returns:
        Its outcome.
    """
    result = evaluate(
        ForceParams.defaults(),
        task.scenario,
        task.seed,
        DEFAULT_MAX_TICKS,
        wall_clock_cap_s=math.inf,
        history_interval=DEFAULT_MAX_TICKS + 1,
        agent_count=task.agents,
        events=task.events,
    )
    return make_outcome(
        task.run_index, task.seed, task.events, task.agents,
        result.throughput_series, result.moving_speed_mps,
    )


def _run_inline(
    tasks: list[RunTask],
    notify: Callable[[RunOutcome], None],
    should_cancel: Callable[[], bool],
) -> list[RunOutcome]:
    """Run tasks one after another in this process."""
    outcomes: list[RunOutcome] = []
    for task in tasks:
        if should_cancel():
            break
        outcome = run_one(task)
        outcomes.append(outcome)
        notify(outcome)
    return outcomes


def _run_pool(
    tasks: list[RunTask],
    workers: int,
    notify: Callable[[RunOutcome], None],
    should_cancel: Callable[[], bool],
) -> list[RunOutcome]:
    """Run tasks in worker processes; on cancel, keep what finishes.

    Runs are submitted only as workers free up, so cancelling stops every
    run that has not started.  Runs already started finish and are kept.
    """
    outcomes: list[RunOutcome] = []
    queue = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: set[Future[RunOutcome]] = set()
        cancelled = False
        while True:
            while not cancelled and len(pending) < workers:
                if should_cancel():
                    cancelled = True
                    break
                task = next(queue, None)
                if task is None:
                    break
                pending.add(pool.submit(run_one, task))
            if not pending:
                break
            done, pending = wait(
                pending, timeout=_CANCEL_POLL_S, return_when=FIRST_COMPLETED
            )
            for future in done:
                outcome = future.result()
                outcomes.append(outcome)
                notify(outcome)
    return outcomes


def _execute(
    config: MonteCarloConfig,
    tasks: list[RunTask],
    listener: MonteCarloListener | None,
    should_cancel: Callable[[], bool],
) -> list[RunOutcome]:
    """Pick visual, in-process or pooled execution and run the tasks."""
    total = len(tasks)
    done = 0

    def notify(outcome: RunOutcome) -> None:
        """Count a finished run and tell the listener."""
        nonlocal done
        done += 1
        if listener is not None:
            listener.on_run_complete(outcome, done, total)

    if config.visual:
        # Lazy: only visual batches need arcade.
        from crowd_evac.montecarlo.visual import run_visual

        return run_visual(
            tasks, notify, should_cancel, start_paused=config.start_paused
        )
    workers = min(config.max_workers or os.cpu_count() or 1, total)
    if workers == 1:
        return _run_inline(tasks, notify, should_cancel)
    return _run_pool(tasks, workers, notify, should_cancel)


def _spawn_seeds(config: MonteCarloConfig, base_seed: int) -> tuple[int, ...]:
    """Return the per-run spawn seeds: the base seed itself if direct."""
    if config.direct_seed:
        return (base_seed,)
    return derive_seeds(base_seed, config.runs)


def run_monte_carlo(
    config: MonteCarloConfig,
    listener: MonteCarloListener | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> MonteCarloResult:
    """Run a Monte Carlo batch.

    The interface the future dashboard calls directly.  Hazards are checked
    and sampled for every run before the first run starts, so a bad range
    fails at once.  Cancelling takes effect between runs; runs already
    started finish and are kept.

    Args:
        config: Batch settings.
        listener: Told after each finished run.
        should_cancel: Polled to stop the batch early.

    Returns:
        The finished runs, their statistics and graph data.  Same config
        and base seed give equal outcomes, for any worker count and for
        headless or visual mode; only ``elapsed_s`` differs.

    Raises:
        HazardSpecError: If a hand-built hazard has invalid values.
        HazardOutOfRoomError: If a hazard bound lies outside the room.
        HazardRangeError: If a sampled position never leaves the walls.
        MalformedScenarioError: If the room cannot be loaded.
    """
    started = time.monotonic()
    floor_plan, _ = load_bundled_scenario(config.scenario)
    for spec in config.hazards:
        validate_spec(spec)
    check_specs_in_room(config.hazards, floor_plan)
    base_seed, was_random = resolve_base_seed(config.seed)
    mask = floor_plan.walkable_mask(GRID_CELL_SIZE)
    tasks = [
        RunTask(
            config.scenario, config.agents, index, seed,
            sample_events(config.hazards, mask, base_seed, index),
        )
        for index, seed in enumerate(_spawn_seeds(config, base_seed))
    ]
    outcomes = _execute(
        config, tasks, listener, should_cancel or (lambda: False)
    )
    ordered = tuple(sorted(outcomes, key=lambda o: o.run_index))
    return MonteCarloResult(
        config=config,
        base_seed=base_seed,
        seed_was_random=was_random,
        outcomes=ordered,
        summary=summarize(ordered),
        curves=build_curves(ordered),
        cancelled=len(ordered) < len(tasks),
        elapsed_s=time.monotonic() - started,
    )
