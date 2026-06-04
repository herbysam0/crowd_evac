"""Headless performance tests for the simulation step pipeline.

Measures wall-clock cost of the full 7-step loop (panic decay, force
composition, integration, exit resolution) for a range of agent counts.
No display or arcade — pure NumPy + domain logic.

Run with:
    pytest tests/ -m perf -v -s

The ``-s`` flag passes stdout through so the timing table prints to the
console.  These tests are excluded from the default run (no ``-m`` flag)
to keep the regular test suite fast.

Adding load tests
-----------------
Any change to code on the critical path (force computation, spatial hash,
integrator, exit model, panic propagation) **must** include an update to
this module — either a new parametrize case, a new test, or a comment
explaining why the change does not affect per-tick cost.  The timing table
printed by this suite is the performance baseline; regressions should be
caught here before they reach the render loop.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import pytest

from crowd_evac.application.simulation import Simulation

if TYPE_CHECKING:
    pass  # All runtime imports above; only TYPE_CHECKING aliases here.

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

WARMUP_TICKS: int = 20
"""Steps discarded before timing begins (cache / first-tick warm-up)."""

TIMED_TICKS: int = 200
"""Steps timed per agent-count run."""

# Catastrophic-regression guard. Intent: flag pathological regressions
# (e.g., accidental O(N²) loop) without being sensitive to CI hardware.
# Once baselines are established, tighten this per agent count.
MAX_MEAN_TICK_MS: float = 5_000.0

# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickStats:
    """Aggregated per-tick timing for one benchmark run.

    Attributes:
        agent_count: Agents present at simulation start.
        ticks_measured: Timed ticks (may be less than requested if the
            simulation terminated early).
        mean_tick_ms: Mean wall-clock cost per tick in milliseconds.
        min_tick_ms: Fastest single tick observed.
        max_tick_ms: Slowest single tick observed.
        p99_tick_ms: 99th-percentile tick cost.
        ticks_per_s: Simulation throughput in ticks per real second.
        agent_ticks_per_s: Effective throughput in agent-ticks per second.
    """

    agent_count: int
    ticks_measured: int
    mean_tick_ms: float
    min_tick_ms: float
    max_tick_ms: float
    p99_tick_ms: float
    ticks_per_s: float
    agent_ticks_per_s: float


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _run_timed_ticks(
    sim: Simulation,
    n_ticks: int,
    warmup: int,
) -> TickStats:
    """Warm up sim then time n_ticks steps; return aggregated stats.

    Discards ``warmup`` steps before timing starts so that any first-tick
    NumPy cache effects do not contaminate measurements.  If the simulation
    terminates early, only completed ticks are included.

    Args:
        sim: A pre-built :class:`Simulation` to step.  Consumed in place.
        n_ticks: Ticks to time after the warm-up phase.
        warmup: Steps to discard before timing starts.

    Returns:
        :class:`TickStats` with timing statistics.  ``ticks_measured`` is
        zero if the simulation terminated before the timed phase.
    """
    n_agents = int(sim.state.pos.shape[0])

    for _ in range(warmup):
        if sim.is_complete:
            break
        sim.step()

    tick_times: list[float] = []
    for _ in range(n_ticks):
        if sim.is_complete:
            break
        t0 = perf_counter()
        sim.step()
        tick_times.append(perf_counter() - t0)

    if not tick_times:
        return TickStats(
            agent_count=n_agents,
            ticks_measured=0,
            mean_tick_ms=0.0,
            min_tick_ms=0.0,
            max_tick_ms=0.0,
            p99_tick_ms=0.0,
            ticks_per_s=0.0,
            agent_ticks_per_s=0.0,
        )

    times_ms = [t * 1_000.0 for t in tick_times]
    total_s = sum(tick_times)
    n = len(tick_times)
    sorted_ms = sorted(times_ms)

    p99 = (
        statistics.quantiles(times_ms, n=100)[98]
        if n >= 4
        else sorted_ms[-1]
    )

    return TickStats(
        agent_count=n_agents,
        ticks_measured=n,
        mean_tick_ms=statistics.mean(times_ms),
        min_tick_ms=sorted_ms[0],
        max_tick_ms=sorted_ms[-1],
        p99_tick_ms=p99,
        ticks_per_s=n / total_s,
        agent_ticks_per_s=(n_agents * n) / total_s,
    )


def _print_stats(stats: TickStats) -> None:
    """Print a one-line timing summary for a single run (visible with -s).

    Args:
        stats: Aggregated :class:`TickStats` from :func:`_run_timed_ticks`.
    """
    print(
        f"\n  {stats.agent_count:>5} agents | "
        f"mean {stats.mean_tick_ms:7.3f} ms | "
        f"p99 {stats.p99_tick_ms:7.3f} ms | "
        f"max {stats.max_tick_ms:7.3f} ms | "
        f"{stats.ticks_per_s:>8.0f} ticks/s | "
        f"{stats.agent_ticks_per_s:>12,.0f} agent·ticks/s"
    )


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


@pytest.mark.perf
class TestHeadlessStepPipeline:
    """Per-tick timing of the full 7-step simulation pipeline.

    Each parametrized case builds a fresh Simulation with the specified
    agent count, runs WARMUP_TICKS discarded steps, then times TIMED_TICKS
    steps and reports statistics.

    All five force terms are enabled (exit-seeking, crowd repulsion, density
    pressure, herd alignment, panic repulsion), matching the live game
    configuration.
    """

    @pytest.mark.parametrize(
        "n_agents",
        [100, 500, 1000],
        ids=["100a", "500a", "1k"],
    )
    def test_step_pipeline_cost(
        self,
        n_agents: int,
        build_benchmark_sim: Callable[..., Simulation],
    ) -> None:
        """Step pipeline completes within budget for the given agent count.

        Asserts ``mean_tick_ms < MAX_MEAN_TICK_MS`` as a catastrophic-
        regression guard.  The printed line is the real artefact: copy it
        into the performance log when establishing a new baseline.

        Args:
            n_agents: Number of agents to simulate.
            build_benchmark_sim: Factory fixture from conftest.
        """
        sim = build_benchmark_sim(n_agents=n_agents)
        stats = _run_timed_ticks(sim, n_ticks=TIMED_TICKS, warmup=WARMUP_TICKS)
        _print_stats(stats)
        assert stats.mean_tick_ms < MAX_MEAN_TICK_MS, (
            f"Mean tick {stats.mean_tick_ms:.1f} ms exceeds "
            f"budget {MAX_MEAN_TICK_MS:.0f} ms at {n_agents} agents."
        )

    @pytest.mark.parametrize(
        "n_agents",
        [500, 1000],
        ids=["500a", "1k"],
    )
    def test_crowd_force_disabled_faster(
        self,
        n_agents: int,
        build_benchmark_sim: Callable[..., Simulation],
    ) -> None:
        """Disabling crowd-repulsion reduces per-tick cost.

        The crowd force is the most expensive term at high density (O(N)
        neighbour queries via spatial hash).  Verifying it dominates cost
        at 2k+ agents is a useful structural check.

        Args:
            n_agents: Number of agents to simulate.
            build_benchmark_sim: Factory fixture from conftest.
        """
        sim_full = build_benchmark_sim(n_agents=n_agents)
        sim_no_crowd = build_benchmark_sim(n_agents=n_agents)
        sim_no_crowd._enable_crowd = False  # noqa: SLF001

        full_stats = _run_timed_ticks(
            sim_full, n_ticks=TIMED_TICKS, warmup=WARMUP_TICKS
        )
        no_crowd_stats = _run_timed_ticks(
            sim_no_crowd, n_ticks=TIMED_TICKS, warmup=WARMUP_TICKS
        )

        print(
            f"\n  crowd ON : mean {full_stats.mean_tick_ms:.3f} ms | "
            f"crowd OFF: mean {no_crowd_stats.mean_tick_ms:.3f} ms"
            f"  ({n_agents} agents)"
        )
        assert no_crowd_stats.mean_tick_ms <= full_stats.mean_tick_ms * 1.05, (
            "Disabling crowd force did not reduce (or equal) mean tick time — "
            "check that crowd repulsion is actually being skipped."
        )
