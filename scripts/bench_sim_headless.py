"""Headless simulation benchmark — pure CPU-bound step cost measurement.

Runs the simulation without any rendering, collecting raw step times
for profiling and optimization work.

Usage (PowerShell, .venv activated):
    python scripts/bench_sim_headless.py --agents 500 --ticks 200
    python scripts/bench_sim_headless.py --agents 2000 --ticks 100
    python -m cProfile -o prof.out scripts/bench_sim_headless.py --agents 2000 --ticks 100
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from time import perf_counter

from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.pathfinding.flow_field import FlowField

# Benchmark defaults (overridden by CLI)
DEFAULT_AGENT_COUNT: int = 500
DEFAULT_TICKS: int = 200
DEFAULT_WARMUP_TICKS: int = 20
DEFAULT_SEED: int = 42

# Benchmark floor — identical to bench_sim_render.py for comparability
FLOOR_WIDTH_M: float = 30.0
FLOOR_HEIGHT_M: float = 30.0
EXIT_CAPACITY_PER_S: int = 2


@dataclass(frozen=True)
class SimHeadlessConfig:
    """Parameters for a headless sim benchmark run.

    Attributes:
        agent_count: Number of agents to simulate.
        ticks: Simulation ticks to measure before exit.
        warmup_ticks: Ticks discarded before timing starts.
        seed: RNG seed for reproducibility.
    """

    agent_count: int = DEFAULT_AGENT_COUNT
    ticks: int = DEFAULT_TICKS
    warmup_ticks: int = DEFAULT_WARMUP_TICKS
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class SimHeadlessStats:
    """Aggregated timing from a headless simulation run.

    All times are in milliseconds.

    Attributes:
        tick_count: Timed ticks collected.
        mean_ms: Mean Simulation.step() cost per tick.
        p99_ms: 99th-percentile step cost.
        min_ms: Minimum step cost.
        max_ms: Maximum step cost.
        total_ms: Total time for all timed ticks.
    """

    tick_count: int
    mean_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    total_ms: float


def _build_simulation(n_agents: int, seed: int) -> Simulation:
    """Construct a benchmark Simulation on the standard 30 m × 30 m floor.

    Args:
        n_agents: Number of agents to spawn uniformly in the walkable region.
        seed: RNG seed for reproducibility.

    Returns:
        A fully wired :class:`~crowd_evac.application.simulation.Simulation`
        with all five force terms enabled.
    """
    floor = FloorPlan(
        width_m=FLOOR_WIDTH_M,
        height_m=FLOOR_HEIGHT_M,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=FLOOR_WIDTH_M,
                y=FLOOR_HEIGHT_M / 2.0,
                width_m=6.0,
                side=ExitSide.EAST,
                capacity_per_second=EXIT_CAPACITY_PER_S,
                label="east",
            ),
        ),
    )
    flow = FlowField.build(floor)
    rng = SeededRNG(seed)
    state = spawn(floor, n_agents, rng.generator)
    return Simulation(
        state=state,
        flow_field=flow,
        panic_field=PanicField(),
        exit_model=ExitModel(floor),
        rng=rng,
    )


def run_headless_bench(config: SimHeadlessConfig) -> SimHeadlessStats:
    """Run the headless benchmark and return timing statistics.

    Args:
        config: Benchmark parameters.

    Returns:
        :class:`SimHeadlessStats` with step timing information.
    """
    sim = _build_simulation(config.agent_count, config.seed)
    tick_times: list[float] = []
    total_ticks = config.warmup_ticks + config.ticks

    for tick in range(total_ticks):
        t0 = perf_counter()
        sim.step()
        elapsed_ms = (perf_counter() - t0) * 1_000.0

        if tick >= config.warmup_ticks:
            tick_times.append(elapsed_ms)

        if sim.is_complete:
            break

    if not tick_times:
        raise ValueError(
            "No timed ticks collected — simulation may have terminated "
            "during the warmup phase."
        )

    def _p99(vals: list[float]) -> float:
        """Return the 99th-percentile value."""
        return (
            statistics.quantiles(vals, n=100)[98]
            if len(vals) >= 4
            else max(vals)
        )

    return SimHeadlessStats(
        tick_count=len(tick_times),
        mean_ms=statistics.mean(tick_times),
        p99_ms=_p99(tick_times),
        min_ms=min(tick_times),
        max_ms=max(tick_times),
        total_ms=sum(tick_times),
    )


def format_report(stats: SimHeadlessStats, config: SimHeadlessConfig) -> str:
    """Render a human-readable benchmark report.

    Args:
        stats: Aggregated timing from :func:`run_headless_bench`.
        config: Benchmark config.

    Returns:
        Multi-line report string (no trailing newline).
    """
    return (
        f"headless sim benchmark @ {config.agent_count} agents\n"
        f"  ticks timed       : {stats.tick_count}\n"
        f"  total time        : {stats.total_ms:8.1f} ms\n"
        f"  mean step ms      : {stats.mean_ms:8.3f}\n"
        f"  p99  step ms      : {stats.p99_ms:8.3f}\n"
        f"  min  step ms      : {stats.min_ms:8.3f}\n"
        f"  max  step ms      : {stats.max_ms:8.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the headless benchmark CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="bench_sim_headless",
        description=(
            "Headless simulation benchmark — "
            "measures pure step() cost without rendering."
        ),
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=DEFAULT_AGENT_COUNT,
        help=f"Number of agents to simulate (default: {DEFAULT_AGENT_COUNT}).",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_TICKS,
        help=f"Timed ticks to collect before exit (default: {DEFAULT_TICKS}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_TICKS,
        help=(
            f"Ticks discarded before timing starts "
            f"(default: {DEFAULT_WARMUP_TICKS})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the benchmark, and print the report.

    Args:
        argv: Argument vector excluding the program name.  Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)
    config = SimHeadlessConfig(
        agent_count=args.agents,
        ticks=args.ticks,
        warmup_ticks=args.warmup,
        seed=args.seed,
    )
    stats = run_headless_bench(config)
    print(format_report(stats, config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
