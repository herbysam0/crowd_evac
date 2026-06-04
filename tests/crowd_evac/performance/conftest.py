"""Shared fixtures for crowd_evac performance tests.

Provides a factory fixture that builds a :class:`Simulation` on a
synthetic benchmark floor (30 m × 30 m, single east exit at low
throughput capacity). All five force terms are enabled by default.

Floor details
-------------
- **Size:** 30 m × 30 m open room — no internal walls or obstacles.
- **Exit:** Single east-wall doorway, 6 m wide, capacity 2 agents/s.
  Low capacity keeps agents active throughout all timed ticks so
  per-tick cost consistently reflects the full N-agent population
  rather than a shrinking one.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import pytest

from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Benchmark floor constants
# ---------------------------------------------------------------------------

FLOOR_WIDTH_M: float = 30.0
FLOOR_HEIGHT_M: float = 30.0

# Low throughput keeps the population stable across the timed window.
EXIT_CAPACITY_PER_S: int = 2

DEFAULT_BENCH_SEED: int = 42

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

#: Callable produced by :func:`build_benchmark_sim`: ``(n_agents, seed=42) -> Simulation``.
SimFactory: TypeAlias = Callable[..., Simulation]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def build_benchmark_sim() -> SimFactory:
    """Return a factory that creates benchmark :class:`Simulation` instances.

    The returned factory accepts ``n_agents: int`` (required) and an
    optional ``seed: int`` (default 42).  Each call creates a fresh
    :class:`FloorPlan`, :class:`FlowField`, agent population, and
    :class:`Simulation` with all five force terms enabled.

    Returns:
        Callable ``(n_agents, seed=42) -> Simulation``.

    Example:
        >>> sim = build_benchmark_sim(n_agents=500)
        >>> sim.step()
    """

    def _make(n_agents: int, seed: int = DEFAULT_BENCH_SEED) -> Simulation:
        """Build a fresh benchmark Simulation with n_agents."""
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

    return _make
