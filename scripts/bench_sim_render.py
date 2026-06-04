"""Simulation + render load test — per-frame cost with the full step pipeline.

Opens an arcade window, runs one full ``Simulation.step()`` call per frame,
then renders live agent positions through a :class:`arcade.SpriteList`.
Sim-step time and draw time are measured and reported separately so you
can attribute frame-budget cost to each phase.

Requires a display.  Run on the target machine — not in a headless server.

Usage (PowerShell, .venv activated):
    python scripts/bench_sim_render.py
    python scripts/bench_sim_render.py --agents 1000 --frames 300
    python scripts/bench_sim_render.py --agents 2000 --frames 200 --seed 7
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from time import perf_counter

import arcade

from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation, SimSnapshot
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Window and rendering constants
# ---------------------------------------------------------------------------

WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
AGENT_RADIUS_PX: int = 4
FLOOR_MARGIN_PX: int = 30

BACKGROUND_COLOR: arcade.types.RGBA255 = (24, 24, 28, 255)
ALIVE_COLOR: arcade.types.RGBA255 = (40, 120, 220, 200)

# Rate small enough to remove arcade's built-in FPS cap so the benchmark
# measures raw throughput, not an artificial ceiling.
UNCAPPED_RATE: float = 1.0 / 10_000.0

# ---------------------------------------------------------------------------
# Benchmark defaults (overridden by CLI)
# ---------------------------------------------------------------------------

DEFAULT_AGENT_COUNT: int = 500
DEFAULT_MAX_FRAMES: int = 300
DEFAULT_WARMUP_FRAMES: int = 30
DEFAULT_SEED: int = 42

# ---------------------------------------------------------------------------
# Benchmark floor — identical dimensions to the headless perf test so that
# step costs are directly comparable between the two scripts.
# ---------------------------------------------------------------------------

FLOOR_WIDTH_M: float = 30.0
FLOOR_HEIGHT_M: float = 30.0

# Low throughput keeps agents active so per-frame cost reflects N agents.
EXIT_CAPACITY_PER_S: int = 2


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimRenderConfig:
    """Parameters for a sim + render benchmark run.

    Attributes:
        agent_count: Number of agents to simulate and render.
        max_frames: Timed frames to collect before auto-close.
        warmup_frames: Frames discarded before timing starts (shader warmup).
        seed: RNG seed for reproducibility.
    """

    agent_count: int = DEFAULT_AGENT_COUNT
    max_frames: int = DEFAULT_MAX_FRAMES
    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class SimRenderStats:
    """Aggregated timing from a combined simulation + render run.

    All times are in milliseconds.

    Attributes:
        frame_count: Timed frames collected.
        mean_step_ms: Mean ``Simulation.step()`` + ``snapshot()`` cost.
        p99_step_ms: 99th-percentile step cost.
        mean_draw_ms: Mean sprite-position update + arcade draw call cost.
        p99_draw_ms: 99th-percentile draw cost.
        mean_frame_ms: Mean total frame cost (step + draw).
        p99_frame_ms: 99th-percentile total frame cost.
        mean_fps: Frames per second derived from mean frame time.
        p1_low_fps: 1%-low FPS (mean of the slowest 1% of frames).
    """

    frame_count: int
    mean_step_ms: float
    p99_step_ms: float
    mean_draw_ms: float
    p99_draw_ms: float
    mean_frame_ms: float
    p99_frame_ms: float
    mean_fps: float
    p1_low_fps: float


# ---------------------------------------------------------------------------
# Simulation factory
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Arcade window
# ---------------------------------------------------------------------------


class SimBenchWindow(arcade.Window):
    """Arcade window that steps the simulation and renders agents each frame.

    Timing is split into two phases per frame:

    * **step_ms** — ``Simulation.step()`` + ``Simulation.snapshot()``
      (CPU cost of advancing physics and copying state for rendering).
    * **draw_ms** — sprite position update loop + the arcade draw call
      (CPU cost of submitting geometry to the GPU pipeline).

    After ``config.warmup_frames`` frames the window enters the timed
    phase.  It auto-closes once ``config.max_frames`` timed frames are
    collected or the simulation evacuates all agents.

    Attributes:
        step_times: Per-frame step durations in milliseconds (timed phase only).
        draw_times: Per-frame draw durations in milliseconds (timed phase only).
    """

    def __init__(self, config: SimRenderConfig) -> None:
        """Initialise the window, simulation, and pre-allocated sprite list.

        Args:
            config: Benchmark parameters (agent count, frame caps, seed).
        """
        super().__init__(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            f"crowd_evac sim+render bench @ {config.agent_count} agents",
            vsync=False,
            update_rate=UNCAPPED_RATE,
            draw_rate=UNCAPPED_RATE,
        )
        self.background_color = BACKGROUND_COLOR
        self._config = config

        # Scale: fit the floor into the window with a fixed pixel margin.
        usable_w = float(WINDOW_WIDTH - 2 * FLOOR_MARGIN_PX)
        usable_h = float(WINDOW_HEIGHT - 2 * FLOOR_MARGIN_PX)
        self._scale: float = min(usable_w / FLOOR_WIDTH_M, usable_h / FLOOR_HEIGHT_M)

        self._sim = _build_simulation(config.agent_count, config.seed)

        # Pre-allocate one sprite per agent.  Egressed agents move off-screen
        # rather than being removed, keeping the draw list size constant.
        self._sprites: arcade.SpriteList[arcade.SpriteCircle] = arcade.SpriteList()
        for _ in range(config.agent_count):
            self._sprites.append(arcade.SpriteCircle(AGENT_RADIUS_PX, ALIVE_COLOR))

        self.step_times: list[float] = []
        self.draw_times: list[float] = []

        self._total_frames: int = 0
        self._snap: SimSnapshot | None = None

    def on_update(self, delta_time: float) -> None:  # noqa: ARG002
        """Advance the simulation by one tick and record the step cost.

        Args:
            delta_time: Real seconds since last update (unused; the
                simulation advances by a fixed DT per step).
        """
        if self._sim.is_complete or len(self.step_times) >= self._config.max_frames:
            self.close()
            return

        t0 = perf_counter()
        self._sim.step()
        self._snap = self._sim.snapshot()
        step_ms = (perf_counter() - t0) * 1_000.0

        self._total_frames += 1
        if self._total_frames > self._config.warmup_frames:
            self.step_times.append(step_ms)

    def on_draw(self) -> None:
        """Update sprite positions from the last snapshot and draw them."""
        t0 = perf_counter()
        self.clear()

        snap = self._snap
        if snap is not None:
            for i, sprite in enumerate(self._sprites):
                if snap.alive[i]:
                    sprite.center_x = (
                        FLOOR_MARGIN_PX + snap.positions[i, 0] * self._scale
                    )
                    sprite.center_y = (
                        FLOOR_MARGIN_PX + snap.positions[i, 1] * self._scale
                    )
                else:
                    # Move off-screen; cost still charged to maintain N sprites.
                    sprite.center_x = float(-AGENT_RADIUS_PX * 4)
                    sprite.center_y = float(-AGENT_RADIUS_PX * 4)

        self._sprites.draw()
        draw_ms = (perf_counter() - t0) * 1_000.0

        if self._total_frames > self._config.warmup_frames:
            self.draw_times.append(draw_ms)


# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------


def aggregate_stats(window: SimBenchWindow) -> SimRenderStats:
    """Compute :class:`SimRenderStats` from a completed window run.

    Args:
        window: A :class:`SimBenchWindow` after its run has finished.

    Returns:
        :class:`SimRenderStats` summarising step and draw costs.

    Raises:
        ValueError: If no timed frames were collected.
    """
    if not window.step_times:
        raise ValueError(
            "No timed frames collected — simulation may have terminated "
            "during the warmup phase."
        )

    step_ms = window.step_times
    draw_ms = window.draw_times
    n = len(step_ms)
    frame_ms = [s + d for s, d in zip(step_ms, draw_ms)]

    def _p99(vals: list[float]) -> float:
        """Return the 99th-percentile value."""
        return statistics.quantiles(vals, n=100)[98] if len(vals) >= 4 else max(vals)

    n_low = max(1, n // 100)
    slowest = sorted(frame_ms, reverse=True)[:n_low]
    p1_low_fps = n_low / max(sum(f / 1_000.0 for f in slowest), 1e-9)

    return SimRenderStats(
        frame_count=n,
        mean_step_ms=statistics.mean(step_ms),
        p99_step_ms=_p99(step_ms),
        mean_draw_ms=statistics.mean(draw_ms),
        p99_draw_ms=_p99(draw_ms),
        mean_frame_ms=statistics.mean(frame_ms),
        p99_frame_ms=_p99(frame_ms),
        mean_fps=1_000.0 / statistics.mean(frame_ms),
        p1_low_fps=p1_low_fps,
    )


# ---------------------------------------------------------------------------
# Runner and formatter
# ---------------------------------------------------------------------------


def run_sim_render_bench(config: SimRenderConfig) -> SimRenderStats:
    """Open the benchmark window, run it to completion, and return stats.

    Args:
        config: Benchmark parameters.

    Returns:
        :class:`SimRenderStats` from the completed run.
    """
    window = SimBenchWindow(config)
    arcade.run()
    return aggregate_stats(window)


def format_report(stats: SimRenderStats, config: SimRenderConfig) -> str:
    """Render a human-readable benchmark report for CLI output.

    Args:
        stats: Aggregated timing from :func:`run_sim_render_bench`.
        config: Benchmark config (used for the header line).

    Returns:
        Multi-line report string (no trailing newline).
    """
    return (
        f"sim + render benchmark @ {config.agent_count} agents\n"
        f"  frames timed      : {stats.frame_count}\n"
        f"  --- sim step (step() + snapshot()) ---\n"
        f"  mean step ms      : {stats.mean_step_ms:8.3f}\n"
        f"  p99  step ms      : {stats.p99_step_ms:8.3f}\n"
        f"  --- arcade draw (sprite pos update + draw call) ---\n"
        f"  mean draw ms      : {stats.mean_draw_ms:8.3f}\n"
        f"  p99  draw ms      : {stats.p99_draw_ms:8.3f}\n"
        f"  --- combined frame ---\n"
        f"  mean frame ms     : {stats.mean_frame_ms:8.3f}\n"
        f"  p99  frame ms     : {stats.p99_frame_ms:8.3f}\n"
        f"  mean FPS          : {stats.mean_fps:8.1f}\n"
        f"  1% low FPS        : {stats.p1_low_fps:8.1f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the sim + render benchmark CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="bench_sim_render",
        description=(
            "Simulation + render load test — "
            "measures sim step and draw cost per frame."
        ),
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=DEFAULT_AGENT_COUNT,
        help=f"Number of agents to simulate and render (default: {DEFAULT_AGENT_COUNT}).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Timed frames to collect before auto-close (default: {DEFAULT_MAX_FRAMES}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help=(
            f"Frames discarded before timing starts (shader warmup) "
            f"(default: {DEFAULT_WARMUP_FRAMES})."
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
    config = SimRenderConfig(
        agent_count=args.agents,
        max_frames=args.frames,
        warmup_frames=args.warmup,
        seed=args.seed,
    )
    stats = run_sim_render_bench(config)
    print(format_report(stats, config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
