"""Render-scale spike for the RK-1 gate (Step 1.3, NFR-P2).

Throwaway-but-kept benchmark that draws ``N`` instanced agents (random
positions, animated, no simulation) through the ``arcade`` adapter and
measures sustained render FPS. Its purpose is to de-risk the iGPU-at-scale
risk *before* the real renderer (Step 1.16) is built: if ``arcade`` cannot
approach 30 FPS at 10k agents on the target laptop, the ``pyglet``/``moderngl``
fallback (PRD §9) is chosen here, while a pivot is still cheap.

The module is import-safe headless: importing it never opens a window or
touches the GPU. ``run_spike`` (which does) is invoked only from
``scripts/bench_render.py`` on the target laptop. The pure timing/aggregation
helper :func:`summarize_frame_times` is unit-tested; FPS itself is measured
manually.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import arcade
import numpy as np

from crowd_evac.application.rng import SeededRNG

# -- Spike defaults (display-only; not domain constants) -------------------

DEFAULT_WINDOW_WIDTH: int = 1280
DEFAULT_WINDOW_HEIGHT: int = 720
DEFAULT_AGENT_COUNT: int = 2000
DEFAULT_AGENT_RADIUS_PX: int = 3
DEFAULT_AGENT_SPEED_PX_PER_S: float = 120.0
DEFAULT_SEED: int = 1234

DEFAULT_WARMUP_FRAMES: int = 30
"""Frames discarded before timing, to skip first-frame shader/buffer warmup."""

DEFAULT_MAX_FRAMES: int = 600
"""Timed frames to collect before the spike auto-closes."""

DEFAULT_MAX_SECONDS: float = 20.0
"""Wall-clock safety cap so the spike always terminates."""

UNCAPPED_DRAW_RATE: float = 1.0 / 10000.0
"""``draw_rate`` small enough to remove arcade's cap (measure raw FPS)."""

LOW_PERCENTILE_DIVISOR: int = 100
"""Denominator for the "1% low" FPS metric (slowest count // 100 frames)."""

AGENT_COLOR: arcade.types.RGBA255 = (40, 120, 220, 255)
BACKGROUND_COLOR: arcade.types.RGBA255 = (24, 24, 28, 255)


@dataclass(frozen=True)
class FrameStats:
    """Aggregated render-timing result for a spike run.

    Attributes:
        frame_count: Number of timed frames (excludes warmup).
        duration_s: Summed wall-clock time of the timed frames, in seconds.
        mean_fps: Frames per second over the whole run (count / duration).
        min_fps: FPS of the single slowest frame.
        max_fps: FPS of the single fastest frame.
        p1_low_fps: Mean FPS across the slowest 1% of frames (stutter floor).
        mean_frame_ms: Mean frame time in milliseconds.
    """

    frame_count: int
    duration_s: float
    mean_fps: float
    min_fps: float
    max_fps: float
    p1_low_fps: float
    mean_frame_ms: float


@dataclass(frozen=True)
class SpikeConfig:
    """Parameters for a single render-spike run.

    Attributes:
        agent_count: Number of animated agents to draw.
        width: Window width in pixels.
        height: Window height in pixels.
        agent_radius: Agent sprite radius in pixels.
        agent_speed: Agent speed magnitude in pixels per second.
        warmup_frames: Leading frames excluded from timing.
        max_frames: Timed frames to collect before auto-close.
        max_seconds: Wall-clock cap before auto-close.
        seed: Seed for reproducible agent layout and velocities.
        title: Window title.
    """

    agent_count: int = DEFAULT_AGENT_COUNT
    width: int = DEFAULT_WINDOW_WIDTH
    height: int = DEFAULT_WINDOW_HEIGHT
    agent_radius: int = DEFAULT_AGENT_RADIUS_PX
    agent_speed: float = DEFAULT_AGENT_SPEED_PX_PER_S
    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    max_frames: int = DEFAULT_MAX_FRAMES
    max_seconds: float = DEFAULT_MAX_SECONDS
    seed: int = DEFAULT_SEED
    title: str = "crowd_evac render spike"


def summarize_frame_times(frame_times_s: Sequence[float]) -> FrameStats:
    """Aggregate per-frame durations into FPS statistics.

    Args:
        frame_times_s: Per-frame wall-clock durations in seconds. Must be
            non-empty and strictly positive.

    Returns:
        A :class:`FrameStats` summarizing mean/min/max/1%-low FPS.

    Raises:
        ValueError: If ``frame_times_s`` is empty or contains a non-positive
            duration.

    Example:
        >>> stats = summarize_frame_times([0.01, 0.01, 0.02, 0.02])
        >>> round(stats.mean_fps, 2)
        66.67
    """
    times = list(frame_times_s)
    if not times:
        raise ValueError("frame_times_s must be non-empty")
    if any(t <= 0.0 for t in times):
        raise ValueError("all frame times must be strictly positive")

    count = len(times)
    total = sum(times)
    fps_values = [1.0 / t for t in times]

    # 1% low: mean FPS over the slowest frames (at least one), a standard
    # stutter-floor metric that ignores best-case spikes.
    n_low = max(1, count // LOW_PERCENTILE_DIVISOR)
    slowest = sorted(times, reverse=True)[:n_low]

    return FrameStats(
        frame_count=count,
        duration_s=total,
        mean_fps=count / total,
        min_fps=min(fps_values),
        max_fps=max(fps_values),
        p1_low_fps=n_low / sum(slowest),
        mean_frame_ms=(total / count) * 1000.0,
    )


def format_report(stats: FrameStats, agent_count: int) -> str:
    """Render a human-readable benchmark report for CLI output.

    Args:
        stats: Aggregated timing result from :func:`summarize_frame_times`.
        agent_count: Number of agents the run drew, for the header line.

    Returns:
        A multi-line report string (no trailing newline).
    """
    return (
        f"render spike @ {agent_count} agents\n"
        f"  frames timed : {stats.frame_count}\n"
        f"  mean FPS     : {stats.mean_fps:8.1f}\n"
        f"  1% low FPS   : {stats.p1_low_fps:8.1f}\n"
        f"  min / max FPS: {stats.min_fps:8.1f} / {stats.max_fps:.1f}\n"
        f"  mean frame   : {stats.mean_frame_ms:8.2f} ms"
    )


class RenderSpikeWindow(arcade.Window):
    """Arcade window that animates ``N`` agents and records frame times.

    Agents are placed at seeded-random positions with seeded-random
    velocities and bounce off the window edges. Each drawn frame's duration
    is recorded (after a warmup period); the window auto-closes once enough
    timed frames are collected or the wall-clock cap is hit.

    This class is defined at import time but only constructed by
    :func:`run_spike`; constructing it requires a usable GL context.

    Attributes:
        frame_times: Per-frame durations in seconds, populated as the run
            proceeds (excludes warmup frames).
    """

    def __init__(self, config: SpikeConfig) -> None:
        """Build the window, agent arrays, and sprite list.

        Args:
            config: Spike parameters (agent count, window size, seed, caps).
        """
        super().__init__(
            config.width,
            config.height,
            config.title,
            vsync=False,
            update_rate=UNCAPPED_DRAW_RATE,
            draw_rate=UNCAPPED_DRAW_RATE,
        )
        self.background_color = BACKGROUND_COLOR
        self._config = config
        self.frame_times: list[float] = []

        rng = SeededRNG(config.seed)
        radius = float(config.agent_radius)
        self._min = np.array([radius, radius], dtype=np.float64)
        self._max = np.array(
            [config.width - radius, config.height - radius], dtype=np.float64
        )
        self._pos = rng.draw_uniform(0.0, 1.0, (config.agent_count, 2))
        self._pos = self._min + self._pos * (self._max - self._min)
        heading = rng.draw_uniform(0.0, 2.0 * np.pi, config.agent_count)
        self._vel = np.stack(
            [np.cos(heading), np.sin(heading)], axis=1
        ) * config.agent_speed

        self._sprites: arcade.SpriteList[arcade.SpriteCircle] = (
            arcade.SpriteList()
        )
        self._build_sprites(config)

        self._start: float = perf_counter()
        self._last_draw: float | None = None
        self._drawn: int = 0

    def _build_sprites(self, config: SpikeConfig) -> None:
        """Populate the sprite list with one circle per agent."""
        for x, y in self._pos:
            self._sprites.append(
                arcade.SpriteCircle(
                    config.agent_radius,
                    AGENT_COLOR,
                    center_x=float(x),
                    center_y=float(y),
                )
            )

    def on_update(self, delta_time: float) -> None:
        """Integrate agent positions, bounce off edges, sync to sprites.

        Args:
            delta_time: Seconds since the previous update, from arcade.
        """
        self._pos += self._vel * delta_time
        for axis in (0, 1):
            below = self._pos[:, axis] < self._min[axis]
            above = self._pos[:, axis] > self._max[axis]
            self._vel[below | above, axis] *= -1.0
        np.clip(self._pos, self._min, self._max, out=self._pos)

        for sprite, (x, y) in zip(self._sprites, self._pos):
            sprite.position = (float(x), float(y))

    def on_draw(self) -> None:
        """Clear, draw all agents, record the frame time, request exit.

        Recording skips the first ``warmup_frames`` measured intervals so
        first-frame shader/buffer compilation does not skew the result. The
        loop is stopped with :func:`arcade.exit` rather than ``self.close``:
        closing here would destroy the GL context mid-frame, before pyglet's
        own ``flip`` runs. The window is torn down in :func:`run_spike`.
        """
        self.clear()
        self._sprites.draw()

        now = perf_counter()
        if self._last_draw is not None:
            self._drawn += 1
            if self._drawn > self._config.warmup_frames:
                self.frame_times.append(now - self._last_draw)
        self._last_draw = now

        elapsed = now - self._start
        if (
            len(self.frame_times) >= self._config.max_frames
            or elapsed >= self._config.max_seconds
        ):
            arcade.exit()


def run_spike(config: SpikeConfig) -> FrameStats:
    """Run the render spike to completion and return its timing stats.

    Opens a window, animates ``config.agent_count`` agents, and blocks until
    the window auto-closes. Requires a usable display/GL context, so this is
    called only from the benchmark script on the target laptop — never from
    the headless test suite.

    Args:
        config: Spike parameters.

    Returns:
        Aggregated :class:`FrameStats` for the run.

    Raises:
        ValueError: If the window closed before any frame was timed (for
            example, closed manually during warmup).
    """
    window = RenderSpikeWindow(config)
    try:
        arcade.run()
        frame_times = list(window.frame_times)
    finally:
        window.close()
    return summarize_frame_times(frame_times)
