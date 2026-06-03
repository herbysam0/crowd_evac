"""Render-scale benchmark CLI for the RK-1 gate (Step 1.3).

Opens the arcade render spike, animates ``--agents`` agents with no
simulation, and prints sustained-FPS statistics. Run on the *target laptop*
to decide whether arcade clears the NFR-P2 bar (≈30 FPS at 10k) or the
pyglet/moderngl fallback (PRD §9) is needed.

Usage (PowerShell, .venv activated):
    python scripts/bench_render.py --agents 2000
    python scripts/bench_render.py --agents 10000

This is a script entry point, so ``print`` is the intended output channel.
All reusable logic lives in ``crowd_evac.adapters.render.spike`` and is
unit-tested there; this file is only argument parsing plus a run + print.
"""
from __future__ import annotations

import argparse
import sys

from crowd_evac.adapters.render.spike import (
    DEFAULT_AGENT_COUNT,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_SECONDS,
    DEFAULT_SEED,
    SpikeConfig,
    format_report,
    run_spike,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the benchmark CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="bench_render",
        description="Measure arcade render FPS at scale (RK-1 gate).",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=DEFAULT_AGENT_COUNT,
        help="Number of agents to draw (default: %(default)s).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Timed frames to collect (default: %(default)s).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Wall-clock cap in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for agent layout/velocities (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the spike, and print the report.

    Args:
        argv: Argument vector excluding the program name. Defaults to
            ``sys.argv[1:]`` when None.

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)
    config = SpikeConfig(
        agent_count=args.agents,
        max_frames=args.frames,
        max_seconds=args.seconds,
        seed=args.seed,
    )
    stats = run_spike(config)
    print(format_report(stats, config.agent_count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
