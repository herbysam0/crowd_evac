"""Command line of ``scripts/monte_carlo.py``.

Usage::

    python scripts/monte_carlo.py
    python scripts/monte_carlo.py -s lecture_hall --agents 150 --runs 50 --seed 7
    python scripts/monte_carlo.py --agents 150 --hazard x=10..40,y=2..28,tick=0..400
    python scripts/monte_carlo.py --runs 5 --visual
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from crowd_evac.adapters.io.scenario_catalog import list_bundled_scenarios
from crowd_evac.application.cli_args import checked, int_at_least
from crowd_evac.application.hazard_args import (
    HazardOutOfRoomError,
    HazardSpecError,
)
from crowd_evac.montecarlo.config import (
    DEFAULT_MC_AGENTS,
    DEFAULT_MC_RUNS,
    DEFAULT_MC_SCENARIO,
    MonteCarloConfig,
)
from crowd_evac.montecarlo.hazards import HazardRangeError, parse_hazard_spec
from crowd_evac.montecarlo.report import format_report
from crowd_evac.montecarlo.runner import run_monte_carlo

_MIN_COUNT: int = 1
_MIN_SEED: int = 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``monte_carlo.py`` argument parser.

    Returns:
        The configured parser.
    """
    available = list_bundled_scenarios()
    parser = argparse.ArgumentParser(
        prog="monte_carlo.py",
        description="Run a Monte Carlo batch of evacuation simulations.",
    )
    parser.add_argument(
        "-s", "--scenario", default=DEFAULT_MC_SCENARIO, choices=available,
        metavar="NAME",
        help=f"bundled room (default: {DEFAULT_MC_SCENARIO}); "
        f"one of: {', '.join(available)}",
    )
    parser.add_argument(
        "--agents", type=checked(int_at_least(_MIN_COUNT)),
        default=DEFAULT_MC_AGENTS, metavar="N",
        help=f"persons per run, >= 1 (default: {DEFAULT_MC_AGENTS})",
    )
    parser.add_argument(
        "--runs", type=checked(int_at_least(_MIN_COUNT)),
        default=DEFAULT_MC_RUNS, metavar="N",
        help=f"number of runs, >= 1 (default: {DEFAULT_MC_RUNS})",
    )
    parser.add_argument(
        "--visual", action="store_true",
        help="play the runs in a window (default: headless)",
    )
    parser.add_argument(
        "--seed", type=checked(int_at_least(_MIN_SEED)), default=None,
        metavar="S",
        help="base seed, >= 0 (default: random, printed in the report)",
    )
    parser.add_argument(
        "--hazard", type=checked(parse_hazard_spec), action="append",
        default=None, metavar="K=V,...",
        help="hazard, repeatable: x=,y= required; x, y and tick take a "
        "value or lo..hi (sampled per run); optional intensity=, radius=, "
        "decay_rate=, block_radius=, blocks_navigation=, source_type=",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> MonteCarloConfig:
    """Parse the command line into a config.

    Args:
        argv: Arguments excluding the program name; ``None`` reads
            ``sys.argv``.

    Returns:
        The batch config.  Exits with status 2 (argparse) on invalid values.
    """
    args = build_parser().parse_args(argv)
    return MonteCarloConfig(
        scenario=args.scenario,
        agents=args.agents,
        runs=args.runs,
        visual=args.visual,
        seed=args.seed,
        hazards=tuple(args.hazard or ()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a batch and print the report.

    Args:
        argv: Arguments excluding the program name; ``None`` reads
            ``sys.argv``.

    Returns:
        Exit status 0.  Hazards the room cannot take exit with status 2.
    """
    config = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_monte_carlo(config)
    except (HazardSpecError, HazardOutOfRoomError, HazardRangeError) as exc:
        build_parser().error(str(exc))
    # CLI output is user-facing, not diagnostic logging.
    print(format_report(result), end="")
    return 0
