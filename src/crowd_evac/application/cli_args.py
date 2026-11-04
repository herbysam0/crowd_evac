"""Command-line argument parsing for the ``crowd-evac`` entry point.

Kept apart from :mod:`crowd_evac.application.cli` so argument handling can
grow without pulling arcade into its tests.  Scenario files describe the room
only; the population (``--agents``, ``--seed``) and scripted hazards
(``--hazard``) are run-time input parsed here.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

from crowd_evac.adapters.io.scenario_catalog import list_bundled_scenarios
from crowd_evac.application.hazard_args import parse_hazard
from crowd_evac.domain.constants import DEFAULT_AGENT_COUNT, DEFAULT_SPAWN_SEED

DEFAULT_SCENARIO: str = "lecture_hall"
"""Bundled scenario name loaded when no user file is specified (FR-0 R0.1)."""

MIN_AGENTS: int = 1
"""Smallest population ``--agents`` accepts."""

_RUN_OPTIONS: tuple[str, ...] = ("agents", "seed", "hazard")
_RUN_FLAGS: str = "--agents, --seed or --hazard"


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def checked(
    convert: Callable[[str], Any],
) -> Callable[[str], Any]:
    """Wrap a converter so ``ValueError`` becomes an argparse type error.

    Public so the Monte Carlo CLI shares the same argument checks.
    """

    def wrapper(text: str) -> Any:
        try:
            return convert(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    wrapper.__name__ = getattr(convert, "__name__", "value")
    return wrapper


def int_at_least(minimum: int) -> Callable[[str], int]:
    """Build an integer converter enforcing ``value >= minimum``.

    Public so the Monte Carlo CLI shares the same argument checks.
    """

    def convert(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise ValueError(f"must be >= {minimum}, got {value}")
        return value

    convert.__name__ = "int"
    return convert


def build_parser() -> argparse.ArgumentParser:
    """Build the ``crowd-evac`` argument parser.

    Returns:
        The configured parser.  Run options default to ``None`` so
        :func:`parse_args` can tell whether they were given.
    """
    available = list_bundled_scenarios()
    parser = argparse.ArgumentParser(
        prog="crowd-evac",
        description="Run a crowd evacuation simulation.",
    )
    parser.add_argument(
        "-s", "--scenario",
        default=DEFAULT_SCENARIO,
        choices=available,
        metavar="NAME",
        help=f"bundled room to run (default: {DEFAULT_SCENARIO}); "
        f"one of: {', '.join(available)}",
    )
    parser.add_argument(
        "-l", "--list", action="store_true",
        help="print bundled scenario names and exit",
    )
    parser.add_argument(
        "-d", "--design", action="store_true",
        help="open the room designer instead of running a scenario",
    )
    parser.add_argument(
        "--agents",
        type=checked(int_at_least(MIN_AGENTS)),
        default=None,
        metavar="N",
        help=f"number of agents to spawn (default: {DEFAULT_AGENT_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=checked(int_at_least(0)),
        default=None,
        metavar="S",
        help=f"spawn RNG seed, >= 0 (default: {DEFAULT_SPAWN_SEED})",
    )
    parser.add_argument(
        "--hazard",
        type=checked(parse_hazard),
        action="append",
        default=None,
        metavar="K=V,...",
        help="scripted hazard, repeatable: x=,y= required; optional tick=, "
        "intensity=, radius=, decay_rate=, block_radius=, "
        "blocks_navigation=, source_type=",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list excluding the program name; ``None`` reads
            ``sys.argv``.

    Returns:
        Namespace with ``scenario`` (validated bundled name), ``list``,
        ``design``, ``agents`` (int, default
        :data:`~crowd_evac.domain.constants.DEFAULT_AGENT_COUNT`), ``seed``
        (int, default :data:`~crowd_evac.domain.constants.DEFAULT_SPAWN_SEED`)
        and ``hazard`` (list of events, possibly empty).  Exits with status 2
        via argparse on any invalid value, if ``--design`` is combined with
        ``--list`` or a non-default ``--scenario``, or if ``--agents``,
        ``--seed`` or ``--hazard`` is combined with ``--design`` or
        ``--list``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.design and (args.list or args.scenario != DEFAULT_SCENARIO):
        parser.error("--design cannot be combined with --list or --scenario")
    given = [o for o in _RUN_OPTIONS if getattr(args, o) is not None]
    if given and (args.design or args.list):
        mode = "--design" if args.design else "--list"
        parser.error(f"{mode} cannot be combined with {_RUN_FLAGS}")
    if args.agents is None:
        args.agents = DEFAULT_AGENT_COUNT
    if args.seed is None:
        args.seed = DEFAULT_SPAWN_SEED
    if args.hazard is None:
        args.hazard = []
    return args
