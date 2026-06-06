"""Offline NSGA-II weight-search launcher (Phase-2, Step 2.8).

Launches (or resumes) the multi-objective search that produces the realism↔time
Pareto front.  The run is multi-hour at full budget and is meant to execute as a
detached background job — nothing in a session awaits it.

Usage (PowerShell, .venv activated):
    # fresh run, all cores
    python scripts/optimize_weights.py --pop 64 --gens 80 --seeds 5 `
        --out artifacts/calibration

    # resume an interrupted run from its checkpoint
    python scripts/optimize_weights.py --resume --out artifacts/calibration

    # tiny smoke run on a single core (debuggable)
    python scripts/optimize_weights.py --pop 4 --gens 2 --seeds 1 --workers 1

To detach in PowerShell:
    Start-Process -NoNewWindow python `
        -ArgumentList "scripts/optimize_weights.py","--pop","64","--gens","80" `
        -RedirectStandardOutput artifacts/calibration/run.log
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from crowd_evac.optimization.fitness import FitnessConfig
from crowd_evac.optimization.nsga import (
    FRONT_FILENAME,
    NSGAConfig,
    run_nsga,
    write_front,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the search launcher.

    Returns:
        Parsed argument namespace.

    Raises:
        SystemExit: On argument-parsing failure.
    """
    parser = argparse.ArgumentParser(
        description="Launch or resume the Phase-2 NSGA-II weight search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pop", type=int, default=64, help="Population size.")
    parser.add_argument("--gens", type=int, default=80, help="Generations.")
    parser.add_argument(
        "--seeds", type=int, default=5, metavar="K",
        help="Number of RNG seeds per scenario (CRN seed set is range(K)).",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="NSGA-II operator seed."
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="W",
        help="Harness worker processes per candidate (default: all cores).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/calibration"),
        help="Output / checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1, metavar="N",
        help="Pickle the algorithm every N generations.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing checkpoint in --out if present.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: build config, run the search, write the front."""
    args = _parse_args()
    fitness_config = FitnessConfig(
        seeds=tuple(range(args.seeds)),
        max_workers=args.workers,
    )
    config = NSGAConfig(
        pop_size=args.pop,
        n_gen=args.gens,
        seed=args.seed,
        fitness_config=fitness_config,
        checkpoint_dir=args.out,
        checkpoint_every=args.checkpoint_every,
    )
    logger.info(
        "launch: pop=%d gens=%d seeds=%d workers=%s out=%s resume=%s",
        args.pop, args.gens, args.seeds,
        args.workers if args.workers is not None else "all",
        args.out, args.resume,
    )

    result = run_nsga(config, resume=args.resume)

    front_path = args.out / FRONT_FILENAME
    write_front(front_path, result)
    logger.info(
        "wrote %d Pareto points to %s (%.1f s, %d generations)",
        len(result.front), front_path, result.wall_clock_s,
        result.n_gen_completed,
    )
    feasible = sum(1 for p in result.front if p.is_feasible)
    logger.info(
        "front: %d/%d points feasible (stuck_count == 0)",
        feasible, len(result.front),
    )


if __name__ == "__main__":
    main()
