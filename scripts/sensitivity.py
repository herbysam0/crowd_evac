"""Phase-2 sensitivity pre-pass launcher (Step 2.7).

Draws *n* quasi-random parameter vectors from the search space, evaluates each
via the composite fitness function (Step 2.6), and ranks each parameter by its
Spearman rank-correlation influence on the realism and evac-time objectives.

The ranked influence table guides which parameters to fix at their defaults
before the expensive NSGA-II run (Step 2.8).

Usage (PowerShell, .venv activated):
    python scripts/sensitivity.py --samples 256
    python scripts/sensitivity.py --samples 64 --method lhs --seed 1
    python scripts/sensitivity.py --samples 128 --workers 4
"""
from __future__ import annotations

import argparse
import logging
import sys

from crowd_evac.optimization.fitness import FitnessConfig
from crowd_evac.optimization.space import (
    BOUNDS,
    SensitivityResult,
    run_sensitivity_prepass,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Flag threshold: |rho| >= this value on at least one objective is "influential".
_INFLUENCE_THRESHOLD: float = 0.10


def _build_config(workers: int | None) -> FitnessConfig:
    """Build a FitnessConfig with the requested worker count.

    Args:
        workers: Worker process count for the harness; ``None`` uses all cores.

    Returns:
        A :class:`~crowd_evac.optimization.fitness.FitnessConfig` with the
        default down-scaled search suite and ``K = 3`` seeds.
    """
    return FitnessConfig(max_workers=workers)


def _print_table(sensitivity: SensitivityResult) -> None:
    """Print the ranked influence table to stdout.

    Args:
        sensitivity: Completed sensitivity result from
            :func:`~crowd_evac.optimization.space.run_sensitivity_prepass`.
    """
    col_w = dict(rank=5, name=35, realism=16, evac=16)
    header = (
        f"\n{'Rank':<{col_w['rank']}} {'Parameter':<{col_w['name']}} "
        f"{'|rho| realism':<{col_w['realism']}} {'|rho| evac_time':<{col_w['evac']}}"
    )
    separator = "-" * (sum(col_w.values()) + 4)
    print("\n=== Sensitivity Ranking — combined influence (Step 2.7) ===")
    print(header)
    print(separator)
    for rank, name in enumerate(sensitivity.rank_combined, start=1):
        idx = sensitivity.param_names.index(name)
        rho_r = sensitivity.abs_spearman_realism[idx]
        rho_e = sensitivity.abs_spearman_evac[idx]
        flag = "***" if max(rho_r, rho_e) >= _INFLUENCE_THRESHOLD else "   "
        print(
            f"{rank:<{col_w['rank']}} "
            f"{name:<{col_w['name']}} "
            f"{rho_r:<{col_w['realism']}.4f} "
            f"{rho_e:<{col_w['evac']}.4f} "
            f"{flag}"
        )
    print(
        f"\n*** = |rho| >= {_INFLUENCE_THRESHOLD} on at least one objective "
        f"— likely influential"
    )
    print(
        f"\nAnalysed {sensitivity.n_samples} samples over {len(BOUNDS)} parameters."
    )
    print("Review rankings and fix non-influential parameters before Step 2.8.")


def main() -> None:
    """Entry point: parse args, run pre-pass, print table.

    Raises:
        SystemExit: On argument-parsing failure.
    """
    parser = argparse.ArgumentParser(
        description="Phase-2 parameter sensitivity pre-pass (Step 2.7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        metavar="N",
        help="Number of quasi-random samples (power of 2 recommended for sobol).",
    )
    parser.add_argument(
        "--method",
        choices=["sobol", "lhs"],
        default="sobol",
        help="Quasi-random sampling method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="W",
        help=(
            "Worker processes per fitness evaluation (default: all cores). "
            "Use 1 for serial in-process evaluation (slow but debuggable)."
        ),
    )
    args = parser.parse_args()

    logger.info(
        "Starting sensitivity pre-pass: %d samples, method=%s, seed=%d, "
        "workers=%s",
        args.samples,
        args.method,
        args.seed,
        args.workers if args.workers is not None else "all",
    )

    config = _build_config(args.workers)
    sensitivity, fitness_results = run_sensitivity_prepass(
        n_samples=args.samples,
        config=config,
        seed=args.seed,
        method=args.method,
    )

    _print_table(sensitivity)

    n_violated = sum(1 for fr in fitness_results if fr.stuck_count > 0)
    logger.info(
        "Pre-pass complete: %d/%d samples had stuck_count > 0.",
        n_violated,
        len(fitness_results),
    )


if __name__ == "__main__":
    main()
