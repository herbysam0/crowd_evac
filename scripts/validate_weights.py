"""Select and full-scale-validate the calibrated weight set (Phase-2, Step 2.9).

Reads the Pareto front produced by the offline NSGA-II run (Step 2.8), applies
the realism-gated selection rule, and re-validates the chosen weight set at full
Tier-A agent count on the un-down-scaled scenario suite plus a hazard-reroute
scenario.  The validated winner (and the full audit trail) is written to
``<out>/chosen_weights.json`` for Step 2.10 to ship as the R0.3 defaults.

Usage (PowerShell, .venv activated):
    # select + validate the winner at full scale (the documented command)
    python scripts/validate_weights.py --front artifacts/calibration/front.json --tier-a

    # tighten the realism gate, fewer held-out seeds, all cores
    python scripts/validate_weights.py --threshold 0.10 --seeds 3

    # skip the hazard-reroute check (hazard-free validation only)
    python scripts/validate_weights.py --no-hazard

Re-running the whole calibration (future operator procedure)
------------------------------------------------------------
1. Sensitivity pre-pass (prune non-influential weights, Step 2.7):
       python scripts/sensitivity.py --samples 256
2. Launch the offline NSGA-II search (Step 2.8; runs detached for hours):
       python scripts/optimize_weights.py --pop 64 --gens 80 --seeds 5 `
           --out artifacts/calibration
3. Select + validate the winner at full scale (this script, Step 2.9):
       python scripts/validate_weights.py --front artifacts/calibration/front.json --tier-a
4. Ship the chosen weights as defaults and set the §8 thresholds (Step 2.10).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from crowd_evac.optimization.select import (
    DEFAULT_HAZARD_EVAC_FLOOR,
    DEFAULT_REALISM_THRESHOLD,
    HAZARD_VALIDATION_SCENARIO,
    SelectionError,
    default_validation_config,
    load_front,
    select_and_validate,
    write_outcome,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_HELD_OUT_SEED_BASE: int = 101
"""First held-out validation seed; the suite is range(base, base + n)."""

OUTCOME_FILENAME: str = "chosen_weights.json"
"""Validated-selection report written under the output directory."""


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the validation launcher.

    Returns:
        Parsed argument namespace.

    Raises:
        SystemExit: On argument-parsing failure.
    """
    parser = argparse.ArgumentParser(
        description="Select and full-scale-validate the calibrated weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--front", type=Path, default=Path("artifacts/calibration/front.json"),
        help="Pareto front JSON written by optimize_weights.py.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/calibration"),
        help="Output directory for the chosen-weights report.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_REALISM_THRESHOLD,
        help="Maximum accepted realism distance (selection gate).",
    )
    parser.add_argument(
        "--seeds", type=int, default=5, metavar="N",
        help="Number of held-out validation seeds (range(101, 101+N)).",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="W",
        help="Harness worker processes (default: all cores).",
    )
    parser.add_argument(
        "--tier-a", action="store_true",
        help="Validate at full Tier-A scale (the default behaviour).",
    )
    parser.add_argument(
        "--no-hazard", action="store_true",
        help="Skip the hazard-reroute validation scenario.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: load the front, select + validate, write the report.

    Returns:
        Process exit code: ``0`` on a validated selection, ``1`` if no front
        point survives the gate and full-scale validation.
    """
    args = _parse_args()
    front = load_front(args.front)
    logger.info("loaded %d front point(s) from %s", len(front), args.front)

    seeds = tuple(_HELD_OUT_SEED_BASE + i for i in range(args.seeds))
    config = default_validation_config(seeds=seeds, max_workers=args.workers)
    hazard = None if args.no_hazard else HAZARD_VALIDATION_SCENARIO
    logger.info(
        "validating: threshold=%.3f seeds=%s hazard=%s evac_floor=%.2f",
        args.threshold, seeds, hazard or "disabled", DEFAULT_HAZARD_EVAC_FLOOR,
    )

    try:
        outcome = select_and_validate(
            front,
            validation_config=config,
            hazard_scenario=hazard,
            threshold=args.threshold,
        )
    except SelectionError as exc:
        logger.error("selection failed: %s", exc)
        return 1

    out_path = args.out / OUTCOME_FILENAME
    write_outcome(out_path, outcome)
    _log_summary(outcome, out_path)
    return 0


def _log_summary(outcome: object, out_path: Path) -> None:
    """Log a concise summary of the validated selection.

    Args:
        outcome: The :class:`~crowd_evac.optimization.select.SelectionOutcome`.
        out_path: Path the report was written to.
    """
    # Local import keeps the type reference without widening the public surface.
    from crowd_evac.optimization.select import SelectionOutcome

    assert isinstance(outcome, SelectionOutcome)
    val = outcome.validation
    logger.info(
        "CHOSEN: realism=%.4f evac=%.2f stuck=%d evac_frac=%.3f "
        "(stepped past %d point(s))",
        val.realism_distance, val.evac_time, val.stuck_count,
        val.evacuated_fraction, len(outcome.rejected),
    )
    if outcome.hazard is not None:
        haz = outcome.hazard
        logger.info(
            "HAZARD (%s): stuck=%d evac_frac=%.3f passed=%s",
            haz.scenario, haz.stuck_count, haz.evacuated_fraction, haz.passed,
        )
    logger.info("wrote chosen-weights report to %s", out_path)


if __name__ == "__main__":
    raise SystemExit(main())
