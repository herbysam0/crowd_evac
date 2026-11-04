"""Turn batch outcomes into the text the results panel shows.

Pure: no arcade.  Rebuilt after every finished run, so it reads only the
outcomes (and the seed) and uses the existing
:func:`~crowd_evac.montecarlo.results.summarize` for statistics.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from crowd_evac.montecarlo.results import RunOutcome, RunSummary, summarize

NO_VALUE: str = "-"
"""Shown where a value does not exist (yet)."""


@dataclass(frozen=True)
class ResultsView:
    """Display lines of the results panel.

    Attributes:
        status: Progress such as ``Run 7/20``, ``Done`` or ``Cancelled``.
        time_lines: Time to last evacuation, or ``n/m evacuated`` when a
            single run did not finish.
        speed_lines: Average moving velocity lines.
        count_lines: Complete/incomplete and hazards fired.
        seed_line: The seed, marked ``(random)`` when it was drawn.
    """

    status: str
    time_lines: tuple[str, ...]
    speed_lines: tuple[str, ...]
    count_lines: tuple[str, ...]
    seed_line: str


def _seconds(value: float | None) -> str:
    """Format seconds, or the placeholder."""
    return NO_VALUE if value is None else f"{value:.1f} s"


def _mps(value: float | None) -> str:
    """Format metres per second, or the placeholder."""
    return NO_VALUE if value is None else f"{value:.2f} m/s"


def _single_time(outcome: RunOutcome) -> tuple[str, ...]:
    """Time lines of a single run: the time, or how many got out."""
    if outcome.complete and outcome.last_evac_s is not None:
        return (_seconds(outcome.last_evac_s),)
    return (f"{outcome.evacuated}/{outcome.total} evacuated",)


def _batch_time(summary: RunSummary) -> tuple[str, ...]:
    """Time lines of a batch, over complete runs only."""
    if summary.mean_s is None:
        return ("no complete runs",)
    lines = [f"mean {_seconds(summary.mean_s)}"]
    if summary.std_s is not None:
        lines.append(f"std dev {_seconds(summary.std_s)}")
    lines.append(f"min {_seconds(summary.min_s)}")
    lines.append(f"max {_seconds(summary.max_s)}")
    return tuple(lines)


def _speed_lines(summary: RunSummary, single: bool) -> tuple[str, ...]:
    """Velocity lines: the value, with std dev for a batch."""
    if single or summary.speed_std_mps is None:
        return (_mps(summary.speed_mean_mps),)
    return (_mps(summary.speed_mean_mps),
            f"std dev {_mps(summary.speed_std_mps)}")


def _count_lines(
    outcomes: Sequence[RunOutcome], summary: RunSummary, single: bool
) -> tuple[str, ...]:
    """Evacuated/incomplete counts and hazards fired."""
    fired = sum(o.hazards_fired for o in outcomes)
    planned = sum(len(o.hazards) for o in outcomes)
    hazards = f"hazards fired {fired}/{planned}"
    if single:
        return (hazards,)
    return (f"complete {summary.complete}, incomplete {summary.incomplete}",
            hazards)


def _status(done: int, total: int, finished: bool, cancelled: bool) -> str:
    """Progress text."""
    if cancelled:
        return f"Cancelled after {done}/{total}"
    if finished:
        return f"Done {done}/{total}"
    return f"Run {done}/{total}"


def build_view(
    outcomes: Sequence[RunOutcome],
    *,
    total_runs: int,
    single: bool,
    base_seed: int,
    seed_was_random: bool,
    finished: bool = False,
    cancelled: bool = False,
) -> ResultsView:
    """Build the panel text from the runs finished so far.

    Args:
        outcomes: Finished runs, in run order.
        total_runs: Runs the batch will make.
        single: ``True`` for one run (shows its value, not statistics).
        base_seed: Seed used (the exact spawn seed for a single run).
        seed_was_random: Whether the seed was drawn, not typed.
        finished: Whether the batch ended.
        cancelled: Whether it ended early on request.

    Returns:
        The lines to show.
    """
    summary = summarize(outcomes)
    if not outcomes:
        time_lines: tuple[str, ...] = (NO_VALUE,)
    elif single:
        time_lines = _single_time(outcomes[0])
    else:
        time_lines = _batch_time(summary)
    seed = f"seed {base_seed}" + (" (random)" if seed_was_random else "")
    return ResultsView(
        status=_status(len(outcomes), total_runs, finished, cancelled),
        time_lines=time_lines,
        speed_lines=_speed_lines(summary, single),
        count_lines=_count_lines(outcomes, summary, single),
        seed_line=seed,
    )
