"""Result types of a Monte Carlo batch and their statistics.

Everything here is derived from each run's exits-per-tick series, so a
headless run and a visual run of the same seed give equal outcomes.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.constants import DT
from crowd_evac.montecarlo.config import MonteCarloConfig
from crowd_evac.scenarios.schema import EmergencyEventData


@dataclass(frozen=True)
class RunOutcome:
    """What one run produced.

    Attributes:
        run_index: Zero-based position in the batch.
        seed: Spawn seed; replay it in the game with ``--seed``.
        hazards: The concrete hazards this run was given.
        hazards_fired: Hazards whose tick came before the run ended.
        evacuated: Persons that left.
        total: Persons that started.
        last_evac_s: Time (s) the last person left; ``None`` if nobody did.
        exits_per_tick: Persons leaving at each tick, from tick 1 on.
    """

    run_index: int
    seed: int
    hazards: tuple[EmergencyEventData, ...]
    hazards_fired: int
    evacuated: int
    total: int
    last_evac_s: float | None
    exits_per_tick: tuple[int, ...]

    @property
    def complete(self) -> bool:
        """Whether everyone left.

        A stalled run ends without being "terminal", so only
        ``evacuated == total`` tells trapped agents apart.
        """
        return self.evacuated == self.total


@dataclass(frozen=True)
class RunSummary:
    """Time-to-last-evacuation statistics over complete runs only.

    Incomplete runs are counted, not averaged: mixing them in would hide
    trapped agents.

    Attributes:
        complete: Runs where everyone left.
        incomplete: Runs where someone did not.
        mean_s: Mean time (s); ``None`` with no complete run.
        std_s: Sample standard deviation (``ddof=1``); ``None`` with fewer
            than two complete runs.
        min_s: Shortest time (s), or ``None``.
        max_s: Longest time (s), or ``None``.
    """

    complete: int
    incomplete: int
    mean_s: float | None
    std_s: float | None
    min_s: float | None
    max_s: float | None


@dataclass(frozen=True, eq=False)
class EvacuationCurves:
    """Persons evacuated versus time, ready for a dashboard graph.

    Every series is cumulative, starts at 0 and is padded to the longest
    run by holding its final count, so ``mean`` stays correct after short
    runs end.

    Attributes:
        times_s: Time (s) of each point, ``0, DT, 2 DT, ...``.
        per_run: Cumulative evacuated count, shape ``(runs, len(times_s))``.
        mean: Mean of ``per_run`` over runs, one value per point.
    """

    times_s: npt.NDArray[np.float64]
    per_run: npt.NDArray[np.int64]
    mean: npt.NDArray[np.float64]


@dataclass(frozen=True)
class MonteCarloResult:
    """Everything a batch returns.

    Attributes:
        config: The settings the batch ran with.
        base_seed: Seed all run seeds derive from.
        seed_was_random: Whether ``base_seed`` was generated.
        outcomes: Finished runs, in run order.
        summary: Statistics over ``outcomes``.
        curves: Graph data over ``outcomes``.
        cancelled: Whether the batch stopped before every run finished.
        elapsed_s: Wall-clock time (s) the batch took, including setup.
            The only field that differs between identical batches.
    """

    config: MonteCarloConfig
    base_seed: int
    seed_was_random: bool
    outcomes: tuple[RunOutcome, ...]
    summary: RunSummary
    curves: EvacuationCurves
    cancelled: bool
    elapsed_s: float = 0.0


def make_outcome(
    run_index: int,
    seed: int,
    hazards: tuple[EmergencyEventData, ...],
    total: int,
    exits_per_tick: Sequence[int],
) -> RunOutcome:
    """Build a :class:`RunOutcome` from a run's exits-per-tick series.

    Args:
        run_index: Zero-based run number.
        seed: Spawn seed of the run.
        hazards: Hazards given to the run.
        total: Persons that started.
        exits_per_tick: Persons leaving at ticks 1, 2, ...

    Returns:
        The outcome.  A hazard counts as fired when its tick is before the
        number of ticks run, since hazards fire before each step.
    """
    exits = tuple(int(n) for n in exits_per_tick)
    leaving = [i for i, n in enumerate(exits) if n > 0]
    last = float(leaving[-1] + 1) * DT if leaving else None
    fired = sum(1 for h in hazards if h["tick"] < len(exits))
    return RunOutcome(
        run_index=run_index,
        seed=seed,
        hazards=hazards,
        hazards_fired=fired,
        evacuated=sum(exits),
        total=total,
        last_evac_s=last,
        exits_per_tick=exits,
    )


def summarize(outcomes: Sequence[RunOutcome]) -> RunSummary:
    """Summarize time to last evacuation over the complete runs.

    Args:
        outcomes: Finished runs.

    Returns:
        Counts of complete and incomplete runs and, from the complete ones,
        mean, sample standard deviation, minimum and maximum.
    """
    times = [
        o.last_evac_s for o in outcomes
        if o.complete and o.last_evac_s is not None
    ]
    incomplete = len(outcomes) - len(times)
    if not times:
        return RunSummary(0, incomplete, None, None, None, None)
    arr = np.asarray(times, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(times) > 1 else None
    return RunSummary(
        complete=len(times),
        incomplete=incomplete,
        mean_s=float(arr.mean()),
        std_s=std,
        min_s=float(arr.min()),
        max_s=float(arr.max()),
    )


def build_curves(outcomes: Sequence[RunOutcome]) -> EvacuationCurves:
    """Build cumulative evacuation series for the dashboard graph.

    Args:
        outcomes: Finished runs.

    Returns:
        Curves padded to the longest run; a single ``t = 0`` point when
        there are no outcomes.
    """
    length = max((len(o.exits_per_tick) for o in outcomes), default=0) + 1
    per_run = np.zeros((len(outcomes), length), dtype=np.int64)
    for row, outcome in enumerate(outcomes):
        counts = np.cumsum(outcome.exits_per_tick, dtype=np.int64)
        per_run[row, 1:len(counts) + 1] = counts
        # Hold the final count so later points do not drop to zero.
        per_run[row, len(counts) + 1:] = counts[-1] if counts.size else 0
    mean = (
        per_run.mean(axis=0) if len(outcomes) else np.zeros(length)
    ).astype(np.float64)
    times = np.arange(length, dtype=np.float64) * DT
    return EvacuationCurves(times_s=times, per_run=per_run, mean=mean)
