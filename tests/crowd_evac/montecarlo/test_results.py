"""Tests for crowd_evac.montecarlo.results.

Covers:
  - make_outcome: last evacuation time, evacuated count, hazards fired.
  - summarize: mean / std dev / min / max over complete runs only.
  - build_curves: cumulative, padded to the longest run, mean by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.constants import DT
from crowd_evac.montecarlo.results import (
    RunOutcome,
    build_curves,
    make_outcome,
    summarize,
)
from crowd_evac.scenarios.schema import EmergencyEventData


def _hazard(tick: int) -> EmergencyEventData:
    """A default hazard firing at ``tick``."""
    return {"tick": tick, "type": "place_panic_source", "pos": [1.0, 1.0]}


def _outcome(
    index: int, exits: tuple[int, ...], total: int = 2
) -> RunOutcome:
    """Outcome with no hazards from an exits-per-tick series."""
    return make_outcome(index, 100 + index, (), total, exits)


class TestMakeOutcome:
    """make_outcome derives results from the exits-per-tick series."""

    def test_last_evac_from_last_nonzero_tick(self) -> None:
        """Verify time is the end of the last tick that had an exit."""
        outcome = _outcome(0, (0, 0, 1, 0, 1, 0, 0))
        assert outcome.last_evac_s == pytest.approx(5 * DT)
        assert outcome.evacuated == 2
        assert outcome.complete

    def test_nobody_left_gives_none(self) -> None:
        """Verify no exits gives None, not the end time."""
        outcome = _outcome(0, (0, 0, 0))
        assert outcome.last_evac_s is None
        assert not outcome.complete

    def test_partial_run_is_incomplete(self) -> None:
        """Verify evacuated < total is incomplete even with exits."""
        outcome = _outcome(0, (0, 1, 0), total=2)
        assert outcome.evacuated == 1
        assert not outcome.complete
        assert outcome.last_evac_s == pytest.approx(2 * DT)

    def test_hazard_fires_only_before_run_end(self) -> None:
        """Verify a hazard at tick >= run length never fired."""
        hazards = (_hazard(0), _hazard(2), _hazard(3))
        outcome = make_outcome(0, 1, hazards, 1, (0, 0, 1))
        assert outcome.hazards_fired == 2
        assert outcome.hazards == hazards

    def test_seed_and_index_kept(self) -> None:
        """Verify identity fields are carried through."""
        outcome = _outcome(4, (1, 1))
        assert (outcome.run_index, outcome.seed) == (4, 104)


class TestSummarize:
    """summarize aggregates complete runs only."""

    def test_known_mean_and_sample_std(self) -> None:
        """Verify mean and ddof=1 std dev on known values."""
        # last-evac ticks 2, 4, 6 -> 0.10 s, 0.20 s, 0.30 s
        outcomes = [
            _outcome(0, (0, 2)),
            _outcome(1, (0, 0, 0, 2)),
            _outcome(2, (0, 0, 0, 0, 0, 2)),
        ]
        summary = summarize(outcomes)
        assert summary.complete == 3
        assert summary.incomplete == 0
        assert summary.mean_s == pytest.approx(0.20)
        assert summary.std_s == pytest.approx(0.10)
        assert summary.min_s == pytest.approx(0.10)
        assert summary.max_s == pytest.approx(0.30)

    def test_single_complete_run_has_no_std(self) -> None:
        """Verify one complete run gives std None."""
        summary = summarize([_outcome(0, (0, 2))])
        assert summary.std_s is None
        assert summary.mean_s == pytest.approx(2 * DT)

    def test_no_complete_runs(self) -> None:
        """Verify all-incomplete gives None statistics."""
        summary = summarize([_outcome(0, (0, 1))])
        assert summary.complete == 0
        assert summary.incomplete == 1
        assert summary.mean_s is None
        assert summary.min_s is None
        assert summary.max_s is None

    def test_incomplete_excluded_and_counted(self) -> None:
        """Verify an incomplete run does not enter the statistics."""
        summary = summarize([_outcome(0, (0, 2)), _outcome(1, (0, 0, 0, 1))])
        assert summary.complete == 1
        assert summary.incomplete == 1
        assert summary.max_s == pytest.approx(2 * DT)

    def test_empty(self) -> None:
        """Verify no outcomes at all is valid."""
        summary = summarize([])
        assert (summary.complete, summary.incomplete) == (0, 0)
        assert summary.mean_s is None


class TestCurves:
    """build_curves gives cumulative, padded series and their mean."""

    def test_cumulative_and_padded(self) -> None:
        """Verify series hold their final count up to the longest run."""
        curves = build_curves([_outcome(0, (1, 1)), _outcome(1, (0, 1, 0, 1))])
        assert curves.per_run.tolist() == [
            [0, 1, 2, 2, 2],
            [0, 0, 1, 1, 2],
        ]

    def test_times_start_at_zero_in_dt_steps(self) -> None:
        """Verify times_s runs 0, DT, 2 DT, ..."""
        curves = build_curves([_outcome(0, (1, 1, 0))])
        np.testing.assert_allclose(curves.times_s, [0.0, DT, 2 * DT, 3 * DT])

    def test_mean_by_hand(self) -> None:
        """Verify the mean stays correct after the short run ends."""
        curves = build_curves([_outcome(0, (1, 1)), _outcome(1, (0, 1, 0, 1))])
        np.testing.assert_allclose(curves.mean, [0.0, 0.5, 1.5, 1.5, 2.0])

    def test_zero_evacuated(self) -> None:
        """Verify a run with no exits is a flat zero line."""
        curves = build_curves([_outcome(0, (0, 0, 0))])
        assert curves.per_run.tolist() == [[0, 0, 0, 0]]
        assert curves.mean.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_no_outcomes(self) -> None:
        """Verify an empty batch gives a single t=0 point."""
        curves = build_curves([])
        assert curves.per_run.shape == (0, 1)
        assert curves.times_s.tolist() == [0.0]
        assert curves.mean.tolist() == [0.0]
