"""Tests for crowd_evac.dashboard.chart_model.

Covers:
  - reduce_series: point limit, final count kept, empty and one-point input
  - nice_ticks: round values, degenerate range
  - build_chart: lines per run, mean, axes, no runs
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.dashboard.chart_model import (
    build_chart,
    nice_ticks,
    reduce_series,
)
from crowd_evac.montecarlo.results import build_curves, make_outcome

_COLUMNS = 100


def _steps(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A cumulative staircase of ``n`` points: one exit every 50 samples."""
    xs = np.arange(n, dtype=np.float64) * 0.05
    ys = (np.arange(n) // 50).astype(np.float64)
    return xs, ys


class TestReduceSeries:
    """Point limit with exact steps."""

    def test_long_series_cut_to_columns(self) -> None:
        """Verify at most one point per column plus the first."""
        xs, ys = _steps(10_000)
        out = reduce_series(xs, ys, _COLUMNS)
        assert len(out.xs) <= _COLUMNS + 1
        assert len(out.xs) == len(out.ys)

    def test_final_count_and_time_preserved(self) -> None:
        """Verify the last point is the original last point."""
        xs, ys = _steps(10_000)
        out = reduce_series(xs, ys, _COLUMNS)
        assert out.xs[-1] == xs[-1]
        assert out.ys[-1] == ys[-1]
        assert out.xs[0] == xs[0]

    def test_reduced_stays_non_decreasing(self) -> None:
        """Verify reduction never invents a drop."""
        xs, ys = _steps(10_000)
        out = reduce_series(xs, ys, _COLUMNS)
        assert np.all(np.diff(out.ys) >= 0)
        assert np.all(np.diff(out.xs) > 0)

    def test_values_come_from_original(self) -> None:
        """Verify every kept point exists in the input."""
        xs, ys = _steps(5_000)
        out = reduce_series(xs, ys, 50)
        original = set(zip(xs.tolist(), ys.tolist()))
        assert set(zip(out.xs.tolist(), out.ys.tolist())) <= original

    def test_short_series_unchanged(self) -> None:
        """Verify a series that already fits is returned as is."""
        xs, ys = _steps(30)
        out = reduce_series(xs, ys, _COLUMNS)
        assert out.xs is xs and out.ys is ys

    def test_empty_input(self) -> None:
        """Verify empty input gives empty output without error."""
        out = reduce_series(np.empty(0), np.empty(0), _COLUMNS)
        assert len(out.xs) == 0

    def test_single_point(self) -> None:
        """Verify one point passes through."""
        out = reduce_series(np.array([0.0]), np.array([0.0]), _COLUMNS)
        assert len(out.xs) == 1

    def test_zero_columns_treated_as_one(self) -> None:
        """Verify a collapsed plot width does not divide by zero."""
        xs, ys = _steps(1_000)
        out = reduce_series(xs, ys, 0)
        assert out.xs[-1] == xs[-1]

    def test_zero_span_keeps_last(self) -> None:
        """Verify identical x values reduce to the last point."""
        xs = np.zeros(500)
        ys = np.arange(500, dtype=np.float64)
        out = reduce_series(xs, ys, 10)
        assert out.ys.tolist() == [499.0]


class TestNiceTicks:
    """Axis ticks."""

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [
            (0.0, 10.0, (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)),
            (0.0, 1.0, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
            (0.0, 100.0, (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)),
            (0.0, 47.3, (0.0, 10.0, 20.0, 30.0, 40.0)),
        ],
    )
    def test_round_values(
        self, low: float, high: float, expected: tuple[float, ...]
    ) -> None:
        """Verify ticks are multiples of 1, 2 or 5 powers of ten."""
        assert nice_ticks(low, high) == pytest.approx(expected)

    def test_empty_range_gives_single_tick(self) -> None:
        """Verify a collapsed range returns just the start."""
        assert nice_ticks(3.0, 3.0) == (3.0,)

    def test_ticks_inside_range(self) -> None:
        """Verify no tick falls outside the axis."""
        ticks = nice_ticks(0.0, 123.4)
        assert min(ticks) >= 0.0 and max(ticks) <= 123.4


class TestBuildChart:
    """Whole-chart reduction."""

    def test_line_per_run_plus_mean(self) -> None:
        """Verify one line per run and a mean line."""
        outcomes = [
            make_outcome(0, 1, (), 2, (0, 1, 0, 1)),
            make_outcome(1, 2, (), 2, (1, 1)),
        ]
        chart = build_chart(build_curves(outcomes), _COLUMNS)
        assert len(chart.runs) == 2
        assert chart.mean is not None
        assert chart.y_max == 2.0
        assert chart.x_max == pytest.approx(0.2)
        assert chart.mean.ys[-1] == pytest.approx(2.0)

    def test_long_runs_are_reduced(self) -> None:
        """Verify 10,000-tick runs end up with few points."""
        exits = [0] * 9_999 + [1]
        chart = build_chart(
            build_curves([make_outcome(0, 1, (), 1, exits)]), _COLUMNS
        )
        assert len(chart.runs[0].xs) <= _COLUMNS + 1
        assert chart.runs[0].ys[-1] == 1.0

    def test_no_runs_gives_empty_unit_chart(self) -> None:
        """Verify an empty batch gives an empty chart with axes."""
        chart = build_chart(build_curves([]), _COLUMNS)
        assert chart.runs == () and chart.mean is None
        assert chart.x_ticks and chart.y_ticks
