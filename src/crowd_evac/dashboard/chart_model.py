"""Reduce evacuation curves to what a chart of a given width can show.

Pure: no arcade.  A curve can hold 10,000 points per run; drawing them all
every frame would be too slow, and a plot only has so many pixel columns.
The series here are cumulative (never decrease), so keeping the last sample
of each pixel column keeps every step at pixel precision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crowd_evac.montecarlo.results import EvacuationCurves

Floats = npt.NDArray[np.float64]

DEFAULT_TICK_TARGET: int = 5
"""Axis ticks aimed for; the nice-number rule may give one or two more."""

_NICE_STEPS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

_EPS: float = 1e-9
"""Slack so a tick on the range edge survives float rounding."""

_TICK_DECIMALS: int = 10
"""Rounding of tick values, hiding accumulated float error."""


@dataclass(frozen=True, eq=False)
class ChartSeries:
    """One drawable line.

    Attributes:
        xs: Time (s) of each point.
        ys: Persons evacuated at each point.
    """

    xs: Floats
    ys: Floats


@dataclass(frozen=True, eq=False)
class ChartData:
    """Everything the chart widget draws.

    Attributes:
        runs: One thin line per run, in run order.
        mean: The bold average line, or ``None`` with no runs.
        x_max: Right edge of the time axis (s).
        y_max: Top edge of the count axis.
        x_ticks: Time axis tick values.
        y_ticks: Count axis tick values.
    """

    runs: tuple[ChartSeries, ...]
    mean: ChartSeries | None
    x_max: float
    y_max: float
    x_ticks: tuple[float, ...]
    y_ticks: tuple[float, ...]


def reduce_series(xs: Floats, ys: Floats, columns: int) -> ChartSeries:
    """Cut a cumulative series to at most one point per pixel column.

    Args:
        xs: Increasing x values.
        ys: Matching y values; must not decrease for exact steps.
        columns: Plot width in pixels; values below 1 are treated as 1.

    Returns:
        The series itself when it already fits; else the first point plus
        the last point of each column, so the final value is kept.
    """
    columns = max(columns, 1)
    if len(xs) <= columns + 1:
        return ChartSeries(xs, ys)
    span = float(xs[-1] - xs[0])
    if span <= 0.0:
        return ChartSeries(xs[-1:], ys[-1:])
    col = np.minimum(((xs - xs[0]) / span * columns).astype(np.int64),
                     columns - 1)
    # Index of the last sample in each column: where the next one differs.
    keep = np.flatnonzero(np.append(col[1:] != col[:-1], True))
    if keep[0] != 0:
        keep = np.insert(keep, 0, 0)
    return ChartSeries(xs[keep], ys[keep])


def nice_ticks(
    low: float, high: float, target: int = DEFAULT_TICK_TARGET
) -> tuple[float, ...]:
    """Pick round axis tick values covering ``[low, high]``.

    Args:
        low: Axis start.
        high: Axis end.
        target: Roughly how many ticks to aim for; at least 1.

    Returns:
        Evenly spaced multiples of 1, 2 or 5 times a power of ten inside
        the range; just ``(low,)`` when the range is empty.
    """
    if not high > low:
        return (low,)
    raw = (high - low) / max(target, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(
        m * magnitude for m in _NICE_STEPS if m * magnitude >= raw
    )
    first = math.ceil(low / step - _EPS) * step
    count = int(math.floor((high - first) / step + _EPS)) + 1
    return tuple(round(first + i * step, _TICK_DECIMALS) for i in range(count))


def build_chart(curves: EvacuationCurves, plot_width_px: int) -> ChartData:
    """Reduce curves and choose axes for a plot of the given width.

    Args:
        curves: Cumulative curves from the batch results.
        plot_width_px: Plot width in pixels.

    Returns:
        Reduced lines and axis ticks.  With no runs, an empty chart on a
        unit-sized axis.
    """
    runs = tuple(
        reduce_series(curves.times_s, row.astype(np.float64), plot_width_px)
        for row in curves.per_run
    )
    if not runs:
        return ChartData((), None, 1.0, 1.0, nice_ticks(0.0, 1.0),
                         nice_ticks(0.0, 1.0))
    x_max = max(float(curves.times_s[-1]), _EPS)
    y_max = max(float(np.max(curves.per_run)), 1.0)
    mean = reduce_series(curves.times_s, curves.mean, plot_width_px)
    return ChartData(
        runs=runs, mean=mean, x_max=x_max, y_max=y_max,
        x_ticks=nice_ticks(0.0, x_max), y_ticks=nice_ticks(0.0, y_max),
    )
