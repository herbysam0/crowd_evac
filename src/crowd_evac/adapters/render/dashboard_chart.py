"""Evacuation chart for the dashboard: thin line per run, bold average.

Curves can hold 10,000 points per run, so the lines are reduced to the plot
width by :mod:`crowd_evac.dashboard.chart_model` and drawn from a cached
``ShapeElementList``.  The cache is rebuilt only when data or size changes.
"""
from __future__ import annotations

import arcade
from arcade.shape_list import (
    Shape,
    ShapeElementList,
    create_line,
    create_line_strip,
    create_rectangle_filled,
)
from arcade.types import RGBA255

from crowd_evac.adapters.render.designer_style import (
    FONT_NAME,
    MUTED_COLOR,
    TEXT_COLOR,
)
from crowd_evac.dashboard.chart_model import ChartData, ChartSeries, build_chart
from crowd_evac.montecarlo.results import EvacuationCurves, build_curves

MARGIN_LEFT_PX: float = 52.0
MARGIN_BOTTOM_PX: float = 56.0
MARGIN_TOP_PX: float = 30.0
MARGIN_RIGHT_PX: float = 14.0
TICK_FONT_SIZE: int = 11
TITLE_FONT_SIZE: int = 12
TICK_LABEL_GAP_PX: float = 6.0
RUN_LINE_WIDTH_PX: float = 1.0
MEAN_LINE_WIDTH_PX: float = 3.0
RUN_COLOR: RGBA255 = (110, 170, 255, 110)
MEAN_COLOR: RGBA255 = (255, 205, 70, 255)
BACKGROUND: RGBA255 = (14, 16, 24, 255)
GRID_COLOR: RGBA255 = (60, 66, 84, 255)
MIN_PLOT_PX: float = 20.0
"""Smallest plot side worth drawing; below this the chart stays blank."""

Rect = tuple[float, float, float, float]


class ChartView:
    """Draws the chart inside a window rectangle.

    Attributes:
        rebuild_count: How many times the shape cache was rebuilt; tests use
            it to check the cache holds between frames.
    """

    def __init__(self) -> None:
        """Start empty and blank."""
        self._curves: EvacuationCurves = build_curves([])
        self._rect: Rect = (0.0, 0.0, 0.0, 0.0)
        self._dirty = True
        self._shapes: ShapeElementList[Shape] | None = None
        self._texts: list[arcade.Text] = []
        self.rebuild_count = 0

    def set_curves(self, curves: EvacuationCurves) -> None:
        """Replace the data; the cache is rebuilt on the next draw."""
        self._curves = curves
        self._dirty = True

    def set_rect(self, x: float, y: float, width: float, height: float) -> None:
        """Move or resize the chart; unchanged values keep the cache."""
        rect = (x, y, width, height)
        if rect != self._rect:
            self._rect = rect
            self._dirty = True

    def draw(self) -> None:
        """Draw the cached chart, rebuilding it first if needed."""
        if self._dirty:
            self._rebuild()
        if self._shapes is not None:
            self._shapes.draw()
        for text in self._texts:
            text.draw()

    # -- Cache -------------------------------------------------------------

    def _plot_rect(self) -> Rect:
        """Return the inner plot area, inside the axis margins."""
        x, y, w, h = self._rect
        return (
            x + MARGIN_LEFT_PX,
            y + MARGIN_BOTTOM_PX,
            w - MARGIN_LEFT_PX - MARGIN_RIGHT_PX,
            h - MARGIN_BOTTOM_PX - MARGIN_TOP_PX,
        )

    def _rebuild(self) -> None:
        """Rebuild shapes and texts for the current data and size."""
        self._dirty = False
        self.rebuild_count += 1
        self._texts = []
        px, py, pw, ph = self._plot_rect()
        if pw < MIN_PLOT_PX or ph < MIN_PLOT_PX:
            self._shapes = None
            return
        chart = build_chart(self._curves, int(pw))
        shapes: ShapeElementList[Shape] = ShapeElementList()
        shapes.append(
            create_rectangle_filled(px + pw / 2, py + ph / 2, pw, ph,
                                    BACKGROUND)
        )
        self._add_axes(shapes, chart, (px, py, pw, ph))
        for series in chart.runs:
            self._add_line(shapes, series, chart, (px, py, pw, ph),
                           RUN_COLOR, RUN_LINE_WIDTH_PX)
        if chart.mean is not None:
            self._add_line(shapes, chart.mean, chart, (px, py, pw, ph),
                           MEAN_COLOR, MEAN_LINE_WIDTH_PX)
        self._shapes = shapes
        self._add_titles(px, py, pw, ph)

    def _add_line(
        self, shapes: ShapeElementList[Shape], series: ChartSeries,
        chart: ChartData, plot: Rect, color: RGBA255, width: float,
    ) -> None:
        """Append one polyline in plot pixels; fewer than two points skip."""
        if len(series.xs) < 2:
            return
        px, py, pw, ph = plot
        xs = px + series.xs / chart.x_max * pw
        ys = py + series.ys / chart.y_max * ph
        points = list(zip(xs.tolist(), ys.tolist(), strict=True))
        shapes.append(create_line_strip(points, color, width))

    def _add_axes(
        self, shapes: ShapeElementList[Shape], chart: ChartData, plot: Rect
    ) -> None:
        """Append grid lines and tick labels."""
        px, py, pw, ph = plot
        for tick in chart.x_ticks:
            x = px + tick / chart.x_max * pw
            shapes.append(create_line(x, py, x, py + ph, GRID_COLOR, 1))
            self._texts.append(self._text(
                f"{tick:g}", x, py - TICK_LABEL_GAP_PX, "center", "top"))
        for tick in chart.y_ticks:
            y = py + tick / chart.y_max * ph
            shapes.append(create_line(px, y, px + pw, y, GRID_COLOR, 1))
            self._texts.append(self._text(
                f"{tick:g}", px - TICK_LABEL_GAP_PX, y, "right", "center"))

    def _add_titles(self, px: float, py: float, pw: float, ph: float) -> None:
        """Add the axis titles and the legend caption."""
        self._texts.append(self._text(
            "Time (s)", px + pw / 2, self._rect[1] + 6.0, "center", "bottom",
            TITLE_FONT_SIZE))
        self._texts.append(self._text(
            "People evacuated", px, py + ph + 8.0, "left", "bottom",
            TITLE_FONT_SIZE, TEXT_COLOR))
        self._texts.append(self._text(
            "thin: runs   bold: average", px + pw, py + ph + 8.0, "right",
            "bottom", TICK_FONT_SIZE))

    @staticmethod
    def _text(
        text: str, x: float, y: float, anchor_x: str, anchor_y: str,
        size: int = TICK_FONT_SIZE, color: RGBA255 = MUTED_COLOR,
    ) -> arcade.Text:
        """Create a cached text object."""
        return arcade.Text(
            text, x, y, color, size, font_name=FONT_NAME,
            anchor_x=anchor_x, anchor_y=anchor_y,
        )
