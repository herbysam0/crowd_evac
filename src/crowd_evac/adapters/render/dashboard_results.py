"""Results text of the dashboard: progress, time, velocity, counts, seed."""
from __future__ import annotations

import arcade.gui as gui

from crowd_evac.adapters.render.dashboard_widgets import LineBlock
from crowd_evac.adapters.render.designer_widgets import (
    SPACING_PX,
    make_heading,
)
from crowd_evac.dashboard.results_model import NO_VALUE, ResultsView

TIME_LINES: int = 4
SPEED_LINES: int = 2
COUNT_LINES: int = 3
COLUMN_WIDTH_PX: int = 210
COLUMN_GAP_PX: int = 10
HEIGHT_PX: int = 190
"""Vertical space the two result columns need above the chart."""
WIDTH_PX: int = 2 * COLUMN_WIDTH_PX + COLUMN_GAP_PX


class ResultsPanel:
    """Labels that show a :class:`~crowd_evac.dashboard.results_model.ResultsView`.

    Attributes:
        widget: Layout to place in the window.
    """

    def __init__(self) -> None:
        """Create the labels, all showing placeholders."""
        width = COLUMN_WIDTH_PX
        self._status = LineBlock(1, width=width)
        self._time = LineBlock(TIME_LINES, width=width)
        self._speed = LineBlock(SPEED_LINES, width=width)
        self._counts = LineBlock(COUNT_LINES, width=width)
        left = self._column([
            make_heading("Results"), self._status.widget,
            make_heading("Time to last evacuation"), self._time.widget,
        ])
        right = self._column([
            make_heading("Avg velocity (moving)"), self._speed.widget,
            make_heading("Runs"), self._counts.widget,
        ])
        self.widget = gui.UIBoxLayout(
            vertical=False, align="top", space_between=COLUMN_GAP_PX,
            children=[left, right],
        )
        self.show(None)

    @staticmethod
    def _column(rows: list[gui.UIWidget]) -> gui.UIBoxLayout:
        """Stack ``rows`` top-down, left-aligned."""
        return gui.UIBoxLayout(
            vertical=True, align="left", space_between=SPACING_PX,
            children=rows,
        )

    def show(self, view: ResultsView | None) -> None:
        """Show ``view``; ``None`` clears everything to placeholders."""
        if view is None:
            self._status.set_lines([NO_VALUE])
            self._time.set_lines([NO_VALUE])
            self._speed.set_lines([NO_VALUE])
            self._counts.set_lines([NO_VALUE])
            return
        self._status.set_lines([view.status])
        self._time.set_lines(view.time_lines)
        self._speed.set_lines(view.speed_lines)
        self._counts.set_lines([*view.count_lines, view.seed_line])

    def set_status(self, text: str) -> None:
        """Replace only the progress line, e.g. with ``Cancelling...``."""
        self._status.set_lines([text])

    @property
    def status_text(self) -> str:
        """The progress line as shown."""
        return self._status.text

    @property
    def time_text(self) -> str:
        """The time-to-last-evacuation block as shown."""
        return self._time.text

    @property
    def speed_text(self) -> str:
        """The average-velocity block as shown."""
        return self._speed.text

    @property
    def counts_text(self) -> str:
        """The counts block (runs, hazards, seed) as shown."""
        return self._counts.text
