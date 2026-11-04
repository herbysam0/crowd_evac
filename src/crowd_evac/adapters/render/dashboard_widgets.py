"""Small widget helpers for the dashboard.

``arcade.gui`` labels shrink to their text inside a box layout, so aligned
columns need fixed-size cells, and wrapped text is built from single-line
labels (a multi-line label collapses to one word per line there).
"""
from __future__ import annotations

import textwrap
from collections.abc import Sequence

import arcade.gui as gui
from arcade.types import RGBA255

from crowd_evac.adapters.render.designer_style import (
    FONT_NAME,
    FONT_SIZE_BODY,
    TEXT_COLOR,
)
from crowd_evac.adapters.render.designer_widgets import (
    CONTENT_WIDTH_PX,
    ROW_HEIGHT_PX,
)

LINE_HEIGHT_PX: int = 20
WRAP_CHARS: int = 34
"""Characters per line that fit :data:`CONTENT_WIDTH_PX` at body size."""


def fixed_cell(
    widget: gui.UIWidget, width: int, height: int = ROW_HEIGHT_PX
) -> gui.UIAnchorLayout:
    """Wrap ``widget`` in a fixed-size cell so neighbours line up."""
    cell = gui.UIAnchorLayout(width=width, height=height, size_hint=None)
    cell.add(widget, anchor_x="left", anchor_y="center")
    return cell


class LineBlock:
    """A fixed number of single-line labels showing wrapped text.

    Attributes:
        widget: Layout holding the labels.
    """

    def __init__(
        self,
        lines: int,
        color: RGBA255 = TEXT_COLOR,
        width: int = CONTENT_WIDTH_PX,
    ) -> None:
        """Create ``lines`` empty labels.

        Args:
            lines: How many lines the block can show.
            color: Text colour.
            width: Cell width in pixels.
        """
        self._labels = [
            gui.UILabel(
                " ", width=width, font_name=FONT_NAME,
                font_size=FONT_SIZE_BODY, text_color=color,
            )
            for _ in range(lines)
        ]
        self.widget = gui.UIBoxLayout(
            vertical=True, align="left",
            children=[fixed_cell(label, width, LINE_HEIGHT_PX)
                      for label in self._labels],
        )

    def set_lines(self, lines: Sequence[str]) -> None:
        """Show ``lines``; extra labels go blank, extra lines are dropped."""
        for index, label in enumerate(self._labels):
            label.text = lines[index] if index < len(lines) else " "

    def set_wrapped(self, text: str) -> None:
        """Wrap ``text`` to the block width and show it."""
        lines: list[str] = []
        for paragraph in text.splitlines():
            lines.extend(textwrap.wrap(paragraph, WRAP_CHARS) or [""])
        self.set_lines(lines)

    @property
    def text(self) -> str:
        """The shown lines joined by newlines, without blank padding."""
        shown = [str(label.text) for label in self._labels]
        return "\n".join(line for line in shown if line.strip())
