"""Widget factories and number helpers shared by the designer panel."""
from __future__ import annotations

import arcade.gui as gui

from crowd_evac.adapters.render.designer_style import (
    FONT_NAME,
    FONT_SIZE_BODY,
    FONT_SIZE_HEADING,
    FONT_SIZE_STATUS,
    INPUT_PAD_PX,
    MUTED_COLOR,
    STATUS_COLOR,
    TEXT_COLOR,
    input_style,
)

CONTENT_WIDTH_PX: int = 280
LABEL_WIDTH_PX: int = 140
FIELD_WIDTH_PX: int = 136
ROW_HEIGHT_PX: int = 30
SPACING_PX: int = 4
SECTION_GAP_PX: int = 8
_DECIMALS = 3


def fmt(value: float) -> str:
    """Format a metre value without float noise (``3.0`` -> ``3``)."""
    return format(round(value, _DECIMALS), "g")


def parse_number(text: str) -> float | None:
    """Parse a user-typed number; None if it is not one."""
    try:
        return float(text.strip())
    except ValueError:
        return None


def gap() -> gui.UIWidget:
    """Return a vertical spacer between panel sections."""
    return gui.UISpace(height=SECTION_GAP_PX)  # type: ignore[no-untyped-call]


def make_input(text: str, width: int = FIELD_WIDTH_PX) -> gui.UIInputText:
    """Create a single-line text field."""
    box = gui.UIInputText(
        text=text,
        width=width,
        height=ROW_HEIGHT_PX,
        font_name=FONT_NAME,
        font_size=FONT_SIZE_BODY,
        text_color=TEXT_COLOR,
        style=input_style(),
    )
    box.padding = (0, 0, 0, INPUT_PAD_PX)
    return box


def make_label(text: str, width: int = LABEL_WIDTH_PX) -> gui.UILabel:
    """Create a body-text label."""
    return gui.UILabel(
        text,
        width=width,
        font_name=FONT_NAME,
        font_size=FONT_SIZE_BODY,
        text_color=TEXT_COLOR,
    )


def make_heading(text: str) -> gui.UILabel:
    """Create a small bold section heading."""
    return gui.UILabel(
        text.upper(),
        width=CONTENT_WIDTH_PX,
        font_name=FONT_NAME,
        font_size=FONT_SIZE_HEADING,
        text_color=MUTED_COLOR,
        bold=True,
    )


def make_status() -> gui.UILabel:
    """Create the wrapped status-line label."""
    return gui.UILabel(
        "",
        width=CONTENT_WIDTH_PX,
        multiline=True,
        font_name=FONT_NAME,
        font_size=FONT_SIZE_STATUS,
        text_color=STATUS_COLOR,
    )


def make_cell(label: gui.UILabel) -> gui.UIAnchorLayout:
    """Wrap a label in a fixed-width cell so fields line up."""
    cell = gui.UIAnchorLayout(
        width=LABEL_WIDTH_PX, height=ROW_HEIGHT_PX, size_hint=None
    )
    cell.add(label, anchor_x="left", anchor_y="center")
    return cell


def make_row(
    label: gui.UILabel, field: gui.UIInputText
) -> gui.UIBoxLayout:
    """Aligned label + field on one line."""
    return gui.UIBoxLayout(
        vertical=False,
        space_between=SPACING_PX,
        children=[make_cell(label), field],
    )
