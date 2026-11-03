"""Shared fonts, sizes and colours for the room designer UI."""
from __future__ import annotations

import arcade.gui as gui
from arcade.gui.widgets.text import UIInputTextStyle
from arcade.types import RGBA255

# First installed family wins; Segoe UI ships with Windows 11.
FONT_NAME: tuple[str, ...] = ("Segoe UI", "Calibri", "Arial")
FONT_SIZE_BODY: int = 14
FONT_SIZE_HEADING: int = 12
FONT_SIZE_STATUS: int = 13
FONT_SIZE_CANVAS: int = 18
FONT_SIZE_HINT: int = 14

TEXT_COLOR: RGBA255 = (235, 235, 240, 255)
MUTED_COLOR: RGBA255 = (175, 190, 215, 255)
STATUS_COLOR: RGBA255 = (255, 200, 120, 255)
LABEL_BG_COLOR: RGBA255 = (20, 20, 26, 220)

_BUTTON_BG: RGBA255 = (52, 64, 88, 255)
_BUTTON_HOVER: RGBA255 = (70, 88, 124, 255)
_BUTTON_PRESS: RGBA255 = (96, 124, 180, 255)
_BUTTON_BORDER: RGBA255 = (110, 125, 160, 255)
_ACTIVE_BG: RGBA255 = (40, 120, 200, 255)
_INPUT_BG: RGBA255 = (14, 16, 24, 255)
_INPUT_BORDER: RGBA255 = (110, 125, 160, 255)
_INPUT_FOCUS: RGBA255 = (90, 170, 255, 255)
INPUT_PAD_PX: int = 6


def input_style() -> dict[str, UIInputTextStyle]:
    """Return dark text-field styles with a visible focus border."""
    style = UIInputTextStyle
    return {
        "normal": style(bg=_INPUT_BG, border=_INPUT_BORDER, border_width=1),
        "hover": style(bg=_INPUT_BG, border=_INPUT_FOCUS, border_width=1),
        "press": style(bg=_INPUT_BG, border=_INPUT_FOCUS, border_width=2),
        "disabled": style(bg=_INPUT_BG, border=_INPUT_BORDER, border_width=1),
        "invalid": style(bg=_INPUT_BG, border=(230, 76, 60, 255), border_width=2),
    }


def button_style(active: bool = False) -> dict[str, gui.UIFlatButton.UIStyle]:
    """Return per-state flat-button styles; ``active`` marks the tool in use."""

    def make(bg: RGBA255, border_width: int = 1) -> gui.UIFlatButton.UIStyle:
        return gui.UIFlatButton.UIStyle(
            font_size=FONT_SIZE_BODY,
            font_name=FONT_NAME,
            font_color=TEXT_COLOR,
            bg=bg,
            border=_BUTTON_BORDER,
            border_width=border_width,
        )

    return {
        "normal": make(_ACTIVE_BG if active else _BUTTON_BG),
        "hover": make(_BUTTON_HOVER),
        "press": make(_BUTTON_PRESS),
        "disabled": make(_BUTTON_BG),
    }
