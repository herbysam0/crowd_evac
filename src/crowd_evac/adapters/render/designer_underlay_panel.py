"""Panel widgets for the designer's photo underlay section."""
from __future__ import annotations

from collections.abc import Callable

import arcade.gui as gui

from crowd_evac.adapters.render.designer_style import button_style
from crowd_evac.adapters.render.designer_widgets import (
    CONTENT_WIDTH_PX,
    ROW_HEIGHT_PX,
    SPACING_PX,
    make_heading,
    make_label,
)
from crowd_evac.designer.underlay import Underlay

_PERCENT = 100
_STEP_BUTTON_WIDTH_PX = 36
_SHOW_BUTTON_WIDTH_PX = 80
_NO_PHOTO_TEXT = "No photo"
# Keeps the label narrower than the column: a longer text raises the
# label's minimum width and would widen the whole right-anchored panel.
_MAX_NAME_CHARS = 18


def short_name(name: str) -> str:
    """Return ``name`` cut to ``_MAX_NAME_CHARS`` with an ellipsis."""
    if len(name) <= _MAX_NAME_CHARS:
        return name
    return name[: _MAX_NAME_CHARS - 3] + "..."


class UnderlayControls:
    """Load / show-hide / opacity / clear buttons and an opacity label."""

    def __init__(
        self,
        on_load: Callable[[], None],
        on_toggle: Callable[[], None],
        on_less: Callable[[], None],
        on_more: Callable[[], None],
        on_clear: Callable[[], None],
    ) -> None:
        """Build the widgets.

        Args:
            on_load: Called by ``Load photo...``.
            on_toggle: Called by ``Show/Hide``.
            on_less: Called by the opacity ``-`` button.
            on_more: Called by the opacity ``+`` button.
            on_clear: Called by ``Clear``.
        """
        self._label = make_label(_NO_PHOTO_TEXT, CONTENT_WIDTH_PX)
        self._load = self._button("Load photo...", on_load, CONTENT_WIDTH_PX)
        self._toggle = self._button(
            "Show/Hide", on_toggle, _SHOW_BUTTON_WIDTH_PX
        )
        self._less = self._button("-", on_less, _STEP_BUTTON_WIDTH_PX)
        self._more = self._button("+", on_more, _STEP_BUTTON_WIDTH_PX)
        self._clear = self._button("Clear", on_clear, _SHOW_BUTTON_WIDTH_PX)

    @staticmethod
    def _button(
        text: str, handler: Callable[[], None], width: int
    ) -> gui.UIFlatButton:
        """Create a button wired to ``handler``."""
        btn = gui.UIFlatButton(
            text=text,
            width=width,
            height=ROW_HEIGHT_PX,
            style=button_style(),
        )
        btn.event("on_click")(lambda _e: handler())
        return btn

    def rows(self) -> list[gui.UIWidget]:
        """Return heading, buttons and label for the panel column."""
        return [
            make_heading("Underlay photo"),
            self._load,
            gui.UIBoxLayout(
                vertical=False,
                space_between=SPACING_PX,
                children=[self._toggle, self._less, self._more, self._clear],
            ),
            self._label,
        ]

    def sync(self, underlay: Underlay | None) -> None:
        """Show the photo name, opacity and visibility."""
        if underlay is None:
            text = _NO_PHOTO_TEXT
        else:
            pct = round(underlay.opacity * _PERCENT)
            state = "" if underlay.visible else " (hidden)"
            text = f"{short_name(underlay.path.name)}  {pct} %{state}"
        if self._label.text != text:
            self._label.text = text
