"""Run settings and buttons of the dashboard.

Scenario, persons, mode, seed and iterations, plus the Run single, Run
Monte Carlo and Cancel buttons and a status line.  The panel only holds
widgets; the window decides what a press does.
"""
from __future__ import annotations

from collections.abc import Callable

import arcade.gui as gui

from crowd_evac.adapters.render.dashboard_widgets import LineBlock
from crowd_evac.adapters.render.designer_style import (
    FONT_NAME,
    FONT_SIZE_BODY,
    MUTED_COLOR,
    STATUS_COLOR,
    button_style,
)
from crowd_evac.adapters.render.designer_widgets import (
    CONTENT_WIDTH_PX,
    FIELD_WIDTH_PX,
    ROW_HEIGHT_PX,
    SPACING_PX,
    gap,
    make_heading,
    make_cell,
    make_input,
    make_label,
    make_row,
)
from crowd_evac.dashboard.form import FormFields, HazardFields

MODE_TEXT = {False: "Headless", True: "Visual"}
STATUS_LINES = 4


class DashboardPanel:
    """Widget tree of the run settings.

    Attributes:
        widget: Layout to place in the window.
    """

    def __init__(
        self,
        scenarios: list[str],
        default_scenario: str,
        on_run_single: Callable[[], None],
        on_run_batch: Callable[[], None],
        on_cancel: Callable[[], None],
        on_scenario: Callable[[str], None],
    ) -> None:
        """Build all widgets.

        Args:
            scenarios: Room names for the dropdown.
            default_scenario: Room selected at start.
            on_run_single: Run single pressed.
            on_run_batch: Run Monte Carlo pressed.
            on_cancel: Cancel pressed.
            on_scenario: Called with the room name when it changes.
        """
        self._visual = False
        self._persons = make_input("1")
        self._seed = make_input("")
        self._iterations = make_input("20")
        self._mode = self._button(MODE_TEXT[False], self._toggle_mode,
                                  FIELD_WIDTH_PX)
        self._dropdown = self._make_dropdown(scenarios, default_scenario)
        self._dropdown.event("on_change")(
            lambda _e: on_scenario(self.scenario))
        self._hint = gui.UILabel(
            "", width=CONTENT_WIDTH_PX, font_name=FONT_NAME,
            font_size=FONT_SIZE_BODY, text_color=MUTED_COLOR,
        )
        self.single_button = self._button("Run single", on_run_single)
        self.batch_button = self._button("Run Monte Carlo", on_run_batch)
        self.cancel_button = self._button("Cancel", on_cancel)
        self.cancel_button.disabled = True
        self._status = LineBlock(STATUS_LINES, STATUS_COLOR)
        self.widget = self._build()

    @staticmethod
    def _button(
        text: str, handler: Callable[[], None],
        width: int = CONTENT_WIDTH_PX,
    ) -> gui.UIFlatButton:
        """Create a button that calls ``handler`` on click."""
        btn = gui.UIFlatButton(
            text=text, width=width, height=ROW_HEIGHT_PX,
            style=button_style(),
        )
        btn.event("on_click")(lambda _e: handler())
        return btn

    @staticmethod
    def _make_dropdown(names: list[str], default: str) -> gui.UIDropdown:
        """Create the scenario dropdown."""
        return gui.UIDropdown(
            default=default if default in names else (names[0] if names
                                                      else None),
            options=list(names), width=CONTENT_WIDTH_PX,
            height=ROW_HEIGHT_PX, primary_style=button_style(),
            dropdown_style=button_style(), active_style=button_style(True),
        )

    def _build(self) -> gui.UIBoxLayout:
        """Assemble the column."""
        return gui.UIBoxLayout(
            vertical=True, align="left", space_between=SPACING_PX,
            children=[
                make_heading("Run settings"),
                self._dropdown, self._hint,
                make_row(make_label("Persons"), self._persons),
                gui.UIBoxLayout(
                    vertical=False, space_between=SPACING_PX,
                    children=[make_cell(make_label("Mode")), self._mode],
                ),
                make_row(make_label("Seed (optional)"), self._seed),
                make_row(make_label("Iterations (batch)"), self._iterations),
                gap(), self.single_button, self.batch_button,
                self.cancel_button, self._status.widget,
            ],
        )

    def _toggle_mode(self) -> None:
        """Switch between headless and visual."""
        self._visual = not self._visual
        self._mode.text = MODE_TEXT[self._visual]

    @property
    def scenario(self) -> str:
        """Room name chosen in the dropdown."""
        return str(self._dropdown.value or "")

    def read_fields(
        self, hazards: tuple[HazardFields, ...]
    ) -> FormFields:
        """Collect every input into a :class:`FormFields`."""
        return FormFields(
            scenario=self.scenario, persons=self._persons.text,
            visual=self._visual, seed=self._seed.text,
            iterations=self._iterations.text, hazards=hazards,
        )

    def set_hint(self, text: str) -> None:
        """Show the room size under the dropdown."""
        self._hint.text = text

    def set_status(self, text: str) -> None:
        """Show an error or progress message under the buttons."""
        self._status.set_wrapped(text)

    @property
    def status_text(self) -> str:
        """The status line as shown."""
        return self._status.text

    def set_running(self, running: bool) -> None:
        """Disable Run while a batch is active; enable Cancel."""
        self.single_button.disabled = running
        self.batch_button.disabled = running
        self.cancel_button.disabled = not running

    def is_typing(self) -> bool:
        """Whether any text field has keyboard focus."""
        return any(
            box.active
            for box in (self._persons, self._seed, self._iterations)
        )
