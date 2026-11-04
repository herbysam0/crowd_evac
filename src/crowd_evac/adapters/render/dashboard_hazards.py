"""Hazard list of the dashboard: typed min/max fields, Add and Remove.

Each row is two lines: type and position/time ranges, then the hazard's
numeric parameters.  Selecting a type prefills the parameters; what is in
the fields is what runs.  The list holds at most
:data:`~crowd_evac.dashboard.form.MAX_HAZARDS` rows because ``arcade.gui``
has no scroll widget here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import arcade.gui as gui

from crowd_evac.adapters.io.hazard_catalog import HazardType
from crowd_evac.adapters.render.dashboard_widgets import fixed_cell
from crowd_evac.adapters.render.designer_style import (
    FONT_NAME,
    FONT_SIZE_HEADING,
    MUTED_COLOR,
    button_style,
)
from crowd_evac.adapters.render.designer_widgets import (
    ROW_HEIGHT_PX,
    SPACING_PX,
    make_heading,
    make_input,
)
from crowd_evac.dashboard.form import (
    MAX_HAZARDS,
    HazardFields,
    hazard_from_type,
)

TYPE_WIDTH_PX: int = 100
FIELD_PX: int = 48
REMOVE_WIDTH_PX: int = 64
SMALL_LABEL_PX: int = 42
NAV_WIDTH_PX: int = 80
LIST_WIDTH_PX: int = (
    TYPE_WIDTH_PX + 6 * FIELD_PX + REMOVE_WIDTH_PX + 7 * SPACING_PX
)
NO_TYPES_LABEL: str = "(none)"
_RANGE_KEYS = ("x_min", "x_max", "y_min", "y_max", "t_min", "t_max")
_PARAM_KEYS = (
    ("intensity", "inten"), ("radius", "radius"), ("decay_rate", "decay"),
    ("block_radius", "block"),
)
_HEADERS = ("x min", "x max", "y min", "y max", "t0 (s)", "t1 (s)")
_NAV_TEXT = {True: "Blocks", False: "No block"}


def _header_label(text: str, width: int) -> gui.UIAnchorLayout:
    """Small muted column caption in a fixed-width cell."""
    label = gui.UILabel(
        text, width=width, font_name=FONT_NAME,
        font_size=FONT_SIZE_HEADING, text_color=MUTED_COLOR,
    )
    return fixed_cell(label, width)


class HazardRow:
    """Widgets of one hazard.

    Attributes:
        widget: The two-line layout for the list.
    """

    def __init__(
        self,
        types: Mapping[str, HazardType],
        initial: HazardFields,
        on_remove: Callable[[HazardRow], None],
    ) -> None:
        """Build the row from starting text.

        Args:
            types: Hazard types offered in the dropdown.
            initial: Starting field text.
            on_remove: Called with this row when Remove is pressed.
        """
        self._types = types
        self._labels = {t.label: n for n, t in types.items()}
        self._blocks = initial.blocks_navigation
        self._source = initial.source_type
        self.inputs = {
            key: make_input(getattr(initial, key), FIELD_PX)
            for key in (*_RANGE_KEYS, *(k for k, _ in _PARAM_KEYS))
        }
        self._dropdown = self._make_dropdown(initial.source_type)
        self._nav = gui.UIFlatButton(
            text=_NAV_TEXT[self._blocks], width=NAV_WIDTH_PX,
            height=ROW_HEIGHT_PX, style=button_style(),
        )
        self._nav.event("on_click")(lambda _e: self._toggle_nav())
        remove = gui.UIFlatButton(
            text="Remove", width=REMOVE_WIDTH_PX, height=ROW_HEIGHT_PX,
            style=button_style(),
        )
        remove.event("on_click")(lambda _e: on_remove(self))
        self.widget = self._layout(remove)

    def _make_dropdown(self, current: str) -> gui.UIDropdown:
        """Create the type dropdown and wire the prefill on change."""
        options: list[str | None] = [*self._labels] or [NO_TYPES_LABEL]
        known = self._types.get(current)
        dropdown = gui.UIDropdown(
            default=known.label if known else options[0],
            options=options, width=TYPE_WIDTH_PX, height=ROW_HEIGHT_PX,
            primary_style=button_style(), dropdown_style=button_style(),
            active_style=button_style(True),
        )
        dropdown.event("on_change")(lambda _e: self._on_type_changed())
        return dropdown

    def _layout(self, remove: gui.UIFlatButton) -> gui.UIBoxLayout:
        """Assemble the two lines."""
        line1: list[gui.UIWidget] = [
            self._dropdown, *[self.inputs[k] for k in _RANGE_KEYS], remove,
        ]
        line2: list[gui.UIWidget] = []
        for key, caption in _PARAM_KEYS:
            line2.append(
                _header_label(caption, SMALL_LABEL_PX)
            )
            line2.append(self.inputs[key])
        line2.append(self._nav)
        return gui.UIBoxLayout(
            vertical=True, space_between=SPACING_PX, children=[
                gui.UIBoxLayout(vertical=False, space_between=SPACING_PX,
                                children=line1),
                gui.UIBoxLayout(vertical=False, space_between=SPACING_PX,
                                children=line2),
            ])

    def _toggle_nav(self) -> None:
        """Flip whether the hazard blocks the route."""
        self._blocks = not self._blocks
        self._nav.text = _NAV_TEXT[self._blocks]

    def _selected_name(self) -> str:
        """Return the type name chosen in the dropdown ('' if none)."""
        return self._labels.get(str(self._dropdown.value or ""), "")

    def _on_type_changed(self) -> None:
        """Prefill the parameters from the chosen type."""
        hazard_type = self._types.get(self._selected_name())
        if hazard_type is None:
            return
        fresh = hazard_from_type(hazard_type)
        for key, _ in _PARAM_KEYS:
            self.inputs[key].text = getattr(fresh, key)
        self._blocks = fresh.blocks_navigation
        self._nav.text = _NAV_TEXT[self._blocks]

    def read(self) -> HazardFields:
        """Return the current field text."""
        texts = {key: box.text for key, box in self.inputs.items()}
        return HazardFields(
            source_type=self._selected_name(),
            blocks_navigation=self._blocks, **texts,
        )

    def is_typing(self) -> bool:
        """Whether any field of the row has keyboard focus."""
        return any(box.active for box in self.inputs.values())


class HazardList:
    """The Hazards section: header, rows and the Add button.

    Attributes:
        widget: Layout to place in the window.
        rows: The current rows, in order.
    """

    def __init__(self, types: Mapping[str, HazardType]) -> None:
        """Create an empty list.

        Args:
            types: Hazard types available (rescanned by the window each
                time the dashboard opens).
        """
        self._types = types
        self._center = ("", "")
        self.rows: list[HazardRow] = []
        self._rows_box = gui.UIBoxLayout(vertical=True, align="left",
                                         space_between=SPACING_PX)
        self._add = gui.UIFlatButton(
            text="Add hazard", width=LIST_WIDTH_PX, height=ROW_HEIGHT_PX,
            style=button_style(),
        )
        self._add.event("on_click")(lambda _e: self.add_row())
        self.widget = gui.UIBoxLayout(
            vertical=True, align="left", space_between=SPACING_PX,
            children=[make_heading("Hazards"), self._header(),
                      self._rows_box, self._add],
        )

    @staticmethod
    def _header() -> gui.UIBoxLayout:
        """Column captions aligned with the first line of each row."""
        cells: list[gui.UIWidget] = [_header_label("type", TYPE_WIDTH_PX)]
        cells += [_header_label(t, FIELD_PX) for t in _HEADERS]
        return gui.UIBoxLayout(vertical=False, space_between=SPACING_PX,
                               children=cells)

    def set_room_center(self, x: float, y: float) -> None:
        """Set the position prefilled into newly added hazards."""
        self._center = (f"{x:g}", f"{y:g}")

    def add_row(self) -> None:
        """Add a hazard row of the first type, if the list is not full."""
        if len(self.rows) >= MAX_HAZARDS:
            return
        first = next(iter(self._types.values()), None)
        base = hazard_from_type(first) if first else HazardFields()
        initial = replace(
            base, x_min=self._center[0], y_min=self._center[1]
        )
        self.rows.append(HazardRow(self._types, initial, self.remove_row))
        self._sync()

    def remove_row(self, row: HazardRow) -> None:
        """Remove ``row`` from the list."""
        if row in self.rows:
            self.rows.remove(row)
            self._sync()

    def _sync(self) -> None:
        """Re-add the rows to the layout and update the Add button."""
        self._rows_box.clear()  # type: ignore[no-untyped-call]
        for row in self.rows:
            self._rows_box.add(row.widget)
        self._add.disabled = len(self.rows) >= MAX_HAZARDS

    def read(self) -> tuple[HazardFields, ...]:
        """Return the text of every row."""
        return tuple(row.read() for row in self.rows)

    def is_typing(self) -> bool:
        """Whether any row field has keyboard focus."""
        return any(row.is_typing() for row in self.rows)
