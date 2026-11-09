"""``arcade.gui`` side panel for the room designer.

Holds tool buttons, room and doorway-default fields, numeric editors for
the current selection, and Save. All behaviour is delegated to the
:class:`~crowd_evac.designer.controller.DesignController`.
"""
from __future__ import annotations

from collections.abc import Callable

import arcade.gui as gui

from crowd_evac.adapters.render.designer_style import button_style
from crowd_evac.adapters.render.designer_widgets import (
    CONTENT_WIDTH_PX,
    ROW_HEIGHT_PX,
    SPACING_PX,
    fmt,
    gap,
    make_heading,
    make_input,
    make_label,
    make_row,
    make_status,
    parse_number,
)
from crowd_evac.adapters.render.designer_labels import ROOM_LABELS
from crowd_evac.adapters.render.designer_selection import SelectionEditor
from crowd_evac.designer.controller import DesignController, Tool
from crowd_evac.designer.layout import RoomSettings

_COLUMN_GAP_PX = 16
_PANEL_MARGIN_PX = 8
# Two columns plus the gap and both margins.
PANEL_WIDTH_PX: int = (
    2 * CONTENT_WIDTH_PX + _COLUMN_GAP_PX + 2 * _PANEL_MARGIN_PX
)
_TOOL_GAP_PX = 2
_LOAD_BUTTON_WIDTH_PX = 80
# Select is the default (no button), so only the drawing tools get buttons.
_BUTTON_TOOLS = tuple(t for t in Tool if t is not Tool.SELECT)
# The buttons plus their gaps must exactly fit the content width.
_TOOL_BUTTON_WIDTH_PX = (
    CONTENT_WIDTH_PX - (len(_BUTTON_TOOLS) - 1) * _TOOL_GAP_PX
) // len(_BUTTON_TOOLS)


class DesignerPanel:
    """Widget tree and event wiring for the designer side panel.

    Attributes:
        widget: Root layout to add to a ``UIManager``.
    """

    def __init__(
        self,
        ctl: DesignController,
        on_save: Callable[[], None],
        on_load: Callable[[], None],
        scenarios: list[str],
        extra_rows: list[gui.UIWidget] | None = None,
    ) -> None:
        """Build all widgets.

        Args:
            ctl: Controller the widgets drive.
            on_save: Called when Save is pressed.
            on_load: Called when Load is pressed.
            scenarios: Saved scenario names offered for loading.
            extra_rows: Optional widgets appended to the left column.
        """
        self._ctl = ctl
        self._extra_rows = extra_rows or []
        self._active: dict[Tool, bool] = {}
        self._tool_buttons = {t: self._tool_button(t) for t in _BUTTON_TOOLS}
        self._room_inputs = self._room_fields(ctl.model.layout.room)
        self._wall_t = make_input(fmt(ctl.wall_thickness_m))
        self._door_w = make_input(fmt(ctl.door_width_m))
        self._door_cap = make_input(str(ctl.door_capacity))
        self._selection = SelectionEditor(ctl)
        self._status = make_status()
        self._dropdown: gui.UIDropdown | None = None
        self._load_box = gui.UIBoxLayout(vertical=False)
        self.set_scenarios(scenarios)
        self.widget = self._build(on_save, on_load)

    # -- Construction ------------------------------------------------------

    def _tool_button(self, tool: Tool) -> gui.UIFlatButton:
        """Create a tool-selection button."""
        btn = gui.UIFlatButton(
            text=tool.value.title(),
            width=_TOOL_BUTTON_WIDTH_PX,
            height=ROW_HEIGHT_PX,
            style=button_style(),
        )
        btn.event("on_click")(lambda _e, t=tool: self.toggle_tool(t))
        return btn

    def toggle_tool(self, tool: Tool) -> None:
        """Toggle a tool like a radio button.

        Pressing a tool activates it (replacing any other); pressing the
        active tool again deactivates it. With no tool active the pointer
        behaves as Select.
        """
        active = self._ctl.tool is tool
        self._ctl.set_tool(Tool.SELECT if active else tool)

    def _room_fields(self, room: RoomSettings) -> dict[str, gui.UIInputText]:
        """Create the room-settings fields with current values."""
        values = {
            "name": room.name,
            "width_m": fmt(room.width_m),
            "height_m": fmt(room.height_m),
        }
        return {k: make_input(v) for k, v in values.items()}

    def _action(
        self,
        text: str,
        handler: Callable[[], None],
        width: int = CONTENT_WIDTH_PX,
    ) -> gui.UIFlatButton:
        """Create an action button (full width by default)."""
        btn = gui.UIFlatButton(
            text=text,
            width=width,
            height=ROW_HEIGHT_PX,
            style=button_style(),
        )
        btn.event("on_click")(lambda _e: handler())
        return btn

    def _build(
        self, on_save: Callable[[], None], on_load: Callable[[], None]
    ) -> gui.UIAnchorLayout:
        """Assemble the panel as two columns: setup (left), selection (right)."""
        columns = gui.UIBoxLayout(
            vertical=False,
            space_between=_COLUMN_GAP_PX,
            align="top",
            children=[
                self._column(self._setup_rows(on_load)),
                self._column(self._selection_rows(on_save)),
            ],
        )
        anchor = gui.UIAnchorLayout()
        anchor.add(
            columns,
            anchor_x="right",
            anchor_y="top",
            align_x=-_PANEL_MARGIN_PX,
            align_y=-_PANEL_MARGIN_PX,
        )
        return anchor

    def _setup_rows(self, on_load: Callable[[], None]) -> list[gui.UIWidget]:
        """Left column: open saved room, tools, room settings, new items."""
        load_btn = self._action("Load", on_load, _LOAD_BUTTON_WIDTH_PX)
        return [
            make_heading("Open saved room"),
            gui.UIBoxLayout(
                vertical=False,
                space_between=SPACING_PX,
                children=[self._load_box, load_btn],
            ),
            gap(),
            make_heading("Tools"),
            gui.UIBoxLayout(
                vertical=False,
                space_between=_TOOL_GAP_PX,
                children=list(self._tool_buttons.values()),
            ),
            gap(),
            make_heading("Room"),
            *[
                make_row(make_label(ROOM_LABELS[k]), f)
                for k, f in self._room_inputs.items()
            ],
            self._action("Apply room", self._apply_room),
            gap(),
            make_heading("New items"),
            make_row(make_label("Wall thickness"), self._wall_t),
            make_row(make_label("Door width"), self._door_w),
            make_row(make_label("Door capacity"), self._door_cap),
            *self._extra_rows,
        ]

    def _selection_rows(
        self, on_save: Callable[[], None]
    ) -> list[gui.UIWidget]:
        """Right column: selection editor, delete, save and status."""
        return [
            *self._selection.rows(),
            self._action("Apply selection", self._selection.apply),
            self._action("Delete selected (Del)", self._ctl.delete_selected),
            gap(),
            self._action("Save", on_save),
            self._status,
        ]

    def _column(self, rows: list[gui.UIWidget]) -> gui.UIBoxLayout:
        """Stack ``rows`` top-down, left-aligned."""
        return gui.UIBoxLayout(
            vertical=True,
            space_between=SPACING_PX,
            align="left",
            children=rows,
        )

    # -- Behaviour ---------------------------------------------------------

    def read_defaults(self) -> None:
        """Copy the new-item default fields into the controller."""
        thickness = parse_number(self._wall_t.text)
        if thickness is not None:
            self._ctl.wall_thickness_m = thickness
        width = parse_number(self._door_w.text)
        cap = parse_number(self._door_cap.text)
        if width is not None:
            self._ctl.door_width_m = width
        if cap is not None:
            self._ctl.door_capacity = cap

    def is_typing(self) -> bool:
        """Return True while any text field has keyboard focus."""
        boxes = [
            *self._room_inputs.values(),
            self._wall_t,
            self._door_w,
            self._door_cap,
        ]
        return self._selection.is_typing() or any(
            box.active for box in boxes
        )

    def _apply_room(self) -> None:
        """Parse the room fields and update the model."""
        texts = {k: f.text.strip() for k, f in self._room_inputs.items()}
        nums = {k: parse_number(v) for k, v in texts.items() if k != "name"}
        if any(v is None for v in nums.values()):
            self._ctl.status = "Room fields must be numbers"
            return
        applied = self._ctl.update_room(
            name=texts["name"],
            width_m=nums["width_m"],
            height_m=nums["height_m"],
        )
        if not applied:
            self.sync_room_fields()

    @property
    def selected_scenario(self) -> str:
        """Name chosen in the open-room dropdown (empty if none)."""
        if self._dropdown is None:
            return ""
        return str(self._dropdown.value or "")

    def set_scenarios(
        self, names: list[str], select: str | None = None
    ) -> None:
        """Rebuild the dropdown, e.g. after a save adds a file."""
        chosen = select or self.selected_scenario
        default = chosen if chosen in names else (names[0] if names else None)
        self._dropdown = gui.UIDropdown(
            default=default,
            options=list(names),
            width=CONTENT_WIDTH_PX - _LOAD_BUTTON_WIDTH_PX - SPACING_PX,
            height=ROW_HEIGHT_PX,
            primary_style=button_style(),
            dropdown_style=button_style(),
            active_style=button_style(True),
        )
        self._load_box.clear()  # type: ignore[no-untyped-call]
        self._load_box.add(self._dropdown)

    def sync_room_fields(self) -> None:
        """Show the room's real values in the Room fields."""
        room = self._ctl.model.layout.room
        self._room_inputs["name"].text = room.name
        self._room_inputs["width_m"].text = fmt(room.width_m)
        self._room_inputs["height_m"].text = fmt(room.height_m)

    def refresh(self) -> None:
        """Sync tool highlight, selection editors and status line."""
        for tool, btn in self._tool_buttons.items():
            active = tool is self._ctl.tool
            if active != self._active.get(tool, False):
                btn.style = button_style(active)
                self._active[tool] = active
        self._selection.sync()
        self._status.text = self._ctl.status
