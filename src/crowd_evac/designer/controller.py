"""Tool state machine translating pointer events into model edits.

Headless: the arcade window forwards mouse events in world metres and
draws ``rubber_band``, ``selection`` and ``status``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

from crowd_evac.designer.geometry import DEFAULT_WALL_THICKNESS_M, Rect
from crowd_evac.designer.layout import Layout
from crowd_evac.designer.model import DEFAULT_DOOR_CAPACITY, DesignModel
from crowd_evac.designer.mover import MoveDrag
from crowd_evac.designer.selection_edit import apply_field, selected_values
from crowd_evac.designer.placement import (
    Selection,
    axis_end,
    clamp,
    door_center,
    hit_test,
    snap,
)
from crowd_evac.domain.errors import DesignValidationError

DEFAULT_DOOR_WIDTH_M: float = 0.85


class Tool(Enum):
    """Active editing tool."""

    SELECT = "select"
    WALL = "wall"
    DOORWAY = "doorway"
    OBSTACLE = "obstacle"


@dataclass(frozen=True)
class RubberBand:
    """In-progress drag shown while drawing a wall or obstacle."""

    kind: Literal["wall", "obstacle"]
    x0: float
    y0: float
    x1: float
    y1: float
    thickness: float = DEFAULT_WALL_THICKNESS_M

    @property
    def length_m(self) -> float:
        """Live length readout (wall) in metres."""
        return abs(self.x1 - self.x0) + abs(self.y1 - self.y0)

    @property
    def rect(self) -> Rect:
        """Normalised rectangle spanned by the two corners."""
        return (
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            abs(self.x1 - self.x0),
            abs(self.y1 - self.y0),
        )


class DesignController:
    """Drives a :class:`DesignModel` from pointer and panel input.

    Attributes:
        model: The design being edited.
        tool: Active tool.
        selection: Selected entity or None.
        rubber_band: Drag preview or None.
        status: Last message for the status line.
        alert: Last room-change rejection, shown as a loud banner.
        alert_seq: Incremented on every new alert so repeats re-trigger.
        notice: Last success message, shown as a green banner.
        notice_seq: Incremented on every new notice.
        wall_thickness_m: Thickness used for new walls.
        door_width_m: Width used for new doorways.
        door_capacity: Capacity used for new doorways.
    """

    def __init__(self, model: DesignModel | None = None) -> None:
        """Create a controller, with a fresh model by default."""
        self.model = model or DesignModel()
        self.tool = Tool.SELECT
        self.selection: Selection | None = None
        self.rubber_band: RubberBand | None = None
        self._move: MoveDrag | None = None
        self.status = ""
        self.alert = ""
        self.alert_seq = 0
        self.notice = ""
        self.notice_seq = 0
        self.wall_thickness_m = DEFAULT_WALL_THICKNESS_M
        self.door_width_m = DEFAULT_DOOR_WIDTH_M
        self.door_capacity = DEFAULT_DOOR_CAPACITY

    def set_tool(self, tool: Tool) -> None:
        """Switch tool, cancelling any drag in progress."""
        self.tool = tool
        self.rubber_band = None
        self._move = None

    # -- Pointer events (world metres) -------------------------------------

    def press(self, x: float, y: float) -> None:
        """Handle mouse press."""
        if self.tool in (Tool.WALL, Tool.OBSTACLE):
            px, py = self._snapped(x, y)
            kind: Literal["wall", "obstacle"] = (
                "wall" if self.tool is Tool.WALL else "obstacle"
            )
            self.rubber_band = RubberBand(
                kind, px, py, px, py, self.wall_thickness_m
            )
        elif self.tool is Tool.DOORWAY:
            self._place_door(x, y)
        else:
            self.selection = hit_test(self.model.layout, x, y)
            self.status = ""
            self._move = None
            if self.selection is not None:
                self._move = MoveDrag(
                    self.model, self.selection, *self._snapped(x, y)
                )

    def drag(self, x: float, y: float) -> None:
        """Handle mouse drag; updates the rubber band."""
        px, py = self._snapped(x, y)
        if self._move is not None:
            self.status = self._move.update(px, py) or ""
            return
        band = self.rubber_band
        if band is None:
            return
        if band.kind == "wall":
            px, py = axis_end(band.x0, band.y0, px, py)
        self.rubber_band = replace(band, x1=px, y1=py)

    def release(self, x: float, y: float) -> None:
        """Handle mouse release; commits the drawn entity."""
        self.drag(x, y)
        move, self._move = self._move, None
        if move is not None and move.moved:
            self.status = "Moved"
        band, self.rubber_band = self.rubber_band, None
        if band is None:
            return
        if band.kind == "wall":
            self._try(self._add_wall, band)
        else:
            self._try(self._add_obstacle, band)

    # -- Panel actions -----------------------------------------------------

    def delete_selected(self) -> None:
        """Delete the selected entity (a wall takes its doorways along)."""
        sel = self.selection
        if sel is None:
            self.status = "Nothing selected"
            return
        if self._try(self._delete, sel):
            self.selection = None

    def selected_values(self) -> dict[str, float]:
        """Return the numeric fields of the selection for the panel."""
        return selected_values(self.model, self.selection)

    def apply_edit(self, field: str, value: float) -> bool:
        """Numerically edit one field of the selection.

        Args:
            field: A key of :meth:`selected_values`.
            value: New value.

        Returns:
            True if applied; False if rejected (see ``status``).
        """
        if field not in self.selected_values():
            raise ValueError(f"Cannot edit {field!r} of the selection")
        return self._try(self._edit, (field, value))

    def update_room(self, **changes: Any) -> bool:
        """Change room settings; entities never move.

        Returns:
            True if applied. False if rejected: ``status`` and ``alert``
            then explain which entity blocks the new size.
        """
        room = self.model.layout.room
        width = changes.get("width_m", room.width_m)
        height = changes.get("height_m", room.height_m)
        if self._try(lambda c: self.model.update_room(**c), changes):
            self.alert = ""
            return True
        self.show_alert(
            f"ROOM SIZE REJECTED - cannot resize to {width:g} x {height:g} m: "
            f"{self.status}"
        )
        return False

    def show_alert(self, message: str) -> None:
        """Raise a loud on-screen alert (re-triggers even if repeated)."""
        self.alert = message
        self.alert_seq += 1

    def show_notice(self, message: str) -> None:
        """Show a green confirmation banner (e.g. after a save)."""
        self.notice = message
        self.notice_seq += 1

    def load_layout(self, layout: Layout) -> None:
        """Replace the whole design with ``layout`` (e.g. from a file).

        Raises:
            DesignValidationError: If the layout is invalid; the current
                design is kept.
        """
        self.model = DesignModel(layout=layout)
        self.selection = None
        self.rubber_band = None
        self._move = None
        self.alert = ""
        self.status = f"Loaded {layout.room.name}"

    # -- Internals ---------------------------------------------------------

    def _snapped(self, x: float, y: float) -> tuple[float, float]:
        """Snap a point to the grid and keep it inside the room."""
        room = self.model.layout.room
        return (
            clamp(snap(x), 0.0, room.width_m),
            clamp(snap(y), 0.0, room.height_m),
        )

    def _try(self, action: Any, arg: Any) -> bool:
        """Run a model action; on rejection show the reason, keep state."""
        try:
            action(arg)
        except DesignValidationError as exc:
            self.status = str(exc)
            return False
        return True

    def _add_wall(self, band: RubberBand) -> None:
        """Commit a wall drag and select the new wall."""
        wall = self.model.add_wall(
            band.x0, band.y0, band.x1, band.y1, band.thickness
        )
        self.selection = Selection("wall", wall.id)
        self.status = f"Added wall ({wall.length:.2f} m)"

    def _add_obstacle(self, band: RubberBand) -> None:
        """Commit an obstacle drag and select the new obstacle."""
        x, y, w, h = band.rect
        obs = self.model.add_obstacle(x, y, w, h)
        self.selection = Selection("obstacle", obs.id)
        self.status = "Added obstacle"

    def _place_door(self, x: float, y: float) -> None:
        """Place a doorway on the clicked perimeter wall."""
        hit = hit_test(self.model.layout, x, y)
        if hit is None or hit.kind != "wall":
            self.status = "Click a wall on the room boundary"
            return
        wall = self.model.layout.walls[hit.id]
        centre = door_center(
            self.model.layout, wall, x, y, self.door_width_m
        )
        self._try(self._add_door, (wall.id, centre))

    def _add_door(self, args: tuple[int, float]) -> None:
        """Add the doorway and select it."""
        door = self.model.add_doorway(
            args[0], args[1], self.door_width_m, self.door_capacity
        )
        self.selection = Selection("doorway", door.id)
        self.status = "Added doorway"

    def _delete(self, sel: Selection) -> None:
        """Delete the entity and report cascades."""
        if sel.kind == "wall":
            n = self.model.delete_wall(sel.id)
            self.status = f"Deleted wall and {n} doorway(s)"
        elif sel.kind == "doorway":
            self.model.delete_doorway(sel.id)
            self.status = "Deleted doorway"
        else:
            self.model.delete_obstacle(sel.id)
            self.status = "Deleted obstacle"

    def _edit(self, change: tuple[str, float]) -> None:
        """Apply a numeric edit to the selection."""
        assert self.selection is not None
        apply_field(self.model, self.selection, *change)
        self.status = "Updated"
