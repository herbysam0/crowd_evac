"""Numeric editor widgets for the currently selected entity."""
from __future__ import annotations

import arcade.gui as gui

from crowd_evac.adapters.render.designer_labels import ANCHORS, FIELD_LABELS
from crowd_evac.adapters.render.designer_widgets import (
    CONTENT_WIDTH_PX,
    FIELD_WIDTH_PX,
    fmt,
    make_heading,
    make_input,
    make_label,
    make_row,
    parse_number,
)
from crowd_evac.designer.controller import DesignController
from crowd_evac.designer.layout import Layout
from crowd_evac.designer.placement import Selection

MAX_EDIT_SLOTS = 4
_EPS = 1e-9


class SelectionEditor:
    """Title plus a few label/field slots showing the selection's values.

    The slots are filled whenever the selection or the layout changes, and
    :meth:`apply` pushes edited values back through the controller.
    """

    def __init__(self, ctl: DesignController) -> None:
        """Create the widgets.

        Args:
            ctl: Controller whose selection is shown and edited.
        """
        self._ctl = ctl
        self._synced: tuple[Selection | None, Layout] | None = None
        self._title = make_label("Nothing selected", CONTENT_WIDTH_PX)
        self.slots = [
            (make_label(""), make_input("", FIELD_WIDTH_PX))
            for _ in range(MAX_EDIT_SLOTS)
        ]

    def rows(self) -> list[gui.UIWidget]:
        """Return the heading, title and slot rows for layout."""
        return [
            make_heading("Selection"),
            self._title,
            *[make_row(label, box) for label, box in self.slots],
        ]

    def is_typing(self) -> bool:
        """Return True while any slot field has keyboard focus."""
        return any(box.active for _, box in self.slots)

    def invalidate(self) -> None:
        """Force the slots to refill on the next :meth:`sync`."""
        self._synced = None

    def apply(self) -> None:
        """Apply every changed field; stop at the first failure."""
        current = self._ctl.selected_values()
        for field, (_label, box) in zip(current, self.slots):
            value = parse_number(box.text)
            if value is None:
                self._ctl.status = f"{FIELD_LABELS[field]} must be a number"
                return
            if abs(value - current[field]) > _EPS and not self._ctl.apply_edit(
                field, value
            ):
                return
        self.invalidate()

    def sync(self) -> None:
        """Refill the slots if the selection or layout changed."""
        key = (self._ctl.selection, self._ctl.model.layout)
        if self._synced is not None and key == self._synced:
            return
        self._synced = key
        values = self._ctl.selected_values()
        sel = self._ctl.selection
        self._title.text = (
            f"{sel.kind.title()} {sel.id}  (X, Y = {ANCHORS[sel.kind]})"
            if sel
            else "Nothing selected"
        )
        fields = list(values)
        for i, (label, box) in enumerate(self.slots):
            if i < len(fields):
                label.text = FIELD_LABELS[fields[i]]
                box.text = fmt(values[fields[i]])
            else:
                label.text, box.text = "", ""
