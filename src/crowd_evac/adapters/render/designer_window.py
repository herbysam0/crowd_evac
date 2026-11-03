"""Arcade window for the room designer (``python -m crowd_evac --design``).

Forwards mouse input to the headless
:class:`~crowd_evac.designer.controller.DesignController`, draws the design
and hosts the :class:`~crowd_evac.adapters.render.designer_panel.DesignerPanel`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import arcade
import arcade.gui as gui

from crowd_evac.adapters.io.scenario_catalog import SCENARIOS_DIR
from crowd_evac.adapters.render.designer_io import (
    Pending,
    list_scenarios,
    load_design,
    save_design,
)
from crowd_evac.adapters.render.designer_panel import (
    PANEL_WIDTH_PX,
    DesignerPanel,
)
from crowd_evac.adapters.render.designer_style import (
    FONT_NAME,
    FONT_SIZE_CANVAS,
    FONT_SIZE_HINT,
    MUTED_COLOR,
    TEXT_COLOR,
)
from crowd_evac.adapters.render.designer_view import (
    CANVAS_MARGIN_PX,
    MIN_WINDOW_H,
    MIN_WINDOW_W,
    TOOL_HINTS,
    ALERT_BG_COLOR,
    NOTICE_BG_COLOR,
    draw_alert,
    draw_overlays,
    draw_room,
    fit_window_size,
)
from crowd_evac.adapters.render.viewport import Viewport, fit_viewport
from crowd_evac.designer.controller import DesignController, Tool

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Crowd Evac - Room Designer"
BACKGROUND_COLOR = (24, 24, 30, 255)
HINT_MARGIN_PX = 12
WINDOW_TOP_PX = 40
ALERT_SECONDS: float = 7.0
ALERT_COLOR = (255, 255, 255, 255)
MOUSE_BUTTON_LEFT = arcade.MOUSE_BUTTON_LEFT


class DesignerWindow(arcade.Window):
    """Interactive room designer window."""

    def __init__(
        self,
        ctl: DesignController | None = None,
        directory: Path = SCENARIOS_DIR,
    ) -> None:
        """Open the window.

        Args:
            ctl: Controller to edit; a fresh design by default.
            directory: Where Save writes scenarios.
        """
        super().__init__(
            MIN_WINDOW_W, MIN_WINDOW_H, WINDOW_TITLE, resizable=True
        )
        self._canvas_w = float(MIN_WINDOW_W - PANEL_WIDTH_PX)
        self._fit_to_screen()
        self.ctl = ctl or DesignController()
        self._directory = directory
        self._pending_save: Pending | None = None
        self._pending_load: Pending | None = None
        self._clean = self.ctl.model.layout
        self._canvas_w = float(self.width - PANEL_WIDTH_PX)
        self.vmap = self._fit()
        self._make_texts()
        self._alert_seq = 0
        self._notice_seq = 0
        self._banner_text = ""
        self._banner_color = ALERT_BG_COLOR
        self.alert_left = 0.0
        self.manager = gui.UIManager()
        self.panel = DesignerPanel(
            self.ctl, self._on_save, self._on_load, list_scenarios(directory)
        )
        self.manager.add(self.panel.widget)
        self.manager.enable()
        self.background_color = BACKGROUND_COLOR

    def _fit_to_screen(self) -> None:
        """Shrink and centre the window so it lies fully on the display."""
        screen = self.screen
        if screen is None:
            return
        ratio = self.get_pixel_ratio()
        size = fit_window_size((screen.width, screen.height), ratio)
        self.set_size(*size)
        # Apply the resize now so the viewport and UI match the new size.
        self.dispatch_events()
        self.set_location(
            max(0, int((screen.width - size[0] * ratio) / 2)), WINDOW_TOP_PX
        )

    def _make_texts(self) -> None:
        """Create the cached text objects used by the canvas."""
        self._label = arcade.Text(
            "", 0, 0, TEXT_COLOR, FONT_SIZE_CANVAS,
            font_name=FONT_NAME, bold=True,
        )
        self._alert_text = arcade.Text(
            "", 0, 0, ALERT_COLOR, FONT_SIZE_CANVAS, font_name=FONT_NAME,
            bold=True, multiline=True, width=int(self._canvas_w) - 28,
            anchor_y="top",
        )
        self._hint = arcade.Text(
            "", HINT_MARGIN_PX, HINT_MARGIN_PX, MUTED_COLOR, FONT_SIZE_HINT,
            font_name=FONT_NAME,
        )

    def on_resize(self, width: int, height: int) -> Literal[True] | None:
        """Keep the canvas and banner matched to the window size."""
        result = super().on_resize(width, height)
        self._canvas_w = float(max(width - PANEL_WIDTH_PX, 1))
        alert = getattr(self, "_alert_text", None)
        if alert is not None:
            alert.width = int(self._canvas_w) - 2 * HINT_MARGIN_PX
        return result

    def _fit(self) -> Viewport:
        """Compute the viewport for the current room size."""
        room = self.ctl.model.layout.room
        return fit_viewport(
            room.width_m,
            room.height_m,
            self._canvas_w,
            self.height,
            margin_px=CANVAS_MARGIN_PX,
        )

    def _on_save(self) -> None:
        """Save button handler; refreshes the open-room list on success."""
        self._pending_save, saved = save_design(
            self.ctl, self._pending_save, self._directory
        )
        if saved:
            self._clean = self.ctl.model.layout
            self.panel.set_scenarios(
                list_scenarios(self._directory), self.ctl.model.layout.room.name
            )

    def _on_load(self) -> None:
        """Load button handler; asks first if there are unsaved edits."""
        self._pending_load, loaded = load_design(
            self.ctl,
            self.panel.selected_scenario,
            self._pending_load,
            self._clean,
            self._directory,
        )
        if loaded:
            self._clean = self.ctl.model.layout
            self.panel.sync_room_fields()

    def on_update(self, delta_time: float) -> None:
        """Start the alert timer on a new rejection and count it down."""
        if self.ctl.alert_seq != self._alert_seq:
            self._alert_seq = self.ctl.alert_seq
            self._banner_text = self.ctl.alert
            self._banner_color = ALERT_BG_COLOR
            self.alert_left = ALERT_SECONDS
        elif self.ctl.notice_seq != self._notice_seq:
            self._notice_seq = self.ctl.notice_seq
            self._banner_text = self.ctl.notice
            self._banner_color = NOTICE_BG_COLOR
            self.alert_left = ALERT_SECONDS
        elif self.alert_left > 0.0:
            self.alert_left = max(0.0, self.alert_left - delta_time)

    def on_draw(self) -> None:
        """Draw canvas, overlays and panel."""
        self.clear()
        self.vmap = self._fit()
        self.panel.refresh()
        draw_room(self.vmap, self.ctl.model)
        draw_overlays(
            self.vmap,
            self.ctl.model,
            self.ctl.selection,
            self.ctl.rubber_band,
            self._label,
        )
        if self.alert_left > 0.0 and self._banner_text:
            self._alert_text.text = self._banner_text
            draw_alert(
                self._alert_text, self._canvas_w, self.height,
                self._banner_color,
            )
        self._hint.text = TOOL_HINTS[self.ctl.tool]
        self._hint.draw()
        self.manager.draw()

    def _world(self, px: float, py: float) -> tuple[float, float] | None:
        """Convert a canvas click to metres; None over the panel."""
        if px >= self._canvas_w:
            return None
        return self.vmap.to_world(px, py)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        """Forward a left press on the canvas to the controller."""
        world = self._world(x, y)
        if button != MOUSE_BUTTON_LEFT or world is None:
            return
        self.panel.read_defaults()
        self.ctl.press(*world)

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> None:
        """Forward drags (the controller clamps to the room)."""
        self.ctl.drag(*self.vmap.to_world(x, y))

    def on_mouse_release(
        self, x: int, y: int, button: int, modifiers: int
    ) -> None:
        """Forward a left release to the controller."""
        if button == MOUSE_BUTTON_LEFT:
            self.ctl.release(*self.vmap.to_world(x, y))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """Delete removes the selection (not while typing); Esc selects."""
        if symbol == arcade.key.DELETE and not self.panel.is_typing():
            self.ctl.delete_selected()
        elif symbol == arcade.key.ESCAPE:
            self.ctl.set_tool(Tool.SELECT)


def run_designer() -> None:
    """Open the designer and block until the window is closed."""
    DesignerWindow()
    arcade.run()
