"""Render tests for the designer window (need a display).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from crowd_evac.adapters.render.designer_panel import PANEL_WIDTH_PX
from crowd_evac.designer.controller import DesignController, Tool

from .designer_helpers import ready_controller


@pytest.mark.render
class TestDesignerWindowGL:
    """Real-window smoke test (needs a display)."""

    def test_window_draws_and_handles_input(self, tmp_path: Path) -> None:
        """Open the window, draw a design plus a drag, then close."""
        from crowd_evac.adapters.render.designer_window import DesignerWindow

        window = DesignerWindow(ready_controller(), tmp_path)
        try:
            window.ctl.set_tool(Tool.OBSTACLE)
            px, py = window.vmap.to_px(5.0, 5.0)
            window.on_mouse_press(int(px), int(py), 1, 0)
            window.on_mouse_drag(int(px) + 40, int(py) + 40, 0, 0, 1, 0)
            window.on_draw()
            window.on_mouse_release(int(px) + 40, int(py) + 40, 1, 0)
            window.on_draw()
            assert len(window.ctl.model.layout.obstacles) == 1
            window._on_save()
            assert (tmp_path / "saved_room.json").exists()
        finally:
            window.close()

    def test_panel_buttons_and_fields_drive_controller(
        self, tmp_path: Path
    ) -> None:
        """Click tool/apply buttons through arcade.gui events."""
        import arcade.gui as gui

        from crowd_evac.adapters.render.designer_window import DesignerWindow

        def click(button: gui.UIFlatButton) -> None:
            button.dispatch_event(
                "on_click", gui.UIOnClickEvent(button, 0, 0, 1, 0)
            )

        window = DesignerWindow(ready_controller(), tmp_path)
        try:
            panel = window.panel
            click(panel._tool_buttons[Tool.WALL])
            assert window.ctl.tool is Tool.WALL
            window.ctl.set_tool(Tool.SELECT)
            window.ctl.press(*window.vmap.to_world(*window.vmap.to_px(3.0, 0.25)))
            window.on_draw()
            assert [b.text for _, b in panel._selection.slots[:3]] == ["0", "0", "15"]
            panel._selection.slots[2][1].text = "10"
            panel._selection.apply()
            assert window.ctl.status == "Updated"
            panel._room_inputs["width_m"].text = "16"
            panel._apply_room()
            assert window.ctl.model.layout.room.width_m == 16.0
        finally:
            window.close()

    def test_rejected_resize_shows_alert_and_resets_fields(
        self, tmp_path: Path
    ) -> None:
        """A blocked resize raises the banner, keeps the room, resets fields."""
        from crowd_evac.adapters.render.designer_window import (
            ALERT_SECONDS,
            DesignerWindow,
        )

        ctl = ready_controller()
        ctl.set_tool(Tool.OBSTACLE)
        ctl.press(10.0, 4.0)
        ctl.release(12.0, 5.0)
        window = DesignerWindow(ctl, tmp_path)
        try:
            panel = window.panel
            panel._room_inputs["width_m"].text = "8"
            panel._apply_room()
            window.on_update(0.016)
            assert window.alert_left == ALERT_SECONDS
            window.on_draw()
            assert panel._room_inputs["width_m"].text == "15"
            assert ctl.model.layout.room.width_m == 15.0
            window.on_update(ALERT_SECONDS + 1)
            assert window.alert_left == 0.0
        finally:
            window.close()

    def test_canvas_rescales_after_resize(self, tmp_path: Path) -> None:
        """Verify the canvas mapping follows an accepted resize."""
        from crowd_evac.adapters.render.designer_window import DesignerWindow

        window = DesignerWindow(ready_controller(), tmp_path)
        try:
            window.on_draw()
            before = window.vmap.ppm
            window.panel._room_inputs["width_m"].text = "30"
            window.panel._apply_room()
            window.on_draw()
            assert window.vmap.ppm < before
        finally:
            window.close()


@pytest.mark.render
class TestLoadGL:
    """Window-level load and save refresh."""

    def test_load_button_flow_and_save_refreshes_list(
        self, tmp_path: Path
    ) -> None:
        """Load a bundled room via the panel, edit, save, see it listed."""
        import shutil

        from crowd_evac.adapters.io.scenario_catalog import SCENARIOS_DIR
        from crowd_evac.adapters.render.designer_window import DesignerWindow

        shutil.copy(SCENARIOS_DIR / "fixture_minimal.json", tmp_path)
        window = DesignerWindow(DesignController(), tmp_path)
        try:
            assert window.panel.selected_scenario == "fixture_minimal"
            window._on_load()
            assert window.ctl.model.layout.room.name == "fixture_minimal"
            assert window.panel._room_inputs["name"].text == "fixture_minimal"
            window.ctl.update_room(name="edited_copy")
            window._on_save()
            assert (tmp_path / "edited_copy.json").exists()
            assert window.panel.selected_scenario == "edited_copy"
            window.on_draw()
        finally:
            window.close()


@pytest.mark.render
class TestToolToggleGL:
    """Radio-toggle behaviour of the tool buttons."""

    def test_no_select_button_and_radio_toggle(self, tmp_path: Path) -> None:
        """Tools are exclusive; pressing the active one returns to Select."""
        import arcade.gui as gui

        from crowd_evac.adapters.render.designer_window import DesignerWindow

        def click(button: gui.UIFlatButton) -> None:
            button.dispatch_event(
                "on_click", gui.UIOnClickEvent(button, 0, 0, 1, 0)
            )

        window = DesignerWindow(DesignController(), tmp_path)
        try:
            buttons = window.panel._tool_buttons
            assert Tool.SELECT not in buttons
            assert set(buttons) == {Tool.WALL, Tool.DOORWAY, Tool.OBSTACLE}
            click(buttons[Tool.WALL])
            assert window.ctl.tool is Tool.WALL
            click(buttons[Tool.OBSTACLE])
            assert window.ctl.tool is Tool.OBSTACLE
            click(buttons[Tool.OBSTACLE])
            assert window.ctl.tool is Tool.SELECT
            window.on_draw()
            assert not any(window.panel._active.values())
        finally:
            window.close()

    def test_escape_deselects_tool(self, tmp_path: Path) -> None:
        """Verify Escape returns to the no-tool Select behaviour."""
        import arcade

        from crowd_evac.adapters.render.designer_window import DesignerWindow

        window = DesignerWindow(DesignController(), tmp_path)
        try:
            window.panel.toggle_tool(Tool.DOORWAY)
            window.on_key_press(arcade.key.ESCAPE, 0)
            assert window.ctl.tool is Tool.SELECT
        finally:
            window.close()


@pytest.mark.render
class TestWindowFitsDisplayGL:
    """The real window must lie fully on the screen."""

    def test_window_and_panel_within_screen(self, tmp_path: Path) -> None:
        """Verify the framebuffer rectangle is inside the screen."""
        from crowd_evac.adapters.render.designer_window import DesignerWindow

        window = DesignerWindow(DesignController(), tmp_path)
        try:
            screen = window.screen
            assert screen is not None
            x, y = window.get_location()
            fb_w, fb_h = window.get_framebuffer_size()
            assert x + fb_w <= screen.width
            assert y + fb_h <= screen.height
            assert window._canvas_w == window.width - PANEL_WIDTH_PX
        finally:
            window.close()

    def test_resize_keeps_canvas_matched(self, tmp_path: Path) -> None:
        """Verify on_resize updates the canvas width."""
        from crowd_evac.adapters.render.designer_window import DesignerWindow

        window = DesignerWindow(DesignController(), tmp_path)
        try:
            window.on_resize(1000, 700)
            assert window._canvas_w == 1000 - PANEL_WIDTH_PX
        finally:
            window.close()


@pytest.mark.render
class TestSaveBannerGL:
    """Banners appear on screen after Save."""

    def test_success_and_failure_banners(self, tmp_path: Path) -> None:
        """Verify the window shows a green then a red banner."""
        from crowd_evac.adapters.render.designer_view import (
            ALERT_BG_COLOR,
            NOTICE_BG_COLOR,
        )
        from crowd_evac.adapters.render.designer_window import (
            ALERT_SECONDS,
            DesignerWindow,
        )

        window = DesignerWindow(ready_controller(), tmp_path)
        try:
            window._on_save()
            window.on_update(0.016)
            assert window._banner_color == NOTICE_BG_COLOR
            assert window.alert_left == ALERT_SECONDS
            window.on_draw()
            window.ctl.update_room(name="Bad Name")
            window._on_save()
            window.on_update(0.016)
            assert window._banner_color == ALERT_BG_COLOR
            assert "NOT SAVED" in window._banner_text
            window.on_draw()
        finally:
            window.close()


@pytest.mark.render
class TestDesignerUnderlayGL:
    """Photo underlay through the real window with a fake picker."""

    @staticmethod
    def _png(tmp_path: Path) -> Path:
        from PIL import Image

        png = tmp_path / "plan.png"
        Image.new("RGB", (40, 20), (200, 10, 10)).save(png)
        return png

    def test_load_toggle_opacity_and_clean_save(self, tmp_path: Path) -> None:
        """Photo loads, toggles via U, steps opacity, never touches JSON."""
        import arcade

        from crowd_evac.adapters.render.designer_window import DesignerWindow

        png = self._png(tmp_path)
        out_a, out_b = tmp_path / "a", tmp_path / "b"
        out_a.mkdir()
        out_b.mkdir()
        window = DesignerWindow(ready_controller(), out_a, lambda: png)
        try:
            clean = window._clean
            window._on_save()
            seq = window.ctl.notice_seq
            window.underlay.load()
            assert window.underlay.underlay is not None
            assert window.ctl.notice_seq == seq + 1
            window.on_key_press(arcade.key.U, 0)
            assert window.underlay.underlay.visible is False
            window.on_key_press(arcade.key.U, 0)
            window.underlay.step_opacity(1)
            assert window.underlay.underlay.opacity == pytest.approx(0.6)
            window.on_draw()
            assert window.ctl.model.layout == clean
            window._directory = out_b
            window._pending_save = None
            window._on_save()
            assert (out_a / "saved_room.json").read_text(
                encoding="utf-8"
            ) == (out_b / "saved_room.json").read_text(encoding="utf-8")
            assert "plan.png" not in (out_b / "saved_room.json").read_text(
                encoding="utf-8"
            )
        finally:
            window.close()

    def test_cancel_noop_and_bad_file_alerts(self, tmp_path: Path) -> None:
        """Cancel changes nothing; unreadable file raises an alert."""
        from crowd_evac.adapters.render.designer_window import DesignerWindow

        bad = tmp_path / "bad.png"
        bad.write_text("nope", encoding="utf-8")
        choice: list[Path | None] = [None]
        window = DesignerWindow(
            ready_controller(), tmp_path, lambda: choice[0]
        )
        try:
            window.underlay.load()
            assert window.underlay.underlay is None
            assert window.ctl.alert_seq == 0
            choice[0] = bad
            window.underlay.load()
            assert window.underlay.underlay is None
            assert window.ctl.alert.startswith("PHOTO NOT LOADED")
        finally:
            window.close()

    def test_u_key_ignored_while_typing(self, tmp_path: Path) -> None:
        """U does not toggle while a text field has focus."""
        import arcade

        from crowd_evac.adapters.render.designer_window import DesignerWindow

        png = self._png(tmp_path)
        window = DesignerWindow(ready_controller(), tmp_path, lambda: png)
        try:
            window.underlay.load()
            window.panel.is_typing = lambda: True  # type: ignore[method-assign]
            window.on_key_press(arcade.key.U, 0)
            assert window.underlay.underlay is not None
            assert window.underlay.underlay.visible is True
        finally:
            window.close()
