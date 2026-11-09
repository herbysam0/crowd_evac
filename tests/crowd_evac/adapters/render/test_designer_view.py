"""Tests for designer view helpers: viewport, labels, numbers, window fit.
"""
from __future__ import annotations

import pytest

from crowd_evac.adapters.render.designer_view import (
    band_label,
    band_preview_rect,
    fit_window_size,
    selection_rect,
)
from crowd_evac.adapters.render.designer_widgets import fmt, parse_number
from crowd_evac.designer.controller import DesignController, RubberBand
from crowd_evac.designer.placement import Selection

from .designer_helpers import ready_controller


class TestLabels:
    """Rubber band readouts."""

    def test_wall_label_is_length(self) -> None:
        """Verify the live wall readout shows metres."""
        band = RubberBand("wall", 0.0, 0.0, 4.3, 0.0)
        assert band_label(band) == "4.3 m"

    def test_obstacle_label_is_size(self) -> None:
        """Verify the obstacle readout shows width x height."""
        band = RubberBand("obstacle", 1.0, 1.0, 3.0, 2.5)
        assert band_label(band) == "2.0 x 1.5 m"

    def test_zero_length_wall_has_no_preview(self) -> None:
        """Verify no preview rectangle for a degenerate wall drag."""
        ctl = DesignController()
        band = RubberBand("wall", 2.0, 2.0, 2.0, 2.0)
        assert band_preview_rect(band, ctl.model.layout) is None


class TestSelectionRect:
    """Highlight rectangles."""

    def test_none_and_stale_selection(self) -> None:
        """Verify no rect for no/unknown selection."""
        ctl = DesignController()
        assert selection_rect(ctl.model, None) is None
        assert selection_rect(ctl.model, Selection("wall", 99)) is None

    def test_wall_selection_rect(self) -> None:
        """Verify a selected wall resolves to its drawn rectangle."""
        ctl = ready_controller()
        wall_id = next(iter(ctl.model.layout.walls))
        rect = selection_rect(ctl.model, Selection("wall", wall_id))
        assert rect == ctl.model.wall_rect(wall_id)


class TestNumbers:
    """Panel number helpers."""

    @pytest.mark.parametrize(
        "value, text", [(3.0, "3"), (0.1 * 3, "0.3"), (2.55, "2.55")]
    )
    def test_fmt(self, value: float, text: str) -> None:
        """Verify float noise is stripped."""
        assert fmt(value) == text

    @pytest.mark.parametrize(
        "text, expected", [("1.5", 1.5), (" 2 ", 2.0), ("abc", None), ("", None)]
    )
    def test_parse_number(self, text: str, expected: float | None) -> None:
        """Verify parsing and rejection."""
        assert parse_number(text) == expected


class TestFitWindowSize:
    """Window is sized to the display so nothing falls off-screen."""

    def test_shrinks_on_scaled_small_display(self) -> None:
        """A 1920x1200 display at 150% leaves ~1280x800 logical units."""
        width, height = fit_window_size((1920, 1200), 1.5)
        assert width * 1.5 <= 1920 and height * 1.5 <= 1200 - 60

    def test_keeps_desired_size_on_large_display(self) -> None:
        """A big unscaled display gets the preferred size."""
        assert fit_window_size((3840, 2160), 1.0) == (1320, 800)

    def test_never_below_minimum(self) -> None:
        """Tiny displays still get a usable window."""
        width, height = fit_window_size((800, 600), 1.0)
        assert width >= 900 and height >= 500


def test_draw_room_uses_shared_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """draw_room delegates the grid to draw_grid with viewport values."""
    from unittest.mock import MagicMock

    from crowd_evac.adapters.render import designer_view as dv
    from crowd_evac.adapters.render.viewport import Viewport

    grid = MagicMock()
    monkeypatch.setattr(dv, "draw_grid", grid)
    monkeypatch.setattr(dv.arcade, "draw_lrbt_rectangle_filled", MagicMock())
    model = ready_controller().model
    dv.draw_room(Viewport(ppm=20.0, ox=5.0, oy=7.0), model)
    room = model.layout.room
    grid.assert_called_once_with(room.width_m, room.height_m, 20.0, 5.0, 7.0)


def test_draw_floor_and_items_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grid belongs to draw_items so the photo sits between floor and grid."""
    from unittest.mock import MagicMock

    from crowd_evac.adapters.render import designer_view as dv
    from crowd_evac.adapters.render.viewport import Viewport

    grid = MagicMock()
    monkeypatch.setattr(dv, "draw_grid", grid)
    monkeypatch.setattr(dv.arcade, "draw_lrbt_rectangle_filled", MagicMock())
    model = ready_controller().model
    vp = Viewport(ppm=20.0, ox=5.0, oy=7.0)
    dv.draw_floor(vp, model)
    grid.assert_not_called()
    dv.draw_items(vp, model)
    grid.assert_called_once()
