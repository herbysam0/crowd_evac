"""Tests for the designer controller: numeric edits, X/Y, thickness.
"""
from __future__ import annotations

import pytest

from crowd_evac.designer.controller import DesignController, Tool

from .helpers import ROOM_W, bottom_wall_with_door, drag


class TestNumericEdit:
    """Panel edits of the selection."""

    def test_wall_length_edit_keeps_start(self, ctl: DesignController) -> None:
        """Verify the start stays fixed and the end moves."""
        drag(ctl, Tool.WALL, (2.0, 3.0), (6.0, 3.0))
        assert ctl.apply_edit("length", 7.5)
        wall = next(iter(ctl.model.layout.walls.values()))
        assert (wall.x0, wall.y0) == (2.0, 3.0)
        assert (wall.x1, wall.y1) == (9.5, 3.0)
        assert ctl.selected_values() == {
            "x": 2.0, "y": 3.0, "length": 7.5, "thickness": 0.5,
        }

    def test_invalid_length_rejected_state_intact(
        self, ctl: DesignController
    ) -> None:
        """Verify an over-long edit is rejected and reported."""
        drag(ctl, Tool.WALL, (2.0, 3.0), (6.0, 3.0))
        before = ctl.model.layout
        assert not ctl.apply_edit("length", 99)
        assert ctl.model.layout is before and ctl.status

    def test_unknown_field_raises(self, ctl: DesignController) -> None:
        """Verify editing a field the selection lacks is a programming error."""
        drag(ctl, Tool.WALL, (2.0, 3.0), (6.0, 3.0))
        with pytest.raises(ValueError):
            ctl.apply_edit("width", 1.0)

    def test_edit_without_selection_raises(
        self, ctl: DesignController
    ) -> None:
        """Verify there is nothing to edit without a selection."""
        with pytest.raises(ValueError):
            ctl.apply_edit("length", 1.0)

    def test_obstacle_size_edit(self, ctl: DesignController) -> None:
        """Verify obstacle width/height can be edited numerically."""
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (7.0, 5.0))
        assert ctl.apply_edit("width", 3.0)
        assert ctl.apply_edit("height", 2.0)
        obs = next(iter(ctl.model.layout.obstacles.values()))
        assert obs.rect == (5.0, 4.0, 3.0, 2.0)

    def test_doorway_width_and_capacity_edit(
        self, ctl: DesignController
    ) -> None:
        """Verify doorway width and (integer) capacity edits."""
        bottom_wall_with_door(ctl)
        assert ctl.apply_edit("width_m", 3.0)
        assert ctl.apply_edit("capacity_per_second", 7.0)
        door = next(iter(ctl.model.layout.doorways.values()))
        assert (door.width_m, door.capacity_per_second) == (3.0, 7)

    def test_update_room_reports_rejection(self, ctl: DesignController) -> None:
        """Verify room changes pass through with status on failure."""
        assert ctl.update_room(name="hall")
        drag(ctl, Tool.OBSTACLE, (10.0, 4.0), (12.0, 5.0))
        assert not ctl.update_room(width_m=8.0)
        assert ctl.model.layout.room.width_m == ROOM_W


class TestXYFields:
    """Selection panel shows and edits X/Y."""

    def test_wall_xy_edit_moves_wall(self, ctl: DesignController) -> None:
        """Verify editing a wall's X/Y translates it, length unchanged."""
        drag(ctl, Tool.WALL, (2.0, 3.0), (6.0, 3.0))
        assert ctl.apply_edit("x", 4.0) and ctl.apply_edit("y", 5.0)
        wall = next(iter(ctl.model.layout.walls.values()))
        assert (wall.x0, wall.y0, wall.x1, wall.y1) == (4.0, 5.0, 8.0, 5.0)

    def test_obstacle_xy_edit(self, ctl: DesignController) -> None:
        """Verify an obstacle's X/Y fields move it."""
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (7.0, 5.0))
        assert ctl.selected_values()["x"] == 5.0
        assert ctl.apply_edit("x", 8.0) and ctl.apply_edit("y", 2.0)
        obs = next(iter(ctl.model.layout.obstacles.values()))
        assert obs.rect == (8.0, 2.0, 2.0, 1.0)

    def test_doorway_shows_centre_xy(self, ctl: DesignController) -> None:
        """Verify a doorway reports its centre point."""
        bottom_wall_with_door(ctl)
        values = ctl.selected_values()
        assert (values["x"], values["y"]) == (5.0, 0.0)

    def test_doorway_x_edit_slides_and_y_edit_rejected(
        self, ctl: DesignController
    ) -> None:
        """Verify only the along-wall coordinate of a doorway is editable."""
        bottom_wall_with_door(ctl)
        assert ctl.apply_edit("x", 9.0)
        assert ctl.selected_values()["x"] == 9.0
        assert not ctl.apply_edit("y", 3.0)
        assert "slide along" in ctl.status


class TestWallThicknessControl:
    """Default and per-wall thickness through the controller."""

    def test_new_walls_use_default_thickness(
        self, ctl: DesignController
    ) -> None:
        """Verify the controller default is applied to drawn walls."""
        ctl.wall_thickness_m = 0.8
        drag(ctl, Tool.WALL, (5.0, 4.0), (9.0, 4.0))
        wall = next(iter(ctl.model.layout.walls.values()))
        assert wall.thickness == 0.8

    def test_rubber_band_carries_thickness(
        self, ctl: DesignController
    ) -> None:
        """Verify the live preview uses the chosen thickness."""
        ctl.wall_thickness_m = 0.3
        ctl.set_tool(Tool.WALL)
        ctl.press(2.0, 3.0)
        assert ctl.rubber_band is not None
        assert ctl.rubber_band.thickness == 0.3

    def test_panel_thickness_edit(self, ctl: DesignController) -> None:
        """Verify thickness is an editable selection field."""
        drag(ctl, Tool.WALL, (5.0, 4.0), (9.0, 4.0))
        assert ctl.selected_values()["thickness"] == 0.5
        assert ctl.apply_edit("thickness", 1.0)
        assert next(iter(ctl.model.layout.walls.values())).thickness == 1.0

    def test_invalid_thickness_edit_reports(
        self, ctl: DesignController
    ) -> None:
        """Verify a bad thickness shows the reason."""
        drag(ctl, Tool.WALL, (5.0, 4.0), (9.0, 4.0))
        assert not ctl.apply_edit("thickness", -1.0)
        assert "thickness" in ctl.status
