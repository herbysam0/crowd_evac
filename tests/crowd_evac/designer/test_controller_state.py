"""Tests for the designer controller: selection, move, resize alert, load.
"""
from __future__ import annotations

from crowd_evac.designer.controller import DesignController, Tool

from .helpers import ROOM_W, bottom_wall_with_door, drag


class TestSelectAndDelete:
    """Selection priority and deletion."""

    def test_click_empty_clears_selection(self, ctl: DesignController) -> None:
        """Verify clicking nothing deselects."""
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (7.0, 5.0))
        ctl.set_tool(Tool.SELECT)
        ctl.press(12.0, 8.0)
        assert ctl.selection is None

    def test_doorway_selected_over_wall(self, ctl: DesignController) -> None:
        """Verify a doorway on a wall wins the click."""
        bottom_wall_with_door(ctl)
        ctl.set_tool(Tool.SELECT)
        ctl.press(5.0, 0.25)
        assert ctl.selection is not None and ctl.selection.kind == "doorway"
        ctl.press(12.0, 0.25)
        assert ctl.selection is not None and ctl.selection.kind == "wall"

    def test_delete_wall_reports_doorway_count(
        self, ctl: DesignController
    ) -> None:
        """Verify deleting a wall cascades and says how many doorways."""
        bottom_wall_with_door(ctl)
        ctl.set_tool(Tool.SELECT)
        ctl.press(12.0, 0.25)
        ctl.delete_selected()
        assert not ctl.model.layout.walls and not ctl.model.layout.doorways
        assert "1 doorway" in ctl.status and ctl.selection is None

    def test_delete_doorway_and_obstacle(self, ctl: DesignController) -> None:
        """Verify doorway and obstacle deletion."""
        bottom_wall_with_door(ctl)
        ctl.delete_selected()
        assert not ctl.model.layout.doorways
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (7.0, 5.0))
        ctl.delete_selected()
        assert not ctl.model.layout.obstacles

    def test_delete_without_selection(self, ctl: DesignController) -> None:
        """Verify delete with nothing selected only sets a status."""
        ctl.delete_selected()
        assert ctl.status == "Nothing selected"


class TestMove:
    """Drag-to-move in the Select tool."""

    def test_drag_moves_obstacle(self, ctl: DesignController) -> None:
        """Verify pressing an obstacle and dragging translates it."""
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (7.0, 5.0))
        drag(ctl, Tool.SELECT, (6.0, 4.5), (8.0, 6.5))
        obs = next(iter(ctl.model.layout.obstacles.values()))
        assert obs.rect == (7.0, 6.0, 2.0, 1.0)
        assert ctl.status == "Moved"

    def test_drag_moves_wall(self, ctl: DesignController) -> None:
        """Verify a wall is moved by the pointer delta."""
        drag(ctl, Tool.WALL, (5.0, 4.0), (9.0, 4.0))
        drag(ctl, Tool.SELECT, (7.0, 4.0), (7.0, 6.0))
        wall = next(iter(ctl.model.layout.walls.values()))
        assert (wall.y0, wall.y1) == (6.0, 6.0)

    def test_drag_doorway_slides_along_wall_only(
        self, ctl: DesignController
    ) -> None:
        """Verify doorway moves follow the wall axis, ignoring the other."""
        bottom_wall_with_door(ctl)
        drag(ctl, Tool.SELECT, (5.0, 0.25), (8.0, 4.0))
        door = next(iter(ctl.model.layout.doorways.values()))
        assert door.center == 8.0

    def test_blocked_move_slides_along_boundary(
        self, ctl: DesignController
    ) -> None:
        """Verify a rejected diagonal step still moves on the valid axis."""
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (7.0, 5.0))
        ctl.set_tool(Tool.SELECT)
        ctl.press(6.0, 4.5)
        ctl.drag(6.0 + 20.0, 4.5 + 1.0)  # x clamps at the room edge
        ctl.release(6.0 + 20.0, 4.5 + 1.0)
        obs = next(iter(ctl.model.layout.obstacles.values()))
        assert obs.rect[0] + obs.rect[2] <= ROOM_W
        assert obs.rect[1] == 5.0

    def test_move_into_wall_rejected_with_reason(
        self, ctl: DesignController
    ) -> None:
        """Verify an overlapping move keeps the old spot and reports why."""
        drag(ctl, Tool.WALL, (0.0, 0.0), (ROOM_W, 0.0))
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (7.0, 5.0))
        ctl.set_tool(Tool.SELECT)
        ctl.press(6.0, 4.5)
        ctl.drag(6.0, 0.8)
        obs = next(iter(ctl.model.layout.obstacles.values()))
        assert obs.rect[1] > 0.0 and "overlap" in ctl.status

    def test_press_on_empty_does_not_move(self, ctl: DesignController) -> None:
        """Verify dragging from empty space changes nothing."""
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (7.0, 5.0))
        before = ctl.model.layout
        drag(ctl, Tool.SELECT, (12.0, 8.0), (13.0, 9.0))
        assert ctl.model.layout is before


class TestRoomResizeAlert:
    """Loud rejection of room-size changes."""

    def test_rejection_sets_alert_naming_size_and_entity(
        self, ctl: DesignController
    ) -> None:
        """Verify the alert states the attempted size and the blocker."""
        drag(ctl, Tool.OBSTACLE, (10.0, 4.0), (12.0, 5.0))
        assert not ctl.update_room(width_m=8.0)
        assert "ROOM SIZE REJECTED" in ctl.alert
        assert "8 x 10 m" in ctl.alert and "Obstacle" in ctl.alert
        assert ctl.alert_seq == 1
        assert ctl.model.layout.room.width_m == ROOM_W

    def test_repeat_rejection_retriggers(self, ctl: DesignController) -> None:
        """Verify an identical rejection bumps the sequence again."""
        drag(ctl, Tool.OBSTACLE, (10.0, 4.0), (12.0, 5.0))
        ctl.update_room(width_m=8.0)
        ctl.update_room(width_m=8.0)
        assert ctl.alert_seq == 2

    def test_success_clears_alert(self, ctl: DesignController) -> None:
        """Verify a later valid resize removes the alert."""
        drag(ctl, Tool.OBSTACLE, (10.0, 4.0), (12.0, 5.0))
        ctl.update_room(width_m=8.0)
        assert ctl.update_room(width_m=20.0)
        assert ctl.alert == ""
        obs = next(iter(ctl.model.layout.obstacles.values()))
        assert obs.rect == (10.0, 4.0, 2.0, 1.0)


class TestLoadLayout:
    """Replacing the design with a loaded layout."""

    def test_load_replaces_design_and_resets_state(
        self, ctl: DesignController
    ) -> None:
        """Verify load swaps the model and clears selection/drag/alert."""
        other = DesignController()
        drag(other, Tool.OBSTACLE, (2.0, 2.0), (3.0, 3.0))
        drag(ctl, Tool.WALL, (5.0, 4.0), (9.0, 4.0))
        ctl.alert = "old"
        ctl.load_layout(other.model.layout)
        assert len(ctl.model.layout.obstacles) == 1
        assert not ctl.model.layout.walls
        assert ctl.selection is None and ctl.alert == ""
        assert ctl.status.startswith("Loaded")

    def test_show_alert_bumps_sequence(self, ctl: DesignController) -> None:
        """Verify show_alert re-triggers on repeats."""
        ctl.show_alert("x")
        ctl.show_alert("x")
        assert ctl.alert == "x" and ctl.alert_seq == 2
