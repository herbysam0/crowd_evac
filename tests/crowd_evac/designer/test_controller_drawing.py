"""Tests for the designer controller: snapping and drawing tools.
"""
from __future__ import annotations

import pytest

from crowd_evac.designer.controller import DEFAULT_DOOR_WIDTH_M, DesignController, Tool
from crowd_evac.designer.placement import Selection, snap

from .helpers import ROOM_W, bottom_wall_with_door, drag


class TestSnap:
    """Grid snapping."""

    @pytest.mark.parametrize(
        "value, expected",
        [(0.0, 0.0), (0.04, 0.0), (0.16, 0.2), (3.62, 3.6), (0.3, 0.3)],
    )
    def test_snap_to_tenth_metre(self, value: float, expected: float) -> None:
        """Verify values snap to 0.1 m with no float residue."""
        assert snap(value) == expected


class TestWallTool:
    """Wall drawing."""

    def test_rubber_band_follows_dominant_axis_with_length(
        self, ctl: DesignController
    ) -> None:
        """Verify live band is horizontal for a mostly-horizontal drag."""
        ctl.set_tool(Tool.WALL)
        ctl.press(2.0, 3.0)
        ctl.drag(6.04, 3.4)
        band = ctl.rubber_band
        assert band is not None
        assert (band.x0, band.y0, band.x1, band.y1) == (2.0, 3.0, 6.0, 3.0)
        assert band.length_m == 4.0

    def test_vertical_drag_gives_vertical_wall(
        self, ctl: DesignController
    ) -> None:
        """Verify a mostly-vertical drag produces a vertical wall."""
        drag(ctl, Tool.WALL, (4.0, 2.0), (4.4, 7.0))
        wall = next(iter(ctl.model.layout.walls.values()))
        assert (wall.x0, wall.y0, wall.x1, wall.y1) == (4.0, 2.0, 4.0, 7.0)

    def test_release_commits_selects_and_clears_band(
        self, ctl: DesignController
    ) -> None:
        """Verify release adds the wall, selects it, clears the band."""
        drag(ctl, Tool.WALL, (2.0, 3.0), (6.0, 3.0))
        assert ctl.rubber_band is None
        assert ctl.selection == Selection("wall", 1)
        assert len(ctl.model.layout.walls) == 1

    def test_drag_beyond_room_is_clamped(self, ctl: DesignController) -> None:
        """Verify the pointer is clamped to the room."""
        drag(ctl, Tool.WALL, (2.0, 3.0), (40.0, 3.0))
        wall = next(iter(ctl.model.layout.walls.values()))
        assert wall.x1 == ROOM_W

    def test_zero_length_drag_rejected(self, ctl: DesignController) -> None:
        """Verify a click without drag adds no wall and reports why."""
        drag(ctl, Tool.WALL, (2.0, 3.0), (2.04, 3.03))
        assert not ctl.model.layout.walls
        assert "length" in ctl.status

    def test_drag_without_press_is_ignored(
        self, ctl: DesignController
    ) -> None:
        """Verify stray drag/release events do nothing."""
        ctl.set_tool(Tool.WALL)
        ctl.drag(5.0, 5.0)
        ctl.release(5.0, 5.0)
        assert ctl.rubber_band is None and not ctl.model.layout.walls

    def test_switching_tool_cancels_drag(self, ctl: DesignController) -> None:
        """Verify a tool change discards the rubber band."""
        ctl.set_tool(Tool.WALL)
        ctl.press(2.0, 3.0)
        ctl.set_tool(Tool.SELECT)
        assert ctl.rubber_band is None


class TestDoorwayTool:
    """Doorway placement."""

    def test_click_mid_wall_snaps_and_selects(
        self, ctl: DesignController
    ) -> None:
        """Verify a mid-wall click places a centred, selected doorway."""
        bottom_wall_with_door(ctl)
        door = next(iter(ctl.model.layout.doorways.values()))
        assert door.center == 5.0 and door.width_m == DEFAULT_DOOR_WIDTH_M
        assert ctl.selection == Selection("doorway", door.id)

    def test_click_uses_settable_width(self, ctl: DesignController) -> None:
        """Verify the configured width is used."""
        ctl.door_width_m = 3.0
        bottom_wall_with_door(ctl)
        assert next(iter(ctl.model.layout.doorways.values())).width_m == 3.0

    def test_click_near_wall_end_places_outward(
        self, ctl: DesignController
    ) -> None:
        """Verify an end click extends the opening past the wall end."""
        drag(ctl, Tool.WALL, (0.0, 0.0), (0.0, 6.0))  # west, partial
        ctl.door_width_m = 2.0
        ctl.set_tool(Tool.DOORWAY)
        ctl.press(0.25, 5.9)
        door = next(iter(ctl.model.layout.doorways.values()))
        assert door.center == 7.0  # opening y 6..8

    def test_click_near_corner_end_falls_back_inward(
        self, ctl: DesignController
    ) -> None:
        """Verify an end that cannot extend outward is placed flush inside."""
        drag(ctl, Tool.WALL, (0.0, 0.0), (ROOM_W, 0.0))
        ctl.door_width_m = 2.0
        ctl.set_tool(Tool.DOORWAY)
        ctl.press(0.2, 0.25)
        assert next(iter(ctl.model.layout.doorways.values())).center == 1.0

    def test_click_on_interior_wall_rejected(
        self, ctl: DesignController
    ) -> None:
        """Verify interior walls take no doorways."""
        drag(ctl, Tool.WALL, (5.0, 4.0), (9.0, 4.0))
        ctl.set_tool(Tool.DOORWAY)
        ctl.press(7.0, 4.0)
        assert not ctl.model.layout.doorways
        assert "boundary" in ctl.status

    def test_click_on_empty_space_rejected(self, ctl: DesignController) -> None:
        """Verify clicking nothing reports guidance."""
        ctl.set_tool(Tool.DOORWAY)
        ctl.press(5.0, 5.0)
        assert not ctl.model.layout.doorways and ctl.status


class TestObstacleTool:
    """Obstacle drawing."""

    @pytest.mark.parametrize(
        "a, b",
        [((5.0, 4.0), (7.0, 5.0)), ((7.0, 5.0), (5.0, 4.0))],
        ids=["forward", "reverse"],
    )
    def test_drag_creates_rect_any_direction(
        self,
        ctl: DesignController,
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> None:
        """Verify corners are normalised."""
        drag(ctl, Tool.OBSTACLE, a, b)
        obs = next(iter(ctl.model.layout.obstacles.values()))
        assert obs.rect == (5.0, 4.0, 2.0, 1.0)

    def test_zero_size_rejected(self, ctl: DesignController) -> None:
        """Verify a degenerate obstacle drag is rejected."""
        drag(ctl, Tool.OBSTACLE, (5.0, 4.0), (5.0, 4.0))
        assert not ctl.model.layout.obstacles and ctl.status

    def test_overlap_with_wall_rejected(self, ctl: DesignController) -> None:
        """Verify an obstacle drawn across a wall is rejected."""
        drag(ctl, Tool.WALL, (0.0, 0.0), (ROOM_W, 0.0))
        drag(ctl, Tool.OBSTACLE, (3.0, 0.0), (5.0, 2.0))
        assert not ctl.model.layout.obstacles
        assert "overlap" in ctl.status
