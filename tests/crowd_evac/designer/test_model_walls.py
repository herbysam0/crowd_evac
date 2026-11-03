"""Tests for designer walls: rules, sides, thickness.
"""
from __future__ import annotations

import pytest

from crowd_evac.designer.model import DesignModel
from crowd_evac.domain.errors import DesignValidationError, ScenarioError

from .helpers import ROOM_H, ROOM_W, snapshot


class TestErrorType:
    """DesignValidationError placement in the hierarchy."""

    def test_is_scenario_error(self) -> None:
        """Verify the designer error is a ScenarioError."""
        assert issubclass(DesignValidationError, ScenarioError)


class TestWalls:
    """Wall add / edit / delete rules."""

    def test_horizontal_wall_on_bottom_boundary_is_flush(
        self, model: DesignModel
    ) -> None:
        """Verify a wall drawn on y=0 is shifted inward to rect y=0."""
        wall = model.add_wall(0, 0, ROOM_W, 0)
        assert model.wall_rect(wall.id) == (0.0, 0.0, ROOM_W, 0.5)

    def test_vertical_wall_on_right_boundary_is_flush(
        self, model: DesignModel
    ) -> None:
        """Verify a wall drawn on x=width ends flush inside the room."""
        wall = model.add_wall(ROOM_W, 0, ROOM_W, ROOM_H)
        assert model.wall_rect(wall.id) == (ROOM_W - 0.5, 0.0, 0.5, ROOM_H)

    def test_interior_wall_is_centred_on_line(
        self, model: DesignModel
    ) -> None:
        """Verify an interior wall rect is centred on its centreline."""
        wall = model.add_wall(5, 4, 9, 4)
        assert model.wall_rect(wall.id) == (5.0, 3.75, 4.0, 0.5)

    def test_wall_drawn_right_to_left_is_normalised_in_rect(
        self, model: DesignModel
    ) -> None:
        """Verify reversed endpoints still produce a positive-width rect."""
        wall = model.add_wall(9, 4, 5, 4)
        assert model.wall_rect(wall.id)[:3] == (5.0, 3.75, 4.0)

    def test_overlapping_walls_are_allowed(self, model: DesignModel) -> None:
        """Verify wall-wall overlap (corners) is accepted."""
        model.add_wall(0, 0, ROOM_W, 0)
        model.add_wall(0, 0, 0, ROOM_H)
        assert len(model.layout.walls) == 2

    @pytest.mark.parametrize(
        "coords",
        [
            (3, 3, 3, 3),
            (0, 0, 5, 5),
            (-1, 5, 4, 5),
            (5, 5, 16, 5),
            (5, -1, 5, 4),
            (5, 5, 5, 10.5),
        ],
        ids=["zero", "diagonal", "left", "right", "below", "above"],
    )
    def test_invalid_wall_rejected_and_state_intact(
        self, model: DesignModel, coords: tuple[float, ...]
    ) -> None:
        """Verify degenerate/diagonal/outside walls are rejected."""
        before = snapshot(model)
        with pytest.raises(DesignValidationError):
            model.add_wall(*coords)
        assert model.layout is before

    def test_wall_overlapping_obstacle_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify a wall may not cross an obstacle."""
        model.add_obstacle(5, 3, 2, 2)
        with pytest.raises(DesignValidationError, match="obstacle"):
            model.add_wall(4, 4, 8, 4)

    def test_wall_touching_obstacle_edge_allowed(
        self, model: DesignModel
    ) -> None:
        """Verify touching edges are not overlap."""
        model.add_obstacle(5, 4.25, 2, 2)
        model.add_wall(4, 4, 8, 4)
        assert len(model.layout.walls) == 1

    def test_set_wall_length_keeps_start_point(
        self, model: DesignModel
    ) -> None:
        """Verify the end moves while the start stays fixed."""
        wall = model.add_wall(2, 4, 6, 4)
        updated = model.set_wall_length(wall.id, 7.5)
        assert (updated.x0, updated.y0) == (2, 4)
        assert (updated.x1, updated.y1) == (9.5, 4)

    def test_set_wall_length_preserves_direction(
        self, model: DesignModel
    ) -> None:
        """Verify a wall drawn downward keeps extending downward."""
        wall = model.add_wall(5, 8, 5, 4)
        updated = model.set_wall_length(wall.id, 6)
        assert (updated.x1, updated.y1) == (5, 2)

    @pytest.mark.parametrize("length", [0, -1])
    def test_set_wall_length_non_positive_rejected(
        self, model: DesignModel, length: float
    ) -> None:
        """Verify non-positive lengths raise."""
        wall = model.add_wall(2, 4, 6, 4)
        with pytest.raises(DesignValidationError):
            model.set_wall_length(wall.id, length)

    def test_set_wall_length_beyond_room_rejected_state_intact(
        self, model: DesignModel
    ) -> None:
        """Verify extending past the room is rejected without change."""
        wall = model.add_wall(2, 4, 6, 4)
        before = snapshot(model)
        with pytest.raises(DesignValidationError):
            model.set_wall_length(wall.id, 20)
        assert model.layout is before

    def test_delete_wall_cascades_doorways(self, model: DesignModel) -> None:
        """Verify deleting a wall removes its doorways and reports count."""
        wall = model.add_wall(0, 0, ROOM_W, 0)
        model.add_doorway(wall.id, 3, 2)
        model.add_doorway(wall.id, 9, 2)
        assert model.delete_wall(wall.id) == 2
        assert not model.layout.walls and not model.layout.doorways

    def test_delete_unknown_wall_raises(self, model: DesignModel) -> None:
        """Verify deleting an unknown id raises KeyError."""
        with pytest.raises(KeyError):
            model.delete_wall(99)

    def test_edit_wall_moves_it(self, model: DesignModel) -> None:
        """Verify edit_wall replaces the endpoints."""
        wall = model.add_wall(2, 4, 6, 4)
        model.edit_wall(wall.id, 2, 5, 6, 5)
        assert model.wall_rect(wall.id)[1] == 4.75


class TestSideMapping:
    """north=y0 bottom, south=y=height top, per renderer convention."""

    @pytest.mark.parametrize(
        "coords, side",
        [
            ((0, 0, ROOM_W, 0), "north"),
            ((0, ROOM_H, ROOM_W, ROOM_H), "south"),
            ((0, 0, 0, ROOM_H), "west"),
            ((ROOM_W, 0, ROOM_W, ROOM_H), "east"),
            ((5, 4, 9, 4), None),
        ],
        ids=["north-bottom", "south-top", "west", "east", "interior"],
    )
    def test_side_of(
        self,
        model: DesignModel,
        coords: tuple[float, float, float, float],
        side: str | None,
    ) -> None:
        """Verify perimeter side derivation for every wall position."""
        wall = model.add_wall(*coords)
        assert model.side_of(wall.id) == side


class TestWallThickness:
    """Per-wall thickness."""

    def test_default_thickness_is_half_metre(self, model: DesignModel) -> None:
        """Verify walls keep the historical 0.5 m default."""
        wall = model.add_wall(5, 4, 9, 4)
        assert wall.thickness == 0.5

    def test_custom_thickness_in_rect_horizontal(
        self, model: DesignModel
    ) -> None:
        """Verify a thick interior wall is centred on its line."""
        wall = model.add_wall(5, 4, 9, 4, thickness=1.0)
        assert model.wall_rect(wall.id) == (5.0, 3.5, 4.0, 1.0)

    def test_custom_thickness_in_rect_vertical(
        self, model: DesignModel
    ) -> None:
        """Verify thickness applies across a vertical wall."""
        wall = model.add_wall(6, 2, 6, 8, thickness=0.2)
        assert model.wall_rect(wall.id) == pytest.approx((5.9, 2.0, 0.2, 6.0))

    def test_perimeter_wall_stays_flush_when_thicker(
        self, model: DesignModel
    ) -> None:
        """Verify a boundary wall grows inward, staying flush."""
        wall = model.add_wall(0, ROOM_H, ROOM_W, ROOM_H, thickness=1.0)
        assert model.wall_rect(wall.id) == (0.0, ROOM_H - 1.0, ROOM_W, 1.0)
        assert model.side_of(wall.id) == "south"

    def test_set_thickness(self, model: DesignModel) -> None:
        """Verify thickness can be edited and endpoints are kept."""
        wall = model.add_wall(5, 4, 9, 4)
        updated = model.set_wall_thickness(wall.id, 0.8)
        assert updated.thickness == 0.8
        assert (updated.x0, updated.x1) == (5, 9)

    @pytest.mark.parametrize("value", [0, -0.5])
    def test_non_positive_thickness_rejected(
        self, model: DesignModel, value: float
    ) -> None:
        """Verify zero/negative thickness is rejected, state intact."""
        wall = model.add_wall(5, 4, 9, 4)
        before = snapshot(model)
        with pytest.raises(DesignValidationError, match="thickness"):
            model.set_wall_thickness(wall.id, value)
        assert model.layout is before

    def test_thickness_into_obstacle_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify thickening a wall into an obstacle is rejected."""
        wall = model.add_wall(5, 4, 9, 4)
        model.add_obstacle(6, 4.5, 1, 1)
        with pytest.raises(DesignValidationError):
            model.set_wall_thickness(wall.id, 2.0)

    def test_thickness_survives_edit_and_move(
        self, model: DesignModel
    ) -> None:
        """Verify length edits and moves keep the thickness."""
        wall = model.add_wall(5, 4, 9, 4, thickness=0.8)
        model.set_wall_length(wall.id, 3)
        model.move_wall(wall.id, 1, 1)
        assert model.layout.walls[wall.id].thickness == 0.8

    def test_doorway_footprint_follows_thickness(
        self, model: DesignModel
    ) -> None:
        """Verify an opening spans the whole (thicker) wall."""
        wall = model.add_wall(0, 0, ROOM_W, 0, thickness=0.8)
        door = model.add_doorway(wall.id, 5, 2)
        assert model.doorway_rect(door.id)[3] == 0.8
