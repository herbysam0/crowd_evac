"""Tests for designer room settings, moves and resize rules.
"""
from __future__ import annotations

import pytest

from crowd_evac.designer.model import DesignModel
from crowd_evac.domain.errors import DesignValidationError

from .helpers import ROOM_H, ROOM_W, snapshot


class TestRoom:
    """Room-level settings."""

    def test_update_name_and_agents(self, model: DesignModel) -> None:
        """Verify the room name can change."""
        model.update_room(name="hall")
        assert model.layout.room.name == "hall"

    def test_grow_room_allowed(self, model: DesignModel) -> None:
        """Verify enlarging the room with interior entities is accepted."""
        model.add_obstacle(5, 4, 2, 1)
        model.update_room(width_m=20.0)
        assert model.layout.room.width_m == 20.0

    def test_shrink_room_with_entity_outside_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify shrinking past an entity is rejected, state intact."""
        model.add_obstacle(10, 4, 2, 1)
        before = snapshot(model)
        with pytest.raises(DesignValidationError):
            model.update_room(width_m=8.0)
        assert model.layout is before

    @pytest.mark.parametrize(
        "changes",
        [{"width_m": 0}, {"height_m": -1}],
        ids=["zero-width", "neg-height"],
    )
    def test_invalid_room_values_rejected(
        self, model: DesignModel, changes: dict[str, float]
    ) -> None:
        """Verify non-positive room values are rejected."""
        with pytest.raises(DesignValidationError):
            model.update_room(**changes)


class TestMoves:
    """Translate walls, doorways and obstacles."""

    def test_move_obstacle(self, model: DesignModel) -> None:
        """Verify an obstacle is translated, size unchanged."""
        obs = model.add_obstacle(5, 4, 2, 1)
        model.move_obstacle(obs.id, 1.5, -1)
        assert model.layout.obstacles[obs.id].rect == (6.5, 3, 2, 1)

    def test_move_obstacle_out_of_room_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify moves are validated and leave state intact."""
        obs = model.add_obstacle(5, 4, 2, 1)
        before = snapshot(model)
        with pytest.raises(DesignValidationError):
            model.move_obstacle(obs.id, 20, 0)
        assert model.layout is before

    def test_move_interior_wall(self, model: DesignModel) -> None:
        """Verify an interior wall moves in both axes."""
        wall = model.add_wall(5, 4, 9, 4)
        moved = model.move_wall(wall.id, 1, 2)
        assert (moved.x0, moved.y0, moved.x1, moved.y1) == (6, 6, 10, 6)

    def test_move_wall_carries_doorways_along_axis(
        self, model: DesignModel
    ) -> None:
        """Verify a perimeter wall slid along its axis keeps its doorway."""
        wall = model.add_wall(0, 0, 10, 0)
        door = model.add_doorway(wall.id, 5, 2)
        model.move_wall(wall.id, 2, 0)
        assert model.layout.doorways[door.id].center == 7

    def test_move_wall_with_doorway_off_boundary_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify a doorway-bearing wall cannot leave the boundary."""
        wall = model.add_wall(0, 0, 10, 0)
        model.add_doorway(wall.id, 5, 2)
        before = snapshot(model)
        with pytest.raises(DesignValidationError):
            model.move_wall(wall.id, 0, 3)
        assert model.layout is before

    def test_move_doorway_along_wall(self, model: DesignModel) -> None:
        """Verify a doorway slides along its wall."""
        wall = model.add_wall(0, 0, ROOM_W, 0)
        door = model.add_doorway(wall.id, 5, 2)
        model.move_doorway(door.id, 3)
        assert model.layout.doorways[door.id].center == 8

    def test_move_doorway_beyond_room_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify sliding a doorway out of the room is rejected."""
        wall = model.add_wall(0, 0, ROOM_W, 0)
        door = model.add_doorway(wall.id, 5, 2)
        with pytest.raises(DesignValidationError):
            model.move_doorway(door.id, 20)


class TestRoomResizeKeepsEntities:
    """Resizing never moves entities and names what blocks a resize."""

    def test_accepted_resize_leaves_positions_unchanged(
        self, model: DesignModel
    ) -> None:
        """Verify walls and obstacles keep their coordinates on growth."""
        wall = model.add_wall(5, 4, 9, 4)
        obs = model.add_obstacle(10, 6, 2, 1)
        model.update_room(width_m=30.0, height_m=20.0)
        assert model.layout.walls[wall.id] == wall
        assert model.layout.obstacles[obs.id] == obs

    def test_shrink_names_the_blocking_obstacle(
        self, model: DesignModel
    ) -> None:
        """Verify the error identifies the entity outside the new size."""
        obs = model.add_obstacle(10, 4, 2, 1)
        with pytest.raises(
            DesignValidationError, match=f"Obstacle {obs.id} is outside"
        ):
            model.update_room(width_m=8.0)

    def test_shrink_names_the_blocking_wall(self, model: DesignModel) -> None:
        """Verify an edge wall blocks shrinking and is named."""
        wall = model.add_wall(ROOM_W, 0, ROOM_W, ROOM_H)
        with pytest.raises(
            DesignValidationError, match=f"Wall {wall.id} is outside"
        ):
            model.update_room(width_m=12.0)

    def test_growth_rejected_when_doorway_wall_leaves_edge(
        self, model: DesignModel
    ) -> None:
        """Verify growing is rejected if a doorway's wall stops being an edge."""
        wall = model.add_wall(ROOM_W, 0, ROOM_W, ROOM_H)
        door = model.add_doorway(wall.id, 5, 2)
        before = snapshot(model)
        with pytest.raises(
            DesignValidationError, match=f"Doorway {door.id} is only allowed"
        ):
            model.update_room(width_m=20.0)
        assert model.layout is before
