"""Tests for designer doorways and obstacles.
"""
from __future__ import annotations

import pytest

from crowd_evac.designer.model import DesignModel
from crowd_evac.domain.errors import DesignValidationError

from .helpers import ROOM_W, snapshot


class TestDoorways:
    """Doorway rules."""

    @pytest.fixture
    def bottom(self, model: DesignModel) -> int:
        """Add a bottom perimeter wall and return its id."""
        return model.add_wall(0, 0, ROOM_W, 0).id

    def test_doorway_in_perimeter_wall(
        self, model: DesignModel, bottom: int
    ) -> None:
        """Verify a doorway mid-wall is accepted with footprint."""
        door = model.add_doorway(bottom, 5, 2)
        assert model.doorway_rect(door.id) == (4.0, 0.0, 2.0, 0.5)

    def test_doorway_default_capacity_positive(
        self, model: DesignModel, bottom: int
    ) -> None:
        """Verify the default capacity is a positive int."""
        assert model.add_doorway(bottom, 5, 2).capacity_per_second > 0

    def test_doorway_at_wall_end_extends_outward(
        self, model: DesignModel
    ) -> None:
        """Verify an opening starting at a wall end is accepted."""
        wall = model.add_wall(0, 0, 0, 6)  # west wall, partial
        door = model.add_doorway(wall.id, 6 + 1.0, 2)  # y 6..8
        assert model.doorway_rect(door.id)[1:4:2] == (6.0, 2.0)

    def test_doorway_beyond_room_rejected(
        self, model: DesignModel, bottom: int
    ) -> None:
        """Verify an opening leaving the room is rejected."""
        with pytest.raises(DesignValidationError, match="fit"):
            model.add_doorway(bottom, ROOM_W, 2)

    def test_doorway_detached_from_wall_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify an opening disjoint from the wall span is rejected."""
        wall = model.add_wall(0, 0, 4, 0)
        with pytest.raises(DesignValidationError, match="fit"):
            model.add_doorway(wall.id, 10, 2)

    def test_doorway_on_interior_wall_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify interior walls take no doorways."""
        wall = model.add_wall(5, 4, 9, 4)
        with pytest.raises(DesignValidationError, match="boundary"):
            model.add_doorway(wall.id, 7, 2)

    @pytest.mark.parametrize("width, cap", [(0, 20), (-1, 20), (2, 0), (2, -5)])
    def test_doorway_non_positive_values_rejected(
        self, model: DesignModel, bottom: int, width: float, cap: int
    ) -> None:
        """Verify width and capacity must be positive."""
        with pytest.raises(DesignValidationError):
            model.add_doorway(bottom, 5, width, cap)

    def test_doorway_overlapping_doorway_rejected(
        self, model: DesignModel, bottom: int
    ) -> None:
        """Verify two openings may not overlap."""
        model.add_doorway(bottom, 5, 2)
        with pytest.raises(DesignValidationError, match="Doorways overlap"):
            model.add_doorway(bottom, 5.5, 2)

    def test_adjacent_doorways_allowed(
        self, model: DesignModel, bottom: int
    ) -> None:
        """Verify touching openings are accepted."""
        model.add_doorway(bottom, 5, 2)
        model.add_doorway(bottom, 7, 2)
        assert len(model.layout.doorways) == 2

    def test_doorway_overlapping_obstacle_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify an end-placed opening may not cover an obstacle."""
        wall = model.add_wall(0, 0, 4, 0)
        model.add_obstacle(4.5, 0.1, 1, 0.3)
        with pytest.raises(DesignValidationError, match="Doorway and obstacle"):
            model.add_doorway(wall.id, 5, 2)

    def test_wall_edit_dropping_doorway_rejected(
        self, model: DesignModel, bottom: int
    ) -> None:
        """Verify shortening a wall away from its doorway is rejected."""
        model.add_doorway(bottom, 12, 2)
        before = snapshot(model)
        with pytest.raises(DesignValidationError, match="fit"):
            model.set_wall_length(bottom, 5)
        assert model.layout is before

    def test_wall_edit_changing_side_with_doorway_rejected(
        self, model: DesignModel, bottom: int
    ) -> None:
        """Verify moving a wall off its boundary is rejected."""
        model.add_doorway(bottom, 5, 2)
        with pytest.raises(DesignValidationError):
            model.edit_wall(bottom, 0, 4, ROOM_W, 4)

    def test_edit_doorway(self, model: DesignModel, bottom: int) -> None:
        """Verify centre, width and capacity can be edited."""
        door = model.add_doorway(bottom, 5, 2)
        new = model.edit_doorway(door.id, 8, 3, 10)
        assert (new.center, new.width_m, new.capacity_per_second) == (8, 3, 10)

    def test_edit_doorway_invalid_leaves_state(
        self, model: DesignModel, bottom: int
    ) -> None:
        """Verify an invalid doorway edit is rejected."""
        door = model.add_doorway(bottom, 5, 2)
        before = snapshot(model)
        with pytest.raises(DesignValidationError):
            model.edit_doorway(door.id, 5, -1, 10)
        assert model.layout is before

    def test_delete_doorway(self, model: DesignModel, bottom: int) -> None:
        """Verify a doorway can be deleted without touching its wall."""
        door = model.add_doorway(bottom, 5, 2)
        model.delete_doorway(door.id)
        assert not model.layout.doorways and bottom in model.layout.walls

    def test_delete_unknown_doorway_raises(self, model: DesignModel) -> None:
        """Verify deleting an unknown id raises KeyError."""
        with pytest.raises(KeyError):
            model.delete_doorway(99)


class TestObstacles:
    """Obstacle rules."""

    def test_add_obstacle(self, model: DesignModel) -> None:
        """Verify a valid obstacle is stored."""
        obs = model.add_obstacle(5, 4, 2, 1)
        assert model.layout.obstacles[obs.id].rect == (5, 4, 2, 1)

    def test_obstacle_overlap_obstacle_allowed(
        self, model: DesignModel
    ) -> None:
        """Verify obstacles may overlap each other."""
        model.add_obstacle(5, 4, 2, 2)
        model.add_obstacle(6, 5, 2, 2)
        assert len(model.layout.obstacles) == 2

    def test_obstacle_overlapping_wall_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify obstacles may not overlap walls."""
        model.add_wall(0, 0, ROOM_W, 0)
        with pytest.raises(DesignValidationError, match="Wall and obstacle"):
            model.add_obstacle(3, 0.25, 1, 1)

    def test_obstacle_touching_wall_allowed(self, model: DesignModel) -> None:
        """Verify an obstacle touching a wall edge is accepted."""
        model.add_wall(0, 0, ROOM_W, 0)
        model.add_obstacle(3, 0.5, 1, 1)
        assert len(model.layout.obstacles) == 1

    @pytest.mark.parametrize(
        "rect",
        [(5, 4, 0, 1), (5, 4, 1, -1), (-1, 4, 2, 2), (14, 4, 2, 2), (5, 9.5, 1, 1)],
        ids=["zero-w", "neg-h", "left", "right", "top"],
    )
    def test_invalid_obstacle_rejected(
        self, model: DesignModel, rect: tuple[float, ...]
    ) -> None:
        """Verify non-positive or out-of-room obstacles are rejected."""
        with pytest.raises(DesignValidationError):
            model.add_obstacle(*rect)

    def test_edit_obstacle(self, model: DesignModel) -> None:
        """Verify numeric resize of an obstacle."""
        obs = model.add_obstacle(5, 4, 2, 1)
        model.edit_obstacle(obs.id, 5, 4, 3, 2)
        assert model.layout.obstacles[obs.id].rect == (5, 4, 3, 2)

    def test_edit_obstacle_into_wall_rejected(
        self, model: DesignModel
    ) -> None:
        """Verify editing into a wall is rejected and state is intact."""
        model.add_wall(0, 0, ROOM_W, 0)
        obs = model.add_obstacle(5, 4, 2, 1)
        before = snapshot(model)
        with pytest.raises(DesignValidationError):
            model.edit_obstacle(obs.id, 5, 0, 2, 1)
        assert model.layout is before

    def test_delete_obstacle(self, model: DesignModel) -> None:
        """Verify an obstacle can be deleted."""
        obs = model.add_obstacle(5, 4, 2, 1)
        model.delete_obstacle(obs.id)
        assert not model.layout.obstacles

    def test_unknown_obstacle_raises(self, model: DesignModel) -> None:
        """Verify edit/delete of unknown ids raise KeyError."""
        with pytest.raises(KeyError):
            model.edit_obstacle(99, 1, 1, 1, 1)
        with pytest.raises(KeyError):
            model.delete_obstacle(99)
