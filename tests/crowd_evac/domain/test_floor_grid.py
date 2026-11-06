"""Tests for crowd_evac.domain.floor_grid — door carving and goal cells.

Covers (issue #17):
  - Interior walls touching a door edge stay blocked (room_model_3 wall_5)
  - Host wall carved across its full thickness along the door span
  - Carved width never exceeds the authored width (inward quantisation)
  - Door span narrower than one cell → ScenarioValidationError
  - Obstacles inside a door opening stay blocked
  - exit_cells: outermost cells of the opening, never in a room interior
"""
from __future__ import annotations

import math

import pytest

from crowd_evac.domain.errors import ScenarioValidationError
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan, Obstacle, Wall

CS = 0.05


def _row(y: float) -> int:
    """Row index of the cell containing world y at the 5 cm grid."""
    return math.floor(y / CS + 1e-6)


def _col(x: float) -> int:
    """Column index of the cell containing world x at the 5 cm grid."""
    return math.floor(x / CS + 1e-6)


def _west_door_room(width_m: float, obstacles: tuple[Obstacle, ...] = ()) -> FloorPlan:
    """6 x 6 m room, 1 m west wall, door of *width_m* centred at y = 3."""
    return FloorPlan(
        6.0, 6.0,
        (Wall(0.0, 0.0, 1.0, 6.0, "west"),),
        obstacles,
        (Exit(1.0, 3.0, width_m, ExitSide.WEST, 1.0, "door"),),
    )


class TestWalkableMaskDoors:
    """Door openings carve only their host wall."""

    def test_wall_5_cells_stay_blocked(self, room_model_3: FloorPlan) -> None:
        """wall_5 (y 7.3-7.4, x 1.0-4.7) is blocked over its whole length."""
        mask = room_model_3.walkable_mask(CS)
        rows = slice(_row(7.3), _row(7.4))
        cols = slice(_col(1.0), _col(4.7))
        assert not mask[rows, cols].any()

    def test_host_wall_carved_full_thickness(
        self, room_model_3: FloorPlan
    ) -> None:
        """wall_4 (x 0-1) is open for every row of the 7.4-9.0 span."""
        mask = room_model_3.walkable_mask(CS)
        assert mask[_row(7.4):_row(9.0), 0:_col(1.0)].all()

    def test_host_wall_closed_outside_span(
        self, room_model_3: FloorPlan
    ) -> None:
        """Rows just below and above the span stay blocked in wall_4."""
        mask = room_model_3.walkable_mask(CS)
        assert not mask[_row(7.4) - 1, 0:_col(1.0)].any()
        assert not mask[_row(9.0), 0:_col(1.0)].any()

    def test_lower_room_not_adjacent_to_door(
        self, room_model_3: FloorPlan
    ) -> None:
        """No walkable lower-room cell 4-neighbours a carved door cell."""
        mask = room_model_3.walkable_mask(CS)
        top_room_row = _row(7.3) - 1
        door_rows = range(_row(7.4), _row(9.0))
        assert top_room_row + 1 not in door_rows
        assert not mask[top_room_row + 1, 0:_col(4.7)].any()

    @pytest.mark.parametrize("width", [0.85, 0.87, 1.6], ids=str)
    def test_carved_width_never_exceeds_authored(self, width: float) -> None:
        """Inward rounding: open rows in the wall ≤ width / cell."""
        mask = _west_door_room(width).walkable_mask(CS)
        open_rows = int(mask[:, 0].sum())
        assert 0 < open_rows * CS <= width + 1e-9

    def test_door_narrower_than_cell_raises(self) -> None:
        """A 4 cm door covers no whole 5 cm cell → unusable door."""
        with pytest.raises(ScenarioValidationError, match="no whole"):
            _west_door_room(0.04).walkable_mask(CS)

    def test_obstacle_in_opening_stays_blocked(self) -> None:
        """An obstacle inside the carved gap is re-blocked."""
        pillar = Obstacle(0.4, 2.9, 0.2, 0.2, "pillar")
        mask = _west_door_room(1.6, (pillar,)).walkable_mask(CS)
        assert not mask[_row(2.95), _col(0.45)]
        assert mask[_row(3.5), _col(0.45)]


class TestExitCells:
    """Goal cells sit at the outer edge of each opening."""

    def test_goals_on_outer_boundary_columns(
        self, room_model_3: FloorPlan
    ) -> None:
        """West goals in column 0, east goals in the last column."""
        mask = room_model_3.walkable_mask(CS)
        cols = {c for _, c in room_model_3.exit_cells(CS)}
        assert cols == {0, mask.shape[1] - 1}

    def test_goals_within_door_span_and_walkable(
        self, room_model_3: FloorPlan
    ) -> None:
        """Every goal row lies inside y 7.4-9.0 and is walkable."""
        mask = room_model_3.walkable_mask(CS)
        cells = room_model_3.exit_cells(CS)
        assert len(cells) == 2 * (_row(9.0) - _row(7.4))
        for r, c in cells:
            assert _row(7.4) <= r < _row(9.0)
            assert mask[r, c]

    def test_boundary_door_without_wall_goals_at_edge(self) -> None:
        """A door on the room edge with no wall uses the edge cells."""
        fp = FloorPlan(
            4.0, 4.0, (), (),
            (Exit(2.0, 0.0, 1.0, ExitSide.SOUTH, 1.0, "s"),),
        )
        cells = fp.exit_cells(CS)
        assert {r for r, _ in cells} == {0}
        assert len(cells) == round(1.0 / CS)

    def test_exit_cells_raise_for_sub_cell_door(self) -> None:
        """exit_cells validates the span even without a host wall."""
        fp = FloorPlan(
            4.0, 4.0, (), (),
            (Exit(2.02, 0.0, 0.04, ExitSide.SOUTH, 1.0, "tiny"),),
        )
        with pytest.raises(ScenarioValidationError, match="tiny"):
            fp.exit_cells(CS)
