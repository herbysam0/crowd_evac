"""Tests for crowd_evac.domain.collision.CollisionMap (step 1.19a item 1).

Covers:
  - from_floor_plan(): blocks wall/obstacle cells, leaves walkable cells free,
    invalid cell_size raises.
  - __init__(): validates cell_size and grid dimensionality.
  - blocked property: returns a read-only view.
  - is_blocked(): inside/outside classification, out-of-bounds, shape failure.
  - resolve(): axis-separated sliding — straight block, diagonal slide, clear
    move, fully-blocked corner, velocity zeroing, dead-agent immutability,
    empty state, prev_pos shape failure, and the core invariant that no
    resolved position ever lands in a blocked cell.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan, Obstacle

from .conftest import MakeState


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------


def _wall_column_map() -> CollisionMap:
    """5x5 unit-cell grid whose column c == 2 is a vertical wall."""
    blocked = np.zeros((5, 5), dtype=np.bool_)
    blocked[:, 2] = True
    return CollisionMap(blocked, cell_size=1.0)


def _cross_map() -> CollisionMap:
    """5x5 unit-cell grid with both column 2 and row 2 blocked (a cross)."""
    blocked = np.zeros((5, 5), dtype=np.bool_)
    blocked[:, 2] = True
    blocked[2, :] = True
    return CollisionMap(blocked, cell_size=1.0)


# ---------------------------------------------------------------------------
# from_floor_plan
# ---------------------------------------------------------------------------


class TestFromFloorPlan:
    """from_floor_plan blocks static geometry and leaves openings free."""

    @pytest.fixture
    def floor_with_obstacle(self) -> FloorPlan:
        """4 m x 4 m room with a 2x2 m central obstacle and one exit."""
        return FloorPlan(
            width_m=4.0,
            height_m=4.0,
            walls=(),
            obstacles=(Obstacle(x=1.0, y=1.0, width=2.0, height=2.0),),
            exits=(
                Exit(
                    x=4.0,
                    y=2.0,
                    width_m=1.0,
                    side=ExitSide.EAST,
                    capacity_per_second=5,
                ),
            ),
        )

    def test_obstacle_cells_blocked(
        self, floor_with_obstacle: FloorPlan
    ) -> None:
        """Cells covered by the obstacle are blocked at unit resolution."""
        cmap = CollisionMap.from_floor_plan(floor_with_obstacle, cell_size=1.0)
        # Obstacle spans x,y in [1, 3) -> rows 1..2, cols 1..2.
        assert cmap.blocked[1, 1]
        assert cmap.blocked[2, 2]

    def test_open_cells_walkable(self, floor_with_obstacle: FloorPlan) -> None:
        """Cells away from the obstacle remain passable (not blocked)."""
        cmap = CollisionMap.from_floor_plan(floor_with_obstacle, cell_size=1.0)
        assert not cmap.blocked[0, 0]
        assert not cmap.blocked[3, 3]

    def test_open_room_blocks_nothing(self) -> None:
        """A room with no walls or obstacles blocks no interior cell (edge)."""
        floor = FloorPlan(
            width_m=4.0,
            height_m=4.0,
            walls=(),
            obstacles=(),
            exits=(
                Exit(
                    x=4.0,
                    y=2.0,
                    width_m=1.0,
                    side=ExitSide.EAST,
                    capacity_per_second=5,
                ),
            ),
        )
        cmap = CollisionMap.from_floor_plan(floor, cell_size=1.0)
        assert not np.any(cmap.blocked)

    def test_non_positive_cell_size_raises(
        self, floor_with_obstacle: FloorPlan
    ) -> None:
        """A non-positive cell size raises ValueError (failure path)."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            CollisionMap.from_floor_plan(floor_with_obstacle, cell_size=0.0)


# ---------------------------------------------------------------------------
# __init__ and blocked property
# ---------------------------------------------------------------------------


class TestConstruction:
    """__init__ validation and the read-only blocked view."""

    def test_stores_cell_size(self) -> None:
        """A valid grid and cell size are stored on the instance."""
        cmap = CollisionMap(np.zeros((3, 3), dtype=np.bool_), cell_size=0.5)
        assert cmap.cell_size == 0.5

    def test_non_positive_cell_size_raises(self) -> None:
        """cell_size <= 0 raises ValueError (failure path)."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            CollisionMap(np.zeros((3, 3), dtype=np.bool_), cell_size=-1.0)

    def test_non_2d_grid_raises(self) -> None:
        """A non-2-D blocking grid raises ValueError (failure path)."""
        with pytest.raises(ValueError, match="2-D grid"):
            CollisionMap(np.zeros(5, dtype=np.bool_), cell_size=1.0)

    def test_blocked_view_is_read_only(self) -> None:
        """The blocked property returns a non-writeable view (edge)."""
        cmap = _wall_column_map()
        view = cmap.blocked
        with pytest.raises(ValueError):
            view[0, 0] = True


# ---------------------------------------------------------------------------
# is_blocked
# ---------------------------------------------------------------------------


class TestIsBlocked:
    """is_blocked classifies world points against the grid."""

    def test_point_in_blocked_cell_is_true(self) -> None:
        """A point inside the wall column reports blocked (happy)."""
        cmap = _wall_column_map()
        pts = np.array([[2.5, 1.5]], dtype=np.float64)  # col 2 -> blocked
        assert bool(cmap.is_blocked(pts)[0])

    def test_point_in_open_cell_is_false(self) -> None:
        """A point in an open cell reports not blocked (happy)."""
        cmap = _wall_column_map()
        pts = np.array([[0.5, 0.5]], dtype=np.float64)  # col 0 -> open
        assert not bool(cmap.is_blocked(pts)[0])

    def test_out_of_bounds_is_blocked(self) -> None:
        """Points outside the grid bounding box report blocked (edge)."""
        cmap = _wall_column_map()
        pts = np.array(
            [[-1.0, 0.5], [0.5, 99.0]], dtype=np.float64
        )
        result = cmap.is_blocked(pts)
        assert bool(result[0]) and bool(result[1])

    def test_vectorised_mixed_points(self) -> None:
        """A batch returns one boolean per point in order."""
        cmap = _wall_column_map()
        pts = np.array(
            [[0.5, 0.5], [2.5, 0.5], [4.5, 0.5]], dtype=np.float64
        )
        np.testing.assert_array_equal(
            cmap.is_blocked(pts), np.array([False, True, False])
        )

    def test_wrong_shape_raises(self) -> None:
        """A non-(N, 2) array raises ValueError (failure path)."""
        cmap = _wall_column_map()
        with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
            cmap.is_blocked(np.zeros((3, 3), dtype=np.float64))


# ---------------------------------------------------------------------------
# cells_blocked — integer cell-index classification
# ---------------------------------------------------------------------------


class TestCellsBlocked:
    """cells_blocked classifies integer cell coordinates (OOB == blocked)."""

    def test_blocked_and_open_cells(self) -> None:
        """Blocked and open in-bounds cells classify correctly (happy)."""
        cmap = _wall_column_map()
        rows = np.array([0, 0], dtype=np.intp)
        cols = np.array([2, 0], dtype=np.intp)  # col 2 blocked, col 0 open
        np.testing.assert_array_equal(
            cmap.cells_blocked(rows, cols), np.array([True, False])
        )

    def test_out_of_bounds_is_blocked(self) -> None:
        """Cells outside the grid report blocked (edge)."""
        cmap = _wall_column_map()
        rows = np.array([-1, 99, 0], dtype=np.intp)
        cols = np.array([0, 0, -1], dtype=np.intp)
        assert np.all(cmap.cells_blocked(rows, cols))


# ---------------------------------------------------------------------------
# resolve — axis-separated sliding
# ---------------------------------------------------------------------------


class TestResolveBlocking:
    """resolve rejects moves into blocked cells and slides along walls."""

    def test_straight_move_into_wall_is_rejected(
        self, make_state: MakeState
    ) -> None:
        """A move straight into the wall column keeps the agent on its side."""
        cmap = _wall_column_map()
        prev = np.array([[1.5, 0.5]], dtype=np.float64)  # col 1, open
        state = make_state([[2.5, 0.5]], vel=[[1.0, 0.0]])  # col 2, blocked
        cmap.resolve(state, prev)
        # Agent must not be in the blocked column.
        np.testing.assert_array_equal(state.pos[0], prev[0])
        assert state.vel[0, 0] == 0.0  # x-velocity zeroed

    def test_diagonal_move_slides_along_wall(
        self, make_state: MakeState
    ) -> None:
        """A diagonal move blocked on x slides along the wall on y."""
        cmap = _wall_column_map()
        prev = np.array([[1.5, 0.5]], dtype=np.float64)
        state = make_state([[2.5, 1.5]], vel=[[1.0, 1.0]])
        cmap.resolve(state, prev)
        # x stays (blocked), y advances (free): slide.
        assert state.pos[0, 0] == pytest.approx(1.5)
        assert state.pos[0, 1] == pytest.approx(1.5)
        assert state.vel[0, 0] == 0.0  # blocked axis zeroed
        assert state.vel[0, 1] == pytest.approx(1.0)  # free axis preserved

    def test_clear_move_is_unchanged(self, make_state: MakeState) -> None:
        """A move between two open cells is accepted in full."""
        cmap = _wall_column_map()
        prev = np.array([[0.2, 0.5]], dtype=np.float64)
        state = make_state([[1.5, 0.5]], vel=[[1.0, 0.0]])
        cmap.resolve(state, prev)
        assert state.pos[0, 0] == pytest.approx(1.5)
        assert state.vel[0, 0] == pytest.approx(1.0)

    def test_fully_blocked_corner_stays_put(
        self, make_state: MakeState
    ) -> None:
        """When full, x-only, and y-only are all blocked, the agent halts."""
        cmap = _cross_map()  # column 2 and row 2 blocked
        prev = np.array([[1.5, 1.5]], dtype=np.float64)  # open
        state = make_state([[2.5, 2.5]], vel=[[1.0, 1.0]])  # blocked corner
        cmap.resolve(state, prev)
        np.testing.assert_array_equal(state.pos[0], prev[0])
        np.testing.assert_array_equal(state.vel[0], np.zeros(2))


class TestResolveInvariant:
    """No resolved position ever lands in a blocked cell."""

    def test_swept_moves_never_end_blocked(
        self, make_state: MakeState
    ) -> None:
        """Many sub-cell moves from open cells never resolve into a block.

        Agents start in guaranteed-open cells and step by less than one cell
        in random directions toward the wall; after resolution none may occupy
        a blocked cell — the core 'obstacles are never crossed' guarantee.
        """
        cmap = _wall_column_map()
        rng = np.random.default_rng(7)
        n = 200
        # Previous positions confined to the open left band (cols 0..1).
        prev: npt.NDArray[np.float64] = rng.uniform(
            low=[0.05, 0.05], high=[1.95, 4.95], size=(n, 2)
        )
        # Integrated step < one cell (0.9 m) in any direction.
        delta = rng.uniform(low=-0.9, high=0.9, size=(n, 2))
        integrated = prev + delta
        state = make_state(integrated, vel=delta.copy())
        cmap.resolve(state, prev)
        assert not np.any(cmap.is_blocked(state.pos))


class TestResolveEdgeCases:
    """resolve edge and failure paths."""

    def test_dead_agents_untouched(self, make_state: MakeState) -> None:
        """A dead agent inside a blocked cell is left unchanged."""
        cmap = _wall_column_map()
        prev = np.array(
            [[1.5, 0.5], [9.9, 9.9]], dtype=np.float64
        )
        state = make_state(
            [[2.5, 0.5], [2.5, 2.5]],
            vel=[[1.0, 0.0], [3.0, 4.0]],
            alive=[True, False],
        )
        dead_pos_before = state.pos[1].copy()
        dead_vel_before = state.vel[1].copy()
        cmap.resolve(state, prev)
        np.testing.assert_array_equal(state.pos[1], dead_pos_before)
        np.testing.assert_array_equal(state.vel[1], dead_vel_before)

    def test_empty_state_no_op(self, make_state: MakeState) -> None:
        """A zero-agent state resolves without error (edge)."""
        cmap = _wall_column_map()
        state = make_state(np.empty((0, 2), dtype=np.float64))
        cmap.resolve(state, np.empty((0, 2), dtype=np.float64))

    def test_prev_pos_wrong_shape_raises(
        self, make_state: MakeState
    ) -> None:
        """A prev_pos whose length disagrees with count raises (failure)."""
        cmap = _wall_column_map()
        state = make_state([[0.5, 0.5]])
        with pytest.raises(ValueError, match="prev_pos must have shape"):
            cmap.resolve(state, np.zeros((3, 2), dtype=np.float64))
