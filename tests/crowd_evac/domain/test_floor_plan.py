"""Tests for crowd_evac.domain.floor_plan.

Covers FloorPlan construction validation, walkable_mask grid computation,
and exit_cells identification. Wall, Obstacle, and Exit are exercised as
components throughout.
"""
from __future__ import annotations

import math

import pytest

from crowd_evac.domain.errors import ScenarioValidationError
from crowd_evac.domain.floor_plan import (
    Exit,
    ExitSide,
    FloorPlan,
    Obstacle,
    Rect,
    Wall,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_fp() -> FloorPlan:
    """Minimal 6 m × 4 m room: one south wall (y=0..1) and one south exit.

    Grid at cell_size=1.0 → 4 rows × 6 cols.
    Exit centre (3, 0.5), width=2 m → bbox x∈[2,4], y∈[0,1]
    → row 0, cols 2–3 are exit cells.
    """
    return FloorPlan(
        width_m=6.0,
        height_m=4.0,
        walls=(Wall(x=0.0, y=0.0, width=6.0, height=1.0, label="south_wall"),),
        obstacles=(),
        exits=(
            Exit(
                x=3.0,
                y=0.5,
                width_m=2.0,
                side=ExitSide.SOUTH,
                capacity_per_second=5,
                label="main_exit",
            ),
        ),
    )


@pytest.fixture
def fp_with_obstacle(minimal_fp: FloorPlan) -> FloorPlan:
    """Extends minimal_fp with one interior obstacle at x=1..3, y=2..3."""
    return FloorPlan(
        width_m=minimal_fp.width_m,
        height_m=minimal_fp.height_m,
        walls=minimal_fp.walls,
        obstacles=(Obstacle(x=1.0, y=2.0, width=2.0, height=1.0, label="block"),),
        exits=minimal_fp.exits,
    )


# ---------------------------------------------------------------------------
# Rect / Wall / Obstacle primitives
# ---------------------------------------------------------------------------


class TestRectPrimitive:
    """Test the Rect base dataclass."""

    def test_rect_stores_fields(self) -> None:
        """Verify Rect preserves all four fields."""
        r = Rect(x=1.0, y=2.0, width=3.0, height=4.0)
        assert r.x == 1.0
        assert r.y == 2.0
        assert r.width == 3.0
        assert r.height == 4.0

    def test_rect_is_frozen(self) -> None:
        """Verify Rect cannot be mutated after construction."""
        r = Rect(x=0.0, y=0.0, width=1.0, height=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            r.x = 99.0  # type: ignore[misc]

    def test_wall_inherits_rect(self) -> None:
        """Verify Wall is a Rect subclass with an optional label."""
        w = Wall(x=0.0, y=0.0, width=5.0, height=0.5)
        assert isinstance(w, Rect)
        assert w.label == ""

    def test_obstacle_inherits_rect(self) -> None:
        """Verify Obstacle is a Rect subclass with an optional label."""
        o = Obstacle(x=1.0, y=1.0, width=2.0, height=1.0, label="pillar")
        assert isinstance(o, Rect)
        assert o.label == "pillar"


# ---------------------------------------------------------------------------
# FloorPlan construction
# ---------------------------------------------------------------------------


class TestFloorPlanConstruction:
    """Test FloorPlan construction and validation."""

    # -- Happy path ---------------------------------------------------------

    def test_valid_plan_constructs(self, minimal_fp: FloorPlan) -> None:
        """Verify a well-formed FloorPlan constructs without error."""
        assert minimal_fp.width_m == 6.0
        assert minimal_fp.height_m == 4.0
        assert len(minimal_fp.walls) == 1
        assert len(minimal_fp.exits) == 1

    def test_no_obstacles_is_valid(self) -> None:
        """Verify a FloorPlan with empty obstacles is valid."""
        fp = FloorPlan(
            width_m=5.0,
            height_m=5.0,
            walls=(),
            obstacles=(),
            exits=(
                Exit(
                    x=2.5, y=0.0, width_m=1.0,
                    side=ExitSide.SOUTH, capacity_per_second=3,
                ),
            ),
        )
        assert fp.obstacles == ()

    def test_multiple_exits_allowed(self) -> None:
        """Verify FloorPlan accepts multiple exits."""
        fp = FloorPlan(
            width_m=10.0,
            height_m=8.0,
            walls=(),
            obstacles=(),
            exits=(
                Exit(x=5.0, y=0.0, width_m=2.0,
                     side=ExitSide.SOUTH, capacity_per_second=5),
                Exit(x=0.0, y=4.0, width_m=1.5,
                     side=ExitSide.WEST, capacity_per_second=3),
            ),
        )
        assert len(fp.exits) == 2

    # -- Edge case ----------------------------------------------------------

    def test_immutable_after_construction(self, minimal_fp: FloorPlan) -> None:
        """Verify FloorPlan fields cannot be mutated (frozen dataclass)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            minimal_fp.width_m = 99.0  # type: ignore[misc]

    # -- Failure path -------------------------------------------------------

    def test_zero_width_raises(self) -> None:
        """Verify width_m=0 raises ScenarioValidationError."""
        with pytest.raises(ScenarioValidationError, match="width_m"):
            FloorPlan(
                width_m=0.0,
                height_m=4.0,
                walls=(),
                obstacles=(),
                exits=(Exit(x=0.0, y=0.0, width_m=1.0,
                            side=ExitSide.SOUTH, capacity_per_second=1),),
            )

    def test_zero_height_m_raises(self) -> None:
        """Verify height_m=0 raises ScenarioValidationError.

        height_m is the y-axis span of the floor plan; both axes must be
        positive to form a valid 2-D walkable area for grid rasterisation.
        """
        with pytest.raises(ScenarioValidationError, match="height_m"):
            FloorPlan(
                width_m=5.0,
                height_m=0.0,
                walls=(),
                obstacles=(),
                exits=(Exit(x=0.0, y=0.0, width_m=1.0,
                            side=ExitSide.SOUTH, capacity_per_second=1),),
            )

    def test_no_exits_raises(self) -> None:
        """Verify an exits-free FloorPlan raises ScenarioValidationError."""
        with pytest.raises(ScenarioValidationError, match="at least one exit"):
            FloorPlan(
                width_m=5.0,
                height_m=5.0,
                walls=(),
                obstacles=(),
                exits=(),
            )

    def test_zero_exit_width_raises(self) -> None:
        """Verify Exit with width_m=0 raises ScenarioValidationError."""
        with pytest.raises(ScenarioValidationError, match="width_m"):
            FloorPlan(
                width_m=5.0,
                height_m=5.0,
                walls=(),
                obstacles=(),
                exits=(Exit(x=1.0, y=0.0, width_m=0.0,
                            side=ExitSide.SOUTH, capacity_per_second=1),),
            )

    def test_zero_exit_capacity_raises(self) -> None:
        """Verify Exit with capacity_per_second=0 raises ScenarioValidationError."""
        with pytest.raises(ScenarioValidationError, match="capacity_per_second"):
            FloorPlan(
                width_m=5.0,
                height_m=5.0,
                walls=(),
                obstacles=(),
                exits=(Exit(x=1.0, y=0.0, width_m=1.0,
                            side=ExitSide.SOUTH, capacity_per_second=0),),
            )

    def test_zero_wall_width_raises(self) -> None:
        """Verify Wall with width=0 raises ScenarioValidationError."""
        with pytest.raises(ScenarioValidationError, match="Wall"):
            FloorPlan(
                width_m=5.0,
                height_m=5.0,
                walls=(Wall(x=0.0, y=0.0, width=0.0, height=0.5),),
                obstacles=(),
                exits=(Exit(x=1.0, y=0.0, width_m=1.0,
                            side=ExitSide.SOUTH, capacity_per_second=1),),
            )


# ---------------------------------------------------------------------------
# walkable_mask
# ---------------------------------------------------------------------------


class TestWalkableMask:
    """Test FloorPlan.walkable_mask grid computation."""

    def test_interior_cells_are_walkable(self, minimal_fp: FloorPlan) -> None:
        """Verify rows above the south wall are all True."""
        mask = minimal_fp.walkable_mask(cell_size=1.0)
        assert mask[1:, :].all()

    def test_wall_cells_are_blocked(self, minimal_fp: FloorPlan) -> None:
        """Verify non-exit cells in the wall row are False."""
        mask = minimal_fp.walkable_mask(cell_size=1.0)
        # Row 0 cols 0, 1, 4, 5 are covered by the wall and not by the exit
        assert not mask[0, 0]
        assert not mask[0, 1]
        assert not mask[0, 4]
        assert not mask[0, 5]

    def test_exit_overrides_wall(self, minimal_fp: FloorPlan) -> None:
        """Verify cells within the exit bbox are True despite wall placement."""
        mask = minimal_fp.walkable_mask(cell_size=1.0)
        # Exit bbox covers row 0, cols 2 and 3
        assert mask[0, 2]
        assert mask[0, 3]

    def test_mask_shape_matches_room_dimensions(
        self, minimal_fp: FloorPlan
    ) -> None:
        """Verify mask shape is (ceil(height_m/cs), ceil(width_m/cs))."""
        cell_size = 0.5
        mask = minimal_fp.walkable_mask(cell_size=cell_size)
        expected_rows = math.ceil(minimal_fp.height_m / cell_size)
        expected_cols = math.ceil(minimal_fp.width_m / cell_size)
        assert mask.shape == (expected_rows, expected_cols)

    def test_obstacle_blocks_cells(
        self, fp_with_obstacle: FloorPlan
    ) -> None:
        """Verify obstacle cells are marked non-walkable."""
        mask = fp_with_obstacle.walkable_mask(cell_size=1.0)
        # Obstacle at x=1..3, y=2..3 → row 2, cols 1 and 2
        assert not mask[2, 1]
        assert not mask[2, 2]
        assert mask[2, 0]
        assert mask[2, 3]

    def test_wall_free_room_is_fully_walkable(self) -> None:
        """Verify a room with no walls produces an all-True mask."""
        fp = FloorPlan(
            width_m=4.0,
            height_m=4.0,
            walls=(),
            obstacles=(),
            exits=(
                Exit(x=2.0, y=0.0, width_m=1.0,
                     side=ExitSide.SOUTH, capacity_per_second=5),
            ),
        )
        mask = fp.walkable_mask(cell_size=1.0)
        assert mask.all()


# ---------------------------------------------------------------------------
# exit_cells
# ---------------------------------------------------------------------------


class TestExitCells:
    """Test FloorPlan.exit_cells returns the correct grid indices."""

    def test_single_exit_returns_correct_cells(
        self, minimal_fp: FloorPlan
    ) -> None:
        """Verify exit_cells returns the expected (row, col) pairs."""
        cells = minimal_fp.exit_cells(cell_size=1.0)
        assert (0, 2) in cells
        assert (0, 3) in cells
        assert len(cells) == 2

    def test_two_exits_return_cells_from_both(self) -> None:
        """Verify two exits contribute independent cell sets."""
        fp = FloorPlan(
            width_m=10.0,
            height_m=8.0,
            walls=(),
            obstacles=(),
            exits=(
                Exit(x=2.0, y=0.5, width_m=2.0,
                     side=ExitSide.SOUTH, capacity_per_second=5),
                Exit(x=8.0, y=0.5, width_m=2.0,
                     side=ExitSide.SOUTH, capacity_per_second=5),
            ),
        )
        cells = fp.exit_cells(cell_size=1.0)
        # First exit bbox x∈[1,3], row 0 → cols 1,2
        assert (0, 1) in cells
        assert (0, 2) in cells
        # Second exit bbox x∈[7,9], row 0 → cols 7,8
        assert (0, 7) in cells
        assert (0, 8) in cells

    def test_exit_cells_are_walkable_in_mask(
        self, minimal_fp: FloorPlan
    ) -> None:
        """Verify every exit cell is True in the corresponding walkable_mask."""
        cell_size = 0.5
        mask = minimal_fp.walkable_mask(cell_size=cell_size)
        for r, c in minimal_fp.exit_cells(cell_size=cell_size):
            assert mask[r, c], f"Exit cell ({r},{c}) is not walkable in mask"


# ---------------------------------------------------------------------------
# Cell-size validation (shared for both grid methods)
# ---------------------------------------------------------------------------


class TestGridMethodCellSizeValidation:
    """Verify both grid methods reject non-positive cell_size values."""

    @pytest.mark.parametrize("cell_size", [0.0, -0.5, -1.0])
    def test_walkable_mask_rejects_invalid_cell_size(
        self, minimal_fp: FloorPlan, cell_size: float
    ) -> None:
        """Verify walkable_mask raises ValueError for non-positive cell_size."""
        with pytest.raises(ValueError, match="cell_size"):
            minimal_fp.walkable_mask(cell_size=cell_size)

    @pytest.mark.parametrize("cell_size", [0.0, -0.5, -1.0])
    def test_exit_cells_rejects_invalid_cell_size(
        self, minimal_fp: FloorPlan, cell_size: float
    ) -> None:
        """Verify exit_cells raises ValueError for non-positive cell_size."""
        with pytest.raises(ValueError, match="cell_size"):
            minimal_fp.exit_cells(cell_size=cell_size)


# ---------------------------------------------------------------------------
# ExitSide enum
# ---------------------------------------------------------------------------


class TestExitSide:
    """Test ExitSide enum values and construction."""

    def test_all_four_cardinal_sides_defined(self) -> None:
        """Verify all four cardinal directions are present."""
        sides = {s.value for s in ExitSide}
        assert sides == {"north", "south", "east", "west"}

    def test_exit_side_from_string(self) -> None:
        """Verify ExitSide can be constructed from its string value."""
        assert ExitSide("south") is ExitSide.SOUTH
        assert ExitSide("north") is ExitSide.NORTH

    def test_invalid_side_raises_value_error(self) -> None:
        """Verify an unrecognised string raises ValueError."""
        with pytest.raises(ValueError):
            ExitSide("diagonal")
