"""Tests for crowd_evac.pathfinding.flow_field.

Covers FlowField construction, the integration/flow solve, bilinear
sampling, multi-exit lowest-cost routing, equidistant tie-breaking, and
bounded re-routing via recompute. Exercises the FR-4 acceptance behaviours:
  - following the field from any walkable cell reaches an exit (no walls);
  - two exits route each cell to the lower-cost exit;
  - an equidistant cell still resolves to a valid draining direction;
  - blocking an exit re-routes affected cells to the other exit;
  - recompute leaves cells far from the block unchanged (bounded).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from crowd_evac.domain.errors import PathfindingError
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan, Wall
from crowd_evac.pathfinding.flow_field import (
    FlowField,
    build_danger_field,
    cells_in_radius,
)

CELL = 1.0

# Exit-opening cells for the two-exit fixtures (cell_size=1.0). The west exit
# (x=1, width 2) spans south-row cols 0-1; the east exit (x=9, width 2) spans
# cols 8-9.
WEST_EXIT_CELLS: list[tuple[int, int]] = [(0, 0), (0, 1)]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def open_room_fp() -> FloorPlan:
    """8 m x 6 m wall-free room with a single south exit at x=4.

    At cell_size=1.0 -> 6 rows x 8 cols, fully walkable; the exit bbox spans
    row 0, cols 3-4.
    """
    return FloorPlan(
        width_m=8.0,
        height_m=6.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(x=4.0, y=0.0, width_m=2.0, side=ExitSide.SOUTH,
                 capacity_per_second=5, label="south"),
        ),
    )


@pytest.fixture
def two_exit_fp() -> FloorPlan:
    """10 m x 6 m room with south exits near each end (x=1 and x=9).

    Cells in the left half are closer to the west exit, the right half to the
    east exit. At cell_size=1.0 -> 6 rows x 10 cols.
    """
    return FloorPlan(
        width_m=10.0,
        height_m=6.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(x=1.0, y=0.0, width_m=2.0, side=ExitSide.SOUTH,
                 capacity_per_second=5, label="west"),
            Exit(x=9.0, y=0.0, width_m=2.0, side=ExitSide.SOUTH,
                 capacity_per_second=5, label="east"),
        ),
    )


@pytest.fixture
def symmetric_two_exit_fp() -> FloorPlan:
    """7 m x 5 m room, single-column south exits symmetric about col 3.

    Each exit is one grid column wide (west -> col 0, east -> col 6), so any
    cell in column 3 is exactly equidistant from both exits — the tie case.
    """
    return FloorPlan(
        width_m=7.0,
        height_m=5.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(x=0.5, y=0.0, width_m=1.0, side=ExitSide.SOUTH,
                 capacity_per_second=5, label="west"),
            Exit(x=6.5, y=0.0, width_m=1.0, side=ExitSide.SOUTH,
                 capacity_per_second=5, label="east"),
        ),
    )


def _follow_to_exit(
    field: FlowField,
    start: tuple[int, int],
    max_steps: int = 1_000,
) -> tuple[int, int]:
    """Walk the flow field from a start cell to a zero-cost (exit) cell.

    Returns the terminal cell. Asserts each step lands on a walkable cell
    with strictly lower cost — i.e. no wall crossing and guaranteed progress.
    """
    r, c = start
    walkable = field._walkable
    for _ in range(max_steps):
        if field.cost[r, c] == 0.0:
            return (r, c)
        dx = field.direction[r, c, 0]
        dy = field.direction[r, c, 1]
        assert (dx, dy) != (0.0, 0.0), f"stuck at {(r, c)} with finite cost"
        nr = r + int(round(dy))
        nc = c + int(round(dx))
        assert walkable[nr, nc], f"stepped onto blocked cell {(nr, nc)}"
        assert field.cost[nr, nc] < field.cost[r, c], "cost did not decrease"
        r, c = nr, nc
    raise AssertionError(f"did not reach an exit within {max_steps} steps")


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestBuild:
    """Test FlowField.build and constructor validation."""

    def test_build_produces_expected_shapes(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Verify cost is (rows, cols) and direction is (rows, cols, 2)."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        assert field.cost.shape == (6, 8)
        assert field.direction.shape == (6, 8, 2)

    def test_exit_cells_have_zero_cost(self, open_room_fp: FloorPlan) -> None:
        """Verify the exit-opening cells are the zero-cost sources."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        assert field.cost[0, 3] == 0.0
        assert field.cost[0, 4] == 0.0

    def test_cost_increases_with_distance_from_exit(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Verify a far cell costs more than a near cell on the same column."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        assert field.cost[5, 4] > field.cost[1, 4]

    def test_invalid_cell_size_raises(self, open_room_fp: FloorPlan) -> None:
        """Verify a non-positive cell_size raises ValueError."""
        with pytest.raises(ValueError, match="cell_size"):
            FlowField.build(open_room_fp, cell_size=0.0)

    def test_no_walkable_exit_raises(self) -> None:
        """Verify a field whose every exit cell is blocked raises.

        Constructed directly with a mask whose exit row is impassable, so the
        Dijkstra solve has no walkable source.
        """
        walkable: npt.NDArray[np.bool_] = np.ones((4, 4), dtype=np.bool_)
        walkable[0, :] = False
        with pytest.raises(PathfindingError, match="exit"):
            FlowField(CELL, walkable, [(0, 1), (0, 2)])


# ---------------------------------------------------------------------------
# Reachability (R4.1 / R4.2)
# ---------------------------------------------------------------------------


class TestReachability:
    """Following the field reaches an exit without crossing walls."""

    def test_every_walkable_cell_reaches_exit_open_room(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Verify every walkable cell drains to an exit in an open room."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        rows, cols = field.cost.shape
        for r in range(rows):
            for c in range(cols):
                if field._walkable[r, c]:
                    terminal = _follow_to_exit(field, (r, c))
                    assert field.cost[terminal] == 0.0

    def test_path_routes_around_wall_without_crossing(self) -> None:
        """Verify cells behind a wall finger detour around it, not through it.

        A vertical wall splits the room except for a gap at the top; a cell on
        the far side must detour through the gap (the follow helper asserts no
        blocked cell is ever entered).
        """
        fp = FloorPlan(
            width_m=6.0,
            height_m=6.0,
            walls=(Wall(x=3.0, y=0.0, width=1.0, height=5.0, label="divider"),),
            obstacles=(),
            exits=(
                Exit(x=1.0, y=0.0, width_m=2.0, side=ExitSide.SOUTH,
                     capacity_per_second=5, label="south"),
            ),
        )
        field = FlowField.build(fp, cell_size=CELL)
        terminal = _follow_to_exit(field, (0, 5))
        assert field.cost[terminal] == 0.0


# ---------------------------------------------------------------------------
# Multi-exit routing (R4.2)
# ---------------------------------------------------------------------------


class TestMultiExitRouting:
    """Cells route toward the lower-cost of two exits."""

    def test_left_cell_routes_west(self, two_exit_fp: FloorPlan) -> None:
        """Verify a far-left cell flows toward negative x (west exit)."""
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        assert field.direction[5, 0, 0] <= 0.0

    def test_right_cell_routes_east(self, two_exit_fp: FloorPlan) -> None:
        """Verify a far-right cell flows toward positive x (east exit)."""
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        assert field.direction[5, 9, 0] >= 0.0

    def test_cost_is_distance_to_nearer_exit(
        self, two_exit_fp: FloorPlan
    ) -> None:
        """Verify each cell's cost is the distance to its nearer exit.

        Cell (0, 3) is 2 cells from the west opening (col 1) and 5 from the
        east (col 8); cell (0, 6) is the mirror. A cost of 2.0 at each proves
        the cell took the lower-cost exit, not a single fixed one.
        """
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        assert np.isclose(field.cost[0, 3], 2.0)  # via west, not 5.0 via east
        assert np.isclose(field.cost[0, 6], 2.0)  # via east, not 5.0 via west


# ---------------------------------------------------------------------------
# Equidistant tie-breaking (R4.2 edge case)
# ---------------------------------------------------------------------------


class TestEquidistantTie:
    """A cell equidistant from two exits resolves to one valid direction."""

    def test_equidistant_costs_are_symmetric(
        self, symmetric_two_exit_fp: FloorPlan
    ) -> None:
        """Verify mirror cells about the centre column have equal cost."""
        field = FlowField.build(symmetric_two_exit_fp, cell_size=CELL)
        assert np.isclose(field.cost[2, 1], field.cost[2, 5])
        assert np.isclose(field.cost[4, 0], field.cost[4, 6])

    def test_equidistant_cell_has_nonzero_direction(
        self, symmetric_two_exit_fp: FloorPlan
    ) -> None:
        """Verify the tied centre cell is not stuck (deterministic break)."""
        field = FlowField.build(symmetric_two_exit_fp, cell_size=CELL)
        dx = field.direction[2, 3, 0]
        dy = field.direction[2, 3, 1]
        assert (dx, dy) != (0.0, 0.0)

    def test_equidistant_cell_still_drains_to_exit(
        self, symmetric_two_exit_fp: FloorPlan
    ) -> None:
        """Verify following the field from the tied cell reaches an exit."""
        field = FlowField.build(symmetric_two_exit_fp, cell_size=CELL)
        terminal = _follow_to_exit(field, (2, 3))
        assert field.cost[terminal] == 0.0


# ---------------------------------------------------------------------------
# Sampling (bilinear -> f_exit direction)
# ---------------------------------------------------------------------------


class TestSample:
    """Bilinear sampling produces unit exit-seeking directions."""

    def test_sample_returns_unit_vectors(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Verify sampled directions are unit length away from the exit."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        pos = np.array([[4.0, 4.5], [2.0, 3.0]], dtype=np.float64)
        norms = np.linalg.norm(field.sample(pos), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_sample_points_toward_exit(self, open_room_fp: FloorPlan) -> None:
        """Verify an agent above the exit is pushed in -y (toward south)."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        out = field.sample(np.array([[4.0, 4.5]], dtype=np.float64))
        assert out[0, 1] < 0.0

    def test_sample_clamps_out_of_bounds(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Verify a position outside the grid clamps instead of erroring."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        out = field.sample(np.array([[-5.0, 100.0]], dtype=np.float64))
        assert out.shape == (1, 2)
        assert np.isfinite(out).all()

    def test_sample_empty_input_returns_empty(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Verify an empty position array yields an empty (0, 2) result."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        out = field.sample(np.empty((0, 2), dtype=np.float64))
        assert out.shape == (0, 2)

    def test_sample_rejects_bad_shape(self, open_room_fp: FloorPlan) -> None:
        """Verify a non-(N, 2) array raises ValueError."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        with pytest.raises(ValueError, match="shape"):
            field.sample(np.array([1.0, 2.0, 3.0], dtype=np.float64))


# ---------------------------------------------------------------------------
# Re-routing via recompute (R4.3 / R4.4)
# ---------------------------------------------------------------------------


class TestRecompute:
    """Blocking cells re-routes affected agents; far cells are untouched."""

    def test_blocking_exit_reroutes_to_other_exit(
        self, two_exit_fp: FloorPlan
    ) -> None:
        """Verify blocking the west exit flips a left cell toward the east."""
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        assert field.direction[3, 1, 0] <= 0.0  # routes west / straight down
        rerouted = field.recompute(WEST_EXIT_CELLS)
        assert rerouted.direction[3, 1, 0] > 0.0  # now routes east

    def test_recompute_leaves_far_cells_unchanged(
        self, two_exit_fp: FloorPlan
    ) -> None:
        """Verify cells already draining to the surviving exit do not change."""
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        rerouted = field.recompute(WEST_EXIT_CELLS)
        assert np.array_equal(
            field.direction[:, 9, :], rerouted.direction[:, 9, :]
        )

    def test_recompute_blocking_all_exits_raises(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Verify removing the last exit's cells raises PathfindingError."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        with pytest.raises(PathfindingError, match="exit"):
            field.recompute([(0, 3), (0, 4)])

    def test_recompute_ignores_out_of_bounds(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Verify out-of-bounds blocked cells are silently ignored."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        rerouted = field.recompute([(-1, -1), (99, 99)])
        assert np.array_equal(field.direction, rerouted.direction)


# ---------------------------------------------------------------------------
# Absolute re-blocking via with_blocks (restoration — bug-fix for the
# permanent-block defect)
# ---------------------------------------------------------------------------


class TestWithBlocks:
    """with_blocks replaces the block set, restoring previously-blocked cells."""

    def test_blocked_empty_on_fresh_field(
        self, open_room_fp: FloorPlan
    ) -> None:
        """A freshly built field has an empty blocked set (happy)."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        assert field.blocked == frozenset()

    def test_with_blocks_records_blocked_set(
        self, two_exit_fp: FloorPlan
    ) -> None:
        """with_blocks records exactly the in-bounds cells it blocked."""
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        blocked = field.with_blocks([(2, 5), (3, 5)])
        assert blocked.blocked == frozenset({(2, 5), (3, 5)})
        assert not np.isfinite(blocked.cost[2, 5])

    def test_with_blocks_restores_previously_blocked_cell(
        self, two_exit_fp: FloorPlan
    ) -> None:
        """Re-solving with a smaller set restores the freed cell's cost (edge).

        This is the core of the permanent-block fix: a cell blocked by one
        call becomes walkable again when the next absolute set omits it.
        """
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        original = float(field.cost[3, 5])

        blocked = field.with_blocks([(3, 5)])
        assert not np.isfinite(blocked.cost[3, 5])

        restored = blocked.with_blocks([])  # clear all blocks
        assert restored.blocked == frozenset()
        assert restored.cost[3, 5] == pytest.approx(original)

    def test_with_blocks_preserves_pristine_base_across_chain(
        self, two_exit_fp: FloorPlan
    ) -> None:
        """Chained blocks always re-solve from the pristine mask, not the last.

        Block cell A, then (separately) block cell B; A must be walkable again
        in the B-only field — proving accumulation does not leak across calls.
        """
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        only_b = field.with_blocks([(2, 4)]).with_blocks([(2, 6)])
        assert only_b.blocked == frozenset({(2, 6)})
        assert np.isfinite(only_b.cost[2, 4])  # A restored
        assert not np.isfinite(only_b.cost[2, 6])  # B blocked

    def test_with_blocks_all_exits_raises(
        self, open_room_fp: FloorPlan
    ) -> None:
        """Blocking every exit cell raises PathfindingError (failure)."""
        field = FlowField.build(open_room_fp, cell_size=CELL)
        with pytest.raises(PathfindingError, match="exit"):
            field.with_blocks([(0, 3), (0, 4)])


# ---------------------------------------------------------------------------
# cells_in_radius helper
# ---------------------------------------------------------------------------


class TestCellsInRadius:
    """cells_in_radius selects grid cells whose centres are within range."""

    def test_centre_cell_selected(self) -> None:
        """A small radius selects at least the cell containing the point."""
        cells = cells_in_radius((2.5, 2.5), 0.4, 1.0, (6, 6))
        assert (2, 2) in cells

    def test_radius_scales_count(self) -> None:
        """A larger radius selects strictly more cells (happy)."""
        small = cells_in_radius((2.5, 2.5), 0.5, 1.0, (6, 6))
        large = cells_in_radius((2.5, 2.5), 2.0, 1.0, (6, 6))
        assert len(large) > len(small)

    def test_out_of_bounds_clipped(self) -> None:
        """A point at a corner yields only in-bounds cells (edge)."""
        cells = cells_in_radius((0.0, 0.0), 1.5, 1.0, (6, 6))
        assert cells  # non-empty
        assert all(0 <= r < 6 and 0 <= c < 6 for r, c in cells)


# ---------------------------------------------------------------------------
# Danger-cost routing: agents divert to the next-best exit around a hazard
# ---------------------------------------------------------------------------


class TestDangerField:
    """build_danger_field shape and graded-cost rerouting via with_hazards."""

    def test_danger_peaks_at_centre_zero_outside(self) -> None:
        """Danger is 1 at the hazard centre cell and 0 beyond its radius."""
        # Hazard centred on a cell centre (col 4 -> x=4.5, row 3 -> y=3.5).
        danger = build_danger_field((6, 8), 1.0, [(4.5, 3.5, 2.0)])
        assert danger[3, 4] == pytest.approx(1.0)  # exact core cell
        assert danger[0, 0] == 0.0  # corner, well outside the 2 m radius

    def test_no_hazards_is_all_zero(self) -> None:
        """An empty hazard list yields an all-zero field (edge)."""
        danger = build_danger_field((5, 5), 1.0, [])
        assert not np.any(danger)

    def test_non_positive_radius_skipped(self) -> None:
        """A hazard with non-positive radius contributes nothing (edge)."""
        danger = build_danger_field((5, 5), 1.0, [(2.0, 2.0, 0.0)])
        assert not np.any(danger)

    def test_danger_reroutes_to_other_exit(
        self, two_exit_fp: FloorPlan
    ) -> None:
        """A danger field near the east exit flips an east-bound cell west.

        At (7.5, 3.0) the cell is closer to the east exit, so the plain field
        routes east; overlaying a strong danger cost around (8, 3) makes the
        eastward route more expensive than the longer westward detour, so the
        direction flips — the crowd diverts to the other exit.
        """
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        probe = np.array([[7.5, 3.0]])
        assert field.sample(probe)[0, 0] > 0.0  # routes east by default

        danger = build_danger_field(field.cost.shape, CELL, [(8.0, 3.0, 3.0)])
        hazardous = field.with_hazards([], 50.0 * danger)
        assert hazardous.sample(probe)[0, 0] < 0.0  # now routes west

    def test_danger_keeps_cells_walkable(
        self, two_exit_fp: FloorPlan
    ) -> None:
        """Danger raises cost but never blocks: hazard cells stay finite.

        This is the freeze-avoidance guarantee — a soft cost never produces a
        zero-direction trap the way a hard block does.
        """
        field = FlowField.build(two_exit_fp, cell_size=CELL)
        danger = build_danger_field(field.cost.shape, CELL, [(8.0, 3.0, 3.0)])
        hazardous = field.with_hazards([], 50.0 * danger)
        assert np.all(np.isfinite(hazardous.cost[field.cost != np.inf]))
        # A cell in the danger core still routes somewhere (non-zero dir).
        core = hazardous.sample(np.array([[8.0, 3.0]]))
        assert np.linalg.norm(core[0]) > 0.0
