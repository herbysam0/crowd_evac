"""Tests for crowd_evac.domain.exit_model (FR-5 R5.1 / R5.2).

Covers:
  - ExitModel construction: valid, invalid dt, invalid capture_radius
  - Egress rate bounded by capacity_per_second over a 1 s window (R5.1)
  - Queue forms when arrivals exceed per-tick capacity
  - Evacuated count equals initial population on a fully solvable scenario
  - Egressed agent is removed from AgentState (R5.2)
  - ticks_since_last_egress resets on egress, accumulates otherwise
  - Segment-distance capture: wide exit captures agents along its full span
  - Agent outside exit's lateral span and beyond capture_radius not captured
  - Multi-exit: agent is routed to nearest exit; goal entry is updated
  - _segment_distances pure-function geometry
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.constants import DT
from crowd_evac.domain.exit_model import ExitModel, _segment_distances
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan

from .conftest import MakeState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def south_exit() -> Exit:
    """2 m-wide SOUTH exit centred at (5, 0), capacity 5 agents/s.

    Opening segment spans x ∈ [4, 6] at y = 0.
    """
    return Exit(
        x=5.0,
        y=0.0,
        width_m=2.0,
        side=ExitSide.SOUTH,
        capacity_per_second=5,
        label="south",
    )


@pytest.fixture
def simple_floor(south_exit: Exit) -> FloorPlan:
    """10 × 10 m room with a single SOUTH exit."""
    return FloorPlan(
        width_m=10.0,
        height_m=10.0,
        walls=(),
        obstacles=(),
        exits=(south_exit,),
    )


@pytest.fixture
def two_exit_floor() -> FloorPlan:
    """10 × 10 m room with SOUTH exits at x = 1 (west) and x = 9 (east)."""
    return FloorPlan(
        width_m=10.0,
        height_m=10.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=1.0,
                y=0.0,
                width_m=1.0,
                side=ExitSide.SOUTH,
                capacity_per_second=5,
                label="west",
            ),
            Exit(
                x=9.0,
                y=0.0,
                width_m=1.0,
                side=ExitSide.SOUTH,
                capacity_per_second=5,
                label="east",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestExitModelConstruction:
    """ExitModel.__init__ — valid and invalid arguments."""

    def test_init_valid_sets_zero_counters(
        self, simple_floor: FloorPlan
    ) -> None:
        """Freshly constructed model has zero evacuated and stall counters."""
        model = ExitModel(simple_floor, dt=DT)
        assert model.evacuated_count == 0
        assert model.ticks_since_last_egress == 0

    def test_init_nonpositive_dt_raises(
        self, simple_floor: FloorPlan
    ) -> None:
        """dt ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            ExitModel(simple_floor, dt=0.0)

    def test_init_nonpositive_capture_radius_raises(
        self, simple_floor: FloorPlan
    ) -> None:
        """capture_radius ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="capture_radius must be positive"):
            ExitModel(simple_floor, capture_radius=-0.5)


# ---------------------------------------------------------------------------
# Egress capacity (R5.1)
# ---------------------------------------------------------------------------


class TestEgressCapacity:
    """Throughput never exceeds exit capacity; queue forms on overflow."""

    def test_egress_bounded_over_one_second_window(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Total egress in 20 ticks (1 s at DT=0.05) ≤ capacity_per_second."""
        state = make_state([(5.0, 0.3)] * 50)
        model = ExitModel(simple_floor, dt=DT)
        for _ in range(round(1.0 / DT)):  # 20 ticks
            model.step(state)
        assert model.evacuated_count <= 5  # capacity_per_second == 5

    def test_queue_forms_when_arrivals_exceed_per_tick_capacity(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """No egress on tick 1 (tokens = 0.25 < 1); all 10 agents still alive."""
        state = make_state([(5.0, 0.3)] * 10)
        model = ExitModel(simple_floor, dt=DT)
        model.step(state)
        assert model.evacuated_count == 0
        assert state.active_indices.size == 10

    def test_evacuated_count_equals_initial_on_solvable_scenario(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """All 5 agents egress within 1 s on a fully solvable scenario."""
        n = 5
        state = make_state([(5.0, 0.3)] * n)
        model = ExitModel(simple_floor, dt=DT)
        # 20 ticks × 0.25 tokens/tick = 5.0 tokens → 5 egresses.
        for _ in range(round(1.0 / DT)):
            model.step(state)
        assert model.evacuated_count == n
        assert state.active_indices.size == 0

    def test_egress_removes_agent_from_state(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent alive flag is cleared on egress (R5.2)."""
        state = make_state([(5.0, 0.3)])
        model = ExitModel(simple_floor, dt=DT)
        # Tokens reach 1.0 on tick 4 (4 × 0.25 = 1.0).
        for _ in range(4):
            model.step(state)
        assert not state.alive[0]
        assert model.evacuated_count == 1


# ---------------------------------------------------------------------------
# Stall counter
# ---------------------------------------------------------------------------


class TestStallCounter:
    """ticks_since_last_egress tracks the no-egress streak."""

    def test_counter_increments_when_no_egress(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Counter grows each tick when no agents enter the exit zone."""
        state = make_state([(5.0, 9.0)])  # far from exit
        model = ExitModel(simple_floor, dt=DT)
        model.step(state)
        assert model.ticks_since_last_egress == 1
        model.step(state)
        assert model.ticks_since_last_egress == 2

    def test_counter_resets_to_zero_on_egress_tick(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Counter resets to zero on the tick in which egress occurs."""
        state = make_state([(5.0, 0.3)])
        model = ExitModel(simple_floor, dt=DT)
        for _ in range(3):
            model.step(state)
        assert model.ticks_since_last_egress > 0
        model.step(state)  # tick 4: tokens = 1.0, agent egresses
        assert model.ticks_since_last_egress == 0

    def test_trapped_agent_accumulates_stall_ticks(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Stall counter reaches target for an immobile agent beyond capture."""
        state = make_state([(5.0, 9.0)])  # beyond capture_radius, no velocity
        model = ExitModel(simple_floor, dt=DT)
        stall_window = 10
        for _ in range(stall_window):
            model.step(state)
        assert model.ticks_since_last_egress >= stall_window


# ---------------------------------------------------------------------------
# Segment-distance capture geometry
# ---------------------------------------------------------------------------


class TestSegmentDistanceCapture:
    """Capture uses perpendicular distance to the exit segment."""

    def test_agent_at_exit_lateral_edge_is_captured(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent near the lateral edge of a wide exit is captured.

        Opening spans x ∈ [4, 6].  Agent at (4.1, 0.3):
        segment distance = 0.3 m < capture_radius (1.0 m).
        """
        state = make_state([(4.1, 0.3)])
        model = ExitModel(simple_floor, dt=DT)
        for _ in range(4):  # 4 ticks → 1 token → 1 egress
            model.step(state)
        assert model.evacuated_count == 1

    def test_agent_outside_lateral_span_and_far_not_captured(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent past exit lateral extent is not captured.

        Opening spans x ∈ [4, 6].  Agent at (0.0, 0.5):
        nearest segment point is (4, 0); distance ≈ 4.03 m > 1.0 m.
        """
        state = make_state([(0.0, 0.5)])
        model = ExitModel(simple_floor, dt=DT)
        for _ in range(round(1.0 / DT)):
            model.step(state)
        assert model.evacuated_count == 0
        assert state.active_indices.size == 1

    def test_segment_distances_horizontal_exit(self) -> None:
        """_segment_distances returns correct values for a SOUTH exit."""
        centres = np.array([[5.0, 0.0]], dtype=np.float64)
        half_widths = np.array([1.0], dtype=np.float64)
        is_horiz = np.array([True], dtype=np.bool_)

        # Agent directly above centre: nearest point (5, 0), dist = 0.5.
        d = _segment_distances(
            np.array([[5.0, 0.5]], dtype=np.float64),
            centres, half_widths, is_horiz,
        )
        assert abs(float(d[0, 0]) - 0.5) < 1e-9

        # Agent outside left edge: nearest point (4, 0), dist = 1.0.
        d2 = _segment_distances(
            np.array([[3.0, 0.0]], dtype=np.float64),
            centres, half_widths, is_horiz,
        )
        assert abs(float(d2[0, 0]) - 1.0) < 1e-9

    def test_segment_distances_vertical_exit(self) -> None:
        """_segment_distances returns correct values for an EAST exit."""
        # EAST exit at (10, 5), width 2 → segment y ∈ [4, 6] at x = 10.
        centres = np.array([[10.0, 5.0]], dtype=np.float64)
        half_widths = np.array([1.0], dtype=np.float64)
        is_horiz = np.array([False], dtype=np.bool_)  # EAST → vertical

        # Agent at (9.5, 5.0): nearest point (10, 5), dist = 0.5.
        d = _segment_distances(
            np.array([[9.5, 5.0]], dtype=np.float64),
            centres, half_widths, is_horiz,
        )
        assert abs(float(d[0, 0]) - 0.5) < 1e-9

        # Agent at (10.0, 2.0): nearest point (10, 4), dist = 2.0.
        d2 = _segment_distances(
            np.array([[10.0, 2.0]], dtype=np.float64),
            centres, half_widths, is_horiz,
        )
        assert abs(float(d2[0, 0]) - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Multi-exit routing
# ---------------------------------------------------------------------------


class TestMultiExitRouting:
    """Agents are queued to the nearest exit and goal entry is updated."""

    def test_agent_queued_to_nearest_exit(
        self, two_exit_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent at (1.0, 0.5) routes to west exit (index 0), not east."""
        state = make_state([(1.0, 0.5)])
        model = ExitModel(two_exit_floor, dt=DT)
        model.step(state)
        assert len(model._queues[0]) == 1  # west queue
        assert len(model._queues[1]) == 0  # east queue empty

    def test_goal_set_to_exit_index_on_capture(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """goal[i] is updated to the exit index when agent is captured."""
        state = make_state([(5.0, 0.3)])
        assert state.goal[0] == -1  # unassigned before capture
        model = ExitModel(simple_floor, dt=DT)
        model.step(state)
        assert state.goal[0] == 0  # exit index 0

    def test_agents_split_to_respective_nearest_exits(
        self, two_exit_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Two agents near different exits each join the correct queue."""
        # Agent 0 near west (1, 0.3); agent 1 near east (9, 0.3).
        state = make_state([(1.0, 0.3), (9.0, 0.3)])
        model = ExitModel(two_exit_floor, dt=DT)
        model.step(state)
        assert len(model._queues[0]) == 1  # west
        assert len(model._queues[1]) == 1  # east
