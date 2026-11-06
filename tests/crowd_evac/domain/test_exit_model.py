"""Tests for crowd_evac.domain.exit_model (FR-5 R5.1 / R5.2).

Covers:
  - ExitModel construction: valid, invalid dt, invalid agent_radius
  - Egress rate bounded by capacity_per_second over a 1 s window (R5.1)
  - Queue forms when arrivals exceed per-tick capacity
  - Evacuated count equals initial population on a fully solvable scenario
  - Egressed agent is removed from AgentState (R5.2)
  - ticks_since_last_egress resets on egress, accumulates otherwise
  - Door clearing: agents queue only once past the door plane, within span
  - room_model_3 regression: no egress through a wall (issue #17)
  - Door gate: held at the line, capacity-metered release, placement past
    the line, evacuation at the end of the egress path
  - Float capacity token bucket (1.05 /s)
  - Multi-exit: agent is routed to the exit it cleared; goal entry updated
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.constants import AGENT_RADIUS, DT
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan

from .conftest import MakeState

RADIUS = AGENT_RADIUS


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

    def test_init_nonpositive_agent_radius_raises(
        self, simple_floor: FloorPlan
    ) -> None:
        """agent_radius ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            ExitModel(simple_floor, agent_radius=-0.5)


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

    def test_step_return_counts_removal_on_release(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Boundary door: step() reports agents removed directly on release.

        Regression: the return value (which feeds the tick event, throughput
        series and dashboard) omitted these, so headless runs showed zero.
        """
        state = make_state([(5.0, 0.3)] * 8)
        model = ExitModel(simple_floor, dt=DT)
        reported = 0
        for _ in range(round(2.0 / DT)):
            egressed = model.step(state)
            assert egressed == sum(model.last_egress_per_exit)
            reported += egressed
        assert reported == model.evacuated_count == 8


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
        """Stall counter reaches target for an immobile agent far from the door."""
        state = make_state([(5.0, 9.0)])  # far from the door, no velocity
        model = ExitModel(simple_floor, dt=DT)
        stall_window = 10
        for _ in range(stall_window):
            model.step(state)
        assert model.ticks_since_last_egress >= stall_window


# ---------------------------------------------------------------------------
# Door clearing
# ---------------------------------------------------------------------------


class TestDoorClearing:
    """Agents queue only once their body has cleared the door plane."""

    def test_agent_at_exit_lateral_edge_is_queued(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent pressed against a boundary door near its lateral edge egresses.

        Opening spans x ∈ [4, 6] on the y = 0 boundary (nothing behind it),
        so the clearing depth shrinks to the reachable ``-(R + margin)``.
        """
        state = make_state([(4.1, 0.3)])
        model = ExitModel(simple_floor, dt=DT)
        for _ in range(4):  # 4 ticks → 1 token → 1 egress
            model.step(state)
        assert model.evacuated_count == 1

    def test_agent_outside_lateral_span_not_queued(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent beside the opening (x = 0, span [4, 6]) is never queued."""
        state = make_state([(0.0, 0.3)])
        model = ExitModel(simple_floor, dt=DT)
        for _ in range(round(1.0 / DT)):
            model.step(state)
        assert model.evacuated_count == 0
        assert state.active_indices.size == 1

    def test_agent_inside_room_near_door_not_queued(
        self, simple_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent 0.5 m inside the plane — within the old 1 m radius — waits."""
        state = make_state([(5.0, 0.5)])
        model = ExitModel(simple_floor, dt=DT)
        model.step(state)
        assert len(model._queues[0]) == 0


class TestRoomModel3Regression:
    """Issue #17: lower-left room agents must not egress through wall_5."""

    def test_corner_agent_below_wall_is_never_queued(
        self, room_model_3: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent in the lower room's top-left corner (1.28, 7.0) stays put."""
        state = make_state([(1.28, 7.0)])
        model = ExitModel(room_model_3, dt=DT)
        for _ in range(round(5.0 / DT)):
            model.step(state)
        assert model.evacuated_count == 0
        assert state.goal[0] == -1
        np.testing.assert_allclose(state.pos[0], [1.28, 7.0])


class TestDoorGate:
    """Agents are held at the door line until the door releases them.

    room_model_3 west door: line x = 1.0, outward -x, span y 7.4-9.0,
    1 m carved gap behind it, capacity 1.0 /s.
    """

    def test_agent_crossing_line_is_held_and_queued(
        self, room_model_3: FloorPlan, make_state: MakeState
    ) -> None:
        """An agent pushing past the line is clamped flush and queued."""
        state = make_state([(0.9, 8.2)], vel=[(-1.0, 0.3)])
        model = ExitModel(room_model_3, dt=DT)
        model.step(state)
        np.testing.assert_allclose(state.pos[0], [1.0 + RADIUS, 8.2])
        np.testing.assert_allclose(state.vel[0], [0.0, 0.3])
        assert model.queue_lengths == (1, 0)
        assert state.goal[0] == 0

    def test_release_places_agent_past_line(
        self, room_model_3: FloorPlan, make_state: MakeState
    ) -> None:
        """At 1.0 /s the first token arrives on tick 20; agent jumps the line."""
        state = make_state([(1.0 + RADIUS, 8.2)])
        model = ExitModel(room_model_3, dt=DT)
        for _ in range(19):
            model.step(state)
        assert model.queue_lengths == (1, 0)
        model.step(state)
        np.testing.assert_allclose(state.pos[0], [1.0 - RADIUS, 8.2])
        assert model.last_release_per_exit == (1, 0)
        assert state.alive[0]
        assert model.evacuated_count == 0
        assert model.queue_lengths == (0, 0)

    def test_released_agent_free_to_walk_and_evacuated_at_gap_end(
        self, room_model_3: FloorPlan, make_state: MakeState
    ) -> None:
        """Released agents are not gated; they count at the gap's outer end."""
        state = make_state([(1.0 + RADIUS, 8.2)])
        model = ExitModel(room_model_3, dt=DT)
        for _ in range(20):
            model.step(state)
        state.pos[0] = (0.5, 8.2)  # walked on; not yet at the outer end
        model.step(state)
        assert state.alive[0]
        np.testing.assert_allclose(state.pos[0], [0.5, 8.2])
        state.pos[0] = (0.3, 8.2)  # reached the outer end of the gap
        assert model.step(state) == 1
        assert not state.alive[0]
        assert model.evacuated_count == 1
        assert model.last_egress_per_exit == (1, 0)

    def test_release_rate_bounded_by_capacity(
        self, room_model_3: FloorPlan, make_state: MakeState
    ) -> None:
        """Six agents at a 1.0 /s door: exactly 3 released in 3 s, FIFO."""
        ys = [7.7, 8.0, 8.3, 8.6, 8.9, 7.5]
        state = make_state([(1.0 + RADIUS, y) for y in ys])
        model = ExitModel(room_model_3, dt=DT)
        released = 0
        for _ in range(round(3.0 / DT)):
            model.step(state)
            released += model.last_release_per_exit[0]
        assert released == 3
        past = state.pos[:, 0] < 1.0
        np.testing.assert_array_equal(past, [True] * 3 + [False] * 3)
        assert model.queue_lengths == (3, 0)

    def test_agent_already_past_line_is_not_pulled_back(
        self, room_model_3: FloorPlan, make_state: MakeState
    ) -> None:
        """An agent spawned inside the gap counts as released, not gated."""
        state = make_state([(0.5, 8.2)])
        model = ExitModel(room_model_3, dt=DT)
        model.step(state)
        np.testing.assert_allclose(state.pos[0], [0.5, 8.2])
        assert model.queue_lengths == (0, 0)

    def test_placement_kept_inside_span(
        self, room_model_3: FloorPlan, make_state: MakeState
    ) -> None:
        """An agent queued at the span edge is placed a radius inside it."""
        state = make_state([(1.0 + RADIUS, 7.3)])
        model = ExitModel(room_model_3, dt=DT)
        for _ in range(20):
            model.step(state)
        np.testing.assert_allclose(
            state.pos[0], [1.0 - RADIUS, 7.4 + RADIUS]
        )


class TestFloatCapacity:
    """Fractional capacity accumulates in the token bucket."""

    def test_capacity_1_05_drains_21_in_20_seconds(
        self, make_state: MakeState
    ) -> None:
        """1.05 agents/s x 20 s = 21 egresses from a large queue."""
        floor = FloorPlan(
            10.0, 10.0, (), (),
            (Exit(5.0, 0.0, 2.0, ExitSide.SOUTH, 1.05, "s"),),
        )
        state = make_state([(5.0, 0.3)] * 40)
        model = ExitModel(floor, dt=DT)
        for _ in range(round(20.0 / DT)):
            model.step(state)
        assert model.evacuated_count == 21


# ---------------------------------------------------------------------------
# Multi-exit routing
# ---------------------------------------------------------------------------


class TestMultiExitRouting:
    """Agents are queued to the exit they cleared; goal entry is updated."""

    def test_agent_queued_to_nearest_exit(
        self, two_exit_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Agent at (1.0, 0.3) routes to west exit (index 0), not east."""
        state = make_state([(1.0, 0.3)])
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
