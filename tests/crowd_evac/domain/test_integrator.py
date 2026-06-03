"""Tests for crowd_evac.domain.integrator.step (FR-1 R1.3).

Covers:
  - step(): semi-implicit Euler updates position and velocity in place,
    acceleration clamping to MAX_ACCEL, speed clamping to the
    panic-modulated cap, dead-agent immutability, empty-state no-op,
    and shape-mismatch failure path.

Integration tests combine f_exit + step to verify an agent on an empty
floor moves toward and reaches the exit under the full physics pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.constants import (
    DT,
    MAX_ACCEL,
    MAX_SPEED,
    PANIC_SPEED_MULTIPLIER,
)
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.forces import f_exit
from crowd_evac.domain.integrator import step
from crowd_evac.pathfinding.flow_field import FlowField


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corridor_floor() -> FloorPlan:
    """4 m × 1 m corridor with a single exit on the east wall."""
    return FloorPlan(
        width_m=4.0,
        height_m=1.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=4.0,
                y=0.5,
                width_m=0.5,
                side=ExitSide.EAST,
                capacity_per_second=5,
                label="east",
            ),
        ),
    )


@pytest.fixture
def field(corridor_floor: FloorPlan) -> FlowField:
    """Flow field built from the corridor floor plan."""
    return FlowField.build(corridor_floor)


@pytest.fixture
def agent_at_rest() -> AgentState:
    """Single calm agent at (0.5, 0.5) with zero velocity."""
    return AgentState(
        pos=np.array([[0.5, 0.5]], dtype=np.float64),
        vel=np.zeros((1, 2), dtype=np.float64),
        panic=np.zeros(1, dtype=np.float64),
        goal=np.full(1, -1, dtype=np.intp),
        alive=np.ones(1, dtype=np.bool_),
    )


@pytest.fixture
def agent_moving() -> AgentState:
    """Single calm agent at (0.5, 0.5) with velocity (1.0, 0.0)."""
    return AgentState(
        pos=np.array([[0.5, 0.5]], dtype=np.float64),
        vel=np.array([[1.0, 0.0]], dtype=np.float64),
        panic=np.zeros(1, dtype=np.float64),
        goal=np.full(1, -1, dtype=np.intp),
        alive=np.ones(1, dtype=np.bool_),
    )


@pytest.fixture
def agent_dead() -> AgentState:
    """Single dead agent at (1.0, 1.0) with velocity (3.0, 4.0)."""
    return AgentState(
        pos=np.array([[1.0, 1.0]], dtype=np.float64),
        vel=np.array([[3.0, 4.0]], dtype=np.float64),
        panic=np.zeros(1, dtype=np.float64),
        goal=np.full(1, -1, dtype=np.intp),
        alive=np.zeros(1, dtype=np.bool_),
    )


@pytest.fixture
def empty_state() -> AgentState:
    """AgentState with zero agents."""
    return AgentState(
        pos=np.empty((0, 2), dtype=np.float64),
        vel=np.empty((0, 2), dtype=np.float64),
        panic=np.empty(0, dtype=np.float64),
        goal=np.empty(0, dtype=np.intp),
        alive=np.empty(0, dtype=np.bool_),
    )


# ---------------------------------------------------------------------------
# Integration: agent moves toward and reaches exit (happy path, FR-1 R1.2)
# ---------------------------------------------------------------------------


class TestIntegratorHappyPath:
    """Agent dynamics under f_exit + step on a simple corridor."""

    def test_single_agent_moves_toward_exit(
        self, agent_at_rest: AgentState, field: FlowField
    ) -> None:
        """Agent on left of corridor reaches exit area after 80 ticks (4 s)."""
        initial_x = float(agent_at_rest.pos[0, 0])
        for _ in range(80):
            step(agent_at_rest, f_exit(agent_at_rest, field))
        # Must have moved at least 2.5 m toward the east exit at x = 4.0.
        assert agent_at_rest.pos[0, 0] > initial_x + 2.5

    def test_x_position_increases_monotonically_early(
        self, agent_at_rest: AgentState, field: FlowField
    ) -> None:
        """x-position increases each tick during the initial approach phase."""
        prev_x = float(agent_at_rest.pos[0, 0])
        for _ in range(30):
            step(agent_at_rest, f_exit(agent_at_rest, field))
            cur_x = float(agent_at_rest.pos[0, 0])
            assert cur_x >= prev_x
            prev_x = cur_x

    def test_velocity_non_zero_after_first_step(
        self, agent_at_rest: AgentState, field: FlowField
    ) -> None:
        """Agent at rest must gain velocity on the first step."""
        step(agent_at_rest, f_exit(agent_at_rest, field))
        assert np.linalg.norm(agent_at_rest.vel[0]) > 0.0

    def test_position_change_equals_v_new_times_dt(
        self, agent_moving: AgentState, field: FlowField
    ) -> None:
        """pos_new − pos_old must equal v_new * DT (semi-implicit Euler invariant)."""
        pos_before = agent_moving.pos[0].copy()
        a = f_exit(agent_moving, field)
        step(agent_moving, a)
        v_new = agent_moving.vel[0]
        np.testing.assert_array_almost_equal(
            agent_moving.pos[0], pos_before + v_new * DT
        )


# ---------------------------------------------------------------------------
# Acceleration clamping
# ---------------------------------------------------------------------------


class TestAccelerationClamp:
    """Large accelerations must be clamped to MAX_ACCEL."""

    def test_huge_accel_limits_velocity_change_per_tick(
        self, agent_at_rest: AgentState
    ) -> None:
        """From rest, a 1000 m/s² force gives delta_v == MAX_ACCEL * DT."""
        step(agent_at_rest, np.array([[1000.0, 0.0]], dtype=np.float64))
        delta_v = float(np.linalg.norm(agent_at_rest.vel[0]))
        assert delta_v == pytest.approx(MAX_ACCEL * DT, rel=1e-6)

    def test_diagonal_huge_accel_clamped_and_direction_preserved(
        self, agent_at_rest: AgentState
    ) -> None:
        """Large diagonal force is clamped while preserving direction (x == y)."""
        step(agent_at_rest, np.array([[500.0, 500.0]], dtype=np.float64))
        speed = float(np.linalg.norm(agent_at_rest.vel[0]))
        assert speed == pytest.approx(MAX_ACCEL * DT, rel=1e-6)
        np.testing.assert_allclose(
            agent_at_rest.vel[0, 0], agent_at_rest.vel[0, 1], rtol=1e-6
        )

    def test_small_accel_not_clamped(self, agent_at_rest: AgentState) -> None:
        """Acceleration well below MAX_ACCEL is applied exactly."""
        step(agent_at_rest, np.array([[0.1, 0.0]], dtype=np.float64))
        assert agent_at_rest.vel[0, 0] == pytest.approx(0.1 * DT, rel=1e-6)

    def test_zero_accel_leaves_velocity_unchanged(
        self, agent_moving: AgentState
    ) -> None:
        """Zero acceleration must not change velocity magnitude."""
        vel_before = agent_moving.vel[0].copy()
        step(agent_moving, np.zeros((1, 2), dtype=np.float64))
        # Velocity is unchanged (no force; just pos is updated).
        np.testing.assert_array_almost_equal(agent_moving.vel[0], vel_before)


# ---------------------------------------------------------------------------
# Speed clamping
# ---------------------------------------------------------------------------


class TestSpeedClamp:
    """Agent speed must stay within the panic-modulated cap."""

    def test_speed_stays_below_max_speed_no_panic(
        self, agent_at_rest: AgentState, field: FlowField
    ) -> None:
        """Sustained f_exit with no panic keeps speed at or below MAX_SPEED."""
        for _ in range(100):
            step(agent_at_rest, f_exit(agent_at_rest, field))
            speed = float(np.linalg.norm(agent_at_rest.vel[0]))
            assert speed <= MAX_SPEED + 1e-9, (
                f"Speed {speed:.6f} exceeded MAX_SPEED {MAX_SPEED}"
            )

    def test_panic_speed_never_exceeds_documented_cap(
        self, field: FlowField
    ) -> None:
        """Full panic keeps speed at or below MAX_SPEED * PANIC_SPEED_MULTIPLIER."""
        state = AgentState(
            pos=np.array([[0.5, 0.5]], dtype=np.float64),
            vel=np.zeros((1, 2), dtype=np.float64),
            panic=np.ones(1, dtype=np.float64),
            goal=np.full(1, -1, dtype=np.intp),
            alive=np.ones(1, dtype=np.bool_),
        )
        cap = MAX_SPEED * PANIC_SPEED_MULTIPLIER
        for _ in range(100):
            step(state, f_exit(state, field))
            speed = float(np.linalg.norm(state.vel[0]))
            assert speed <= cap + 1e-9, (
                f"Speed {speed:.6f} exceeded panic cap {cap}"
            )

    def test_initial_overspeed_clamped_in_one_step(self) -> None:
        """Agent already exceeding the speed cap is clamped on the first step."""
        state = AgentState(
            pos=np.array([[0.5, 0.5]], dtype=np.float64),
            vel=np.array([[MAX_SPEED * 10, 0.0]], dtype=np.float64),
            panic=np.zeros(1, dtype=np.float64),
            goal=np.full(1, -1, dtype=np.intp),
            alive=np.ones(1, dtype=np.bool_),
        )
        step(state, np.zeros((1, 2), dtype=np.float64))
        speed = float(np.linalg.norm(state.vel[0]))
        assert speed <= MAX_SPEED + 1e-9

    def test_partial_panic_cap_is_interpolated(self) -> None:
        """panic=0.5 cap is between MAX_SPEED and the full-panic cap."""
        panic = 0.5
        expected_cap = MAX_SPEED * (1.0 + panic * (PANIC_SPEED_MULTIPLIER - 1.0))
        state = AgentState(
            pos=np.array([[0.5, 0.5]], dtype=np.float64),
            vel=np.array([[expected_cap * 2.0, 0.0]], dtype=np.float64),
            panic=np.array([panic], dtype=np.float64),
            goal=np.full(1, -1, dtype=np.intp),
            alive=np.ones(1, dtype=np.bool_),
        )
        step(state, np.zeros((1, 2), dtype=np.float64))
        speed = float(np.linalg.norm(state.vel[0]))
        assert speed <= expected_cap + 1e-9


# ---------------------------------------------------------------------------
# Dead-agent immutability
# ---------------------------------------------------------------------------


class TestDeadAgentImmutability:
    """Dead agents must not be moved by step()."""

    def test_dead_agent_position_unchanged(self, agent_dead: AgentState) -> None:
        """step() must not alter the position of a dead agent."""
        pos_before = agent_dead.pos[0].copy()
        step(agent_dead, np.array([[999.0, 999.0]], dtype=np.float64))
        np.testing.assert_array_equal(agent_dead.pos[0], pos_before)

    def test_dead_agent_velocity_unchanged(self, agent_dead: AgentState) -> None:
        """step() must not alter the velocity of a dead agent."""
        vel_before = agent_dead.vel[0].copy()
        step(agent_dead, np.array([[999.0, 999.0]], dtype=np.float64))
        np.testing.assert_array_equal(agent_dead.vel[0], vel_before)

    def test_mixed_alive_dead_only_moves_alive(self) -> None:
        """Only alive agents are updated; dead agents stay put."""
        state = AgentState(
            pos=np.array([[0.5, 0.5], [2.0, 0.5]], dtype=np.float64),
            vel=np.zeros((2, 2), dtype=np.float64),
            panic=np.zeros(2, dtype=np.float64),
            goal=np.full(2, -1, dtype=np.intp),
            alive=np.array([True, False], dtype=np.bool_),
        )
        dead_pos_before = state.pos[1].copy()
        step(state, np.ones((2, 2), dtype=np.float64))
        np.testing.assert_array_equal(state.pos[1], dead_pos_before)
        assert np.linalg.norm(state.pos[0] - np.array([0.5, 0.5])) > 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestIntegratorEdgeCases:
    """step() with boundary inputs."""

    def test_empty_state_no_crash(self, empty_state: AgentState) -> None:
        """Zero-count AgentState must complete without error."""
        step(empty_state, np.empty((0, 2), dtype=np.float64))

    def test_shape_mismatch_raises_value_error(
        self, agent_at_rest: AgentState
    ) -> None:
        """Acceleration with wrong N must raise ValueError."""
        with pytest.raises(ValueError, match="state.count"):
            step(agent_at_rest, np.zeros((5, 2), dtype=np.float64))

    def test_all_dead_state_leaves_arrays_unchanged(self) -> None:
        """When all agents are dead, pos and vel are not touched."""
        state = AgentState(
            pos=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            vel=np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
            panic=np.zeros(2, dtype=np.float64),
            goal=np.full(2, -1, dtype=np.intp),
            alive=np.zeros(2, dtype=np.bool_),
        )
        pos_before = state.pos.copy()
        vel_before = state.vel.copy()
        step(state, np.ones((2, 2), dtype=np.float64))
        np.testing.assert_array_equal(state.pos, pos_before)
        np.testing.assert_array_equal(state.vel, vel_before)
