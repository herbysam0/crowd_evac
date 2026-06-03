"""Tests for crowd_evac.domain.forces.f_exit (FR-1 R1.2).

Covers:
  - f_exit(): output shape, zero rows for dead agents, correct exit-seeking
    direction for active agents, panic-modulated desired speed, extensibility
    (additive zero term), and failure paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.constants import MAX_SPEED, PANIC_SPEED_MULTIPLIER
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.forces import f_exit
from crowd_evac.pathfinding.flow_field import FlowField


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corridor_floor() -> FloorPlan:
    """5 m × 1 m corridor with a single exit on the east wall.

    Flow field directions are approximately (+1, 0) throughout — simple
    geometry makes the expected direction easy to assert.
    """
    return FloorPlan(
        width_m=5.0,
        height_m=1.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=5.0,
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


def _make_state(
    pos: list[list[float]],
    vel: list[list[float]] | None = None,
    panic: list[float] | None = None,
    alive: list[bool] | None = None,
) -> AgentState:
    """Construct an AgentState from plain lists for test convenience."""
    n = len(pos)
    vel_ = vel if vel is not None else [[0.0, 0.0]] * n
    panic_ = panic if panic is not None else [0.0] * n
    alive_ = alive if alive is not None else [True] * n
    return AgentState(
        pos=np.array(pos, dtype=np.float64),
        vel=np.array(vel_, dtype=np.float64),
        panic=np.array(panic_, dtype=np.float64),
        goal=np.full(n, -1, dtype=np.intp),
        alive=np.array(alive_, dtype=np.bool_),
    )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestFExitShape:
    """f_exit must return an (N, 2) float64 array for any N."""

    def test_shape_single_agent(self, field: FlowField) -> None:
        """Single active agent produces shape (1, 2)."""
        state = _make_state([[0.5, 0.5]])
        result = f_exit(state, field)
        assert result.shape == (1, 2)

    def test_shape_multiple_agents(self, field: FlowField) -> None:
        """Five active agents produce shape (5, 2)."""
        state = _make_state([[float(i) * 0.5 + 0.1, 0.5] for i in range(5)])
        result = f_exit(state, field)
        assert result.shape == (5, 2)

    def test_shape_empty_state(self, field: FlowField) -> None:
        """Empty population (N=0) produces shape (0, 2)."""
        state = AgentState(
            pos=np.empty((0, 2), dtype=np.float64),
            vel=np.empty((0, 2), dtype=np.float64),
            panic=np.empty(0, dtype=np.float64),
            goal=np.empty(0, dtype=np.intp),
            alive=np.empty(0, dtype=np.bool_),
        )
        result = f_exit(state, field)
        assert result.shape == (0, 2)

    def test_dtype_is_float64(self, field: FlowField) -> None:
        """Output dtype must be float64."""
        state = _make_state([[0.5, 0.5]])
        result = f_exit(state, field)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Dead-agent rows are zero
# ---------------------------------------------------------------------------


class TestFExitDeadAgents:
    """Dead agents must receive a zero acceleration row."""

    def test_single_dead_agent_gets_zero(self, field: FlowField) -> None:
        """An agent with alive=False gets a (0, 0) row."""
        state = _make_state([[0.5, 0.5]], alive=[False])
        result = f_exit(state, field)
        np.testing.assert_array_equal(result[0], [0.0, 0.0])

    def test_mixed_alive_dead(self, field: FlowField) -> None:
        """Dead agents in a mixed population get zero; live agents do not."""
        state = _make_state(
            [[0.5, 0.5], [1.0, 0.5], [2.0, 0.5]],
            alive=[True, False, True],
        )
        result = f_exit(state, field)
        np.testing.assert_array_equal(result[1], [0.0, 0.0])
        # Live agents have non-zero exit-seeking force
        assert np.linalg.norm(result[0]) > 0.0
        assert np.linalg.norm(result[2]) > 0.0

    def test_all_dead_returns_all_zeros(self, field: FlowField) -> None:
        """All dead population produces an all-zero output."""
        state = _make_state(
            [[0.5, 0.5], [1.0, 0.5]], alive=[False, False]
        )
        result = f_exit(state, field)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Direction correctness
# ---------------------------------------------------------------------------


class TestFExitDirection:
    """Active agents on the left of the corridor must seek the east exit."""

    def test_x_component_positive_for_agent_on_left(
        self, field: FlowField
    ) -> None:
        """Agent at x=0.5 gets a positive x acceleration (toward east exit)."""
        state = _make_state([[0.5, 0.5]])
        a = f_exit(state, field)
        assert a[0, 0] > 0.0, "Expected positive x-component toward east exit"

    def test_force_reduces_when_velocity_matches_desired(
        self, field: FlowField
    ) -> None:
        """Force magnitude drops as agent velocity approaches desired velocity."""
        pos = [[0.5, 0.5]]
        # Agent at rest: large exit force
        state_rest = _make_state(pos, vel=[[0.0, 0.0]])
        a_rest = np.linalg.norm(f_exit(state_rest, field))

        # Agent already moving at MAX_SPEED rightward: near-zero force
        state_fast = _make_state(pos, vel=[[MAX_SPEED, 0.0]])
        a_fast = np.linalg.norm(f_exit(state_fast, field))

        assert a_rest > a_fast


# ---------------------------------------------------------------------------
# Panic modulation
# ---------------------------------------------------------------------------


class TestFExitPanicModulation:
    """Panic raises the desired speed and thereby the exit-seeking force."""

    def test_panic_increases_force_magnitude(self, field: FlowField) -> None:
        """Full panic produces a larger force than no panic (same pos, vel=0)."""
        pos = [[0.5, 0.5]]
        state_calm = _make_state(pos, panic=[0.0])
        state_panic = _make_state(pos, panic=[1.0])
        a_calm = np.linalg.norm(f_exit(state_calm, field))
        a_panic = np.linalg.norm(f_exit(state_panic, field))
        assert a_panic > a_calm

    def test_zero_panic_desired_speed_equals_max_speed(
        self, field: FlowField
    ) -> None:
        """With no panic the desired speed is exactly MAX_SPEED."""
        # Agent at rest on the corridor centreline; direction is (+1, 0).
        state = _make_state([[0.5, 0.5]], panic=[0.0])
        from crowd_evac.domain.constants import RELAXATION_TIME

        a = f_exit(state, field)
        # Desired velocity = MAX_SPEED * direction; a = (desired - 0) / tau.
        # Magnitude should be MAX_SPEED / tau (along x).
        expected = MAX_SPEED / RELAXATION_TIME
        assert np.linalg.norm(a[0]) == pytest.approx(expected, rel=0.05)

    def test_full_panic_desired_speed_equals_cap(
        self, field: FlowField
    ) -> None:
        """Full panic gives desired speed = MAX_SPEED * PANIC_SPEED_MULTIPLIER."""
        state = _make_state([[0.5, 0.5]], panic=[1.0])
        from crowd_evac.domain.constants import RELAXATION_TIME

        a = f_exit(state, field)
        cap = MAX_SPEED * PANIC_SPEED_MULTIPLIER
        expected = cap / RELAXATION_TIME
        assert np.linalg.norm(a[0]) == pytest.approx(expected, rel=0.05)


# ---------------------------------------------------------------------------
# Extensibility: adding a zero term does not change the result (R1.2 AC)
# ---------------------------------------------------------------------------


class TestFExitExtensibility:
    """Callers may add zero-value terms without altering the trajectory."""

    def test_adding_zero_array_leaves_result_unchanged(
        self, field: FlowField
    ) -> None:
        """f_exit + zeros(N,2) must equal f_exit."""
        state = _make_state([[0.5, 0.5], [1.5, 0.5]])
        a_ref = f_exit(state, field)
        a_with_zero = f_exit(state, field) + np.zeros((state.count, 2))
        np.testing.assert_array_equal(a_ref, a_with_zero)

    def test_adding_zero_produces_identical_trajectory(
        self, field: FlowField
    ) -> None:
        """Running two identical states — one with a zero term — stays equal."""
        from crowd_evac.domain.integrator import step

        def make() -> AgentState:
            return _make_state([[0.5, 0.5]])

        state_a = make()
        state_b = make()

        for _ in range(10):
            step(state_a, f_exit(state_a, field))
            step(state_b, f_exit(state_b, field) + np.zeros((1, 2)))

        np.testing.assert_array_almost_equal(state_a.pos, state_b.pos)
        np.testing.assert_array_almost_equal(state_a.vel, state_b.vel)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFExitFailurePaths:
    """f_exit must raise on invalid arguments."""

    def test_zero_relaxation_time_raises(self, field: FlowField) -> None:
        """relaxation_time=0.0 must raise ValueError."""
        state = _make_state([[0.5, 0.5]])
        with pytest.raises(ValueError, match="relaxation_time must be positive"):
            f_exit(state, field, relaxation_time=0.0)

    def test_negative_relaxation_time_raises(self, field: FlowField) -> None:
        """Negative relaxation_time must raise ValueError."""
        state = _make_state([[0.5, 0.5]])
        with pytest.raises(ValueError, match="relaxation_time must be positive"):
            f_exit(state, field, relaxation_time=-1.0)
