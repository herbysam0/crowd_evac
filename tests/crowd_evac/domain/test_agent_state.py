"""Tests for crowd_evac.domain.agent_state.

Covers:
  - spawn(): agent placement in the walkable region, seed reproducibility
    (R6.2), count=0 edge case, and ValueError failure paths.
  - AgentState: active_indices, remove(), agent_view().
  - Agent: four-field read-access view (pos, vel, panic, goal).
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.agent_state import Agent, AgentState, spawn
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan, Wall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_floor() -> FloorPlan:
    """10 m × 10 m room with a south wall band and one south exit.

    Wall covers y ∈ [0, 1); walkable region at cell_size=1.0 is rows 1-9.
    """
    return FloorPlan(
        width_m=10.0,
        height_m=10.0,
        walls=(Wall(x=0.0, y=0.0, width=10.0, height=1.0, label="south"),),
        obstacles=(),
        exits=(
            Exit(
                x=5.0,
                y=0.5,
                width_m=2.0,
                side=ExitSide.SOUTH,
                capacity_per_second=5,
                label="south_exit",
            ),
        ),
    )


@pytest.fixture
def rng_42() -> np.random.Generator:
    """Seeded generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def state_10(
    simple_floor: FloorPlan, rng_42: np.random.Generator
) -> AgentState:
    """10-agent state spawned on simple_floor with seed 42."""
    return spawn(simple_floor, count=10, rng=rng_42, cell_size=1.0)


# ---------------------------------------------------------------------------
# spawn() — happy path
# ---------------------------------------------------------------------------


class TestSpawnHappyPath:
    """spawn() with valid inputs."""

    def test_returns_agent_state(self, simple_floor: FloorPlan) -> None:
        """spawn() must return an AgentState."""
        state = spawn(simple_floor, count=5, rng=np.random.default_rng(0))
        assert isinstance(state, AgentState)

    def test_correct_count(self, simple_floor: FloorPlan) -> None:
        """state.count must equal the requested number of agents."""
        state = spawn(simple_floor, count=20, rng=np.random.default_rng(1))
        assert state.count == 20

    def test_all_agents_alive(self, simple_floor: FloorPlan) -> None:
        """Every freshly spawned agent must be alive."""
        state = spawn(simple_floor, count=15, rng=np.random.default_rng(2))
        assert state.alive.all()

    def test_agents_in_walkable_region(
        self, simple_floor: FloorPlan
    ) -> None:
        """All agent positions must fall inside a walkable grid cell."""
        cell_size = 1.0
        state = spawn(
            simple_floor,
            count=50,
            rng=np.random.default_rng(3),
            cell_size=cell_size,
        )
        mask = simple_floor.walkable_mask(cell_size)
        row_idx = np.clip(
            (state.pos[:, 1] / cell_size).astype(int),
            0,
            mask.shape[0] - 1,
        )
        col_idx = np.clip(
            (state.pos[:, 0] / cell_size).astype(int),
            0,
            mask.shape[1] - 1,
        )
        assert mask[row_idx, col_idx].all(), (
            "Some spawned agents lie in non-walkable cells"
        )

    def test_positions_within_room_bounds(
        self, simple_floor: FloorPlan
    ) -> None:
        """Agent x/y must lie within the room bounding box."""
        state = spawn(simple_floor, count=40, rng=np.random.default_rng(4))
        assert (state.pos[:, 0] >= 0).all()
        assert (state.pos[:, 0] < simple_floor.width_m).all()
        assert (state.pos[:, 1] >= 0).all()
        assert (state.pos[:, 1] < simple_floor.height_m).all()

    def test_initial_velocity_zero(self, simple_floor: FloorPlan) -> None:
        """Initial velocity must be (0, 0) for every agent."""
        state = spawn(simple_floor, count=10, rng=np.random.default_rng(5))
        assert (state.vel == 0.0).all()

    def test_initial_panic_zero(self, simple_floor: FloorPlan) -> None:
        """Initial panic must be 0.0 for every agent."""
        state = spawn(simple_floor, count=10, rng=np.random.default_rng(6))
        assert (state.panic == 0.0).all()

    def test_initial_goal_unassigned(self, simple_floor: FloorPlan) -> None:
        """Goal must be -1 (unassigned) for every freshly spawned agent."""
        state = spawn(simple_floor, count=10, rng=np.random.default_rng(7))
        assert (state.goal == -1).all()

    def test_no_two_agents_same_cell_when_count_le_cells(
        self, simple_floor: FloorPlan
    ) -> None:
        """When count ≤ n_cells, each agent occupies a distinct cell."""
        cell_size = 1.0
        state = spawn(
            simple_floor,
            count=20,
            rng=np.random.default_rng(8),
            cell_size=cell_size,
        )
        row_idx = (state.pos[:, 1] / cell_size).astype(int)
        col_idx = (state.pos[:, 0] / cell_size).astype(int)
        cells = list(zip(row_idx.tolist(), col_idx.tolist()))
        assert len(cells) == len(set(cells)), "Duplicate cells found"


# ---------------------------------------------------------------------------
# spawn() — edge case: count = 0
# ---------------------------------------------------------------------------


class TestSpawnZeroCount:
    """spawn() with count=0 must return a valid but empty AgentState."""

    def test_zero_count_returns_empty_state(
        self, simple_floor: FloorPlan
    ) -> None:
        """count=0 must produce arrays with shape (0, …)."""
        state = spawn(simple_floor, count=0, rng=np.random.default_rng(0))
        assert state.count == 0
        assert state.pos.shape == (0, 2)
        assert state.vel.shape == (0, 2)
        assert state.panic.shape == (0,)
        assert state.goal.shape == (0,)
        assert state.alive.shape == (0,)

    def test_zero_count_no_active_indices(
        self, simple_floor: FloorPlan
    ) -> None:
        """Empty state must report no active indices."""
        state = spawn(simple_floor, count=0, rng=np.random.default_rng(0))
        assert state.active_indices.size == 0


# ---------------------------------------------------------------------------
# spawn() — failure paths
# ---------------------------------------------------------------------------


class TestSpawnFailurePaths:
    """spawn() must raise on invalid arguments."""

    def test_negative_count_raises_value_error(
        self, simple_floor: FloorPlan
    ) -> None:
        """Negative count must raise ValueError."""
        with pytest.raises(ValueError, match="count must be"):
            spawn(simple_floor, count=-1, rng=np.random.default_rng(0))

    def test_zero_cell_size_raises_value_error(
        self, simple_floor: FloorPlan
    ) -> None:
        """cell_size=0 must raise ValueError."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            spawn(
                simple_floor,
                count=1,
                rng=np.random.default_rng(0),
                cell_size=0.0,
            )

    def test_negative_cell_size_raises_value_error(
        self, simple_floor: FloorPlan
    ) -> None:
        """Negative cell_size must raise ValueError."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            spawn(
                simple_floor,
                count=1,
                rng=np.random.default_rng(0),
                cell_size=-0.5,
            )


# ---------------------------------------------------------------------------
# spawn() — seed reproducibility (R6.2)
# ---------------------------------------------------------------------------


class TestSpawnSeedReproducibility:
    """Identical seeds must produce identical spawn layouts."""

    def test_same_seed_same_positions(self, simple_floor: FloorPlan) -> None:
        """Two calls with the same seed must yield the same positions."""
        s1 = spawn(simple_floor, count=50, rng=np.random.default_rng(99))
        s2 = spawn(simple_floor, count=50, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(s1.pos, s2.pos)

    def test_different_seed_different_positions(
        self, simple_floor: FloorPlan
    ) -> None:
        """Distinct seeds must produce different positions (prob. 1)."""
        s1 = spawn(simple_floor, count=50, rng=np.random.default_rng(1))
        s2 = spawn(simple_floor, count=50, rng=np.random.default_rng(2))
        assert not np.array_equal(s1.pos, s2.pos)


# ---------------------------------------------------------------------------
# AgentState.active_indices + remove()
# ---------------------------------------------------------------------------


class TestAgentStateActiveAndRemove:
    """active_indices and remove() mutation."""

    def test_all_alive_gives_full_index_range(
        self, state_10: AgentState
    ) -> None:
        """active_indices must be [0..N-1] when all agents are alive."""
        expected = np.arange(10, dtype=np.intp)
        np.testing.assert_array_equal(state_10.active_indices, expected)

    def test_remove_single_agent(self, state_10: AgentState) -> None:
        """Removing agent 3 must exclude it from active_indices."""
        state_10.remove(3)
        assert 3 not in state_10.active_indices
        assert len(state_10.active_indices) == 9

    def test_remove_multiple_agents(self, state_10: AgentState) -> None:
        """Removing a list of indices must exclude all from active."""
        state_10.remove([1, 5, 7])
        for i in (1, 5, 7):
            assert i not in state_10.active_indices
        assert len(state_10.active_indices) == 7

    def test_remove_idempotent(self, state_10: AgentState) -> None:
        """Removing an already-dead agent must not change the active count."""
        state_10.remove(4)
        count_after = len(state_10.active_indices)
        state_10.remove(4)
        assert len(state_10.active_indices) == count_after

    def test_remove_all_yields_empty_active(
        self, state_10: AgentState
    ) -> None:
        """Removing every agent must leave active_indices empty."""
        state_10.remove(list(range(10)))
        assert state_10.active_indices.size == 0

    def test_remove_out_of_bounds_raises(
        self, state_10: AgentState
    ) -> None:
        """Out-of-range index must propagate an IndexError from NumPy."""
        with pytest.raises(IndexError):
            state_10.remove(999)


# ---------------------------------------------------------------------------
# AgentState.agent_view() and Agent properties
# ---------------------------------------------------------------------------


class TestAgentView:
    """agent_view() and the four Agent read properties."""

    def test_view_exposes_four_fields(self, state_10: AgentState) -> None:
        """Agent view must expose pos, vel, goal, and panic."""
        view = state_10.agent_view(0)
        assert hasattr(view, "pos")
        assert hasattr(view, "vel")
        assert hasattr(view, "goal")
        assert hasattr(view, "panic")

    def test_pos_shape(self, state_10: AgentState) -> None:
        """Agent.pos must be a 1-D array of length 2."""
        assert state_10.agent_view(0).pos.shape == (2,)

    def test_vel_shape(self, state_10: AgentState) -> None:
        """Agent.vel must be a 1-D array of length 2."""
        assert state_10.agent_view(0).vel.shape == (2,)

    def test_panic_is_float(self, state_10: AgentState) -> None:
        """Agent.panic must be a Python float."""
        assert isinstance(state_10.agent_view(0).panic, float)

    def test_goal_is_int(self, state_10: AgentState) -> None:
        """Agent.goal must be a Python int."""
        assert isinstance(state_10.agent_view(0).goal, int)

    def test_initial_goal_sentinel(self, state_10: AgentState) -> None:
        """Agent.goal must be -1 at spawn (unassigned)."""
        assert state_10.agent_view(0).goal == -1

    def test_view_reflects_state_mutation(
        self, state_10: AgentState
    ) -> None:
        """Agent.panic must reflect an in-place update to the parent array."""
        view = state_10.agent_view(2)
        state_10.panic[2] = 0.75
        assert view.panic == pytest.approx(0.75)

    def test_view_out_of_bounds_raises(self, state_10: AgentState) -> None:
        """agent_view() must raise IndexError for an out-of-range index."""
        with pytest.raises(IndexError):
            state_10.agent_view(100)

    def test_view_negative_index_raises(self, state_10: AgentState) -> None:
        """agent_view() must raise IndexError for a negative index."""
        with pytest.raises(IndexError):
            state_10.agent_view(-1)
