"""Tests for crowd_evac.domain.agent_state.

Covers:
  - spawn(): agent placement in the walkable region, seed reproducibility
    (R6.2), count=0 edge case, ValueError failure paths, and the overlap-
    free invariant (item 14 / step 1.19a): agents never overlap walls,
    obstacles, or each other at spawn time.
  - AgentState: active_indices, remove(), agent_view().
  - Agent: four-field read-access view (pos, vel, panic, goal).
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.agent_state import Agent, AgentState, spawn
from crowd_evac.domain.constants import AGENT_RADIUS
from crowd_evac.domain.floor_plan import (
    Exit,
    ExitSide,
    FloorPlan,
    Obstacle,
    Wall,
)


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
        """When count ≤ n_safe_cells, each agent occupies a distinct cell.

        Uses cell_size=2.0 so adjacent cells are 2 m apart, well above the
        2 * AGENT_RADIUS = 1.1 m exclusion distance.  This means the greedy
        placement never skips a cell for inter-agent reasons, and the
        without-replacement cell invariant holds unconditionally.
        """
        cell_size = 2.0
        state = spawn(
            simple_floor,
            count=5,
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


# ---------------------------------------------------------------------------
# Helpers for overlap tests
# ---------------------------------------------------------------------------


def _min_dist_to_rect(
    pos: np.ndarray, rx: float, ry: float, rw: float, rh: float
) -> np.ndarray:
    """Euclidean distance from each position to a rectangle (vectorised)."""
    px, py = pos[:, 0], pos[:, 1]
    dx = np.maximum(rx - px, np.maximum(px - (rx + rw), 0.0))
    dy = np.maximum(ry - py, np.maximum(py - (ry + rh), 0.0))
    return np.sqrt(dx**2 + dy**2)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# spawn() — overlap-free invariant (step 1.19a item 14)
# ---------------------------------------------------------------------------


@pytest.fixture
def floor_with_obstacle() -> FloorPlan:
    """10 × 10 room with a 2 × 2 centre obstacle and one south exit.

    Outer walls are 0.5 m thick on all four sides; obstacle sits at (4, 4).
    """
    return FloorPlan(
        width_m=10.0,
        height_m=10.0,
        walls=(
            Wall(x=0.0, y=0.0, width=10.0, height=0.5, label="south"),
            Wall(x=0.0, y=9.5, width=10.0, height=0.5, label="north"),
            Wall(x=0.0, y=0.0, width=0.5, height=10.0, label="west"),
            Wall(x=9.5, y=0.0, width=0.5, height=10.0, label="east"),
        ),
        obstacles=(
            Obstacle(x=4.0, y=4.0, width=2.0, height=2.0, label="pillar"),
        ),
        exits=(
            Exit(
                x=5.0,
                y=0.25,
                width_m=2.0,
                side=ExitSide.SOUTH,
                capacity_per_second=5,
                label="south_exit",
            ),
        ),
    )


class TestSpawnNoOverlap:
    """spawn() must produce an overlap-free initial placement (item 14)."""

    def test_no_agent_agent_overlap(self, simple_floor: FloorPlan) -> None:
        """All pairwise agent distances must be >= 2 * AGENT_RADIUS."""
        state = spawn(simple_floor, count=30, rng=np.random.default_rng(10))
        pos = state.pos  # (30, 2)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = (dx * dx + dy * dy) ** 0.5
                assert dist >= 2.0 * AGENT_RADIUS - 1e-9, (
                    f"Agents {i} and {j} overlap: distance {dist:.4f} m "
                    f"< {2 * AGENT_RADIUS:.4f} m"
                )

    def test_no_agent_wall_overlap(self, simple_floor: FloorPlan) -> None:
        """Agent centres must be >= AGENT_RADIUS from the south wall surface."""
        state = spawn(simple_floor, count=30, rng=np.random.default_rng(11))
        south_wall = simple_floor.walls[0]  # x=0, y=0, w=10, h=1
        dists = _min_dist_to_rect(
            state.pos,
            south_wall.x,
            south_wall.y,
            south_wall.width,
            south_wall.height,
        )
        assert (dists >= AGENT_RADIUS - 1e-9).all(), (
            f"Agents overlap south wall; min distance = {dists.min():.4f} m"
        )

    def test_no_agent_room_boundary_overlap(
        self, simple_floor: FloorPlan
    ) -> None:
        """Agent centres must be >= AGENT_RADIUS from every room boundary."""
        state = spawn(simple_floor, count=30, rng=np.random.default_rng(12))
        pos = state.pos
        margin = AGENT_RADIUS - 1e-9
        assert (pos[:, 0] >= margin).all(), "Agent too close to west boundary"
        assert (pos[:, 0] <= simple_floor.width_m - margin).all(), (
            "Agent too close to east boundary"
        )
        assert (pos[:, 1] >= margin).all(), "Agent too close to south boundary"
        assert (pos[:, 1] <= simple_floor.height_m - margin).all(), (
            "Agent too close to north boundary"
        )

    def test_no_agent_obstacle_overlap(
        self, floor_with_obstacle: FloorPlan
    ) -> None:
        """Agent centres must be >= AGENT_RADIUS from every obstacle surface."""
        state = spawn(
            floor_with_obstacle, count=20, rng=np.random.default_rng(13)
        )
        pillar = floor_with_obstacle.obstacles[0]  # x=4, y=4, w=2, h=2
        dists = _min_dist_to_rect(
            state.pos,
            pillar.x,
            pillar.y,
            pillar.width,
            pillar.height,
        )
        assert (dists >= AGENT_RADIUS - 1e-9).all(), (
            f"Agents overlap obstacle; min distance = {dists.min():.4f} m"
        )
