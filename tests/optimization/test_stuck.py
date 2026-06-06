"""Tests for crowd_evac.optimization.stuck (Phase 2 Step 2.5).

Covers the three plan success criteria and the public/private surface:
  - a hand-built deadlock fixture reports stuck > 0;
  - a legitimately queued agent at a saturated exit reports stuck == 0;
  - a freely moving agent reports 0;
plus edge cases (empty history, zero agents, window longer than the run,
agent sitting on an exit) and failure paths (out-of-range tuning args), and
the private helpers `_blocking_range`, `_window_samples`, `_sustained`,
`_cell_cost`, `_blocked_by_wall`.

FlowFields are built directly from a uniform walkable grid with an exit column,
so the descent direction is the known unit vector toward -x; RunResults are
hand-built so each deadlock condition is driven in isolation.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from crowd_evac.optimization import stuck as S
from crowd_evac.optimization.harness import RunResult
from crowd_evac.optimization.stuck import (
    has_stuck,
    stuck_agents,
    stuck_count,
)
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

_CELL = 1.0
_GRID = 10


def _open_field() -> FlowField:
    """A 10x10 unit-cell field whose only exits are column 0 (descent = -x).

    Every interior cell routes one step west toward the exit column, so
    ``sample`` returns ``(-1, 0)`` and the integration cost equals the column
    index — convenient for driving the deadlock conditions explicitly.
    """
    walkable = np.ones((_GRID, _GRID), dtype=np.bool_)
    exits = [(r, 0) for r in range(_GRID)]
    return FlowField(_CELL, walkable, exits)


@pytest.fixture
def open_field() -> FlowField:
    """Value fixture wrapping :func:`_open_field`."""
    return _open_field()


@pytest.fixture
def make_run() -> Callable[..., RunResult]:
    """Return a builder for synthetic RunResults driving the stuck detector.

    The builder accepts ``positions`` (T, N, 2), ``velocities`` (T, N, 2),
    optional ``alive`` (T, N, default all True), and ``sample_ticks`` (len T).
    Unused RunResult fields (density/throughput series) are left empty.
    """

    def _build(
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        sample_ticks: Sequence[int],
        alive: np.ndarray | None = None,
    ) -> RunResult:
        pos = np.asarray(positions, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)
        n = pos.shape[1] if pos.ndim == 3 else 0
        if alive is None:
            alive_arr = np.ones(pos.shape[:2], dtype=np.bool_)
        else:
            alive_arr = np.asarray(alive, dtype=np.bool_)
        panic = np.zeros(pos.shape[:2], dtype=np.float64)
        return RunResult(
            evac_time=1.0,
            evacuated_fraction=0.0,
            is_terminal=True,
            total_ticks=int(sample_ticks[-1]) if len(sample_ticks) else 0,
            initial_count=n,
            throughput_series=(),
            density_series=(),
            sample_ticks=tuple(sample_ticks),
            positions_history=pos,
            velocities_history=vel,
            panics_history=panic,
            alive_history=alive_arr,
        )

    return _build


def _stationary(
    points: list[tuple[float, float]], n_samples: int = 6
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build (positions, velocities, sample_ticks) for fixed, motionless agents.

    All agents hold ``points`` across ``n_samples`` samples at tick spacing 10
    (sample_dt = 0.5 s), with zero velocity.
    """
    n = len(points)
    pos = np.tile(
        np.asarray(points, dtype=np.float64)[np.newaxis, :, :],
        (n_samples, 1, 1),
    )
    vel = np.zeros((n_samples, n, 2), dtype=np.float64)
    ticks = [10 * (k + 1) for k in range(n_samples)]
    return pos, vel, ticks


# ---------------------------------------------------------------------------
# Plan success criteria
# ---------------------------------------------------------------------------


class TestStuckCriteria:
    """The three Step-2.5 success criteria for the public detector."""

    def test_deadlocked_agent_reported(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """A lone stalled agent with a clear route is flagged (stuck > 0)."""
        pos, vel, ticks = _stationary([(5.5, 5.5)])
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        assert stuck_count(run, open_field) == 1
        assert has_stuck(run, open_field) is True

    def test_queued_agent_not_reported(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """A queue feeding a saturated exit yields no false positives (== 0).

        Agents are spaced 1 m apart (< the blocking range) along the descent
        axis: the front agent sits at the exit; every follower has a neighbour
        directly ahead, so none is deadlocked.
        """
        pos, vel, ticks = _stationary(
            [(0.5, 5.5), (1.5, 5.5), (2.5, 5.5), (3.5, 5.5)]
        )
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        assert stuck_count(run, open_field) == 0

    def test_free_moving_agent_not_reported(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """An agent moving above the speed floor is never stuck (== 0)."""
        pos, _, ticks = _stationary([(5.5, 5.5)])
        vel = np.zeros_like(pos)
        vel[:, 0, 0] = -1.0  # moving west toward the exit at ~1 m/s
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        assert stuck_count(run, open_field) == 0


# ---------------------------------------------------------------------------
# Conditions in isolation
# ---------------------------------------------------------------------------


class TestDeadlockConditions:
    """Each instantaneous deadlock condition gates the result correctly."""

    def test_agent_on_exit_has_no_route(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """A stalled agent sitting on an exit cell is not stuck (zero dir)."""
        pos, vel, ticks = _stationary([(0.5, 5.5)])  # column 0 == exit
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        assert stuck_count(run, open_field) == 0

    def test_agent_near_exit_excluded(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """A lone stalled agent within the capture radius is queue, not stuck.

        At (1.2, 5.5) the path cost (~1.0 m) sits at the exit-capture radius;
        a lone agent there is treated as awaiting egress, not deadlocked.
        """
        pos, vel, ticks = _stationary([(1.2, 5.5)])
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        assert stuck_count(
            run, open_field, exit_capture_radius=2.0
        ) == 0

    def test_dead_agent_not_reported(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """A non-alive (egressed) agent is never flagged."""
        pos, vel, ticks = _stationary([(5.5, 5.5)])
        alive = np.zeros((len(ticks), 1), dtype=np.bool_)
        run = make_run(
            positions=pos, velocities=vel, sample_ticks=ticks, alive=alive
        )
        assert stuck_count(run, open_field) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty/degenerate inputs return an empty result without raising."""

    def test_zero_agents(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """A run with no agents returns an empty mask / zero count."""
        run = make_run(
            positions=np.zeros((3, 0, 2)),
            velocities=np.zeros((3, 0, 2)),
            sample_ticks=[10, 20, 30],
        )
        assert stuck_agents(run, open_field).shape == (0,)
        assert stuck_count(run, open_field) == 0

    def test_empty_history(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """No sampled ticks yields a zero mask sized to the agent count."""
        run = make_run(
            positions=np.zeros((0, 2, 2)),
            velocities=np.zeros((0, 2, 2)),
            sample_ticks=[],
        )
        # No sampled ticks -> early return of a zero mask regardless of count.
        assert stuck_count(run, open_field) == 0

    def test_window_longer_than_run(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """When the stall window exceeds the run length nothing is flagged."""
        pos, vel, ticks = _stationary([(5.5, 5.5)], n_samples=3)
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        # 3 samples at 0.5 s spacing = 1.5 s < a 100 s required stall.
        assert stuck_count(run, open_field, min_stall_s=100.0) == 0


# ---------------------------------------------------------------------------
# Consistency between the three public entry points
# ---------------------------------------------------------------------------


class TestPublicConsistency:
    """stuck_count, has_stuck, and stuck_agents agree."""

    def test_count_equals_mask_sum(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """stuck_count equals the number of True entries in stuck_agents."""
        pos, vel, ticks = _stationary([(5.5, 5.5), (6.5, 2.5)])
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        mask = stuck_agents(run, open_field)
        assert stuck_count(run, open_field) == int(mask.sum())
        assert has_stuck(run, open_field) == bool(mask.any())


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestValidation:
    """Out-of-range tuning arguments raise ValueError."""

    def test_non_positive_speed_eps_raises(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """speed_eps <= 0 raises."""
        pos, vel, ticks = _stationary([(5.5, 5.5)])
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        with pytest.raises(ValueError, match="speed_eps must be > 0"):
            stuck_count(run, open_field, speed_eps=0.0)

    def test_negative_exit_capture_radius_raises(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """exit_capture_radius < 0 raises."""
        pos, vel, ticks = _stationary([(5.5, 5.5)])
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        with pytest.raises(ValueError, match="exit_capture_radius must be >= 0"):
            stuck_count(run, open_field, exit_capture_radius=-1.0)

    def test_non_positive_min_stall_raises(
        self, open_field: FlowField, make_run: Callable[..., RunResult]
    ) -> None:
        """min_stall_s <= 0 raises."""
        pos, vel, ticks = _stationary([(5.5, 5.5)])
        run = make_run(positions=pos, velocities=vel, sample_ticks=ticks)
        with pytest.raises(ValueError, match="min_stall_s must be > 0"):
            stuck_count(run, open_field, min_stall_s=0.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestBlockingRange:
    """_blocking_range adds two agent radii to the repulsion radius."""

    def test_body_to_body(self) -> None:
        """0.5 m repulsion → 2*0.55 + 0.5 = 1.6 m centre-distance threshold."""
        assert S._blocking_range(0.5) == pytest.approx(1.6)


class TestWindowSamples:
    """_window_samples maps a stall duration onto the sampling rate."""

    def test_regular_spacing(self) -> None:
        """Ticks spaced 10 (0.5 s) need 4 samples for a 2 s stall."""
        assert S._window_samples((10, 20, 30), 2.0) == 4

    def test_single_sample_spacing_one(self) -> None:
        """A single tick falls back to unit spacing (DT)."""
        assert S._window_samples((5,), 0.1) == 2


class TestSustained:
    """_sustained flags only runs of `window` consecutive True samples."""

    def test_full_run_flagged_gap_not(self) -> None:
        """An unbroken column is flagged; one broken by a gap is not."""
        matrix = np.array(
            [[True, True],
             [True, True],
             [True, False],
             [True, True],
             [True, True]],
            dtype=np.bool_,
        )
        result = S._sustained(matrix, window=3)
        assert result[0]  # 5 consecutive
        assert not result[1]  # longest run is 2

    def test_window_longer_than_history(self) -> None:
        """A window longer than the matrix yields an all-False mask."""
        matrix = np.ones((2, 3), dtype=np.bool_)
        assert not S._sustained(matrix, window=5).any()


class TestCellCostAndWall:
    """_cell_cost and _blocked_by_wall read the integration field correctly."""

    def test_cost_inf_out_of_bounds(self, open_field: FlowField) -> None:
        """A point outside the grid maps to infinite cost."""
        pts = np.array([[-1.0, -1.0], [5.5, 5.5]])
        cost = S._cell_cost(open_field, pts)
        assert not np.isfinite(cost[0])
        assert np.isfinite(cost[1])

    def test_wall_ahead_detected(self) -> None:
        """A blocked cell one probe step ahead is reported as a wall."""
        walkable = np.ones((5, 5), dtype=np.bool_)
        walkable[2, 3] = False  # block the cell east of (2.5, 2.5)
        field = FlowField(1.0, walkable, [(r, 0) for r in range(5)])
        pos = np.array([[2.5, 2.5]])
        east = np.array([[1.0, 0.0]])  # probe toward the blocked cell
        west = np.array([[-1.0, 0.0]])  # probe toward open space
        assert S._blocked_by_wall(field, pos, east, 0.55)[0]
        assert not S._blocked_by_wall(field, pos, west, 0.55)[0]
