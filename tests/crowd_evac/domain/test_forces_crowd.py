"""Tests for crowd-dynamics forces: f_crowd, f_density, f_herd (FR-2).

Covers:
  - f_crowd (R2.1): repulsion direction/shape, dead-agent zero rows, agents
    beyond range, overlap resolution and collision-course deflection.
  - f_density (R2.2): zero below threshold, deceleration opposing velocity
    growing with crowding, and reduced effective forward speed in a dense
    group versus a sparse one.
  - f_herd (R2.5): zero at no panic, alignment toward the local mean velocity
    increasing with panic, isolated/dead agents zero.
  - Failure paths for each term.

Test inputs come from factory fixtures (``make_state`` from the domain
``conftest``; ``cluster`` defined here).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.constants import AGENT_RADIUS, REPULSION_RADIUS
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.forces import f_crowd, f_density, f_exit, f_herd
from crowd_evac.domain.integrator import step
from crowd_evac.domain.spatial_hash import SpatialHash
from crowd_evac.pathfinding.flow_field import FlowField

from .conftest import MakeState

# 0.4 m: centre distance below which two agents are deemed to overlap.
_CONTACT = 2.0 * AGENT_RADIUS

Cluster = Callable[[int, tuple[float, float], float], list[list[float]]]


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster() -> Cluster:
    """Return a factory producing a square-ish grid of positions."""

    def _make(
        n: int, centre: tuple[float, float], spacing: float
    ) -> list[list[float]]:
        """Lay n points on a grid of side ceil(sqrt(n)) from ``centre``."""
        side = int(np.ceil(np.sqrt(n)))
        pts: list[list[float]] = []
        for k in range(n):
            r, c = divmod(k, side)
            pts.append([centre[0] + c * spacing, centre[1] + r * spacing])
        return pts

    return _make


@pytest.fixture
def corridor_field() -> FlowField:
    """Flow field for a long corridor with a single east exit."""
    floor = FloorPlan(
        width_m=12.0,
        height_m=4.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=12.0,
                y=2.0,
                width_m=1.0,
                side=ExitSide.EAST,
                capacity_per_second=5,
                label="east",
            ),
        ),
    )
    return FlowField.build(floor)


# ===========================================================================
# f_crowd — short-range repulsion (R2.1)
# ===========================================================================


class TestFCrowdShape:
    """f_crowd returns an (N, 2) float64 array with dead rows zeroed."""

    def test_shape_and_dtype(self, make_state: MakeState) -> None:
        """Output is (N, 2) float64."""
        out = f_crowd(make_state([[1.0, 1.0], [1.2, 1.0]]))
        assert out.shape == (2, 2)
        assert out.dtype == np.float64

    def test_empty_population(self, make_state: MakeState) -> None:
        """Zero agents -> shape (0, 2)."""
        assert f_crowd(make_state([])).shape == (0, 2)

    def test_dead_agent_row_zero(self, make_state: MakeState) -> None:
        """A dead agent neither exerts nor receives repulsion."""
        state = make_state([[1.0, 1.0], [1.2, 1.0]], alive=[True, False])
        out = f_crowd(state)
        np.testing.assert_array_equal(out[1], [0.0, 0.0])
        # The live agent has no live neighbour, so it too is zero.
        np.testing.assert_array_equal(out[0], [0.0, 0.0])


class TestFCrowdDirection:
    """Repulsion pushes neighbours apart along their connecting line."""

    def test_two_agents_repel_apart(self, make_state: MakeState) -> None:
        """Left/right neighbours get opposing forces along x."""
        out = f_crowd(make_state([[1.0, 1.0], [1.3, 1.0]]))
        assert out[0, 0] < 0.0, "left agent pushed -x"
        assert out[1, 0] > 0.0, "right agent pushed +x"
        np.testing.assert_allclose(out[0], -out[1], atol=1e-9)

    def test_agents_beyond_radius_no_force(
        self, make_state: MakeState
    ) -> None:
        """Agents farther apart than the radius do not interact."""
        state = make_state([[0.0, 0.0], [REPULSION_RADIUS + 0.2, 0.0]])
        np.testing.assert_array_equal(f_crowd(state), np.zeros((2, 2)))

    def test_closer_agents_repel_harder(
        self, make_state: MakeState
    ) -> None:
        """Repulsion magnitude grows as separation shrinks."""
        near = f_crowd(make_state([[1.0, 1.0], [1.1, 1.0]]))
        far = f_crowd(make_state([[1.0, 1.0], [1.4, 1.0]]))
        assert np.linalg.norm(near[0]) > np.linalg.norm(far[0])


class TestFCrowdNoOverlap:
    """Repulsion resolves overlap and prevents collisions (R2.1)."""

    def test_overlapping_agents_separate_monotonically(
        self, make_state: MakeState
    ) -> None:
        """Two overlapping, at-rest agents push apart past contact distance."""
        state = make_state([[1.0, 1.0], [1.3, 1.0]])  # 0.3 m < contact 0.4
        distances = []
        for _ in range(120):
            step(state, f_crowd(state))
            distances.append(
                float(np.linalg.norm(state.pos[0] - state.pos[1]))
            )
        diffs = np.diff(distances)
        assert np.all(diffs >= -1e-9), "separation must never decrease"
        assert distances[0] > 0.3, "agents start separating immediately"
        assert distances[-1] >= _CONTACT, "overlap resolved past contact"

    def test_collision_course_deflects(self, make_state: MakeState) -> None:
        """Offset head-on agents deflect; repulsion keeps them farther apart."""

        def run(strength: float) -> tuple[float, AgentState]:
            state = make_state(
                [[0.0, 0.15], [3.0, -0.15]],
                vel=[[0.3, 0.0], [-0.3, 0.0]],
            )
            min_dist = float("inf")
            for _ in range(220):
                step(state, f_crowd(state, strength=strength))
                min_dist = min(
                    min_dist,
                    float(np.linalg.norm(state.pos[0] - state.pos[1])),
                )
            return min_dist, state

        min_repel, state_repel = run(strength=1.0)
        min_inert, _ = run(strength=0.0)

        assert min_repel > min_inert, "repulsion increases closest approach"
        assert min_repel >= 0.3, "agents do not overlap on a near miss"
        assert state_repel.pos[0, 0] > state_repel.pos[1, 0], "agents passed"


class TestFCrowdFailures:
    """f_crowd validates its tuning arguments."""

    def test_non_positive_radius_raises(self, make_state: MakeState) -> None:
        """radius <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            f_crowd(make_state([[1.0, 1.0]]), radius=0.0)

    def test_negative_strength_raises(self, make_state: MakeState) -> None:
        """Negative strength must raise ValueError."""
        with pytest.raises(ValueError, match="strength must be non-negative"):
            f_crowd(make_state([[1.0, 1.0]]), strength=-1.0)

    def test_non_positive_min_distance_raises(
        self, make_state: MakeState
    ) -> None:
        """min_distance <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="min_distance must be positive"):
            f_crowd(make_state([[1.0, 1.0]]), min_distance=0.0)


# ===========================================================================
# f_density — density pressure reducing effective speed (R2.2)
# ===========================================================================


class TestFDensityShape:
    """f_density returns an (N, 2) float64 array."""

    def test_shape_and_dtype(self, make_state: MakeState) -> None:
        """Output is (N, 2) float64."""
        out = f_density(make_state([[1.0, 1.0], [1.2, 1.0]]))
        assert out.shape == (2, 2)
        assert out.dtype == np.float64

    def test_empty_population(self, make_state: MakeState) -> None:
        """Zero agents -> shape (0, 2)."""
        assert f_density(make_state([])).shape == (0, 2)


class TestFDensityThreshold:
    """No pressure below the density threshold; drag opposes velocity above."""

    def test_sparse_crowd_no_force(self, make_state: MakeState) -> None:
        """A lone moving agent feels no density pressure."""
        state = make_state([[5.0, 5.0]], vel=[[2.0, 0.0]])
        np.testing.assert_array_equal(f_density(state), np.zeros((1, 2)))

    def test_dense_crowd_decelerates_along_velocity(
        self, make_state: MakeState, cluster: Cluster
    ) -> None:
        """In a dense pack the central agent's drag opposes its motion."""
        pts = cluster(60, (5.0, 5.0), 0.18)
        state = make_state(pts, vel=[[1.5, 0.0]] * len(pts))
        out = f_density(state)
        centre_idx = len(pts) // 2
        assert out[centre_idx, 0] < 0.0, "drag opposes +x velocity"

    def test_drag_grows_with_crowding(
        self, make_state: MakeState, cluster: Cluster
    ) -> None:
        """Denser surroundings yield a larger opposing force (directional)."""

        def central_drag(n: int) -> float:
            pts = cluster(n, (5.0, 5.0), 0.18)
            state = make_state(pts, vel=[[1.5, 0.0]] * len(pts))
            return float(np.linalg.norm(f_density(state)[len(pts) // 2]))

        assert central_drag(64) > central_drag(16)


class TestFDensityEffectiveSpeed:
    """A dense group makes slower forward progress than a sparse one (R2.2)."""

    def test_dense_group_advances_slower(
        self,
        make_state: MakeState,
        cluster: Cluster,
        corridor_field: FlowField,
    ) -> None:
        """Same agents, packed vs spread: the dense pack lags downfield."""

        def mean_x_after(positions: list[list[float]]) -> float:
            state = make_state(positions)
            for _ in range(120):
                a = (
                    f_exit(state, corridor_field)
                    + f_crowd(state)
                    + f_density(state)
                )
                step(state, a)
            return float(state.pos[:, 0].mean())

        n = 49
        dense = cluster(n, (1.0, 1.6), 0.22)
        sparse = cluster(n, (1.0, 1.0), 0.5)
        assert mean_x_after(dense) < mean_x_after(sparse)


class TestFDensityFailures:
    """f_density validates its tuning arguments."""

    def test_non_positive_radius_raises(self, make_state: MakeState) -> None:
        """radius <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            f_density(make_state([[1.0, 1.0]]), radius=0.0)

    def test_negative_threshold_raises(self, make_state: MakeState) -> None:
        """Negative threshold must raise ValueError."""
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            f_density(make_state([[1.0, 1.0]]), threshold=-1.0)

    def test_negative_strength_raises(self, make_state: MakeState) -> None:
        """Negative strength must raise ValueError."""
        with pytest.raises(ValueError, match="strength must be non-negative"):
            f_density(make_state([[1.0, 1.0]]), strength=-1.0)


# ===========================================================================
# f_herd — panic-scaled alignment to local mean velocity (R2.5)
# ===========================================================================


class TestFHerdShape:
    """f_herd returns an (N, 2) float64 array."""

    def test_shape_and_dtype(self, make_state: MakeState) -> None:
        """Output is (N, 2) float64."""
        out = f_herd(make_state([[1.0, 1.0], [1.5, 1.0]], panic=[0.5, 0.5]))
        assert out.shape == (2, 2)
        assert out.dtype == np.float64

    def test_empty_population(self, make_state: MakeState) -> None:
        """Zero agents -> shape (0, 2)."""
        assert f_herd(make_state([])).shape == (0, 2)


class TestFHerdPanicScaling:
    """Herd alignment is zero without panic and grows with it (R2.5)."""

    @pytest.fixture
    def herd_scene(self, make_state: MakeState) -> Callable[[float], AgentState]:
        """Factory: agent 0 at rest amid neighbours moving +y, given panic."""

        def _scene(panic: float) -> AgentState:
            pos = [[5.0, 5.0], [5.4, 5.0], [4.6, 5.0], [5.0, 5.4]]
            vel = [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            return make_state(pos, vel=vel, panic=[panic, 0.0, 0.0, 0.0])

        return _scene

    def test_no_panic_no_herd(
        self, herd_scene: Callable[[float], AgentState]
    ) -> None:
        """A calm agent ignores the herd."""
        np.testing.assert_array_equal(f_herd(herd_scene(0.0))[0], [0.0, 0.0])

    def test_herd_pulls_toward_mean_velocity(
        self, herd_scene: Callable[[float], AgentState]
    ) -> None:
        """A panicked agent is accelerated toward the neighbours' +y flow."""
        assert f_herd(herd_scene(1.0))[0, 1] > 0.0

    def test_herd_increases_with_panic(
        self, herd_scene: Callable[[float], AgentState]
    ) -> None:
        """Alignment magnitude rises monotonically with panic."""
        mags = [
            float(np.linalg.norm(f_herd(herd_scene(p))[0]))
            for p in (0.0, 0.25, 0.5, 1.0)
        ]
        assert mags[0] == 0.0
        assert mags[1] < mags[2] < mags[3]

    def test_isolated_agent_no_herd(self, make_state: MakeState) -> None:
        """An agent with no neighbours within radius feels no herd."""
        state = make_state(
            [[0.0, 0.0], [50.0, 50.0]],
            vel=[[0.0, 0.0], [0.0, 2.0]],
            panic=[1.0, 1.0],
        )
        np.testing.assert_array_equal(f_herd(state)[0], [0.0, 0.0])


class TestFHerdFailures:
    """f_herd validates its tuning arguments."""

    def test_non_positive_radius_raises(self, make_state: MakeState) -> None:
        """radius <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            f_herd(make_state([[1.0, 1.0]]), radius=0.0)

    def test_negative_strength_raises(self, make_state: MakeState) -> None:
        """Negative strength must raise ValueError."""
        with pytest.raises(ValueError, match="strength must be non-negative"):
            f_herd(make_state([[1.0, 1.0]]), strength=-1.0)


# ===========================================================================
# Shared-hash optimisation path
# ===========================================================================


class TestSharedHash:
    """Passing a pre-built hash must match the internally-built result."""

    def test_shared_hash_matches_internal(
        self, make_state: MakeState
    ) -> None:
        """f_crowd with a supplied hash equals the auto-built version."""
        state = make_state([[1.0, 1.0], [1.2, 1.0], [1.1, 1.3]])
        sh = SpatialHash.build(state, cell_size=REPULSION_RADIUS)
        np.testing.assert_array_equal(f_crowd(state, sh), f_crowd(state))

    def test_too_coarse_hash_raises(self, make_state: MakeState) -> None:
        """A hash with a smaller cell than the radius is rejected."""
        state = make_state([[1.0, 1.0], [1.2, 1.0]])
        sh = SpatialHash.build(state, cell_size=0.3)
        with pytest.raises(ValueError, match="must be"):
            f_crowd(state, sh, radius=REPULSION_RADIUS)
