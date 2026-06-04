"""Tests for crowd_evac.domain.overlap (step 1.19a item 12).

Covers the hard no-overlap projection:
  - Module constants derive from AGENT_RADIUS.
  - Agent-agent separation: overlapping pairs pushed to >= 2R, far pairs and
    dead agents untouched, coincident centres separated deterministically.
  - Wall clearance: an agent within a radius of a wall is pushed out to >= R.
  - Backstop: no active centre ever ends inside a blocked cell.
  - Edge/failure paths: empty state, no-map run, iterations=0, negative
    iterations.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.constants import AGENT_RADIUS
from crowd_evac.domain.overlap import (
    MIN_AGENT_SEPARATION,
    WALL_CLEARANCE,
    resolve_overlaps,
)

from .conftest import MakeState


def _wall_column_map() -> CollisionMap:
    """5x5 unit-cell grid whose column c == 2 is a vertical wall."""
    blocked = np.zeros((5, 5), dtype=np.bool_)
    blocked[:, 2] = True
    return CollisionMap(blocked, cell_size=1.0)


def _min_pairwise(pos: npt.NDArray[np.float64]) -> float:
    """Smallest centre-to-centre distance among the given positions."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Derived separation constants track the agent radius."""

    def test_min_separation_is_two_radii(self) -> None:
        """Minimum centre distance equals one agent diameter."""
        assert MIN_AGENT_SEPARATION == pytest.approx(2.0 * AGENT_RADIUS)

    def test_wall_clearance_is_one_radius(self) -> None:
        """Minimum centre-to-wall distance equals the agent radius."""
        assert WALL_CLEARANCE == pytest.approx(AGENT_RADIUS)


# ---------------------------------------------------------------------------
# Agent-agent separation
# ---------------------------------------------------------------------------


class TestAgentSeparation:
    """Overlapping live pairs are pushed apart to at least one diameter."""

    def test_overlapping_pair_pushed_to_min_separation(
        self, make_state: MakeState
    ) -> None:
        """Two overlapping agents end exactly one diameter apart (happy).

        A single isolated pair converges in one pass: each moves half the
        penetration, so the post-projection distance equals MIN_AGENT_SEPARATION.
        """
        gap = 0.3  # well inside 2R = 1.1 m
        state = make_state([[0.0, 0.0], [gap, 0.0]])
        resolve_overlaps(state, iterations=1)
        dist = float(np.linalg.norm(state.pos[0] - state.pos[1]))
        assert dist == pytest.approx(MIN_AGENT_SEPARATION)

    def test_symmetric_push(self, make_state: MakeState) -> None:
        """Both agents move by an equal and opposite half-penetration."""
        state = make_state([[0.0, 0.0], [0.4, 0.0]])
        resolve_overlaps(state, iterations=1)
        # Midpoint is preserved; each moved the same distance outward.
        assert state.pos[0, 0] == pytest.approx(-state.pos[1, 0] + 0.4)
        assert state.pos[:, 1] == pytest.approx([0.0, 0.0])

    def test_far_pair_unchanged(self, make_state: MakeState) -> None:
        """Agents already beyond one diameter are not moved (edge)."""
        pos = [[0.0, 0.0], [3.0, 0.0]]
        state = make_state(pos)
        resolve_overlaps(state, iterations=4)
        np.testing.assert_array_equal(state.pos, np.asarray(pos))

    def test_coincident_centres_separated_deterministically(
        self, make_state: MakeState
    ) -> None:
        """Two agents at the same point separate to 2R along a fixed axis.

        With no defined separation axis the projection picks +x deterministically
        so seeded runs stay reproducible; the pair ends one diameter apart.
        """
        state_a = make_state([[1.0, 1.0], [1.0, 1.0]])
        state_b = make_state([[1.0, 1.0], [1.0, 1.0]])
        resolve_overlaps(state_a, iterations=1)
        resolve_overlaps(state_b, iterations=1)
        np.testing.assert_array_equal(state_a.pos, state_b.pos)
        dist = float(np.linalg.norm(state_a.pos[0] - state_a.pos[1]))
        assert dist == pytest.approx(MIN_AGENT_SEPARATION)

    def test_dead_agent_excluded(self, make_state: MakeState) -> None:
        """A dead agent neither moves nor pushes a live overlapping neighbour."""
        pos = [[0.0, 0.0], [0.2, 0.0]]
        state = make_state(pos, alive=[True, False])
        resolve_overlaps(state, iterations=4)
        np.testing.assert_array_equal(state.pos, np.asarray(pos))

    def test_crowd_converges_below_tolerance(
        self, make_state: MakeState
    ) -> None:
        """A clustered crowd relaxes so no pair stays grossly overlapped.

        Nine agents packed on a 0.4 m lattice (deep mutual overlap) are
        separated by the default iteration count until the closest pair clears
        a documented tolerance of 0.9 x one diameter — finite Jacobi passes do
        not reach the exact bound from a deep start in a single call.
        """
        xs, ys = np.meshgrid(np.arange(3) * 0.4, np.arange(3) * 0.4)
        pos = np.column_stack((xs.ravel(), ys.ravel()))
        state = make_state(pos)
        resolve_overlaps(state, iterations=20)
        assert _min_pairwise(state.pos) >= 0.9 * MIN_AGENT_SEPARATION


# ---------------------------------------------------------------------------
# Wall clearance and backstop
# ---------------------------------------------------------------------------


class TestWallClearance:
    """Agents are pushed clear of walls and never end inside a blocked cell."""

    def test_agent_pushed_to_clearance_from_wall(
        self, make_state: MakeState
    ) -> None:
        """An agent within a radius of the wall is pushed out to exactly R.

        The wall column starts at x = 2.0; an agent at x = 1.7 is 0.3 m from it
        (< 0.55 m) and must be pushed to x = 2.0 - 0.55 = 1.45.
        """
        cmap = _wall_column_map()
        state = make_state([[1.7, 0.5]])
        prev = state.pos.copy()
        resolve_overlaps(state, cmap, prev, iterations=4)
        assert state.pos[0, 0] == pytest.approx(2.0 - WALL_CLEARANCE)
        assert 2.0 - state.pos[0, 0] >= WALL_CLEARANCE - 1e-9

    def test_agent_clear_of_wall_unchanged(
        self, make_state: MakeState
    ) -> None:
        """An agent beyond the clearance from all geometry is not pushed (edge).

        ``(1.0, 2.5)`` sits 1.0 m from the wall column (x = 2.0) and well over a
        radius from every room boundary, so no clearance push applies.
        """
        cmap = _wall_column_map()
        state = make_state([[1.0, 2.5]])
        prev = state.pos.copy()
        resolve_overlaps(state, cmap, prev, iterations=4)
        np.testing.assert_array_equal(state.pos[0], np.array([1.0, 2.5]))

    def test_backstop_keeps_centres_out_of_blocked_cells(
        self, make_state: MakeState
    ) -> None:
        """No active centre ends inside a blocked cell after projection.

        Agents are scattered through the open columns and pushed by both passes;
        the final backstop guarantees the hard containment invariant.
        """
        cmap = _wall_column_map()
        rng = np.random.default_rng(11)
        pos = rng.uniform(low=[0.1, 0.1], high=[1.9, 4.9], size=(30, 2))
        # Pack several near the wall so the clearance pass is exercised.
        pos[:10, 0] = rng.uniform(1.6, 1.95, size=10)
        state = make_state(pos)
        prev = state.pos.copy()
        resolve_overlaps(state, cmap, prev, iterations=4)
        assert not np.any(cmap.is_blocked(state.pos))


# ---------------------------------------------------------------------------
# Edge and failure paths
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Degenerate inputs and argument validation."""

    def test_empty_state_no_op(self, make_state: MakeState) -> None:
        """A zero-agent state projects without error (edge)."""
        cmap = _wall_column_map()
        state = make_state(np.empty((0, 2), dtype=np.float64))
        resolve_overlaps(state, cmap, state.pos.copy(), iterations=4)
        assert state.pos.shape == (0, 2)

    def test_single_agent_no_map_unchanged(
        self, make_state: MakeState
    ) -> None:
        """One agent with no map and no neighbours is untouched (edge)."""
        state = make_state([[1.0, 1.0]])
        resolve_overlaps(state, iterations=4)
        np.testing.assert_array_equal(state.pos[0], np.array([1.0, 1.0]))

    def test_zero_iterations_skips_projection(
        self, make_state: MakeState
    ) -> None:
        """iterations=0 leaves overlapping agents in place (no map, no backstop)."""
        pos = [[0.0, 0.0], [0.2, 0.0]]
        state = make_state(pos)
        resolve_overlaps(state, iterations=0)
        np.testing.assert_array_equal(state.pos, np.asarray(pos))

    def test_negative_iterations_raises(self, make_state: MakeState) -> None:
        """A negative iteration count raises ValueError (failure path)."""
        state = make_state([[0.0, 0.0]])
        with pytest.raises(ValueError, match="iterations must be non-negative"):
            resolve_overlaps(state, iterations=-1)
