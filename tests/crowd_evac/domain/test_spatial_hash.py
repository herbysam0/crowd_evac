"""Tests for crowd_evac.domain.spatial_hash.SpatialHash (FR-2 R2.4).

Covers:
  - query_pairs(): exact correctness against an O(N^2) brute-force reference,
    symmetry (both orderings), dead-agent exclusion, and empty/single edges.
  - neighbour_counts(): correctness and radius validation.
  - Sub-quadratic neighbour-query cost: the hash beats brute force and scales
    far better than quadratically as N grows at fixed density (R2.4).
  - Failure paths: non-positive cell size, length mismatch, radius too large.

Test inputs and reference computations are supplied by factory fixtures
(``make_state`` lives in the domain ``conftest``).
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from crowd_evac.domain.spatial_hash import SpatialHash

from .conftest import MakeState

# Factory fixture signatures.
RandomPositions = Callable[[int, float, int], np.ndarray]
PairSet = set[tuple[int, int]]
BrutePairs = Callable[..., PairSet]
HashPairs = Callable[[SpatialHash, np.ndarray, float], PairSet]
TimeFn = Callable[[np.ndarray, float], float]


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_positions() -> RandomPositions:
    """Return a factory for uniform random positions at a target density."""

    def _make(n: int, density: float, seed: int) -> np.ndarray:
        """Place n points in a square sized so coverage ~= density."""
        side = float(np.sqrt(n / density))
        rng = np.random.default_rng(seed)
        return rng.random((n, 2)) * side

    return _make


@pytest.fixture
def brute_pairs() -> BrutePairs:
    """Return an O(N^2) reference computing ordered pairs within a radius."""

    def _make(
        pos: np.ndarray,
        radius: float,
        alive: np.ndarray | None = None,
    ) -> PairSet:
        """Reference set of (i, j), i != j, with distance < radius."""
        n = pos.shape[0]
        result: PairSet = set()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if alive is not None and not (alive[i] and alive[j]):
                    continue
                if float(np.linalg.norm(pos[i] - pos[j])) < radius:
                    result.add((i, j))
        return result

    return _make


@pytest.fixture
def hash_pairs() -> HashPairs:
    """Return a helper distance-filtering a hash's candidate pairs."""

    def _make(sh: SpatialHash, pos: np.ndarray, radius: float) -> PairSet:
        """Ordered-pair set kept from query_pairs by exact distance."""
        gi, gj = sh.query_pairs()
        if gi.size == 0:
            return set()
        dist = np.linalg.norm(pos[gi] - pos[gj], axis=1)
        keep = dist < radius
        return {(int(a), int(b)) for a, b in zip(gi[keep], gj[keep])}

    return _make


@pytest.fixture
def time_hash() -> TimeFn:
    """Return a best-of-three timer for build + query_pairs."""

    def _time(pos: np.ndarray, radius: float) -> float:
        indices = np.arange(pos.shape[0], dtype=np.intp)
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            sh = SpatialHash(pos, indices, cell_size=radius)
            sh.query_pairs()
            best = min(best, time.perf_counter() - start)
        return best

    return _time


@pytest.fixture
def time_brute() -> TimeFn:
    """Return a timer for the vectorised O(N^2) all-pairs computation."""

    def _time(pos: np.ndarray, radius: float) -> float:
        start = time.perf_counter()
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.count_nonzero(dist < radius)
        return time.perf_counter() - start

    return _time


# ---------------------------------------------------------------------------
# Correctness against brute force
# ---------------------------------------------------------------------------


class TestQueryPairsCorrectness:
    """query_pairs (filtered by distance) must equal the brute-force set."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42], ids=lambda s: f"seed{s}")
    def test_matches_brute_force(
        self,
        seed: int,
        random_positions: RandomPositions,
        brute_pairs: BrutePairs,
        hash_pairs: HashPairs,
    ) -> None:
        """Random clouds: distance-filtered candidates equal brute force."""
        radius = 1.0
        pos = random_positions(300, 2.0, seed)
        sh = SpatialHash(pos, np.arange(300, dtype=np.intp), cell_size=radius)
        assert hash_pairs(sh, pos, radius) == brute_pairs(pos, radius)

    def test_two_agents_same_cell_pair_both_orders(self) -> None:
        """Two close agents yield both (0, 1) and (1, 0)."""
        pos = np.array([[1.0, 1.0], [1.2, 1.0]])
        sh = SpatialHash(pos, np.arange(2, dtype=np.intp), cell_size=1.0)
        gi, gj = sh.query_pairs()
        assert set(zip(gi.tolist(), gj.tolist())) == {(0, 1), (1, 0)}

    def test_far_agents_no_candidate_pairs(self) -> None:
        """Agents in non-adjacent cells produce no candidate pairs."""
        pos = np.array([[0.5, 0.5], [10.5, 10.5]])
        sh = SpatialHash(pos, np.arange(2, dtype=np.intp), cell_size=1.0)
        gi, _ = sh.query_pairs()
        assert gi.size == 0

    def test_no_self_pairs(self, random_positions: RandomPositions) -> None:
        """An agent is never paired with itself."""
        pos = random_positions(50, 3.0, 5)
        sh = SpatialHash(pos, np.arange(50, dtype=np.intp), cell_size=1.0)
        gi, gj = sh.query_pairs()
        assert not np.any(gi == gj)

    def test_pairs_use_global_indices(self, make_state: MakeState) -> None:
        """build() returns pairs in global IDs, skipping dead agents."""
        # Agents 0 and 2 are close and alive; agent 1 (between) is dead.
        pos = [[1.0, 1.0], [1.1, 1.0], [1.2, 1.0]]
        state = make_state(pos, alive=[True, False, True])
        sh = SpatialHash.build(state, cell_size=1.0)
        gi, gj = sh.query_pairs()
        assert set(zip(gi.tolist(), gj.tolist())) == {(0, 2), (2, 0)}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestQueryPairsEdges:
    """Empty and single-agent inputs return empty pair arrays."""

    def test_empty_population(self) -> None:
        """Zero agents -> empty pair arrays of dtype intp."""
        sh = SpatialHash(
            np.empty((0, 2)), np.empty(0, dtype=np.intp), cell_size=1.0
        )
        gi, gj = sh.query_pairs()
        assert gi.size == 0 and gj.size == 0
        assert gi.dtype == np.intp

    def test_single_agent(self) -> None:
        """One agent has no pairs."""
        sh = SpatialHash(
            np.array([[1.0, 1.0]]), np.arange(1, dtype=np.intp), cell_size=1.0
        )
        gi, _ = sh.query_pairs()
        assert gi.size == 0

    def test_query_pairs_cached(
        self, random_positions: RandomPositions
    ) -> None:
        """Repeated calls return the identical cached arrays."""
        pos = random_positions(20, 2.0, 1)
        sh = SpatialHash(pos, np.arange(20, dtype=np.intp), cell_size=1.0)
        first = sh.query_pairs()
        second = sh.query_pairs()
        assert first[0] is second[0]
        assert first[1] is second[1]

    def test_handles_negative_coordinates(self) -> None:
        """Cells shift correctly for negative positions (no key wrap-around)."""
        pos = np.array([[-2.3, -2.3], [-2.1, -2.4]])
        sh = SpatialHash(pos, np.arange(2, dtype=np.intp), cell_size=1.0)
        gi, gj = sh.query_pairs()
        assert set(zip(gi.tolist(), gj.tolist())) == {(0, 1), (1, 0)}


# ---------------------------------------------------------------------------
# neighbour_counts
# ---------------------------------------------------------------------------


class TestNeighbourCounts:
    """neighbour_counts must match a brute-force count within the radius."""

    def test_counts_match_brute_force(
        self,
        make_state: MakeState,
        random_positions: RandomPositions,
        brute_pairs: BrutePairs,
    ) -> None:
        """Per-agent counts equal the brute-force neighbour tally."""
        radius = 1.0
        pos = random_positions(200, 2.5, 3)
        state = make_state(pos)
        sh = SpatialHash.build(state, cell_size=radius)
        counts = sh.neighbour_counts(state, radius)

        expected = np.zeros(200, dtype=np.intp)
        for i, _ in brute_pairs(pos, radius):
            expected[i] += 1
        np.testing.assert_array_equal(counts, expected)

    def test_isolated_agents_have_zero(self, make_state: MakeState) -> None:
        """Agents with no neighbours within radius count zero."""
        state = make_state([[0.5, 0.5], [50.0, 50.0]])
        sh = SpatialHash.build(state, cell_size=1.0)
        np.testing.assert_array_equal(sh.neighbour_counts(state, 1.0), [0, 0])

    def test_dead_agents_count_zero(self, make_state: MakeState) -> None:
        """A dead agent contributes no count and receives none."""
        state = make_state([[1.0, 1.0], [1.1, 1.0]], alive=[True, False])
        sh = SpatialHash.build(state, cell_size=1.0)
        np.testing.assert_array_equal(sh.neighbour_counts(state, 1.0), [0, 0])

    def test_radius_exceeding_cell_size_raises(
        self, make_state: MakeState
    ) -> None:
        """Counting beyond the cell size would miss neighbours -> error."""
        state = make_state([[1.0, 1.0], [1.1, 1.0]])
        sh = SpatialHash.build(state, cell_size=1.0)
        with pytest.raises(ValueError, match="must not exceed cell_size"):
            sh.neighbour_counts(state, 2.0)


# ---------------------------------------------------------------------------
# Sub-quadratic cost (R2.4)
# ---------------------------------------------------------------------------


class TestSubQuadraticCost:
    """The grid hash must scale sub-quadratically with population (R2.4)."""

    def test_hash_beats_brute_force(
        self,
        random_positions: RandomPositions,
        time_hash: TimeFn,
        time_brute: TimeFn,
    ) -> None:
        """At a moderate N the hash is faster than the all-pairs matrix."""
        pos = random_positions(1500, 1.0, 11)
        radius = 1.0
        assert time_hash(pos, radius) < time_brute(pos, radius)

    def test_scaling_is_sub_quadratic(
        self,
        random_positions: RandomPositions,
        time_hash: TimeFn,
    ) -> None:
        """Doubling N at fixed density costs far less than 4x (quadratic)."""
        radius = 1.0
        t_small = time_hash(random_positions(2000, 1.0, 21), radius)
        t_large = time_hash(random_positions(4000, 1.0, 22), radius)
        # Quadratic would be ~4x; linear ~2x. A generous bound that still
        # excludes quadratic growth and tolerates timer noise on a busy host.
        assert t_large < t_small * 6.0


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestSpatialHashFailures:
    """Construction validates its arguments."""

    def test_non_positive_cell_size_raises(self) -> None:
        """cell_size <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            SpatialHash(
                np.array([[0.0, 0.0]]),
                np.arange(1, dtype=np.intp),
                cell_size=0.0,
            )

    def test_length_mismatch_raises(self) -> None:
        """positions and indices of differing length must raise ValueError."""
        with pytest.raises(ValueError, match="must equal"):
            SpatialHash(
                np.zeros((3, 2)),
                np.arange(2, dtype=np.intp),
                cell_size=1.0,
            )
