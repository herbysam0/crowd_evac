"""Uniform-grid spatial hash for fast fixed-radius neighbour queries (FR-2 R2.4).

The crowd-dynamics forces (repulsion, density pressure, herd alignment) all
need, for every agent, the set of nearby agents within some interaction
radius. Computing that with an all-pairs distance matrix is ``O(N^2)`` in time
and memory and is infeasible at the Tier A target (~10k agents). This module
buckets agents into a uniform grid whose cell side equals the query radius, so
every neighbour within that radius falls in the agent's own cell or one of the
eight adjacent cells. Querying then costs ``O(N + P)`` where ``P`` is the
number of close pairs — sub-quadratic for the bounded densities of a crowd.

:class:`SpatialHash` is built once per tick from the live agents and exposes
:meth:`query_pairs`, returning every ordered close pair ``(i, j)`` with
``i != j`` as **global** agent indices (indices into the parent
:class:`~crowd_evac.domain.agent_state.AgentState` arrays). Dead agents are
excluded at build time, so they neither appear in nor influence any pair.

Pure NumPy — no engine or I/O imports.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.agent_state import AgentState, Int1D, Vec2Array

# 3x3 cell neighbourhood (including the agent's own cell), as (dx, dy) in
# cell-coordinate space. With cell side == query radius this covers every
# neighbour within the radius.
_NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
)


class SpatialHash:
    """Uniform-grid index over agent positions for fast neighbour queries.

    Construct via :meth:`build` (the common path) or directly from a
    positions array and matching global indices. Cells are linearised to
    int64 keys; queries join each agent's nine candidate cells against the
    sorted key table with :func:`numpy.searchsorted`, then expand the matched
    ranges into pair index arrays. The result of :meth:`query_pairs` is cached
    on first call, so the three crowd forces can share one build per tick.

    Attributes:
        cell_size: Grid cell side length in metres (equals the query radius
            this hash is valid for).
    """

    def __init__(
        self,
        positions: Vec2Array,
        indices: Int1D,
        cell_size: float,
    ) -> None:
        """Build the grid index from active positions and their global IDs.

        Args:
            positions: Active-agent positions, shape ``(A, 2)`` in metres.
            indices: Global agent indices for each row of ``positions``,
                shape ``(A,)``. Returned pairs are expressed in these IDs.
            cell_size: Grid cell side length in metres. Must be positive and
                should equal the neighbour-query radius.

        Raises:
            ValueError: If ``cell_size`` is not positive, or ``positions`` and
                ``indices`` disagree on length.
        """
        if cell_size <= 0.0:
            raise ValueError(
                f"cell_size must be positive, got {cell_size!r}"
            )
        if positions.shape[0] != indices.shape[0]:
            raise ValueError(
                f"positions length ({positions.shape[0]!r}) must equal "
                f"indices length ({indices.shape[0]!r})"
            )
        self.cell_size = float(cell_size)
        self._indices: Int1D = indices.astype(np.intp)
        self._count = int(positions.shape[0])
        self._pairs: tuple[Int1D, Int1D] | None = None
        # Per-pair (delta, squared-distance), computed once and shared by the
        # crowd force terms that reuse this hash within a tick.
        self._pair_off: tuple[Vec2Array, npt.NDArray[np.float64]] | None = None

        if self._count == 0:
            # Degenerate grid; query_pairs short-circuits on empty input.
            self._cx = np.empty(0, dtype=np.int64)
            self._cy = np.empty(0, dtype=np.int64)
            self._stride = 1
            self._cell_key = np.empty(0, dtype=np.int64)
            return

        cells = np.floor(positions / self.cell_size).astype(np.int64)
        # Shift so the minimum cell index is 1; neighbour queries then span
        # [0, max+1] without underflow, and the stride (max cy + 2) leaves
        # room for the +/-1 offset so linear keys never wrap between rows.
        self._cx = cells[:, 0] - cells[:, 0].min() + 1
        self._cy = cells[:, 1] - cells[:, 1].min() + 1
        self._stride = int(self._cy.max()) + 2
        self._cell_key = self._cx * self._stride + self._cy

    @classmethod
    def build(cls, state: AgentState, cell_size: float) -> SpatialHash:
        """Build a spatial hash over the live agents of ``state``.

        Only agents with ``alive == True`` are indexed; returned pairs use
        global agent indices, so dead agents are transparently excluded.

        Args:
            state: Agent population to index.
            cell_size: Grid cell side length in metres (the query radius).
                Must be positive.

        Returns:
            A :class:`SpatialHash` over the active agents.

        Raises:
            ValueError: If ``cell_size`` is not positive.
        """
        active = state.active_indices
        return cls(state.pos[active], active, cell_size)

    def query_pairs(self) -> tuple[Int1D, Int1D]:
        """Return every ordered close pair of agents in adjacent cells.

        A pair ``(i, j)`` is returned (as global indices) when agent ``j`` lies
        in the same cell as ``i`` or one of the eight surrounding cells, and
        ``i != j``. Both orderings ``(i, j)`` and ``(j, i)`` are returned, so a
        caller can accumulate a per-agent force by scattering over the first
        index. Pairs are *candidates*: callers must still filter by exact
        distance against their interaction radius (cells may be up to ~2x the
        radius across diagonally).

        The arrays are computed once and cached on the instance.

        Returns:
            A tuple ``(i, j)`` of equal-length ``intp`` arrays of global agent
            indices. Empty arrays when fewer than two agents are indexed.
        """
        if self._pairs is not None:
            return self._pairs

        n = self._count
        if n < 2:
            empty: Int1D = np.empty(0, dtype=np.intp)
            self._pairs = (empty, empty.copy())
            return self._pairs

        order = np.argsort(self._cell_key, kind="stable")
        sorted_keys = self._cell_key[order]
        base = np.arange(n, dtype=np.intp)

        i_parts: list[Int1D] = []
        j_parts: list[Int1D] = []
        for dx, dy in _NEIGHBOUR_OFFSETS:
            query_key = (self._cx + dx) * self._stride + (self._cy + dy)
            lo = np.searchsorted(sorted_keys, query_key, side="left")
            hi = np.searchsorted(sorted_keys, query_key, side="right")
            counts = hi - lo
            total = int(counts.sum())
            if total == 0:
                continue

            # Home agent (local) repeated once per matched neighbour.
            i_local = np.repeat(base, counts)
            # Expand each [lo, hi) range into flat positions in the sorted
            # array: repeat(lo) plus the within-group running index.
            exclusive_prefix = np.zeros(n, dtype=np.intp)
            exclusive_prefix[1:] = np.cumsum(counts)[:-1]
            within = np.arange(total, dtype=np.intp) - np.repeat(
                exclusive_prefix, counts
            )
            sorted_pos = np.repeat(lo, counts) + within
            j_local = order[sorted_pos]

            i_parts.append(i_local)
            j_parts.append(j_local)

        i_all = np.concatenate(i_parts)
        j_all = np.concatenate(j_parts)
        keep = i_all != j_all
        gi: Int1D = self._indices[i_all[keep]]
        gj: Int1D = self._indices[j_all[keep]]
        self._pairs = (gi, gj)
        return self._pairs

    def pair_offsets(
        self, state: AgentState
    ) -> tuple[Vec2Array, npt.NDArray[np.float64]]:
        """Return per-pair displacement and squared distance, computed once.

        For the candidate pairs from :meth:`query_pairs`, this computes
        ``delta = pos[i] - pos[j]`` and ``|delta|**2`` a single time and caches
        the result on the instance.  The three crowd force terms that share
        one hash per tick (repulsion, density, herd) then reuse this rather
        than recomputing the displacement and norm three times.

        Squared distance — not distance — is returned so callers needing only
        a radius mask (herd alignment, density counting) avoid the square root
        entirely; :func:`~crowd_evac.domain.forces.f_crowd` takes the root of
        just the in-range subset.

        The squared distance is computed with :func:`numpy.einsum`, which is
        markedly faster than :func:`numpy.linalg.norm` for the ``(P, 2)`` case
        as it avoids the intermediate ``abs`` and ufunc-reduce overhead.

        Args:
            state: The population this hash indexes; positions are read by the
                global indices returned from :meth:`query_pairs`.  The same
                state must be passed across calls within a tick, since the
                result is cached against the first call's positions.

        Returns:
            Tuple ``(delta, dist_sq)`` where ``delta`` has shape ``(P, 2)`` in
            metres and ``dist_sq`` shape ``(P,)`` in metres squared.  Both are
            empty when fewer than two agents are indexed.
        """
        if self._pair_off is not None:
            return self._pair_off

        gi, gj = self.query_pairs()
        if gi.size == 0:
            delta: Vec2Array = np.empty((0, 2), dtype=np.float64)
            dist_sq: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        else:
            delta = state.pos[gi] - state.pos[gj]
            dist_sq = np.einsum("ij,ij->i", delta, delta)
        self._pair_off = (delta, dist_sq)
        return self._pair_off

    def neighbour_counts(self, state: AgentState, radius: float) -> Int1D:
        """Count, per agent, how many other agents lie within ``radius``.

        Uses :meth:`query_pairs` candidates filtered by exact distance, so
        ``radius`` must not exceed :attr:`cell_size` or true neighbours beyond
        one cell would be missed.

        Args:
            state: The agent population this hash was built from (positions
                are read by global index).
            radius: Counting radius in metres. Must be ``<= cell_size``.

        Returns:
            An ``intp`` array of length ``state.count``; entry ``i`` is the
            number of distinct live agents within ``radius`` of agent ``i``.
            Dead agents have a count of zero.

        Raises:
            ValueError: If ``radius`` exceeds :attr:`cell_size`.
        """
        if radius > self.cell_size:
            raise ValueError(
                f"radius ({radius!r}) must not exceed cell_size "
                f"({self.cell_size!r})"
            )
        counts: Int1D = np.zeros(state.count, dtype=np.intp)
        gi, _ = self.query_pairs()
        if gi.size == 0:
            return counts
        _, dist_sq = self.pair_offsets(state)
        # Compare squared distances to avoid a full-array square root.
        within = gi[dist_sq < radius * radius]
        if within.size == 0:
            return counts
        return np.bincount(within, minlength=state.count).astype(
            np.intp, copy=False
        )
