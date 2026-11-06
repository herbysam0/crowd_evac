"""Hard no-overlap projection — agents never overlap agents or walls.

The additive ``f_crowd`` repulsion term (FR-2 R2.1) *discourages* overlap but
cannot guarantee it: inertia, panic boosts, and dense queues at a constriction
can still drive two integrated centres closer than a body-width apart, or push
a centre to within a radius of a wall. This module adds the missing hard
constraint required by step 1.19a item 12 — a per-tick geometric position
projection run after integration and static collision resolution:

1. **Agent-agent** — every pair of live agents whose centre distance is below
   ``2 * AGENT_RADIUS`` is pushed apart along the line joining their centres,
   each moving half the penetration (position-based-dynamics contact rule).
2. **Agent-wall** — every agent centre closer than ``AGENT_RADIUS`` to a
   blocked or out-of-bounds cell is pushed back out to that clearance, using
   the static grid owned by :class:`~crowd_evac.domain.collision.CollisionMap`.

Both passes are repeated ``OVERLAP_RESOLUTION_ITERATIONS`` times so chained
overlaps (an agent shoved by a neighbour into a third, or off a wall into a
peer) converge. A final :meth:`CollisionMap.resolve` backstop guarantees the
non-negotiable invariant that no active centre ever ends a tick inside a
blocked cell — obstacles are never crossed even when a separation push would
otherwise nudge a centre across a wall.

The projection only repositions agents; velocities are left to the next tick's
force/integration cycle (the backstop still zeroes velocity into a blocked
axis). All operations are deterministic so seeded runs stay reproducible
(R6.2). Pure NumPy — no engine or I/O imports.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from crowd_evac.domain.agent_state import AgentState, Int1D, Vec2Array
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.constants import (
    AGENT_RADIUS,
    OVERLAP_RESOLUTION_ITERATIONS,
)
from crowd_evac.domain.spatial_hash import SpatialHash

MIN_AGENT_SEPARATION: float = 2.0 * AGENT_RADIUS
"""Minimum allowed centre-to-centre distance between two agents (metres)."""

WALL_CLEARANCE: float = AGENT_RADIUS
"""Minimum allowed centre-to-wall distance for any agent (metres)."""


def resolve_overlaps(
    state: AgentState,
    collision_map: CollisionMap | None = None,
    prev_pos: Vec2Array | None = None,
    *,
    iterations: int = OVERLAP_RESOLUTION_ITERATIONS,
    constraints: Sequence[Callable[[AgentState], None]] = (),
) -> None:
    """Enforce the hard no-overlap invariant by projecting agent positions.

    Mutates ``state.pos`` in place so that, after the call, no two live agents
    are closer than ``2 * AGENT_RADIUS`` and (when a collision map is supplied)
    no agent centre is nearer than ``AGENT_RADIUS`` to a wall — best-effort
    within ``iterations`` passes — while the backstop guarantees no centre rests
    inside a blocked cell.

    Args:
        state: Agent state whose positions are corrected in place. Velocities
            are not modified here (the backstop may zero a blocked axis).
        collision_map: Static blocking grid used for the wall-clearance pass and
            the final containment backstop. ``None`` runs the agent-agent pass
            only (e.g. open-room headless tests with no geometry).
        prev_pos: Per-agent positions before this tick's integration, shape
            ``(state.count, 2)``; required for the backstop. When ``None`` the
            backstop is skipped.
        iterations: Number of separation/wall projection passes. Must be
            non-negative; ``0`` is a no-op apart from the backstop.
        constraints: Extra position constraints applied after the wall pass
            in every iteration (e.g. the exit model's door gate), so the
            separation push cannot drive agents through them.

    Raises:
        ValueError: If ``iterations`` is negative.
    """
    if iterations < 0:
        raise ValueError(
            f"iterations must be non-negative, got {iterations!r}"
        )
    for _ in range(iterations):
        _separate_agents(state)
        if collision_map is not None:
            _push_off_walls(state, collision_map)
        for constrain in constraints:
            constrain(state)

    # Hard backstop: a separation or wall push may have moved a centre into a
    # blocked cell; resolve slides it back so obstacles are never crossed.
    if collision_map is not None and prev_pos is not None:
        collision_map.resolve(state, prev_pos)


def _separate_agents(state: AgentState) -> None:
    """Push every overlapping live-agent pair apart by half the penetration."""
    active = state.active_indices
    if active.size < 2:
        return
    spatial_hash = SpatialHash.build(state, MIN_AGENT_SEPARATION)
    gi, gj = spatial_hash.query_pairs()
    if gi.size == 0:
        return

    # query_pairs yields both (i, j) and (j, i); keep one per unordered pair.
    keep = gi < gj
    gi, gj = gi[keep], gj[keep]
    delta: Vec2Array = state.pos[gi] - state.pos[gj]  # points j -> i
    dist = np.linalg.norm(delta, axis=1)
    overlap = dist < MIN_AGENT_SEPARATION
    if not np.any(overlap):
        return
    gi, gj, delta, dist = gi[overlap], gj[overlap], delta[overlap], dist[overlap]

    # Coincident centres have no separation axis; pick +x deterministically
    # so seeded runs stay reproducible (R6.2).
    coincident = dist <= 0.0
    safe_dist = np.where(coincident, 1.0, dist)
    direction: Vec2Array = delta / safe_dist[:, np.newaxis]
    direction[coincident] = (1.0, 0.0)

    penetration = MIN_AGENT_SEPARATION - dist  # > 0; zero for coincident pairs
    shift: Vec2Array = 0.5 * penetration[:, np.newaxis] * direction
    np.add.at(state.pos, gi, shift)
    np.add.at(state.pos, gj, -shift)


def _push_off_walls(state: AgentState, collision_map: CollisionMap) -> None:
    """Move every live agent at least ``WALL_CLEARANCE`` from blocked cells."""
    active = state.active_indices
    if active.size == 0:
        return
    push = _wall_pushout(collision_map, state.pos[active], WALL_CLEARANCE)
    state.pos[active] += push


def _wall_pushout(
    collision_map: CollisionMap, points: Vec2Array, clearance: float
) -> Vec2Array:
    """Per-point displacement to ``clearance`` from the deepest blocked contact.

    Scans the grid neighbourhood reachable within ``clearance`` around each
    point and returns the outward displacement that clears the most deeply
    penetrating blocked (or out-of-bounds) cell. Points already clear, or whose
    centre sits exactly inside a blocked cell (no outward normal; left to the
    backstop), receive a zero row. A flat wall yields a purely normal push
    from the perpendicular cell; corners are resolved one axis per outer
    projection pass. Ties keep the first cell in row-major offset order.

    All ``(2 * reach + 1) ** 2`` candidate cells are evaluated in one
    vectorised batch: at a 5 cm grid the neighbourhood is 13 x 13 cells, and a
    per-offset loop dominated the tick.

    Args:
        collision_map: Source of the static blocking grid and cell size.
        points: World coordinates of shape ``(N, 2)`` as ``(x, y)``.
        clearance: Minimum required centre-to-wall distance in metres.

    Returns:
        Float64 array of shape ``(N, 2)`` of per-point displacements.
    """
    pts = np.asarray(points, dtype=np.float64)
    push: Vec2Array = np.zeros_like(pts)
    n = pts.shape[0]
    if n == 0:
        return push

    cell = collision_map.cell_size
    reach = int(np.ceil(clearance / cell))
    offsets = np.arange(-reach, reach + 1, dtype=np.intp)
    d_row, d_col = (a.ravel() for a in np.meshgrid(offsets, offsets, indexing="ij"))
    rows: Int1D = np.floor(pts[:, 1] / cell).astype(np.intp)[:, None] + d_row
    cols: Int1D = np.floor(pts[:, 0] / cell).astype(np.intp)[:, None] + d_col

    px = pts[:, 0:1]
    py = pts[:, 1:2]
    vec_x = px - np.clip(px, cols * cell, (cols + 1) * cell)
    vec_y = py - np.clip(py, rows * cell, (rows + 1) * cell)
    dist = np.hypot(vec_x, vec_y)
    pen = clearance - dist
    blocked = collision_map.cells_blocked(rows.ravel(), cols.ravel())
    hit = blocked.reshape(rows.shape) & (pen > 0.0) & (dist > 0.0)

    best = np.argmax(np.where(hit, pen, -np.inf), axis=1)
    agents = np.flatnonzero(hit.any(axis=1))
    k = best[agents]
    scale = pen[agents, k] / dist[agents, k]
    push[agents, 0] = vec_x[agents, k] * scale
    push[agents, 1] = vec_y[agents, k] * scale
    return push
