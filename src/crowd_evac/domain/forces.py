"""Force terms acting on the agent population (FR-1 R1.2 / FR-14).

Each function returns a ``(N, 2)`` float64 acceleration array, where
``N == state.count``.  Dead agents (``alive[i] == False``) always receive
a zero row.  Force terms are additive; callers sum them before passing the
total to :func:`~crowd_evac.domain.integrator.step`.

Functions defined here:

- :func:`f_exit` (step 1.7): Exit-seeking self-driven force from the flow
  field.
- :func:`f_crowd` (step 1.8): Short-range agent-agent repulsion (R2.1).
- :func:`f_density` (step 1.8): Density-pressure deceleration that reduces
  effective speed in dense regions (R2.2).
- :func:`f_herd` (step 1.8): Panic-scaled alignment toward the mean velocity
  of nearby agents (R2.5).
- :func:`f_panic_repulsion` (step 1.10): Panic-source repulsion pushing
  agents down-gradient away from hazard sources (FR-11 R11.4).
- :func:`compose` (step 1.11): Combine all enabled force terms into a single
  acceleration array ready for the integrator (FR-14 R14.1).

The short-range pair terms (:func:`f_crowd`, :func:`f_density`) find
neighbours through :class:`~crowd_evac.domain.spatial_hash.SpatialHash`; a
caller may pass a pre-built hash (so the terms share one build per tick) or let
each term build its own at the appropriate radius.  :func:`compose` builds one
shared hash at the larger of those two radii so only one index is constructed
per tick.  :func:`f_herd` does *not* enumerate pairs: its perception radius is
comparable to the floor span, so it aggregates velocity per coarse grid cell in
``O(N)`` rather than degenerating to an all-pairs query.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.agent_state import AgentState, Int1D, Vec2Array
from crowd_evac.domain.constants import (
    DENSITY_PRESSURE_STRENGTH,
    DENSITY_SENSING_RADIUS,
    HERD_ATTRACTION_STRENGTH,
    HERD_PERCEPTION_RADIUS,
    HIGH_DENSITY_THRESHOLD,
    MAX_SPEED,
    PANIC_REPULSION_STRENGTH,
    PANIC_SPEED_MULTIPLIER,
    RELAXATION_TIME,
    REPULSION_MIN_DISTANCE,
    REPULSION_RADIUS,
    REPULSION_STRENGTH,
)
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.params import ForceParams
from crowd_evac.domain.spatial_hash import SpatialHash
from crowd_evac.pathfinding.flow_field import FlowField


def f_exit(
    state: AgentState,
    field: FlowField,
    relaxation_time: float = RELAXATION_TIME,
    *,
    max_speed: float = MAX_SPEED,
    panic_speed_multiplier: float = PANIC_SPEED_MULTIPLIER,
) -> Vec2Array:
    """Compute exit-seeking acceleration for all agents (FR-1 R1.2).

    Samples the flow field at each active agent's world position to get the
    desired exit-seeking direction, scales it by the agent's panic-modulated
    desired speed, and returns the social-force steering acceleration::

        a_i = (v_desired_i * direction_i − v_i) / relaxation_time

    where::

        v_desired_i = max_speed * (1 + panic_i * (panic_speed_multiplier − 1))

    Dead agents (``alive[i] == False``) receive a zero row in the output.

    Args:
        state: Current agent state (positions, velocities, panic levels).
        field: Pre-computed flow field providing exit-seeking unit directions.
        relaxation_time: Characteristic time in seconds for velocity
            relaxation. Must be positive.
        max_speed: Base maximum agent speed in m/s for the desired velocity
            computation. Defaults to the module constant ``MAX_SPEED``.
        panic_speed_multiplier: Speed boost factor at full panic. Defaults to
            the module constant ``PANIC_SPEED_MULTIPLIER``.

    Returns:
        Float64 array of shape ``(N, 2)`` with per-agent acceleration vectors
        in m/s². Dead agents have zero rows.

    Raises:
        ValueError: If relaxation_time is not positive.
    """
    if relaxation_time <= 0.0:
        raise ValueError(
            f"relaxation_time must be positive, got {relaxation_time!r}"
        )
    n = state.count
    out: Vec2Array = np.zeros((n, 2), dtype=np.float64)
    active = state.active_indices
    if active.size == 0:
        return out

    dirs: npt.NDArray[np.float64] = field.sample(state.pos[active])  # (A, 2)

    # Linearly interpolate desired speed between max_speed (no panic) and
    # max_speed * panic_speed_multiplier (full panic).
    desired_speed: npt.NDArray[np.float64] = max_speed * (
        1.0 + state.panic[active] * (panic_speed_multiplier - 1.0)
    )  # (A,)

    desired_vel: Vec2Array = dirs * desired_speed[:, np.newaxis]  # (A, 2)
    out[active] = (desired_vel - state.vel[active]) / relaxation_time
    return out


def _resolve_hash(
    state: AgentState,
    spatial_hash: SpatialHash | None,
    radius: float,
) -> SpatialHash:
    """Return a usable spatial hash, building one at ``radius`` if needed.

    Args:
        state: Agent population to index when building a fresh hash.
        spatial_hash: Caller-supplied hash, or ``None`` to build one.
        radius: Interaction radius this force needs to query.

    Returns:
        A spatial hash whose cell size is at least ``radius``.

    Raises:
        ValueError: If a supplied hash has a cell size smaller than
            ``radius`` (it would miss neighbours beyond one cell).
    """
    if spatial_hash is None:
        return SpatialHash.build(state, radius)
    if spatial_hash.cell_size < radius:
        raise ValueError(
            f"spatial_hash.cell_size ({spatial_hash.cell_size!r}) must be "
            f">= radius ({radius!r})"
        )
    return spatial_hash


def f_crowd(
    state: AgentState,
    spatial_hash: SpatialHash | None = None,
    *,
    radius: float = REPULSION_RADIUS,
    strength: float = REPULSION_STRENGTH,
    min_distance: float = REPULSION_MIN_DISTANCE,
) -> Vec2Array:
    """Compute short-range agent-agent repulsion for all agents (FR-2 R2.1).

    Every pair of live agents within ``radius`` pushes apart along the line
    joining their centres. The per-pair magnitude is::

        strength * (radius - d) / max(d, min_distance)

    which is zero at ``d == radius`` and rises steeply as the separation ``d``
    shrinks, so agents strongly resist overlapping (the integrator clamps the
    summed acceleration to ``MAX_ACCEL``). Dead agents are excluded by the
    spatial hash and receive a zero row.

    Args:
        state: Current agent state (positions, liveness).
        spatial_hash: Optional pre-built neighbour index. If ``None``, one is
            built at ``radius``. If supplied, its cell size must be
            ``>= radius``.
        radius: Repulsion cut-off distance in metres. Must be positive.
        strength: Repulsion magnitude scale. Must be non-negative.
        min_distance: Lower clamp on the pair separation in the denominator,
            keeping the unclamped force finite near contact. Must be positive.

    Returns:
        Float64 array of shape ``(N, 2)`` of repulsion accelerations in m/s².
        Dead agents have zero rows.

    Raises:
        ValueError: If ``radius`` or ``min_distance`` is not positive, if
            ``strength`` is negative, or if a supplied hash is too coarse.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if min_distance <= 0.0:
        raise ValueError(
            f"min_distance must be positive, got {min_distance!r}"
        )
    if strength < 0.0:
        raise ValueError(f"strength must be non-negative, got {strength!r}")

    out: Vec2Array = np.zeros((state.count, 2), dtype=np.float64)
    sh = _resolve_hash(state, spatial_hash, radius)
    gi, _ = sh.query_pairs()
    if gi.size == 0:
        return out

    delta, dist_sq = sh.pair_offsets(state)  # delta points j -> i
    # Mask on squared distance; the costly square root is taken only for the
    # in-range subset below.
    close = (dist_sq < radius * radius) & (dist_sq > 0.0)
    if not np.any(close):
        return out

    gi_c = gi[close]
    delta_c = delta[close]
    dist_c = np.sqrt(dist_sq[close])
    d_clamped = np.maximum(dist_c, min_distance)
    magnitude = strength * (radius - dist_c) / d_clamped  # (P,)
    # contrib = (delta_c / dist_c) * magnitude; fold the two scalar divides
    # into one per-pair factor to halve the elementwise work.
    factor: npt.NDArray[np.float64] = magnitude / dist_c  # (P,)
    n = state.count
    # bincount is far faster than the unbuffered np.add.at scatter.
    out[:, 0] = np.bincount(gi_c, weights=delta_c[:, 0] * factor, minlength=n)
    out[:, 1] = np.bincount(gi_c, weights=delta_c[:, 1] * factor, minlength=n)
    return out


def f_density(
    state: AgentState,
    spatial_hash: SpatialHash | None = None,
    *,
    radius: float = DENSITY_SENSING_RADIUS,
    threshold: float = HIGH_DENSITY_THRESHOLD,
    strength: float = DENSITY_PRESSURE_STRENGTH,
) -> Vec2Array:
    """Compute density-pressure deceleration in crowded regions (FR-2 R2.2).

    Local density is estimated per agent as the number of neighbours within
    ``radius`` divided by the disc area ``pi * radius**2``. Where density
    exceeds ``threshold``, a drag acceleration opposes the agent's velocity::

        a_i = -strength * (density_i - threshold) * v_i

    so an agent's *effective* speed drops as the crowd around it thickens;
    below the threshold the term is zero. This is the additive-force form of
    the fundamental-diagram speed reduction and, combined with repulsion,
    makes throughput fall as upstream density rises. Dead agents receive a
    zero row.

    Args:
        state: Current agent state (positions, velocities, liveness).
        spatial_hash: Optional pre-built neighbour index. If ``None``, one is
            built at ``radius``. If supplied, its cell size must be
            ``>= radius``.
        radius: Density-sensing radius in metres. Must be positive.
        threshold: Density (agents/m^2) above which pressure applies. Must be
            non-negative.
        strength: Drag scale. Must be non-negative.

    Returns:
        Float64 array of shape ``(N, 2)`` of deceleration in m/s². Dead agents
        have zero rows.

    Raises:
        ValueError: If ``radius`` is not positive, ``threshold`` or
            ``strength`` is negative, or a supplied hash is too coarse.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold!r}")
    if strength < 0.0:
        raise ValueError(f"strength must be non-negative, got {strength!r}")

    out: Vec2Array = np.zeros((state.count, 2), dtype=np.float64)
    sh = _resolve_hash(state, spatial_hash, radius)
    counts = sh.neighbour_counts(state, radius)
    if not np.any(counts):
        return out

    disc_area = math.pi * radius * radius
    density: npt.NDArray[np.float64] = counts / disc_area
    excess: npt.NDArray[np.float64] = np.maximum(density - threshold, 0.0)
    out = -strength * excess[:, np.newaxis] * state.vel
    return out


def _box_sum_3x3(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the 3x3 neighbourhood sum of a once-zero-padded 2D grid.

    Args:
        grid: Array of shape ``(gx, gy)`` carrying exactly one cell of zero
            padding on every side.

    Returns:
        Array of shape ``(gx - 2, gy - 2)`` where entry ``(i, j)`` is the sum
        of ``grid[i:i+3, j:j+3]`` — i.e. the total over the 3x3 block of
        *unpadded* cells centred on cell ``(i, j)``.
    """
    return (
        grid[0:-2, 0:-2] + grid[0:-2, 1:-1] + grid[0:-2, 2:]
        + grid[1:-1, 0:-2] + grid[1:-1, 1:-1] + grid[1:-1, 2:]
        + grid[2:, 0:-2] + grid[2:, 1:-1] + grid[2:, 2:]
    )


def f_herd(
    state: AgentState,
    *,
    radius: float = HERD_PERCEPTION_RADIUS,
    strength: float = HERD_ATTRACTION_STRENGTH,
) -> Vec2Array:
    """Compute panic-scaled herd alignment toward local mean velocity (R2.5).

    Each agent is drawn toward the mean velocity of the live agents in its
    neighbourhood, scaled by its own panic level::

        a_i = strength * panic_i * (mean_velocity_i - v_i)

    A calm agent (``panic == 0``) is unaffected; a panicked agent increasingly
    follows the surrounding crowd flow. Agents with no neighbours, and dead
    agents, receive a zero row.

    **Approximation (accuracy traded for speed).** The neighbourhood is the
    agent's own cell plus the eight adjacent cells of a uniform grid whose side
    equals ``radius`` — a 3x3 box covering ``[radius, 2*radius)`` in each
    direction rather than the exact disc of radius ``radius``. Velocity sums
    and counts are aggregated *per cell* and read back per agent, making the
    term ``O(N)`` instead of the ``O(N^2)`` pair enumeration an exact disc
    requires. This matters because the tuned ``radius`` (~15 m) is comparable
    to the floor span, so an exact query degenerates to all-pairs. Herd
    alignment is a diffuse, large-radius drift, so the square-neighbourhood
    approximation has negligible behavioural effect while removing the dominant
    per-tick cost. The result is fully deterministic (R6.2).

    Args:
        state: Current agent state (positions, velocities, panic, liveness).
        radius: Herd-perception grid cell side in metres. Must be positive.
        strength: Alignment scale. Must be non-negative.

    Returns:
        Float64 array of shape ``(N, 2)`` of alignment accelerations in m/s².
        Dead and isolated agents have zero rows.

    Raises:
        ValueError: If ``radius`` is not positive or ``strength`` is negative.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if strength < 0.0:
        raise ValueError(f"strength must be non-negative, got {strength!r}")

    out: Vec2Array = np.zeros((state.count, 2), dtype=np.float64)
    active = state.active_indices
    if active.size < 2:
        # Fewer than two live agents: no neighbour can exist.
        return out

    pos: Vec2Array = state.pos[active]
    vel: Vec2Array = state.vel[active]
    panic: npt.NDArray[np.float64] = state.panic[active]

    # Bin active agents into a uniform grid of cell side ``radius`` and shift so
    # the minimum cell index is 0; an agent's 3x3 cell block is then its
    # herd neighbourhood.
    inv_cell = 1.0 / radius
    cx: npt.NDArray[np.intp] = np.floor(pos[:, 0] * inv_cell).astype(np.intp)
    cy: npt.NDArray[np.intp] = np.floor(pos[:, 1] * inv_cell).astype(np.intp)
    cx -= cx.min()
    cy -= cy.min()
    n_cx = int(cx.max()) + 1
    n_cy = int(cy.max()) + 1

    # Accumulate per-cell velocity sums and counts, padding by one cell on every
    # side so the 3x3 box sum needs no bounds handling. Padded index = cell + 1.
    sum_vx = np.zeros((n_cx + 2, n_cy + 2), dtype=np.float64)
    sum_vy = np.zeros((n_cx + 2, n_cy + 2), dtype=np.float64)
    counts = np.zeros((n_cx + 2, n_cy + 2), dtype=np.float64)
    np.add.at(sum_vx, (cx + 1, cy + 1), vel[:, 0])
    np.add.at(sum_vy, (cx + 1, cy + 1), vel[:, 1])
    np.add.at(counts, (cx + 1, cy + 1), 1.0)

    box_vx = _box_sum_3x3(sum_vx)  # (n_cx, n_cy), indexed by (cx, cy)
    box_vy = _box_sum_3x3(sum_vy)
    box_cnt = _box_sum_3x3(counts)

    # Per-agent neighbourhood totals exclude the agent itself.
    neigh_cnt = box_cnt[cx, cy] - 1.0
    has = neigh_cnt > 0.0
    if not np.any(has):
        return out

    inv = 1.0 / neigh_cnt[has]
    mean0 = (box_vx[cx, cy][has] - vel[has, 0]) * inv
    mean1 = (box_vy[cx, cy][has] - vel[has, 1]) * inv
    scale = strength * panic[has]
    rows: Int1D = active[has]
    out[rows, 0] = scale * (mean0 - vel[has, 0])
    out[rows, 1] = scale * (mean1 - vel[has, 1])
    return out


def f_panic_repulsion(
    state: AgentState,
    panic_field: PanicField,
    *,
    strength: float = PANIC_REPULSION_STRENGTH,
) -> Vec2Array:
    """Compute panic-gradient repulsion for all agents (FR-11 R11.4).

    For each active agent within a panic source's influence radius, an
    acceleration is applied in the direction *away* from that source,
    proportional to the local panic field value::

        a_i = strength * sum_s[ v_s(p_i) * (p_i − p_s) / ||p_i − p_s|| ]

    where ``v_s(p_i)`` is the scalar field contribution from source ``s``
    at position ``p_i`` (zero outside the source radius) and the direction
    ``(p_i − p_s) / d`` points away from the source.

    Dead agents (``alive[i] == False``) receive a zero row.  Agents whose
    positions lie outside all active source radii also receive a zero row.

    Args:
        state: Current agent state (positions, liveness).
        panic_field: Aggregated panic field from one or more
            :class:`~crowd_evac.domain.panic_field.PanicField` sources.
            Inactive sources (intensity at/below threshold) contribute
            nothing.
        strength: Repulsion acceleration scale in m/s².  Must be >= 0.

    Returns:
        Float64 array of shape ``(N, 2)`` of repulsion accelerations in
        m/s².  Dead and out-of-range agents have zero rows.

    Raises:
        ValueError: If ``strength`` is negative.
    """
    if strength < 0.0:
        raise ValueError(
            f"strength must be non-negative, got {strength!r}"
        )

    out: Vec2Array = np.zeros((state.count, 2), dtype=np.float64)
    active = state.active_indices
    if active.size == 0:
        return out

    raw = panic_field.repulsion_at(state.pos[active])  # (A, 2)
    out[active] = strength * raw
    return out


def compose(
    state: AgentState,
    field: FlowField,
    panic_field: PanicField,
    *,
    params: ForceParams | None = None,
    spatial_hash: SpatialHash | None = None,
    enable_exit: bool = True,
    enable_crowd: bool = True,
    enable_density: bool = True,
    enable_herd: bool = True,
    enable_panic_repulsion: bool = True,
) -> Vec2Array:
    """Compose all enabled force terms into a single per-agent acceleration.

    Sums the active force contributions into one ``(N, 2)`` array ready for
    :func:`~crowd_evac.domain.integrator.step`::

        a_i = f_exit + f_crowd + f_density + f_herd + f_panic_repulsion

    Each term is independently togglable for debugging (R14.1 AC).  A single
    :class:`~crowd_evac.domain.spatial_hash.SpatialHash` is built — or reused
    from the caller — at the largest radius required by the enabled crowd
    terms, so repulsion, density, and herd share one index build per tick.

    When ``params`` is ``None`` (the default) every force term uses its own
    constant-derived defaults — behaviour is identical to the pre-Phase-2
    baseline.  When a :class:`~crowd_evac.domain.params.ForceParams` is
    supplied each field is forwarded to the corresponding term kwarg, enabling
    the Phase-2 optimiser to vary weights without mutating module globals.

    Phase 4 signage and other future additive terms slot in here without
    touching the integrator.

    Args:
        state: Current agent state (positions, velocities, panic, liveness).
        field: Pre-computed flow field for exit-seeking direction.
        panic_field: Aggregated panic field from active panic sources.
        params: Optional injectable force weights.  ``None`` uses the
            Phase-1 constant defaults (behaviour-preserving).
        spatial_hash: Optional pre-built neighbour index whose
            :attr:`~SpatialHash.cell_size` is at least the largest enabled
            crowd-term radius.  When ``None``, one is built automatically.
        enable_exit: Include the exit-seeking force (default ``True``).
        enable_crowd: Include agent-agent repulsion (default ``True``).
        enable_density: Include density-pressure deceleration (default ``True``).
        enable_herd: Include herd alignment toward local mean velocity
            (default ``True``).
        enable_panic_repulsion: Include panic-gradient repulsion away from
            hazard sources (default ``True``).

    Returns:
        Float64 array of shape ``(N, 2)`` with summed accelerations in m/s².
        Dead agents receive zero rows regardless of enabled terms.
    """
    _p: ForceParams = ForceParams.defaults() if params is None else params

    out: Vec2Array = np.zeros((state.count, 2), dtype=np.float64)

    if enable_exit:
        out = out + f_exit(
            state,
            field,
            _p.relaxation_time,
            max_speed=_p.max_speed,
            panic_speed_multiplier=_p.panic_speed_multiplier,
        )

    # Build one shared spatial hash for the pair-enumerating crowd terms
    # (repulsion, density) at the larger of their two radii.  Herd alignment
    # no longer enumerates pairs — it aggregates per grid cell internally — so
    # its large perception radius is deliberately excluded here: including it
    # would inflate the cell size and force a near-all-pairs candidate set.
    sh: SpatialHash | None = spatial_hash
    if sh is None:
        _needed: list[float] = []
        if enable_crowd:
            _needed.append(_p.repulsion_radius)
        if enable_density:
            _needed.append(_p.density_sensing_radius)
        if _needed:
            sh = SpatialHash.build(state, max(_needed))

    if enable_crowd:
        out = out + f_crowd(
            state, sh,
            radius=_p.repulsion_radius,
            strength=_p.repulsion_strength,
        )
    if enable_density:
        out = out + f_density(
            state, sh,
            radius=_p.density_sensing_radius,
            threshold=_p.high_density_threshold,
            strength=_p.density_pressure_strength,
        )
    if enable_herd:
        out = out + f_herd(
            state,
            radius=_p.herd_perception_radius,
            strength=_p.herd_attraction_strength,
        )

    if enable_panic_repulsion:
        out = out + f_panic_repulsion(
            state, panic_field, strength=_p.panic_repulsion_strength
        )

    return out
