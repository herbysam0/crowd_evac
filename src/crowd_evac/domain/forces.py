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

The crowd terms find neighbours through
:class:`~crowd_evac.domain.spatial_hash.SpatialHash`; a caller may pass a
pre-built hash (so the terms share one build per tick) or let each term build
its own at the appropriate radius.

Further terms — ``f_panic_repulsion``, ``compose`` — will be added in steps
1.10–1.11 to this same module without changing the integrator.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.agent_state import AgentState, Vec2Array
from crowd_evac.domain.constants import (
    DENSITY_PRESSURE_STRENGTH,
    DENSITY_SENSING_RADIUS,
    HERD_ATTRACTION_STRENGTH,
    HERD_PERCEPTION_RADIUS,
    HIGH_DENSITY_THRESHOLD,
    MAX_SPEED,
    PANIC_SPEED_MULTIPLIER,
    RELAXATION_TIME,
    REPULSION_MIN_DISTANCE,
    REPULSION_RADIUS,
    REPULSION_STRENGTH,
)
from crowd_evac.domain.spatial_hash import SpatialHash
from crowd_evac.pathfinding.flow_field import FlowField


def f_exit(
    state: AgentState,
    field: FlowField,
    relaxation_time: float = RELAXATION_TIME,
) -> Vec2Array:
    """Compute exit-seeking acceleration for all agents (FR-1 R1.2).

    Samples the flow field at each active agent's world position to get the
    desired exit-seeking direction, scales it by the agent's panic-modulated
    desired speed, and returns the social-force steering acceleration::

        a_i = (v_desired_i * direction_i − v_i) / relaxation_time

    where::

        v_desired_i = MAX_SPEED * (1 + panic_i * (PANIC_SPEED_MULTIPLIER − 1))

    Dead agents (``alive[i] == False``) receive a zero row in the output.

    Args:
        state: Current agent state (positions, velocities, panic levels).
        field: Pre-computed flow field providing exit-seeking unit directions.
        relaxation_time: Characteristic time in seconds for velocity
            relaxation. Must be positive.

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

    # Linearly interpolate desired speed between MAX_SPEED (no panic) and
    # MAX_SPEED * PANIC_SPEED_MULTIPLIER (full panic).
    desired_speed: npt.NDArray[np.float64] = MAX_SPEED * (
        1.0 + state.panic[active] * (PANIC_SPEED_MULTIPLIER - 1.0)
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
    gi, gj = sh.query_pairs()
    if gi.size == 0:
        return out

    delta: Vec2Array = state.pos[gi] - state.pos[gj]  # points j -> i
    dist: npt.NDArray[np.float64] = np.linalg.norm(delta, axis=1)
    close = (dist < radius) & (dist > 0.0)
    if not np.any(close):
        return out

    gi_c = gi[close]
    delta_c = delta[close]
    dist_c = dist[close]
    d_clamped = np.maximum(dist_c, min_distance)
    magnitude = strength * (radius - dist_c) / d_clamped  # (P,)
    direction = delta_c / dist_c[:, np.newaxis]           # unit j -> i
    contrib: Vec2Array = direction * magnitude[:, np.newaxis]
    np.add.at(out, gi_c, contrib)
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


def f_herd(
    state: AgentState,
    spatial_hash: SpatialHash | None = None,
    *,
    radius: float = HERD_PERCEPTION_RADIUS,
    strength: float = HERD_ATTRACTION_STRENGTH,
) -> Vec2Array:
    """Compute panic-scaled herd alignment toward local mean velocity (R2.5).

    Each agent is drawn toward the mean velocity of all live agents within
    ``radius``, scaled by its own panic level::

        a_i = strength * panic_i * (mean_velocity_i - v_i)

    A calm agent (``panic == 0``) is unaffected; a panicked agent increasingly
    follows the surrounding crowd flow. Agents with no neighbours, and dead
    agents, receive a zero row.

    Args:
        state: Current agent state (velocities, panic, liveness).
        spatial_hash: Optional pre-built neighbour index. If ``None``, one is
            built at ``radius``. If supplied, its cell size must be
            ``>= radius``.
        radius: Herd-perception radius in metres. Must be positive.
        strength: Alignment scale. Must be non-negative.

    Returns:
        Float64 array of shape ``(N, 2)`` of alignment accelerations in m/s².
        Dead and isolated agents have zero rows.

    Raises:
        ValueError: If ``radius`` is not positive, ``strength`` is negative,
            or a supplied hash is too coarse.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if strength < 0.0:
        raise ValueError(f"strength must be non-negative, got {strength!r}")

    out: Vec2Array = np.zeros((state.count, 2), dtype=np.float64)
    sh = _resolve_hash(state, spatial_hash, radius)
    gi, gj = sh.query_pairs()
    if gi.size == 0:
        return out

    dist: npt.NDArray[np.float64] = np.linalg.norm(
        state.pos[gi] - state.pos[gj], axis=1
    )
    within = dist < radius
    if not np.any(within):
        return out

    gi_w = gi[within]
    gj_w = gj[within]
    vel_sum: Vec2Array = np.zeros((state.count, 2), dtype=np.float64)
    np.add.at(vel_sum, gi_w, state.vel[gj_w])
    counts: npt.NDArray[np.intp] = np.zeros(state.count, dtype=np.intp)
    np.add.at(counts, gi_w, 1)

    has_neighbours = counts > 0
    mean_vel: Vec2Array = (
        vel_sum[has_neighbours] / counts[has_neighbours][:, np.newaxis]
    )
    panic_h = state.panic[has_neighbours][:, np.newaxis]
    out[has_neighbours] = (
        strength * panic_h * (mean_vel - state.vel[has_neighbours])
    )
    return out
