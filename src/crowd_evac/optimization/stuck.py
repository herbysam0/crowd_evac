"""Stuck-agent detector: realism constraint and structural-bug canary (2.5).

Implements the user's added hard realism constraint — *no agent stays stuck
while a viable route to a live exit still exists* — as a scalar
:func:`stuck_count` the Phase-2 optimiser treats as a near-hard constraint
(``stuck_count <= 0``).

An agent is **stuck** when, over a sustained time window, *all* of the
following hold at every sampled tick in that window:

1. **Stalled** — its speed stays at or below :data:`STUCK_SPEED_EPS_MPS`.
2. **Has a route** — the flow field offers a non-zero descent direction toward
   a live exit at its position (the field is solved only from live exits, so a
   non-zero direction is, by construction, toward a live one), and its cell is
   reachable (finite integration-field cost).
3. **Not at an exit** — its path-distance to the nearest exit exceeds the exit
   capture radius, so it is *not* in the egress queue legitimately rate-limited
   by exit capacity.
4. **Not blocked ahead** — no wall and no other live agent sit within blocking
   range in its descent direction.  This is the discriminator that separates a
   genuine deadlock (the classic equal-and-opposite ``f_exit`` vs
   ``f_crowd`` / ``f_panic_repulsion`` cancellation, with clear space ahead)
   from a legitimately queued or wall-blocked agent.

Blocking range note: the agent no-overlap invariant keeps centres at least
``2 * AGENT_RADIUS`` (1.1 m) apart, which already exceeds ``REPULSION_RADIUS``
(0.5 m).  A literal "neighbour within ``repulsion_radius`` of my *centre*" test
would therefore never fire and would false-flag every queued agent.  The
neighbour test is consequently body-to-body: a neighbour blocks when its centre
is within ``2 * AGENT_RADIUS + repulsion_radius`` (surfaces within the
repulsion range).  See :func:`_blocking_range`.

Canary note (per ``docs/plan_phase_2.md`` Step 2.5): this detector directly
probes FR-4 routing/flow-field faults.  The original Phase-1 freeze — injected
hazards permanently blocking the flow field so engulfed agents lost all
exit-seeking direction — is exactly what it is built to catch.  If *no* weight
set in Step 2.9 can drive ``stuck_count`` to zero, the fault is in routing, not
the weights — surface it as a Phase-1 bug rather than masking it by detuning.

All inputs are :class:`~crowd_evac.optimization.harness.RunResult` arrays plus
a :class:`~crowd_evac.pathfinding.flow_field.FlowField`; the module is headless
and unit-testable against hand-built fixtures.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from crowd_evac.domain.constants import (
    AGENT_RADIUS,
    DT,
    EXIT_CAPTURE_RADIUS,
    REPULSION_RADIUS,
)
from crowd_evac.optimization.harness import RunResult
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Detection tuning constants
# ---------------------------------------------------------------------------

STUCK_SPEED_EPS_MPS: float = 0.05
"""Speed (m/s) at or below which an agent is treated as stalled.

Matches the moving-speed floor used by the realism extractors
(:data:`~crowd_evac.optimization.realism.MOVING_SPEED_EPS_MPS`): below this an
agent is, for evacuation purposes, not making progress.
"""

STUCK_MIN_STALL_S: float = 2.0
"""Minimum sustained stall duration (s) before an agent counts as stuck.

A momentary stop (one agent yielding for a tick, a brief jostle) is normal
crowd behaviour; only a stall held continuously for this long with a clear
path is a deadlock.  Deliberately shorter than the global
:data:`~crowd_evac.domain.constants.STALL_TICKS` (10 s) so a *single* trapped
agent is caught well before the whole evacuation is declared stalled.
"""

STUCK_WALL_PROBE_M: float = AGENT_RADIUS
"""Distance (m) ahead of an agent at which an obstructing wall is probed.

One body radius ahead along the descent direction: far enough to detect a wall
the agent would walk into, short enough not to overshoot a nearby live exit.
"""

_DIR_EPS: float = 1e-9
"""Norm below which a flow-field direction is treated as zero (at an exit)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stuck_agents(
    result: RunResult,
    flow_field: FlowField,
    *,
    speed_eps: float = STUCK_SPEED_EPS_MPS,
    min_stall_s: float = STUCK_MIN_STALL_S,
    repulsion_radius: float = REPULSION_RADIUS,
    probe_distance_m: float = STUCK_WALL_PROBE_M,
    exit_capture_radius: float = EXIT_CAPTURE_RADIUS,
) -> npt.NDArray[np.bool_]:
    """Return a per-agent boolean mask of agents stuck at any point in the run.

    An agent is flagged when it is *deadlocked* (stalled, with a clear routed
    path to a live exit and nothing blocking it ahead — see the module
    docstring) continuously across a sustained window of sampled ticks.

    Args:
        result: The headless run whose sampled history is analysed.
        flow_field: Navigation field reflecting the run's live exits; supplies
            the descent direction and integration-field cost at each position.
        speed_eps: Stall speed floor (m/s). Must be > 0.
        min_stall_s: Minimum sustained stall duration (s) to count as stuck.
            Must be > 0.
        repulsion_radius: Agent-agent repulsion radius (m) used to derive the
            body-to-body blocking range. Must be > 0.
        probe_distance_m: Distance ahead (m) at which an obstructing wall is
            probed. Must be > 0.
        exit_capture_radius: Path-distance to an exit (m) below which an agent
            is treated as a legitimate egress-queue member, not stuck. Must be
            >= 0.

    Returns:
        Boolean array of shape ``(result.initial_count,)``; ``True`` where the
        agent was stuck for at least one sustained window.

    Raises:
        ValueError: If any tuning argument is out of range.
    """
    _validate(
        speed_eps, min_stall_s, repulsion_radius,
        probe_distance_m, exit_capture_radius,
    )
    n = result.initial_count
    if n == 0 or len(result.sample_ticks) == 0:
        return np.zeros(n, dtype=np.bool_)

    blocking_range = _blocking_range(repulsion_radius)
    deadlocked = _deadlocked_matrix(
        result, flow_field, speed_eps,
        blocking_range, probe_distance_m, exit_capture_radius,
    )
    window = _window_samples(result.sample_ticks, min_stall_s)
    return _sustained(deadlocked, window)


def stuck_count(
    result: RunResult,
    flow_field: FlowField,
    *,
    speed_eps: float = STUCK_SPEED_EPS_MPS,
    min_stall_s: float = STUCK_MIN_STALL_S,
    repulsion_radius: float = REPULSION_RADIUS,
    probe_distance_m: float = STUCK_WALL_PROBE_M,
    exit_capture_radius: float = EXIT_CAPTURE_RADIUS,
) -> int:
    """Return the number of agents stuck at any point in the run.

    Thin wrapper over :func:`stuck_agents`; see it for the full criterion and
    argument semantics.

    Args:
        result: The headless run to analyse.
        flow_field: Navigation field reflecting the run's live exits.
        speed_eps: Stall speed floor (m/s). Must be > 0.
        min_stall_s: Minimum sustained stall duration (s). Must be > 0.
        repulsion_radius: Agent-agent repulsion radius (m). Must be > 0.
        probe_distance_m: Wall-probe distance ahead (m). Must be > 0.
        exit_capture_radius: Egress-queue exclusion radius (m). Must be >= 0.

    Returns:
        Count of distinct agents flagged as stuck (``>= 0``).

    Raises:
        ValueError: If any tuning argument is out of range.
    """
    mask = stuck_agents(
        result, flow_field,
        speed_eps=speed_eps, min_stall_s=min_stall_s,
        repulsion_radius=repulsion_radius, probe_distance_m=probe_distance_m,
        exit_capture_radius=exit_capture_radius,
    )
    return int(np.count_nonzero(mask))


def has_stuck(
    result: RunResult,
    flow_field: FlowField,
    *,
    speed_eps: float = STUCK_SPEED_EPS_MPS,
    min_stall_s: float = STUCK_MIN_STALL_S,
    repulsion_radius: float = REPULSION_RADIUS,
    probe_distance_m: float = STUCK_WALL_PROBE_M,
    exit_capture_radius: float = EXIT_CAPTURE_RADIUS,
) -> bool:
    """Return whether any agent was stuck during the run.

    Boolean form of the :func:`stuck_count` ``> 0`` constraint, suited to the
    composite-fitness constraint vector (Step 2.6).

    Args:
        result: The headless run to analyse.
        flow_field: Navigation field reflecting the run's live exits.
        speed_eps: Stall speed floor (m/s). Must be > 0.
        min_stall_s: Minimum sustained stall duration (s). Must be > 0.
        repulsion_radius: Agent-agent repulsion radius (m). Must be > 0.
        probe_distance_m: Wall-probe distance ahead (m). Must be > 0.
        exit_capture_radius: Egress-queue exclusion radius (m). Must be >= 0.

    Returns:
        ``True`` if at least one agent was stuck, else ``False``.

    Raises:
        ValueError: If any tuning argument is out of range.
    """
    mask = stuck_agents(
        result, flow_field,
        speed_eps=speed_eps, min_stall_s=min_stall_s,
        repulsion_radius=repulsion_radius, probe_distance_m=probe_distance_m,
        exit_capture_radius=exit_capture_radius,
    )
    return bool(np.any(mask))


# ---------------------------------------------------------------------------
# Per-sample deadlock detection
# ---------------------------------------------------------------------------


def _deadlocked_matrix(
    result: RunResult,
    flow_field: FlowField,
    speed_eps: float,
    blocking_range: float,
    probe_distance: float,
    exit_capture_radius: float,
) -> npt.NDArray[np.bool_]:
    """Build the ``(T, N)`` per-sample, per-agent deadlock boolean matrix.

    Entry ``[k, i]`` is ``True`` when agent ``i`` satisfies every instantaneous
    deadlock condition at sample ``k`` (see the module docstring).  The
    sustained-window test is applied separately by :func:`_sustained`.

    Args:
        result: Run supplying the sampled position/velocity/liveness history.
        flow_field: Field giving descent direction and cost at each position.
        speed_eps: Stall speed floor (m/s).
        blocking_range: Centre-distance (m) within which a forward neighbour
            counts as blocking.
        probe_distance: Distance ahead (m) at which a wall is probed.
        exit_capture_radius: Cost (m) below which an agent is treated as
            queued at an exit, not deadlocked.

    Returns:
        Boolean matrix of shape ``(len(sample_ticks), initial_count)``.
    """
    speeds = np.linalg.norm(result.velocities_history, axis=-1)  # (T, N)
    n_samples = speeds.shape[0]
    out = np.zeros((n_samples, result.initial_count), dtype=np.bool_)
    for k in range(n_samples):
        pos = result.positions_history[k]
        alive = result.alive_history[k]
        stalled = alive & (speeds[k] <= speed_eps)
        if not bool(np.any(stalled)):
            continue
        dirs = flow_field.sample(pos)
        has_route = np.linalg.norm(dirs, axis=1) > _DIR_EPS
        cost = _cell_cost(flow_field, pos)
        routed = has_route & np.isfinite(cost) & (cost >= exit_capture_radius)
        neighbour = _blocked_by_neighbour(pos, dirs, alive, blocking_range)
        wall = _blocked_by_wall(flow_field, pos, dirs, probe_distance)
        out[k] = stalled & routed & ~neighbour & ~wall
    return out


def _blocked_by_neighbour(
    positions: npt.NDArray[np.float64],
    directions: npt.NDArray[np.float64],
    alive: npt.NDArray[np.bool_],
    blocking_range: float,
) -> npt.NDArray[np.bool_]:
    """Return a per-agent mask: a live neighbour blocks the descent direction.

    A neighbour blocks agent ``i`` when its centre is within ``blocking_range``
    of ``i`` *and* lies in ``i``'s forward half-plane (positive projection onto
    ``i``'s descent direction).

    Args:
        positions: Agent positions at one sample, shape ``(N, 2)``.
        directions: Per-agent flow-field descent directions, shape ``(N, 2)``.
        alive: Per-agent liveness flags, shape ``(N,)``.
        blocking_range: Centre-distance cut-off (m).

    Returns:
        Boolean array of shape ``(N,)``.
    """
    n = positions.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.bool_)
    # offset[i, j] = position(j) - position(i): vector from i toward j.
    offset = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(offset, axis=2)  # (N, N)
    within = (dist > _DIR_EPS) & (dist < blocking_range) & alive[np.newaxis, :]
    forward = np.einsum("ijk,ik->ij", offset, directions) > 0.0
    blocked: npt.NDArray[np.bool_] = np.any(within & forward, axis=1)
    return blocked


def _blocked_by_wall(
    flow_field: FlowField,
    positions: npt.NDArray[np.float64],
    directions: npt.NDArray[np.float64],
    probe_distance: float,
) -> npt.NDArray[np.bool_]:
    """Return a per-agent mask: a wall lies one probe step ahead.

    Probes the integration field one ``probe_distance`` step along each agent's
    descent direction; an infinite (blocked or out-of-grid) cost there means a
    wall obstructs the path.

    Args:
        flow_field: Field whose cost grid is probed.
        positions: Agent positions at one sample, shape ``(N, 2)``.
        directions: Per-agent descent directions, shape ``(N, 2)``.
        probe_distance: Step length ahead (m).

    Returns:
        Boolean array of shape ``(N,)``; ``True`` where a wall is ahead.
    """
    probe = positions + directions * probe_distance
    return ~np.isfinite(_cell_cost(flow_field, probe))


def _cell_cost(
    flow_field: FlowField,
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Gather the integration-field cost at each world point.

    Points outside the grid map to ``inf`` (treated as blocked/unreachable).

    Args:
        flow_field: Field supplying the cost grid and cell size.
        points: World ``(x, y)`` coordinates, shape ``(M, 2)``.

    Returns:
        Float64 array of shape ``(M,)`` with per-point cost; ``inf`` where the
        point falls outside the grid.
    """
    cell_size = flow_field.cell_size
    cost = flow_field.cost
    rows, cols = cost.shape
    col = np.floor(points[:, 0] / cell_size).astype(np.intp)
    row = np.floor(points[:, 1] / cell_size).astype(np.intp)
    in_bounds = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    out = np.full(points.shape[0], np.inf, dtype=np.float64)
    rr = np.clip(row, 0, rows - 1)
    cc = np.clip(col, 0, cols - 1)
    out[in_bounds] = cost[rr[in_bounds], cc[in_bounds]]
    return out


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------


def _sustained(
    matrix: npt.NDArray[np.bool_],
    window: int,
) -> npt.NDArray[np.bool_]:
    """Reduce a ``(T, N)`` per-sample matrix to a per-agent sustained mask.

    An agent is flagged when some run of ``window`` consecutive samples is
    entirely ``True`` for it.

    Args:
        matrix: Per-sample, per-agent boolean matrix, shape ``(T, N)``.
        window: Required number of consecutive ``True`` samples (``>= 1``).

    Returns:
        Boolean array of shape ``(N,)``.  All ``False`` when ``T < window``.
    """
    n_samples, n_agents = matrix.shape
    if window <= 0 or n_samples < window:
        return np.zeros(n_agents, dtype=np.bool_)
    windows = sliding_window_view(matrix, window, axis=0)  # (T-w+1, N, w)
    flagged: npt.NDArray[np.bool_] = np.any(np.all(windows, axis=2), axis=0)
    return flagged


def _window_samples(
    sample_ticks: tuple[int, ...],
    min_stall_s: float,
) -> int:
    """Convert a minimum stall duration to a count of consecutive samples.

    Derives the inter-sample spacing from the sampled tick indices (their
    median gap) and the fixed timestep :data:`~crowd_evac.domain.constants.DT`,
    so the physical ``min_stall_s`` maps onto the run's actual sampling rate.

    Args:
        sample_ticks: Tick indices at which state was captured.
        min_stall_s: Minimum sustained stall duration (s).

    Returns:
        Number of consecutive samples spanning at least ``min_stall_s``
        (``>= 1``).
    """
    ticks = np.asarray(sample_ticks, dtype=np.float64)
    if ticks.size >= 2:
        spacing = float(np.median(np.diff(ticks)))
    else:
        spacing = 1.0
    spacing = max(spacing, 1.0)
    sample_dt = spacing * DT
    return max(1, int(np.ceil(min_stall_s / sample_dt)))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _blocking_range(repulsion_radius: float) -> float:
    """Return the centre-distance (m) within which a neighbour blocks.

    Two agents touch at a centre distance of ``2 * AGENT_RADIUS`` (the
    no-overlap invariant), so a neighbour whose *surface* is within
    ``repulsion_radius`` sits at a centre distance up to
    ``2 * AGENT_RADIUS + repulsion_radius``.

    Args:
        repulsion_radius: Agent-agent repulsion radius (m).

    Returns:
        Centre-distance blocking threshold (m).
    """
    return 2.0 * AGENT_RADIUS + repulsion_radius


def _validate(
    speed_eps: float,
    min_stall_s: float,
    repulsion_radius: float,
    probe_distance_m: float,
    exit_capture_radius: float,
) -> None:
    """Validate the public tuning arguments; raise on any out-of-range value.

    Args:
        speed_eps: Stall speed floor (m/s). Must be > 0.
        min_stall_s: Minimum sustained stall duration (s). Must be > 0.
        repulsion_radius: Agent-agent repulsion radius (m). Must be > 0.
        probe_distance_m: Wall-probe distance ahead (m). Must be > 0.
        exit_capture_radius: Egress-queue exclusion radius (m). Must be >= 0.

    Raises:
        ValueError: If any argument violates its stated bound.
    """
    if speed_eps <= 0.0:
        raise ValueError(f"speed_eps must be > 0, got {speed_eps!r}")
    if min_stall_s <= 0.0:
        raise ValueError(f"min_stall_s must be > 0, got {min_stall_s!r}")
    if repulsion_radius <= 0.0:
        raise ValueError(
            f"repulsion_radius must be > 0, got {repulsion_radius!r}"
        )
    if probe_distance_m <= 0.0:
        raise ValueError(
            f"probe_distance_m must be > 0, got {probe_distance_m!r}"
        )
    if exit_capture_radius < 0.0:
        raise ValueError(
            f"exit_capture_radius must be >= 0, got {exit_capture_radius!r}"
        )
