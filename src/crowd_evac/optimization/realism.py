"""Realism metric: empirical reference targets + fundamental-diagram distance.

Turns the qualitative goal "the simulated crowd behaves like a real evacuation"
into a single scalar :func:`realism_distance` that the Phase-2 optimiser
minimises.  The distance is a weighted sum of four normalised components, each
anchored to a cited pedestrian-dynamics reference:

1. **Free-walking speed** — the cruising speed of uncongested agents should sit
   in the empirical 1.2-1.4 m/s band (Weidmann 1993; Fruin 1971).
2. **Bottleneck specific flow** — steady-state throughput per metre of effective
   width should sit near 1.2-1.3 persons/(m·s) (Seyfried et al. 2005; SFPE).
3. **Speed-density fundamental diagram** — the emergent speed-vs-density relation
   should follow the Weidmann (1993) curve *shape* (level is captured by
   component 1, so the FD term is normalised by the *measured* free speed and
   tests shape only, avoiding double-counting).
4. **Emergent-behaviour fidelity** — a realistic bottleneck evacuation must form
   transient congestion; a run that never reaches a plausible jam density is the
   frictionless degenerate the Phase-2 plan warns against (the global
   time-optimum and maximally unrealistic), and is penalised here.

Every reference constant carries a source comment and is range-checked by the
unit tests.  All extractors operate purely on :class:`~crowd_evac.optimization.
harness.RunResult` arrays, so the whole module is headless and unit-testable
against hand-built synthetic runs.

Risk flag (per ``docs/plan_phase_2.md`` Step 2.4): this metric *is* the
calibration — a wrong reference set or component weighting silently yields a
confidently-wrong default.  The component weights below
(:data:`W_SPEED`, :data:`W_FLOW`, :data:`W_FD`, :data:`W_EB`) and the reference
bands must be reviewed before Step 2.8 commits compute to them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crowd_evac.optimization.harness import RunResult

# ---------------------------------------------------------------------------
# Empirical reference targets (each carries a literature source)
# ---------------------------------------------------------------------------

FREE_WALK_SPEED_TARGET_MPS: float = 1.34
"""Mean free-walking speed on the level (Weidmann 1993, ~1.34 m/s)."""

FREE_WALK_SPEED_BAND_MPS: tuple[float, float] = (1.20, 1.40)
"""Accepted free-walking-speed band (Fruin 1971; Daamen & Hoogendoorn 2006).

Distance is zero inside this band and rises linearly outside it, reflecting
genuine spread in reported means rather than a single point estimate.
"""

SPECIFIC_FLOW_TARGET_PMS: float = 1.25
"""Target bottleneck specific flow in persons/(m·s) (band midpoint)."""

SPECIFIC_FLOW_BAND_PMS: tuple[float, float] = (1.20, 1.30)
"""Accepted bottleneck specific-flow band, persons/(m·s).

Flat-bottleneck steady-state values cluster near 1.2-1.3 P/(m·s)
(Seyfried et al. 2005; Kretz et al. 2006; SFPE Handbook, 5th ed.).
"""

WEIDMANN_FREE_SPEED_MPS: float = 1.34
"""``v0`` in the Weidmann (1993) walking fundamental diagram."""

WEIDMANN_JAM_DENSITY_M2: float = 5.4
"""Jam density ``rho_max`` (agents/m²) at which Weidmann speed reaches zero."""

WEIDMANN_GAMMA: float = 1.913
"""Shape coefficient ``gamma`` of the Weidmann (1993) speed-density curve."""

# ---------------------------------------------------------------------------
# Component weights (REVIEW BEFORE STEP 2.8 — see module risk flag)
# ---------------------------------------------------------------------------

W_SPEED: float = 0.30
"""Weight of the free-walking-speed component in the composite distance."""

W_FLOW: float = 0.30
"""Weight of the bottleneck-specific-flow component."""

W_FD: float = 0.25
"""Weight of the speed-density fundamental-diagram-shape component."""

W_EB: float = 0.15
"""Weight of the emergent-behaviour (congestion-formation) component."""

# ---------------------------------------------------------------------------
# Extraction tuning constants
# ---------------------------------------------------------------------------

FREE_WALK_PERCENTILE: float = 85.0
"""Percentile of moving-agent speeds taken as the free-walking estimate.

Congested agents are slow and free walkers are fast, so a high percentile of
the moving-agent speed distribution isolates the free-walking cohort without
needing a per-agent local-density estimate.
"""

MOVING_SPEED_EPS_MPS: float = 0.05
"""Speed floor (m/s) below which an agent is treated as stationary, not walking."""

FD_DENSITY_FLOOR_M2: float = 0.2
"""Density floor (agents/m²) below which a sample is too sparse for the FD term."""

EB_CONGESTION_FLOOR_M2: float = 0.8
"""Peak density (agents/m²) a realistic bottleneck run must transiently reach.

Below this floor no jam ever forms — the frictionless degenerate — so the
emergent-behaviour component is penalised in proportion to the shortfall.

Scaled to the model's packing limit, *not* the Weidmann jam density
(:data:`WEIDMANN_JAM_DENSITY_M2` = 5.4 /m²): with ``AGENT_RADIUS`` = 0.55 m the
hard no-overlap invariant forces a ≥ 1.1 m centre spacing, capping reachable
local density near ~1.0 /m² on the metrics 1 m grid.  A 0.8 /m² floor cleanly
separates a formed queue from free flow (~0.1-0.4 /m²) while staying physically
attainable.  If a future Phase-1 change shrinks ``AGENT_RADIUS`` toward
crowd-dynamics values, raise this floor toward the Weidmann jam density.
"""

_EPS: float = 1e-9
"""Numerical guard against division by zero."""


# ---------------------------------------------------------------------------
# Input bundle and diagnostic report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRunSet:
    """The headless runs a single realism evaluation consumes.

    Attributes:
        flow_results: Runs of the search-suite scenarios, used to estimate
            free-walking speed (the least-congested run dominates). Must be
            non-empty.
        bottleneck_result: Run of the controlled bottleneck micro-rig, used
            for specific flow, the speed-density trace, and the congestion
            (EB) signature.
        bottleneck_door_width_m: Effective width (m) of the controlling
            bottleneck opening, used to normalise egress to specific flow.
            Must be positive.
    """

    flow_results: tuple[RunResult, ...]
    bottleneck_result: RunResult
    bottleneck_door_width_m: float


@dataclass(frozen=True)
class RealismReport:
    """Per-component breakdown of one realism evaluation.

    Exposes both the raw extracted statistics and the normalised per-component
    distances so callers (and tests) can attribute the composite
    :attr:`distance` to its drivers.

    Attributes:
        free_walk_speed_mps: Estimated free-walking speed (m/s).
        specific_flow_pms: Estimated bottleneck specific flow, persons/(m·s).
        peak_density_m2: Peak density observed in the bottleneck run (m⁻²).
        speed_distance: Normalised free-walking-speed component distance (≥ 0).
        flow_distance: Normalised specific-flow component distance (≥ 0).
        fd_distance: Normalised FD-shape component distance (≥ 0).
        eb_penalty: Normalised congestion-formation penalty (≥ 0).
        distance: Weighted composite distance; 0 means a perfect match.
    """

    free_walk_speed_mps: float
    specific_flow_pms: float
    peak_density_m2: float
    speed_distance: float
    flow_distance: float
    fd_distance: float
    eb_penalty: float
    distance: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def realism_distance(runs: CalibrationRunSet) -> float:
    """Return the scalar realism distance for a set of calibration runs.

    Thin wrapper over :func:`realism_report`; returns only the composite
    :attr:`RealismReport.distance`.

    Args:
        runs: The flow and bottleneck runs to score.

    Returns:
        Weighted composite distance in ``[0, ∞)``. ``0`` means every component
        matches its empirical reference within tolerance.

    Raises:
        ValueError: If ``runs.flow_results`` is empty or
            ``runs.bottleneck_door_width_m`` is not positive.
    """
    return realism_report(runs).distance


def realism_report(runs: CalibrationRunSet) -> RealismReport:
    """Compute the full per-component realism breakdown.

    Args:
        runs: The flow and bottleneck runs to score.

    Returns:
        A :class:`RealismReport` with raw statistics, per-component distances,
        and the weighted composite distance.

    Raises:
        ValueError: If ``runs.flow_results`` is empty or
            ``runs.bottleneck_door_width_m`` is not positive.
    """
    if not runs.flow_results:
        raise ValueError("flow_results must be non-empty")
    if runs.bottleneck_door_width_m <= 0.0:
        raise ValueError(
            "bottleneck_door_width_m must be > 0, got "
            f"{runs.bottleneck_door_width_m!r}"
        )

    # Free-walking speed: the least-congested run gives the best estimate, so
    # take the maximum free-speed estimate across the flow scenarios.
    v_free = max(free_walking_speed(r) for r in runs.flow_results)

    q_s = specific_flow(runs.bottleneck_result, runs.bottleneck_door_width_m)
    density, speed = speed_density_trace(runs.bottleneck_result)
    peak_density = peak_run_density(runs.bottleneck_result)

    speed_d = _band_distance(v_free, *FREE_WALK_SPEED_BAND_MPS)
    flow_d = _band_distance(q_s, *SPECIFIC_FLOW_BAND_PMS)
    fd_d = _fd_distance(density, speed, v_free)
    eb_d = _eb_penalty(peak_density)

    distance = (
        W_SPEED * speed_d
        + W_FLOW * flow_d
        + W_FD * fd_d
        + W_EB * eb_d
    )

    return RealismReport(
        free_walk_speed_mps=v_free,
        specific_flow_pms=q_s,
        peak_density_m2=peak_density,
        speed_distance=speed_d,
        flow_distance=flow_d,
        fd_distance=fd_d,
        eb_penalty=eb_d,
        distance=distance,
    )


def free_walking_speed(
    result: RunResult,
    *,
    percentile: float = FREE_WALK_PERCENTILE,
    moving_eps: float = MOVING_SPEED_EPS_MPS,
) -> float:
    """Estimate the free-walking speed from a run's velocity history.

    Computes per-agent speeds over every sampled tick, keeps only *alive,
    moving* agents (speed above ``moving_eps``), and returns the given
    percentile of that distribution.  A high percentile isolates the
    free-walking cohort because congested agents move slowly.

    Args:
        result: The run whose velocity history is analysed.
        percentile: Percentile of the moving-speed distribution to return,
            in ``[0, 100]``.
        moving_eps: Speed (m/s) below which an agent is treated as stationary.

    Returns:
        Free-walking speed estimate in m/s; ``0.0`` if no agent ever moved.
    """
    if result.velocities_history.size == 0:
        return 0.0
    speeds = np.linalg.norm(result.velocities_history, axis=-1)  # (T, N)
    moving = result.alive_history & (speeds > moving_eps)
    sample = speeds[moving]
    if sample.size == 0:
        return 0.0
    return float(np.percentile(sample, percentile))


def specific_flow(result: RunResult, door_width_m: float) -> float:
    """Estimate bottleneck specific flow in persons/(m·s).

    Uses the mean-flow definition ``q = N / (T · w)`` where ``N`` is the total
    number of agents egressed, ``T`` is the time of the last egress, and ``w``
    is the controlling door width.  This averages over the active egress
    period (a coarse but standard steady-state proxy).

    Args:
        result: The bottleneck-rig run.
        door_width_m: Effective controlling-opening width (m). Must be > 0.

    Returns:
        Specific flow in persons/(m·s); ``0.0`` if nothing egressed or the
        egress window has zero duration.

    Raises:
        ValueError: If ``door_width_m`` is not positive.
    """
    if door_width_m <= 0.0:
        raise ValueError(f"door_width_m must be > 0, got {door_width_m!r}")
    egressed = sum(result.throughput_series)
    if egressed <= 0 or result.evac_time <= _EPS:
        return 0.0
    return float(egressed) / (result.evac_time * door_width_m)


def speed_density_trace(
    result: RunResult,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract paired (density, mean-speed) samples for the fundamental diagram.

    For each sampled tick (after tick 0) pairs the peak local density recorded
    that tick with the mean speed of the alive, moving agents.  As the
    bottleneck run progresses density rises into a jam and falls again, so the
    samples trace out an empirical speed-density relation.

    Args:
        result: The bottleneck-rig run.

    Returns:
        A ``(density, speed)`` tuple of equal-length float arrays, one entry
        per usable sampled tick.  Both are empty when no tick yields a moving
        agent paired with a recorded density.
    """
    n_ticks = len(result.density_series)
    densities: list[float] = []
    speeds: list[float] = []
    vel_hist = result.velocities_history
    for k, tick in enumerate(result.sample_ticks):
        if tick <= 0 or tick - 1 >= n_ticks or k >= vel_hist.shape[0]:
            continue
        speed = _mean_moving_speed(vel_hist[k], result.alive_history[k])
        if speed is None:
            continue
        densities.append(result.density_series[tick - 1])
        speeds.append(speed)
    return (
        np.asarray(densities, dtype=np.float64),
        np.asarray(speeds, dtype=np.float64),
    )


def peak_run_density(result: RunResult) -> float:
    """Return the peak local density (agents/m²) observed over the whole run.

    Args:
        result: The run whose per-tick density series is scanned.

    Returns:
        Maximum value of the density series; ``0.0`` if the series is empty.
    """
    if not result.density_series:
        return 0.0
    return float(max(result.density_series))


# ---------------------------------------------------------------------------
# Component-distance helpers
# ---------------------------------------------------------------------------


def _band_distance(value: float, lo: float, hi: float) -> float:
    """Return the normalised distance of a value from an accepted band.

    Zero inside ``[lo, hi]``; outside, the shortfall/excess normalised by the
    band midpoint, giving a relative error that rises monotonically away from
    the band.

    Args:
        value: Measured statistic.
        lo: Lower edge of the accepted band.
        hi: Upper edge of the accepted band.

    Returns:
        Normalised distance in ``[0, ∞)``.
    """
    scale = max(0.5 * (lo + hi), _EPS)
    if value < lo:
        return (lo - value) / scale
    if value > hi:
        return (value - hi) / scale
    return 0.0


def _weidmann_speed(
    density: npt.NDArray[np.float64], v_free: float
) -> npt.NDArray[np.float64]:
    """Evaluate the Weidmann (1993) speed-density curve at given densities.

    ``v(rho) = v_free · [1 - exp(-gamma · (1/rho - 1/rho_jam))]``, clamped at
    zero for densities at or beyond the jam density.

    Args:
        density: Strictly-positive densities (agents/m²).
        v_free: Free-walking speed used as the curve's ``v0`` level.

    Returns:
        Predicted speeds (m/s), element-wise, clamped to be non-negative.
    """
    inv = 1.0 / density - 1.0 / WEIDMANN_JAM_DENSITY_M2
    predicted = v_free * (1.0 - np.exp(-WEIDMANN_GAMMA * inv))
    return np.clip(predicted, 0.0, None)


def _fd_distance(
    density: npt.NDArray[np.float64],
    speed: npt.NDArray[np.float64],
    v_free: float,
) -> float:
    """Return the RMS shape distance of a speed-density trace from Weidmann.

    Each sample's residual ``(measured - predicted)`` is normalised by the
    *measured* free speed so the term tests curve shape only; the absolute
    speed level is already scored by the free-walking-speed component.

    Args:
        density: Per-sample densities (agents/m²).
        speed: Per-sample mean speeds (m/s), aligned with ``density``.
        v_free: Measured free-walking speed used to predict and normalise.

    Returns:
        Normalised RMS residual in ``[0, ∞)``; ``0.0`` when no sample clears
        the density floor or when ``v_free`` is effectively zero (those cases
        carry no FD evidence and are scored by other components).
    """
    if v_free <= _EPS or density.size == 0:
        return 0.0
    mask = (
        (density > FD_DENSITY_FLOOR_M2)
        & np.isfinite(density)
        & np.isfinite(speed)
    )
    if not bool(np.any(mask)):
        return 0.0
    predicted = _weidmann_speed(density[mask], v_free)
    residual = (speed[mask] - predicted) / v_free
    return float(np.sqrt(np.mean(residual**2)))


def _eb_penalty(peak_density: float) -> float:
    """Penalise runs that never reach a plausible bottleneck jam density.

    A realistic bottleneck evacuation forms transient congestion; a run whose
    peak density never reaches :data:`EB_CONGESTION_FLOOR_M2` is the
    frictionless degenerate.  The penalty is the normalised shortfall below the
    floor and zero once the floor is reached.

    Args:
        peak_density: Peak local density observed over the run (agents/m²).

    Returns:
        Normalised congestion-shortfall penalty in ``[0, 1]``.
    """
    if peak_density >= EB_CONGESTION_FLOOR_M2:
        return 0.0
    return (EB_CONGESTION_FLOOR_M2 - peak_density) / EB_CONGESTION_FLOOR_M2


def _mean_moving_speed(
    velocities: npt.NDArray[np.float64],
    alive: npt.NDArray[np.bool_],
) -> float | None:
    """Return the mean speed of alive, moving agents at one tick, or None.

    Args:
        velocities: Per-agent velocity vectors at the tick, shape ``(N, 2)``.
        alive: Per-agent liveness flags at the tick, shape ``(N,)``.

    Returns:
        Mean speed (m/s) of alive agents moving above the speed floor, or
        ``None`` if no such agent exists at this tick.
    """
    speeds = np.linalg.norm(velocities, axis=-1)
    moving = alive & (speeds > MOVING_SPEED_EPS_MPS)
    if not bool(np.any(moving)):
        return None
    return float(np.mean(speeds[moving]))
