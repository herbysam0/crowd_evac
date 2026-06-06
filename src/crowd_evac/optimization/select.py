"""Pareto-front selection and full-scale validation (Phase 2, Step 2.9).

Closes the optimiser loop: given the realism↔time Pareto front produced by the
NSGA-II run (Step 2.8), pick the single weight set to ship and prove it still
holds when re-run at full Tier-A agent count on the un-down-scaled scenario
suite — including a hazard scenario that exercises ``hazard_avoidance_cost``.

Selection — the realism-gated rule
----------------------------------
The two Phase-2 objectives conflict by design (realism anchors the search;
evacuation time is the lever traded against it — ``docs/plan_phase_2.md``
§Context).  The shipped default is therefore *not* the global time optimum (the
frictionless degenerate) but the **fastest weight set that is still realistic
enough and never deadlocks an agent**:

    among front points with ``stuck_count == 0`` **and**
    ``realism_distance <= threshold``, choose the minimum ``evac_time``;
    ties are broken by the knee (the point nearest the utopia corner).

The knee of the feasible front is also reported as a gate-free secondary
reference (:attr:`SelectionOutcome.knee`).

Validation — two checks, stepping back on regression
----------------------------------------------------
A point selected on the cheap, down-scaled search suite can regress at full
scale.  :func:`select_and_validate` re-scores gated points in selection order
(cheapest evac time first) and returns the **first** that clears both:

1. **Realism + stuck**, on the hazard-free full-scale suite
   (:func:`validation_suite`), via the composite fitness (Step 2.6).  The
   hazard scenario is *excluded* here on purpose: ``realism_distance`` has no
   hazard reference component, and a congested re-routing run would pollute the
   free-walking-speed estimate.
2. **Hazard reroute** (:func:`hazard_check`): on a scenario carrying a scripted
   emergency, the crowd must route around the hazard and still clear, with no
   agent deadlocked against the *re-routed* field.  Stuck is measured against
   :func:`~crowd_evac.optimization.harness.rerouted_flow_field`, not the base
   field, so a legitimate reroute is never mistaken for a deadlock.

Any point that regresses on either check is recorded in
:attr:`SelectionOutcome.rejected` and stepped past.  If *no* gated point
survives — in particular if none can reach ``stuck_count == 0`` at scale — that
is surfaced as a hard failure (:class:`SelectionError`): per the plan's RK-2.4
an unreachable ``stuck_count == 0`` is a Phase-1 flow-field/routing bug, not a
weighting problem, and must not be papered over by detuning.

Decoupling
----------
This module imports no ``pymoo``.  Selection consumes front points
*structurally* (:class:`FrontPointLike`), so it works both with an in-memory
:class:`~crowd_evac.optimization.nsga.ParetoPoint` and with points reloaded
from ``front.json`` (:func:`load_front`) — the latter being the offline-run →
next-session interface the plan prescribes.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from crowd_evac.domain.errors import CrowdEvacError
from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization.fitness import FitnessConfig, evaluate_fitness
from crowd_evac.optimization.harness import (
    DEFAULT_HISTORY_INTERVAL,
    DEFAULT_MAX_TICKS,
    DEFAULT_WALL_CLOCK_CAP_S,
    evaluate_batch,
    rerouted_flow_field,
)
from crowd_evac.optimization.stuck import stuck_count
from crowd_evac.optimization.suite import SearchScenario, calibration_rigs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate / validation defaults
# ---------------------------------------------------------------------------

DEFAULT_REALISM_THRESHOLD: float = 0.15
"""Maximum realism distance accepted by the selection gate (Step-2.4 threshold).

:func:`~crowd_evac.optimization.realism.realism_distance` is a convex-weighted
sum (``W_SPEED + W_FLOW + W_FD + W_EB == 1.0``) of per-component *normalised*
relative deviations from the empirical reference set, so the composite is itself
a normalised aggregate error: ``0.15`` admits weight sets within ≈ 15 %
aggregate deviation of the cited pedestrian-dynamics targets.  This is the
provisional acceptance gate; the final shipped threshold is fixed in Step 2.10
from the distribution actually observed on the real Pareto front, and every
call here takes ``threshold`` explicitly so it can be tightened without code
change.
"""

DEFAULT_VALIDATION_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
"""Held-out RNG seeds for full-scale validation.

Disjoint from the search seed sets (``range(K)`` with ``K`` up to ~10), so the
winner is re-scored on seeds it was never tuned against — a guard against
seed-overfitting (plan RK-2.3).
"""

HAZARD_VALIDATION_SCENARIO: str = "hazard_lecture_hall"
"""Bundled full-scale scenario carrying a scripted mid-run fire event.

The only validation scenario that exercises ``hazard_avoidance_cost``: at its
event tick a panic source blocks the main-exit approach, so the crowd must
re-route to the side exits.
"""

DEFAULT_HAZARD_EVAC_FLOOR: float = 0.95
"""Minimum mean evacuated fraction required of a hazard run to pass.

A re-route that strands part of the crowd is a regression even with
``stuck_count == 0``; the floor catches that.
"""

_LECTURE_HALL_FULL: str = "lecture_hall"
"""Bundled full-scale (150-agent) lecture hall — the un-down-scaled sibling."""

_OPEN_ROOM: str = "open_room_search"
"""Open-room scenario; already at full scale (no down-scaled sibling)."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SelectionError(CrowdEvacError):
    """Raised when no front point can be selected or survives validation.

    Covers an empty gate (no feasible, in-threshold point), a front that fails
    to deserialise, and the case where every gated point regresses at full
    scale (a likely Phase-1 routing bug per plan RK-2.4).
    """


# ---------------------------------------------------------------------------
# Front-point protocol and the concrete reloaded type
# ---------------------------------------------------------------------------


@runtime_checkable
class FrontPointLike(Protocol):
    """Structural view of one scored Pareto point used by the selector.

    Both :class:`~crowd_evac.optimization.nsga.ParetoPoint` and
    :class:`LoadedFrontPoint` satisfy this protocol, so selection never imports
    the NSGA-II (``pymoo``) stack.
    """

    @property
    def params(self) -> ForceParams: ...

    @property
    def realism_distance(self) -> float: ...

    @property
    def evac_time(self) -> float: ...

    @property
    def stuck_count(self) -> float: ...


_T = TypeVar("_T", bound=FrontPointLike)


@dataclasses.dataclass(frozen=True)
class LoadedFrontPoint:
    """A Pareto point reconstructed from a serialised ``front.json`` entry.

    Attributes:
        params: The candidate weight set.
        realism_distance: First objective (minimised).
        evac_time: Second objective (minimised).
        stuck_count: Constraint value; feasible (no deadlock) at ``<= 0``.
    """

    params: ForceParams
    realism_distance: float
    evac_time: float
    stuck_count: float


# ---------------------------------------------------------------------------
# Validation result objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    """Outcome of re-scoring one candidate on the hazard-free full-scale suite.

    Attributes:
        params: The candidate that was validated.
        realism_distance: Aggregated realism distance at full scale.
        evac_time: Aggregated effective evac time at full scale.
        stuck_count: Worst-seed stuck-agent count at full scale.
        evacuated_fraction: Mean evacuated fraction across seeds/scenarios.
        threshold: Realism-distance gate the result was checked against.
    """

    params: ForceParams
    realism_distance: float
    evac_time: float
    stuck_count: int
    evacuated_fraction: float
    threshold: float

    @property
    def passed(self) -> bool:
        """True iff no agent is stuck and realism is within ``threshold``."""
        return self.stuck_count == 0 and self.realism_distance <= self.threshold


@dataclasses.dataclass(frozen=True)
class HazardCheckResult:
    """Outcome of the hazard-reroute validation run.

    Attributes:
        scenario: Hazard scenario name that was run.
        stuck_count: Worst-seed stuck count against the *re-routed* field.
        evacuated_fraction: Mean evacuated fraction across the held-out seeds.
        evac_floor: Minimum evacuated fraction required to pass.
    """

    scenario: str
    stuck_count: int
    evacuated_fraction: float
    evac_floor: float

    @property
    def passed(self) -> bool:
        """True iff nothing deadlocks and the crowd clears above the floor."""
        return self.stuck_count == 0 and self.evacuated_fraction >= self.evac_floor


@dataclasses.dataclass(frozen=True)
class RejectedPoint:
    """A gated point that regressed at scale and was stepped past.

    Attributes:
        point: The front point that failed validation.
        validation: Its hazard-free validation result.
        hazard: Its hazard-check result, or ``None`` if it failed before the
            hazard check ran.
        reason: Short human-readable cause of rejection.
    """

    point: FrontPointLike
    validation: ValidationResult
    hazard: HazardCheckResult | None
    reason: str


@dataclasses.dataclass(frozen=True)
class SelectionOutcome:
    """The validated winner plus the selection/validation audit trail.

    Attributes:
        chosen: The shipped weight set (first gated point to clear both checks).
        chosen_point: The winner's front point (search-scale objectives).
        validation: The winner's hazard-free :class:`ValidationResult`.
        hazard: The winner's :class:`HazardCheckResult`, or ``None`` if hazard
            validation was disabled.
        knee: Knee of the feasible front (nearest utopia), a gate-free
            secondary reference; ``None`` if the front has no feasible point.
        rejected: Points that regressed and were stepped past, in order tried.
        threshold: Realism-distance gate applied throughout.
    """

    chosen: ForceParams
    chosen_point: FrontPointLike
    validation: ValidationResult
    hazard: HazardCheckResult | None
    knee: FrontPointLike | None
    rejected: tuple[RejectedPoint, ...]
    threshold: float


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def gate(
    front: Sequence[_T],
    threshold: float = DEFAULT_REALISM_THRESHOLD,
) -> list[_T]:
    """Filter a front to feasible, in-threshold points, fastest first.

    A point passes when it is feasible (``stuck_count <= 0``) and its
    ``realism_distance`` does not exceed ``threshold``.  Survivors are sorted by
    ``evac_time`` ascending (then ``realism_distance``) so the head is the
    realism-gated minimum-time candidate.

    Args:
        front: Scored Pareto points (any :class:`FrontPointLike`).
        threshold: Maximum accepted realism distance.

    Returns:
        Gated points, ascending by ``(evac_time, realism_distance)``; empty when
        none qualify.
    """
    passing = [
        p
        for p in front
        if p.stuck_count <= 0.0 and p.realism_distance <= threshold
    ]
    passing.sort(key=lambda p: (p.evac_time, p.realism_distance))
    return passing


def choose(
    front: Sequence[_T],
    *,
    threshold: float = DEFAULT_REALISM_THRESHOLD,
) -> ForceParams:
    """Apply the realism-gated rule to pick one weight set (no validation).

    Among gated points (see :func:`gate`) returns the minimum-``evac_time``
    candidate; ties on evac time are broken by the knee (utopia-nearest).

    Args:
        front: Scored Pareto points.
        threshold: Maximum accepted realism distance.

    Returns:
        The selected :class:`~crowd_evac.domain.params.ForceParams`.

    Raises:
        SelectionError: If no point satisfies the gate.
    """
    gated = gate(front, threshold)
    if not gated:
        raise SelectionError(_empty_gate_message(len(front), threshold))
    best_time = gated[0].evac_time
    tied = [p for p in gated if math.isclose(p.evac_time, best_time)]
    if len(tied) == 1:
        return tied[0].params
    winner = knee_point(tied)
    assert winner is not None  # tied is non-empty
    return winner.params


def knee_point(points: Sequence[_T]) -> _T | None:
    """Return the point nearest the utopia corner of the two objectives.

    Both objectives are min-max normalised across ``points`` (a flat objective
    contributes 0), and the point minimising Euclidean distance to the ideal
    ``(0, 0)`` corner is returned.  Standard knee proxy, used here as a
    secondary report and evac-time tie-break.

    Args:
        points: Candidate points to rank.

    Returns:
        The knee point, or ``None`` if ``points`` is empty.
    """
    pts = list(points)
    if not pts:
        return None
    norm_r = _minmax([p.realism_distance for p in pts])
    norm_e = _minmax([p.evac_time for p in pts])
    dists = [math.hypot(r, e) for r, e in zip(norm_r, norm_e)]
    best = min(range(len(pts)), key=lambda i: dists[i])
    return pts[best]


# ---------------------------------------------------------------------------
# Full-scale validation configuration
# ---------------------------------------------------------------------------


def validation_suite() -> tuple[SearchScenario, ...]:
    """Return the hazard-free full-scale scenarios for Tier-A validation.

    Mirrors the search suite's topologies at full agent count: the 150-agent
    lecture hall (full-scale sibling of ``lecture_hall_small``) and the
    open-room plan (already full scale).  The hazard scenario is validated
    separately by :func:`hazard_check`, not here.

    Returns:
        Non-empty tuple of full-scale :class:`SearchScenario` entries.
    """
    return (
        SearchScenario(
            name=_LECTURE_HALL_FULL,
            scenario_ref=_LECTURE_HALL_FULL,
            full_agent_count=150,
            description=(
                "Full-scale lecture hall (150 agents) — un-down-scaled sibling "
                "of lecture_hall_small; validates the chosen weights at Tier-A "
                "count on the primary seating-row bottleneck."
            ),
        ),
        SearchScenario(
            name=_OPEN_ROOM,
            scenario_ref=_OPEN_ROOM,
            full_agent_count=25,
            description=(
                "Open room with two competing exits — validates routing and "
                "herding at its native scale (no down-scaled sibling)."
            ),
        ),
    )


def default_validation_config(
    *,
    seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS,
    max_workers: int | None = None,
) -> FitnessConfig:
    """Build the hazard-free full-scale validation fitness config.

    Args:
        seeds: Held-out RNG seeds (disjoint from the search seeds).
        max_workers: Harness worker processes; ``None`` uses all cores.

    Returns:
        A :class:`~crowd_evac.optimization.fitness.FitnessConfig` over the
        full-scale :func:`validation_suite` and the bottleneck rig.
    """
    return FitnessConfig(
        seeds=seeds,
        scenarios=validation_suite(),
        rig=calibration_rigs()[0],
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# Validation runs
# ---------------------------------------------------------------------------


def validate_at_scale(
    params: ForceParams,
    config: FitnessConfig | None = None,
    *,
    threshold: float = DEFAULT_REALISM_THRESHOLD,
) -> ValidationResult:
    """Re-score one candidate on the hazard-free suite and check it vs the gate.

    Args:
        params: Weight set to validate.
        config: Validation fitness config; ``None`` uses
            :func:`default_validation_config` (full scale, held-out seeds).
        threshold: Realism-distance gate the result is checked against.

    Returns:
        A :class:`ValidationResult`; inspect :attr:`ValidationResult.passed`.
    """
    cfg = config if config is not None else default_validation_config()
    res = evaluate_fitness(params, cfg)
    return ValidationResult(
        params=params,
        realism_distance=res.realism_distance,
        evac_time=res.evac_time,
        stuck_count=res.stuck_count,
        evacuated_fraction=res.evacuated_fraction,
        threshold=threshold,
    )


def hazard_check(
    params: ForceParams,
    *,
    scenario_ref: str | Path = HAZARD_VALIDATION_SCENARIO,
    seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS,
    evac_floor: float = DEFAULT_HAZARD_EVAC_FLOOR,
    max_workers: int | None = None,
) -> HazardCheckResult:
    """Validate that the crowd re-routes around a scripted hazard and clears.

    Runs the hazard scenario once per held-out seed (in parallel) and measures
    the stuck count against the *re-routed* flow field — the field agents
    actually navigate while the hazard is active — so a legitimate reroute is
    never counted as a deadlock.

    Args:
        params: Weight set to validate (supplies ``hazard_avoidance_cost``).
        scenario_ref: Hazard scenario name or path.
        seeds: Held-out RNG seeds to average over. Must be non-empty.
        evac_floor: Minimum mean evacuated fraction to pass.
        max_workers: Harness worker processes; ``None`` uses all cores.

    Returns:
        A :class:`HazardCheckResult`; inspect :attr:`HazardCheckResult.passed`.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    field = rerouted_flow_field(scenario_ref, params)
    results = evaluate_batch(
        [params],
        scenario_ref,
        list(seeds),
        DEFAULT_MAX_TICKS,
        wall_clock_cap_s=DEFAULT_WALL_CLOCK_CAP_S,
        history_interval=DEFAULT_HISTORY_INTERVAL,
        max_workers=max_workers,
    )
    worst_stuck = max(stuck_count(r, field) for r in results)
    mean_evac = sum(r.evacuated_fraction for r in results) / len(results)
    name = scenario_ref if isinstance(scenario_ref, str) else scenario_ref.name
    return HazardCheckResult(
        scenario=name,
        stuck_count=worst_stuck,
        evacuated_fraction=mean_evac,
        evac_floor=evac_floor,
    )


def select_and_validate(
    front: Sequence[_T],
    *,
    validation_config: FitnessConfig | None = None,
    hazard_scenario: str | Path | None = HAZARD_VALIDATION_SCENARIO,
    threshold: float = DEFAULT_REALISM_THRESHOLD,
) -> SelectionOutcome:
    """Select the realism-gated winner and prove it holds at Tier-A scale.

    Validates gated points in selection order (cheapest evac time first) and
    returns the first that clears both the hazard-free realism/stuck check and
    the hazard reroute check; regressing points are recorded and stepped past.

    Args:
        front: Scored Pareto points.
        validation_config: Hazard-free full-scale config; ``None`` uses
            :func:`default_validation_config`.
        hazard_scenario: Hazard scenario name/path, or ``None`` to skip the
            hazard reroute check entirely.
        threshold: Realism-distance gate applied to selection and validation.

    Returns:
        A :class:`SelectionOutcome` for the validated winner.

    Raises:
        SelectionError: If the gate is empty or every gated point regresses at
            scale (the latter likely a Phase-1 routing bug — plan RK-2.4).
    """
    gated = gate(front, threshold)
    if not gated:
        raise SelectionError(_empty_gate_message(len(front), threshold))
    cfg = (
        validation_config
        if validation_config is not None
        else default_validation_config()
    )
    knee = knee_point([p for p in front if p.stuck_count <= 0.0])
    rejected: list[RejectedPoint] = []
    for point in gated:
        outcome = _try_point(point, cfg, hazard_scenario, threshold, knee, rejected)
        if outcome is not None:
            return outcome
    raise SelectionError(
        f"all {len(gated)} gated point(s) regressed at full scale; if no point "
        "can reach stuck_count == 0, suspect a Phase-1 flow-field/routing bug "
        "(plan RK-2.4), not the weights"
    )


def _try_point(
    point: _T,
    cfg: FitnessConfig,
    hazard_scenario: str | Path | None,
    threshold: float,
    knee: FrontPointLike | None,
    rejected: list[RejectedPoint],
) -> SelectionOutcome | None:
    """Validate one gated point; return a winning outcome or record rejection.

    Args:
        point: The gated candidate to validate.
        cfg: Hazard-free validation fitness config.
        hazard_scenario: Hazard scenario, or ``None`` to skip the hazard check.
        threshold: Realism-distance gate.
        knee: Pre-computed feasible-front knee for the outcome.
        rejected: Mutable accumulator of prior rejections; appended to on a
            regression.

    Returns:
        A :class:`SelectionOutcome` if *point* clears both checks, else ``None``
        (with a :class:`RejectedPoint` appended to ``rejected``).
    """
    vr = validate_at_scale(point.params, cfg, threshold=threshold)
    if not vr.passed:
        rejected.append(
            RejectedPoint(point, vr, None, "realism/stuck regressed at scale")
        )
        logger.warning(
            "point regressed (realism/stuck): realism=%.4f stuck=%d — stepping on",
            vr.realism_distance, vr.stuck_count,
        )
        return None
    hr: HazardCheckResult | None = None
    if hazard_scenario is not None:
        hr = hazard_check(
            point.params, scenario_ref=hazard_scenario,
            seeds=cfg.seeds, max_workers=cfg.max_workers,
        )
        if not hr.passed:
            rejected.append(
                RejectedPoint(point, vr, hr, "hazard reroute regressed")
            )
            logger.warning(
                "point regressed (hazard): stuck=%d evac_frac=%.3f — stepping on",
                hr.stuck_count, hr.evacuated_fraction,
            )
            return None
    logger.info(
        "validated winner: realism=%.4f evac=%.2f stuck=%d (after %d rejection(s))",
        vr.realism_distance, vr.evac_time, vr.stuck_count, len(rejected),
    )
    return SelectionOutcome(
        chosen=point.params,
        chosen_point=point,
        validation=vr,
        hazard=hr,
        knee=knee,
        rejected=tuple(rejected),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Front (de)serialisation and outcome reporting
# ---------------------------------------------------------------------------


def load_front(path: Path) -> tuple[LoadedFrontPoint, ...]:
    """Load Pareto points from a ``front.json`` written by the NSGA-II driver.

    Args:
        path: Path to the ``front.json`` file produced by
            :func:`~crowd_evac.optimization.nsga.write_front`.

    Returns:
        The front's points as :class:`LoadedFrontPoint` instances, in file
        order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        SelectionError: If the file is not valid front JSON or a ``params``
            block cannot be reconstructed into :class:`ForceParams`.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    try:
        points = tuple(
            LoadedFrontPoint(
                params=ForceParams(**entry["params"]),
                realism_distance=float(entry["realism_distance"]),
                evac_time=float(entry["evac_time"]),
                stuck_count=float(entry["stuck_count"]),
            )
            for entry in payload["front"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionError(f"malformed front file {path}: {exc}") from exc
    return points


def write_outcome(path: Path, outcome: SelectionOutcome) -> None:
    """Serialise a :class:`SelectionOutcome` to JSON for the calibration record.

    Args:
        path: Destination ``.json`` file. Parent directories are created.
        outcome: The validated selection to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": outcome.threshold,
        "chosen_params": dataclasses.asdict(outcome.chosen),
        "chosen_point": _point_dict(outcome.chosen_point),
        "validation": _validation_dict(outcome.validation),
        "hazard": (
            _hazard_dict(outcome.hazard) if outcome.hazard is not None else None
        ),
        "knee_point": (
            _point_dict(outcome.knee) if outcome.knee is not None else None
        ),
        "n_rejected": len(outcome.rejected),
        "rejected": [
            {
                "point": _point_dict(rp.point),
                "validation": _validation_dict(rp.validation),
                "hazard": (
                    _hazard_dict(rp.hazard) if rp.hazard is not None else None
                ),
                "reason": rp.reason,
            }
            for rp in outcome.rejected
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _empty_gate_message(n_points: int, threshold: float) -> str:
    """Build the SelectionError message for an empty gate."""
    return (
        f"no front point satisfies the gate (stuck_count == 0 and "
        f"realism_distance <= {threshold}); {n_points} point(s) considered"
    )


def _minmax(values: list[float]) -> list[float]:
    """Min-max normalise a list to ``[0, 1]``; a flat list maps to all-zeros.

    Args:
        values: Values to normalise.

    Returns:
        Normalised values; all ``0.0`` when ``max == min`` (no spread).
    """
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0.0:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


def _point_dict(point: FrontPointLike) -> dict[str, object]:
    """Serialise one front point's params and objectives to a plain dict."""
    return {
        "params": dataclasses.asdict(point.params),
        "realism_distance": point.realism_distance,
        "evac_time": point.evac_time,
        "stuck_count": point.stuck_count,
        "feasible": point.stuck_count <= 0.0,
    }


def _validation_dict(result: ValidationResult) -> dict[str, object]:
    """Serialise a :class:`ValidationResult` to a plain dict."""
    return {
        "realism_distance": result.realism_distance,
        "evac_time": result.evac_time,
        "stuck_count": result.stuck_count,
        "evacuated_fraction": result.evacuated_fraction,
        "threshold": result.threshold,
        "passed": result.passed,
    }


def _hazard_dict(result: HazardCheckResult) -> dict[str, object]:
    """Serialise a :class:`HazardCheckResult` to a plain dict."""
    return {
        "scenario": result.scenario,
        "stuck_count": result.stuck_count,
        "evacuated_fraction": result.evacuated_fraction,
        "evac_floor": result.evac_floor,
        "passed": result.passed,
    }
