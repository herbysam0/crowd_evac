"""Composite multi-objective fitness for Phase-2 weight optimisation (2.6).

Assembles the optimiser-facing objective/constraint vector that NSGA-II
(Step 2.8) minimises.  A single :func:`evaluate_fitness` call scores one
:class:`~crowd_evac.domain.params.ForceParams` candidate over ``K`` seeds ×
``M`` suite scenarios (Step 2.3) plus the bottleneck calibration rig, and
returns:

* **two objectives to minimise** — ``(realism_distance, evac_time)`` — the
  structurally-conflicting pair the Phase-2 plan trades off (realism anchors,
  evacuation time is the lever); and
* **one constraint** in pymoo's ``g(x) <= 0`` form — ``stuck_count`` (Step 2.5),
  feasible only when no agent is deadlocked while a viable exit route exists.

Aggregation, noise, generalisation
-----------------------------------
Per candidate, every (seed, scenario) pair is evaluated by the headless harness
(Step 2.2).  Per-seed statistics are aggregated across seeds with a **mean plus
a robustness quantile** (:func:`_aggregate`): raising ``K`` lowers the variance
of the aggregated objective, and the quantile term penalises candidates that
are merely lucky on the mean but fragile on a bad seed.  Because the seed set is
fixed in :class:`FitnessConfig`, *every* candidate is scored on the **same**
seeds — common random numbers (CRN) across candidates, which cuts the variance
of pairwise comparisons NSGA-II makes during selection.

Parallelism
-----------
All ``K × (M + 1)`` harness runs for one candidate are dispatched as a single
flat job set across a :class:`~concurrent.futures.ProcessPoolExecutor`
(``max_workers`` cores), so no single evaluation blocks the others and the whole
``K × M`` grid runs concurrently rather than one scenario at a time.  Setting
``max_workers = 1`` selects an in-process serial path (used by the unit tests so
they can monkeypatch the evaluator).  Workers are module-level functions, so
they are picklable under the Windows ``spawn`` start method.

Scope note (stuck constraint): the stuck detector reads the *base* flow field
built from the scenario floor plan, which is exact for the **hazard-free** search
suite (the Phase-2 search scenarios carry no hazards — see the plan's decision
variables note).  Hazard scenarios re-route the live field mid-run and are
handled only in the full-scale validation of Step 2.9, not here.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from crowd_evac.adapters.io.scenario_loader import (
    load_bundled_scenario,
    load_scenario_file,
)
from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization.harness import (
    DEFAULT_HISTORY_INTERVAL,
    DEFAULT_MAX_TICKS,
    DEFAULT_WALL_CLOCK_CAP_S,
    RunResult,
    evaluate,
)
from crowd_evac.optimization.realism import CalibrationRunSet, realism_distance
from crowd_evac.optimization.stuck import stuck_count
from crowd_evac.optimization.suite import (
    CalibrationRig,
    SearchScenario,
    calibration_rigs,
    search_suite,
)
from crowd_evac.pathfinding.flow_field import FlowField

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aggregation / penalty defaults (tunable via FitnessConfig)
# ---------------------------------------------------------------------------

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
"""Default seed set (``K = 3``); fixed across candidates to provide CRN."""

DEFAULT_ROBUSTNESS_QUANTILE: float = 0.75
"""Upper quantile of the per-seed objective used as the robustness anchor."""

DEFAULT_ROBUSTNESS_WEIGHT: float = 0.5
"""Blend weight on ``(quantile - mean)``; ``0`` reduces aggregation to the mean."""

DEFAULT_INCOMPLETE_EVAC_PENALTY_S: float = 500.0
"""Evac-time penalty (s) per unit unevacuated fraction.

Equals :data:`~crowd_evac.optimization.harness.DEFAULT_MAX_TICKS` × ``DT``
(10 000 × 0.05 s), so a run that strands agents is scored worse than any run
that fully clears within the tick cap — partial clears cannot win on time.
"""


# ---------------------------------------------------------------------------
# Configuration and result value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FitnessConfig:
    """Knobs controlling the cost/fidelity of one fitness evaluation.

    ``K`` is ``len(seeds)`` and ``M`` is ``len(scenarios)``; agent-count scale
    is governed by *which* scenarios are supplied (down-scaled search entries
    during the search, full-scale entries during Step-2.9 validation).

    Attributes:
        seeds: RNG seeds evaluated per scenario. Fixed across candidates to
            give common random numbers. Must be non-empty.
        scenarios: Search-suite scenarios scored for both objectives and the
            stuck constraint. Must be non-empty.
        rig: Bottleneck calibration rig supplying the realism metric's
            specific-flow / speed-density signal.
        max_ticks: Per-run tick cap forwarded to the harness.
        wall_clock_cap_s: Per-run wall-clock cap forwarded to the harness.
        history_interval: Per-run state-sampling interval forwarded to the
            harness.
        robustness_quantile: Per-seed upper quantile (in ``[0, 1]``) blended
            into each aggregated objective.
        robustness_weight: Non-negative weight on ``(quantile - mean)``.
        incomplete_evac_penalty_s: Evac-time penalty (s) per unit unevacuated
            fraction. Must be >= 0.
        max_workers: Worker process count for the harness job set. ``None``
            uses all logical cores; ``1`` runs serially in-process.
    """

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    scenarios: tuple[SearchScenario, ...] = dataclasses.field(
        default_factory=lambda: tuple(search_suite())
    )
    rig: CalibrationRig = dataclasses.field(
        default_factory=lambda: calibration_rigs()[0]
    )
    max_ticks: int = DEFAULT_MAX_TICKS
    wall_clock_cap_s: float = DEFAULT_WALL_CLOCK_CAP_S
    history_interval: int = DEFAULT_HISTORY_INTERVAL
    robustness_quantile: float = DEFAULT_ROBUSTNESS_QUANTILE
    robustness_weight: float = DEFAULT_ROBUSTNESS_WEIGHT
    incomplete_evac_penalty_s: float = DEFAULT_INCOMPLETE_EVAC_PENALTY_S
    max_workers: int | None = None

    def __post_init__(self) -> None:
        """Validate the configuration; raise ValueError on any bad value."""
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not self.scenarios:
            raise ValueError("scenarios must be non-empty")
        if self.max_ticks <= 0:
            raise ValueError(f"max_ticks must be > 0, got {self.max_ticks!r}")
        if not 0.0 <= self.robustness_quantile <= 1.0:
            raise ValueError(
                "robustness_quantile must be in [0, 1], got "
                f"{self.robustness_quantile!r}"
            )
        if self.robustness_weight < 0.0:
            raise ValueError(
                f"robustness_weight must be >= 0, got {self.robustness_weight!r}"
            )
        if self.incomplete_evac_penalty_s < 0.0:
            raise ValueError(
                "incomplete_evac_penalty_s must be >= 0, got "
                f"{self.incomplete_evac_penalty_s!r}"
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(
                f"max_workers must be >= 1 or None, got {self.max_workers!r}"
            )


@dataclasses.dataclass(frozen=True)
class FitnessResult:
    """The objective/constraint vector for one candidate, plus diagnostics.

    The optimiser consumes :attr:`objectives` (minimise) and
    :attr:`constraints` (``g(x) <= 0`` feasible); the remaining fields expose
    the aggregation drivers for analysis and tests.  :attr:`wall_clock_s` is
    excluded from equality so two runs of the same candidate compare equal
    despite differing timing.

    Attributes:
        objectives: ``(realism_distance, evac_time)`` to minimise.
        constraints: ``(stuck_count,)`` in pymoo form; feasible at ``<= 0``.
        realism_distance: Aggregated realism objective (== ``objectives[0]``).
        evac_time: Aggregated effective evac-time objective
            (== ``objectives[1]``).
        stuck_count: Worst-seed total stuck-agent count (== ``constraints[0]``).
        evacuated_fraction: Mean evacuated fraction across seeds and suite
            scenarios, in ``[0, 1]``.
        per_seed_realism: Per-seed realism distances, length ``K``.
        per_seed_evac_time: Per-seed mean effective evac times, length ``K``.
        per_seed_stuck: Per-seed total stuck counts, length ``K``.
        wall_clock_s: Wall-clock time (s) spent dispatching the harness runs.
    """

    objectives: tuple[float, float]
    constraints: tuple[float, ...]
    realism_distance: float
    evac_time: float
    stuck_count: int
    evacuated_fraction: float
    per_seed_realism: tuple[float, ...]
    per_seed_evac_time: tuple[float, ...]
    per_seed_stuck: tuple[int, ...]
    wall_clock_s: float = dataclasses.field(default=0.0, compare=False)


# ---------------------------------------------------------------------------
# Internal job specification (picklable under spawn)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _JobSpec:
    """One harness run, fully specified so a worker process can execute it."""

    params: ForceParams
    scenario_ref: str | Path
    seed: int
    max_ticks: int
    wall_clock_cap_s: float
    history_interval: int


def _run_job(spec: _JobSpec) -> RunResult:
    """Execute one :class:`_JobSpec` via the harness (process-pool worker).

    Module-level so it is picklable under the Windows ``spawn`` start method.

    Args:
        spec: The single run to evaluate.

    Returns:
        The harness :class:`RunResult` for the run.
    """
    return evaluate(
        spec.params,
        spec.scenario_ref,
        spec.seed,
        spec.max_ticks,
        wall_clock_cap_s=spec.wall_clock_cap_s,
        history_interval=spec.history_interval,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_fitness(
    params: ForceParams,
    config: FitnessConfig | None = None,
) -> FitnessResult:
    """Score one candidate into the optimiser's objective/constraint vector.

    Runs ``params`` on every (seed, scenario) pair plus the rig (one run per
    seed) through the headless harness in parallel, then aggregates per-seed
    realism distance, effective evac time, and stuck count into two minimised
    objectives and one ``g(x) <= 0`` constraint.

    The same ``params`` and ``config`` always yield an equal
    :class:`FitnessResult` (timing aside) — every underlying run is
    deterministic in its seed.

    Args:
        params: Force-weight candidate to evaluate.
        config: Evaluation knobs; ``None`` uses :class:`FitnessConfig`
            defaults (the down-scaled search suite, ``K = 3`` seeds, all cores).

    Returns:
        A :class:`FitnessResult` whose :attr:`~FitnessResult.objectives` are
        ``(realism_distance, evac_time)`` and whose
        :attr:`~FitnessResult.constraints` are ``(stuck_count,)``.
    """
    cfg = config if config is not None else FitnessConfig()

    specs = _build_specs(params, cfg)
    t0 = time.monotonic()
    runs = _dispatch(specs, cfg.max_workers)
    wall_s = time.monotonic() - t0

    flow_fields = [_flow_field_for(s.scenario_ref) for s in cfg.scenarios]
    grid = _regroup(runs, len(cfg.seeds), len(cfg.scenarios))

    realism_per_seed: list[float] = []
    evac_per_seed: list[float] = []
    stuck_per_seed: list[int] = []
    evac_frac_per_seed: list[float] = []

    for suite_runs, rig_run in grid:
        realism_per_seed.append(
            realism_distance(
                CalibrationRunSet(
                    flow_results=tuple(suite_runs),
                    bottleneck_result=rig_run,
                    bottleneck_door_width_m=cfg.rig.door_width_m,
                )
            )
        )
        evac_per_seed.append(
            float(
                np.mean(
                    [
                        _effective_evac_time(r, cfg.incomplete_evac_penalty_s)
                        for r in suite_runs
                    ]
                )
            )
        )
        stuck_per_seed.append(
            sum(stuck_count(r, ff) for r, ff in zip(suite_runs, flow_fields))
        )
        evac_frac_per_seed.append(
            float(np.mean([r.evacuated_fraction for r in suite_runs]))
        )

    realism_obj = _aggregate(
        realism_per_seed, cfg.robustness_quantile, cfg.robustness_weight
    )
    evac_obj = _aggregate(
        evac_per_seed, cfg.robustness_quantile, cfg.robustness_weight
    )
    stuck_constraint = max(stuck_per_seed)

    logger.info(
        "evaluate_fitness: realism=%.4f evac=%.2f stuck=%d (K=%d M=%d) "
        "in %.2f s",
        realism_obj,
        evac_obj,
        stuck_constraint,
        len(cfg.seeds),
        len(cfg.scenarios),
        wall_s,
    )

    return FitnessResult(
        objectives=(realism_obj, evac_obj),
        constraints=(float(stuck_constraint),),
        realism_distance=realism_obj,
        evac_time=evac_obj,
        stuck_count=stuck_constraint,
        evacuated_fraction=float(np.mean(evac_frac_per_seed)),
        per_seed_realism=tuple(realism_per_seed),
        per_seed_evac_time=tuple(evac_per_seed),
        per_seed_stuck=tuple(stuck_per_seed),
        wall_clock_s=wall_s,
    )


# ---------------------------------------------------------------------------
# Dispatch and grouping
# ---------------------------------------------------------------------------


def _build_specs(params: ForceParams, cfg: FitnessConfig) -> list[_JobSpec]:
    """Build the flat job set: per seed, every suite scenario then the rig.

    Ordering is ``seed-major``: for each seed the ``M`` suite runs precede that
    seed's single rig run, so the flat result list regroups cleanly by stride
    ``M + 1`` in :func:`_regroup`.

    Args:
        params: Candidate evaluated by every job.
        cfg: Configuration supplying seeds, scenarios, rig, and caps.

    Returns:
        Ordered list of ``len(seeds) × (len(scenarios) + 1)`` job specs.
    """
    specs: list[_JobSpec] = []
    for seed in cfg.seeds:
        for scen in cfg.scenarios:
            specs.append(
                _JobSpec(
                    params, scen.scenario_ref, seed, cfg.max_ticks,
                    cfg.wall_clock_cap_s, cfg.history_interval,
                )
            )
        specs.append(
            _JobSpec(
                params, cfg.rig.scenario_ref, seed, cfg.max_ticks,
                cfg.wall_clock_cap_s, cfg.history_interval,
            )
        )
    return specs


def _dispatch(specs: list[_JobSpec], max_workers: int | None) -> list[RunResult]:
    """Run every job spec, in parallel processes or serially in-process.

    A single worker (``max_workers == 1``) takes the in-process path so unit
    tests can monkeypatch the module-level :func:`evaluate`; otherwise all jobs
    are mapped across a :class:`~concurrent.futures.ProcessPoolExecutor`,
    preserving input order.

    Args:
        specs: Ordered job set to execute. Must be non-empty.
        max_workers: Worker count; ``None`` uses all cores, ``1`` is serial.

    Returns:
        Results in the same order as ``specs``.
    """
    n_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    n_workers = min(n_workers, len(specs))
    if n_workers <= 1:
        return [_run_job(s) for s in specs]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_run_job, specs))


def _regroup(
    runs: list[RunResult],
    n_seeds: int,
    n_scenarios: int,
) -> list[tuple[list[RunResult], RunResult]]:
    """Regroup the flat result list into ``(suite_runs, rig_run)`` per seed.

    Inverts the seed-major, stride ``M + 1`` layout produced by
    :func:`_build_specs`.

    Args:
        runs: Flat results, length ``n_seeds × (n_scenarios + 1)``.
        n_seeds: Number of seeds (``K``).
        n_scenarios: Number of suite scenarios (``M``).

    Returns:
        One ``(suite_runs, rig_run)`` tuple per seed, in seed order.
    """
    stride = n_scenarios + 1
    grouped: list[tuple[list[RunResult], RunResult]] = []
    for s in range(n_seeds):
        base = s * stride
        suite_runs = runs[base : base + n_scenarios]
        rig_run = runs[base + n_scenarios]
        grouped.append((suite_runs, rig_run))
    return grouped


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate(values: list[float], quantile: float, weight: float) -> float:
    """Blend a sample mean with an upper-quantile robustness penalty.

    Returns ``mean + weight · (Q_quantile - mean)``.  With ``weight == 0`` (or a
    single sample) this is the plain mean; a positive weight shifts the score
    toward the worse (higher, since both objectives are minimised) seeds, so a
    candidate that is fragile on one seed is penalised.

    Args:
        values: Per-seed objective values. Must be non-empty.
        quantile: Quantile in ``[0, 1]`` taken as the robustness anchor.
        weight: Non-negative blend weight on ``(quantile - mean)``.

    Returns:
        The blended scalar objective.
    """
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if weight == 0.0 or arr.size < 2:
        return mean
    q = float(np.quantile(arr, quantile))
    return mean + weight * (q - mean)


def _effective_evac_time(result: RunResult, penalty_s: float) -> float:
    """Return evac time penalised for any unevacuated fraction.

    ``evac_time + penalty_s · (1 - evacuated_fraction)``.  A fully-evacuated run
    is scored on raw clearance time; a partial clear is pushed worse in
    proportion to the stranded fraction, so time-minimisation cannot reward a
    candidate that simply abandons agents.

    Args:
        result: The run to score.
        penalty_s: Penalty (s) per unit unevacuated fraction.

    Returns:
        Effective evacuation time in seconds.
    """
    return result.evac_time + penalty_s * (1.0 - result.evacuated_fraction)


def _flow_field_for(scenario_ref: str | Path) -> FlowField:
    """Build the base flow field for a scenario reference.

    Loads the scenario's floor plan (bundled name or filesystem path) and
    solves the hazard-free flow field — the exact field the harness uses for
    the hazard-free search suite, so the stuck detector reads the same
    navigation signal the run experienced.

    Args:
        scenario_ref: Bundled scenario name (``str``) or path (:class:`Path`).

    Returns:
        A solved :class:`~crowd_evac.pathfinding.flow_field.FlowField`.

    Raises:
        TypeError: If ``scenario_ref`` is neither ``str`` nor :class:`Path`.
    """
    if isinstance(scenario_ref, str):
        floor_plan, _ = load_bundled_scenario(scenario_ref)
    elif isinstance(scenario_ref, Path):
        floor_plan, _ = load_scenario_file(scenario_ref)
    else:
        raise TypeError(
            "scenario_ref must be str or Path, got "
            f"{type(scenario_ref).__name__!r}"
        )
    return FlowField.build(floor_plan)
