"""Parameter search space bounds and Sobol sensitivity pre-pass (Phase 2, Step 2.7).

Declares the per-weight lower/upper bounds that constrain the NSGA-II search
(Step 2.8), provides quasi-random sampling (:func:`sample_space`) via
``scipy.stats.qmc``, and implements a lightweight Spearman rank-correlation
sensitivity pre-pass (:func:`run_sensitivity_prepass`) that ranks each weight's
influence on both objectives before the expensive multi-objective search.

The canonical bound ordering mirrors :data:`~crowd_evac.domain.params._FIELD_ORDER`
so a raw float array can be converted directly to/from
:class:`~crowd_evac.domain.params.ForceParams` via
:meth:`~crowd_evac.domain.params.ForceParams.from_array`.

Bounds source: ``docs/plan_phase_2.md`` §Decision variables.

Note on ``max_speed``: the empirical free-walking speed is ~1.34 m/s (Weidmann
1993), so [1.2, 3.0] is deliberately wider on the upper end to accommodate the
Phase-1 default (2.5 m/s); the realism metric in Step 2.4 will drive
calibration toward the lower end of the range.  Hazard-avoidance cost is not
exercised by the hazard-free search suite (see plan §Decision variables); the
sensitivity pre-pass will typically read it as non-influential and the NSGA-II
run fixes it at its default accordingly.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.params import ForceParams, N_PARAMS, _FIELD_ORDER
from crowd_evac.optimization.fitness import FitnessConfig, FitnessResult, evaluate_fitness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search bounds
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SearchBound:
    """Lower / upper search bound for one optimisable parameter.

    Attributes:
        name: Parameter name matching the corresponding
            :class:`~crowd_evac.domain.params.ForceParams` field.
        low: Inclusive lower bound.  Must be strictly less than ``high``.
        high: Inclusive upper bound.
        description: Human-readable annotation for analysis output.
    """

    name: str
    low: float
    high: float
    description: str

    def __post_init__(self) -> None:
        """Raise ValueError if ``low >= high``."""
        if self.low >= self.high:
            raise ValueError(
                f"{self.name!r}: low ({self.low!r}) must be < high ({self.high!r})"
            )


# Canonical bounds in _FIELD_ORDER sequence.
# Width classes: behavioural (wide) | gain (wide) | radius (wide) | physical (narrow).
BOUNDS: tuple[SearchBound, ...] = (
    SearchBound("relaxation_time", 0.1, 2.0,
                "f_exit velocity-relaxation time (s) — behavioural"),
    SearchBound("panic_speed_multiplier", 1.0, 2.5,
                "f_exit panic speed boost — behavioural"),
    SearchBound("repulsion_strength", 0.0, 12.0,
                "f_crowd short-range repulsion gain — gain"),
    SearchBound("repulsion_radius", 0.2, 2.5,
                "f_crowd cut-off radius (m) — radius"),
    SearchBound("high_density_threshold", 0.5, 10.0,
                "f_density activation threshold (agents/m²) — behavioural"),
    SearchBound("density_pressure_strength", 0.0, 5.0,
                "f_density drag gain — gain"),
    SearchBound("density_sensing_radius", 0.5, 5.0,
                "f_density sensing radius (m) — radius"),
    SearchBound("herd_attraction_strength", 0.0, 3.0,
                "f_herd velocity-alignment gain — gain"),
    SearchBound("herd_perception_radius", 1.0, 20.0,
                "f_herd perception radius (m) — radius"),
    SearchBound("panic_repulsion_strength", 0.0, 12.0,
                "f_panic_repulsion gain — gain"),
    SearchBound("max_accel", 0.5, 6.0,
                "integrator max acceleration (m/s²) — responsiveness"),
    SearchBound("max_speed", 1.2, 3.0,
                "base max walking speed (m/s) — physical; empirical target ~1.34"),
    SearchBound("hazard_avoidance_cost", 0.0, 100.0,
                "flow-field hazard cost multiplier — navigation"),
)

# Module-level consistency guards — fail fast if the table drifts.
assert len(BOUNDS) == N_PARAMS, (
    f"BOUNDS has {len(BOUNDS)} entries but N_PARAMS == {N_PARAMS}"
)
assert tuple(b.name for b in BOUNDS) == _FIELD_ORDER, (
    "BOUNDS names do not match _FIELD_ORDER"
)


def bounds_array() -> npt.NDArray[np.float64]:
    """Return the search bounds as a (N_PARAMS, 2) float64 array [low, high] per row.

    Returns:
        Array of shape ``(N_PARAMS, 2)`` where ``arr[i, 0]`` and ``arr[i, 1]``
        are the lower and upper bounds for parameter ``i`` in
        :data:`~crowd_evac.domain.params._FIELD_ORDER` order.
    """
    return np.array([[b.low, b.high] for b in BOUNDS], dtype=np.float64)


def validate_defaults() -> None:
    """Assert that :meth:`ForceParams.defaults` lies inside every search bound.

    Intended as a module-level sanity check after any change to either the
    :class:`SearchBound` table or ``domain.constants``.

    Raises:
        ValueError: If any default value falls outside its declared bound.
    """
    arr = ForceParams.defaults().to_array()
    violations: list[str] = []
    for i, b in enumerate(BOUNDS):
        v = float(arr[i])
        if not (b.low <= v <= b.high):
            violations.append(
                f"  {b.name!r}: default {v!r} outside [{b.low}, {b.high}]"
            )
    if violations:
        raise ValueError(
            "ForceParams defaults outside search bounds:\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Quasi-random sampling
# ---------------------------------------------------------------------------


def sample_space(
    n: int,
    method: Literal["sobol", "lhs"] = "sobol",
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Draw *n* quasi-random parameter vectors from the search space.

    Samples are scaled to the per-parameter bounds in :data:`BOUNDS` and
    returned in canonical :data:`~crowd_evac.domain.params._FIELD_ORDER`.

    For Sobol sequences, *n* that is a power of two gives optimal coverage;
    non-power-of-two values are accepted but may trigger a scipy warning.

    Args:
        n: Number of sample vectors to draw.  Must be >= 1.
        method: ``"sobol"`` for a scrambled Sobol low-discrepancy sequence;
            ``"lhs"`` for Latin Hypercube Sampling.
        seed: RNG seed for reproducibility.

    Returns:
        Float64 array of shape ``(n, N_PARAMS)`` with every row inside the
        declared bounds (up to floating-point precision).

    Raises:
        ValueError: If ``n < 1`` or ``method`` is unrecognised.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    try:
        from scipy.stats import qmc  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for space sampling; "
            "install it with: pip install 'scipy>=1.11'"
        ) from exc

    lo = np.array([b.low for b in BOUNDS], dtype=np.float64)
    hi = np.array([b.high for b in BOUNDS], dtype=np.float64)

    if method == "sobol":
        sampler = qmc.Sobol(d=N_PARAMS, scramble=True, seed=seed)
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=N_PARAMS, seed=seed)
    else:
        raise ValueError(f"method must be 'sobol' or 'lhs', got {method!r}")

    unit = sampler.random(n)
    return np.asarray(qmc.scale(unit, lo, hi), dtype=np.float64)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SensitivityResult:
    """Spearman rank-correlation sensitivity for all search parameters.

    Attributes:
        param_names: Parameter names in :data:`~crowd_evac.domain.params._FIELD_ORDER`.
        abs_spearman_realism: ``|rho|`` per parameter vs. the realism objective.
        abs_spearman_evac: ``|rho|`` per parameter vs. the evac-time objective.
        rank_by_realism: Parameter names sorted most- to least-influential on
            the realism objective (highest ``|rho|`` first).
        rank_by_evac: Parameter names sorted most- to least-influential on
            the evac-time objective.
        rank_combined: Parameter names sorted by ``max(|rho_realism|,
            |rho_evac|)`` — the combined influence on either objective.
        n_samples: Number of samples used to compute the correlations.
    """

    param_names: tuple[str, ...]
    abs_spearman_realism: tuple[float, ...]
    abs_spearman_evac: tuple[float, ...]
    rank_by_realism: tuple[str, ...]
    rank_by_evac: tuple[str, ...]
    rank_combined: tuple[str, ...]
    n_samples: int


def compute_sensitivity(
    samples: npt.NDArray[np.float64],
    objectives: npt.NDArray[np.float64],
) -> SensitivityResult:
    """Compute Spearman rank-correlation sensitivity from a sample / objective matrix.

    For each parameter column in *samples*, computes the absolute Spearman
    rank correlation against each objective column in *objectives*.  A
    ``|rho|`` near 1 indicates strong monotone influence; near 0 indicates
    negligible influence over the sampled range.

    Args:
        samples: Float64 array of shape ``(n, N_PARAMS)`` — the quasi-random
            parameter vectors drawn from the search space.
        objectives: Float64 array of shape ``(n, 2)`` — column 0 is
            ``realism_distance``, column 1 is ``evac_time``.

    Returns:
        A :class:`SensitivityResult` with per-parameter correlations and
        ranked name lists.

    Raises:
        ValueError: If ``samples`` or ``objectives`` have wrong shapes or
            if ``n < 3`` (minimum for a meaningful rank correlation).
    """
    n, d = samples.shape
    if d != N_PARAMS:
        raise ValueError(
            f"samples must have {N_PARAMS} columns, got {d!r}"
        )
    if objectives.shape != (n, 2):
        raise ValueError(
            f"objectives must have shape ({n}, 2), got {objectives.shape!r}"
        )
    if n < 3:
        raise ValueError(f"Need at least 3 samples for rank correlation, got {n!r}")

    rho_realism = np.array(
        [_safe_spearman(samples[:, i], objectives[:, 0]) for i in range(N_PARAMS)],
        dtype=np.float64,
    )
    rho_evac = np.array(
        [_safe_spearman(samples[:, i], objectives[:, 1]) for i in range(N_PARAMS)],
        dtype=np.float64,
    )

    names = list(_FIELD_ORDER)
    rank_realism = sorted(
        range(N_PARAMS), key=lambda i: float(rho_realism[i]), reverse=True
    )
    rank_evac = sorted(
        range(N_PARAMS), key=lambda i: float(rho_evac[i]), reverse=True
    )
    combined = np.maximum(rho_realism, rho_evac)
    rank_combined = sorted(
        range(N_PARAMS), key=lambda i: float(combined[i]), reverse=True
    )

    return SensitivityResult(
        param_names=tuple(names),
        abs_spearman_realism=tuple(float(v) for v in rho_realism),
        abs_spearman_evac=tuple(float(v) for v in rho_evac),
        rank_by_realism=tuple(names[i] for i in rank_realism),
        rank_by_evac=tuple(names[i] for i in rank_evac),
        rank_combined=tuple(names[i] for i in rank_combined),
        n_samples=n,
    )


def run_sensitivity_prepass(
    n_samples: int,
    config: FitnessConfig | None = None,
    seed: int = 0,
    method: Literal["sobol", "lhs"] = "sobol",
) -> tuple[SensitivityResult, list[FitnessResult]]:
    """Evaluate *n_samples* quasi-random candidates and rank parameter influence.

    Draws *n_samples* parameter vectors from the search space, evaluates each
    via :func:`~crowd_evac.optimization.fitness.evaluate_fitness`, then
    computes Spearman rank-correlation sensitivity on the resulting objective
    matrix.  Use the returned :class:`SensitivityResult` to identify parameters
    with negligible influence (fix at default) and to tighten bounds around
    promising regions before the expensive NSGA-II run.

    Parameters with ``|rho|`` < 0.05 on both objectives are typically
    candidates for fixing at their default; the decision is documented in
    ``docs/plan_phase_2.md`` §Results (Step 2.7 sensitivity ranking).

    Args:
        n_samples: Number of quasi-random candidates to evaluate.  Must
            be >= 3 (minimum for rank correlation).  Powers of two are
            recommended when ``method="sobol"``.
        config: Fitness evaluation knobs forwarded to each
            :func:`~crowd_evac.optimization.fitness.evaluate_fitness` call.
            ``None`` uses the default (down-scaled search suite, ``K = 3``
            seeds, all available cores).
        seed: RNG seed for the quasi-random sampler.  Fixed across re-runs
            for reproducibility.
        method: Sampling method, either ``"sobol"`` or ``"lhs"``.

    Returns:
        Tuple ``(sensitivity, fitness_results)`` where *sensitivity* is the
        ranked :class:`SensitivityResult` and *fitness_results* is the
        list of :class:`~crowd_evac.optimization.fitness.FitnessResult` for
        each sample in sampling order.

    Raises:
        ValueError: If ``n_samples < 3``.
    """
    if n_samples < 3:
        raise ValueError(f"n_samples must be >= 3, got {n_samples!r}")

    cfg = config if config is not None else FitnessConfig()
    samples = sample_space(n_samples, method=method, seed=seed)

    fitness_results: list[FitnessResult] = []
    objectives = np.zeros((n_samples, 2), dtype=np.float64)

    for i, row in enumerate(samples):
        params = ForceParams.from_array(row)
        result = evaluate_fitness(params, cfg)
        fitness_results.append(result)
        objectives[i, 0] = result.realism_distance
        objectives[i, 1] = result.evac_time
        logger.info(
            "pre-pass %d/%d: realism=%.4f evac=%.2f stuck=%d",
            i + 1,
            n_samples,
            result.realism_distance,
            result.evac_time,
            result.stuck_count,
        )

    sensitivity = compute_sensitivity(samples, objectives)
    return sensitivity, fitness_results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_spearman(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Return absolute Spearman correlation, or 0.0 when undefined.

    A zero-variance input (all identical values) makes Spearman undefined;
    return 0.0 to treat it as non-influential rather than propagating NaN.

    Args:
        x: First 1-D array.
        y: Second 1-D array of the same length.

    Returns:
        Absolute Spearman rank correlation in ``[0, 1]``.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError as exc:
        raise ImportError(
            "scipy is required for sensitivity analysis; "
            "install it with: pip install 'scipy>=1.11'"
        ) from exc

    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return 0.0
    result = spearmanr(x, y)
    # result[0] == statistic (renamed from .correlation in scipy < 1.7).
    rho = float(result[0])
    return abs(rho) if np.isfinite(rho) else 0.0
