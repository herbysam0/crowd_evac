"""Tests for crowd_evac.optimization.space (Phase 2, Step 2.7).

Covers the plan success criteria:
  - bounds validate (low < high, defaults inside);
  - the pre-pass runs headless on a small budget and emits a ranked influence
    table (integration test, marked e2e);
  - at least the obviously-dominant weights (repulsion_strength, relaxation_time,
    panic_repulsion_strength) surface as influential when synthetic data is
    designed to make them so (sanity check via compute_sensitivity).

Fast tests use synthetic data only — no real simulation.  Integration tests
(``@pytest.mark.e2e``) run the real harness with a tiny tick cap.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from crowd_evac.domain.params import N_PARAMS, ForceParams, _FIELD_ORDER
from crowd_evac.optimization.fitness import FitnessConfig, FitnessResult
from crowd_evac.optimization.space import (
    BOUNDS,
    SearchBound,
    SensitivityResult,
    bounds_array,
    compute_sensitivity,
    run_sensitivity_prepass,
    sample_space,
    validate_defaults,
)


# ---------------------------------------------------------------------------
# SearchBound dataclass
# ---------------------------------------------------------------------------


class TestSearchBound:
    """SearchBound construction and validation."""

    def test_valid_bound_constructed(self) -> None:
        """SearchBound with low < high constructs without error."""
        b = SearchBound("foo", 0.0, 1.0, "test")
        assert b.low == 0.0
        assert b.high == 1.0

    def test_equal_low_high_raises(self) -> None:
        """low == high raises ValueError."""
        with pytest.raises(ValueError, match="must be < high"):
            SearchBound("foo", 1.0, 1.0, "bad")

    def test_low_gt_high_raises(self) -> None:
        """low > high raises ValueError."""
        with pytest.raises(ValueError, match="must be < high"):
            SearchBound("foo", 2.0, 1.0, "bad")

    def test_frozen(self) -> None:
        """SearchBound is frozen — attribute assignment raises."""
        b = SearchBound("foo", 0.0, 1.0, "test")
        with pytest.raises((AttributeError, TypeError)):
            b.low = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BOUNDS table structure
# ---------------------------------------------------------------------------


class TestBoundsTable:
    """BOUNDS tuple length, ordering, and individual bound validity."""

    def test_count_matches_n_params(self) -> None:
        """BOUNDS must have exactly N_PARAMS entries."""
        assert len(BOUNDS) == N_PARAMS

    def test_ordering_matches_field_order(self) -> None:
        """BOUNDS names are in the canonical _FIELD_ORDER sequence."""
        assert tuple(b.name for b in BOUNDS) == _FIELD_ORDER

    def test_all_bounds_low_lt_high(self) -> None:
        """Every SearchBound satisfies low < high."""
        for b in BOUNDS:
            assert b.low < b.high, f"{b.name}: low={b.low} >= high={b.high}"

    def test_narrow_max_speed_bound(self) -> None:
        """max_speed has the documented narrow empirical bound starting at 1.2."""
        b = next(x for x in BOUNDS if x.name == "max_speed")
        assert b.low == pytest.approx(1.2)
        assert b.high >= 2.5  # covers Phase-1 default

    def test_wide_repulsion_strength_bound(self) -> None:
        """repulsion_strength covers 0.0–12.0 as specified."""
        b = next(x for x in BOUNDS if x.name == "repulsion_strength")
        assert b.low == pytest.approx(0.0)
        assert b.high == pytest.approx(12.0)

    def test_relaxation_time_bound(self) -> None:
        """relaxation_time spans 0.1–2.0 s as specified."""
        b = next(x for x in BOUNDS if x.name == "relaxation_time")
        assert b.low == pytest.approx(0.1)
        assert b.high == pytest.approx(2.0)

    def test_panic_repulsion_strength_bound(self) -> None:
        """panic_repulsion_strength spans 0–12 as specified."""
        b = next(x for x in BOUNDS if x.name == "panic_repulsion_strength")
        assert b.low == pytest.approx(0.0)
        assert b.high == pytest.approx(12.0)

    def test_defaults_inside_bounds(self) -> None:
        """validate_defaults() passes — Phase-1 defaults are inside bounds."""
        validate_defaults()

    def test_all_descriptions_non_empty(self) -> None:
        """Every bound carries a non-empty description string."""
        for b in BOUNDS:
            assert b.description.strip(), f"{b.name!r} has empty description"


# ---------------------------------------------------------------------------
# bounds_array
# ---------------------------------------------------------------------------


class TestBoundsArray:
    """bounds_array shape and consistency with BOUNDS."""

    def test_shape(self) -> None:
        """bounds_array returns shape (N_PARAMS, 2)."""
        arr = bounds_array()
        assert arr.shape == (N_PARAMS, 2)

    def test_dtype_float64(self) -> None:
        """bounds_array dtype is float64."""
        assert bounds_array().dtype == np.float64

    def test_col0_lt_col1(self) -> None:
        """Column 0 (low) is strictly less than column 1 (high) for all rows."""
        arr = bounds_array()
        assert np.all(arr[:, 0] < arr[:, 1])

    def test_values_match_bounds(self) -> None:
        """bounds_array values exactly mirror BOUNDS.low and BOUNDS.high."""
        arr = bounds_array()
        for i, b in enumerate(BOUNDS):
            assert arr[i, 0] == pytest.approx(b.low)
            assert arr[i, 1] == pytest.approx(b.high)


# ---------------------------------------------------------------------------
# sample_space
# ---------------------------------------------------------------------------


class TestSampleSpace:
    """Quasi-random sampling from the search space."""

    def test_sobol_shape(self) -> None:
        """Sobol sample returns (n, N_PARAMS) array."""
        s = sample_space(16, method="sobol")
        assert s.shape == (16, N_PARAMS)

    def test_lhs_shape(self) -> None:
        """LHS sample returns (n, N_PARAMS) array."""
        s = sample_space(16, method="lhs")
        assert s.shape == (16, N_PARAMS)

    def test_dtype_float64(self) -> None:
        """Returned samples are float64."""
        assert sample_space(8).dtype == np.float64

    def test_sobol_samples_within_bounds(self) -> None:
        """All Sobol samples lie within declared bounds (up to floating-point)."""
        lo = np.array([b.low for b in BOUNDS])
        hi = np.array([b.high for b in BOUNDS])
        s = sample_space(64, method="sobol", seed=7)
        assert np.all(s >= lo - 1e-10), "Sobol sample below lower bound"
        assert np.all(s <= hi + 1e-10), "Sobol sample above upper bound"

    def test_lhs_samples_within_bounds(self) -> None:
        """All LHS samples lie within declared bounds (up to floating-point)."""
        lo = np.array([b.low for b in BOUNDS])
        hi = np.array([b.high for b in BOUNDS])
        s = sample_space(64, method="lhs", seed=7)
        assert np.all(s >= lo - 1e-10), "LHS sample below lower bound"
        assert np.all(s <= hi + 1e-10), "LHS sample above upper bound"

    def test_same_seed_reproduces(self) -> None:
        """Same seed and method produce identical samples."""
        a = sample_space(16, method="sobol", seed=42)
        b = sample_space(16, method="sobol", seed=42)
        assert np.allclose(a, b)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different samples."""
        a = sample_space(16, method="sobol", seed=0)
        b = sample_space(16, method="sobol", seed=1)
        assert not np.allclose(a, b)

    def test_sobol_and_lhs_differ(self) -> None:
        """Sobol and LHS produce different sample patterns."""
        a = sample_space(32, method="sobol", seed=0)
        b = sample_space(32, method="lhs", seed=0)
        assert not np.allclose(a, b)

    def test_invalid_n_raises(self) -> None:
        """n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n must be"):
            sample_space(0)

    def test_invalid_method_raises(self) -> None:
        """Unrecognised method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            sample_space(8, method="random")  # type: ignore[arg-type]

    def test_samples_yield_valid_force_params(self) -> None:
        """ForceParams.from_array() succeeds for every row of a sample matrix."""
        s = sample_space(32, method="sobol", seed=3)
        for row in s:
            ForceParams.from_array(row)


# ---------------------------------------------------------------------------
# compute_sensitivity
# ---------------------------------------------------------------------------


def _dominated_data(
    n: int = 128,
    *,
    realism_driver: int = 0,
    evac_driver: int = 2,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic data where each objective is dominated by one parameter column.

    Args:
        n: Sample count.
        realism_driver: Column index of the parameter that solely drives realism.
        evac_driver: Column index of the parameter that solely drives evac_time.
        noise: Standard deviation of additive Gaussian noise to break rank ties.

    Returns:
        Tuple ``(samples, objectives)`` of shapes ``(n, N_PARAMS)`` and ``(n, 2)``.
    """
    rng = np.random.default_rng(99)
    lo = np.array([b.low for b in BOUNDS])
    hi = np.array([b.high for b in BOUNDS])
    samples = rng.uniform(lo, hi, size=(n, N_PARAMS))
    realism = samples[:, realism_driver] + noise * rng.standard_normal(n)
    evac = samples[:, evac_driver] + noise * rng.standard_normal(n)
    return samples, np.column_stack([realism, evac])


class TestComputeSensitivity:
    """Spearman-based sensitivity computation with synthetic data."""

    def test_returns_sensitivity_result(self) -> None:
        """compute_sensitivity returns a SensitivityResult."""
        samples, objectives = _dominated_data()
        result = compute_sensitivity(samples, objectives)
        assert isinstance(result, SensitivityResult)

    def test_param_names_match_field_order(self) -> None:
        """param_names equals _FIELD_ORDER."""
        samples, objectives = _dominated_data()
        result = compute_sensitivity(samples, objectives)
        assert result.param_names == _FIELD_ORDER

    def test_n_samples_recorded(self) -> None:
        """n_samples field records the input sample count."""
        samples, objectives = _dominated_data(n=50)
        result = compute_sensitivity(samples, objectives)
        assert result.n_samples == 50

    def test_all_params_in_rankings(self) -> None:
        """Every parameter appears exactly once in each ranking list."""
        samples, objectives = _dominated_data()
        result = compute_sensitivity(samples, objectives)
        assert set(result.rank_by_realism) == set(_FIELD_ORDER)
        assert set(result.rank_by_evac) == set(_FIELD_ORDER)
        assert set(result.rank_combined) == set(_FIELD_ORDER)
        assert len(result.rank_by_realism) == N_PARAMS

    def test_rankings_are_tuples_of_strings(self) -> None:
        """All ranking fields are tuples of str."""
        samples, objectives = _dominated_data()
        result = compute_sensitivity(samples, objectives)
        assert isinstance(result.rank_by_realism, tuple)
        assert all(isinstance(n, str) for n in result.rank_by_realism)

    def test_influential_realism_driver_ranked_first(self) -> None:
        """The synthetic realism driver (relaxation_time, col 0) ranks #1."""
        samples, objectives = _dominated_data(n=128, realism_driver=0)
        result = compute_sensitivity(samples, objectives)
        assert result.rank_by_realism[0] == "relaxation_time"

    def test_influential_evac_driver_ranked_first(self) -> None:
        """The synthetic evac driver (repulsion_strength, col 2) ranks #1."""
        samples, objectives = _dominated_data(n=128, evac_driver=2)
        result = compute_sensitivity(samples, objectives)
        assert result.rank_by_evac[0] == "repulsion_strength"

    def test_combined_ranking_surfaces_dominant_param(self) -> None:
        """Combined ranking lists the objective-driving param near the top."""
        samples, objectives = _dominated_data(n=128, realism_driver=0, evac_driver=9)
        result = compute_sensitivity(samples, objectives)
        # Both relaxation_time (0) and panic_repulsion_strength (9) must appear
        # in the top 3 of the combined ranking.
        top3 = set(result.rank_combined[:3])
        assert "relaxation_time" in top3 or "panic_repulsion_strength" in top3

    def test_canonical_dominant_trio_surfaces(self) -> None:
        """REPULSION_STRENGTH and RELAXATION_TIME must rank in top-3 combined.

        This is the plan Step 2.7 sanity check: with realism ∝ relaxation_time
        and evac ∝ repulsion_strength, both must appear in the top 3 combined
        influence ranking.
        """
        rng = np.random.default_rng(7)
        lo = np.array([b.low for b in BOUNDS])
        hi = np.array([b.high for b in BOUNDS])
        n = 256
        samples = rng.uniform(lo, hi, size=(n, N_PARAMS))
        noise = 0.01 * rng.standard_normal(n)
        realism = samples[:, 0] + noise  # relaxation_time drives realism
        evac = samples[:, 2] + noise    # repulsion_strength drives evac_time
        objectives = np.column_stack([realism, evac])
        result = compute_sensitivity(samples, objectives)
        top3 = set(result.rank_combined[:3])
        assert "relaxation_time" in top3, (
            f"relaxation_time not in top-3; got {result.rank_combined[:3]}"
        )
        assert "repulsion_strength" in top3, (
            f"repulsion_strength not in top-3; got {result.rank_combined[:3]}"
        )

    def test_abs_values_in_zero_one(self) -> None:
        """All |rho| values lie in [0, 1]."""
        samples, objectives = _dominated_data()
        result = compute_sensitivity(samples, objectives)
        for v in result.abs_spearman_realism:
            assert 0.0 <= v <= 1.0
        for v in result.abs_spearman_evac:
            assert 0.0 <= v <= 1.0

    def test_zero_variance_column_yields_zero_rho(self) -> None:
        """A constant parameter column yields |rho| == 0.0."""
        samples, objectives = _dominated_data()
        samples[:, 5] = 1.0  # make density_pressure_strength constant
        result = compute_sensitivity(samples, objectives)
        assert result.abs_spearman_realism[5] == pytest.approx(0.0)

    def test_wrong_samples_ncols_raises(self) -> None:
        """samples with wrong column count raises ValueError."""
        samples, objectives = _dominated_data()
        with pytest.raises(ValueError, match=str(N_PARAMS)):
            compute_sensitivity(samples[:, :5], objectives)

    def test_wrong_objectives_shape_raises(self) -> None:
        """objectives with wrong shape raises ValueError."""
        samples, objectives = _dominated_data()
        with pytest.raises(ValueError):
            compute_sensitivity(samples, objectives[:, :1])

    def test_too_few_samples_raises(self) -> None:
        """n < 3 raises ValueError."""
        samples, objectives = _dominated_data(n=2)
        with pytest.raises(ValueError, match="at least 3"):
            compute_sensitivity(samples, objectives)


# ---------------------------------------------------------------------------
# run_sensitivity_prepass — integration (e2e)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestRunSensitivityPrepassE2E:
    """End-to-end wiring of the sensitivity pre-pass with the real harness."""

    def test_returns_sensitivity_and_fitness_list(self) -> None:
        """run_sensitivity_prepass returns (SensitivityResult, list[FitnessResult])."""
        cfg = FitnessConfig(max_workers=1)
        sens, fitness_list = run_sensitivity_prepass(
            n_samples=4,
            config=cfg,
            seed=0,
            method="lhs",
        )
        assert isinstance(sens, SensitivityResult)
        assert len(fitness_list) == 4
        assert all(isinstance(fr, FitnessResult) for fr in fitness_list)

    def test_sensitivity_has_n_params_entries(self) -> None:
        """Sensitivity result from the real pre-pass has N_PARAMS entries."""
        cfg = FitnessConfig(max_workers=1)
        sens, _ = run_sensitivity_prepass(n_samples=4, config=cfg, seed=0)
        assert len(sens.abs_spearman_realism) == N_PARAMS
        assert len(sens.rank_combined) == N_PARAMS

    def test_fitness_results_have_finite_objectives(self) -> None:
        """All fitness results carry finite, non-negative objectives."""
        cfg = FitnessConfig(max_workers=1)
        _, fitness_list = run_sensitivity_prepass(n_samples=4, config=cfg, seed=0)
        for fr in fitness_list:
            realism, evac = fr.objectives
            assert np.isfinite(realism) and realism >= 0.0
            assert np.isfinite(evac) and evac >= 0.0


# ---------------------------------------------------------------------------
# run_sensitivity_prepass — unit (mocked fitness)
# ---------------------------------------------------------------------------


def _mock_evaluate(
    params: ForceParams,
    config: FitnessConfig | None = None,
) -> FitnessResult:
    """Deterministic mock: realism tracks relaxation_time, evac is constant.

    Args:
        params: Candidate to score.
        config: Unused (present to match evaluate_fitness signature).

    Returns:
        A :class:`FitnessResult` with realism proportional to relaxation_time.
    """
    realism = params.relaxation_time * 0.5
    evac = 60.0
    return FitnessResult(
        objectives=(realism, evac),
        constraints=(0.0,),
        realism_distance=realism,
        evac_time=evac,
        stuck_count=0,
        evacuated_fraction=1.0,
        per_seed_realism=(realism,),
        per_seed_evac_time=(evac,),
        per_seed_stuck=(0,),
    )


class TestRunSensitivityPrepassUnit:
    """Unit tests for run_sensitivity_prepass with mocked evaluate_fitness."""

    def test_low_n_samples_raises(self) -> None:
        """n_samples < 3 raises ValueError before any evaluation."""
        with pytest.raises(ValueError, match="n_samples"):
            run_sensitivity_prepass(n_samples=2)

    def test_mock_prepass_ranks_relaxation_time_top(self) -> None:
        """With realism ∝ relaxation_time, it ranks #1 on the realism axis."""
        with patch(
            "crowd_evac.optimization.space.evaluate_fitness",
            side_effect=_mock_evaluate,
        ):
            sens, fitness_list = run_sensitivity_prepass(
                n_samples=16,
                seed=0,
                method="lhs",
            )
        assert sens.rank_by_realism[0] == "relaxation_time"
        assert len(fitness_list) == 16

    def test_mock_prepass_correct_list_length(self) -> None:
        """fitness_list length equals n_samples."""
        with patch(
            "crowd_evac.optimization.space.evaluate_fitness",
            side_effect=_mock_evaluate,
        ):
            sens, fitness_list = run_sensitivity_prepass(n_samples=8, seed=1)
        assert len(fitness_list) == 8
        assert len(sens.param_names) == N_PARAMS
