"""Tests for crowd_evac.optimization.select (Phase 2, Step 2.9).

Covers the plan success criteria:
  - selection returns a single ForceParams via the realism-gated rule;
  - the gate never returns a stuck-violating or over-distance point;
  - on a synthetic front, the winner is the minimum-evac-time gated point,
    ties broken by the knee;
  - full-scale validation steps past points that regress (realism/stuck or the
    hazard reroute) and raises when none survive;
  - load_front / write_outcome round-trip; validation suite / config shape.

All simulation calls are mocked, so the suite is fast and headless (no pymoo,
no real harness runs).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization.fitness import FitnessConfig, FitnessResult
from crowd_evac.optimization.select import (
    HazardCheckResult,
    LoadedFrontPoint,
    RejectedPoint,
    SelectionError,
    SelectionOutcome,
    ValidationResult,
    choose,
    default_validation_config,
    gate,
    hazard_check,
    knee_point,
    load_front,
    select_and_validate,
    validate_at_scale,
    validation_suite,
    write_outcome,
)

_SELECT = "crowd_evac.optimization.select"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _point(
    *,
    realism: float,
    evac: float,
    stuck: float = 0.0,
    relaxation_time: float = 0.5,
) -> LoadedFrontPoint:
    """Build a LoadedFrontPoint, tagging params via relaxation_time."""
    params = ForceParams(relaxation_time=relaxation_time)
    return LoadedFrontPoint(
        params=params,
        realism_distance=realism,
        evac_time=evac,
        stuck_count=stuck,
    )


def _fitness(
    realism: float, evac: float, stuck: int, evac_frac: float = 1.0
) -> FitnessResult:
    """Build a FitnessResult standing in for evaluate_fitness output."""
    return FitnessResult(
        objectives=(realism, evac),
        constraints=(float(stuck),),
        realism_distance=realism,
        evac_time=evac,
        stuck_count=stuck,
        evacuated_fraction=evac_frac,
        per_seed_realism=(realism,),
        per_seed_evac_time=(evac,),
        per_seed_stuck=(stuck,),
    )


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


class TestGate:
    """The realism-gated, feasibility-filtered, evac-sorted front filter."""

    def test_sorts_by_evac_time_ascending(self) -> None:
        """Gated points come back fastest-first."""
        front = [
            _point(realism=0.1, evac=80.0),
            _point(realism=0.1, evac=50.0),
            _point(realism=0.1, evac=65.0),
        ]
        evacs = [p.evac_time for p in gate(front, 0.15)]
        assert evacs == [50.0, 65.0, 80.0]

    def test_excludes_infeasible_points(self) -> None:
        """A stuck-violating point is filtered even when it is the fastest."""
        front = [
            _point(realism=0.1, evac=40.0, stuck=2.0),
            _point(realism=0.1, evac=70.0, stuck=0.0),
        ]
        gated = gate(front, 0.15)
        assert len(gated) == 1
        assert gated[0].evac_time == 70.0

    def test_excludes_over_threshold_realism(self) -> None:
        """A point whose realism exceeds the threshold is filtered."""
        front = [_point(realism=0.30, evac=40.0)]
        assert gate(front, 0.15) == []

    def test_threshold_boundary_is_inclusive(self) -> None:
        """realism exactly at the threshold passes (<=)."""
        front = [_point(realism=0.15, evac=40.0)]
        assert len(gate(front, 0.15)) == 1


# ---------------------------------------------------------------------------
# choose
# ---------------------------------------------------------------------------


class TestChoose:
    """The realism-gated selection rule."""

    def test_picks_minimum_evac_time(self) -> None:
        """Among gated points, the fastest is chosen."""
        front = [
            _point(realism=0.05, evac=90.0, relaxation_time=0.9),
            _point(realism=0.10, evac=55.0, relaxation_time=0.4),
        ]
        chosen = choose(front, threshold=0.15)
        assert chosen.relaxation_time == 0.4

    def test_never_returns_infeasible(self) -> None:
        """A fast but stuck-violating point is never selected."""
        front = [
            _point(realism=0.05, evac=30.0, stuck=1.0, relaxation_time=0.3),
            _point(realism=0.05, evac=75.0, stuck=0.0, relaxation_time=0.7),
        ]
        chosen = choose(front, threshold=0.15)
        assert chosen.relaxation_time == 0.7

    def test_evac_tie_broken_by_knee(self) -> None:
        """Equal evac time: the utopia-nearest (lower realism) point wins."""
        front = [
            _point(realism=0.12, evac=50.0, relaxation_time=0.9),
            _point(realism=0.04, evac=50.0, relaxation_time=0.3),
        ]
        chosen = choose(front, threshold=0.15)
        assert chosen.relaxation_time == 0.3

    def test_empty_gate_raises(self) -> None:
        """No gated point raises SelectionError."""
        front = [_point(realism=0.5, evac=40.0)]
        with pytest.raises(SelectionError, match="no front point"):
            choose(front, threshold=0.15)


# ---------------------------------------------------------------------------
# knee_point
# ---------------------------------------------------------------------------


class TestKneePoint:
    """Utopia-nearest knee selection."""

    def test_empty_returns_none(self) -> None:
        """No points yields None."""
        assert knee_point([]) is None

    def test_single_point_returns_itself(self) -> None:
        """One point is its own knee."""
        p = _point(realism=0.2, evac=50.0)
        assert knee_point([p]) is p

    def test_picks_utopia_nearest(self) -> None:
        """The point closest to (0, 0) in normalised space is the knee."""
        far = _point(realism=0.0, evac=100.0, relaxation_time=0.9)
        near = _point(realism=0.2, evac=60.0, relaxation_time=0.5)
        other = _point(realism=1.0, evac=0.0, relaxation_time=0.7)
        knee = knee_point([far, near, other])
        assert knee is near


# ---------------------------------------------------------------------------
# Result-object predicates
# ---------------------------------------------------------------------------


class TestResultPredicates:
    """passed properties on ValidationResult and HazardCheckResult."""

    def test_validation_passes_within_gate(self) -> None:
        """stuck 0 and realism <= threshold passes."""
        vr = ValidationResult(
            ForceParams.defaults(), 0.1, 50.0, 0, 1.0, 0.15
        )
        assert vr.passed is True

    def test_validation_fails_on_stuck(self) -> None:
        """Any stuck agent fails."""
        vr = ValidationResult(
            ForceParams.defaults(), 0.1, 50.0, 1, 1.0, 0.15
        )
        assert vr.passed is False

    def test_validation_fails_over_threshold(self) -> None:
        """realism above the threshold fails."""
        vr = ValidationResult(
            ForceParams.defaults(), 0.2, 50.0, 0, 1.0, 0.15
        )
        assert vr.passed is False

    def test_hazard_passes_clear_no_stuck(self) -> None:
        """No stuck and evac above floor passes."""
        hr = HazardCheckResult("h", 0, 0.98, 0.95)
        assert hr.passed is True

    def test_hazard_fails_on_stuck(self) -> None:
        """A deadlock against the rerouted field fails."""
        hr = HazardCheckResult("h", 3, 0.99, 0.95)
        assert hr.passed is False

    def test_hazard_fails_below_evac_floor(self) -> None:
        """Stranding part of the crowd fails even with no stuck."""
        hr = HazardCheckResult("h", 0, 0.80, 0.95)
        assert hr.passed is False


# ---------------------------------------------------------------------------
# validate_at_scale
# ---------------------------------------------------------------------------


class TestValidateAtScale:
    """Hazard-free full-scale re-scoring."""

    def test_builds_result_from_fitness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fitness output maps onto the ValidationResult fields."""
        monkeypatch.setattr(
            f"{_SELECT}.evaluate_fitness",
            lambda params, cfg: _fitness(0.08, 47.0, 0, 0.99),
        )
        vr = validate_at_scale(
            ForceParams.defaults(),
            FitnessConfig(max_workers=1),
            threshold=0.15,
        )
        assert vr.realism_distance == 0.08
        assert vr.evac_time == 47.0
        assert vr.evacuated_fraction == 0.99
        assert vr.passed is True

    def test_stuck_result_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stuck fitness result yields a failing validation."""
        monkeypatch.setattr(
            f"{_SELECT}.evaluate_fitness",
            lambda params, cfg: _fitness(0.08, 47.0, 4),
        )
        vr = validate_at_scale(
            ForceParams.defaults(), FitnessConfig(max_workers=1)
        )
        assert vr.passed is False


# ---------------------------------------------------------------------------
# hazard_check
# ---------------------------------------------------------------------------


class TestHazardCheck:
    """Hazard-reroute validation against the rerouted field."""

    def _patch_harness(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        stuck: int,
        evac_frac: float,
    ) -> None:
        """Stub rerouted_flow_field, evaluate_batch and stuck_count."""
        monkeypatch.setattr(
            f"{_SELECT}.rerouted_flow_field", lambda *a, **k: object()
        )

        class _Run:
            evacuated_fraction = evac_frac

        monkeypatch.setattr(
            f"{_SELECT}.evaluate_batch",
            lambda *a, **k: [_Run(), _Run()],
        )
        monkeypatch.setattr(
            f"{_SELECT}.stuck_count", lambda result, field: stuck
        )

    def test_clear_reroute_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No stuck and a full clear passes."""
        self._patch_harness(monkeypatch, stuck=0, evac_frac=0.99)
        hr = hazard_check(ForceParams.defaults(), seeds=(101, 102))
        assert hr.passed is True
        assert hr.stuck_count == 0

    def test_deadlock_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deadlock against the rerouted field fails."""
        self._patch_harness(monkeypatch, stuck=2, evac_frac=0.99)
        hr = hazard_check(ForceParams.defaults(), seeds=(101,))
        assert hr.passed is False

    def test_empty_seeds_raises(self) -> None:
        """No seeds raises ValueError."""
        with pytest.raises(ValueError, match="seeds must be non-empty"):
            hazard_check(ForceParams.defaults(), seeds=())


# ---------------------------------------------------------------------------
# select_and_validate
# ---------------------------------------------------------------------------


class TestSelectAndValidate:
    """End-to-end selection with full-scale validation and step-back."""

    def _front(self) -> list[LoadedFrontPoint]:
        """Two gated points; the faster is tagged relaxation_time 0.3."""
        return [
            _point(realism=0.10, evac=45.0, relaxation_time=0.3),
            _point(realism=0.08, evac=60.0, relaxation_time=0.6),
        ]

    def test_first_passing_point_is_chosen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the fastest gated point validates, it wins with no rejections."""
        monkeypatch.setattr(
            f"{_SELECT}.evaluate_fitness",
            lambda params, cfg: _fitness(0.10, 45.0, 0),
        )
        monkeypatch.setattr(
            f"{_SELECT}.hazard_check",
            lambda params, **k: HazardCheckResult("h", 0, 0.99, 0.95),
        )
        outcome = select_and_validate(
            self._front(), validation_config=FitnessConfig(max_workers=1)
        )
        assert outcome.chosen.relaxation_time == 0.3
        assert outcome.rejected == ()
        assert outcome.hazard is not None

    def test_steps_past_realism_regression(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A point that regresses on realism/stuck is stepped past."""
        def _fit(params: ForceParams, cfg: FitnessConfig) -> FitnessResult:
            # The fast (0.3) point regresses (stuck); the slower (0.6) is clean.
            return (
                _fitness(0.10, 45.0, 5)
                if params.relaxation_time == 0.3
                else _fitness(0.08, 60.0, 0)
            )

        monkeypatch.setattr(f"{_SELECT}.evaluate_fitness", _fit)
        monkeypatch.setattr(
            f"{_SELECT}.hazard_check",
            lambda params, **k: HazardCheckResult("h", 0, 0.99, 0.95),
        )
        outcome = select_and_validate(
            self._front(), validation_config=FitnessConfig(max_workers=1)
        )
        assert outcome.chosen.relaxation_time == 0.6
        assert len(outcome.rejected) == 1
        assert outcome.rejected[0].hazard is None

    def test_steps_past_hazard_regression(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A point clean on the suite but failing the hazard reroute is skipped."""
        monkeypatch.setattr(
            f"{_SELECT}.evaluate_fitness",
            lambda params, cfg: _fitness(0.09, 45.0, 0),
        )

        def _haz(params: ForceParams, **kwargs: object) -> HazardCheckResult:
            stuck = 4 if params.relaxation_time == 0.3 else 0
            return HazardCheckResult("h", stuck, 0.99, 0.95)

        monkeypatch.setattr(f"{_SELECT}.hazard_check", _haz)
        outcome = select_and_validate(
            self._front(), validation_config=FitnessConfig(max_workers=1)
        )
        assert outcome.chosen.relaxation_time == 0.6
        assert outcome.rejected[0].reason == "hazard reroute regressed"
        assert outcome.rejected[0].hazard is not None

    def test_all_regress_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every gated point regresses, SelectionError is raised."""
        monkeypatch.setattr(
            f"{_SELECT}.evaluate_fitness",
            lambda params, cfg: _fitness(0.10, 45.0, 7),
        )
        monkeypatch.setattr(
            f"{_SELECT}.hazard_check",
            lambda params, **k: HazardCheckResult("h", 0, 0.99, 0.95),
        )
        with pytest.raises(SelectionError, match="regressed at full scale"):
            select_and_validate(
                self._front(), validation_config=FitnessConfig(max_workers=1)
            )

    def test_hazard_disabled_skips_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """hazard_scenario=None yields a winner with no hazard result."""
        monkeypatch.setattr(
            f"{_SELECT}.evaluate_fitness",
            lambda params, cfg: _fitness(0.10, 45.0, 0),
        )

        def _boom(*a: object, **k: object) -> HazardCheckResult:
            raise AssertionError("hazard_check must not run when disabled")

        monkeypatch.setattr(f"{_SELECT}.hazard_check", _boom)
        outcome = select_and_validate(
            self._front(),
            validation_config=FitnessConfig(max_workers=1),
            hazard_scenario=None,
        )
        assert outcome.hazard is None

    def test_empty_gate_raises(self) -> None:
        """A front with no gated point raises before any validation."""
        front = [_point(realism=0.9, evac=40.0)]
        with pytest.raises(SelectionError, match="no front point"):
            select_and_validate(front, hazard_scenario=None)


# ---------------------------------------------------------------------------
# load_front / write_outcome
# ---------------------------------------------------------------------------


class TestLoadFront:
    """Deserialising a front.json into LoadedFrontPoint objects."""

    def _front_payload(self) -> dict[str, object]:
        """A minimal front.json payload with one point."""
        return {
            "pop_size": 8,
            "n_gen_completed": 3,
            "front": [
                {
                    "params": _as_param_dict(ForceParams.defaults()),
                    "realism_distance": 0.12,
                    "evac_time": 55.0,
                    "stuck_count": 0.0,
                    "feasible": True,
                }
            ],
        }

    def test_round_trip(self, tmp_path: Path) -> None:
        """A written front payload loads back into a LoadedFrontPoint."""
        path = tmp_path / "front.json"
        path.write_text(json.dumps(self._front_payload()), encoding="utf-8")
        points = load_front(path)
        assert len(points) == 1
        assert points[0].realism_distance == 0.12
        assert points[0].evac_time == 55.0
        assert isinstance(points[0].params, ForceParams)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_front(tmp_path / "absent.json")

    def test_malformed_front_raises(self, tmp_path: Path) -> None:
        """A front entry missing required keys raises SelectionError."""
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"front": [{"realism_distance": 0.1}]}),
            encoding="utf-8",
        )
        with pytest.raises(SelectionError, match="malformed front"):
            load_front(path)


class TestWriteOutcome:
    """Serialising a SelectionOutcome to JSON."""

    def _outcome(self) -> SelectionOutcome:
        """Build a SelectionOutcome with one rejection for serialisation."""
        point = _point(realism=0.1, evac=50.0)
        rejected_pt = _point(realism=0.09, evac=42.0)
        vr = ValidationResult(point.params, 0.1, 50.0, 0, 0.99, 0.15)
        rej_vr = ValidationResult(rejected_pt.params, 0.09, 42.0, 3, 0.97, 0.15)
        return SelectionOutcome(
            chosen=point.params,
            chosen_point=point,
            validation=vr,
            hazard=HazardCheckResult("hazard_lecture_hall", 0, 0.98, 0.95),
            knee=point,
            rejected=(
                RejectedPoint(rejected_pt, rej_vr, None, "stuck regressed"),
            ),
            threshold=0.15,
        )

    def test_writes_parseable_json(self, tmp_path: Path) -> None:
        """write_outcome emits JSON with the expected top-level keys."""
        path = tmp_path / "chosen_weights.json"
        write_outcome(path, self._outcome())
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["threshold"] == 0.15
        assert "relaxation_time" in payload["chosen_params"]
        assert payload["validation"]["passed"] is True
        assert payload["hazard"]["scenario"] == "hazard_lecture_hall"
        assert payload["n_rejected"] == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """A nested output path has its parents created."""
        path = tmp_path / "nested" / "deep" / "chosen.json"
        write_outcome(path, self._outcome())
        assert path.exists()


# ---------------------------------------------------------------------------
# validation_suite / default_validation_config
# ---------------------------------------------------------------------------


class TestValidationConfig:
    """Full-scale validation suite and config."""

    def test_suite_includes_full_scale_lecture_hall(self) -> None:
        """The validation suite carries the 150-agent lecture hall."""
        names = {s.name for s in validation_suite()}
        assert "lecture_hall" in names
        hall = next(s for s in validation_suite() if s.name == "lecture_hall")
        assert hall.full_agent_count == 150

    def test_config_seeds_disjoint_from_search(self) -> None:
        """Held-out validation seeds do not overlap the default search seeds."""
        cfg = default_validation_config()
        assert set(cfg.seeds).isdisjoint({0, 1, 2, 3, 4})

    def test_config_uses_full_scale_scenarios(self) -> None:
        """The config's scenarios are the full-scale validation suite."""
        cfg = default_validation_config()
        suite_names = {s.name for s in validation_suite()}
        cfg_names = {s.name for s in cfg.scenarios}
        assert cfg_names == suite_names

    def test_custom_seeds_and_workers_respected(self) -> None:
        """Explicit seeds and worker count flow into the config."""
        cfg = default_validation_config(seeds=(7, 8), max_workers=2)
        assert cfg.seeds == (7, 8)
        assert cfg.max_workers == 2


def _as_param_dict(params: ForceParams) -> dict[str, float]:
    """Return a ForceParams as a plain field dict (mirrors dataclasses.asdict)."""
    import dataclasses

    return dataclasses.asdict(params)
