"""Tests for crowd_evac.optimization.nsga (Phase 2, Step 2.8).

Covers the plan success criteria:
  - on a tiny budget (small pop, few generations, cheap mocked fitness) the
    driver produces a valid non-dominated set and a resumable checkpoint;
  - constraint-violating individuals are dominated out of the final front;
  - the problem exposes 2 objectives + 1 ``g(x) <= 0`` constraint of the right
    shape and sign;
  - config validation, Pareto-point feasibility, front extraction, and JSON
    serialisation behave as specified.

``pymoo`` is an optional Phase-2 dependency; the whole module is skipped when it
is not installed (the search cannot run without it).  Fitness is mocked so these
tests never touch the real simulation harness — they stay fast and headless.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("pymoo")  # noqa: E402 — optional Phase-2 search dependency

from crowd_evac.domain.params import N_PARAMS, ForceParams  # noqa: E402
from crowd_evac.optimization.fitness import (  # noqa: E402
    FitnessConfig,
    FitnessResult,
)
from crowd_evac.optimization.nsga import (  # noqa: E402
    CHECKPOINT_FILENAME,
    EvacuationProblem,
    NSGAConfig,
    NSGAResult,
    ParetoPoint,
    detect_devices,
    load_checkpoint,
    run_nsga,
    save_checkpoint,
    write_front,
    _extract_front,
)
from crowd_evac.optimization.space import bounds_array  # noqa: E402


# ---------------------------------------------------------------------------
# Cheap mock fitness — conflicting objectives + an infeasible sub-region
# ---------------------------------------------------------------------------


def _mock_fitness(
    params: ForceParams,
    config: FitnessConfig | None = None,
) -> FitnessResult:
    """Deterministic, sim-free fitness with a real trade-off and a bad region.

    realism rises with ``relaxation_time`` while evac falls with it (a genuine
    conflict, so a non-trivial front exists).  A large ``repulsion_strength``
    marks an infeasible sub-region (``stuck_count > 0``) used to verify NSGA-II
    dominates constraint violators out.

    Args:
        params: Candidate to score.
        config: Unused (present to match ``evaluate_fitness`` signature).

    Returns:
        A :class:`FitnessResult` for the candidate.
    """
    realism = float(params.relaxation_time)
    evac = float(2.1 - params.relaxation_time)
    stuck = 5 if params.repulsion_strength > 11.0 else 0
    return FitnessResult(
        objectives=(realism, evac),
        constraints=(float(stuck),),
        realism_distance=realism,
        evac_time=evac,
        stuck_count=stuck,
        evacuated_fraction=1.0,
        per_seed_realism=(realism,),
        per_seed_evac_time=(evac,),
        per_seed_stuck=(stuck,),
    )


def _midpoint_population(n: int) -> np.ndarray:
    """Return *n* identical rows at the centre of every search bound.

    Args:
        n: Number of rows.

    Returns:
        Array of shape ``(n, N_PARAMS)`` inside the search bounds.
    """
    bounds = bounds_array()
    mid = (bounds[:, 0] + bounds[:, 1]) / 2.0
    return np.tile(mid, (n, 1))


# ---------------------------------------------------------------------------
# NSGAConfig
# ---------------------------------------------------------------------------


class TestNSGAConfig:
    """NSGAConfig construction and validation."""

    def test_defaults_construct(self) -> None:
        """Default config constructs with documented values."""
        cfg = NSGAConfig()
        assert cfg.pop_size == 64
        assert cfg.n_gen == 80
        assert isinstance(cfg.fitness_config, FitnessConfig)

    def test_pop_size_too_small_raises(self) -> None:
        """pop_size < 2 raises ValueError."""
        with pytest.raises(ValueError, match="pop_size"):
            NSGAConfig(pop_size=1)

    def test_n_gen_too_small_raises(self) -> None:
        """n_gen < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_gen"):
            NSGAConfig(n_gen=0)

    def test_checkpoint_every_too_small_raises(self) -> None:
        """checkpoint_every < 1 raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_every"):
            NSGAConfig(checkpoint_every=0)

    def test_frozen(self) -> None:
        """NSGAConfig is frozen — attribute assignment raises."""
        cfg = NSGAConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.pop_size = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ParetoPoint
# ---------------------------------------------------------------------------


class TestParetoPoint:
    """ParetoPoint feasibility logic."""

    def test_feasible_when_stuck_zero(self) -> None:
        """stuck_count == 0 is feasible."""
        p = ParetoPoint(ForceParams.defaults(), 0.1, 50.0, 0.0)
        assert p.is_feasible is True

    def test_infeasible_when_stuck_positive(self) -> None:
        """stuck_count > 0 is infeasible."""
        p = ParetoPoint(ForceParams.defaults(), 0.1, 50.0, 3.0)
        assert p.is_feasible is False

    def test_feasible_boundary_is_inclusive(self) -> None:
        """The g <= 0 boundary (exactly 0) is feasible."""
        p = ParetoPoint(ForceParams.defaults(), 0.0, 0.0, 0.0)
        assert p.is_feasible is True


# ---------------------------------------------------------------------------
# EvacuationProblem
# ---------------------------------------------------------------------------


class TestEvacuationProblem:
    """The pymoo Problem wrapper shape and evaluation."""

    def test_problem_dimensions(self) -> None:
        """Problem declares N_PARAMS vars, 2 objectives, 1 constraint."""
        problem = EvacuationProblem(FitnessConfig(max_workers=1))
        assert problem.n_var == N_PARAMS
        assert problem.n_obj == 2
        assert problem.n_ieq_constr == 1

    def test_bounds_match_search_space(self) -> None:
        """Problem xl/xu match the search-space bounds."""
        problem = EvacuationProblem(FitnessConfig(max_workers=1))
        bounds = bounds_array()
        assert np.allclose(problem.xl, bounds[:, 0])
        assert np.allclose(problem.xu, bounds[:, 1])

    def test_evaluate_fills_f_and_g(self) -> None:
        """_evaluate writes (n, 2) F and (n, 1) G from the fitness function."""
        problem = EvacuationProblem(FitnessConfig(max_workers=1))
        x = _midpoint_population(3)
        out: dict[str, Any] = {}
        with patch(
            "crowd_evac.optimization.nsga.evaluate_fitness",
            side_effect=_mock_fitness,
        ):
            problem._evaluate(x, out)
        assert out["F"].shape == (3, 2)
        assert out["G"].shape == (3, 1)
        # Midpoint relaxation_time == (0.1 + 2.0)/2 == 1.05 → realism objective.
        assert out["F"][0, 0] == pytest.approx(1.05)

    def test_evaluate_signs(self) -> None:
        """Constraint column carries the stuck_count from the fitness result."""
        problem = EvacuationProblem(FitnessConfig(max_workers=1))
        x = _midpoint_population(1)
        out: dict[str, Any] = {}
        with patch(
            "crowd_evac.optimization.nsga.evaluate_fitness",
            side_effect=_mock_fitness,
        ):
            problem._evaluate(x, out)
        # Midpoint repulsion_strength == 6.0 (< 11) → feasible (g == 0).
        assert out["G"][0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# detect_devices
# ---------------------------------------------------------------------------


class TestDetectDevices:
    """Device detection for run logging."""

    def test_reports_cpu_count(self) -> None:
        """cpu_count is a positive int."""
        devices = detect_devices()
        assert isinstance(devices["cpu_count"], int)
        assert devices["cpu_count"] >= 1

    def test_reports_gpu_key(self) -> None:
        """gpu key is present (typically 'none' on CI)."""
        assert "gpu" in detect_devices()


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------


class TestCheckpointIO:
    """save_checkpoint / load_checkpoint round-trip and errors."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """A pickled object reloads equal to the original."""
        path = tmp_path / CHECKPOINT_FILENAME
        save_checkpoint(path, {"gen": 7, "data": [1, 2, 3]})
        assert load_checkpoint(path) == {"gen": 7, "data": [1, 2, 3]}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_checkpoint creates missing parent directories."""
        path = tmp_path / "nested" / "deep" / CHECKPOINT_FILENAME
        save_checkpoint(path, {"x": 1})
        assert path.exists()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "absent.pkl")


# ---------------------------------------------------------------------------
# _extract_front
# ---------------------------------------------------------------------------


class TestExtractFront:
    """Front extraction from a pymoo-style result object."""

    def test_sorted_by_realism(self) -> None:
        """Points are returned ascending by realism_distance."""
        x = _midpoint_population(3)
        result = SimpleNamespace(
            X=x,
            F=np.array([[0.9, 1.1], [0.2, 1.9], [0.5, 1.5]]),
            G=np.array([[0.0], [0.0], [0.0]]),
        )
        front = _extract_front(result)
        realisms = [p.realism_distance for p in front]
        assert realisms == sorted(realisms)
        assert len(front) == 3

    def test_none_x_yields_empty(self) -> None:
        """A result with no solutions yields an empty front."""
        result = SimpleNamespace(X=None, F=None, G=None)
        assert _extract_front(result) == ()

    def test_single_solution_reshaped(self) -> None:
        """A 1-D single-solution result is handled (atleast_2d)."""
        x = _midpoint_population(1)[0]  # 1-D row
        result = SimpleNamespace(
            X=x, F=np.array([0.3, 1.7]), G=np.array([0.0])
        )
        front = _extract_front(result)
        assert len(front) == 1
        assert front[0].realism_distance == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# write_front
# ---------------------------------------------------------------------------


class TestWriteFront:
    """JSON serialisation of a completed run."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        """write_front emits parseable JSON with the expected top-level keys."""
        result = NSGAResult(
            front=(ParetoPoint(ForceParams.defaults(), 0.2, 55.0, 0.0),),
            n_gen_completed=3,
            pop_size=8,
        )
        path = tmp_path / "front.json"
        write_front(path, result)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["pop_size"] == 8
        assert payload["n_gen_completed"] == 3
        assert payload["n_points"] == 1
        assert payload["front"][0]["feasible"] is True
        assert "relaxation_time" in payload["front"][0]["params"]


# ---------------------------------------------------------------------------
# run_nsga — driver on a tiny mocked budget
# ---------------------------------------------------------------------------


class TestRunNSGA:
    """End-to-end driver with cheap mocked fitness (no real simulation)."""

    def _tiny_config(self, out: Path) -> NSGAConfig:
        """Build a fast config: small pop, few gens, single worker.

        Args:
            out: Checkpoint / output directory.

        Returns:
            A tiny :class:`NSGAConfig` for fast runs.
        """
        return NSGAConfig(
            pop_size=8,
            n_gen=5,
            seed=1,
            fitness_config=FitnessConfig(max_workers=1),
            checkpoint_dir=out,
        )

    def test_produces_nondominated_front(self, tmp_path: Path) -> None:
        """The driver returns a non-empty front and writes a checkpoint."""
        cfg = self._tiny_config(tmp_path)
        with patch(
            "crowd_evac.optimization.nsga.evaluate_fitness",
            side_effect=_mock_fitness,
        ):
            result = run_nsga(cfg)
        assert isinstance(result, NSGAResult)
        assert len(result.front) >= 1
        assert (tmp_path / CHECKPOINT_FILENAME).exists()

    def test_front_is_feasible(self, tmp_path: Path) -> None:
        """Constraint-violating individuals are dominated out of the front."""
        cfg = self._tiny_config(tmp_path)
        with patch(
            "crowd_evac.optimization.nsga.evaluate_fitness",
            side_effect=_mock_fitness,
        ):
            result = run_nsga(cfg)
        assert all(p.is_feasible for p in result.front)

    def test_front_sorted_by_realism(self, tmp_path: Path) -> None:
        """Returned front is sorted ascending by realism_distance."""
        cfg = self._tiny_config(tmp_path)
        with patch(
            "crowd_evac.optimization.nsga.evaluate_fitness",
            side_effect=_mock_fitness,
        ):
            result = run_nsga(cfg)
        realisms = [p.realism_distance for p in result.front]
        assert realisms == sorted(realisms)

    def test_progress_callback_invoked(self, tmp_path: Path) -> None:
        """The progress callback fires once per generation."""
        cfg = self._tiny_config(tmp_path)
        calls: list[int] = []
        with patch(
            "crowd_evac.optimization.nsga.evaluate_fitness",
            side_effect=_mock_fitness,
        ):
            run_nsga(cfg, progress_cb=lambda gen, secs: calls.append(gen))
        assert len(calls) == cfg.n_gen
        assert calls == sorted(calls)

    def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        """A completed run resumes from its checkpoint and returns a front."""
        cfg = self._tiny_config(tmp_path)
        with patch(
            "crowd_evac.optimization.nsga.evaluate_fitness",
            side_effect=_mock_fitness,
        ):
            run_nsga(cfg)
            resumed = run_nsga(cfg, resume=True)
        assert isinstance(resumed, NSGAResult)
        assert len(resumed.front) >= 1
