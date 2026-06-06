"""NSGA-II multi-objective driver for Phase-2 weight optimisation (Step 2.8).

Wires the composite fitness (Step 2.6) and search bounds (Step 2.7) into a
`pymoo` NSGA-II run that produces a realism↔time Pareto front.  The front is
selected from in Step 2.9 by the realism-gated rule.

Problem definition
------------------
:class:`EvacuationProblem` exposes the calibration as a pymoo ``Problem``:

* **2 objectives to minimise** — ``(realism_distance, evac_time)`` — the
  structurally-conflicting pair (realism anchors, evac time is the lever);
* **1 inequality constraint** in pymoo's ``g(x) <= 0`` form —
  ``g = stuck_count``.  Because the stuck count is a non-negative integer, the
  feasible region is exactly ``stuck_count == 0``: no agent deadlocked while a
  viable exit route exists.  Constraint-violating individuals are dominated out
  by NSGA-II's constrained tournament / non-dominated sorting.

The class is defined at module scope (not inside a factory) so the algorithm
object — which holds a reference to the problem — pickles cleanly for the
checkpoint/resume path below.  This requires ``pymoo`` at import time; tests
guard with :func:`pytest.importorskip`.

Parallelism
-----------
Each candidate's ``K × (M + 1)`` harness runs are already dispatched across all
logical cores by :func:`~crowd_evac.optimization.fitness.evaluate_fitness`
(a :class:`~concurrent.futures.ProcessPoolExecutor`).  Population members are
evaluated **sequentially** in :meth:`EvacuationProblem._evaluate`: nesting a
second process pool across individuals would oversubscribe cores and, on the
``spawn`` start method, fail (daemonic workers cannot spawn children).  The
per-generation speedup therefore comes from the per-candidate batch, which
saturates the machine whenever ``K × (M + 1) >= cpu_count`` — true for the
default suite.  GPU presence is detected and logged only; the NumPy simulation
core does not offload (see :mod:`crowd_evac.optimization.harness`).

Checkpoint / resume
-------------------
:func:`run_nsga` drives the algorithm one generation at a time, pickling the
whole algorithm object to ``<checkpoint_dir>/checkpoint.pkl`` every
``checkpoint_every`` generations.  A run is resumed by unpickling that file and
continuing (:func:`load_checkpoint`).  The checkpoints are written and read by
this process only — trusted input — so :mod:`pickle` is acceptable here.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem

from crowd_evac.domain.params import N_PARAMS, ForceParams
from crowd_evac.optimization.fitness import FitnessConfig, evaluate_fitness
from crowd_evac.optimization.space import bounds_array

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIR: Path = Path("artifacts/calibration")
"""Directory for checkpoints and the final Pareto front (gitignored)."""

CHECKPOINT_FILENAME: str = "checkpoint.pkl"
"""Pickled algorithm state, rewritten every ``checkpoint_every`` generations."""

FRONT_FILENAME: str = "front.json"
"""Final non-dominated set written at the end of :func:`run_nsga`."""

DEFAULT_POP_SIZE: int = 64
"""Default NSGA-II population size."""

DEFAULT_N_GEN: int = 80
"""Default number of generations."""

# Type alias for the per-generation progress callback.
ProgressCallback = Callable[[int, float], None]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NSGAConfig:
    """Configuration for one NSGA-II run.

    Attributes:
        pop_size: NSGA-II population size. Must be >= 2.
        n_gen: Number of generations to run (total, including any already
            completed when resuming). Must be >= 1.
        seed: RNG seed for the genetic operators (reproducible search).
        fitness_config: Knobs forwarded to every
            :func:`~crowd_evac.optimization.fitness.evaluate_fitness` call
            (seeds, scenarios, scale, worker count).
        checkpoint_dir: Directory for the checkpoint and front files. Created
            if absent.
        checkpoint_every: Pickle the algorithm every this many generations.
            Must be >= 1.
    """

    pop_size: int = DEFAULT_POP_SIZE
    n_gen: int = DEFAULT_N_GEN
    seed: int = 1
    fitness_config: FitnessConfig = dataclasses.field(default_factory=FitnessConfig)
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        """Validate the configuration; raise ValueError on any bad value."""
        if self.pop_size < 2:
            raise ValueError(f"pop_size must be >= 2, got {self.pop_size!r}")
        if self.n_gen < 1:
            raise ValueError(f"n_gen must be >= 1, got {self.n_gen!r}")
        if self.checkpoint_every < 1:
            raise ValueError(
                f"checkpoint_every must be >= 1, got {self.checkpoint_every!r}"
            )


@dataclasses.dataclass(frozen=True)
class ParetoPoint:
    """One non-dominated solution: a candidate plus its scored vector.

    Attributes:
        params: The :class:`~crowd_evac.domain.params.ForceParams` candidate.
        realism_distance: First objective (minimised).
        evac_time: Second objective (minimised).
        stuck_count: Constraint value ``g`` (feasible at ``<= 0``; integral and
            non-negative, so feasible only at ``0``).
    """

    params: ForceParams
    realism_distance: float
    evac_time: float
    stuck_count: float

    @property
    def is_feasible(self) -> bool:
        """True when the stuck-agent constraint is satisfied (``g <= 0``)."""
        return self.stuck_count <= 0.0


@dataclasses.dataclass(frozen=True)
class NSGAResult:
    """The outcome of a completed (or resumed-and-completed) NSGA-II run.

    Attributes:
        front: Non-dominated solutions, ascending by ``realism_distance``.
        n_gen_completed: Generations actually executed by the algorithm.
        pop_size: Population size used.
        wall_clock_s: Total wall-clock time (s) of the run. Excluded from
            equality so two runs of the same search compare equal despite
            timing jitter.
    """

    front: tuple[ParetoPoint, ...]
    n_gen_completed: int
    pop_size: int
    wall_clock_s: float = dataclasses.field(default=0.0, compare=False)


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class EvacuationProblem(Problem):  # type: ignore[misc]  # pymoo base is Any
    """pymoo ``Problem`` scoring ForceParams over the composite fitness.

    Two objectives (``realism_distance``, ``evac_time``) are minimised under one
    inequality constraint ``g = stuck_count <= 0``.  Bounds come from
    :func:`~crowd_evac.optimization.space.bounds_array` in canonical field
    order, so a decision vector maps directly to
    :meth:`~crowd_evac.domain.params.ForceParams.from_array`.
    """

    def __init__(self, fitness_config: FitnessConfig) -> None:
        """Initialise the problem with bounds and the fitness configuration.

        Args:
            fitness_config: Knobs forwarded to each fitness evaluation.
        """
        bounds = bounds_array()
        super().__init__(
            n_var=N_PARAMS,
            n_obj=2,
            n_ieq_constr=1,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
        )
        self.fitness_config = fitness_config

    def _evaluate(
        self,
        x: npt.NDArray[np.float64],
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Score a whole population matrix into pymoo's ``F`` / ``G`` outputs.

        Args:
            x: Decision matrix of shape ``(pop_size, N_PARAMS)``.
            out: Output dict; ``out["F"]`` is set to the ``(pop_size, 2)``
                objective matrix and ``out["G"]`` to the ``(pop_size, 1)``
                constraint matrix.
            *args: Unused pymoo positional extras.
            **kwargs: Unused pymoo keyword extras.
        """
        pop = np.atleast_2d(x)
        n = pop.shape[0]
        f = np.zeros((n, 2), dtype=np.float64)
        g = np.zeros((n, 1), dtype=np.float64)
        for i in range(n):
            params = ForceParams.from_array(np.asarray(pop[i], dtype=np.float64))
            result = evaluate_fitness(params, self.fitness_config)
            f[i, 0], f[i, 1] = result.objectives
            g[i, 0] = result.constraints[0]
        out["F"] = f
        out["G"] = g


# ---------------------------------------------------------------------------
# Device detection (informational)
# ---------------------------------------------------------------------------


def detect_devices() -> dict[str, Any]:
    """Report available compute devices for logging.

    The simulation core is NumPy/CPU-bound; GPU presence is recorded for
    operator visibility only and does not change the evaluation path.

    Returns:
        Mapping with ``cpu_count`` (int) and ``gpu`` (str): either a detected
        backend name or ``"none"``.
    """
    gpu = "none"
    try:
        import cupy  # optional GPU backend, rarely present

        gpu = f"cupy:{cupy.cuda.runtime.getDeviceCount()} device(s)"
    except Exception:
        # Any import or CUDA-runtime failure simply means no usable GPU; the
        # NumPy core runs on CPU regardless, so this is a benign fallback.
        gpu = "none"
    return {"cpu_count": os.cpu_count() or 1, "gpu": gpu}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, algorithm: Any) -> None:
    """Pickle the algorithm state to *path* (atomic via a temp file).

    Args:
        path: Destination ``.pkl`` file. Parent directories are created.
        algorithm: The pymoo algorithm object to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(algorithm, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_checkpoint(path: Path) -> Any:
    """Unpickle a previously saved algorithm state.

    Args:
        path: Checkpoint ``.pkl`` file written by :func:`save_checkpoint`.

    Returns:
        The resumed pymoo algorithm object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_nsga(
    config: NSGAConfig | None = None,
    *,
    resume: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> NSGAResult:
    """Run NSGA-II to completion, checkpointing each generation.

    Drives the algorithm one generation at a time so the full state can be
    pickled for resume.  On ``resume=True`` an existing checkpoint in
    ``config.checkpoint_dir`` is loaded and the search continues from it;
    otherwise a fresh population is initialised.

    Args:
        config: Run configuration; ``None`` uses :class:`NSGAConfig` defaults.
        resume: If True and a checkpoint exists, continue from it.
        progress_cb: Optional callback invoked as ``(n_gen, gen_seconds)``
            after each generation.

    Returns:
        An :class:`NSGAResult` with the final non-dominated front.
    """
    cfg = config if config is not None else NSGAConfig()
    ckpt_path = cfg.checkpoint_dir / CHECKPOINT_FILENAME
    devices = detect_devices()
    logger.info(
        "NSGA-II start: pop=%d gen=%d seed=%d resume=%s devices=%s",
        cfg.pop_size, cfg.n_gen, cfg.seed, resume, devices,
    )

    algorithm = _init_algorithm(cfg, ckpt_path, resume)
    t0 = time.monotonic()
    while algorithm.has_next():
        t_gen = time.monotonic()
        algorithm.next()
        gen_s = time.monotonic() - t_gen
        n_gen = int(algorithm.n_gen)
        if n_gen % cfg.checkpoint_every == 0:
            save_checkpoint(ckpt_path, algorithm)
        logger.info(
            "gen %d/%d: %.2f s (cores=%d, speedup via per-candidate batch)",
            n_gen, cfg.n_gen, gen_s, devices["cpu_count"],
        )
        if progress_cb is not None:
            progress_cb(n_gen, gen_s)

    save_checkpoint(ckpt_path, algorithm)
    wall_s = time.monotonic() - t0
    front = _extract_front(algorithm.result())
    logger.info(
        "NSGA-II done: %d generations, %d front points, %.1f s",
        int(algorithm.n_gen), len(front), wall_s,
    )
    return NSGAResult(
        front=front,
        n_gen_completed=int(algorithm.n_gen),
        pop_size=cfg.pop_size,
        wall_clock_s=wall_s,
    )


def _init_algorithm(cfg: NSGAConfig, ckpt_path: Path, resume: bool) -> Any:
    """Load a checkpointed algorithm or set up a fresh NSGA-II run.

    Args:
        cfg: Run configuration.
        ckpt_path: Path to the checkpoint file.
        resume: Whether to attempt resuming from ``ckpt_path``.

    Returns:
        A pymoo algorithm ready for the generational loop.
    """
    if resume and ckpt_path.exists():
        logger.info("resuming from checkpoint %s", ckpt_path)
        return load_checkpoint(ckpt_path)
    problem = EvacuationProblem(cfg.fitness_config)
    algorithm = NSGA2(pop_size=cfg.pop_size)
    algorithm.setup(
        problem,
        termination=("n_gen", cfg.n_gen),
        seed=cfg.seed,
        verbose=False,
    )
    return algorithm


def _extract_front(result: Any) -> tuple[ParetoPoint, ...]:
    """Build the sorted Pareto-point tuple from a pymoo result.

    Args:
        result: A pymoo ``Result`` with ``X`` (decisions), ``F`` (objectives),
            and ``G`` (constraints) for the non-dominated set.

    Returns:
        Non-dominated points sorted ascending by ``realism_distance``. Empty if
        the result carries no solutions.
    """
    if result.X is None:
        return ()
    x = np.atleast_2d(np.asarray(result.X, dtype=np.float64))
    f = np.atleast_2d(np.asarray(result.F, dtype=np.float64))
    g = np.atleast_2d(np.asarray(result.G, dtype=np.float64))
    points = [
        ParetoPoint(
            params=ForceParams.from_array(x[i]),
            realism_distance=float(f[i, 0]),
            evac_time=float(f[i, 1]),
            stuck_count=float(g[i, 0]),
        )
        for i in range(x.shape[0])
    ]
    points.sort(key=lambda p: p.realism_distance)
    return tuple(points)


# ---------------------------------------------------------------------------
# Front serialisation
# ---------------------------------------------------------------------------


def write_front(path: Path, result: NSGAResult) -> None:
    """Write the Pareto front and run metadata to a JSON file.

    Args:
        path: Destination ``.json`` file. Parent directories are created.
        result: The completed run to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pop_size": result.pop_size,
        "n_gen_completed": result.n_gen_completed,
        "wall_clock_s": result.wall_clock_s,
        "n_points": len(result.front),
        "front": [
            {
                "params": dataclasses.asdict(p.params),
                "realism_distance": p.realism_distance,
                "evac_time": p.evac_time,
                "stuck_count": p.stuck_count,
                "feasible": p.is_feasible,
            }
            for p in result.front
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
