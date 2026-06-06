"""Performance test: parallel evaluate_fitness dispatch across cores.

Excluded from the standard exit-bar run — execute explicitly with::

    pytest tests/optimization/test_fitness_perf.py -v -s -m perf

Why separate from test_fitness.py
---------------------------------
On Windows the ``spawn`` start method costs ~0.5 s per worker process.  Only
when each harness run's simulation cost (here ~1-5 s at 1000 ticks) exceeds the
spawn overhead does the flat parallel dispatch genuinely beat the serial path.
At the short tick caps used by the correctness tests the spawn cost dominates,
so the comparison is unreliable there and lives here under the ``perf`` marker
(consistent with ``test_suite_perf.py``).
"""
from __future__ import annotations

import logging
import os
import time

import pytest

from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization.fitness import FitnessConfig, evaluate_fitness

logger = logging.getLogger(__name__)

_TICK_CAP: int = 1000


@pytest.mark.perf
def test_parallel_fitness_faster_than_serial() -> None:
    """Parallel K x (M+1) dispatch beats serial at 1000 ticks/run.

    Uses K=3 seeds x (2 suite scenarios + 1 rig) = 9 runs.  At 1000 ticks the
    per-run cost comfortably exceeds the Windows spawn overhead, so the
    all-cores dispatch is reliably faster than the in-process serial path.

    Skipped automatically on single-core machines.
    """
    n_cores = os.cpu_count() or 1
    if n_cores < 2:
        pytest.skip("single-core machine — no parallel speedup possible")

    params = ForceParams.defaults()
    seeds = (0, 1, 2)

    serial_cfg = FitnessConfig(seeds=seeds, max_ticks=_TICK_CAP, max_workers=1)
    t0 = time.monotonic()
    evaluate_fitness(params, serial_cfg)
    serial_s = time.monotonic() - t0

    parallel_cfg = FitnessConfig(
        seeds=seeds, max_ticks=_TICK_CAP, max_workers=n_cores
    )
    t1 = time.monotonic()
    res = evaluate_fitness(params, parallel_cfg)
    parallel_s = time.monotonic() - t1

    n_runs = len(seeds) * (len(parallel_cfg.scenarios) + 1)
    speedup = serial_s / parallel_s if parallel_s > 0 else float("inf")
    logger.info(
        "parallel fitness: %d runs / %d cores — serial=%.2f s parallel=%.2f s "
        "speedup=%.2fx (dispatch wall=%.2f s)",
        n_runs, n_cores, serial_s, parallel_s, speedup, res.wall_clock_s,
    )
    print(
        f"\n[perf] {n_runs} runs / {n_cores} cores — "
        f"serial={serial_s:.2f}s  parallel={parallel_s:.2f}s  "
        f"speedup={speedup:.2f}x"
    )

    assert parallel_s < serial_s, (
        f"parallel ({parallel_s:.3f} s) should beat serial ({serial_s:.3f} s) "
        f"with {n_cores} cores at {_TICK_CAP} ticks/run"
    )
