"""Performance test: parallel evaluate_batch speedup across the search suite.

Excluded from the standard exit-bar run — execute explicitly with::

    pytest tests/optimization/test_suite_perf.py -v -s -m perf

Why separate from test_suite.py
--------------------------------
On Windows the ``spawn`` start method costs ~0.5 s per worker process.
At short tick counts (≤ 50 ticks, ~50 ms/task) spawn overhead dominates
and ``parallel < serial`` is unreliable.  At 1000 ticks (~1–5 s/task on
lecture_hall_small) each task's simulation cost exceeds the spawn cost,
so the parallel path genuinely wins.

The test is marked ``perf`` so it is skipped in the default pytest run
(consistent with ``tests/crowd_evac/performance/``).
"""
from __future__ import annotations

import logging
import os
import time

import pytest

from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization.harness import evaluate_batch
from crowd_evac.optimization.suite import search_suite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Enough ticks that per-task simulation cost (~1–5 s) exceeds Windows spawn
# overhead (~0.5 s/worker), making the parallel comparison meaningful.
_TICK_CAP: int = 1000

_CANDIDATES: list[ForceParams] = [
    ForceParams.defaults(),
    ForceParams.defaults(),
]

_SEEDS: list[int] = [42, 43, 44]


# ---------------------------------------------------------------------------
# Perf test
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_parallel_batch_faster_than_serial() -> None:
    """Parallel evaluate_batch beats serial for the first suite scenario at 1000 ticks.

    Uses 2 candidates × 3 seeds = 6 tasks on lecture_hall_small (30 agents).
    At 1000 ticks the per-task wall-clock cost comfortably exceeds the Windows
    spawn overhead, so ``parallel_s < serial_s`` is a reliable assertion.

    Skipped automatically on single-core machines.

    Run with:
        pytest tests/optimization/test_suite_perf.py -v -s -m perf
    """
    n_cores = os.cpu_count() or 1
    if n_cores < 2:
        pytest.skip("single-core machine — no parallel speedup possible")

    scenario_ref = search_suite()[0].scenario_ref
    n_tasks = len(_CANDIDATES) * len(_SEEDS)

    t0 = time.monotonic()
    evaluate_batch(
        _CANDIDATES,
        scenario_ref,
        _SEEDS,
        max_ticks=_TICK_CAP,
        max_workers=1,
    )
    serial_s = time.monotonic() - t0

    t1 = time.monotonic()
    evaluate_batch(
        _CANDIDATES,
        scenario_ref,
        _SEEDS,
        max_ticks=_TICK_CAP,
        max_workers=n_cores,
    )
    parallel_s = time.monotonic() - t1

    logger.info(
        "parallel batch: %d tasks / %d cores — serial=%.2f s  parallel=%.2f s"
        "  speedup=%.2fx",
        n_tasks,
        n_cores,
        serial_s,
        parallel_s,
        serial_s / parallel_s if parallel_s > 0 else float("inf"),
    )
    print(
        f"\n[perf] {n_tasks} tasks / {n_cores} cores — "
        f"serial={serial_s:.2f}s  parallel={parallel_s:.2f}s  "
        f"speedup={serial_s / parallel_s:.2f}x"
    )

    assert parallel_s < serial_s, (
        f"parallel ({parallel_s:.3f} s) should be faster than "
        f"serial ({serial_s:.3f} s) with {n_cores} cores at "
        f"{_TICK_CAP} ticks/task"
    )
