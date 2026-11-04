"""Integration tests: form -> child process -> results, and seed parity.

Covers:
  - a dashboard batch with a randomized hazard under a real ``spawn`` child
  - same config and seed from the dashboard path and run_monte_carlo agree
  - a single run with seed S spawns where the game's ``--seed S`` does
"""
from __future__ import annotations

import time

import numpy as np

from crowd_evac.application.cli import build_simulation_from_scenario
from crowd_evac.dashboard.batch_process import (
    BatchEvent,
    BatchFinished,
    BatchProcess,
    RunFinished,
)
from crowd_evac.dashboard.form import (
    FormFields,
    HazardFields,
    build_config,
)
from crowd_evac.domain.params import ForceParams
from crowd_evac.montecarlo.runner import run_monte_carlo
from crowd_evac.optimization.harness import evaluate

_ROOM = "fixture_minimal"
_SEED = 7
_TIMEOUT_S = 120.0
_HAZARD = HazardFields(
    source_type="fire", x_min="1", x_max="9", y_min="1", y_max="7",
    t_min="0", t_max="2",
)


def _run_to_end(batch: BatchProcess) -> list[BatchEvent]:
    """Start the child and collect events until the last one."""
    batch.start()
    events: list[BatchEvent] = []
    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline:
        events.extend(batch.poll())
        if events and isinstance(events[-1], BatchFinished):
            return events
        time.sleep(0.05)
    raise AssertionError("batch never finished")


def test_dashboard_batch_equals_direct_run() -> None:
    """Verify the child's outcomes equal an in-process run of the config."""
    fields = FormFields(
        scenario=_ROOM, persons="2", seed=str(_SEED), iterations="3",
        hazards=(_HAZARD,),
    )
    config = build_config(fields, monte_carlo=True, max_workers=1)
    events = _run_to_end(BatchProcess(config))
    final = events[-1]
    assert isinstance(final, BatchFinished)
    assert sum(isinstance(e, RunFinished) for e in events) == 3
    direct = run_monte_carlo(config)
    assert final.result.outcomes == direct.outcomes
    assert any(o.hazards for o in final.result.outcomes)


def test_single_run_spawns_like_game_seed() -> None:
    """Verify a single run with seed S starts from the game's --seed S."""
    fields = FormFields(scenario=_ROOM, persons="3", seed=str(_SEED))
    config = build_config(fields, monte_carlo=False, max_workers=1)
    result = run_monte_carlo(config)
    assert result.outcomes[0].seed == _SEED
    game, _ = build_simulation_from_scenario(_ROOM, 3, _SEED, [])
    run = evaluate(
        ForceParams.defaults(), _ROOM, result.outcomes[0].seed,
        agent_count=3, history_interval=10**9,
    )
    assert np.array_equal(
        run.positions_history[0], game.snapshot().positions
    )


def test_single_run_differs_from_derived_batch_seed() -> None:
    """Verify the batch path still derives seeds, so only single is exact."""
    batch = build_config(
        FormFields(scenario=_ROOM, seed=str(_SEED), iterations="1"),
        monte_carlo=True, max_workers=1,
    )
    single = build_config(
        FormFields(scenario=_ROOM, seed=str(_SEED)),
        monte_carlo=False, max_workers=1,
    )
    assert run_monte_carlo(batch).outcomes[0].seed != _SEED
    assert run_monte_carlo(single).outcomes[0].seed == _SEED
