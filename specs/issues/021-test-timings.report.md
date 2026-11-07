# Issue 21 — Test timing report (pass A: everything except `-m render`)

Source: `%TEMP%\durlog_A.csv` (per-test setup+call+teardown, via scratchpad plugin).
Run: 1674 tests, 711.4 s summed test time, 2 failed (perf), 0 render tests run.

## By category (top-level dir under tests/)

| Category | n | total s | median s | p90 s | max s |
|---|---|---|---|---|---|
| tests root (optimization, e2e, scripts, ports) | 415 | 496.2 | 0.000 | 0.302 | 106.62 |
| application | 206 | 129.3 | 0.004 | 2.482 | 10.48 |
| performance | 5 | 25.8 | 4.841 | 5.844 | 10.28 |
| montecarlo | 122 | 24.0 | 0.001 | 0.497 | 3.76 |
| dashboard | 77 | 12.9 | 0.000 | 0.403 | 3.96 |
| adapters | 228 | 8.9 | 0.001 | 0.008 | 1.39 |
| designer | 165 | 8.2 | 0.000 | 0.001 | 2.22 |
| domain | 366 | 2.6 | 0.000 | 0.011 | 0.26 |
| pathfinding | 37 | 2.4 | 0.001 | 0.002 | 2.19 |
| metrics | 29 | 1.0 | 0.000 | 0.090 | 0.09 |
| scenarios | 24 | 0.1 | 0.000 | 0.000 | 0.13 |

## Slowest (top 12, full list in CSV)

| s | test |
|---|---|
| 106.6 | optimization/test_space.py::TestRunSensitivityPrepassE2E::test_fitness_results_have_finite_objectives |
| 106.6 | ...::test_sensitivity_has_n_params_entries |
| 105.7 | ...::test_returns_sensitivity_and_fitness_list |
| 31.3 | optimization/test_fitness_perf.py::test_parallel_fitness_faster_than_serial |
| 31.3 | optimization/test_suite_perf.py::test_parallel_batch_faster_than_serial |
| 10.5 | application/test_cli_run_settings.py::TestReset::test_reset_replays_the_hazard |
| 10.3 | performance/test_perf_headless.py::test_crowd_force_disabled_faster[1k] (FAILED) |
| 7.9 | scripts/test_migrate_door_apron.py::test_all_bundled_scenarios_migrate_load_and_run |
| 7.8 | optimization/test_harness.py::TestRerouteFlowField::test_hazard_blocks_cells |
| 7.8 / 7.7 | application/test_cli_run_settings.py::TestReset (2 more) |
| 7.7 | optimization/test_suite.py::TestParallelBatchAcrossSuite::test_full_suite_parallel_batch_dispatches_all_cores |

## Outlier findings (for Phase 3 discussion)

1. `TestRunSensitivityPrepassE2E`: 3 tests x ~106 s = 318 s = 45% of all
   test time. Each test appears to rerun the same prepass (setup = 0 s, so no
   shared fixture).
2. Two `*_perf.py` tests, 31 s each, in optimization/, marked `@pytest.mark.perf`
   (verified). The `TestRunSensitivityPrepassE2E` class is marked `e2e` (verified).
3. `application/test_cli.py` (and test_cli_run_settings.py): dozens of tests
   at a flat ~2.47 s each; flat cost = repeated per-test build (likely default
   scenario build or arcade import), not test logic.
4. `TestReset` in test_cli_run_settings.py: 8 s call + 2.5 s setup each.
5. Failed: `test_crowd_force_disabled_faster[500a]` and `[1k]` — crowd OFF
   measured slower than crowd ON (6.62 vs 5.26 ms; 16.99 vs 13.65 ms). Timing
   comparison across runs; order/warm-up bias suspected, not a regression
   signal yet.

## Not covered yet

- `render`-marked tests (real windows) not run. A prior run likely hung in
  `montecarlo/test_visual.py::TestRealWindow::test_closing_early_returns_finished_runs`
  (unconfirmed). Pass B needed.
- `test_dashboard_gl.py` is `pytestmark = render` (verified), so it was deselected in
  pass A (0 rows in CSV). Its 2 run-1 failures (`test_scenario_change_updates_hint`,
  `test_add_and_remove`) belong to pass B.

## Pass B (render, `-m render`): 31 tests, 47.9 s, all passed

Slowest: dashboard visual batch 7.9 s, cancel batch 7.1 s, headless batch 5.8 s,
montecarlo `test_visual_matches_headless` 4.8 s. All dominated by real child
processes/windows. Keep, as `render`.

## Phase 3 — outlier discussion and recommendations

Finding 0: `pyproject.toml` `addopts` has no `-m` filter, so `perf`, `e2e` and
`render` tests all run in a bare `pytest`, although the marker text and the
`*_perf.py` docstrings say "excluded by default". Docs and config disagree.

| Group | n | s | Why slow | Needed? | Recommendation |
|---|---|---|---|---|---|
| `test_space.py::TestRunSensitivityPrepassE2E` | 3 | 318 | Each test reruns the same 4-sample prepass (~106 s, real harness) | Wiring check: 1 run is enough, 3 asserts on one result | SHRINK: class-scoped fixture runs once (-212 s). Keep `e2e`; make e2e opt-in |
| `optimization/test_fitness_perf.py`, `test_suite_perf.py` | 2 | 62.6 | Serial vs parallel at 1000 ticks | Only for tuning; assert is hardware dependent | MOVE: already `perf`; exclude from default run |
| `crowd_evac/performance/*` | 5 | 25.8 | Timed 200 ticks at 500/1000 agents | Budget guard | KEEP as `perf` (opt-in) |
| `application/test_cli.py`, `test_cli_run_settings.py` | 57 | 114 | Flat ~2.5 s: `build_simulation_from_scenario("lecture_hall")` per test (0.05 m grid after #20) | Most only need *a* sim | SHRINK: use `fixture_minimal`, or module-scoped build + copy (~-100 s) |
| `TestReset`, `TestScriptedHazardFiring` | ~8 | 55 | Hazard placement reroutes the 0.05 m flow field | Behaviour needed, room size not | SHRINK: `fixture_minimal` (needs hazard pos change) |
| `optimization/test_harness.py`, `test_suite.py` | 84 | 84 | Real harness runs on lecture_hall variants | Integration of harness | KEEP most; cut ticks where assert is structural. Optional `slow` group |
| `tests/e2e/*` | 20 | 16.6 | Full evacuations | Core acceptance | KEEP (own marker `e2e`) |
| `scripts/test_migrate_door_apron.py::..all_bundled..` | 1 | 7.9 | Migrates+runs every bundled scenario | One-off migration guard | MOVE to `slow`/opt-in or delete after migration is final |
| `render` tests | 31 | 47.9 | Real windows/processes | Needed | KEEP, opt-in (display needed) |

Failing perf tests `test_crowd_force_disabled_faster[500a|1k]`: single wall-clock
sample each, fixed order (ON first, then OFF), and the docstring itself says the
crowd term dominates at 2k+ agents, yet it asserts at 500/1000. OFF came out
slower (6.6 vs 5.3 ms; 17.0 vs 13.7 ms): noise/thermal order effect, not a code
regression. Fix: interleave and take min of several runs, and/or assert only
at 2k. Keep as `perf`.

Estimated default-run saving if all above adopted: perf (88 s) + e2e prepass
(212 s) + cli builds (~100 s) = ~400 s of 711 s.

## Outcome (implemented, v1.5.1)

- `pyproject.toml`: default `-m 'not perf and not e2e and not render'`.
- Render hang fixed: `run_dashboard` now closes its window after Ctrl+C
  (pyglet's `on_close` is a no-op once the event loop has exited, so the window
  leaked and the next `arcade.run()` never returned). Stale dashboard test
  expectations updated for the 1 m door aprons.
- `TestRunSensitivityPrepassE2E`: one shared prepass (class-scoped fixture).
- `test_cli.py` / `test_cli_run_settings.py`: small fixture room (64 s -> 5 s,
  50 s -> 14 s).
- Deleted `tests/scripts/test_migrate_door_apron.py` (one-off migration guard).
- Deleted `test_crowd_force_disabled_faster`: the premise (crowd force dominates
  cost) no longer holds after the O(N) herd work; crowd-off measured slower in
  every interleaved run. The flag's behaviour stays covered by
  `domain/test_compose.py` and `optimization/test_params.py`.

## After (final run, v1.5.1 working tree)

Method: same `durlog` plugin; pass 1 = bare `pytest` (new default filter),
pass 2 = `-m "perf or e2e or render"`. Wall times from a Stopwatch.
Result: 0 failures, 0 unfinished tests in both passes (1651 + 45 tests).

| | Before | After |
|---|---|---|
| Default run: tests | 1674 (incl. perf, e2e, 2 failures) | 1651, all pass |
| Default run: summed test time | 711.4 s | 200.4 s (-72%) |
| Default run: wall clock | not measured | 214.3 s |
| Opt-in groups (perf, e2e, render): summed test time | in default run (+47.9 s render, run separately) | 294.8 s (45 tests) |
| Opt-in groups: wall clock | n/a | 305.1 s |
| Whole suite in one process | hung (render) | not run; two passes, no hang |

By category (summed seconds, default run):

| Category | Before | After |
|---|---|---|
| tests root (optimization, e2e, scripts) | 496.2 | 105.0 |
| application | 129.3 | 34.4 |
| performance | 25.8 | excluded (opt-in) |
| montecarlo | 24.0 | 26.8 |
| dashboard | 12.9 | 14.1 |
| all others | ~23 | ~20 |

montecarlo and dashboard moved +2 s and +1 s; unchanged code, machine noise.

Opt-in pass, slowest: shared sensitivity pre-pass 120.3 s (once, was 3 x 106 s),
the two parallel-vs-serial perf tests 35.2 s and 34.7 s, render batch tests
8-11 s. Render group (31 tests) all pass, including the three
`TestRealWindow` tests that used to hang.

Remaining outliers in the default run (all kept, real harness runs):
`optimization/test_suite.py::TestParallelBatchAcrossSuite` (7.9-8.8 s),
`test_harness.py::TestRerouteFlowField` (5.2-8.0 s), `TestDownScaleFaster`
(6.5 s), `dashboard/test_integration.py` (4.6 s). Next candidates if more time
is wanted: cut ticks where the assertion is structural.

Caveat: before/after compare summed per-test time; the "before" wall clock was
not captured because the first run was interrupted.
