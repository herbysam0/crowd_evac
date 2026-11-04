# Spec #9 — Performance dashboard [MINOR] [Model: Sonnet 5.5, effort: high]

## Issue Summary
- Title: performance dashboard
- Description: home screen for running one scenario or a Monte Carlo batch, with
  scenario, persons, headless or visual mode, seed, iterations and hazard injection
  (type, plus fixed or randomized time and location). Reports time to last evacuation,
  people evacuated vs time (all runs plus a bold average) and average person velocity.
- Labels: none
- Priority: Medium

## Problem Statement
Batches can only be run from the command line, whose report has tables and no graph.
`run_monte_carlo` already exposes the interface, curve data, progress and cancel that
spec #5 prepared for a dashboard, but nothing uses it.

Three pieces are missing. Average velocity isn't measured, because batches turn off
history. Hazard types are hardcoded in the renderer. Visual batches must own the main
thread of their process, so they can't run beside a dashboard window in one process.

## Decisions
| # | Decision |
|---|---|
| 1 | Every batch, headless or visual, runs in a child process. A visual batch opens its own window there; the dashboard stays responsive. |
| 2 | `python -m crowd_evac` with no flags opens the dashboard. `-s`, `--agents`, `--seed` or `--hazard` starts the game. `--design` and `--list` are unchanged. |
| 3 | Hazard types come from files: `assets/hazards/<name>.json`. Adding a file adds a type, with no code change and no dashboard editor. |
| 4 | Hazards are entered as typed min/max fields for x, y and time, in a list with Add and Remove. Max blank or equal to min means fixed. |
| 5 | Average velocity is moving-only: mean speed over alive agents and ticks, above `MOVING_SPEED_EPS_MPS` (0.05 m/s). Queueing is ignored. |
| 6 | The UI uses `arcade.gui` and reuses the designer's widgets and style. No new dependency. The chart is drawn with arcade line shapes. |
| 7 | Single run: seed is the exact spawn seed, matching game `--seed`. Monte Carlo: seed is the base seed. Blank draws a random seed, shown in the results. |
| 8 | Results update live as runs finish. A new batch replaces them. Nothing is saved. |

## Technical Approach

### Entry point
`--scenario` defaults to `None` so `parse_args` can tell whether it was given, then
resolves to `DEFAULT_SCENARIO`. New `args.dashboard` is true when no scenario, run
option, `--list` or `--design` was given. To get the old default, run
`python -m crowd_evac -s lecture_hall`.

`main` imports the dashboard lazily, as it does the designer. `__main__.py` gets an
`if __name__ == "__main__":` guard so no spawned process can re-run `main`.

### Running a batch
`dashboard/batch_process.py` owns one child process per batch, so there is a single
path for both modes and no threads in the dashboard:

- `multiprocessing.get_context("spawn").Process`, non-daemon, because a daemon process
  can't start the headless worker pool.
- The child calls `run_monte_carlo(config, listener, cancel_event.is_set)`. In visual
  mode that opens `MonteCarloWindow` on the child's main thread, unchanged except for
  cancel below.
- A queue carries `RunFinished(outcome, done, total)`, then one `BatchFinished(result)`
  or `BatchFailed(message)`. The window drains it with `get_nowait` in `on_update`.
- Child ends with no final message: `BatchFailed` with a generic message; details are
  in the child's log.
- Headless `max_workers` is `max(cpu_count - 1, 1)`, keeping a core for the UI.
- One batch at a time: Run buttons are disabled while one is active.

Cancel:
- Headless: as in spec #5, runs already started finish and are kept. The panel shows
  `Cancelling...`.
- Visual: `MonteCarloWindow.on_update` also checks `should_cancel` each tick, discards
  the unfinished run and closes. Today it checks only between runs, so Cancel would
  wait for the current run to play out.
- Closing the dashboard sets cancel and closes the window at once. The child exits by
  itself, bounded by `DEFAULT_MAX_TICKS`.

### Exact seed
New `MonteCarloConfig.direct_seed: bool = False`. When true, the run uses the base seed
as its spawn seed instead of `derive_seeds`; `runs != 1` raises `ValueError`. The CLI
never sets it, so `--runs 1 --seed 7` keeps its current results. "Run single" sets it.

### Average velocity
New `metrics/speed.py`:
- `MovingSpeedTally.add(snapshot)` adds speeds of alive agents above the threshold.
- `.mean_mps` is `None` when no agent ever moved.
- `MOVING_SPEED_EPS_MPS` moves from `optimization/realism.py` to `domain/constants.py`,
  because `metrics` must not import `optimization`. `realism.py` imports it from there.

Both loops call `add` after each step, so visual and headless results stay equal:
- `harness.evaluate`: new last field `RunResult.moving_speed_mps: float | None = None`.
- `MonteCarloWindow.on_update`: same tally, reset per run.

`RunOutcome` gains `moving_speed_mps`, and `RunSummary` gains `speed_mean_mps` and
`speed_std_mps`. Speed stats cover all runs with a value, including incomplete ones: a
trapped agent's walking speed is still valid, unlike its evacuation time. New fields
default to `None`, so existing constructors keep working.

### Hazard type files
`assets/hazards/fire.json` ships with today's defaults, and a test holds it equal to
the constants:

```json
{
  "label": "Fire",
  "symbol": "🔥",
  "symbol_size_m": 3.0,
  "intensity": 1.0,
  "radius": 10.0,
  "decay_rate": 0.0,
  "block_radius": 1.5,
  "blocks_navigation": true
}
```
Numeric values above show the shape only; use the real `PANIC_RANGE`,
`PANIC_DECAY_RATE` and `HAZARD_BLOCK_RADIUS` values.

`adapters/io/hazard_catalog.py` has `HazardType` and `load_hazard_types()`, modelled on
`scenario_catalog.py`:
- The file stem is `source_type`, for example `smoke.json` is `smoke`.
- `symbol` and `symbol_size_m` are optional. Without a symbol, only the panic circle
  is drawn, as unknown types are today.
- Read with `encoding="utf-8"`. Values are validated through `PanicSource`.
- A bad file is logged with `logger.warning` and skipped, so one typo can't stop the app.
- Rescanned each time the dashboard opens.

`ArcadeRenderer` replaces `_SYMBOL_CHAR` and `_SYMBOL_SIZE_M` with the catalog, loaded
once at init. So `--hazard source_type=smoke` also gets its icon in the game and CLI.

A type sets defaults only. Selecting one prefills the panel, and the panel's values
are what run. Behaviour still comes from the numeric parameters, not from the type.

### Form
`dashboard/form.py` is pure, with no arcade, so it's fully testable. It turns field text
into a `MonteCarloConfig`, or field-specific error messages.

| Field | Default | Rule |
|---|---|---|
| Scenario | `lecture_hall` | `list_bundled_scenarios()`, so designer rooms appear |
| Persons | 1 | integer >= 1 |
| Mode | headless | headless or visual |
| Seed | blank | blank or integer >= 0 |
| Iterations | 20 | integer >= 1, Monte Carlo only |

Per hazard: type; x, y and time min/max; intensity; radius; decay rate; block radius;
blocks navigation. Limit of `MAX_HAZARDS = 6`, because `arcade.gui` has no scroll
widget here.

- Time is in seconds in the UI and becomes ticks with `round(s / DT)`.
- The room size is shown as a hint and updates with the scenario.
- Before starting, the parent runs `validate_spec` and `check_specs_in_room`, so a bad
  hazard shows at once, without starting a process.
- A sampled position in a wall (`HazardRangeError`) or too many agents for the room
  arrive as `BatchFailed` and appear in the status line.

### Results panel
Rebuilt from outcomes after each `RunFinished` with the existing `summarize` and
`build_curves`, sorted by run index. `BatchFinished` then replaces it with the final
result.

- Progress: `Run 7/20` with Cancel.
- Time to last evacuation: value for one run; mean, std dev, min and max for a batch,
  over complete runs only, as in spec #5.
- Average velocity (moving): value, with std dev for a batch.
- Evacuated or incomplete count, hazards fired, and the seed with `(random)` if drawn.
- A single run that doesn't finish shows `148/150 evacuated`, not a time.

### Chart
`dashboard/chart_model.py` is pure: it reduces each series to at most one point per
plot pixel, keeping steps exact, and picks axis ticks. `adapters/render/dashboard_chart.py`
draws thin lines per run and a bold mean.

Curves can hold 10,000 points per run, and drawing all of them each frame would be too
slow. So lines go into a cached `arcade.shape_list.ShapeElementList`, rebuilt only when
data or size changes.

### Layout
Headless logic goes in new package `crowd_evac/dashboard/`, tested without a display.
Arcade code goes in `adapters/render/dashboard_*.py`, following the designer split.
Every new file stays under 300 lines. `cli.py` and `harness.py` are already over the
limit: they get only the small changes above and aren't split here.

## Implementation Plan
1. `git checkout -b issue-9-performance-dashboard`.
2. RED, then GREEN: `metrics/speed.py`; move the constant; wire `harness.py`,
   `RunOutcome`, `RunSummary` and `visual.py`. Check visual speed equals headless.
3. RED, then GREEN: `direct_seed` in config and runner.
4. RED, then GREEN: `hazard_catalog.py`, `assets/hazards/fire.json`; switch the renderer
   to the catalog.
5. RED, then GREEN: `dashboard/form.py`.
6. RED, then GREEN: `dashboard/batch_process.py`, and per-tick cancel in `visual.py`.
7. RED, then GREEN: `dashboard/chart_model.py`, `dashboard/results_model.py`.
8. RED, then GREEN: `dashboard_chart.py`, `dashboard_hazards.py`, `dashboard_results.py`,
   `dashboard_panel.py`, `dashboard_window.py` (`render` marker).
9. RED, then GREEN: `cli_args.py` and `cli.py` entry point; `__main__.py` guard; update
   `test_cli_args.py`.
10. Update `README.md`: dashboard, changed default command, adding a hazard type.
11. Run targeted tests for changed files. Hand the user the full-suite commands.
12. Bump `pyproject.toml` to `1.3.0` (minor: new feature; no sync or DB contract exists).
13. Agent drafts commit messages and PR text. User commits and opens the PR with
    `Closes #9`.
14. Close the issue with the prompt at the end of this spec.

## Test Plan
1. Unit Tests:
   - Speed: known velocities give a known mean, at-threshold and dead agents excluded,
     nobody moving gives `None`.
   - Summary: speed mean and std dev, one run gives `None` std dev, incomplete runs
     included.
   - Seed: `direct_seed` uses the seed exactly; `runs=2` raises; default path unchanged.
   - Catalog: fire matches constants, new file is listed, no symbol allowed, malformed
     JSON and out-of-range values skipped with a warning, missing directory gives empty.
   - Form: all defaults, blank seed, fixed vs range, blank max, seconds to ticks,
     reversed range, non-numeric field, `0` persons, hazard limit.
   - Chart model: point limit, final count preserved, empty input, one point, axis ticks.
   - Results model: one run, batch, incomplete single run, no complete runs, cancelled.
   - Args: no flags gives dashboard; each run flag gives game; `-s lecture_hall` gives
     game; `--design -s x` still rejected.
2. Component Tests:
   - Batch process: events arrive in order and final result matches a direct
     `run_monte_carlo`; cancel returns finished runs; bad room gives `BatchFailed`;
     killed child gives `BatchFailed`.
   - Headless child doesn't import arcade.
   - `render` marker: visual speed matches headless; per-tick cancel closes the window
     and drops the unfinished run; renderer draws an icon from a temporary type file;
     dashboard opens, Add and Remove work, a bad field shows its error, chart caches;
     a `KeyboardInterrupt` in the event loop cancels the batch and closes the window
     without raising.
3. Integration Tests:
   - Dashboard batch with a randomized hazard runs under a real `spawn` child and fills
     results, without re-running `main` in any process.
   - Same config and seed from the dashboard and `run_monte_carlo` give equal outcomes.
   - Single run with seed 7 spawns the same positions as the game's `--seed 7`.
4. Manual acceptance **[YOU DO]**: `python -m crowd_evac`, then a visual Monte Carlo
   batch with a hazard; resize; Cancel; close the sim window early; press Ctrl+C in the
   terminal with a headless batch, a visual batch and no batch running; add a
   `smoke.json` and reopen.

Agent runs named test files only. Full-suite runs belong to the user:
```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m mypy src\ --strict
.\.venv\Scripts\python -m flake8 src\
```

## Files to Modify
- `src/crowd_evac/__main__.py`: main guard.
- Ctrl+C handling lives in `dashboard_window.py::run_dashboard` and the child's
  `batch_process.py::_child_main` (ignores the console's SIGINT; the parent cancels).
- `src/crowd_evac/application/cli.py`: open the dashboard.
- `src/crowd_evac/application/cli_args.py`: `--scenario` default `None`, `args.dashboard`.
- `src/crowd_evac/adapters/render/arcade_renderer.py`: icons from the catalog.
- `src/crowd_evac/domain/constants.py`: gains `MOVING_SPEED_EPS_MPS`.
- `src/crowd_evac/optimization/realism.py`: import that constant.
- `src/crowd_evac/optimization/harness.py`: `moving_speed_mps`.
- `src/crowd_evac/montecarlo/config.py`: `direct_seed`.
- `src/crowd_evac/montecarlo/runner.py`: `direct_seed`, pass speed through.
- `src/crowd_evac/montecarlo/results.py`: speed fields and stats.
- `src/crowd_evac/montecarlo/visual.py`: speed tally, per-tick cancel.
- `tests/crowd_evac/application/test_cli_args.py`,
  `tests/crowd_evac/adapters/render/test_arcade_renderer.py`: new behaviour.
- `README.md`, `pyproject.toml` (`1.3.0`).

## Files to Create
- `assets/hazards/fire.json`: default hazard type.
- `src/crowd_evac/adapters/io/hazard_catalog.py`: discover and validate type files.
- `src/crowd_evac/metrics/speed.py`: `MovingSpeedTally`.
- `src/crowd_evac/dashboard/__init__.py`: package docstring.
- `src/crowd_evac/dashboard/form.py`: field text to `MonteCarloConfig`.
- `src/crowd_evac/dashboard/batch_process.py`: child process, events, cancel.
- `src/crowd_evac/dashboard/results_model.py`: display values from outcomes.
- `src/crowd_evac/dashboard/chart_model.py`: reduce series, axis ticks.
- `src/crowd_evac/adapters/render/dashboard_window.py`: window, queue polling.
- `src/crowd_evac/adapters/render/dashboard_panel.py`: run settings and buttons.
- `src/crowd_evac/adapters/render/dashboard_hazards.py`: hazard list and fields.
- `src/crowd_evac/adapters/render/dashboard_results.py`: stats and progress.
- `src/crowd_evac/adapters/render/dashboard_chart.py`: cached chart drawing.
- Matching tests under `tests/crowd_evac/dashboard/` (with `__init__.py`),
  `tests/crowd_evac/adapters/render/`, `tests/crowd_evac/adapters/io/test_hazard_catalog.py`
  and `tests/crowd_evac/metrics/test_speed.py`.

## Existing Utilities to Leverage
- `montecarlo/runner.py::run_monte_carlo`: progress, cancel, curves, results.
- `montecarlo/results.py::summarize`, `build_curves`: live statistics and graph data.
- `montecarlo/hazards.py::validate_spec`, `check_specs_in_room`: pre-run checks.
- `montecarlo/visual.py::MonteCarloWindow`: visual batches.
- `montecarlo/config.py::HazardSpec`, `ValueRange`: form output.
- `adapters/render/designer_widgets.py`, `designer_style.py`: inputs, rows, headings, buttons.
- `adapters/io/scenario_catalog.py`: scenario list and pattern for the hazard catalog.
- `adapters/io/scenario_loader.py::load_bundled_scenario`: room size hint and bounds.

## Success Criteria
- [ ] `python -m crowd_evac` opens the dashboard; `-s NAME` and other run flags still
      start the game.
- [ ] Single run works with issue defaults: `lecture_hall`, 1 person, headless, no seed.
- [ ] Monte Carlo defaults to 20 iterations, with the same settings.
- [ ] Hazards can be added with type and fixed or randomized location and time.
- [ ] A new `assets/hazards/*.json` appears in the dropdown on next open, with its icon
      in the simulation, and no code change.
- [ ] Results show time to last evacuation, average moving velocity and a graph with one
      line per run and a bold average, updating live.
- [ ] Visual batches play in a separate window while the dashboard stays responsive;
      headless and visual results are equal, including speed.
- [ ] Cancel keeps finished runs in both modes.
- [ ] Ctrl+C in the terminal closes the dashboard like the close button: the batch is
      cancelled, no traceback is printed, no child process is left behind.
- [ ] Single run with seed S spawns as game `--seed S`; CLI Monte Carlo results for a
      given seed are unchanged.
- [ ] No new dependency. New files under 300 lines.
- [ ] Targeted tests pass; user-run full suite, mypy and flake8 pass; coverage at least
      80% on new code.
- [ ] Version is `1.3.0`.

## Out of Scope
- Editing or creating hazard types in the dashboard.
- Behaviour per type beyond parameters (such as smoke occlusion), and icon images.
- Clicking a room map to place hazards.
- Session history, comparing batches and file export.
- Velocity in the CLI report.
- Per-run seed table in the dashboard (the CLI lists them).
- A designer button on the dashboard (`--design` stays).
- User room files by path.
- Splitting `cli.py` or `harness.py`.

## Close Issue (prompt for user)
After the PR merges, run:
```powershell
gh issue close 9 --comment "Performance dashboard added: python -m crowd_evac now opens it, with single and Monte Carlo runs, file-based hazard types and live results. Released in v1.3.0."
```
