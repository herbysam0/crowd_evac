# Spec #5 — Monte Carlo testing [MINOR] [Model: Sonnet 5.5, effort: high]

## Issue Summary
- Title: monte carlo testing
- Description: add a script that runs Monte Carlo batches from the CLI or a future
  dashboard. Parameters: scenario, number of persons, number of runs, headless or
  visual, seed, hazard injection (fixed or randomized). Reports time to last
  evacuation and a graph of people evacuated vs time. Only the dashboard interface
  is designed now.
- Labels: none
- Priority: Medium

## Problem Statement
The project can only run one simulation at a time, in the game window, or batches
through the weight optimizer. Nothing measures how much an evacuation varies across
spawn positions and hazards for a given room.

The existing stall rule also makes such a measurement wrong. A run ends after 10 s
with no exit, even when agents are still walking. With 1 agent (this feature's
default), measured over 10 seeds on `main`, `lecture_hall` stopped early with 0
evacuated in 1 of 10 runs and `lecture_hall_no_aisle` in 4 of 10. Those runs need
up to 16.2 s, so results would be cut off at 10 s.

## Decisions
| # | Decision |
|---|---|
| 1 | Fix the early stop by raising the global `STALL_TICKS` from 200 to 1200 (60 s). |
| 2 | CLI output is tables only. The graph is dashboard-only; the result carries its data. |
| 3 | Console and dashboard results both include the standard deviation. |
| 4 | Hazard type stays a label: `source_type` only selects the icon. Behaviour comes from `intensity`, `radius` and `decay_rate`. |
| 5 | Visual mode plays all runs in turn in one window, starting and advancing automatically. Closing it stops the batch and reports finished runs. |
| 6 | With no `--seed`, generate a random base seed and include it in the report. |

## Technical Approach

### Stall limit
Set `STALL_TICKS = 1200`. Measured with a raised limit (2000 ticks): 30 of 30
one-agent runs evacuate in both lecture halls, worst 16.2 s, and at 150 agents the
longest gap between exits is 13 ticks. So 1200 is about 3.7 times the worst walk.

Consequences:
- This is global, so the game also waits 60 s, not 10 s, before ending a run with
  trapped agents.
- Headless runs with trapped agents take up to 6 times as many ticks.
- Calibration and hazard validation change only for runs that previously stalled.
  Weights are not recalibrated.
- It's still a time limit. A designer room with a walk over 60 s would stop early;
  the report shows this as a run with fewer than all evacuated.

### CLI
```powershell
python scripts/monte_carlo.py
python scripts/monte_carlo.py -s lecture_hall --agents 150 --runs 50 --seed 7
python scripts/monte_carlo.py --agents 150 --hazard x=10..40,y=2..28,tick=0..400,source_type=fire
python scripts/monte_carlo.py --runs 5 --visual
```

| Flag | Meaning | Default |
|---|---|---|
| `-s`, `--scenario NAME` | bundled room | `lecture_hall` |
| `--agents N` | persons per run, N >= 1 | 1 |
| `--runs N` | number of runs, N >= 1 | 20 |
| `--visual` | show runs in a window | headless |
| `--seed S` | base seed, S >= 0 | random, reported |
| `--hazard K=V,...` | hazard, repeatable | none |

The defaults are Monte Carlo constants. `DEFAULT_AGENT_COUNT` (3) stays for the game.

### Hazard format
This extends the existing `--hazard` format. `x`, `y` and `tick` each take one value
or a `lo..hi` range, sampled per run. All other keys stay fixed values.

- Reuse `parse_hazard`: replace each range with its lower bound, validate, and keep
  the bounds. Make `_split_pairs` public as `split_pairs`.
- `lo > hi`, or a range on any other key: usage error.
- `x` and `y` remain required. For the whole room, pass its bounds (`x=0..50`).
- Time is in ticks, as in the game flag (`DT` = 0.05 s, so `tick=100` is 5 s).
- Any bound outside the room: usage error after the room loads.
- A sampled position in a wall or obstacle is redrawn, up to
  `MAX_HAZARD_SAMPLE_ATTEMPTS` (100), then raises `HazardRangeError` as a usage error.
- A fixed position in a wall is accepted, as for `--hazard` in the game.
- A hazard whose tick falls after the run ended never fires. The report shows this.

### Seeds
- No `--seed`: base seed = `secrets.randbelow(2**32)`. The report always prints it,
  labelled `(random)` when generated.
- Spawn seeds per run: `SeedSequence(base).generate_state(runs, np.uint32)`.
- Hazard sampling per run: `default_rng([base, HAZARD_STREAM, run_index])`, so
  adding a hazard doesn't change spawn positions.
- Same base seed and config give identical results, headless or visual.
- The table lists each run's seed, so one run can be replayed in the game with
  `--seed`. Pass its sampled hazard as a fixed `--hazard`.

### Running
Headless runs use `harness.evaluate` in a `ProcessPoolExecutor`, through a new
module-level worker. `evaluate_batch` doesn't fit: it uses the same events for every
run, and hazards here vary per run.

Worker settings:
- `wall_clock_cap_s=math.inf`: the 120 s default depends on machine load, so it
  would break reproducibility. `max_ticks` (10,000) remains the cap.
- `history_interval` above `max_ticks`: keeps only tick 0, since position history
  isn't needed and would use a lot of memory.

Visual runs use `MonteCarloWindow(EvacWindow)`, imported lazily so headless runs
never load arcade:
- Starts each run without SPACE and enforces `max_ticks`.
- Ignores mouse hazards, which would break reproducibility.
- Exits when the last run ends. Closing early reports finished runs.

### Results
Per run, `RunOutcome` holds `run_index`, `seed`, `hazards`, `hazards_fired`,
`evacuated`, `total`, `last_evac_s` and `exits_per_tick`.
The batch result also carries `elapsed_s`, the wall-clock time, shown on the last
report line. It is the only value that differs between identical batches.

- `last_evac_s` comes from the last non-zero `exits_per_tick`, and is `None` if
  nobody left. `RunResult.evac_time` isn't used because it returns the end time
  when nobody evacuated.
- A run counts as complete only when `evacuated == total`. `is_terminal` isn't
  enough, because a stalled run has `is_terminal=False`.
- Mean, standard deviation, min and max use complete runs only, with incomplete runs
  counted separately. Mixing them would hide trapped agents.
- Standard deviation is the sample form (`ddof=1`), `None` with fewer than 2 complete
  runs, shown as `n/a`.

### Console report
```text
Scenario: lecture_hall  Agents: 150  Runs: 3  Mode: headless
Base seed: 2718281828 (random)

Run  Seed        Evacuated  Last evac (s)  Hazards fired
1    3141592653  150/150    25.10          1/1
2    1618033988  150/150    22.60          1/1
3    2236067977  149/150    24.05          1/1

Time to last evacuation (2 complete runs)
Mean (s)  Std dev (s)  Min (s)  Max (s)
23.85     1.77         22.60    25.10

Incomplete runs: 1 of 3
```
The sample shows the layout only; its values are illustrative. With randomized
hazards, a line per run below the table gives each sampled position and tick.

### Dashboard interface
Only this interface is built now. The dashboard calls it directly, without the CLI:

```python
def run_monte_carlo(
    config: MonteCarloConfig,
    listener: MonteCarloListener | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> MonteCarloResult: ...

class MonteCarloListener(Protocol):
    def on_run_complete(self, outcome: RunOutcome, done: int, total: int) -> None: ...
```

- `MonteCarloConfig`: frozen dataclass holding the flags above, with `seed: int | None`.
- `HazardSpec`: frozen dataclass with `x`, `y`, `tick` as a value or `ValueRange`, plus
  the fixed fields. The dashboard panel fills this in without any text parsing.
- `MonteCarloResult`: `config`, `base_seed`, `seed_was_random`, `outcomes`, `summary`,
  `curves`, `cancelled`, `elapsed_s`.
- `EvacuationCurves`: `times_s`, one cumulative `per_run` series per run, and bold-line
  `mean` data. Series are padded to the longest run by holding their final count, so
  the mean stays correct after short runs end.
- Cancelling takes effect between runs and returns the runs that finished.

`curves` is computed and tested now but not printed.

### Layout
New package `crowd_evac/montecarlo/`, because it depends on both `application` and
`optimization`, and each module stays under 300 lines. `scripts/monte_carlo.py` is a
thin wrapper, like the other scripts. `harness.py` and `cli.py` are already over the
limit and are not extended; splitting them is out of scope.

## Implementation Plan
1. `git checkout -b issue-5-monte-carlo-testing`.
2. Fix (own commit), tests first: `lecture_hall`, 1 agent, seed 4 evacuates. Set
   `STALL_TICKS = 1200`; update its docstring and the "(10 s)" note at
   `optimization/stuck.py:77`.
3. RED, then GREEN: `config.py` and `hazards.py` (ranges, sampling, wall redraw).
   Make `split_pairs` public.
4. RED, then GREEN: `results.py` (outcome, summary, curves).
5. RED, then GREEN: `runner.py` and the listener port (seeds, process pool, cancel).
6. RED, then GREEN: `report.py` and `cli.py`; add `scripts/monte_carlo.py`.
7. RED, then GREEN: `visual.py`. Check visual and headless results match.
8. Update `README.md` with usage, flags, hazard ranges and the new stall limit.
9. Run targeted tests for changed files. Hand the user the full-suite commands.
10. Bump `pyproject.toml` to `1.2.0` (minor: new feature, backward compatible).
11. Agent drafts commit messages and PR text. User commits and opens the PR with
    `Closes #5`.

## Test Plan
1. Unit Tests:
   - Hazard parsing: fixed value, range, `lo == hi`, `lo > hi`, range on another key,
     bad number, out-of-room bound.
   - Sampling: stays in range, same seed repeats, redraws from a wall, fully walled
     range raises `HazardRangeError`, spawn seeds unaffected by adding a hazard.
   - Seeds: no seed gives a recorded random seed; given seed is used; run seeds are
     stable, distinct and valid for the game's `--seed`.
   - Summary: known mean and std dev, 1 complete run gives `None`, 0 complete runs,
     incomplete runs excluded and counted.
   - Curves: cumulative, padded to longest run, mean checked by hand, zero evacuated.
   - Args: all defaults; `--agents 0` and `--runs 0` rejected; unknown scenario rejected.
   - Report: base seed and `(random)` label, std dev `n/a`, incomplete count, no graph.
2. Component Tests:
   - Same config and seed give identical results, for any worker count.
   - Hazard after the run ends counts as not fired.
   - Listener called once per run; cancel returns finished runs with `cancelled=True`.
   - Headless import of `crowd_evac.montecarlo.runner` doesn't load arcade.
   - `render` marker: visual run matches headless for the same seed; closing early
     returns finished runs.
3. Integration Tests:
   - `lecture_hall`, 1 agent, seed 4 evacuates (the stall regression).
   - Script run with a randomized hazard prints the full report and exits 0.
   - Trapped agents: run ends at the stall limit and is reported as incomplete.

Agent runs named test files only. Full-suite runs belong to the user:
```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m mypy src\ --strict
.\.venv\Scripts\python -m flake8 src\
```

## Files to Modify
- `src/crowd_evac/domain/constants.py`: `STALL_TICKS = 1200`, docstring.
- `src/crowd_evac/optimization/stuck.py`: docstring note on the stall time.
- `src/crowd_evac/application/hazard_args.py`: public `split_pairs`.
- `tests/crowd_evac/application/test_hazard_args.py`: if it uses the old name.
- `README.md`: Monte Carlo section.
- `pyproject.toml`: version `1.2.0`.

## Files to Create
- `src/crowd_evac/montecarlo/__init__.py`: package docstring.
- `src/crowd_evac/montecarlo/config.py`: `MonteCarloConfig`, `HazardSpec`, `ValueRange`.
- `src/crowd_evac/montecarlo/hazards.py`: hazard parsing and per-run sampling.
- `src/crowd_evac/montecarlo/results.py`: `RunOutcome`, summary, `EvacuationCurves`.
- `src/crowd_evac/montecarlo/runner.py`: `run_monte_carlo`, headless worker.
- `src/crowd_evac/montecarlo/report.py`: console tables.
- `src/crowd_evac/montecarlo/cli.py`: argument parsing, `main`.
- `src/crowd_evac/montecarlo/visual.py`: `MonteCarloWindow`.
- `src/crowd_evac/ports/monte_carlo_listener.py`: `MonteCarloListener`.
- `scripts/monte_carlo.py`: thin entry point.
- `tests/crowd_evac/montecarlo/` (with `__init__.py`): `test_hazards.py`,
  `test_results.py`, `test_runner.py`, `test_report.py`, `test_cli.py`, `test_visual.py`.

## Existing Utilities to Leverage
- `optimization/harness.py::evaluate`: headless run; `throughput_series` gives exits per tick.
- `application/hazard_args.py::parse_hazard`, `check_hazards_in_room`: hazard validation.
- `application/injection.py::events_by_tick`, `apply_due_events`: hazard firing.
- `application/cli.py::EvacWindow`, `build_simulation_from_scenario`: visual runs.
- `domain/floor_plan.py::FloorPlan.walkable_mask`: wall check when sampling.
- `metrics/records.py::make_record`: exits per tick in visual runs.
- `adapters/io/scenario_catalog.py::list_bundled_scenarios`: scenario choices.

## Success Criteria
- [x] `python scripts/monte_carlo.py` runs 20 headless runs of `lecture_hall` with 1 person.
- [x] With no `--seed`, a random base seed is generated and printed; rerunning with it
      gives an identical report, except for the `Elapsed` line.
- [x] Each run's seed is listed.
- [x] Report shows time to last evacuation (mean, std dev, min, max), per-run results,
      the incomplete count and the elapsed wall-clock time. No graph in the CLI.
- [x] `--hazard` accepts fixed values and `lo..hi` ranges for `x`, `y` and `tick`.
- [x] `--visual` plays all runs automatically, with results matching headless.
- [x] `run_monte_carlo` is usable without the CLI, with progress, cancel, curve data and
      `elapsed_s`.
- [x] `lecture_hall`, 1 agent, seed 4 evacuates.
- [ ] Targeted tests pass (done); user-run full suite, mypy and flake8 pass; coverage at
      least 80% (`crowd_evac.montecarlo` measured 94% on targeted tests).
- [x] Version is `1.2.0`.

## Out of Scope
- Building the dashboard or drawing the graph.
- File export (CSV, JSON).
- Hazard behaviour presets per type.
- Progress-based stall detection.
- Recalibrating weights.
- Ranges on hazard keys other than `x`, `y`, `tick`.
- User room files by path (bundled names only).
- Splitting `harness.py` or `cli.py`.

## Close Issue (prompt for user)
After the PR merges, run:
```powershell
gh issue close 5 --comment "Monte Carlo batches added: scripts/monte_carlo.py and the run_monte_carlo interface for the future dashboard. Released in v1.2.0."
```
