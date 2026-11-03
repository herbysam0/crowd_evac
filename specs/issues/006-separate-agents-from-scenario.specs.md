# Spec #6 — Separate agents from scenario [MAJOR] [Model: Sonnet 5.5, effort: high]

## Issue Summary
- Title: separate agents from scenario
- Description: scenario files hold room configuration and agents. Leave only
  room configuration. Decide whether agents are runtime input or a separate JSON.
- Labels: none
- Priority: Medium

## Problem Statement
A scenario file mixes room geometry with run settings: `agents` (count, seed),
`simulation` (dt, max_ticks) and `events`. Rooms get duplicated to vary a run:
`lecture_hall_small` and `hazard_lecture_hall` have identical floor plans.

Some of this data misleads. `simulation` is never read at runtime, so
`max_ticks: 8000` in `bottleneck_corridor` has no effect. The harness ignores
`spawn_seed`. The CLI ignores `events`, so `-s hazard_lecture_hall` shows no hazard.

## Decisions (grill session)
| # | Decision |
|---|---|
| 1 | Agents are runtime input: `--agents N --seed S`. No population file. |
| 2 | JSON stays comment-free. README lists a run command for each room. |
| 3 | `agents`, `simulation` and `events` all leave the scenario files. |
| 4 | Events live in code for hazard validation, plus a new CLI `--hazard` flag. Delete `hazard_lecture_hall.json`. |
| 5 | Rename `lecture_hall_small` to `lecture_hall_no_aisle`. Geometry unchanged. |
| 6 | `DEFAULT_AGENT_COUNT = 3`, `DEFAULT_SPAWN_SEED = 42`. |
| 7 | Schema `"2.0"` only. `"1.0"` is rejected with a clear error. Add a migration script. |
| 8 | Hazard format: key=value, repeatable. |
| 9 | Harness `agent_count` is required, with no default. |
| 10 | Calibration counts live in code. |
| 11 | Designer drops `Extras` and the spawn check. |
| 12 | New flags with `--design` or `--list` give an argparse error. |
| 13 | `ScenarioRepository.load_scenario` returns `FloorPlan` only. |
| 14 | Loader keeps returning `(FloorPlan, ScenarioData)`. |
| 15 | "Scenario" terminology stays. |
| 16 | Version `0.2.0` -> `1.0.0`. |

## Technical Approach

### Schema
`ScenarioData` becomes `schema_version`, `name`, `floor_plan`. Remove
`AgentConfigData` and `SimConfigData`. `EmergencyEventData` stays in `schema.py`
as the event type shared by harness and CLI.

The loader removes `_normalize_agents`, `_normalize_events`, `_normalize_event`
and `_build_event`. A `"1.0"` file raises `MalformedScenarioError` naming the
migration script. Value checks stay in `PanicSource`.

### CLI
```powershell
python -m crowd_evac -s lecture_hall --agents 150 --seed 42
python -m crowd_evac -s lecture_hall_no_aisle --agents 150 --seed 42 --hazard x=25,y=3.5,tick=100,source_type=fire
```

`--hazard` keys:
- Required: `x`, `y`.
- Optional: `tick` (default 0), `intensity`, `radius`, `decay_rate`,
  `block_radius`, `blocks_navigation`, `source_type`. Unset keys use the
  `add_panic_source` defaults.
- Unknown key, duplicate key, bad value or negative tick: argparse error.
- Position outside the room: `parser.error` after the room loads, since parsing
  doesn't know room size.
- Position inside a wall: same handling as a mouse-placed source.

`--agents` below 1 is an argparse error. `build_simulation_from_scenario` gains
`agent_count`, `seed` and `events`. `EvacWindow` keeps them so `_reset` rebuilds
the same run. Scripted hazards fire by sim tick in `on_update`, before `step()`.

### Shared event code
Move `_events_by_tick` and `_apply_event` from `harness.py` to
`application/injection.py` as public functions, so the CLI doesn't import the
optimizer stack.

### Harness and calibration
- `evaluate`, `evaluate_batch`, `rerouted_flow_field` and `_build_sim` take a
  required `agent_count` and optional `events=()`. `_eval_task` args extend to match.
- `SearchScenario`, `CalibrationRig` and the `fitness.py:222` spec gain `agent_count`.
- Counts: `lecture_hall_no_aisle` 30, `open_room_search` 25, `bottleneck_corridor`
  60. Validation at `select.py:394` uses `lecture_hall` 150 and `open_room_search` 25.
- `select.py` gets `HAZARD_VALIDATION_EVENTS` (tick 100, pos `[25.0, 3.5]`,
  intensity 1.0, source_type `fire`) and `HAZARD_VALIDATION_AGENT_COUNT = 150`.
  `HAZARD_VALIDATION_SCENARIO` becomes `lecture_hall_no_aisle`.
- Fix `suite.py:123` and related docstrings: this room has solid rows and no
  centre aisle, so it isn't the same geometry as `lecture_hall`.

### Designer
Remove `Extras` and `Layout.extras`. Export writes version, name and floor plan
only. Remove `DESIGN_MAX_TICKS` and the `spawn` call in `check_headless`, because
`spawn` can't fail on a valid count. Keep the floor plan and flow-field checks.
`designer_io` keeps working through the loader.

### Migration script
`scripts/migrate_scenarios.py PATH [PATH ...]`:
- Sets version `"2.0"` and removes the 3 blocks.
- Prints the equivalent run command for each file, including any `--hazard` flags,
  so removed values aren't lost.
- Skips files already on `"2.0"`, so reruns are safe.
- Uses `encoding="utf-8"` and `pathlib`.

Bundled rooms are migrated with this script.

### Default-count consequence
Bare `python -m crowd_evac` drops from 150 to 3 agents. Accepted in decision 6;
README commands keep the old counts.

### File size
`cli.py` is 488 lines, over the 300 limit, and this work adds to it. Move
`parse_args` and the hazard parser to `application/cli_args.py` in a separate
refactor commit first. `harness.py` and `select.py` are also already over the
limit; splitting them is out of scope.

## Implementation Plan
1. `git checkout -b issue-6-separate-agents-from-scenario`.
2. Record baseline `RunResult` values on `main` for `lecture_hall_small` (30),
   `open_room_search` (25), `bottleneck_corridor` (60) and `hazard_lecture_hall`
   (150, with its event), so step 7 can compare them.
3. Refactor: extract `application/cli_args.py`, no behaviour change. Own commit.
4. Tests first (RED): loader rejects `"1.0"` and accepts `"2.0"`; migration script;
   hazard parser; event functions in `injection.py`; harness `agent_count`.
5. Add `scripts/migrate_scenarios.py` and migrate the bundled rooms. Delete
   `hazard_lecture_hall.json` (approved). Rename to `lecture_hall_no_aisle.json`.
6. Update schema and loader.
7. Update injection, harness, fitness, suite and select. Check results match the
   step 2 baseline.
8. Update CLI: flags, `build_simulation_from_scenario`, `EvacWindow`, hazard firing.
9. Update designer: layout, importer, export.
10. Update port signature and `tests/test_ports.py`.
11. Update README: per-room command table below, `--hazard` usage, migration note.
    Update `docs/PRD.md` references if they describe agents in scenario files.
12. Run targeted tests for changed files. Hand the user the full-suite commands.
13. Bump `pyproject.toml` to `1.0.0`.
14. Agent drafts commit message and PR text. User commits and opens the PR with
    `Closes #6`.

### README run commands
| Room | Command |
|---|---|
| `lecture_hall` | `python -m crowd_evac -s lecture_hall --agents 150 --seed 42` |
| `lecture_hall_no_aisle` | `python -m crowd_evac -s lecture_hall_no_aisle --agents 30 --seed 42` |
| `lecture_hall_no_aisle` + hazard (old `hazard_lecture_hall`) | `python -m crowd_evac -s lecture_hall_no_aisle --agents 150 --seed 42 --hazard x=25,y=3.5,tick=100,source_type=fire` |
| `bottleneck_corridor` | `python -m crowd_evac -s bottleneck_corridor --agents 60 --seed 7` |
| `open_room_search` | `python -m crowd_evac -s open_room_search --agents 25 --seed 42` |
| `fixture_minimal` | `python -m crowd_evac -s fixture_minimal --agents 50 --seed 42` |
| `model_room_1` | `python -m crowd_evac -s model_room_1 --agents 1 --seed 42` |

## Test Plan
1. Unit Tests:
   - Loader: `"2.0"` loads; `"1.0"` error names the script; extra blocks don't reach `ScenarioData`.
   - Hazard parser: minimal, all keys, unknown key, duplicate key, bad number, missing `x`, negative tick.
   - Args: defaults 3 and 42; `--agents 0` rejected; new flags with `--design` or `--list` rejected.
   - `injection.py`: events grouped by tick; event applies a panic source; empty events.
   - Migration: removes blocks, prints command, includes hazard flags, safe to rerun, bad JSON fails cleanly.
   - Harness: `agent_count` sets population; same inputs give identical results; missing `agent_count` raises `TypeError`.
2. Component Tests:
   - `build_simulation_from_scenario` uses given count, seed and events.
   - Scripted hazard fires at its tick, not before; `_reset` restores count, seed and hazard.
   - Hazard outside the room gives an error.
   - Designer: export has only version, name, floor plan; import/export round trip keeps geometry.
   - Suite and rigs carry the agreed counts.
3. Integration Tests:
   - All 6 bundled rooms load as `"2.0"` and contain no removed blocks.
   - Calibration and hazard validation match the step 2 baseline.
   - Designer save, then CLI run.

Agent runs named test files only. Full-suite runs belong to the user:
```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m mypy src\ --strict
.\.venv\Scripts\python -m flake8 src\
```

## Files to Modify
- `assets/scenarios/*.json`: migrate; rename `lecture_hall_small.json`; delete `hazard_lecture_hall.json`.
- `src/crowd_evac/scenarios/schema.py`: reduced schema, version `"2.0"`.
- `src/crowd_evac/adapters/io/scenario_loader.py`: remove normalizers, reject `"1.0"`.
- `src/crowd_evac/application/cli.py`: run settings, hazard firing, reset.
- `src/crowd_evac/application/injection.py`: public event functions.
- `src/crowd_evac/domain/constants.py`: default count and seed.
- `src/crowd_evac/optimization/harness.py`: required `agent_count`, `events`.
- `src/crowd_evac/optimization/fitness.py`: pass `agent_count`.
- `src/crowd_evac/optimization/suite.py`: counts, rename, docstring fix.
- `src/crowd_evac/optimization/select.py`: hazard constants, counts, rename.
- `src/crowd_evac/designer/layout.py`, `importer.py`, `export.py`: drop `Extras`.
- `src/crowd_evac/ports/scenario_repository.py`: return type.
- `scripts/validate_weights.py`, `scripts/optimize_weights.py`, `scripts/sensitivity.py`: check callers and names.
- `tests/`: 17 files reference affected APIs, including `test_harness.py`, `test_scenario_loader.py`, `test_cli.py`, `test_cli_args.py`, `test_suite.py`, `test_select.py`, `test_export.py`, `test_importer.py`, `test_schema.py`, `test_end_to_end.py`.
- `README.md`, `docs/PRD.md`, `pyproject.toml`.

## Files to Create
- `src/crowd_evac/application/cli_args.py`: argument and hazard parsing.
- `scripts/migrate_scenarios.py`: `"1.0"` to `"2.0"` converter.
- `tests/crowd_evac/application/test_hazard_args.py`: hazard parser tests.
- `tests/scripts/test_migrate_scenarios.py`: migration tests (add `__init__.py` if needed).

## Existing Utilities to Leverage
- `application/injection.py::add_panic_source`: applies hazards and their defaults.
- `harness.py::_events_by_tick`, `_apply_event`: moved, not rewritten.
- `domain/panic_source.py::PanicSource`: value validation.
- `domain/agent_state.py::spawn`: handles crowding with a warning.
- `adapters/io/scenario_catalog.py`: room listing, unchanged.

## Success Criteria
- [ ] All bundled JSONs hold only `schema_version` (`"2.0"`), `name`, `floor_plan`.
- [ ] `hazard_lecture_hall.json` gone; `lecture_hall_small` renamed, geometry unchanged.
- [ ] Bare run uses 3 agents, seed 42; every README command matches its old count and seed.
- [ ] `--hazard` reproduces the old hazard in the window; reset replays it.
- [ ] Calibration and hazard validation results match the baseline.
- [ ] `"1.0"` error names the migration script; script converts old files.
- [ ] Designer saves room-only files.
- [ ] Targeted tests pass; user-run full suite, mypy and flake8 pass; coverage at least 80%.
- [ ] Version is `1.0.0`.

## Out of Scope
- Population files, spawn zones, per-agent traits.
- Density-scaled default count.
- Renaming "scenario" to "room".
- Designer UI for agents or hazards.
- Splitting `harness.py` or `select.py`.
- Recalibrating weights.

## Close Issue (prompt for user)
After the PR merges, run:
```powershell
gh issue close 6 --comment "Scenario files now hold room configuration only (schema 2.0). Agents and hazards are runtime input. Released in v1.0.0."
```
