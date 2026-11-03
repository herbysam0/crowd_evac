# Spec #3 — Move scenario folder [PATCH] [Model: Sonnet]

## Issue Summary
- Title: move scenario folder
- Description: scenarios live in `src/crowd_evac/assets/scenarios`; should be in `assets/scenarios` (repo root).
- Labels: none
- Priority: Low

## Problem Statement
Bundled scenario JSON sits inside the package. Issue wants it at repo-root
`assets/scenarios/`. A stale, unused, different `assets/scenarios/lecture_hall.json`
(1984 B, from Step 1.18) already exists there. The `src` copy (2852 B) is the one
loaded by default (`cli.py:42` -> `load_bundled_scenario` -> `files("crowd_evac.assets")`).

Decision (user): keep the `src` copy as canonical; delete the root copy.

## Technical Approach
**[YOU DECIDE] loader/packaging — pending.** `importlib.resources` cannot reach
repo-root files. Options:

1. Path-walk loader (Recommended): resolve repo root from `__file__` with
   `pathlib`. Pros: tiny change, KISS. Cons: editable/dev install only; wheel
   loses bundled scenarios (PRD R0.4 weakened for wheels).
2. Build-time copy into package (setuptools hook). Pros: wheels keep R0.4.
   Cons: packaging complexity, two on-disk locations.
3. Keep both copies. Cons: drift (already happened).

Plan below assumes option 1; adjust steps 3-4 if another is chosen.

## Implementation Plan
1. `git checkout -b issue-3-move-scenario-folder`.
2. Tests first: update/add test asserting `list_bundled_scenarios()` and
   `load_bundled_scenario("lecture_hall")` work from a different cwd (RED).
3. `git rm assets/scenarios/lecture_hall.json` (stale root copy).
4. `git mv src/crowd_evac/assets/scenarios/*.json assets/scenarios/`.
5. Add single `SCENARIOS_DIR` resolver (pathlib, from `__file__`); use it in
   `scenario_loader.py:89` and `scenario_catalog.py:20`.
6. Remove `assets/scenarios/*.json` package-data from `pyproject.toml:49`; delete
   `src/crowd_evac/assets/__init__.py` if empty of purpose.
7. Fix docs: `scenario_loader.py:73` docstring, `suite.py:18` docstring, README tree.
8. Run targeted tests (GREEN). Bump version `0.1.0` -> `0.1.1` (patch).
9. Commit; open PR linking `Closes #3`.

## Test Plan
1. Unit: resolver returns existing dir; each bundled name loads; missing name raises `MalformedScenarioError`.
2. Component: catalog lists all 6 scenarios (bottleneck_corridor, fixture_minimal, hazard_lecture_hall, lecture_hall, lecture_hall_small, open_room_search).
3. Integration: CLI default scenario loads from non-repo cwd.

Targeted files only: `tests/crowd_evac/adapters/io/test_scenario_catalog.py`,
`tests/crowd_evac/application/test_cli_args.py`, scenario loader tests.
Full suite: user runs.

## Files to Modify
- `src/crowd_evac/adapters/io/scenario_loader.py`: use resolver, docstring.
- `src/crowd_evac/adapters/io/scenario_catalog.py`: use resolver.
- `src/crowd_evac/optimization/suite.py`: docstring path.
- `pyproject.toml`: drop package-data, bump version.
- `README.md`: structure tree.

## Files to Create
- None beyond resolver constant (place in `scenario_loader.py`; no new file).

## Existing Utilities to Leverage
- `scenario_catalog.py` listing; `_parse_and_build` unchanged.

## Success Criteria
- [ ] `src/crowd_evac/assets/scenarios/` gone; all 6 JSON in `assets/scenarios/`.
- [ ] Root stale `lecture_hall.json` replaced by src version (content = 2852 B file).
- [ ] CLI default scenario loads from any cwd.
- [ ] Targeted tests pass; version 0.1.1.

## Out of Scope
- Wheel packaging if option 1 chosen.
- Scenario content changes.

## Close Issue (prompt for user)
After PR merge, run:
```powershell
gh issue close 3 --comment "Scenarios moved to assets/scenarios in v0.1.1."
```
