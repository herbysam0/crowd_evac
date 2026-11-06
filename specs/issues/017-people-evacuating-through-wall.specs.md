# Technical Specification for Issue #17 — MINOR — Recommended: Opus 5.5, effort high

MINOR because: new door defaults, changed (user-visible) evacuation semantics,
`capacity_per_second` widened int → float, schema `2.0` → `2.1` (old files still
load via migration; no break of existing readers' inputs).

## Issue Summary
- Title: bugfix - people evacuating through wall
- Description: In `room_model_3` (40 agents), agents in the lower-left room crowd
  into its top-left corner and evacuate from there, through the hall wall.
- Labels: none
- Priority: High (physically invalid egress corrupts every metric / MC result)

## Problem Statement
Scenario fix already applied (door now spans y 7.4–9.0, `wall_5` y 7.3–7.4 —
touching, not overlapping). Bug persists. Root cause is in code:

1. **Grid quantisation punches the wall** — `_rect_cell_range`
   (`floor_plan.py:216`) rounds outward (`floor`/`ceil`). Door bbox starts at
   row `floor(7.4/0.25)=29` (y 7.25–7.5), the same row that holds the 0.1 m wall
   and the lower room's top strip. `walkable_mask` (`floor_plan.py:157`)
   re-opens cells (29, 3–4) and `exit_cells` makes them Dijkstra goals. Cell
   (29, 4) is 4-adjacent to room cell (28, 4) → flow field routes lower-room
   agents to the corner. Any door edge within one cell (0.25 m) of a wall thinner
   than a cell leaks.
2. **Capture ignores walls** — `_enqueue_arrivals` (`exit_model.py:180`) queues
   agents by Euclidean distance `< EXIT_CAPTURE_RADIUS` (1.0 m). Corner agent
   (~1.28, 7.0) is ~0.5 m from segment end (1.0, 7.4) → removed through `wall_5`.

## Technical Approach

### A. Exit opening never re-opens foreign walls (`floor_plan.py`)
- Identify each exit's **host wall**: the wall rect containing the exit centre
  (perimeter wall). Exit override carves the opening through the host wall's
  **full thickness** along the door span (agents must physically walk through).
- After the override, **re-block every non-host wall / obstacle**, so interior
  walls always win over door bboxes.
- Door span quantised **inward** (`ceil` start, `floor` end) for override, so
  the carved gap never exceeds the authored width. Guard: span < 1 cell →
  `ScenarioValidationError` (door unusable on grid).
- `exit_cells` → cells on the **outer edge of the apron** (see D), filtered to
  final-mask walkable. Flow field then drives agents through door and apron.

### B'. Door gate (REVISED, supersedes B — user request after phase 2)
- Agents whose body reaches the door line are **held** there (centre one
  radius inside, outward velocity zeroed) and join the door FIFO.
- Token bucket (`capacity_per_second`) **releases** queue heads; a released
  agent is placed one radius past the line (lateral clamped into the span)
  and keeps walking.
- Released agents are **evacuated** (removed, counted) at the outer end of
  the egress path (`reach − radius − margin`). Doors with no room behind for
  a whole body (boundary door, no apron) evacuate on release.
- Gate also runs inside each no-overlap projection pass (constraint) so the
  projection cannot push held agents through the line.
- `last_egress_per_exit` = removals; new `last_release_per_exit`,
  `queue_lengths`. Stall counter resets on release or removal.

### B. Evacuation = body fully cleared door (`exit_model.py`) — SUPERSEDED by B'
Replaces distance capture (subsumes old fix B; LOS not needed — crossing
the door plane is only possible by walking through the carved gap, walls
collide).
- Per exit: outward normal `n` (WEST −x, EAST +x, SOUTH −y, NORTH +y), door
  plane = authored exit line.
- Agent **cleared** when `dot(pos − door_point, n) ≥ AGENT_RADIUS` **and**
  lateral coordinate within the door span ± `AGENT_RADIUS` (apron bounds it
  anyway). Whole body past the plane.
- Cleared agents join exit FIFO; token bucket (`capacity_per_second`, now
  float) meters removal as today. Uncleared-but-queued agents keep walking in
  the apron (apron space absorbs backlog; physical door width becomes the
  emergent bottleneck, capacity is an upper cap).
- `EXIT_CAPTURE_RADIUS` stays only for `optimization/stuck.py` (queue-member
  exclusion); rename doc to reflect that.

### C. Door defaults
- `DEFAULT_DOOR_WIDTH_M = 0.85` (`designer/controller.py:27`, was 1.5).
  **Note:** request said "0.85 cm"; assumed **0.85 m** (standard door leaf;
  0.85 cm < agent diameter 0.56 m would block all egress).
- `DEFAULT_DOOR_CAPACITY = 1.05` persons/s (`designer/model.py:31`, was 20).
- `capacity_per_second: int → float` in `Exit` (`floor_plan.py:89`),
  `schema.py:41`, `designer/layout.py:84`, `designer/model.py:170,188`,
  loader `scenario_loader.py:258` (`float(...)`), export, designer panel input
  parsing. Validate `> 0`.

### D. One-time migration: 1 m egress apron behind every door
- New script `scripts/migrate_door_apron.py` (CLI entry, `print` OK), run
  once over `assets/scenarios/*.json`; writes in place, `schema_version → "2.1"`;
  idempotent (skips files already `2.1`).
- For each side that hosts ≥1 exit, grow floor plan by `APRON_DEPTH_M = 1.0`:
  - EAST: `width_m += 1`; NORTH: `height_m += 1`.
  - WEST / SOUTH: also translate **all** geometry (+1 x or +1 y): walls,
    obstacles, exits, agent spawn regions, hazard `events` positions.
- New strip filled with wall rects **except** an apron corridor: door span ×
  1 m directly behind each door, closed by side walls → agents cannot wander
  along the outside.
- Loader: accept `2.0` and `2.1`; for `2.0` log `WARNING` "pre-apron scenario;
  run migrate_door_apron.py" (still loads; doors get no apron → agents clear
  inside host-wall gap only if wall ≥ radius thick). Designer export writes
  `2.1` and auto-adds aprons on export.
- Agent files (`separate agents from scenario`, issue #6) with positions:
  migration also shifts them when a WEST/SOUTH shift applies — inventory agent
  files referencing each scenario before running.

### E. Grid resolution 0.25 m -> 0.05 m (DECIDED: 0.05 m)
Decision: `GRID_CELL_SIZE = 0.05`. Analysis of rejected 0.01 m kept below.
Originally requested: `GRID_CELL_SIZE = 0.01` (`constants.py:167`), so walls (0.1 m) span
10 cells instead of under 1.

Impact (cells x625):
| Scenario | Cells @0.25 | Cells @0.01 | Flow field RAM* |
|---|---|---|---|
| room_model_3 16.2x10 | 2.6 k | 1.6 M | ~60 MB |
| open_room_search 20x15 | 4.8 k | 3.0 M | ~110 MB |
| lecture_hall 50x30 | 24 k | 15 M | ~550 MB |
*cost f64 + 2-D direction f64 + masks; doubled per `with_blocks` copy.
- `_dijkstra` (`flow_field.py:425`) is pure-Python `heapq`: ~8 relaxations/cell
  -> lecture_hall ~120 M heap ops = minutes per solve. Re-solved on every hazard
  event and every Monte Carlo run -> MC / NSGA batches infeasible.
- `cells_in_radius` (`flow_field.py:303`, used `simulation.py:407`) loops
  Python over `(2r/cs)^2` cells: 2 m hazard -> 160 k iterations per source.
- Spawn placement (`agent_state.py`), `montecarlo/runner.py:266`,
  `montecarlo/hazards.py:201`, `designer/export.py:127` validation solve,
  `collision.py`, `overlap.py`, `stuck.py`: correct (scale-free), but slower
  startup and bigger masks.
- Does NOT remove the leak by itself: outward rounding still leaks up to 1 cell;
  fix A stays required. Navigation smoothness gain is marginal (agents 0.56 m).
- Golden values in flow-field / spawn / realism tests change.

Chosen: `GRID_CELL_SIZE = 0.05` (5 cm): every 0.1 m wall >= 2
cells, cells x25 (lecture_hall 600 k, Dijkstra ~5-15 s, ~25 MB), no new
dependency. 1 cm only viable with a compiled Dijkstra (`scipy.sparse.csgraph`
or numba) -> dependency proposal per governance rule.

## Implementation Plan
One fresh Claude Code session per phase; one commit per phase. Set model/effort
with `/model` and `/effort` at session start. Do not switch model mid-phase.

| # | Phase | Model | Effort | Commit |
|---|---|---|---|---|
| 1 | C — door defaults, float capacity | Sonnet 5.5 | medium | `feat(domain): float door capacity and new defaults` |
| 2 | A + B + E — wall re-block, clearing egress, 0.05 m grid | Opus 5.5 | high | `fix(domain): keep interior walls closed at doors`, `feat(domain): evacuate agents only after clearing door`, `perf(domain): refine grid to 0.05 m` |
| 3 | D — apron migration + finalise | Sonnet 5.5 | medium | `feat(scenarios): add 1 m egress apron migration`, `chore(assets): migrate scenarios to 2.1` |

### Phase 1 — C (Sonnet 5.5, medium)
1. `git checkout -b issue-17-people-evacuating-through-wall`
2. RED: tests for float capacity (loader, schema, designer), defaults 0.85 m / 1.05/s.
3. GREEN: widen `capacity_per_second` int → float end-to-end; set defaults.
4. Targeted tests + `flake8` / `mypy --strict` on touched files. Draft commit.

### Phase 2 — A + B + E (Opus 5.5, high)
1. RED: `room_model_3` regression fixture; tests for A, B, E (Test Plan).
2. A: `_host_wall`, inward span quantisation, re-block non-host walls,
   apron-edge exit cells (falls back to door cells when no apron exists).
3. B: `_cleared_mask(pos, exits)` vectorised in new
   `src/crowd_evac/domain/egress_geometry.py`; wire into
   `ExitModel._enqueue_arrivals` (exit_model.py ≤ 300 lines).
4. E: `GRID_CELL_SIZE = 0.05`; `agent_state.py:350` default → `GRID_CELL_SIZE`;
   perf guard test (lecture_hall build < 15 s); update golden values.
5. Targeted tests + lint/types on touched files. Draft commits.

### Phase 3 — D + finalise (Sonnet 5.5, medium)
1. RED: tests for `add_aprons`, loader 2.0/2.1 handling, export round-trip.
2. GREEN: `src/crowd_evac/scenarios/apron.py` (pure `add_aprons(raw) -> raw`,
   reused by designer export), `scripts/migrate_door_apron.py`, loader version
   handling, export `SCHEMA_VERSION = "2.1"`, shift agent files for WEST/SOUTH.
3. Targeted tests + lint/types on touched files.
4. **[YOU DO]** run migration, review `git diff assets/scenarios`, visual run of
   `room_model_3` 40 agents: lower-left room exits via interior gap (x 4.7–5.6)
   → hall → door → apron.
5. **[YOU DO]** full suite (realism/MC baselines shift; expect golden updates).
6. Bump version **MINOR** (`pyproject.toml`, `src/crowd_evac/__init__.py`).
7. Draft commits; **[YOU DO]** push, open PR with body `Fixes #17`.

## Test Plan
1. Unit Tests:
   - `walkable_mask` (room_model_3): rows 29 cells under `wall_5` blocked; host
     wall carved full thickness along span; carved width ≤ authored width.
   - door span < 1 cell → `ScenarioValidationError`.
   - `exit_cells`: all at apron outer edge; none in any room interior.
   - `_cleared_mask`: centre on plane → False; outward `radius − ε` → False;
     `≥ radius` → True; outside lateral span → False; all four sides.
   - `ExitModel`: corner agent (1.28, 7.0) never queued; agent cleared in apron
     queued; float capacity 1.05 → 21 egress in 20 s (token-bucket math).
   - `add_aprons`: EAST grows width only; WEST shifts all geometry +1 x;
     doors on two sides; idempotency on `2.1`; side walls enclose apron.
   - Loader: `capacity_per_second` 1.05 parsed float; int still accepted;
     `2.0` logs WARNING (`caplog`); unsupported version still raises.
   - Designer: new door uses 0.85 m / 1.05 /s.
2. Component Tests:
   - FlowField on migrated room_model_3: descent from (3.0, 5.0) passes interior
     gap, not corner (1.0, 7.3).
   - Designer export → load round-trip yields `2.1` with aprons.
3. Integration Tests:
   - Headless room_model_3 (migrated), 40 agents, fixed seed: all 40 evacuated;
     every removed agent's final position lies inside an apron; no agent ever
     has a position inside a wall rect.
   - Migration script on `tmp_path` copy of all scenarios → each loads, runs
     100 ticks without error.

## Files to Modify
- `src/crowd_evac/domain/floor_plan.py`: host wall, inward quantisation, re-block, apron exit cells, float capacity.
- `src/crowd_evac/domain/exit_model.py`: clearing-based arrivals.
- `src/crowd_evac/domain/constants.py`: `APRON_DEPTH_M`, `EXIT_CAPTURE_RADIUS` doc.
- `src/crowd_evac/scenarios/schema.py`: capacity float, version 2.1.
- `src/crowd_evac/adapters/io/scenario_loader.py`: float capacity, 2.0/2.1 handling.
- `src/crowd_evac/designer/controller.py`, `model.py`, `layout.py`, `export.py`: defaults, float, export 2.1 + aprons.
- `src/crowd_evac/adapters/render/designer_panel.py`: float capacity input.
- `assets/scenarios/*.json` (7 files): migrated by script.
- `pyproject.toml`, `src/crowd_evac/__init__.py`: MINOR bump.

## Files to Create
- `src/crowd_evac/domain/egress_geometry.py`: outward normals, `_cleared_mask`.
- `src/crowd_evac/scenarios/apron.py`: pure apron transform.
- `scripts/migrate_door_apron.py`: one-time migration CLI.
- Tests mirroring each new module + regression tests in existing floor_plan / exit_model / loader / designer test files.

## Existing Utilities to Leverage
- `_rect_cell_range`, `_exit_bbox` (`floor_plan.py`): cell mapping (add inward variant).
- `_segment_distances` (`exit_model.py`): lateral clamp math reused for span test.
- `designer/export.py`, `designer/importer.py`: JSON shape for transform + round-trip.
- `FlowField.from_floor_plan`, `optimization/harness.py`: component / integration runs.

## Success Criteria
- [ ] No agent egresses through or stands inside any wall (room_model_3 integration test).
- [ ] Interior walls never opened by door bboxes, any door/wall alignment.
- [ ] Agents removed only after whole body passes door plane.
- [ ] New doors default 0.85 m, 1.05 persons/s; float capacity end-to-end.
- [ ] All 7 scenarios migrated to 2.1 with 1 m aprons; load and run.
- [ ] `flake8` + `mypy --strict` clean on touched files; all files ≤ 300 lines.
- [ ] `GRID_CELL_SIZE = 0.05`; lecture_hall flow-field build < 15 s.
- [ ] Version bumped MINOR.

## Out of Scope
- Retuning optimiser / realism targets after semantics change (follow-up issue).
- Designer UI to edit apron depth.
- Doors on interior walls (still unsupported as exits).

## Open Points (confirm before step 2)
- Door width 0.85 **m** (not cm) — assumed.
- Capacity kept as upper cap on removal from apron; alternative (gate agents at
  door threshold) rejected as more complex.

## Close the Issue
After PR merge, user runs:
```powershell
gh issue close 17 --comment "Fixed: door openings no longer reopen interior walls, agents evacuate only after fully clearing the door, new door defaults (0.85 m, 1.05/s), scenarios migrated to 2.1 with 1 m egress aprons."
```
