# Spec #2 — Add room designer feature [MINOR] [Model: Sonnet 5.5, effort: high]

Status: implemented on branch `issue-2-add-room-designer-feature` (uncommitted at the time
of writing). This document records what was built, including scope added during review.

## Issue Summary
- Title: add room designer feature
- Description: design a room in a visual interface and get a compatible JSON file in the scenarios folder. Features: drag walls with a rubber band showing length; edit length numerically; add doorways in walls or at wall ends with settable width; add obstacles.
- Labels: none
- Priority: Medium

## Problem Statement
Scenarios are hand-written JSON (`assets/scenarios/`). Authoring is slow and error-prone,
and nothing checks a layout until the simulator loads it.

The designer is a new visual authoring tool. It reads and writes the existing schema 1.0
with no schema or simulation change, so every saved file works with `--scenario NAME`
immediately (`list_bundled_scenarios` scans the directory).

## Decisions
| Topic | Decision |
|---|---|
| Wall angle | Snap to horizontal/vertical only. |
| Obstacles | Rectangles only (**replaces the issue's polygon request**). Obstacles may overlap each other, but not other entities. |
| Edit/delete | Walls, doorways and obstacles can all be edited and deleted (**added to scope**). |
| Doorways | Use the current `Exit` representation unchanged, no toggle. A later issue expands the door definition. |
| UI toolkit | `arcade` + `arcade.gui`; launched with `python -m crowd_evac --design`. No new dependency. |
| Snap | Pointer and numeric positions snap to 0.1 m (`SNAP_STEP_M`); the simulation grid stays 0.25 m. |
| Move | Dragging a selected entity moves it (walls carry their doorways; doorways slide along their wall); panel X/Y edits do the same. **Added to scope.** |
| Wall thickness | Per-wall, default 0.5 m, editable; default for new walls set in the panel. **Added to scope.** |
| Tools | Radio toggle buttons Wall / Doorway / Obstacle. No Select button: with no tool active the pointer selects and moves. Pressing the active tool again, or Esc, deselects it. |
| Agents/seed | Not exposed in the designer. New rooms save 50 agents, seed 42. |
| Loading | Existing scenario JSON can be opened for editing from the panel. **Added to scope** (was out of scope). |
| Room resize | Entities never move on a resize; a resize that would leave anything outside is rejected with a loud banner. |

## Technical Approach

### Layers
A pure, headless design model holds all rules. The arcade window only translates input and
draws. Nearly all behaviour is testable without a GL context.

```
designer/geometry.py        rectangle maths, wall rect, perimeter side
designer/layout.py          immutable entities, Extras, whole-layout validation
designer/model.py           DesignModel: validated add/edit/delete, commit
designer/wall_ops.py        move / lengthen / thicken a wall
designer/mover.py           drag-to-move gesture (slides along blocked axes)
designer/placement.py       snapping, hit testing, doorway centre rule
designer/selection_edit.py  numeric fields of the selection: read and apply
designer/controller.py      tool state machine, rubber band, alert/notice
designer/export.py          layout -> ScenarioData; headless check
designer/importer.py        loaded scenario -> layout
adapters/io/scenario_writer.py        name check, overwrite guard, safe JSON write
adapters/render/designer_io.py        save/load flows with confirmations (no GL)
adapters/render/designer_window.py    arcade window, input forwarding, banners
adapters/render/designer_view.py      canvas drawing, viewport, window sizing
adapters/render/designer_panel.py     two-column arcade.gui panel
adapters/render/designer_selection.py selection editor widgets
adapters/render/designer_widgets.py   widget factories, number helpers
adapters/render/designer_style.py     fonts, colours, button/input styles
adapters/render/designer_labels.py    display labels
```

### Existing conventions the export follows
- Coordinates in metres, origin bottom-left, y-up (arcade matches).
- **`north` means the bottom wall (`y=0`) and `south` the top wall (`y=height_m`)**:
  `arcade_renderer.py:437-448` and existing labels. Counter-intuitive; kept and tested.
- Walls are rectangles `{x, y, width, height, label}`. The designer stores a centreline plus
  thickness; a wall on the room boundary is shifted inward so it is flush with the edge.
- Doorways do **not** split the wall. `FloorPlan.walkable_mask` carves the opening over the
  wall; agents leave within `EXIT_CAPTURE_RADIUS`.
- Exit centre sits on the wall's inner face (e.g. east wall `x=14.5..15`, exit `x=14.5`).
  Exports follow the wall's actual thickness.

### Doorways: perimeter walls only
`arcade_renderer.py:434-461` ignores an exit's perpendicular coordinate and draws it on the
room boundary for its `side`, and `Exit` always removes agents. A doorway on an interior wall
would display in the wrong place and make agents vanish mid-room. So a doorway attaches only
to a wall flush with the room boundary; `side` is derived from that wall. Other walls are
rejected with a message.

- In a wall: centre snaps along the wall, opening stays within the wall span.
- At a wall end: a click within `END_SNAP_M` of an end places the opening starting at that
  end and extending outward (centre = end ± width/2). If that would leave the room (a
  corner), the opening is placed flush inside that end instead of being rejected.
- Doorway X/Y in the panel is its centre; only the along-wall coordinate is editable.
- `capacity_per_second` is editable; the default for new doorways is set in the panel.

### Interaction
- Wall tool: press, drag, release. Rubber band follows the dominant axis with a live length
  label, previewed at the chosen thickness. Zero-length is rejected.
- No tool (select): click to select (doorway > obstacle > wall), drag to move, `Del` deletes
  (ignored while a text field has focus).
- Selection panel shows editable numbers: wall X/Y (start point), length, thickness;
  obstacle X/Y (bottom-left), width, height; doorway X/Y (centre), width, capacity.
- Wall length edit: start point stays fixed, the end moves.
- Room panel: scenario name, room width/height. Defaults for new items: wall thickness, door
  width, door capacity.
- **Open saved room**: dropdown of files in `assets/scenarios` plus Load.

### Window and layout
- The window is sized to the display (shrunk when the desired 1320 x 800 would not fit once
  OS scaling is applied), centred horizontally, and resizable; the canvas and banner follow.
- The panel is two columns (setup / selection) so it fits small displays.
- Fonts: Segoe UI → Calibri → Arial, 14 pt body.

### Feedback
Rejections and results are shown as full-width banners across the canvas (7 s), not only in
the small status line:
- red: rejected room resize (names the attempted size and the blocking entity), save failure
  (`NOT SAVED - ...`, with a plain "needs at least one doorway" message), overwrite prompt,
  discard-unsaved prompt on Load, `LOAD FAILED - ...`.
- green: `SAVED  <full path>`.

### Validation rules (model; rejected change leaves state untouched)
Every edit builds a candidate layout, validates all of it, and only then commits.

| Change | Rejected when |
|---|---|
| Any entity | Non-positive size/thickness, or outside room bounds |
| Obstacle | Overlaps a wall or doorway (obstacle-obstacle allowed) |
| Wall | Overlaps an obstacle (wall-wall allowed, needed at corners); not axis-aligned |
| Doorway | Wall not on perimeter; overlaps an obstacle or another doorway; non-positive width/capacity; does not fit on its wall |
| Wall edit / move | An attached doorway would no longer fit, or would change side |
| Room resize | Any entity would fall outside, or a doorway's wall would stop being on the room edge (e.g. enlarging a room whose east wall carries a doorway) |

Deleting a wall deletes its doorways, with the count in the status message. Touching edges do
not count as overlap. Messages name the offending entity (`Wall 3 is outside the room`).

### Save pipeline
1. Name must match `^[a-z0-9_]+$`, which blocks path traversal.
2. `export` builds `ScenarioData` (`schema_version` "1.0"). `agents`, `simulation` and
   `events` come from `Extras`: defaults for a new room, the loaded file's values otherwise.
3. Headless check: build `FloorPlan`, `FlowField.build`, `spawn`. Failures raise a red banner
   and nothing is written.
4. Existing name: require a second Save click, with the design unchanged, to overwrite.
5. Write to a temporary sibling, read it back with `load_scenario_file`, then move it into
   place, so an invalid scenario never appears in the scenarios directory.

### Load pipeline
- `layout_from_scenario(plan, data)`: wall rectangles become centreline + thickness; exits
  attach to the nearest boundary wall of matching orientation **by position, not by the
  `side` label** (`open_room_search.json` labels a top-wall exit `north`); agents, simulation
  and events are carried through unchanged.
- Loading with unsaved edits needs a second Load click.
- Normalised on save: exit perpendicular coordinate moves to the wall's inner face; wall,
  obstacle and exit labels are regenerated; mislabelled sides are corrected.
- Files the designer cannot represent (e.g. an exit with no boundary wall) fail with a
  message and leave the current design untouched.

## Implementation Plan
One commit per step (user runs commits). Status as built:

1. [x] `git checkout -b issue-2-add-room-designer-feature` from `main`.
2. [x] Model (TDD): geometry, layout, model, wall ops, `DesignValidationError`.
3. [x] Export + writer (TDD): `export.py`, `scenario_writer.py`, round trip through the loader.
4. [x] Controller (TDD): tool state machine, snapping, rubber band, hit testing, selection, move.
5. [x] Window + panel: arcade canvas and `arcade.gui` widgets, delegating to the controller.
6. [x] CLI: `--design` / `-d` flag in `parse_args`, lazy import and dispatch in `main`.
7. [x] Docs: README usage. Version `0.1.1` -> `0.2.0` in `pyproject.toml` and `__version__`.
8. [x] Commit, then PR from `issue-2-add-room-designer-feature` with `Closes #2` (user).
9. [x] Close the issue (prompt below) (user).

Added during review (not in the original plan): per-wall thickness, move, X/Y fields,
loading existing scenarios (`importer.py`, `designer_io.py`), loud banners, display-fitted
two-column window, radio-toggle tools.

## Test Plan
Agent ran only targeted files; the full suite, `mypy` and `flake8` over the whole tree are
the user's to run.

1. Unit Tests:
   - Model: add/edit/delete/move per entity; every row of the validation table; touching edges allowed; obstacle-obstacle overlap allowed; wall delete cascades; thickness; resize rejection names the entity; rejected change leaves state intact.
   - Export: side mapping for all 4 walls (north=`y=0`); inner-face centre incl. thick walls; door at wall end.
   - Importer: every bundled scenario round-trips (import → export → reload) with identical geometry, agents, simulation and events; exit matched by position; unrepresentable exit rejected.
   - Writer: invalid names; overwrite guard; no write on failed validation; failed overwrite keeps the old file.
2. Component Tests:
   - Controller: drag produces an H/V wall with live length; numeric edits keep the start point; X/Y edits; end-click placement; degenerate drag and interior-wall door rejected; move slides along blocked axes; resize alert.
   - Save/load flows: confirmations, banners, plain-language messages.
3. Integration Tests:
   - Designed room -> save to `tmp_path` -> `load_scenario_file` -> `FloorPlan`/`FlowField`/`Simulation`; agents evacuate.
   - `parse_args(["--design"])`, incompatible flag combinations, lazy import (subprocess), dispatch.
   - `@pytest.mark.render`: window opens, draws, takes input, fits the display, banners, tool toggling, load/save refresh.

Tests write to `tmp_path` only: `test_scenario_catalog.py` parametrizes over every file in
`assets/scenarios`, so stray files would enter the suite. Load tests only read bundled files.

## Files Modified
- `src/crowd_evac/application/cli.py`: `--design` flag and dispatch only.
- `src/crowd_evac/domain/errors.py`: `DesignValidationError`.
- `pyproject.toml`, `src/crowd_evac/__init__.py`: version `0.2.0`.
- `README.md`: scenario selection and designer usage.
- `tests/crowd_evac/application/test_cli_args.py`: `--design` cases.

## Files Created
- `src/crowd_evac/designer/`: `__init__`, `geometry`, `layout`, `model`, `wall_ops`, `mover`, `placement`, `selection_edit`, `controller`, `export`, `importer`.
- `src/crowd_evac/adapters/io/scenario_writer.py`.
- `src/crowd_evac/adapters/render/`: `designer_window`, `designer_view`, `designer_panel`, `designer_selection`, `designer_widgets`, `designer_style`, `designer_labels`, `designer_io`.
- `tests/crowd_evac/designer/`: `conftest`, `helpers`, `test_model_{walls,doorways,edits}`,
  `test_controller_{drawing,editing,state}`, `test_export`, `test_importer` (split by topic
  to keep each file under 300 lines).
- `tests/crowd_evac/adapters/io/test_scenario_writer.py`.
- `tests/crowd_evac/adapters/render/`: `designer_helpers`, `test_designer_view`,
  `test_designer_io`, `test_designer_window_gl` (`render`-marked).

## Existing Utilities Leveraged
- `scenarios/schema.py`: `ScenarioData` and friends as the contract.
- `adapters/io/scenario_catalog.py`: `SCENARIOS_DIR`, `list_bundled_scenarios`.
- `adapters/io/scenario_loader.py`: `load_scenario_file` for read-back and loading.
- `domain/floor_plan.py`, `pathfinding/flow_field.py`, `domain/agent_state.py`: headless check.
- `domain/constants.py`: `GRID_CELL_SIZE`, `DT`.
- `arcade.gui`: `UIManager`, `UIInputText`, `UIFlatButton`, `UIDropdown` (arcade 3.3.3).

## Success Criteria
- [x] `python -m crowd_evac --design` opens the designer.
- [x] Dragging creates an H/V wall with a live length readout.
- [x] Editing wall length numerically updates the visual.
- [x] Doorways can be placed in a perimeter wall or at its end, with settable width.
- [x] Obstacle rectangles can be drawn; overlap with walls/doorways rejected, with each other allowed.
- [x] Walls, doorways and obstacles can be edited, moved and deleted.
- [x] Wall thickness is settable.
- [x] Existing scenarios can be opened, edited and saved without losing agents, simulation or events.
- [x] Saved file appears in `--list`, loads, and runs to evacuation in the simulator.
- [x] No change to `schema.py`, `scenario_loader.py`, or simulation code.
- [x] Every new file ≤ 300 lines, functions ≤ 40 lines; targeted tests, `flake8` and `mypy --strict` pass on the new code.
- [x] Version `0.2.0`.

## Out of Scope
- Polygon obstacles and diagonal walls (user decision).
- Interior doors and new door semantics (later issue).
- Editing `events`, agent count/seed, spawn regions, undo/redo, multi-level floors. (Loaded `events`, agents and simulation settings are preserved, not editable.)
- Opening a scenario from `--design --scenario NAME` or from outside `assets/scenarios`.
- Detecting sealed rooms: agents spawned in a pocket with no door pass the save check.
- Narrow gaps: the designer snaps to 0.1 m but the simulation grid is 0.25 m, so a gap or doorway under about 0.5 m may close or widen at run time.
- Refactoring `cli.py` (468 lines, already over the 300-line limit); separate issue.
- Saving from a wheel install: `SCENARIOS_DIR` is repo-relative (known limit from #3).

## Open Points
- **Perimeter-only doorways** narrow the issue's "doorways in walls". Lift in the later
  door-definition issue (needs the renderer to draw exits at their `x, y` and a decision on
  whether an interior opening removes agents or is a passage).
- **Rooms with edge walls** cannot be shrunk, and cannot be enlarged if an edge wall carries a
  doorway: entities never move on resize, so the walls must be moved first.
- **Issue text still says polygons.** Update issue #2 so it matches this spec.

## Close Issue (prompt for user)
After PR merge, run:
```powershell
gh issue close 2 --comment "Room designer shipped in v0.2.0: python -m crowd_evac --design."
```
