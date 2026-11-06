# Technical Specification for Issue #19 — Meter grid in the canvas
**[MINOR] Implement with: Sonnet 5.5, medium effort** (new user-facing visual, backward compatible; no sync/DB contract change; analysis ran on Sonnet 5.5 / medium)

## Issue Summary
- Title: meter grid in the canvas
- Description: Add a 1x1 m grid to the canvas. Lines sit *under* walls, obstacles, doors and people. Lines are thin and faint.
- Labels: none
- Priority: Low (cosmetic aid for reading scale)

## Problem Statement
The simulation window (`ArcadeRenderer`) draws a plain floor fill, so users cannot judge distances. The room designer canvas (`designer_view.draw_room`) already draws a 1 m grid, but with its own inline loops and an opaque colour `(60, 60, 72)`; the two views are inconsistent and the logic is duplicated.

The issue says "canvas": interpreted as both views. Designer already satisfies the layering; the simulation window is the real gap. Sharing one helper gives identical look in both and removes the duplicate.

## Technical Approach
- New pure, GL-free module `adapters/render/grid_overlay.py`:
  - `GRID_STEP_M = 1.0`, `GRID_LINE_COLOR: RGBA255 = (255, 255, 255, 28)` (white, low alpha = faint on any floor/background), `GRID_LINE_WIDTH_PX = 1.0`.
  - `grid_segments(width_m, height_m, ppm, x_off, y_off) -> list[tuple[float, float]]`: flat point list (pairs = one segment) for interior lines at x = 1..floor(W), y = 1..floor(H) (room edges are already the floor border; matches current designer behaviour). Empty for rooms <= 1 m on an axis.
  - `draw_grid(width_m, height_m, ppm, x_off, y_off)`: one `arcade.draw_lines(points, GRID_LINE_COLOR, GRID_LINE_WIDTH_PX)` call (single batch, not N `draw_line` calls per frame).
- `ArcadeRenderer.render`: call `draw_grid` after `_draw_floor()`, before `_draw_geometry()`. Order gives: background, floor, **grid**, walls/obstacles/exits, panic sources, agents -> grid is under everything.
- `designer_view.draw_room`: replace the two inline loops with `draw_grid(...)` using `vp.ppm/ox/oy`; delete local `GRID_LINE_COLOR`.
- No toggle (not requested; KISS). Pure presentation: no domain/snapshot change (R7.3 intact).

## Implementation Plan
1. Create branch `issue-19-meter-grid-in-the-canvas` from `main` and switch to it. **Note:** current branch is `issue-17-people-evacuating-through-wall` with a large uncommitted diff (incl. version bump to 1.4.0). Land/commit #17 first, or branch from main with stash; do not mix.
2. TDD red: write `tests/crowd_evac/adapters/render/test_grid_overlay.py` (below).
3. Green: add `grid_overlay.py`.
4. Wire into `ArcadeRenderer.render` (one import + one call; `arcade_renderer.py` is already >300 lines, add no logic there).
5. Refactor `designer_view.draw_room` to use the helper; keep `test_designer_view.py` green.
6. Update `ArcadeRenderer.render` docstring layer list (add "grid" between floor and walls).
7. Run targeted tests, flake8/mypy on touched files only.
8. Bump version **1.4.0 -> 1.5.0** (minor) in `pyproject.toml` and `src/crowd_evac/__init__.py`.
9. Commit (`feat(render): add 1 m grid under geometry and agents`) and open PR linking `Closes #19`.
10. Closing prompt (below).

## Test Plan
1. Unit Tests (`test_grid_overlay.py`, no GL):
   - 10x6 m room, ppm 40, offsets 0 -> 9 vertical + 5 horizontal segments (2 pts each = 28 pts); endpoints span full room extent.
   - Offsets/ppm applied: first vertical line at `x_off + 1*ppm`.
   - Non-integer room (5.5 x 3.2) -> 4 vertical, 2 horizontal.
   - Edge: room <= 1 m on an axis -> no lines on that axis; 0-size -> empty list.
   - Failure: `ppm <= 0` raises `ValueError`.
   - Constants: alpha < 64 and width <= 1 (faint/thin guard).
2. Component Tests:
   - `ArcadeRenderer(headless=True).render()` still no-ops/doesn't raise.
   - Monkeypatched `draw_grid` called once per `render` (non-headless path stubbed), and ordered after `_draw_floor`, before `_draw_geometry`.
   - `designer_view.draw_room` calls `draw_grid` with viewport values (patched).
3. Integration Tests:
   - `@pytest.mark.render` smoke: open real window, render a frame, no exception (user-run; needs display).
   - Manual visual check: grid visible, faint, hidden beneath walls/doors/agents in sim window and designer.

## Files to Modify
- `src/crowd_evac/adapters/render/arcade_renderer.py`: import + `draw_grid` call in `render`; docstring.
- `src/crowd_evac/adapters/render/designer_view.py`: use shared helper, remove local grid loops/constant.
- `tests/crowd_evac/adapters/render/test_arcade_renderer.py`: call-order test.
- `tests/crowd_evac/adapters/render/test_designer_view.py`: adjust if it references `GRID_LINE_COLOR`.
- `pyproject.toml`, `src/crowd_evac/__init__.py`: version 1.5.0.

## Files to Create
- `src/crowd_evac/adapters/render/grid_overlay.py`: pure segment math + single-batch draw (~50 lines).
- `tests/crowd_evac/adapters/render/test_grid_overlay.py`: unit tests above.

## Existing Utilities to Leverage
- `adapters/render/viewport.py::Viewport` (`ppm`, `ox`, `oy`): designer mapping.
- `arcade_renderer.py::FLOOR_COLOR`, `_draw_filled_rect`: floor layer the grid sits on.
- `arcade.draw_lines`: batched line drawing (arcade >= 3.3.3).

## Success Criteria
- [ ] Simulation window shows 1 m grid, aligned to metre multiples from room origin.
- [ ] Designer shows identical-looking grid (shared helper, one colour constant).
- [ ] Grid hidden under walls, obstacles, doors/exits, panic sources and agents.
- [ ] Lines 1 px, low alpha (faint) on dark floor and over background image.
- [ ] Grid scales correctly after window resize (`set_viewport`).
- [ ] Single draw call per frame for the grid; no FPS regression.
- [ ] New file <= 300 lines; targeted tests, flake8, mypy --strict pass on touched files.
- [ ] Version bumped to 1.5.0.

## Out of Scope
- Toggle key / UI to hide grid, configurable step or colour.
- Axis labels / metre rulers.
- Grid in Monte Carlo dashboard charts.
- Splitting the pre-existing >300-line `arcade_renderer.py` (separate refactor).

## Close the issue (last step)
Prompt for user to run after PR merge (PowerShell):
```powershell
gh issue close 19 --comment "Implemented in v1.5.0: 1 m faint grid in simulation window and designer (PR linked)."
```
