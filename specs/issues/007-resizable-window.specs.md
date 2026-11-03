# Technical Specification for Issue #7 — Resizable window [MINOR] [Model: Sonnet 5.5, effort: medium]

## Issue Summary
- Title: resizable window
- Description: Let the user resize the UI window with the canvas. Canvas fills the
  window and UI elements (e.g. people icons) are responsive. Works before and during
  a run.
- Labels: none
- Priority: Medium

## Problem Statement
`EvacWindow` (`application/cli.py:267`) is fixed-size: its size and px/m are computed
once from the floor plan and screen. `ArcadeRenderer` and `ArcadeInputSource` take
that scale at construction and never change it, and agent sprites have a fixed pixel
radius. The room therefore cannot follow the window size.

The designer window already resizes through `fit_viewport`, but the simulation window
does not use it. Resize must change only presentation: it must not affect simulation
state, determinism, or the world position of mouse input.

## Technical Approach
**Scaling decision (user): fit and keep aspect.** Use uniform px/m, centre the room,
and leave background bars on the spare sides. Agents stay round and room proportions
stay true. This narrows the issue's word "stretched". Scale is not capped at
`PIXELS_PER_METER` after a resize, so the room grows to fill a large window.

- **Shared viewport:** move `Viewport` and `fit_viewport` from `designer_view.py` to a
  new `adapters/render/viewport.py`. Add a `margin_px` parameter: the designer passes
  `CANVAS_MARGIN_PX`, the simulation passes `0`, so the initial window has no bars.
- **Window:** create with `resizable=True` and call `set_minimum_size`. `on_resize`
  calls `super()`, computes the viewport, and pushes it to renderer and input.
- **Renderer:** add `set_viewport(ppm, x_offset_px, y_offset_px)`. Apply the x offset
  to every layer, as y offset is applied now. Update background image position and
  scale.
- **Agents:** create sprites once from a fixed-size texture and set `sprite.scale`
  from the current ppm. A resize never rebuilds the 10k-capacity sprite list, and
  this avoids the rounding of `int(round(radius * ppm))`.
- **Input:** add `set_viewport(viewport, room_w, room_h)`. Convert with
  `Viewport.to_world` and ignore presses and drags outside the room, including bars.
- **Text:** hints stay at a fixed font size and re-centre through `self.width`.

## Implementation Plan
1. Create and switch to branch `issue-7-resizable-window`.
2. TDD: extract `viewport.py`, add `margin_px`, and update designer imports with no
   designer behaviour change.
3. TDD: add `ArcadeInputSource.set_viewport`, x offset, and out-of-room rejection.
4. TDD: add `ArcadeRenderer.set_viewport`, x offset across all layers, sprite scaling,
   background update, and a minimum symbol font size of 1 so tiny windows are safe.
5. TDD: make `EvacWindow` resizable, add a minimum-size constant, and implement
   `on_resize`. Tolerate `on_resize` firing during `super().__init__`, before renderer
   and input exist.
6. Make `_reset()` apply the current viewport to its new `ArcadeInputSource`. It
   currently rebuilds with the startup ppm, which would misplace clicks after a resize
   followed by reset.
7. Bump version `0.2.0` -> `0.3.0` in `pyproject.toml` and anywhere else it is defined.
8. Update README if it documents window behaviour. Run the targeted checks in the Test
   Plan, then hand the user the full-suite commands.
9. Draft the commit for the user to run, then create a PR from
   `issue-7-resizable-window` with `Closes #7` in the body.
10. Give the user this prompt to close the issue after merge: "Confirm PR merged and
    close issue #7." `Closes #7` closes it automatically on merge to `main`; otherwise
    run `gh issue close 7`.

## Test Plan
1. Unit Tests:
   - `fit_viewport`: wider and taller windows centre the room; `margin_px=0` on a
     matching aspect has no bars; 0×0 and 1×1 stay finite and positive.
   - Input: click maps to correct metres with both offsets and after `set_viewport`;
     clicks and drags in bars are ignored; room-edge boundary values are accepted.
   - Renderer (headless): `set_viewport` updates scale and offsets; invalid ppm
     (zero or negative) raises `ValueError`.
2. Component Tests (GL, following `test_designer_window_gl.py`):
   - `on_resize` updates renderer and input to the same viewport.
   - Agent sprites keep list identity across resize and have the expected centre and
     scale.
   - Reset after resize keeps click mapping correct.
3. Integration Tests:
   - Resizing in all states (initial, running, paused, complete) keeps the state
     unchanged.
   - Same seed with and without resizes during a run gives identical agent positions
     at the same tick.
4. Manual acceptance **[YOU DO]**: drag and maximize before and during a run; place a
   fire after a resize; minimize and restore; check `--design` still resizes correctly.

## Files to Modify
- `src/crowd_evac/application/cli.py`: resizable window, `on_resize`, reset fix.
- `src/crowd_evac/adapters/render/arcade_renderer.py`: `set_viewport`, x offset, sprite
  scaling.
- `src/crowd_evac/adapters/render/arcade_input.py`: `set_viewport`, x offset, bounds.
- `src/crowd_evac/adapters/render/designer_view.py`: import the moved viewport code.
- `pyproject.toml`: version `0.3.0`.
- `tests/crowd_evac/application/test_cli.py`,
  `tests/crowd_evac/adapters/render/test_arcade_renderer.py`,
  `tests/crowd_evac/adapters/render/test_arcade_input.py`: new cases.

## Files to Create
- `src/crowd_evac/adapters/render/viewport.py`: shared `Viewport` and `fit_viewport`.
- `tests/crowd_evac/adapters/render/test_viewport.py`: viewport tests.

## Existing Utilities to Leverage
- `adapters/render/designer_view.py:55,77`: `Viewport`, `fit_viewport`.
- `adapters/render/designer_window.py:130`: resize handling pattern.
- `application/cli.py:108`: `compute_fit_ppm`, kept for initial window size.
- `tests/crowd_evac/adapters/render/test_designer_window_gl.py`: GL test pattern.

## Risks
- **Existing file-size violations:** `cli.py` and `arcade_renderer.py` already exceed
  the 300-line limit (at least 500 lines each by last line read for the renderer, over
  400 for the CLI). Keep additions small here; splitting them is separate refactor work.
- **Windows drag behaviour:** the window message loop may block while the resize border
  is held, pausing the simulation until release. Not verified. Check manually; it does
  not affect determinism because the simulation uses fixed `DT`.

## Success Criteria
- [ ] Window can be resized before and during a run, with state and run results
      unchanged.
- [ ] Room fills the window on the limiting axis, is centred, and keeps its aspect.
- [ ] Agents, walls, exits, and fire markers scale and align together.
- [ ] Clicks and drags map to correct world positions after resize and reset; bars
      ignore input.
- [ ] Designer behaviour is unchanged.
- [ ] Targeted tests pass; flake8 and `mypy --strict` pass on touched files.
- [ ] Version is `0.3.0`; PR is linked and issue #7 closed.

## Out of Scope
- Non-uniform stretching.
- Zoom, pan, and fullscreen toggle.
- Scaling hint and overlay text fonts.
- Splitting `cli.py` and `arcade_renderer.py`.
- `RenderSpikeWindow` in `adapters/render/spike.py`.
