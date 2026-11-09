# Technical Specification for Issue #18 — MINOR — Recommended: Sonnet 5.5, effort medium

## Issue Summary
- Title: underlay photo in designer
- Description: In the designer canvas let the user load a photo as an underlay to assist creating
  the scenario. The photo is not saved and not referenced in the result JSON. User can show / hide /
  set transparency of the photo.
- Labels: none
- Priority: Medium (UX aid; no simulation impact)

## Problem Statement
Users trace real floor plans (photos, scans) by eye while drawing walls, obstacles and doorways in
the room designer. Without an underlay they must measure the plan and re-type every coordinate,
which is slow and error-prone.

The underlay is purely a session-local visual aid: it must never touch the `Layout`, the
`DesignModel`, unsaved-change detection (`_clean` comparison), or the exported scenario JSON.

## Technical Approach
Decisions (user-confirmed 2026-11-07):
- **Placement:** fit inside room rectangle, **keep aspect ratio**, anchored bottom-left at room
  origin (0, 0). Photo follows room resize (recomputed each draw from `room.width_m/height_m`).
- **File pick:** stdlib `tkinter.filedialog.askopenfilename` (png/jpg/jpeg/bmp). No new dependency.

Architecture (ports & adapters, keep domain/designer packages GL-free):
1. **Pure state** — `crowd_evac/designer/underlay.py` (no arcade import):
   - `@dataclass(frozen=True) class Underlay: path: Path; px_w: int; px_h: int; visible: bool = True;
     opacity: float = DEFAULT_OPACITY` (0.5).
   - `fit_rect(px_w, px_h, room_w_m, room_h_m) -> Rect`: largest aspect-preserving rect at origin.
     Raise `UnderlayError(CrowdEvacError)` on non-positive sizes.
   - `with_opacity(u, value) -> Underlay`: clamp to [0.0, 1.0]; `toggled(u) -> Underlay`.
   - Constants: `DEFAULT_OPACITY`, `OPACITY_STEP = 0.1`, `SUPPORTED_SUFFIXES`.
2. **Adapter** — `crowd_evac/adapters/render/designer_underlay.py`:
   - `pick_photo() -> Path | None`: hidden `tk.Tk()` root, `askopenfilename`, `root.destroy()`.
     Injected as a callable into the window so tests replace it.
   - `load_underlay(path) -> tuple[Underlay, arcade.Texture]`: `arcade.load_texture`; map
     `OSError`/PIL `UnidentifiedImageError` → `UnderlayError`.
   - `draw_underlay(vp, underlay, texture, room)`: if visible, `arcade.draw_texture_rect` with
     `vp.lrbt(fit_rect(...))` and `alpha=round(opacity*255)`.
3. **Draw order** in `DesignerWindow.on_draw`: floor → **underlay** → grid → walls/obstacles/doors.
   Requires splitting `draw_room` into `draw_floor` + `draw_items` (grid stays on top of photo).
4. **Panel UI** — new section "Underlay photo" in left column: buttons `Load photo…`,
   `Show/Hide`, `−` / `+` opacity, opacity label (`50 %`), `Clear`. Panel is 289 lines →
   put widgets in new `designer_underlay_panel.py` (`UnderlayControls` returning `rows()`),
   mirroring `designer_selection.py` pattern.
5. **Keyboard:** `U` toggles visibility (ignored while `panel.is_typing()`).
6. Load failure → `ctl.show_alert("PHOTO NOT LOADED - <reason>")`. Success → notice.
7. Underlay state lives on `DesignerWindow` only. Load/save of a scenario does not clear it.

## Implementation Plan
1. `git checkout -b issue-18-underlay-photo-in-designer`
2. RED: `tests/crowd_evac/designer/test_underlay.py` — fit_rect (wide, tall, square, non-positive
   raises), opacity clamp, toggle.
3. GREEN: implement `src/crowd_evac/designer/underlay.py`.
4. RED/GREEN: `tests/crowd_evac/adapters/render/test_designer_underlay.py` — `load_underlay` on a
   `tmp_path` PNG (written with PIL, already transitive via arcade), bad file → `UnderlayError`.
5. Split `draw_room` into `draw_floor` / `draw_items` in `designer_view.py`; keep `draw_room` as
   thin wrapper if other callers exist; update `test_designer_view.py`.
6. Add `designer_underlay_panel.py` (`UnderlayControls`) and wire into
   `DesignerPanel._setup_rows`; window receives `pick_photo` injectable (default real dialog).
7. Wire `DesignerWindow`: hold `_underlay`/`_underlay_tex`, handlers, `U` key, draw order.
   Keep `designer_window.py` ≤ 300 lines (move handlers to adapter module if needed).
8. GL tests in `test_designer_window_gl.py`: load via fake picker, toggle, opacity, save → JSON
   has no photo reference, `_clean` comparison unaffected.
9. Run targeted tests + `flake8`/`mypy --strict` on touched files only.
10. Bump version `1.5.1 → 1.6.0` in `pyproject.toml` (minor: new user-facing feature).
11. Update README designer section (underlay controls, `U` key).
12. Commit (`feat(designer): add photo underlay with show/hide and opacity`), push, open PR with
    `Closes #18`.

## Test Plan
1. Unit Tests:
   - `fit_rect`: 2000×1000 px in 10×10 m room → (0,0,10,5); 1000×2000 → (0,0,5,10); equal aspect
     → full room; zero/negative → `UnderlayError`.
   - `with_opacity` clamps −0.2→0.0, 1.3→1.0; `toggled` flips `visible`, leaves rest.
2. Component Tests:
   - `load_underlay` real PNG → correct `px_w/px_h`; text file renamed `.png` → `UnderlayError`.
   - `UnderlayControls`: buttons disabled/no-op when no photo; label reflects opacity.
3. Integration Tests (GL, existing `designer_window_gl` harness):
   - Fake picker returns tmp PNG → underlay set, notice shown.
   - Picker returns `None` (cancel) → no change, no alert.
   - `U` key toggles; ignored while typing in input.
   - Save after loading photo → written JSON identical to save without photo.
   - Loading photo does not mark design dirty (Load confirmation not triggered).

## Files to Modify
- `src/crowd_evac/adapters/render/designer_window.py`: underlay state, handlers, `U` key, draw order.
- `src/crowd_evac/adapters/render/designer_view.py`: split `draw_room`; add `U` to hints if useful.
- `src/crowd_evac/adapters/render/designer_panel.py`: include `UnderlayControls.rows()`.
- `tests/crowd_evac/adapters/render/test_designer_view.py`, `test_designer_window_gl.py`.
- `pyproject.toml`: version 1.6.0. `README.md`: designer usage.

## Files to Create
- `src/crowd_evac/designer/underlay.py`: pure underlay state + fit geometry.
- `src/crowd_evac/adapters/render/designer_underlay.py`: file dialog, texture load, draw.
- `src/crowd_evac/adapters/render/designer_underlay_panel.py`: panel widgets.
- `tests/crowd_evac/designer/test_underlay.py`, `tests/crowd_evac/adapters/render/test_designer_underlay.py`.

## Existing Utilities to Leverage
- `viewport.Viewport.lrbt` / `fit_viewport`: metre→pixel rect.
- `designer_widgets.make_heading/make_row/make_label`, `designer_panel._action` pattern.
- `DesignController.show_alert/show_notice`: user feedback banner.
- `designer.geometry.Rect`; `domain.errors.CrowdEvacError` base.
- `tests/.../designer_helpers.py`: GL window fixtures.

## Success Criteria
- [ ] User loads png/jpg via dialog; photo drawn under grid and items, aspect kept, fit to room.
- [ ] Show/hide (button + `U`) and opacity 0–100 % in 10 % steps work.
- [ ] Saved scenario JSON contains no photo data/path; dirty-check unaffected.
- [ ] Bad file → alert, design untouched; cancel → no-op.
- [ ] All new/changed files ≤ 300 lines, functions ≤ 40 lines; flake8 + mypy --strict clean.
- [ ] Version 1.6.0; README updated.

## Out of Scope
- Persisting photo between sessions or in scenario JSON.
- Moving / rotating / free-scaling the photo; calibration by two reference points.
- PDF/SVG plans; image editing.

## Close the Issue
After PR merge, run:
```powershell
gh issue close 18 --comment "Implemented in PR #<pr-number>: photo underlay (fit to room, show/hide, opacity) in designer. Released in 1.6.0."
```
