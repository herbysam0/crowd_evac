# Implementation Plan: Phase 1 — Playable MVP (crowd_evac)

## Context

`crowd_evac` is a greenfield build. The repo currently holds only `docs/PRD.md`
and git/claude config — there is **no baseline code**. The PRD (DRAFT, 2026-06-02)
specifies a single-player, real-time 2D crowd-evacuation game built bottom-up:
a pure NumPy simulation core behind a `Renderer` port, with `arcade` as the
chosen renderer adapter.

**Phase 1 = the MVP playable loop** (PRD §15): the from-scratch 2D core
(**FR-1..FR-8**) + a zero-setup default scenario (**FR-0**, Lecture Hall) + one
panic source the player can drop/drag that makes the crowd flee and re-route
(a subset of **FR-11 / FR-12.1 / FR-14**) + a minimal `arcade` render/interaction
loop. Phase 1 also **pins the stack (D6 = arcade)**, **spikes iGPU rendering at
scale early** to de-risk RK-1, and **sets the §8 emergent-behavior thresholds
empirically** once baseline scales are observed.

Outcome: a clean install launches straight into the running, interactive Lecture
Hall; the player drags a fire and watches the crowd flee, re-route, and evacuate
to completion under a fixed-step seeded loop — believable on its own and the
foundation every later layer (FR-9..FR-15) composes onto without a core rewrite.

## Locked Decisions (this planning session)

- **Agent state = SoA / vectorized NumPy.** Population stored as struct-of-arrays
  (`pos (N,2)`, `vel (N,2)`, `panic (N,)`, `goal (N,)`, `alive (N,)`). Forces are
  computed vectorized over the whole population; a thin `Agent` view exposes
  per-agent fields only where an AC needs it. This is the path to Tier A
  (~10k agents @ 30 FPS) and matches the PRD "pure NumPy" core.
- **Render spike is early and lightweight** (Step 1.3), before the full sim
  exists, while a pivot to `pyglet`/`moderngl` is still cheap (RK-1 gate).
- **Per-step exit bar:** every step ends with `flake8 src/` clean,
  `mypy src/ --strict` clean, and `pytest` green for that step's tests. No step
  advances on a red checkpoint. (Commits are at the user's discretion per the
  git-workflow rule — drafted on request, run by the user.)
- **Navigation = grid-based flow field** (BFS/Dijkstra from exits over a
  rasterized walkable grid → per-cell direction vector). Standard for crowd sims,
  supports R4.4 cache + bounded recompute, and is the substrate FR-9 R9.5 extends.

## Architecture (clean-architecture spine, PRD §9)

```
src/crowd_evac/
├── domain/        # AgentState (SoA), FloorPlan, PanicSource, force terms,
│                  #   gradient field — pure NumPy, no engine/IO imports
├── pathfinding/   # grid flow field over walkable region (FR-4)
├── application/   # fixed-step loop, tick model, seeded RNG, event log,
│                  #   injection API, metrics (FR-6, FR-8, FR-12.1)
├── scenarios/     # Lecture Hall default data + scenario schema (FR-0, FR-3, FR-13.1)
├── metrics/       # structured per-tick metric records (FR-8)
├── ports/         # Renderer / InputSource / ScenarioRepository / Clock (FR-7)
└── adapters/
    ├── render/    # arcade renderer behind Renderer port (FR-7)
    └── io/        # scenario loader, bundled-data resolution via pathlib (R0.4, R3.3)
assets/scenarios/  # bundled default Lecture Hall (FR-0)
tests/             # mirrors src/ — core is fully testable headless
docs/              # PRD.md, plan_phase_1.md
```

**Dependency note (governance):** runtime deps `numpy` and `arcade` (D6, PRD
§9 — resolved). Dev deps `pytest`, `pytest-cov`, `pytest-mock`, `mypy`, `flake8`.
Optional `numba` is **out of Phase 1** (Tier B stretch only). `arcade` to be
vetted (Windows wheel, maintenance, license) in Step 1.1 before install.

## Step Overview

| # | Step | Layer | FR | Model |
|---|------|-------|----|-------|
| 1.1 | Project scaffolding, tooling, deps | — | NFR-Q1/Q2 | Haiku |
| 1.2 | Core primitives: constants, errors, seeded RNG, vec helpers | domain/app | FR-6 R6.2 | Haiku |
| 1.3 | `arcade` instanced-render spike + FPS benchmark (RK-1 gate) | adapters/render | NFR-P2 | Opus |
| 1.4 | FloorPlan model + scenario schema + loader | domain, scenarios, io | FR-3 | Sonnet |
| 1.5 | Grid flow field (multi-exit, cached, re-routable) | pathfinding | FR-4 | Opus |
| 1.6 | AgentState SoA container + seeded spawn | domain | FR-1 R1.1/R1.4 | Sonnet |
| 1.7 | `f_exit` + integrator + speed/accel limits | domain | FR-1 R1.2/R1.3 | Sonnet |
| 1.8 | Crowd dynamics: spatial hash, `f_crowd`, density pressure, herd | domain | FR-2 | Opus |
| 1.9 | Exit model: capacity, queue, egress, evac-complete | domain/app | FR-5 | Sonnet |
| 1.10 | PanicSource + gradient field + `f_panic_repulsion` | domain | FR-11 (subset) | Sonnet |
| 1.11 | Combined force composition (toggleable terms) | domain | FR-14 R14.1 (subset) | Opus |
| 1.12 | Fixed-step sim loop + tick model + event log | application | FR-6 | Sonnet |
| 1.13 | Core metrics records (evac, throughput, density) | metrics/app | FR-8 | Sonnet |
| 1.14 | Runtime injection API + re-route trigger | application | FR-12.1, R4.3 | Sonnet |
| 1.15 | Ports: Renderer / InputSource / ScenarioRepository / Clock | ports | FR-7 | Haiku |
| 1.16 | `arcade` Renderer adapter (read-only snapshot) | adapters/render | FR-7 R7.1/R7.3 | Sonnet |
| 1.17 | Input adapter + interaction loop: drop/drag panic source | adapters/render, app | FR-7 R7.2, FR-15 (subset) | Sonnet |
| 1.18 | Bundled Lecture Hall default scenario data | scenarios, assets | FR-0, FR-13.1 | Haiku |
| 1.19 | App entry point: launch into Lecture Hall, run loop | application/cli | FR-0 R0.1 | Sonnet |
| 1.20 | Headless end-to-end + seed-repro tests | tests | FR-6 R6.1, FR-5 R5.3 | Sonnet |
| 1.21 | Tune defaults, set §8 thresholds, 10k benchmark, document fallback | — | §8, NFR-P2/P3 | Opus |

**Dependencies:** 1.1→1.2→{1.3 (independent spike)}; 1.4→1.5; {1.6}→1.7 (needs 1.5);
1.7→1.8; 1.9 needs 1.6; 1.10 needs 1.4/1.6; 1.11 needs 1.7/1.8/1.10; 1.12 needs
1.11; 1.13/1.14 need 1.12; 1.15→1.16→1.17 need snapshot from 1.12; 1.18 needs
1.4; 1.19 needs 1.16/1.17/1.18; 1.20 needs 1.19; 1.21 needs 1.20.

---

## Step Detail

> Every step ends with the **exit bar**: `flake8 src/` clean,
> `mypy src/ --strict` clean, `pytest` green. Tests are written in the same step
> as the code they cover (3 cases minimum per public symbol: happy / edge /
> failure, per testing-philosophy). Steps marked *(no logic)* add only config or
> interfaces and carry minimal or import-only tests.

### 1.1 Project scaffolding, tooling, dependencies — Haiku
- **Do:** create `pyproject.toml` (project metadata, deps, `[tool.mypy] strict`,
  `[tool.flake8]` max-line 88, `[tool.pytest]`), the `src/crowd_evac/` package
  tree above with `__init__.py` files, `tests/` mirror, `assets/scenarios/`,
  `README.md` skeleton, `.gitignore` (`.venv`, `__pycache__`, `.env`).
- **Deps:** vet `arcade` (Windows wheel, last release, license) per
  dependency-governance; pin `numpy`, `arcade` in `[project.dependencies]`,
  dev tools in `[project.optional-dependencies].dev`.
- **Tests:** `pytest --collect-only` runs (0 tests OK); package imports.
- **Success:** `pip install -e ".[dev]"` succeeds in `.venv`; tree matches layout.

### 1.2 Core primitives — Haiku
- **Do:** `domain/constants.py` (UPPER_SNAKE defaults: `DT`, max speed/accel,
  repulsion radius, panic range), `domain/errors.py` (`CrowdEvacError` base),
  `application/rng.py` (single seeded `numpy.random.Generator` wrapper — FR-6
  R6.2, NFR-R2), small vector helpers if not pure-numpy.
- **Tests:** RNG with same seed reproduces the same draws; distinct seeds differ;
  base error is raisable/catchable.

### 1.3 arcade render spike + FPS benchmark (RK-1 gate) — Opus
- **Do:** throwaway-but-kept `adapters/render/spike.py` drawing `N` instanced
  point/sprite agents (no sim), random positions, animated. Measure FPS at
  `N = 2k` and `N = 10k`. A `scripts/bench_render.py` prints FPS.
- **Decision gate:** record numbers in this plan's "Spike Results"; if `arcade`
  can't approach 30 FPS at 10k on the target iGPU, flag the `pyglet`/`moderngl`
  fallback (PRD §9) **before** building the real renderer (1.16).
- **Tests:** headless-importable; benchmark harness has a unit test on its
  timing/aggregation helper (FPS itself is measured manually on the laptop).

### 1.4 FloorPlan model + scenario schema + loader — Sonnet
- **Do:** `domain/floor_plan.py` (`FloorPlan`: walls, obstacles, walkable region,
  exits as typed structures — FR-3 R3.1); JSON scenario schema (`scenarios/schema.py`
  with `TypedDict`s); `adapters/io/scenario_loader.py` resolving bundled data via
  `pathlib` + UTF-8 from a package-relative `assets/` dir (R0.4, R3.3), pure-domain
  output (no render dep).
- **Tests:** a fixture scenario loads headless into `FloorPlan`; malformed JSON
  raises a typed error; walls/obstacles/exits round-trip; loader finds bundled
  data regardless of CWD.

### 1.5 Grid flow field (FR-4) — Opus
- **Do:** `pathfinding/flow_field.py` — rasterize walkable region to a grid;
  multi-source BFS/Dijkstra from exit cells producing per-cell cost + unit
  direction vector toward lowest-cost exit (R4.1/R4.2); bilinear sample at agent
  position → `f_exit` direction. Decouple build from stepping; cache per floor;
  expose bounded `recompute(blocked_cells)` for re-routing (R4.4, supports R4.3).
- **Tests:** following the field from any walkable cell reaches an exit without
  crossing walls; two exits → cells route to lower-cost exit; blocking an exit's
  cells re-routes affected cells to the other exit; recompute touches only the
  affected region (directional).

### 1.6 AgentState SoA + spawn — Sonnet
- **Do:** `domain/agent_state.py` — SoA arrays (`pos`, `vel`, `panic`, `goal`,
  `alive`); seeded spawn within walkable region (uses 1.2 RNG); `Agent` view
  exposing position/velocity/goal/panic for AC (FR-1 R1.1); `remove()`/`alive`
  mask so egressed agents exert/receive no force (R1.4).
- **Tests:** spawned agents lie in walkable region; agent view exposes 4 fields;
  marking dead removes from active set; same seed → same initial layout (R6.2).

### 1.7 f_exit + integrator + limits — Sonnet
- **Do:** `domain/forces.py::f_exit(state, field)` (vectorized field sample);
  `domain/integrator.py` semi-implicit Euler at fixed `DT`; clamp to max speed &
  max acceleration, optional panic-modulated desired speed up to the cap
  (R1.2/R1.3).
- **Tests:** a single agent on an empty floor moves toward and reaches the exit
  under summed forces; no tick exceeds max speed/accel; raising panic never
  exceeds the documented cap; adding a zero extra term doesn't change trajectory
  (extensibility — R1.2 AC).

### 1.8 Crowd dynamics (FR-2) — Opus
- **Do:** `domain/spatial_hash.py` (uniform-grid neighbor index, R2.4);
  `domain/forces.py::f_crowd` short-range repulsion (no overlap, R2.1); density
  pressure reducing effective speed at high local density (R2.2); herd term —
  attraction to local mean velocity scaled by panic (R2.5).
- **Tests:** two agents on a collision course deflect, never overlap (R2.1);
  throughput through a constriction drops monotonically as upstream density rises
  (R2.2 directional); neighbor query cost scales sub-quadratically to ~10k (R2.4
  — timed, lenient bound); herd attraction increases with panic (R2.5 directional).

### 1.9 Exit model + evac-complete (FR-5) — Sonnet
- **Do:** `domain/exit_model.py` — per-exit throughput capacity, queue when
  arrivals exceed capacity (R5.1); egress removes agent + increments evacuated
  count (R5.2); `application/termination.py` evac-complete when all evacuable
  agents egressed or no further egress possible (R5.3).
- **Tests:** egress rate never exceeds capacity over a 1s window; queue forms
  when arrival > capacity; evacuated count == initial evacuable on a fully
  solvable scenario; a trapped-agent scenario reaches terminal "no further
  egress" rather than looping forever.

### 1.10 PanicSource + gradient field + f_panic_repulsion (FR-11 subset) — Sonnet
- **Do:** `domain/panic_source.py` (`PanicSource`: position, intensity, radius,
  decay_rate — fire_visual is enough for Phase 1); `domain/panic_field.py`
  scalar gradient highest at source, decaying with distance (obstacle-aware
  decay; wall-reflection deferred to Phase 3); `forces.py::f_panic_repulsion`
  pushing agents down-gradient (R11.4); decay reduces intensity over time (R11.1).
- **Tests:** field is max at source, decays with distance; an isolated agent near
  a source moves away (down-gradient, R11.4); decay lowers intensity each tick and
  source expires near zero.

### 1.11 Combined force composition (FR-14 subset) — Opus
- **Do:** `domain/forces.py::compose(state, ...)` →
  `a_i = f_exit + f_panic_repulsion + f_crowd` (+ herd), each term individually
  **toggleable** for debugging (R14.1 AC). Superset-ready for `f_signage`
  (Phase 4) without integrator change.
- **Tests:** all present terms sum into acceleration; disabling a term removes its
  effect; toggling panic repulsion on near a source bends trajectories away vs.
  baseline.

### 1.12 Fixed-step sim loop + tick model + event log (FR-6) — Sonnet
- **Do:** `application/simulation.py` — fixed `DT` step orchestrating
  field-sample → compose forces → integrate → resolve exits → metrics; monotonic
  tick + elapsed sim time (R6.3); seeded RNG threaded through (R6.2); event-log
  foundation stamping events to tick (R6.3, NFR-R3); read-only `snapshot()` for
  render/metrics (R7.3 prep).
- **Tests:** tick increments by 1 per step; outcome independent of how often
  `snapshot()` is read (frame-rate independence proxy, R6.1); same seed + same
  scripted events → same recognizable end state (NFR-R1 best-effort); every
  recorded event carries its tick.

### 1.13 Core metrics records (FR-8) — Sonnet
- **Do:** `metrics/records.py` (`TypedDict`/dataclass per-tick record: tick,
  evac progress, per-exit + total throughput, density measure — R8.3) produced by
  the application layer from the read-only snapshot, available headless (R8.2).
- **Tests:** record is produced per tick headless; serializable; contains the
  four required fields; throughput matches egress events; density rises in a
  crowded region (directional).

### 1.14 Runtime injection API + re-route trigger (FR-12.1) — Sonnet
- **Do:** `application/injection.py::add_panic_source(type, pos, intensity,
  radius)` mutating the panic field within the same/next tick (R12.1); triggers a
  bounded flow-field `recompute` (1.5) so affected agents re-route (R12.2 / R4.3),
  logged as a tick-stamped event.
- **Tests:** calling mid-run updates the field by next tick; injection near a path
  causes affected agents' target/route to change (redirection observable,
  directional); event is logged with its tick.

### 1.15 Ports (FR-7) — Haiku *(no logic)*
- **Do:** `ports/` Protocols — `Renderer`, `InputSource`, `ScenarioRepository`,
  `Clock`. Pure interfaces; sim core depends on nothing from render.
- **Tests:** a trivial in-memory fake implements each Protocol and type-checks
  under `mypy --strict` (structural conformance).

### 1.16 arcade Renderer adapter (FR-7) — Sonnet
- **Do:** `adapters/render/arcade_renderer.py` implementing `Renderer`; draws
  walkable region, walls/obstacles, exits, agents from the **read-only snapshot**
  (instanced sprites per spike findings); optional background image (R7.1); never
  mutates domain state (R7.3). Apply spike fallback if 1.3 flagged one.
- **Tests:** renderer consumes a snapshot without mutating it (snapshot equality
  before/after); headless construction path doesn't import-fail; draw-call builder
  maps N agents → N instances (unit-level, no window).

### 1.17 Input adapter + interaction loop: drop/drag panic source (FR-7 R7.2 / FR-15 subset) — Sonnet
- **Do:** `adapters/render/arcade_input.py` implementing `InputSource`; mouse
  place + drag maps to `add_panic_source` / move-source commands routed through
  the application layer (not bypassing it — R7.2); INFO-log each command with its
  CLI-equivalent syntax (seed of R15.4).
- **Tests:** a place event produces the correct injection command; a drag event
  produces a move-source command at the new position; commands go through the app
  API (mocked) — handler registers on the loop.

### 1.18 Bundled Lecture Hall default scenario (FR-0, FR-13.1) — Haiku
- **Do:** author `assets/scenarios/lecture_hall.json` — tiered-seating-ish
  walkable region, walls/obstacles, front-stage exit (+ optional side exits),
  50–300 agents, sensible default parameters (panic decay, force weights, FPS/scale
  tier — R0.3). Versioned with the app, resolved via 1.4 loader (R0.4).
- **Tests:** the bundled file loads via the loader into a valid `FloorPlan` with
  agents in range; declared exits/obstacles present; aisle convergence reachable.

### 1.19 App entry point: launch into Lecture Hall (FR-0 R0.1) — Sonnet
- **Do:** `application/app.py` / `__main__.py` wiring loader → simulation →
  arcade renderer + input ports; on launch with no user data, load the Lecture
  Hall and start the loop (R0.1). PowerShell run steps in README (NFR-Q2).
- **Tests:** headless launch path builds a simulation from the default scenario
  without a window (renderer port mocked); starts at tick 0 with agents in range.

### 1.19a Bugfix after first App entry point
- **Fix:** agents are able to cross obstacles. Obstacles should NEVER be crossed.
- **Add:** simulation should begin paused. Add "Evacuate" button to initiate the exit.
- **Add:** the "Evacuate" button change to "Pause", further pressing "Pause" pauses the app and button change "Continue", and can go back to pause etc.
- **Add:** after evacuation is complete the button change to "Reset", pressing it resets the scenario and back to pause.
- **Add:** emergency sources should also include an icon at the position of the emergency source, shaped like the event type.
- **Fix:** entire area including all walls MUST be visible on the screen..
- **Modify:** emergency 1nfluence area should be more transparent.
- **Modify:** agent symbol size should be proportional to pixel per meter, showing diameter of 40 cm.
- **Fix:** the speed of the events seems not aligned to real time. Explore that and give insights.
- **Add:** velocity slider, from x0.1 to X3, with snap to x1.
- **Fix:** when queueing by the exit, any agent not exiting yet is still bound to forces. Currently it looks like many agents are positioned inside the exit and waiting for their turn, and the correct behavior should be that they can't get to the exit because of the density.


### 1.20 Headless end-to-end + seed-repro tests — Sonnet
- **Do:** `tests/e2e/` — run the default scenario headless to evacuation-complete;
  assert evacuated == initial evacuable (R5.3); same seed twice → recognizably
  same run (NFR-R1); halving snapshot/render cadence doesn't change sim outcome
  (R6.1). Coverage pass to ≥ 85% on core/capability modules (NFR-Q1).
- **Tests:** the above; `pytest --cov=src --cov-branch` ≥ 85%; fill gaps.

### 1.21 Tune defaults, set §8 thresholds, 10k benchmark, document fallback — Opus
- **Do:** run on the **target laptop**; tune default parameters for a readable,
  fun loop; **empirically set the §8 EB thresholds** now that baseline scales are
  observed (PRD §8 — no preset numbers); benchmark combined sim+render at ~2k and
  ~10k; record FPS, re-route latency (NFR-P3 ≤ 200 ms), footprint (NFR-P4 ≤ 6 GB);
  document the NFR-P2 fallback decision (instanced/active-level draw → sprite LOD →
  documented lower count). Write findings into this plan's "Spike Results".
- **Tests:** threshold values land in a config/constants module with a unit test
  asserting they're loaded and in range; benchmark script runs headless+windowed.

### 1.22 Case of no walls
- **Do:** decide how to solve case of floorplan with no walls or not enclosed (eg 3 walls). Open walls should be treated as exits.

### 1.23 Limited knowledge
- **Do:** decide how to reflect agents not knowing where the emergency exit is (eg no sign in sight). Possible it to follow the herd.

### 1.24 Spawn region
- **Do:** define spawn region within the walkable region. For example, in stadium spawn aget in the seating areas but not in the hall areas.

---

## Verification (end-to-end)

```powershell
# from project root, .venv activated
pip install -e ".[dev]"

# quality gates (run after every step)
flake8 src/
mypy src/ --strict
pytest tests/ -v

# coverage gate (Step 1.20)
pytest tests/ --cov=src --cov-branch --cov-report=term-missing   # >= 85%

# render scale spike (Step 1.3) — on the target laptop
python scripts/bench_render.py --agents 2000
python scripts/bench_render.py --agents 10000

# playable MVP (Step 1.19+) — should open directly into the Lecture Hall
python -m crowd_evac
```

**Manual acceptance (the Phase 1 / MVP gate, PRD §3 / §14):**
1. `python -m crowd_evac` launches straight into a running, interactive Lecture
   Hall — no files to author, no config to edit (R0.1).
2. The player drags a fire source around live; the crowd flees, re-routes, and a
   redirection wave is visible (FR-11/12.1/14 subset, NFR-P3).
3. The crowd jams at the aisle/exit constriction, then evacuates to completion
   under the fixed seeded loop (FR-2/FR-5).
4. Interactive scale (~2k) holds ≥ 30 FPS on the target laptop (NFR-P1); the
   10k Tier A benchmark and fallback decision are recorded (Step 1.21).

## Risks & Notes

- **RK-1 (iGPU @ 10k):** front-loaded in Step 1.3 so a `pyglet`/`moderngl` pivot
  (PRD §9) is cheap; the core stays behind the `Renderer` port regardless.
- **Phase-1 deferrals (kept out by design):** multi-level/FR-9, full PanicSource
  types + wall reflection (Phase 3), signage/FR-10, secondary cascade FR-12 R12.3,
  Tier B 50k + Numba (Phase 10). The force-composition API (1.11) and FloorPlan
  (1.4) are built superset-ready so these layers add terms/levels without a core
  rewrite (PRD A3/RK-8).
- **Synthesized core (PRD §0/§12):** FR-1..FR-9 are domain-standard inferences,
  replaceable if the original §1–10 surfaces; the additive-force design keeps a
  later swap from breaking the capability layers.
- **Sequencing:** 1.3 can run in parallel with 1.4/1.5 (independent spike). All
  domain steps (1.4–1.11) are headless-testable before any renderer exists.

## Spike Results (filled during execution)

- **Step 1.3 render-only FPS @ 2k / 10k on target laptop** (600 timed frames,
  30-frame warmup excluded, SpriteList of circles, no sim, vsync off):
  - 2k: mean **247.5** FPS / 1% low **124.5** / min **110.9** / max 332.8
    (mean frame 4.04 ms).
  - 10k: mean **57.6** FPS / 1% low **38.3** / min **36.4** / max 62.6
    (mean frame 17.35 ms).
- **Renderer decision (RK-1 gate): arcade — PASS.** At 10k agents even the
  worst frame (36.4 FPS) and 1% low (38.3 FPS) clear the 30 FPS bar (NFR-P2),
  so the real renderer (1.16) is built on arcade; the pyglet/moderngl fallback
  (PRD §9) is **not** taken. Caveat: render-only; headroom over 30 FPS at 10k
  is thin (~1.3× at the 1% low). The combined sim+render 10k measurement in
  Step 1.21 is the real gate — re-confirm there before committing to Tier A.
- **Env note (resolved 2026-06-03):** the `.venv` was rebuilt on **Python
  3.12.10** (the original 3.14.5 venv had no `pymunk~=6.9.0` wheel and couldn't
  compile it without MSVC). On 3.12, `pip install -e ".[dev]"` installs the full
  tree cleanly — arcade 3.3.3 + its pinned pymunk 6.9.0 (prebuilt wheel) — with
  no `--no-deps` hack. `flake8 src/`, `mypy src/ --strict`, and `pytest` (39
  tests) all pass on 3.12; the earlier mypy `arcade` import-not-found errors are
  gone now that arcade is installed. The pyglet `ctypes._pointer_type_cache`
  DeprecationWarning persists, so the targeted pytest `filterwarnings` ignore
  stays.
- Step 1.21 sim+render FPS @ 2k / 10k, re-route latency, footprint: _TBD_
- §8 EB-1..6 empirical thresholds: _TBD_

## Profiling Guide

Use this sequence to locate per-tick bottlenecks once the load tests
(`pytest -m perf -v -s`) establish a baseline or flag a regression.

### Step 1 — `cProfile` + `snakeviz` (identify the dominant call)

No new dependencies for profiling; install `snakeviz` only for the browser
visualisation.

```powershell
pip install snakeviz
python -m cProfile -o prof.out scripts/bench_sim_headless.py --agents 2000 --ticks 100
snakeviz prof.out
```

Opens an interactive sunburst in the browser.  Cumulative time per call
immediately shows which function owns the budget.  The usual suspects are
`forces.compose`, `spatial_hash.build`, and `_propagate_panic`.

### Step 2 — `line_profiler` (line-by-line inside a suspect function)

Once `cProfile` identifies the dominant function, install `line_profiler`,
decorate that function temporarily with `@profile`, and run:

```powershell
pip install line-profiler
kernprof -l -v scripts/bench_sim_headless.py --agents 2000 --ticks 50
```

Outputs microsecond-level time per line.  Essential for NumPy-heavy code
where a single vectorised expression may account for 80% of a function's
runtime.

### Step 3 — `py-spy` flame graph (zero-code-change validation)

Requires no decorators or code changes.  Run the benchmark as a subprocess
and sample the live call stack:

```powershell
pip install py-spy
py-spy record -o profile.svg -- python scripts/bench_sim_headless.py --agents 2000 --ticks 200
```

Produces a flame graph SVG.  Useful for confirming that an optimisation
actually moved wall time, or for profiling a run that is hard to annotate
(e.g., the arcade render loop inside `bench_sim_render.py`).

### Recommended order

1. Run `cProfile` — free, zero setup, narrows to the top 3 functions.
2. Apply `line_profiler` to whichever of `forces.py`, `spatial_hash.py`,
   or `integrator.py` appears at the top.
3. Use `py-spy` to validate before/after when the change is non-trivial.
