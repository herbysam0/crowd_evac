# PRD: Evacuation Dynamics — Physics + Social-Behavior Game (Spec Addendum §11–18)

| Field | Value |
|---|---|
| Status | DRAFT (revised — awaiting approval) |
| Author | AI (with user context) |
| Date | 2026-06-02 |
| Repo | `C:\Users\user\Documents\projects\crowd_evac` |
| Scope | Build from scratch: capability areas §11–18, framed as an interactive game. |
| Target reader | Solo / small-team engineering effort |

---

## 0. Repo Reconnaissance & Product Reframing

**Repo state.** The repository is empty except for `.claude/settings.local.json`.
There is **no baseline simulation code**. Decision (resolved): **there is no external
baseline codebase — we build from scratch.** All references in the spec to a
"pre-existing baseline (§1–10)" are treated as features *this project will create*,
not inherited code.

**Product reframing (resolved).** This is a **game built on physics + social-behavior
modeling — not a hi-fi research or scientific instrument.** The product is optimized
for **engaging, interactive, real-time play** that also teaches evacuation dynamics.
Plausibility and educational value matter; strict numerical rigor and research-grade
determinism do **not** gate the product. Reproducibility is downgraded to a
**nice-to-have** (seedable scenarios for sharing/replay), not a hard requirement.

Consequences applied throughout this PRD:
1. "Research-grade determinism / scientific validation" framing is **downgraded**.
2. Primary users are **(a) engineers building domain knowledge** and **(b) players** —
   the "safety researcher producing publishable results" persona is dropped as primary.
3. A **playable MVP phase** is added as the first roadmap phase.
4. Out-of-the-box default data/scenarios are a **first-class functional requirement**.
5. Performance/memory targets are set against a **specified integrated-GPU laptop**.
6. A concrete **clean-architecture, non-pygame** stack is recommended (sim core stays
   engine-agnostic).

---

## 1. Overview

A **single-player, real-time evacuation game** for Windows 11 where the player places
hazards and signage and watches a crowd evacuate a **multi-level building** governed by
plausible physics (steering/social forces) and social behavior (panic, signage trust,
herding). The game ships **ready to play out of the box** with pre-built environments,
default signage, sample hazards, and sensible default parameters — zero setup to launch.

It adds, on top of a 2D crowd-evacuation core: a multi-level topology layer (floors
connected by stairs **and elevators**), an emergency signage guidance field, a dynamic
hazard/panic-source system with live injection, a library of tuned scenario templates
(Lecture Hall, Stadium, Movie Theater, Industrial Plant), an interaction model blending
signage attraction with panic repulsion, authoring UI to drop hazards and signage live,
and a set of **observable emergent behaviors** that make play interesting and teach the
underlying dynamics.

## 2. Problem Statement (player/learner framing)

A flat single-floor crowd sim is neither fun nor instructive. The phenomena that make
evacuation dynamics *engaging to play with and worth learning* are exactly the ones a 2D
model can't express:

- **Vertical bottlenecks** — stairwells/elevators are where dramatic crushes and queues
  form; you can't feel the tension of a clogged stairwell in 2D.
- **Guidance vs. panic conflict** — the core "game" is whether your signage out-persuades
  the crowd's panic. No signage field, no game.
- **Dynamic hazards** — dragging a fire around and watching the crowd flee and re-route in
  real time is the central interactive loop.
- **Scenario variety & scale** — different venues (theater, stadium) create different
  puzzles; big crowds create spectacle.

## 3. Goals

1. Deliver a **playable, satisfying real-time loop**: place/drag hazards and signage,
   watch the crowd react and evacuate, read clear visual feedback.
2. Ship **out-of-the-box defaults** so the app launches straight into a playable scenario
   with no configuration.
3. Model evacuation across **multiple connected levels** with stairs **and elevators** as
   capacity-limited vertical bottlenecks, plus cross-floor pathfinding.
4. Provide a **signage guidance field** that visibly redistributes crowds and forms lanes,
   with a **configurable** relationship between panic and signage trust.
5. Provide a **dynamic hazard/panic system** with live injection and a spatial panic
   gradient agents flee, including **wall reflection** in enclosed mode.
6. Compose signage + panic + crowd + exit forces into one **interaction model** handling
   conflict scenarios (misleading signage, panic overriding trust).
7. Ship a **scenario library** of four tuned venues.
8. Provide **authoring UI** for hazards, signage, and multi-level navigation.
9. Make the **emergent behaviors (§8)** observable and (where cheap) auto-detected, to
   reward experimentation and support the "learn the dynamics" use case.
10. Sustain **real-time interactivity** on the target laptop (§7) at the documented scale
    tiers, with an **optional seedable mode** for shareable/replayable scenarios.

### What success looks like
- ✓ App launches into a default playable scenario with no setup; the core loop (place
  hazard/signage → crowd reacts → evacuates) is fun and readable.
- ✓ All four scenario templates load and evacuate to completion.
- ✓ Tier A (full-fidelity ~10k agents) hits the FPS target on the target laptop; the
  game is the release gate, Tier B (≤50k abstracted) is a stretch goal.
- ✓ The six §18 emergent behaviors are reproducible and observable; cheap ones are
  auto-detected for in-game feedback / learning metrics.
- ✓ Seedable mode: same seed + same event log replays the same scenario closely enough to
  share (best-effort, not byte-identical).

## 4. Non-Goals

1. **Research-grade / publishable scientific accuracy.** Plausible, not validated. No
   claim of numerical rigor or clinical casualty prediction.
2. **Byte-identical cross-machine determinism.** Seedable replay is best-effort only.
3. **Photoreal 3D rendering.** Visualization is 2.5D: 2D-per-level + stacked multi-level
   view/minimap. No photoreal 3D engine.
4. **Networked / multiplayer / multi-machine simulation.** Single-player, local only.
5. **Real building-data import** (BIM/IFC/CAD). Out of scope unless raised later.
6. **Validated medical injury/fatality modeling.** "Collapse" is a game state.
7. **Agent learning / ML.** Behavior is rule/force-based.
8. **Mobile / web deployment.**

## 5. Target Users & Use Cases

| User | Primary use case |
|---|---|
| **Engineers / domain learners** | Build intuition for evacuation dynamics by experimenting: vary signage/hazard/panic params, observe stairwell congestion, flow splitting, panic diffusion. Professional curiosity, not publication. |
| **Players** | Sandbox play: drop and drag hazards and signage, watch emergent crowd behavior, try to evacuate everyone / create dramatic failures. |

Representative user stories:

1. As an **engineer/learner**, I want to toggle "panic lowers trust" vs. "panic causes
   over-trust" and re-run, so that I can see how that single behavioral assumption changes
   the outcome and deepen my understanding.
2. As an **engineer/learner**, I want to place signage and re-run, so that I can see which
   exit layout reduces the worst bottleneck.
3. As a **player**, I want to drag a fire source around live, so that I can watch the crowd
   flee and re-route in real time.
4. As a **player**, I want to launch the game and immediately play a good default scenario,
   so that I don't have to build anything before having fun.
5. As an **engineer/learner**, I want to inject a gunfire source mid-run, so that I can see
   a redirection wave form.

---

## 6. Functional Requirements

Grouped by capability area. Each requirement has an ID and acceptance criteria (AC).
Priority: **P0** = must-have for release, **P1** = should-have, **P2** = optional/stretch.
Note: this is a game, so AC favor *observable in-game behavior* over numerical proofs;
unit tests assert directional/qualitative properties, not research-grade exactness.

### FR-10 Core 2D Crowd-Evacuation Engine (built from scratch) — P0

Because there is no baseline, the project must first build the minimal 2D core the
extension assumes.

- **R10.1** Agent model with position, velocity, goal, panic state; additive
  steering/social-force movement so later forces (R16.1) extend it without rewrite.
  - AC: agents move toward an exit on a single 2D floor and evacuate to completion.
- **R10.2** Floor-plan representation (walls/obstacles, walkable space, exits) and a
  navigation field for goal-seeking.
  - AC: agents avoid walls/obstacles and reach exits; a blocked exit forces a detour.
- **R10.3** Fixed-step simulation loop with a single seeded RNG (for seedable scenarios).
  - AC: loop runs at a fixed timestep; seed is recorded in scenario/run metadata.

### FR-0 Out-of-the-Box Defaults (zero-setup launch) — P0

- **R0.1** On first launch with no user data, the app loads the **default playable
  scenario — the Lecture Hall template (R15.1)** (a pre-built environment with crowd, exits,
  default signage, sample hazards, and sensible default parameters) and starts the core loop.
  Lecture Hall is the locked out-of-the-box default: it is the smallest scale and fastest to
  a readable, playable loop on the target iGPU.
  - AC: a clean install launches directly into the running, interactive **Lecture Hall**
    scenario; the player can immediately place/drag a hazard and see the crowd react — no
    files to author, no config to edit.
- **R0.2** All four scenario templates (FR-15) ship as **bundled default data** loadable
  from the UI without external assets.
  - AC: each template loads from the in-app library on a clean install.
- **R0.3** Ship **sensible default parameters** for every tunable (panic decay, signage
  λ, trust mode, force weights, FPS/scale tier) so nothing must be set before play.
  - AC: with zero user edits, all systems behave reasonably; defaults documented.
- **R0.4** Default assets are versioned with the app and resolved via `pathlib` from a
  bundled data directory (UTF-8), not the working directory.
  - AC: launching from any working directory still finds default scenarios.

### FR-11 Multi-Level Architecture (multi-level topology layer) — P0

- **R11.1** Represent a `Level` with: `id`, `elevation_z`, `floor_plan_graph`,
  `stair_nodes`, `elevator_nodes`.
  - AC: a scenario with ≥3 levels loads; each level exposes its own floor-plan graph and
    node sets (including elevator nodes) via the data model.
- **R11.2** Model vertical transitions (stairs/ramps/**elevators**) as directed
  `transition_edge`s with `capacity_per_second`, `directionality ∈ {bidirectional,
  one_way}`; elevators additionally have **batch capacity** and **travel/dwell time**.
  - AC: throughput across a transition edge never exceeds `capacity_per_second` over any
    1-second window; a one-way edge rejects reverse travel; an elevator carries at most
    its batch capacity per trip and incurs travel/dwell delay.
- **R11.3** Transition edges act as **vertical bottlenecks**: when arrival rate >
  capacity, agents **queue** at the entry node (including waiting for an elevator car).
  - AC: with arrival rate > capacity, a queue forms and grows; it drains when arrival rate
    drops; elevator queues form while a car is in transit.
- **R11.4** Panic increases stair **congestion-collapse probability**.
  - AC: holding density constant, higher mean panic raises per-tick collapse probability on
    a stair edge (monotonic relationship verified by a unit test).
- **R11.5** Multi-level pathfinding = level-graph traversal + intra-floor flow field:
  (a) select global target exit, (b) choose optimal level-transition sequence (stairs or
  elevator), (c) compute local movement vector. **Elevator-vs-stairs preference is
  scenario-tunable; the default selects whichever yields the lower estimated travel time**
  (accounting for stair traversal vs. elevator queue + travel/dwell delay per R11.2/R11.3).
  - AC: an agent on level N with an exit on level 0 produces a valid stair/elevator path
    and reaches the exit; with default weights the agent picks the lower-estimated-travel-time
    option; a scenario-level override changes the preference; if the chosen sequence becomes
    blocked, the agent re-routes (ties to R14.2).

### FR-12 Emergency Signage System (directional guidance field) — P0

- **R12.1** `Signage` object: `position`, `direction_vector`, `influence_radius`,
  `strength`, `visibility_score`, `compliance_modifier`.
  - AC: a sign only influences agents within `influence_radius`; influence scales with
    `strength` and `visibility_score`.
- **R12.2** Signage modifies goal force: **f_goal = f_exit + λ · f_signage**, where
  `λ = g(compliance_level, visibility, panic_state)`.
  - AC: with λ=0 agents ignore signage; increasing λ measurably bends trajectories toward
    `direction_vector`.
- **R12.3** Panic affects trust in a **runtime-configurable** direction — a toggle with
  **both** modes selectable at runtime: (a) higher panic → **lower** trust, and
  (b) higher panic → **over-trust**.
  - AC: both modes are selectable from config/UI at runtime; switching the toggle produces
    the documented opposite effect on λ within the same session.
- **R12.4** Visibility degrades signage influence (distance and smoke occlusion).
  - AC: an agent inside a smoke cloud has reduced effective `visibility_score` for signs
    occluded by that smoke.
- **R12.5** Signage enables crowd redistribution, single-bottleneck prevention, and guided
  lane formation.
  - AC: see emergent behavior EB-3 (§8) — flow splitting/re-merging is observable.

### FR-13 Dynamic Hazard & Panic Source System — P0

- **R13.1** `PanicSource` types: `fire_visual`, `smoke`, `explosion_sound`,
  `gunfire_audio`, `structural_collapse`. Each: `position`, `intensity`, `radius`,
  `decay_rate`.
  - AC: each type exists with these fields; `decay_rate` reduces effective intensity over
    time and a source expires near zero intensity.
- **R13.2** Agent perception model:
  **p_i = f(visual) + f(audio) + f(local_density) + f(secondary_reinforcement)**.
  - AC: visual contribution rises fastest; audio is non-zero **beyond line of sight**;
    higher local density amplifies p_i (each term unit-tested for direction).
- **R13.3** Maintain a **spatial panic gradient** scalar field: high near sources, decaying
  with distance and **obstacles**; **wall reflection in enclosed mode** (in scope for
  release).
  - AC: field value is highest at the source and decays with obstacle-aware distance; with
    enclosed mode enabled, field near a wall is higher than the free-field value at the same
    distance (reflection contribution observable).
- **R13.4** Agents **drift away** from gradient peaks (panic repulsion force).
  - AC: an isolated agent near a source moves down-gradient (away) absent other forces.

### FR-14 Dynamic Multi-Source Panic Injection (runtime events) — P0

- **R14.1** `AddPanicSource(type, position, intensity, radius)` injectable at runtime;
  immediately modifies the local panic field.
  - AC: calling the API mid-run updates the panic field within the same or next tick.
- **R14.2** Injection **re-triggers path recalculation**, causing crowd redirection waves.
  - AC: after injection near a path, affected agents recompute routes and a visible
    redirection wave propagates outward from the source.
- **R14.3** **Secondary panic cascade** (rumor propagation / wavefront):
  **p_i(t+1) = p_i(t) + γ · neighbor_panic**, producing an expanding secondary wavefront.
  *(Later-phase feature — post-release / late roadmap, not MVP, not initial release-gating.
  See §15 Phase 9.)*
  - AC: with cascade enabled and γ>0, panic spreads to neighbors not in line of sight of
    the source; with γ=0 it does not.

### FR-15 Pre-Made Environment Library (scenario templates) — P0

Each template is a tuned, loadable scenario shipped as default data (FR-0) with documented
topology, exits, bottlenecks, and agent-count range.

- **R15.1 Lecture Hall** — 50–300 agents; tiered seating; front-stage exit + optional side
  exits; aisle-convergence bottlenecks; high frontal congestion.
  - AC: loads in range; aisle convergence produces observable front congestion.
- **R15.2 Stadium / Arena** — radial seating; multiple gated exits; concourse ring;
  multi-level (seating bowl / concourse / external exits); radial collapse; multi-exit
  competition; stair bottlenecks. **Tier A (full per-agent ~10k) gates release; Tier B
  (≤50k abstracted) is a stretch goal.**
  - AC: loads and plays at Tier A (~10k full per-agent) on the target laptop; exits
    compete; stairs bottleneck. Tier B abstraction (≤50k) is stretch and, if shipped,
    documents its abstraction model.
- **R15.3 Movie Theater** — 80–400 agents; long central aisle; side-wall exits; sloped
  floor; low visibility; row-based congestion; aisle locking.
  - AC: loads in range; low-visibility default applied; row egress produces aisle locking.
- **R15.4 Industrial Plant** — machinery obstacles; narrow service corridors; hazard zones;
  maze navigation; local deadlocks.
  - AC: plant loads with obstacles/corridors/hazard zones; local deadlocks are reachable and
    detectable.

### FR-16 System-Level Interaction (signage + panic + crowd) — P0

- **R16.1** Combined movement vector:
  **a_i = f_exit + f_signage + f_panic_repulsion + f_crowd**.
  - AC: all four terms are present and individually toggleable for debugging; disabling a
    term removes its effect.
- **R16.2** Support conflict scenarios:
  (a) signage directing toward a **blocked** exit,
  (b) panic **overriding** signage trust,
  (c) **contradictory multi-signage** fields,
  (d) dynamic **re-weighting under stress**.
  - AC: (a) agents detour from a blocked exit despite the sign; (b) above a panic threshold,
    panic repulsion dominates signage; (c) opposing signs produce a measurable
    split/oscillation, not a crash; (d) term weights shift with mean panic per a documented
    function (ties to EB-6).

### FR-17 UI Extensions — P0 (a,b,c functional; polish P1)

- **R17.1 Panic Source Editor Panel** — add fire zone / gunshot source / smoke cloud; move
  a source in real time; intensity slider. *(Secondary-cascade trigger is a later-phase
  add, per FR-14.3 / Phase 9.)*
  - AC: each action maps to the underlying API (R14.1) and updates the sim live.
- **R17.2 Signage Placement Tool** — click-to-place directional arrows; define flow
  corridors; adjust influence radius; toggle visibility / smoke occlusion; **toggle
  panic→trust mode** (R12.3).
  - AC: placed signs appear in the field (R12.1) immediately; radius/visibility/trust-mode
    edits update influence live.
- **R17.3 Multi-Level Navigation View** — floor selector tabs; stacked floor minimap;
  stair/elevator-connection overlay graph; per-level heatmap.
  - AC: tabs switch the active level; minimap shows all levels stacked; overlay shows
    transition edges (stairs + elevators); per-level heatmap renders density (default) with
    a toggle to panic.

---

## 7. Non-Functional Requirements

### Target hardware (resolved)

The game runs on the **same modest laptop it ships for**:
- **OS:** Windows 11
- **CPU:** AMD Ryzen 5 Pro 7535U (6c/12t, ~2.90 GHz base)
- **GPU:** **integrated** AMD Radeon Graphics (shares system RAM)
- **RAM:** 16 GB @ 4800 MT/s

This is an **integrated-GPU APU**. Targets below are set realistically against it. The
**integrated GPU is the binding rendering constraint**, especially for 10k-agent Tier A
and multi-level views; rendering 10k+ instanced agents on an iGPU requires batched/instanced
draw calls and culling of off-screen / non-active levels, and the sim core must stay off
the render thread.

### Performance & Scale (targets set against the laptop above)
- **NFR-P1** Interactive scenarios (Lecture Hall, Theater, Industrial Plant, ≤ ~2,000
  agents): **≥ 30 FPS** on the target laptop.
- **NFR-P2** Tier A — **up to ~10k agents at ≥ 30 FPS, full per-agent fidelity** — is the
  **release gate**. If the iGPU cannot sustain 30 FPS at 10k while rendering, the fallback
  order is: (1) instanced/batched rendering + active-level-only draw, (2) reduced render
  detail (point/sprite agents) above a count threshold, (3) lower the guaranteed-30-FPS
  count to the highest count the iGPU sustains and document it. Tier B (10k–50k via
  spatial/flow abstraction, fast-forwardable, ≥ 15 FPS) is a **stretch goal**.
  - AC: Tier A target met and benchmarked on the target laptop.
- **NFR-P3** Runtime panic-injection re-route latency: redirection wave begins within
  **≤ 200 ms** (or ≤ N ticks) of `AddPanicSource`.
- **NFR-P4** Memory budget: total app footprint **≤ 6 GB** at Tier A (leaving headroom on a
  16 GB machine where the iGPU also consumes shared RAM); document footprint if Tier B is
  attempted.

### Seedability / Replay (downgraded from determinism — nice-to-have)
- **NFR-R1** A **seedable mode**: same seed + same ordered event log replays the *same
  scenario* closely enough to **share and re-watch** (best-effort reproducibility, not
  byte-identical, not cross-machine). Determinism is **not** a release gate.
  - AC: replaying a saved seed+event-log reproduces the scenario recognizably (same initial
    layout, same scripted events at the same ticks).
- **NFR-R2** All randomness routed through a single seeded RNG; seed recorded in scenario
  metadata so scenarios are shareable.
- **NFR-R3** Runtime events (signage/hazard edits) are timestamped to sim-tick and recorded
  to a replayable event log.

### Other
- **NFR-Q1 Code quality:** `flake8` clean, `mypy --strict` clean, ≥ 85% test coverage on
  core/extension modules (per project standards).
- **NFR-Q2 Platform:** Windows 11; `pathlib` for all paths; UTF-8 I/O; PowerShell-documented
  run steps.
- **NFR-Q3 Observability:** structured metrics (per-tick queue lengths, density, panic,
  throughput, collapse events) for in-game feedback, learning metrics, and §8 checks.
- **NFR-Q4 Stability:** no scenario crashes or deadlocks the process; in-sim agent deadlocks
  are detected and surfaced (visual/log), not silent.

---

## 8. Emergent-Behavior Scenarios (§18 — observable, partly auto-detected)

These make play interesting and teach the dynamics. Each should be reproducible from a
saved scenario + seed + event log. Where cheap, an **automated detector** provides in-game
feedback / learning metrics; otherwise visual observation is acceptable (this is a game, not
a validation suite). **Thresholds are defined empirically in Phase 1** — no preset numbers.

| ID | Emergent behavior | Setup | Observable / pass signal |
|---|---|---|---|
| **EB-1** | Cross-floor congestion propagation | Multi-level; funnel upper floor to shared stair/elevator | Upper-floor queue growth causes downstream queue/density rise on the connected lower level, with a time lag. |
| **EB-2** | Stairwell collapse under panic pressure | High density + rising panic on one stair edge | A `collapse` event fires; collapse rate increases with panic at fixed density (R11.4). |
| **EB-3** | Signage-induced flow splitting & re-merging | One crowd, two exits, signs steering a fraction to exit B | Exit-B share rises with λ; lanes split then re-merge downstream. |
| **EB-4** | Panic wave diffusion across crowd clusters | Inject one source near a cluster | Panic spreads outward over time with a visible propagation speed (full non-LOS spread needs the later cascade feature, FR-14.3). |
| **EB-5** | Multi-source panic interference patterns | Two sources with overlapping gradients | Combined field shows interference (saddle/ridge); overlap-region drift differs from single-source baseline. |
| **EB-6** | False-optimal routing under misleading signage | Sign points toward a soon-to-be-blocked/hazardous exit | A fraction initially routes to the false-optimal exit, then re-routes after encountering block/panic. |

> **Thresholds:** exact numeric pass thresholds are **defined empirically in Phase 1** once
> baseline scales are observed (resolved — no preset numbers). EB-4's full non-LOS spread is
> gated on the later cascade feature; until then EB-4 is satisfied by line-of-sight diffusion.

---

## 9. Recommended Architecture & Tech Stack

**Non-negotiables (resolved):** clean architecture with the **simulation core fully
decoupled from rendering** (engine-agnostic core), **avoid pygame**, must render a 2.5D /
multi-level game with **~10k agents on an integrated-GPU laptop**, and fit a
solo/small-team effort.

### Clean-architecture spine (engine-agnostic, applies to every option)

```
domain (pure):      Level, TransitionEdge, Agent, Signage, PanicSource, fields, forces
                    — no rendering, no I/O, no engine imports; pure NumPy + Python
application:        fixed-step sim loop, seeded RNG, event log, injection API, metrics
ports (interfaces): Renderer, InputSource, ScenarioRepository, Clock
adapters:           concrete renderer/UI, scenario loaders (bundled default data), input
```

The sim core depends on **nothing** from the render layer. Renderers are adapters behind a
`Renderer` port, so the engine choice is reversible and the core is unit-testable headless.

### D6 — Engine/stack (RESOLVED: `arcade`)

The architectural fork (*game + not pygame + 10k agents on iGPU + clean architecture*) is
**resolved: the chosen default renderer adapter is `arcade`** (Python core stays
NumPy + optional Numba). The core remains Python/NumPy behind the `Renderer` port in all
cases, so the simulation is portable and the choice stays reversible. The two options below
are retained as **documented alternatives / fallbacks**, not open choices.

**Chosen — Python core (NumPy, optional Numba) + `arcade` renderer (OpenGL-based).**
Keep everything in Python; `arcade` gives modern GPU-accelerated, **instanced 2D sprite
rendering** (far better for 10k agents than pygame's per-sprite blits) with a clean,
well-documented API and good Windows support.
- Pros: single language; instanced rendering suits 10k point/sprite agents on an iGPU;
  fast to build solo; trivially keeps core decoupled.
- **Accepted trade-offs:** 2.5D multi-level is **faked via stacked 2D layering, not true
  3D** (no native depth); the **Tier B 50k stretch may require Numba acceleration** of the
  sim core since very high agent counts lean on the CPU; smaller ecosystem than a full
  engine.

**Documented alternatives / fallbacks (not chosen):**

1. **Python core (NumPy/Numba) + `pyglet` or raw `moderngl` renderer.** Same Python core,
   but render with lower-level OpenGL (instanced draws, custom shaders) for maximum control
   over iGPU batching. Fallback if `arcade` can't sustain 30 FPS at 10k, or if the Tier B
   50k stretch needs finer iGPU control than `arcade` exposes. More rendering/UI to write
   yourself; steeper for a solo dev.
2. **Python sim core as a headless service + a Godot 4 front-end.** Keep the NumPy core
   headless; Godot handles rendering/UI/2.5D via the `Renderer`/`Input` ports. Fallback only
   if true 2.5D presentation/UX becomes a primary goal. Two-runtime complexity and a
   process/binding boundary; highest setup cost for a solo dev.

**Decision:** ship on **`arcade`**. Escalate to alternative 1 (`pyglet`/`moderngl`) only if
profiling shows `arcade` can't sustain 30 FPS at 10k on the iGPU **or** the Tier B 50k
stretch needs more iGPU control; escalate to alternative 2 (Godot) only if 2.5D presentation
becomes a primary goal. The core stays behind the `Renderer` port so any such pivot is
cheap. *(`arcade`/`pyglet`/`moderngl`/`Numba` to be vetted via dependency governance before
adoption.)*

Proposed module layout (clean-architecture):
```
src/crowd_evac/
├── domain/         # Agent, Level, TransitionEdge (stairs+elevators), Signage,
│                   #   PanicSource, fields, forces — pure, no engine imports (FR-10..16)
├── pathfinding/    # level-graph traversal + flow field (R11.5)
├── application/    # fixed-step loop, seeded RNG, event log, injection API (FR-14, NFR-R*)
├── scenarios/      # template library + bundled DEFAULT data (FR-0, FR-15)
├── metrics/        # exporters + EB-1..6 detectors (§8)
├── ports/          # Renderer / Input / ScenarioRepository / Clock interfaces
└── adapters/
    ├── render/     # arcade (default) renderer behind Renderer port (FR-17, NFR-P*)
    └── io/         # scenario loaders, default-data resolution via pathlib (R0.4)
assets/             # bundled default scenarios/parameters (FR-0)
tests/              # mirrors src/ (core is testable headless)
docs/               # this PRD, plan.md
```

Architecture is finalized in `/plan`.

---

## 10. Assumptions & Dependencies

**Assumptions**
- **A1:** No external baseline exists — the project **builds the 2D core from scratch**
  (FR-10) before the extension layers (resolved).
- **A2:** The core force model is additive (steering/social-force) so the combined vector in
  R16.1 extends it cleanly.
- **A3:** Target machine is the specified Ryzen 5 Pro 7535U / integrated Radeon / 16 GB
  Windows 11 laptop (resolved, §7).
- **A4:** "Multi-level/3D" means a multi-layer navigable graph + stacked 2.5D views, not
  photoreal 3D.
- **A5:** Stadium 50k (Tier B) is a stretch goal via abstraction; ~10k full per-agent
  (Tier A) gates release.
- **A6:** Reproducibility is best-effort/shareable, not a hard determinism guarantee.

**Dependencies**
- D-ext1: Chosen rendering/engine (D6) — default `arcade`.
- D-ext2: Bundled default scenario data authored before release (FR-0).

---

## 11. Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| RK-1 | 10k agents can't hit 30 FPS on the iGPU (render-bound) | Misses Tier A release gate | Instanced/batched rendering, active-level-only draw, sprite/point LOD; NumPy-vectorized sim off the render thread; profile early (Phase 1 spike); documented FPS/count fallback (NFR-P2). |
| RK-2 | Tier B 50k infeasible | Misses stretch only | Tier B is explicitly stretch; abstraction + optional Numba; not release-gating. |
| RK-3 | "Fun/playable" is subjective | MVP ships but isn't engaging | MVP gate is a concrete playable loop (FR-0 + drag-hazard-and-react); iterate on feel after MVP. |
| RK-4 | Configurable panic→trust toggle doubles behavior surface | Test/verification cost | Single runtime enum with two well-tested modes; ship a default; both covered by directional tests. |
| RK-5 | Emergent thresholds (§8) hard to pin | Confusing feedback | Define empirically in Phase 1; default to qualitative/visual signals where detectors are expensive. |
| RK-6 | Scope creep from later/optional features (cascade, Tier B) | Timeline slip | Cascade (FR-14.3) and Tier B are explicitly post-release/stretch; gate behind approval; default off. |
| RK-7 | Building 2D core from scratch underestimated | Slips everything | MVP scopes the smallest viable core (FR-10) first; defer richness until the loop is playable. |

---

## 12. Open Questions

**None.** All prior questions are resolved and folded into the FRs/NFRs/architecture
(see §0 and §13). For the record, the previously-tracked items closed in this revision:

- **Q-A (closed):** First-launch default scenario = **Lecture Hall** — smallest scale,
  fastest to a readable, playable loop on the iGPU. Locked into R0.1.
- **Q-B (closed):** Elevator-vs-stairs routing = **scenario-tunable, default selects lower
  estimated travel time**. Locked into R11.5.
- **D6 (closed):** Renderer = **`arcade`** (default), with `pyglet`/`moderngl` and Godot as
  documented fallbacks. Locked into §9.

(Resolved earlier: baseline existence, stadium scale gating, elevators, emergent thresholds,
panic→trust direction, wall reflection, secondary cascade timing, and target machine.)

## 13. Resolved Decisions (folded into requirements)

- **No external baseline** → build 2D core from scratch (FR-10; A1).
- **Out-of-the-box defaults** → first-class FR (FR-0).
- **Stadium scale** → Tier A (~10k full) gates release; Tier B (≤50k abstracted) is stretch
  (R15.2, NFR-P2).
- **Elevators** → in scope for release (FR-11; R11.1–R11.3, R11.5; R17.3).
- **Emergent thresholds** → defined empirically in Phase 1 (§8).
- **Panic→signage-trust** → runtime toggle, both "lower trust" and "over-trust" selectable
  (R12.3; R17.2).
- **Wall reflection (enclosed mode)** → in scope for release (R13.3).
- **Secondary panic cascade (rumor wavefront)** → later phase, post-release (FR-14.3;
  §15 Phase 9).
- **Target machine** → Ryzen 5 Pro 7535U / integrated Radeon / 16 GB / Win 11; targets set
  against it; integrated-GPU rendering constraint called out (§7).
- **Tech stack (D6)** → clean architecture, engine-agnostic core, avoid pygame; **chosen:**
  Python+NumPy(+optional Numba) core with an **`arcade`** renderer; `pyglet`/`moderngl` and a
  Godot front-end retained as documented fallbacks (§9). Accepted trade-offs: 2.5D faked via
  layering (not true 3D); Tier B 50k stretch may require Numba acceleration.
- **First-launch default scenario (Q-A)** → **Lecture Hall** (R0.1, R15.1).
- **Elevator-vs-stairs routing (Q-B)** → scenario-tunable, default = lower estimated travel
  time (R11.5).

**No open or remaining decisions.** All forks are resolved and folded into the requirements
above; `/plan` can proceed without further input.

---

## 14. Success Metrics

1. **Playable MVP:** clean install launches into a default scenario; the core loop (place/
   drag hazard → crowd reacts/re-routes → evacuates) works and reads clearly (FR-0, FR-10,
   FR-13/14, FR-16).
2. **Functional completeness:** 100% of P0 requirements pass AC.
3. **Emergent behaviors:** 6/6 §8 scenarios reproducible and observable; cheap ones
   auto-detected.
4. **Performance:** Tier A ~10k ≥ 30 FPS and interactive scenarios ≥ 30 FPS on the target
   laptop; injection re-route ≤ 200 ms (NFR-P1/P2/P3); footprint ≤ 6 GB (NFR-P4).
5. **Quality:** `mypy --strict` + `flake8` clean; ≥ 85% coverage on core/extension modules.
6. **Library:** all 4 templates load and evacuate to completion.

---

## 15. Phased Delivery Roadmap

Phase 1 is now an explicit **playable MVP**: the smallest set of features that delivers a
valuable, playable loop. Later phases layer fidelity, scale, and the post-release cascade.
Each phase is independently completable and ends in a checkpoint.

| # | Phase | Objective | Depends on | Priority |
|---|---|---|---|---|
| 1 | **MVP — playable loop** | From-scratch 2D core (FR-10) + zero-setup default scenario (FR-0) + one panic source the player can drop/drag with crowd flee + re-route (subset of FR-13/14.1/16) + minimal `arcade` render/UI + seeded loop. **Pin stack (D6).** Spike iGPU rendering at scale (RK-1). Set §8 thresholds empirically here. | D6 | P0 |
| 2 | Multi-level topology | Levels, transition edges (stairs **and elevators**: capacity, directionality, batch/dwell), queuing, multi-level pathfinding (FR-11) | 1 | P0 |
| 3 | Panic field & hazards | Full PanicSource types, perception model, spatial gradient with **wall reflection (enclosed mode)**, drift-away force (FR-13) | 1 | P0 |
| 4 | Signage field | Signage objects, f_goal blend with λ, visibility/occlusion, **runtime panic→trust toggle** (FR-12, R12.3) | 1 | P0 |
| 5 | Interaction model | Combine a_i terms; conflict handling + stress re-weighting (FR-16) | 2,3,4 | P0 |
| 6 | Runtime injection | AddPanicSource live; re-route waves (FR-14.1/14.2) | 3,5 | P0 |
| 7 | Scenario library | Lecture Hall, Movie Theater, Industrial Plant, Stadium **Tier A** (~10k); all shipped as bundled default data (FR-15, FR-0) | 2,5 | P0 |
| 8 | UI extensions | Panic editor, signage tool (incl. trust toggle), multi-level navigation view (stairs+elevators overlay) (FR-17) | 5,6,7 | P0 |
| 9 | Emergent behaviors | EB-1..6 detectors + empirically-set thresholds; observable acceptance run (§8, §18) | all | P0 |
| 10 | Post-release / stretch | **Secondary panic cascade / rumor wavefront (FR-14.3)**, **Stadium Tier B (≤50k abstracted)** | as needed | P1/P2 |

**Critical path:** 1 (MVP) → 2 → (3,4 parallel) → 5 → 6 → 7 → 8 → 9. Cascade and Tier B
(Phase 10) follow release.

**Model assignments (per project planning rule):** Phase 1 core/spike — Opus (architectural
fork + scale spike); Phases 2–4 — Sonnet; Phase 5/9 — Opus (interactions / cross-system);
Phases 6–8 — Sonnet; scaffolding/defaults — Haiku.

Phase-level detail (steps, files, success criteria) is produced by `/plan` into
`docs/plan.md` after approval.

---

## 16. Next Step

On approval, run `/plan` to break FR-0/FR-10..FR-17 and EB-1..6 into atomic, phase-sized
tasks with architecture detail in `docs/plan.md`.

[YOU DO] Review. All decisions (D6 renderer, Q-A default scenario, Q-B routing) are now
resolved and locked — no open items remain. Reply with "Approved — proceed to /plan" or any
change requests.
