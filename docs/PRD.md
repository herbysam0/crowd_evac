# PRD: Evacuation Dynamics — Physics + Social-Behavior Evacuation Game

| Field | Value |
|---|---|
| Status | DRAFT (consolidated whole-product PRD — awaiting approval) |
| Author | AI (with user context) |
| Date | 2026-06-02 |
| Repo | `C:\Users\user\Documents\projects\crowd_evac` |
| Scope | Build the entire product from scratch: a 2D crowd-evacuation core plus multi-level, signage, hazard, scenario, interaction, and UI capability layers, framed as an interactive game. |
| Target reader | Solo / small-team engineering effort |

---

## 0. Repo Reconnaissance & Product Reframing

**Repo state.** The repository is empty except for `.claude/settings.local.json`.
There is **no baseline simulation code**. Decision (resolved): **there is no external
baseline codebase — we build the whole product from scratch.** This document is the single,
authoritative product requirements document covering both the core simulation and every
capability layer on top of it.

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

> **Provenance note on the core (FR-1..FR-9).** The original project spec described a
> "pre-existing baseline (§1–10)" but the §1–10 text was **never provided**. The core
> requirements in §6 (**FR-1 through FR-9**) are therefore **synthesized from (a) standard
> 2D crowd-evacuation domain modeling — social-force / steering-based agents — and (b) the
> structural assumptions the original addendum (§11–18, now FR-9..FR-15) depends on**: an
> additive force model (so FR-14's `a_i = f_exit + f_signage + f_panic_repulsion + f_crowd`
> composes cleanly), floor plans with walls/obstacles/walkable space/exits, per-agent panic
> state, a fixed-step seeded sim loop, basic crowd density/collision handling, and a base
> render/UI loop. **These core FRs are domain-standard inferences, not transcribed from the
> user's original spec, and may be reviewed or replaced if the real §1–10 is supplied.** The
> capability layers (FR-9..FR-15) are transcribed from the supplied addendum and carry the
> locked decisions verbatim.

---

## 1. Overview

A **single-player, real-time evacuation game** for Windows 11 where the player places
hazards and signage and watches a crowd evacuate a **multi-level building** governed by
plausible physics (steering/social forces) and social behavior (panic, signage trust,
herding). The game ships **ready to play out of the box** with pre-built environments,
default signage, sample hazards, and sensible default parameters — zero setup to launch.

The product is built bottom-up:

- a **2D crowd-evacuation core** — force-based agents, crowd dynamics, an authored floor-plan
  model, a single-floor navigation field, an exit/egress model, a fixed-step seeded loop,
  a base render/interaction loop, and core metrics (FR-1..FR-9); and
- **capability layers** on top of that core: a multi-level topology layer (floors connected
  by stairs **and elevators**), an emergency signage guidance field, a dynamic hazard/panic
  source system with live injection, a library of tuned scenario templates (Lecture Hall,
  Stadium, Movie Theater, Industrial Plant), an interaction model blending signage attraction
  with panic repulsion, authoring UI to drop hazards and signage live, and a set of
  **observable emergent behaviors** that make play interesting and teach the underlying
  dynamics (FR-9..FR-15).

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

But none of those layers stand alone: each rests on a believable **2D crowd-evacuation core**
(agents that steer, crowd that jams, exits that drain). The product must therefore build that
core first, then the layers that make it a game.

## 3. Goals

1. Deliver a believable **2D crowd-evacuation core**: force-based agents on an authored floor
   plan that steer toward exits, jam realistically, and evacuate to completion under a
   fixed-step seeded loop.
2. Deliver a **playable, satisfying real-time loop**: place/drag hazards and signage,
   watch the crowd react and evacuate, read clear visual feedback.
3. Ship **out-of-the-box defaults** so the app launches straight into a playable scenario
   with no configuration.
4. Model evacuation across **multiple connected levels** with stairs **and elevators** as
   capacity-limited vertical bottlenecks, plus cross-floor pathfinding.
5. Provide a **signage guidance field** that visibly redistributes crowds and forms lanes,
   with a **configurable** relationship between panic and signage trust.
6. Provide a **dynamic hazard/panic system** with live injection and a spatial panic
   gradient agents flee, including **wall reflection** in enclosed mode.
7. Compose signage + panic + crowd + exit forces into one **interaction model** handling
   conflict scenarios (misleading signage, panic overriding trust).
8. Ship a **scenario library** of four tuned venues.
9. Provide **authoring UI** for hazards, signage, and multi-level navigation.
10. Make the **emergent behaviors (§8)** observable and (where cheap) auto-detected, to
    reward experimentation and support the "learn the dynamics" use case.
11. Sustain **real-time interactivity** on the target laptop (§7) at the documented scale
    tiers, with an **optional seedable mode** for shareable/replayable scenarios.

### What success looks like
- ✓ The 2D core is believable on its own: agents steer to exits, crowds jam at constrictions,
  and a scenario evacuates to completion under a fixed, seeded loop.
- ✓ App launches into a default playable scenario with no setup; the core loop (place
  hazard/signage → crowd reacts → evacuates) is fun and readable.
- ✓ All four scenario templates load and evacuate to completion.
- ✓ Tier A (full-fidelity ~10k agents) hits the FPS target on the target laptop; the
  game is the release gate, Tier B (≤50k abstracted) is a stretch goal.
- ✓ The six §8 emergent behaviors are reproducible and observable; cheap ones are
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
5. **Real building-data import** (BIM/IFC/CAD). Out of scope unless raised later. Floor
   plans are authored in the project's own scenario format (FR-3), not imported from CAD.
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
6. As an **engineer/learner**, I want to learn the architecture of the implementation.

---

## 6. Functional Requirements

Grouped by capability area. Each requirement has an ID and acceptance criteria (AC).
Priority: **P0** = must-have for release, **P1** = should-have, **P2** = optional/stretch.
Note: this is a game, so AC favor *observable in-game behavior* over numerical proofs;
unit tests assert directional/qualitative properties, not research-grade exactness.

The functional requirements are organized as **core first** (FR-0 out-of-the-box defaults,
then FR-1..FR-9 the from-scratch 2D core), **then the capability layers** (FR-9..FR-15)
that build on that core.

> **Core synthesis note (applies to FR-1..FR-9).** As stated in §0, FR-1 through FR-9 are
> **synthesized from domain-standard 2D crowd-evacuation modeling and the structural
> assumptions of the FR-9..FR-15 layers**, not transcribed from the user's original §1–10
> (which was never supplied). They are reasonable, replaceable defaults. If the real §1–10
> arrives, treat these FRs as the section to reconcile against it.

### FR-0 Out-of-the-Box Defaults (zero-setup launch) — P0

- **R0.1** On first launch with no user data, the app loads the **default playable
  scenario — the Lecture Hall template (R13.1)** (a pre-built environment with crowd, exits,
  default signage, sample hazards, and sensible default parameters) and starts the core loop.
  Lecture Hall is the locked out-of-the-box default: it is the smallest scale and fastest to
  a readable, playable loop on the target iGPU.
  - AC: a clean install launches directly into the running, interactive **Lecture Hall**
    scenario; the player can immediately place/drag a hazard and see the crowd react — no
    files to author, no config to edit.
- **R0.2** All four scenario templates (FR-13) ship as **bundled default data** loadable
  from the UI without external assets. Only the Lecture Hall reequired on first release, other scenarios can be implemented in later releases.
  - AC: each template loads from the in-app library on a clean install.
- **R0.3** Ship **sensible default parameters** for every tunable (panic decay, signage
  λ, trust mode, force weights, FPS/scale tier) so nothing must be set before play.
  - AC: with zero user edits, all systems behave reasonably; defaults documented.
- **R0.4** Default assets are versioned with the app and resolved via `pathlib` from a
  bundled data directory (UTF-8), not the working directory.
  - AC: launching from any working directory still finds default scenarios.

### FR-1 Agent Model & Movement (core — synthesized) — P0

- **R1.1** Each agent carries explicit state: `position` (2D), `velocity` (2D),
  `goal` (current target node/exit), and `panic` (scalar in a documented range). Panic state
  is first-class from the start so later layers (FR-10 signage trust, FR-11 panic field,
  FR-14 interaction) read/write it without a model rewrite.
  - AC: an agent exposes position, velocity, goal, and panic; panic is readable/writable by
    the application layer.
- **R1.2** Movement is **additive steering / social-force integration**: per tick, a set of
  force terms is summed into a desired acceleration, integrated into velocity then position
  under a fixed timestep. The summation API is explicitly **extensible** so FR-14's
  `a_i = f_exit + f_signage + f_panic_repulsion + f_crowd` is a superset of the core terms,
  not a replacement.
  - AC: an agent moves toward its goal on a single 2D floor purely under summed forces;
    adding a new force term changes trajectory without modifying the integrator.
- **R1.3** Enforce **speed and acceleration limits** per agent (max speed, max acceleration),
  optionally panic-modulated (panic may raise desired speed up to the cap).
  - AC: no agent exceeds its max speed or max acceleration in any tick; raising panic does
    not produce speeds above the documented cap.
- **R1.4** Agents are removed from the active simulation after completing egress (handoff to FR-5); a
  removed agent exerts and receives no further forces.
  - AC: once an agent egresses it no longer appears in force computation or rendering.

### FR-2 Crowd Dynamics (core — synthesized) — P0

- **R2.1** **Inter-agent repulsion / collision avoidance**: agents exert a short-range
  repulsive force on near neighbors so they avoid overlap and steer around each other. This
  is the `f_crowd` term FR-14 composes. Overlap is strictly forbidden.
  - AC: two agents on a collision course deflect rather than overlap; mean pairwise overlap
    stays below a documented tolerance. No overlap at all.
- **R2.2** **Density pressure**: in high local density, agents experience increased resistance
  (reduced effective speed / outward pressure), producing realistic slowdown at constrictions
  rather than free flow.
  - AC: effective throughput through a constriction drops as upstream density rises (monotonic,
    unit-tested directionally).
- **R2.3** **Lane and jam formation as emergent base behavior**: counter-flow self-organizes
  into lanes, and over-capacity constrictions form jams — both arise from R2.1/R2.2 without
  special-case code.
  - AC: in a corridor with bidirectional flow, lanes are observable; at a narrow exit a jam
    forms and persists while arrival rate exceeds capacity.
- **R2.4** Neighbor queries are spatially indexed (e.g., uniform grid / spatial hash) so
  crowd-force cost scales to Tier A counts (FR ties to NFR-P2).
  - AC: per-tick crowd-force time grows sub-quadratically with agent count up to ~10k.
- **R2.5** **Herd Effect**: agents exert a attraction to the average velocity and direction of part of the crowd that is near them. When panic level increases, the attraction level increases.


### FR-3 Environment / Floor-Plan Model (core — synthesized) — P0

- **R3.1** A floor plan defines **walls**, **obstacles**, a **walkable region**, and one or
  more **exits**, in a single authored scenario format (no CAD/BIM import; see §4).
  - AC: a scenario file declares walls, obstacles, walkable space, and exits and loads into
    the domain model.
- **R3.2** Walls and obstacles are **collision boundaries**: agents cannot pass through them
  and receive a boundary-avoidance force near them.
  - AC: no agent penetrates a wall/obstacle; an agent driven toward a wall is deflected along
    it, not through it.
- **R3.3** Floor plans are **authored/loaded as bundled or user scenario files** resolved via
  `pathlib` (UTF-8), parsed into the pure domain model with no engine/render dependency.
  - AC: the same floor plan loads headless (no renderer) in a unit test and in the app.
- **R3.4** The floor-plan model is the per-level substrate the multi-level layer (FR-9)
  composes: one `Level` owns exactly one floor plan.
  - AC: a multi-level scenario (FR-9) reuses this floor-plan model unchanged per level.

### FR-4 Single-Floor Goal-Seeking & Navigation Field (core — synthesized) — P0

- **R4.1** Build an **intra-floor navigation/flow field** over the walkable region that yields,
  at any walkable point, a movement vector toward the agent's target exit, routing around walls
  and obstacles. This is the `f_exit` term and the substrate R9.5 (multi-level pathfinding)
  extends.
  - AC: from any walkable start, following the field reaches the target exit without passing
    through walls/obstacles.
- **R4.2** The field supports **multiple exits**: an agent's target selection picks an exit
  (e.g., nearest/lowest-cost by default) and the field guides it there.
  - AC: with two exits, agents distribute toward their lower-cost exit; changing an agent's
    target redirects it via the field.
- **R4.3** When a target exit becomes **blocked/unavailable** (hazard, capacity, FR-9 blocked
  transition), the field/target supports **re-routing** to an alternative exit or path.
  - AC: blocking an exit forces affected agents to detour and reach a different exit (ties to
    R9.5 re-route and FR-14 conflict handling).
- **R4.4** Field computation is decoupled from per-agent stepping so it can be precomputed and
  cached per floor and recomputed on demand when topology changes (e.g., a new block).
  - AC: a static floor computes the field once; injecting a block triggers a bounded
    recompute, not a per-agent per-tick full solve.

### FR-5 Exit Model & Evacuation-Complete Condition (core — synthesized) — P0

- **R5.1** Each exit has a **throughput capacity** (agents per second / per tick); arrivals
  beyond capacity **queue** at the exit rather than teleporting through.
  - AC: egress rate at an exit never exceeds its capacity over any 1-second window; a queue
    forms when arrival rate exceeds capacity.
- **R5.2** An agent reaching an exit and passing it a small amount (clearing it) is **removed (egressed)** from the active simulation and
  counted as evacuated (handoff from R1.4).
  - AC: evacuated count increases by exactly one per egress; egressed agents are gone from
    the sim.
- **R5.3** **Evacuation-complete** fires when all evacuable agents have egressed (or no
  further egress is possible — e.g., all remaining agents are trapped/collapsed), ending or
  flagging the run.
  - AC: a scenario with all agents able to reach an exit reaches an evacuation-complete state
    with evacuated count == initial evacuable count; a scenario with trapped agents reaches a
    terminal "no further egress" state rather than running forever.

### FR-6 Fixed-Step Simulation Loop & Seeded RNG (core — synthesized) — P0

- **R6.1** The simulation advances on a **fixed timestep** (sim tick decoupled from render
  frame rate), so behavior is frame-rate-independent and the render thread cannot stall the
  sim (ties to §7 / NFR-P2).
  - AC: the loop advances by a constant `dt` regardless of render FPS; halving FPS does not
    change sim outcomes for the same seed+events.
- **R6.2** All randomness is routed through a **single seeded RNG**; the seed is recorded in
  scenario/run metadata (foundation for NFR-R1/R2 seedable replay).
  - AC: no module instantiates its own unseeded RNG; the recorded seed reproduces the same
    initial layout and stochastic choices within a run.
- **R6.3** A **time/tick model** exposes the current sim tick and elapsed sim time to the
  application and metrics layers; runtime events (FR-12 injection, FR-15 edits) are stamped to
  sim-tick (foundation for NFR-R3 event log).
  - AC: every recorded event carries the sim-tick at which it occurred; tick monotonically
    increases by one per step.

### FR-7 Base Rendering & Interaction Loop (core — synthesized) — P0

- **R7.1** A **base render loop** draws the current floor: walkable region, walls/obstacles,
  exits, and agents (color/encoding may reflect panic/state), behind the `Renderer` port (§9)
  so the sim core stays render-agnostic. render may include background of an image (jpg, png).
  - AC: a running scenario renders the floor and live agent motion; swapping the renderer
    adapter requires no change to the domain/application layers.
- **R7.2** A **base interaction loop** maps user input (select, place, drag) to application-
  layer commands behind the `Input` port — the minimal play loop the FR-15 authoring tools
  extend.
  - AC: the player can select a position and issue a place/drag command that the application
    layer receives; FR-15 tools register as handlers on this loop without bypassing it.
- **R7.3** Rendering reads a **read-only snapshot** of sim state per frame; it never mutates
  domain state (preserves the clean-architecture boundary and lets sim run off the render
  thread).
  - AC: with rendering disabled (headless), the sim produces identical results for the same
    seed+events.

### FR-8 Core Metrics & Observability (core — synthesized) — P0

- **R8.1** Emit **core per-tick metrics**: **evacuation time / progress**, **throughput**
  (egress rate, per-exit and total), and **density** (local/areal). These feed §8 emergent
  detectors and §7 NFR-Q3 observability.
  - AC: evac progress, per-exit throughput, and a density measure are queryable per tick from
    the metrics layer.
- **R8.2** Metrics are produced by the application/metrics layer from the read-only sim
  snapshot, not by the renderer, so they are available headless for tests and learning
  metrics.
  - AC: the same metrics are obtainable in a headless run; EB-1..6 detectors (§8) consume
    them.
- **R8.3** Metrics are **structured** (machine-readable records keyed by tick) so they can be
  logged, charted in-game, and asserted in tests.
  - AC: a metrics record per tick is serializable and includes tick, evac progress,
    per-exit throughput, and density (extended by later layers with queue lengths, panic,
    collapse — NFR-Q3).

### FR-9 Multi-Level Architecture (multi-level topology layer) — P0

- **R9.1** Represent a `Level` with: `id`, `elevation_z`, `floor_plan_graph` (the FR-3
  floor-plan model),
  `stair_nodes`, `elevator_nodes`.
  - AC: a scenario with ≥3 levels loads; each level exposes its own floor-plan graph and
    node sets (including elevator nodes) via the data model.
- **R9.2** Model vertical transitions (stairs/ramps/**elevators**) as directed
  `transition_edge`s with `capacity_per_second`, `directionality ∈ {bidirectional,
  one_way}`; elevators additionally have **batch capacity** and **travel/dwell time**.
  - AC: throughput across a transition edge never exceeds `capacity_per_second` over any
    1-second window; a one-way edge rejects reverse travel; an elevator carries at most
    its batch capacity per trip and incurs travel/dwell delay.
- **R9.3** Transition edges act as **vertical bottlenecks**: when arrival rate >
  capacity, agents **queue** at the entry node (including waiting for an elevator car).
  - AC: with arrival rate > capacity, a queue forms and grows; it drains when arrival rate
    drops; elevator queues form while a car is in transit.
- **R9.4** Panic increases stair **congestion-collapse probability**.
  - AC: holding density constant, higher mean panic raises per-tick collapse probability on
    a stair edge (monotonic relationship verified by a unit test).
- **R9.5** Multi-level pathfinding = level-graph traversal + intra-floor flow field (FR-4):
  (a) select global target exit, (b) choose optimal level-transition sequence (stairs or
  elevator), (c) compute local movement vector. **Elevator-vs-stairs preference is
  scenario-tunable; the default selects whichever yields the lower estimated travel time**
  (accounting for stair traversal vs. elevator queue + travel/dwell delay per R9.2/R9.3).
  - AC: an agent on level N with an exit on level 0 produces a valid stair/elevator path
    and reaches the exit; with default weights the agent picks the lower-estimated-travel-time
    option; a scenario-level override changes the preference; if the chosen sequence becomes
    blocked, the agent re-routes (ties to R12.2 and FR-4 R4.3).

### FR-10 Emergency Signage System (directional guidance field) — P0

- **R10.1** `Signage` object: `position`, `direction_vector`, `influence_radius`,
  `strength`, `visibility_score`, `compliance_modifier`.
  - AC: a sign only influences agents within `influence_radius`; influence scales with
    `strength` and `visibility_score`.
- **R10.2** Signage modifies goal force: **f_goal = f_exit + λ · f_signage**, where
  `λ = g(compliance_level, visibility, panic_state)`. `f_exit` is the FR-4 navigation-field
  term.
  - AC: with λ=0 agents ignore signage; increasing λ measurably bends trajectories toward
    `direction_vector`.
- **R10.3** Panic affects trust in a **runtime-configurable** direction — a toggle with
  **both** modes selectable at runtime: (a) higher panic → **lower** trust, and
  (b) higher panic → **over-trust**.
  - AC: both modes are selectable from config/UI at runtime; switching the toggle produces
    the documented opposite effect on λ within the same session.
- **R10.4** Visibility degrades signage influence (distance and smoke occlusion).
  - AC: an agent inside a smoke cloud has reduced effective `visibility_score` for signs
    occluded by that smoke.
- **R10.5** Signage enables crowd redistribution, single-bottleneck prevention, and guided
  lane formation.
  - AC: see emergent behavior EB-3 (§8) — flow splitting/re-merging is observable.

### FR-11 Dynamic Hazard & Panic Source System — P0

- **R11.1** `PanicSource` types: `fire_visual`, `smoke`, `explosion_sound`,
  `gunfire_audio`, `structural_collapse`, `earthquake`. Each: `position`, `intensity`, `radius`,
  `decay_rate`.
  - AC: each type exists with these fields; `decay_rate` reduces effective intensity over
    time and a source expires near zero intensity.
- **R11.2** Agent perception model:
  **p_i = f(visual) + f(audio) + f(local_density) + f(secondary_reinforcement)**, writing the
  FR-1 per-agent panic state.
  - AC: visual contribution rises fastest; audio is non-zero **beyond line of sight**;
    higher local density amplifies p_i (each term unit-tested for direction).
- **R11.3** Maintain a **spatial panic gradient** scalar field: high near sources, decaying
  with distance and **obstacles**; **wall reflection in enclosed mode** (in scope for
  release).
  - AC: field value is highest at the source and decays with obstacle-aware distance; with
    enclosed mode enabled, field near a wall is higher than the free-field value at the same
    distance (reflection contribution observable).
- **R11.4** Agents **drift away** from gradient peaks (panic repulsion force) — the
  `f_panic_repulsion` term FR-14 composes.
  - AC: an isolated agent near a source moves down-gradient (away) absent other forces.

### FR-12 Dynamic Multi-Source Panic Injection (runtime events) — P0

- **R12.1** `AddPanicSource(type, position, intensity, radius)` injectable at runtime;
  immediately modifies the local panic field.
  - AC: calling the API mid-run updates the panic field within the same or next tick.
- **R12.2** Injection **re-triggers path recalculation** (FR-4 R4.3 / R9.5), causing crowd
  redirection waves.
  - AC: after injection near a path, affected agents recompute routes and a visible
    redirection wave propagates outward from the source.
- **R12.3** **Secondary panic cascade** (rumor propagation / wavefront):
  **p_i(t+1) = p_i(t) + γ · neighbor_panic**, producing an expanding secondary wavefront.
  *(Later-phase feature — post-release / late roadmap, not MVP, not initial release-gating.
  See §15 Phase 10.)*
  - AC: with cascade enabled and γ>0, panic spreads to neighbors not in line of sight of
    the source; with γ=0 it does not.

### FR-13 Pre-Made Environment Library (scenario templates) — P0

Each template is a tuned, loadable scenario shipped as default data (FR-0) with documented
topology, exits, bottlenecks, and agent-count range, built on the FR-3 floor-plan and (where
multi-level) FR-9 topology models.

- **R13.1 Lecture Hall** — 50–300 agents; tiered seating; front-stage exit + optional side
  exits; aisle-convergence bottlenecks; high frontal congestion.
  - AC: loads in range; aisle convergence produces observable front congestion.
- **R13.2 Stadium / Arena** — radial seating; multiple gated exits; concourse ring;
  multi-level (seating bowl / concourse / external exits); radial collapse; multi-exit
  competition; stair bottlenecks. **Tier A (full per-agent ~10k) gates release; Tier B
  (≤50k abstracted) is a stretch goal.**
  - AC: loads and plays at Tier A (~10k full per-agent) on the target laptop; exits
    compete; stairs bottleneck. Tier B abstraction (≤50k) is stretch and, if shipped,
    documents its abstraction model.
- **R13.3 Movie Theater** — 80–400 agents; long central aisle; side-wall exits; sloped
  floor; low visibility; row-based congestion; aisle locking.
  - AC: loads in range; low-visibility default applied; row egress produces aisle locking.
- **R13.4 Industrial Plant** — machinery obstacles; narrow service corridors; hazard zones;
  maze navigation; local deadlocks.
  - AC: plant loads with obstacles/corridors/hazard zones; local deadlocks are reachable and
    detectable.

### FR-14 System-Level Interaction (signage + panic + crowd) — P0

- **R14.1** Combined movement vector:
  **a_i = f_exit + f_signage + f_panic_repulsion + f_crowd**, a superset of the FR-1 core
  steering terms (`f_exit` from FR-4, `f_signage` from FR-10, `f_panic_repulsion` from FR-11,
  `f_crowd` from FR-2).
  - AC: all four terms and their individual components are present and individually toggleable for debugging; disabling a
    term removes its effect.
- **R14.2** Support conflict scenarios:
  (a) signage directing toward a **blocked** exit,
  (b) panic **overriding** signage trust,
  (c) **contradictory multi-signage** fields,
  (d) dynamic **re-weighting under stress**.
  - AC: (a) agents detour from a blocked exit despite the sign; (b) above a panic threshold,
    panic repulsion dominates signage; (c) opposing signs produce a measurable
    split/oscillation, not a crash; (d) term weights shift with mean panic per a documented
    function (ties to EB-6).

### FR-15 UI Extensions — P0 (a,b,c functional; polish P1)

These extend the FR-7 base render/interaction loop; they register on it rather than replacing
it.

- **R15.1 Panic Source Editor Panel** — add fire zone / gunshot source / smoke cloud; move
  a source in real time; intensity slider. *(Secondary-cascade trigger is a later-phase
  add, per FR-12 R12.3 / Phase 10.)*
  - AC: each action maps to the underlying API (R12.1) and updates the sim live.
- **R15.2 Signage Placement Tool** — click-to-place directional arrows; define flow
  corridors; adjust influence radius; toggle visibility / smoke occlusion; **toggle
  panic→trust mode** (R10.3).
  - AC: placed signs appear in the field (R10.1) immediately; radius/visibility/trust-mode
    edits update influence live.
- **R15.3 Multi-Level Navigation View** — floor selector tabs; stacked floor minimap;
  stair/elevator-connection overlay graph; per-level heatmap.
  - AC: tabs switch the active level; minimap shows all levels stacked; overlay shows
    transition edges (stairs + elevators); per-level heatmap renders density (default) with
    a toggle to panic.
- **R15.4 CLI** — all UI commands log INFO with their syntax, so the user can copy them as CLI commands. The app will receive CLI commands for all its UI commands..


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
the render thread (FR-6 R6.1, FR-7 R7.3).

### Performance & Scale (targets set against the laptop above)
- **NFR-P1** Interactive scenarios (Lecture Hall, Theater, Industrial Plant, ≤ ~2,000
  agents): **≥ 30 FPS** on the target laptop.
- **NFR-P2** Tier A — **up to ~10k agents at ≥ 30 FPS, full per-agent fidelity** — is the
  **release gate**. If the iGPU cannot sustain 30 FPS at 10k while rendering, the fallback
  order is: (1) instanced/batched rendering + active-level-only draw, (2) reduced render
  detail (point/sprite agents) above a count threshold, (3) lower the guaranteed-30-FPS
  count to the highest count the iGPU sustains and document it. Tier B (10k–50k via
  spatial/flow abstraction, fast-forwardable, ≥ 15 FPS) is a **stretch goal**. Depends on
  the FR-2 R2.4 spatial index and FR-6 fixed-step loop keeping sim cost bounded.
  - AC: Tier A target met and benchmarked on the target laptop.
- **NFR-P3** Runtime panic-injection re-route latency: redirection wave begins within
  **≤ 200 ms** (or ≤ N ticks) of `AddPanicSource` (FR-12 R12.1/R12.2).
- **NFR-P4** Memory budget: total app footprint **≤ 6 GB** at Tier A (leaving headroom on a
  16 GB machine where the iGPU also consumes shared RAM); document footprint if Tier B is
  attempted.

### Seedability / Replay (downgraded from determinism — nice-to-have)
- **NFR-R1** A **seedable mode**: same seed + same ordered event log replays the *same
  scenario* closely enough to **share and re-watch** (best-effort reproducibility, not
  byte-identical, not cross-machine). Determinism is **not** a release gate. Built on the
  FR-6 fixed-step loop and seeded RNG.
  - AC: replaying a saved seed+event-log reproduces the scenario recognizably (same initial
    layout, same scripted events at the same ticks).
- **NFR-R2** All randomness routed through a single seeded RNG (FR-6 R6.2); seed recorded in
  scenario metadata so scenarios are shareable.
- **NFR-R3** Runtime events (signage/hazard edits) are timestamped to sim-tick (FR-6 R6.3)
  and recorded to a replayable event log.

### Other
- **NFR-Q1 Code quality:** `flake8` clean, `mypy --strict` clean, ≥ 85% test coverage on
  core/capability modules (per project standards). All code commented, with type hints.
- **NFR-Q2 Platform:** Windows 11; `pathlib` for all paths; UTF-8 I/O; PowerShell-documented
  run steps.
- **NFR-Q3 Observability:** structured metrics (per-tick queue lengths, density, panic,
  throughput, collapse events) for in-game feedback, learning metrics, and §8 checks — an
  extension of the FR-8 core metrics.
- **NFR-Q4 Stability:** no scenario crashes or deadlocks the process; in-sim agent deadlocks
  are detected and surfaced (visual/log), not silent (ties to FR-13 R13.4, FR-5 R5.3).
- **NFR-Q5 Visuals:** modern looking, aesthetic UI visuals.


---

## 8. Emergent-Behavior Scenarios (observable, partly auto-detected)

These make play interesting and teach the dynamics. Each should be reproducible from a
saved scenario + seed + event log. Where cheap, an **automated detector** (consuming the FR-8
metrics) provides in-game feedback / learning metrics; otherwise visual observation is
acceptable (this is a game, not a validation suite). **Thresholds are defined empirically in
Phase 1** — no preset numbers.

| ID | Emergent behavior | Setup | Observable / pass signal |
|---|---|---|---|
| **EB-1** | Cross-floor congestion propagation | Multi-level; funnel upper floor to shared stair/elevator | Upper-floor queue growth causes downstream queue/density rise on the connected lower level, with a time lag. |
| **EB-2** | Stairwell collapse under panic pressure | High density + rising panic on one stair edge | A `collapse` event fires; collapse rate increases with panic at fixed density (R9.4). |
| **EB-3** | Signage-induced flow splitting & re-merging | One crowd, two exits, signs steering a fraction to exit B | Exit-B share rises with λ; lanes split then re-merge downstream. |
| **EB-4** | Panic wave diffusion across crowd clusters | Inject one source near a cluster | Panic spreads outward over time with a visible propagation speed (full non-LOS spread needs the later cascade feature, FR-12 R12.3). |
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
domain (pure):      Level, FloorPlan, TransitionEdge, Agent, Signage, PanicSource,
                    fields, forces — no rendering, no I/O, no engine imports;
                    pure NumPy + Python  (FR-1..FR-5, FR-9..FR-11, FR-14)
application:        fixed-step sim loop, seeded RNG, tick model, event log, injection
                    API, metrics  (FR-6, FR-8, FR-12, NFR-R*)
ports (interfaces): Renderer, InputSource, ScenarioRepository, Clock  (FR-7)
adapters:           concrete renderer/UI, scenario loaders (bundled default data), input
                    (FR-7, FR-15, FR-0, FR-3, FR-13)
```

The sim core depends on **nothing** from the render layer. Renderers are adapters behind a
`Renderer` port (FR-7), so the engine choice is reversible and the core is unit-testable
headless (FR-7 R7.3 / FR-8 R8.2).

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
├── domain/         # Agent, FloorPlan, Level, TransitionEdge (stairs+elevators), Signage,
│                   #   PanicSource, fields, forces — pure, no engine imports
│                   #   (FR-1..FR-5, FR-9..FR-11, FR-14)
├── pathfinding/    # single-floor navigation field (FR-4) + level-graph traversal (R9.5)
├── application/    # fixed-step loop, tick model, seeded RNG, event log, injection API,
│                   #   metrics (FR-6, FR-8, FR-12, NFR-R*)
├── scenarios/      # template library + bundled DEFAULT data (FR-0, FR-3, FR-13)
├── metrics/        # exporters + EB-1..6 detectors (FR-8, §8)
├── ports/          # Renderer / Input / ScenarioRepository / Clock interfaces (FR-7)
└── adapters/
    ├── render/     # arcade (default) renderer behind Renderer port (FR-7, FR-15, NFR-P*)
    └── io/         # scenario loaders, default-data resolution via pathlib (R0.4, R3.3)
assets/             # bundled default scenarios/parameters (FR-0)
tests/              # mirrors src/ (core is testable headless)
docs/               # this PRD, plan.md
```

Architecture is finalized in `/plan`.

---

## 10. Assumptions & Dependencies

**Assumptions**
- **A1:** No external baseline exists — the project **builds the entire product from
  scratch**, core (FR-1..FR-9) before the capability layers (FR-9..FR-15) (resolved).
- **A2 (core-synthesis):** The core (FR-1..FR-9) is **synthesized from domain-standard 2D
  crowd-evacuation modeling and the structural needs of the capability layers**, not
  transcribed from the user's original §1–10 (never supplied). It is replaceable if §1–10
  arrives (see §0 provenance note).
- **A3:** The core force model is additive (steering/social-force, FR-1 R1.2) so the combined
  vector in FR-14 R14.1 extends it cleanly.
- **A4:** Target machine is the specified Ryzen 5 Pro 7535U / integrated Radeon / 16 GB
  Windows 11 laptop (resolved, §7).
- **A5:** "Multi-level/3D" means a multi-layer navigable graph + stacked 2.5D views, not
  photoreal 3D.
- **A6:** Stadium 50k (Tier B) is a stretch goal via abstraction; ~10k full per-agent
  (Tier A) gates release.
- **A7:** Reproducibility is best-effort/shareable, not a hard determinism guarantee.

**Dependencies**
- D-1: Chosen rendering/engine (D6) — default `arcade`.
- D-2: Bundled default scenario data authored before release (FR-0, FR-3, FR-13).

---

## 11. Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| RK-1 | 10k agents can't hit 30 FPS on the iGPU (render-bound) | Misses Tier A release gate | Instanced/batched rendering, active-level-only draw, sprite/point LOD; NumPy-vectorized sim off the render thread (FR-6/FR-7); FR-2 R2.4 spatial index; profile early (Phase 1 spike); documented FPS/count fallback (NFR-P2). |
| RK-2 | Tier B 50k infeasible | Misses stretch only | Tier B is explicitly stretch; abstraction + optional Numba; not release-gating. |
| RK-3 | "Fun/playable" is subjective | MVP ships but isn't engaging | MVP gate is a concrete playable loop (FR-0 + FR-1..FR-8 core + drag-hazard-and-react); iterate on feel after MVP. |
| RK-4 | Configurable panic→trust toggle doubles behavior surface | Test/verification cost | Single runtime enum with two well-tested modes; ship a default; both covered by directional tests. |
| RK-5 | Emergent thresholds (§8) hard to pin | Confusing feedback | Define empirically in Phase 1; default to qualitative/visual signals where detectors are expensive. |
| RK-6 | Scope creep from later/optional features (cascade, Tier B) | Timeline slip | Cascade (FR-12 R12.3) and Tier B are explicitly post-release/stretch; gate behind approval; default off. |
| RK-7 | Building the 2D core from scratch underestimated | Slips everything | MVP scopes the smallest viable core (FR-1..FR-8) first; defer richness until the loop is playable. |
| RK-8 | Synthesized core (FR-1..FR-9) diverges from the user's real §1–10 if it later appears | Rework of core layer | Core FRs are flagged as synthesized/replaceable (§0, A2); they are additive-force-compatible so the capability layers survive a core swap. |

---

## 12. Open Questions

**None.** All prior questions are resolved and folded into the FRs/NFRs/architecture
(see §0 and §13). For the record, the previously-tracked items closed in earlier revisions:

- **Q-A (closed):** First-launch default scenario = **Lecture Hall** — smallest scale,
  fastest to a readable, playable loop on the iGPU. Locked into R0.1.
- **Q-B (closed):** Elevator-vs-stairs routing = **scenario-tunable, default selects lower
  estimated travel time**. Locked into R9.5.
- **D6 (closed):** Renderer = **`arcade`** (default), with `pyglet`/`moderngl` and Godot as
  documented fallbacks. Locked into §9.

(Resolved earlier: baseline existence, stadium scale gating, elevators, emergent thresholds,
panic→trust direction, wall reflection, secondary cascade timing, and target machine.)

> **One reviewable item (not a blocker):** the core requirements **FR-1..FR-9 are synthesized
> assumptions** (the original §1–10 was never supplied). They are intentionally domain-standard
> and additive-force-compatible. The user may review/replace them if the real §1–10 exists;
> absent that, `/plan` proceeds on the synthesized core.

## 13. Resolved Decisions (folded into requirements)

- **No external baseline** → build the whole product from scratch, core then layers
  (FR-1..FR-9 then FR-9..FR-15; A1).
- **Core is synthesized** → FR-1..FR-9 inferred from domain-standard modeling + the layers'
  structural needs; flagged replaceable (§0; A2; RK-8).
- **Out-of-the-box defaults** → first-class FR (FR-0).
- **Stadium scale** → Tier A (~10k full) gates release; Tier B (≤50k abstracted) is stretch
  (R13.2, NFR-P2).
- **Elevators** → in scope for release (FR-9; R9.1–R9.3, R9.5; R15.3).
- **Emergent thresholds** → defined empirically in Phase 1 (§8).
- **Panic→signage-trust** → runtime toggle, both "lower trust" and "over-trust" selectable
  (R10.3; R15.2).
- **Wall reflection (enclosed mode)** → in scope for release (R11.3).
- **Secondary panic cascade (rumor wavefront)** → later phase, post-release (FR-12 R12.3;
  §15 Phase 10).
- **Target machine** → Ryzen 5 Pro 7535U / integrated Radeon / 16 GB / Win 11; targets set
  against it; integrated-GPU rendering constraint called out (§7).
- **Tech stack (D6)** → clean architecture, engine-agnostic core, avoid pygame; **chosen:**
  Python+NumPy(+optional Numba) core with an **`arcade`** renderer; `pyglet`/`moderngl` and a
  Godot front-end retained as documented fallbacks (§9). Accepted trade-offs: 2.5D faked via
  layering (not true 3D); Tier B 50k stretch may require Numba acceleration.
- **First-launch default scenario (Q-A)** → **Lecture Hall** (R0.1, R13.1).
- **Elevator-vs-stairs routing (Q-B)** → scenario-tunable, default = lower estimated travel
  time (R9.5).

**No open or remaining decisions.** All forks are resolved and folded into the requirements
above; `/plan` can proceed without further input (the synthesized-core flag in §12 is a
review opportunity, not a blocking decision).

---

## 14. Success Metrics

1. **Believable core:** the 2D core (FR-1..FR-8) stands alone — agents steer to exits, crowds
   jam at constrictions, a scenario evacuates to completion under a fixed seeded loop.
2. **Playable MVP:** clean install launches into a default scenario; the core loop (place/
   drag hazard → crowd reacts/re-routes → evacuates) works and reads clearly (FR-0,
   FR-1..FR-8, FR-11/12, FR-14).
3. **Functional completeness:** 100% of P0 requirements pass AC.
4. **Emergent behaviors:** 6/6 §8 scenarios reproducible and observable; cheap ones
   auto-detected.
5. **Performance:** Tier A ~10k ≥ 30 FPS and interactive scenarios ≥ 30 FPS on the target
   laptop; injection re-route ≤ 200 ms (NFR-P1/P2/P3); footprint ≤ 6 GB (NFR-P4).
6. **Quality:** `mypy --strict` + `flake8` clean; ≥ 85% coverage on core/capability modules.
7. **Library:** all 4 templates load and evacuate to completion.

---

## 15. Phased Delivery Roadmap

Phase 1 is an explicit **playable MVP**: the smallest set of features (the from-scratch 2D
core plus a single live hazard) that delivers a valuable, playable loop. Later phases layer
fidelity, scale, and the post-release cascade. Each phase is independently completable and
ends in a checkpoint.

| # | Phase | Objective | Depends on | Priority |
|---|---|---|---|---|
| 1 | **MVP — playable loop** | From-scratch 2D core — agent model & movement, crowd dynamics, floor-plan model, single-floor navigation field, exit/egress, fixed-step seeded loop, base render/interaction loop, core metrics (**FR-1..FR-8**) + zero-setup default scenario (FR-0) + one panic source the player can drop/drag with crowd flee + re-route (subset of FR-11/12.1/14) + minimal `arcade` render/UI. **Pin stack (D6).** Spike iGPU rendering at scale (RK-1). Set §8 thresholds empirically here. | D6 | P0 |
| 2 | Multi-level topology | Levels, transition edges (stairs **and elevators**: capacity, directionality, batch/dwell), queuing, multi-level pathfinding (FR-9) | 1 | P0 |
| 3 | Panic field & hazards | Full PanicSource types, perception model, spatial gradient with **wall reflection (enclosed mode)**, drift-away force (FR-11) | 1 | P0 |
| 4 | Signage field | Signage objects, f_goal blend with λ, visibility/occlusion, **runtime panic→trust toggle** (FR-10, R10.3) | 1 | P0 |
| 5 | Interaction model | Combine a_i terms; conflict handling + stress re-weighting (FR-14) | 2,3,4 | P0 |
| 6 | Runtime injection | AddPanicSource live; re-route waves (FR-12 R12.1/R12.2) | 3,5 | P0 |
| 7 | Scenario library | Lecture Hall, Movie Theater, Industrial Plant, Stadium **Tier A** (~10k); all shipped as bundled default data (FR-13, FR-0) | 2,5 | P0 |
| 8 | UI extensions | Panic editor, signage tool (incl. trust toggle), multi-level navigation view (stairs+elevators overlay) (FR-15) | 5,6,7 | P0 |
| 9 | Emergent behaviors | EB-1..6 detectors + empirically-set thresholds; observable acceptance run (§8) | all | P0 |
| 10 | Post-release / stretch | **Secondary panic cascade / rumor wavefront (FR-12 R12.3)**, **Stadium Tier B (≤50k abstracted)** | as needed | P1/P2 |

**Critical path:** 1 (MVP core) → 2 → (3,4 parallel) → 5 → 6 → 7 → 8 → 9. Cascade and Tier B
(Phase 10) follow release.

**Model assignments (per project planning rule):** Phase 1 core/spike — Opus (architectural
fork + scale spike); Phases 2–4 — Sonnet; Phase 5/9 — Opus (interactions / cross-system);
Phases 6–8 — Sonnet; scaffolding/defaults — Haiku.

Phase-level detail (steps, files, success criteria) is produced by `/plan` into
`docs/plan.md` after approval.

---

## 16. Next Step

On approval, run `/plan` to break FR-0/FR-1..FR-15 and EB-1..6 into atomic, phase-sized
tasks with architecture detail in `docs/plan.md`.

[YOU DO] Review. All decisions (D6 renderer, Q-A default scenario, Q-B routing) are resolved
and locked — no open items remain. The only reviewable (non-blocking) item is the
**synthesized core (FR-1..FR-9)**, which you may replace if you supply the original §1–10.
Reply with "Approved — proceed to /plan" or any change requests.
