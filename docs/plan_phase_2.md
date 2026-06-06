# Implementation Plan: Phase 2 — Behavioural-Weight Optimisation (crowd_evac)

## Context

Phase 1 delivered the playable 2D core: vectorised SoA agents, five additive
force terms (`f_exit`, `f_crowd`, `f_density`, `f_herd`, `f_panic_repulsion`),
a fixed-step seeded loop, exit/egress, headless metrics, and an `arcade`
render/UI loop. Every behavioural weight that shapes the crowd is currently a
**hard-coded module constant** in `src/crowd_evac/domain/constants.py`
(`REPULSION_STRENGTH`, `HERD_ATTRACTION_STRENGTH`, `PANIC_REPULSION_STRENGTH`,
`DENSITY_PRESSURE_STRENGTH`, `RELAXATION_TIME`, the radii, the speed caps, …).
They were set by hand in Phase 1 step 1.21.

**Phase 2 = find the best weight set automatically** (PRD §15 Phase 2). "Best"
has two meanings the user fixed, and they **conflict**:

1. **Realism** — emergent crowd statistics match real evacuations (fundamental
   diagram, bottleneck specific flow, free-walking speed, EB-1..6 fidelity),
   plus a hard realism constraint the user added: **no agent gets stuck while a
   viable route to a live exit exists**.
2. **Minimal evacuation time** — the crowd clears as fast as possible.

The conflict is structural: the terms that *create* realism (`f_crowd`,
`f_herd`, density pressure, panic) are exactly the ones that *cost* time.
Pure time-minimisation degenerates to frictionless, zero-panic particles — the
global time optimum and maximally unrealistic. Therefore **realism anchors the
search; evacuation time is the lever traded against it, not co-maximised
freely.** The one place the two objectives *agree* is the stuck-agent
constraint: a frozen-but-unblocked agent is both unrealistic and a direct time
penalty, so it is treated as a near-hard constraint and doubles as a **bug
canary** (see §Locked Decisions).

Outcome of Phase 2: a single calibrated weight set, validated at Tier A scale
across the scenario suite, shipped as the new R0.3 defaults, with the
calibration recorded in `docs/` and the §8 EB thresholds set from the same
empirical reference data.

## Locked Decisions (this planning session)

- **Weights become an injectable value object, not module constants.** A frozen
  `ForceParams` dataclass (defaults == today's constants) flows
  scenario → `Simulation` → `forces.compose` → each force term. The optimiser
  varies *one `ForceParams` per evaluation*; nothing in the hot path reads
  module globals for a tunable. This is the prerequisite for every later step
  and is a behaviour-preserving refactor (its own commit, per refactoring rule).
- **Optimiser = NSGA-II (multi-objective GA, via `pymoo`), producing a
  realism↔time Pareto front; the shipped default is then *selected* from that
  front by the realism-gated rule** (minimum evac time subject to
  realism_distance ≤ threshold **and** stuck_count == 0). This unifies both
  framings the user weighed: NSGA-II yields the trade-off curve; the
  realism-gated selection picks the one default to ship. NSGA-II is a genetic
  algorithm, so it honours the PRD §15 "use a GA" decision while fitting the
  two-conflicting-objective structure that vanilla single-objective GA does not.
  - **Documented fallback:** single-objective **CMA-ES** (`cma`) on a
    realism-penalised scalar score, used only if NSGA-II proves too
    sample-hungry for the per-evaluation cost measured in Step 2.2/2.7.
  - **Rejected:** RL (PRD Non-Goal #7); Nelder-Mead/Powell (stall in local
    optima, noise-intolerant); vanilla single-objective GA (least
    sample-efficient credible option).
- **Evaluations are headless, multi-seed, multi-scenario, and down-scaled.**
  Each candidate is scored as the mean over **K seeds × M scenarios** at a
  reduced agent count; the winner is re-validated at full Tier A scale (Step
  2.9). Common random numbers across candidates cut comparison variance. This
  guards against overfitting weights to one floor plan or one seed.
  **Parallelization requirement:** The evaluation harness (Step 2.2) and
  fitness composite (Step 2.6) must parallelize across available hardware
  (CPU cores, GPU if available) to minimize wall-clock cost of multi-seed ×
  multi-scenario runs; NSGA-II population evaluation (Step 2.8) is distributed
  across all available logical cores. No single evaluation blocks the others.
- **The optimiser *run* executes offline (background job), not inside a
  session.** Each Phase-2 step below is session-sized (build + unit-test +
  launch/analyse). The multi-hour NSGA-II search itself runs detached and is
  analysed in the following session. No step depends on watching a run finish.
- **Per-step exit bar is run by the user.** Each step ends by running the
  **Exit-Bar Prompt** (§Exit-Bar Prompt below) in PowerShell and pasting the
  output back. No step advances on a red checkpoint. Commits drafted on request,
  run by the user.

## Decision variables (the weight vector)

The continuous parameters the optimiser searches, all currently in
`domain/constants.py` (~13 dims — comfortably in CMA-ES / NSGA-II's sweet spot).
Bounds below are **deliberately wide** for the abstract behavioural weights —
the search, not our Phase-1 hand-tuned priors, decides their values; Step 2.7
tightens around promising regions before the expensive run.

| Weight | Constant | Term | Search range | Class |
|---|---|---|---|---|
| Exit relaxation time | `RELAXATION_TIME` | `f_exit` | 0.1 – 2.0 s | behavioural (wide) |
| Panic speed boost | `PANIC_SPEED_MULTIPLIER` | `f_exit` | 1.0 – 2.5 | behavioural (wide) |
| Repulsion strength | `REPULSION_STRENGTH` | `f_crowd` | 0.0 – 12.0 | gain (wide) |
| Repulsion radius | `REPULSION_RADIUS` | `f_crowd` | 0.2 – 2.5 m | radius (wide) |
| Density threshold | `HIGH_DENSITY_THRESHOLD` | `f_density` | 0.5 – 10.0 /m² | behavioural (wide) |
| Density pressure | `DENSITY_PRESSURE_STRENGTH` | `f_density` | 0.0 – 5.0 | gain (wide) |
| Density sensing radius | `DENSITY_SENSING_RADIUS` | `f_density` | 0.5 – 5.0 m | radius (wide) |
| Herd strength | `HERD_ATTRACTION_STRENGTH` | `f_herd` | 0.0 – 3.0 | gain (wide) |
| Herd radius | `HERD_PERCEPTION_RADIUS` | `f_herd` | 1.0 – 20.0 m | radius (wide) |
| Panic repulsion | `PANIC_REPULSION_STRENGTH` | `f_panic_repulsion` | 0.0 – 12.0 | gain (wide) |
| Max accel | `MAX_ACCEL` | integrator | 0.5 – 6.0 m/s² | responsiveness (wide) |
| Max speed | `MAX_SPEED` | `f_exit` | 1.2 – 1.6 m/s | **physical (narrow)** |
| Hazard avoidance cost | `HAZARD_AVOIDANCE_COST` | flow-field solve | 0.0 – 100.0 | navigation (wide) |

**Hazard avoidance cost** is a navigation weight (it scales a graded danger
cost in the flow-field solve so the crowd routes around a hazard to the
next-best exit), not an additive force term. Its default is deliberately
overpowering. The hazard-free search suite (Step 2.3) does not exercise it, so
the Step 2.7 sensitivity pre-pass will read it as non-influential and fix it at
its default; it is searchable only on hazard scenarios.

**Narrow / fixed by design:** `MAX_SPEED` stays narrow — free-walking speed is
an empirical physical quantity (~1.34 m/s), and it is also a realism *target*
in Step 2.4, so it is bounded tightly rather than left to drift. Excluded from
the search entirely (fixed physical / scenario properties): `DT`,
`AGENT_RADIUS`, `GRID_CELL_SIZE`, `EXIT_CAPACITY_PER_SECOND`.

## Architecture (new package under the Phase-1 spine)

```
src/crowd_evac/
├── domain/
│   └── params.py            # ForceParams frozen dataclass (NEW, Step 2.1)
├── optimization/            # NEW package — calibration, pure of render
│   ├── harness.py           # headless evaluate(weights, scenario, seed) (2.2)
│   ├── suite.py             # search scenario set + down-scaling + micro-rigs (2.3)
│   ├── realism.py           # empirical reference targets + realism distance (2.4)
│   ├── stuck.py             # stuck-agent detector / constraint / canary (2.5)
│   ├── fitness.py           # composite multi-objective + constraint vector (2.6)
│   ├── space.py             # parameter bounds + sensitivity pre-pass (2.7)
│   ├── nsga.py              # pymoo NSGA-II problem + driver + checkpoint (2.8)
│   └── select.py            # Pareto-front selection (realism-gated rule) (2.9)
scripts/
│   ├── bench_sim_headless.py  # (exists; reused for eval timing)
│   └── optimize_weights.py    # launch / resume an offline search run (2.8)
tests/optimization/          # mirrors the package — all headless
docs/                        # this plan; calibration report appended (2.10)
artifacts/calibration/       # Pareto fronts, checkpoints, chosen weights (gitignored)
```

`optimization/` depends on `application` + `domain` only; **no `arcade`
import**, so the whole calibration stack is unit-testable headless and runs on a
CI-style budget.

**Dependency note (governance — vet before install):**
`pymoo` (NSGA-II, Step 2.8), Sobol/LHS sampling via `scipy.stats.qmc`
(preferred — avoids a new dependency) *or* `SALib` (Step 2.7), optional `cma`
(fallback optimiser). Each is vetted (Windows wheel, maintenance < 12 mo,
license, size) in the step that introduces it, per `dependency-governance`.

## Step Overview

| # | Step | Layer | Depends on | Model |
|---|------|-------|-----------|-------|
| 2.1 | `ForceParams` value object; thread through compose + Simulation | domain/app | — | Sonnet |
| 2.2 | Headless evaluation harness (run a weight set → RunResult) | optimization | 2.1 | Sonnet |
| 2.3 | Search scenario suite + down-scaling + bottleneck micro-rig | optimization | 2.2 | Sonnet |
| 2.4 | Realism metric: reference targets + fundamental-diagram distance | optimization | 2.2,2.3 | **Opus** |
| 2.5 | Stuck-agent detector (constraint + bug canary) | optimization | 2.2 | **Opus** |
| 2.6 | Composite fitness: 2 objectives + constraint, multi-seed/scenario | optimization | 2.4,2.5 | **Opus** |
| 2.7 | Parameter bounds + Sobol/LHS sensitivity pre-pass | optimization | 2.6 | Sonnet |
| 2.8 | NSGA-II driver (pymoo), parallel population, checkpoint, launch | optimization/scripts | 2.6,2.7 | **Opus** |
| 2.9 | Pareto selection (realism-gated) + full-scale validation | optimization | 2.8 | **Opus** |
| 2.10 | Ship weights as R0.3 defaults; set §8 thresholds; document | domain/scenarios/docs | 2.9 | Sonnet |

**Critical path:** 2.1 → 2.2 → 2.3 → {2.4, 2.5} → 2.6 → 2.7 → 2.8 → 2.9 → 2.10.
Steps 2.4 and 2.5 are independent of each other and may run in either order.

---

## Exit-Bar Prompt (run at the end of every step)

After each step's code is written, **run this in PowerShell from the repo root
with `.venv` activated, then paste the full output back to Claude.** A step
advances only when all three are clean (`flake8=0 mypy=0 pytest=0`).

```powershell
# Phase-2 step exit bar — paste the full output (including the RESULT line) back.
Write-Host "== flake8 =="; flake8 src/; $flake = $LASTEXITCODE
Write-Host "== mypy ==";   mypy src/ --strict; $mypyc = $LASTEXITCODE
Write-Host "== pytest =="; pytest tests/ -v --ignore=tests/crowd_evac/performance; $pytestc = $LASTEXITCODE
Write-Host "RESULT flake8=$flake mypy=$mypyc pytest=$pytestc"
```

If any value is non-zero, paste the failing section and Claude fixes it before
the step closes. (Per the Windows rule, all commands are PowerShell;
`$LASTEXITCODE` carries each tool's exit code so the RESULT line summarises
pass/fail at a glance.)

---

## Step Detail

> Each step ends by running the **Exit-Bar Prompt** above and reporting the
> output. Three test cases minimum per public symbol (happy / edge / failure).
> All work is headless — no renderer.

### 2.1 `ForceParams` value object — Sonnet

- **Objective:** make every behavioural weight injectable so the optimiser can
  vary it per run, with zero behaviour change at the default.
- **Do:** add `domain/params.py` — a frozen `@dataclass` `ForceParams` with one
  field per decision variable above, a `ForceParams.defaults()` classmethod
  returning today's constant values, and `to_array()` / `from_array()` helpers
  (ordered vector ⇄ object) for the optimiser. Thread it through:
  `forces.compose(..., params: ForceParams)` passing each field to the term
  kwargs (the force functions already accept strength/radius kwargs);
  `Simulation.__init__(..., params: ForceParams = ForceParams.defaults())`
  storing it and passing it to `compose`. `constants.py` values stay as the
  source of `defaults()`.
- **Files:** `domain/params.py` (new); edit `domain/forces.py`,
  `application/simulation.py`; `tests/optimization/test_params.py`,
  update affected force/sim tests.
- **Success:** with `ForceParams.defaults()`, a fixed-seed run produces a
  **bit-identical** tick sequence to pre-refactor (regression-locked);
  `to_array`/`from_array` round-trip; an out-of-range field raises.
- **Note:** behaviour-preserving refactor → its own commit (refactoring rule).

### 2.2 Headless evaluation harness — Sonnet

- **Objective:** one call runs a weight set on a scenario to completion and
  returns the raw signals every metric needs; support parallel batch evaluation
  across CPU cores and GPU where available.
- **Do:** `optimization/harness.py::evaluate(params, scenario, seed,
  max_ticks) -> RunResult`. Builds the `Simulation` from a loaded scenario +
  `params` + seeded RNG, steps until `is_complete` or `max_ticks`, collecting
  per-tick `MetricRecord`s plus the per-tick velocity/position/panic arrays
  needed for the fundamental diagram (Step 2.4) and stuck detection (Step 2.5).
  `RunResult` (frozen dataclass / TypedDict): evac_time (s to last egress or
  terminal), evacuated_fraction, throughput series, density series, and a
  compact sampled state history. Hard wall-clock/tick cap so a pathological
  weight set cannot hang the search. Expose a `evaluate_batch(candidates,
  scenario, seeds) -> list[RunResult]` variant that parallelizes across
  available cores (thread pool or multiprocessing) — no Python GIL assumptions
  in the evaluator design.
- **Files:** `optimization/harness.py`; `tests/optimization/test_harness.py`.
- **Success:** the bundled Lecture Hall evaluates to completion headless under
  default params; same seed → identical `RunResult`; a deliberately broken param
  set hits the cap and returns a terminal result rather than hanging;
  `evaluate_batch()` with N candidates and M seeds runs wall-clock time
  ≤ 1 serial equivalent run × (N×M) / (available CPU cores).

### 2.3 Search scenario suite + down-scaling + bottleneck micro-rig — Sonnet

- **Objective:** define the *small, fast* scenario set scored during search, and
  a controlled rig for measuring physical constants.
- **Do:** `optimization/suite.py` — a curated list of search scenarios
  (down-scaled Lecture Hall + at least one contrasting floor plan) with reduced
  agent counts for cheap evaluation; plus a synthetic **bottleneck corridor
  micro-rig** (a fixed-width door with a feeding crowd) used by Step 2.4 to
  measure specific flow (persons / m·s) and the speed–density relation under
  controlled geometry. Expose `search_suite()` and `calibration_rigs()`.
- **Files:** `optimization/suite.py`, any micro-rig scenario JSON under
  `assets/scenarios/calibration/`; `tests/optimization/test_suite.py`.
- **Success:** every suite scenario loads and evaluates headless via Step 2.2;
  the bottleneck rig produces a measurable steady-state flow; down-scaled eval
  is markedly faster than full scale (timed, lenient bound).

### 2.4 Realism metric — **Opus**

- **Objective:** turn "similar to real evacuation" into a scalar distance — the
  conceptually hard, judgement-heavy core of Phase 2.
- **Do:** `optimization/realism.py` — encode empirical reference targets from
  the pedestrian-dynamics literature as named constants with citations:
  free-walking speed ≈ 1.2–1.4 m/s; bottleneck specific flow ≈ 1.2–1.3
  persons/m·s; the Weidmann speed–density fundamental-diagram shape. Implement
  extractors that compute each statistic from a `RunResult` / the bottleneck
  rig, and a `realism_distance(results) -> float` that returns a weighted,
  normalised distance to the reference set, plus an `EB`-fidelity component
  (do the §8 emergent signatures trigger: exit arching, lane formation,
  faster-is-slower, jam persistence). Document each weight and why.
- **Files:** `optimization/realism.py`; `tests/optimization/test_realism.py`.
- **Success:** distance is 0 (within tolerance) for a synthetic run matching
  the targets and rises monotonically as a controlled statistic is detuned away
  from its reference (directional tests per component); reference constants are
  unit-checked in range and carry source comments.
- **Risk flag:** this metric *is* the calibration — a wrong reference set
  silently yields a confidently-wrong weight set. Escalate the reference-weight
  choices for review before Step 2.8 commits compute to them.

### 2.5 Stuck-agent detector — **Opus**

- **Objective:** implement the user's added realism constraint — *no agent
  stuck while a viable exit route exists* — and use it as a structural-bug
  canary.
- **Do:** `optimization/stuck.py::stuck_count(history, flow_field, ...) -> int`.
  An agent is **stuck** when, over a time window, its speed stays ≈ 0 **and**
  the flow field offers a non-zero descent direction to a live exit **and** no
  wall/obstacle and no agent sits within repulsion range in that direction
  (i.e. it is *not* legitimately queued or blocked — it is deadlocked, the
  classic equal-and-opposite `f_exit` vs `f_crowd`/`f_panic_repulsion`
  cancellation). Return the count; expose a boolean constraint
  `has_stuck(...)`.
- **Files:** `optimization/stuck.py`; `tests/optimization/test_stuck.py`.
- **Success:** a hand-built deadlock fixture reports stuck > 0; a legitimately
  queued agent at a saturated exit reports stuck == 0 (no false positive); a
  freely moving agent reports 0.
- **Canary note:** this directly probes the two open Phase-1 §1.19a bugs
  ("agents don't move though not blocked", "stay in queue when another exit is
  free"). If *no* weight set in Step 2.9 can drive stuck → 0, the fault is in
  routing/flow-field logic (FR-4 R4.2/R4.3), **not** the weights — surface that
  as a Phase-1 bug, do not paper over it by detuning.

### 2.6 Composite fitness — **Opus**

- **Objective:** assemble the optimiser-facing objective/constraint vector with
  proper noise and generalisation handling; parallelize evaluation across all
  CPU cores and GPU resources.
- **Do:** `optimization/fitness.py::evaluate_fitness(params) -> FitnessResult`.
  For each of K seeds × M suite scenarios (Step 2.3) call the harness via the
  batch evaluator (Step 2.2), compute realism distance (2.4) and evac time,
  aggregate (mean + a robustness quantile across seeds), and compute the
  stuck-agent constraint (2.5). Return the **two objectives** `(realism_distance,
  evac_time)` and the **constraint** `stuck_count ≤ 0` in the form pymoo expects
  (objectives to minimise, constraints as `g(x) ≤ 0`). Use common random numbers
  across candidate evaluations. Parallelize all K×M harness runs via the batch
  evaluator; make K, M, scale, and parallelism level configurable (cheap during
  search, rich during validation, max-core utilization for production runs).
- **Files:** `optimization/fitness.py`; `tests/optimization/test_fitness.py`.
- **Success:** returns a 2-objective + 1-constraint vector of correct shape and
  sign; identical params + seeds → identical result; raising K reduces objective
  variance (directional); a stuck-prone param set reports a violated constraint;
  parallel wall-clock time for K×M evaluations is ≤ (K×M / cpu_count) ×
  single-run time (measured and logged).

### 2.7 Parameter bounds + sensitivity pre-pass — Sonnet

- **Objective:** finalise search bounds and prune non-influential weights so the
  expensive optimiser spends its budget where it matters.
- **Do:** `optimization/space.py` — declare per-weight bounds (the wide table
  above, refined), a `to_array`/`from_array` ordering matching Step 2.1, and a
  Sobol/LHS sampling pre-pass (prefer `scipy.stats.qmc` to avoid a new
  dependency; else vet `SALib`) that evaluates ~N samples via Step 2.6 and ranks
  each weight's influence on both objectives. Fix weights with negligible
  influence at their default and tighten ranges around promising regions.
- **Files:** `optimization/space.py`; `scripts/sensitivity.py`;
  `tests/optimization/test_space.py`.
- **Success:** bounds validate (low < high, defaults inside); the pre-pass runs
  headless on a small budget and emits a ranked influence table; at least the
  obviously-dominant weights (`REPULSION_STRENGTH`, `RELAXATION_TIME`,
  `PANIC_REPULSION_STRENGTH`) surface as influential (sanity check).

### 2.8 NSGA-II driver — **Opus**

- **Objective:** wire and launch the multi-objective search with maximal
  hardware utilization (CPU and GPU).
- **Do:** vet + add `pymoo`. `optimization/nsga.py` — a pymoo `Problem`
  wrapping `evaluate_fitness` (2 objectives, 1 constraint, bounds from 2.7),
  population evaluation **fully parallelised across all available CPU cores and
  GPU resources** (detect device, dispatch via `evaluate_batch()` in Step 2.6),
  and checkpoint/resume to `artifacts/calibration/` (runs are multi-hour).
  `scripts/optimize_weights.py` launches or resumes a run with configurable
  worker count and device affinity, writes the Pareto front, and logs per-generation
  wall-clock time + speedup ratio. The run is started **as a background/offline
  job**, not awaited in the session.
- **Files:** `optimization/nsga.py`, `scripts/optimize_weights.py`;
  `tests/optimization/test_nsga.py`; `pyproject.toml` (pymoo pinned).
- **Success:** on a tiny budget (small pop, few generations, mocked/cheap
  fitness) the driver produces a valid non-dominated set and a resumable
  checkpoint; constraint-violating individuals are dominated out; the real
  launch command starts and detaches cleanly; wall-clock speedup for parallel
  evaluation is ≥ 0.8 × (available cores) vs. serial baseline (logged per generation).
- **Fallback:** if per-evaluation cost (measured 2.2/2.6) makes a useful NSGA-II
  population infeasible in available wall-clock even with full parallelization,
  switch to the documented CMA-ES realism-penalised single-objective path (`cma`)
  behind the same fitness, which has lower population overhead.

### 2.9 Pareto selection + full-scale validation — **Opus**

- **Objective:** pick the one weight set to ship and prove it holds at Tier A. Should be ready for future optimizations.
- **Do:** `optimization/select.py::choose(front) -> ForceParams` — apply the
  **realism-gated rule**: among non-dominated points with realism_distance ≤ the
  Step-2.4 threshold **and** stuck_count == 0, pick minimum evac_time (knee as a
  tie-break/secondary report). Re-validate the winner at **full Tier A agent
  count** across the full (non-down-scaled) suite and extra held-out seeds; if it
  regresses (realism or stuck-count) at scale, record it and step back to the
  next front point. Also leave a working procedure for future running of the optimization again, by the user.
- **Files:** `optimization/select.py`; `scripts/validate_weights.py`;
  `tests/optimization/test_select.py`.
- **Success:** selection returns a single `ForceParams`; on a synthetic front it
  honours the gate (never returns a stuck-violating or over-distance point); the
  chosen set validates at Tier A with stuck_count == 0 and realism within
  threshold; results written to `artifacts/calibration/`.

### 2.10 Ship defaults, set §8 thresholds, document — Sonnet

- **Objective:** make the calibrated weights the product defaults and close the
  loop with the PRD.
- **Do:** write the chosen weight set into `ForceParams.defaults()` / the
  scenario default-parameter block (R0.3) — keep the old values in git history,
  no commented-out code. Set the §8 EB-1..6 empirical thresholds from the same
  reference statistics gathered in Step 2.4 (PRD §8 — "defined empirically").
  Append a **calibration report** to `docs/` (method, reference targets, Pareto
  front summary, chosen point, validation numbers) and update PRD §15 Phase 2 /
  R0.3 to point at it.
- **Files:** `domain/params.py` / scenario defaults, `domain/constants.py` (§8
  thresholds), `docs/plan_phase_2.md` (this file — results section),
  `docs/PRD.md`; tests asserting defaults load and thresholds are in range.
- **Success:** a clean run uses the calibrated defaults; §8 thresholds load and
  are unit-checked; the report documents the full calibration; the Exit-Bar
  Prompt is clean; coverage ≥ 85% on the `optimization/` package.

---

## Verification (commands the user runs and reports)

```powershell
# from project root, .venv activated

# step exit bar — after every step (see §Exit-Bar Prompt for the full block)
flake8 src/; mypy src/ --strict; pytest tests/ -v

# coverage gate (Step 2.10) — optimization package
pytest tests/optimization/ --cov=src/crowd_evac/optimization --cov-branch --cov-report=term-missing

# sensitivity pre-pass (Step 2.7) — small budget, headless
python scripts/sensitivity.py --samples 256

# launch the offline NSGA-II search (Step 2.8) — runs detached for hours
python scripts/optimize_weights.py --pop 64 --gens 80 --seeds 5 --out artifacts/calibration

# select + validate the winner at full scale (Step 2.9)
python scripts/validate_weights.py --front artifacts/calibration/front.json --tier-a
```

**Phase-2 acceptance gate:**
1. Calibrated `ForceParams` selected from a real Pareto front via the
   realism-gated rule, with stuck_count == 0 at Tier A.
2. Realism distance within the documented threshold across the full suite at
   full scale.
3. Calibrated weights shipped as R0.3 defaults; §8 EB thresholds set from the
   same reference data; calibration report in `docs/`.
4. All quality gates green; `optimization/` coverage ≥ 85%.

## Risks & Notes

- **RK-2.1 (realism metric is the real risk):** a wrong reference target set
  yields a confidently-wrong default. Mitigation: cite every reference constant,
  escalate the metric weighting for review before Step 2.8 (Step 2.4 risk flag).
- **RK-2.2 (evaluation cost):** NSGA-II needs hundreds–thousands of runs. If
  per-eval cost is too high even down-scaled, fall back to CMA-ES (fewer evals)
  and/or shrink K/M/scale during search, re-validating at scale only in 2.9.
- **RK-2.3 (overfitting to one scenario/seed):** mitigated by the multi-scenario
  suite + multi-seed averaging + held-out validation seeds in 2.9.
- **RK-2.4 (stuck-agent is a code bug, not a weight):** if Step 2.9 can't reach
  stuck_count == 0 on any front point, the fault is in flow-field routing
  (FR-4) — fix it in Phase 1, do not mask it by detuning. The detector exists
  precisely to make this visible.
- **RK-2.5 (objectives conflict):** by design — the Pareto front *is* the
  trade-off; the realism gate, not raw time-minimisation, selects the default.
- **Dependencies:** `pymoo` (2.8), Sobol via `scipy.stats.qmc` preferred over
  new `SALib` (2.7), optional `cma` fallback (2.8) — each vetted in its step.

## Results (filled during execution)

- Step 2.7 sensitivity ranking: _TBD_
- Step 2.8 Pareto front summary (pop/gens/wall-clock): _TBD_
- Step 2.9 chosen weight set + Tier-A validation (realism distance, evac time,
  stuck_count): _TBD_
- Step 2.10 §8 EB-1..6 thresholds set from reference data: _TBD_
```
