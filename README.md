# crowd_evac

Real-time 2D crowd-evacuation game with a pure NumPy simulation core.

## Requirements

- **Python 3.11–3.13** — developed and tested on **3.12.10**. Python 3.14 is
  **not supported**: the `arcade` renderer pins `pymunk~=6.9.0`, which has no
  3.14 wheel and will not compile without MSVC build tools.
- Windows 11 with pip and venv

## Quick Start

```powershell
# Verify a supported interpreter is available (3.11-3.13; 3.14 is unsupported)
py -3.12 --version  # Should print Python 3.12.x

# Clone and navigate to project
cd crowd_evac

# Create and activate virtual environment on Python 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies from requirements.txt
pip install -r requirements.txt

# OR install project in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the game (Lecture Hall, 3 agents, seed 42)
python -m crowd_evac

# List the available rooms
python -m crowd_evac --list

# Run a room with a chosen population (short form: -s)
python -m crowd_evac --scenario bottleneck_corridor --agents 60 --seed 7

# Open the room designer
python -m crowd_evac --design

# Run tests
pytest tests/ -v

# Run linter and type checker
flake8 src/
mypy src/ --strict
```

## Running a Specific Scenario

Scenarios are the JSON files in `assets/scenarios/`; the name is the file name
without `.json`. A scenario file describes the **room only** (`schema_version`
`"2.0"`, `name`, `floor_plan`). Any file placed there (including rooms saved
from the room designer) shows up in `--list` and can be run by name.

The population and hazards are run-time input:

| Flag | Meaning | Default |
|---|---|---|
| `--agents N` | number of agents to spawn (N >= 1) | 3 |
| `--seed S` | spawn RNG seed (S >= 0) | 42 |
| `--hazard K=V,...` | scripted hazard, repeatable | none |

`--hazard` takes comma-separated `key=value` pairs. `x` and `y` (metres) are
required; optional keys are `tick` (default 0), `intensity`, `radius`,
`decay_rate`, `block_radius`, `blocks_navigation` (`true`/`false`) and
`source_type`. Unset keys use the defaults of a mouse-placed source. The same
seed, agent count and hazards reproduce the same run, and Space-reset replays
it. A hazard outside the room, an unknown or repeated key, or a bad value is a
usage error. `--agents`, `--seed` and `--hazard` cannot be combined with
`--design` or `--list`.

Commands for each bundled room:

| Room | Command |
|---|---|
| `lecture_hall` | `python -m crowd_evac -s lecture_hall --agents 150 --seed 42` |
| `lecture_hall_no_aisle` | `python -m crowd_evac -s lecture_hall_no_aisle --agents 30 --seed 42` |
| `lecture_hall_no_aisle` + fire | `python -m crowd_evac -s lecture_hall_no_aisle --agents 150 --seed 42 --hazard x=25,y=3.5,tick=100,source_type=fire` |
| `bottleneck_corridor` | `python -m crowd_evac -s bottleneck_corridor --agents 60 --seed 7` |
| `open_room_search` | `python -m crowd_evac -s open_room_search --agents 25 --seed 42` |
| `fixture_minimal` | `python -m crowd_evac -s fixture_minimal --agents 50 --seed 42` |
| `model_room_1` | `python -m crowd_evac -s model_room_1 --agents 1 --seed 42` |

`lecture_hall_no_aisle` (formerly `lecture_hall_small`) has the lecture hall's
50 x 30 m footprint but solid seating rows and no centre aisle, so it is not
the same geometry as `lecture_hall`. The old `hazard_lecture_hall` room is the
fire command above.

An unknown name is rejected with the list of valid choices. Without
`--scenario`, the default is `lecture_hall`.

## Monte Carlo Testing

`scripts/monte_carlo.py` runs one room many times and reports how long the
last person took to leave. Each run gets its own spawn seed, and hazards can
vary per run.

```powershell
python scripts/monte_carlo.py
python scripts/monte_carlo.py -s lecture_hall --agents 150 --runs 50 --seed 7
python scripts/monte_carlo.py --agents 150 --hazard x=10..40,y=2..28,tick=0..400,source_type=fire
python scripts/monte_carlo.py --runs 5 --visual
```

| Flag | Meaning | Default |
|---|---|---|
| `-s`, `--scenario NAME` | bundled room | `lecture_hall` |
| `--agents N` | persons per run (N >= 1) | 1 |
| `--runs N` | number of runs (N >= 1) | 20 |
| `--visual` | play the runs in a window, one after another | headless |
| `--seed S` | base seed (S >= 0) | random, printed in the report |
| `--hazard K=V,...` | hazard, repeatable | none |

`--hazard` extends the game flag: `x`, `y` and `tick` take a value or a range
`lo..hi`, sampled once per run. Any other key stays a fixed value. `x` and `y`
are required (for the whole room, pass its bounds, e.g. `x=0..50`). `tick` is
in simulation ticks of 0.05 s, so `tick=100` is 5 s. A reversed range, a range
on another key, or a bound outside the room is a usage error. A sampled
position inside a wall is redrawn; a fixed position inside a wall is accepted.
A hazard whose tick falls after the run ended never fires; the report shows
this in the `Hazards fired` column.

The report lists every run with its seed, then the mean, sample standard
deviation, minimum and maximum time to last evacuation. Only runs where
everyone left are averaged; the others are counted as incomplete. The last
line is the elapsed wall-clock time. The same base seed and options give the
same report, headless or visual, except for that line. To replay one
run in the game, use its seed with `--seed` and its sampled hazard (printed
below the table) as a fixed `--hazard`.

A run ends when everyone has left, when 60 s pass without anyone leaving, or
after 10,000 ticks. The 60 s limit (`STALL_TICKS`, 1200 ticks) applies to the
game too: it used to be 10 s, which cut off long single-person walks.

The dashboard will call `crowd_evac.montecarlo.runner.run_monte_carlo`
directly; its result also carries the data for the evacuated-versus-time
graph.

### Migrating schema-1.0 files

Version 1.0 files (with `agents`, `simulation` or `events` blocks) are
rejected. Convert them with:

```powershell
python scripts/migrate_scenarios.py assets\scenarios\my_room.json
```

The script rewrites each file to `"2.0"` and prints the equivalent run
command, including any `--hazard` flags, so no value is lost. Files already on
`"2.0"` are skipped.

## Room Designer

```powershell
python -m crowd_evac --design   # short form: -d
```

Draw walls (drag), add doorways on walls at the room edge, add obstacles, and
edit any item's numbers in the side panel. **Save** writes
`assets/scenarios/<name>.json` (name set under Room); the new room then runs
with `--scenario <name> --agents N`. **Open saved room** loads an existing
scenario for editing. Saved files hold the room only. `--design` cannot be
combined with `--list`, `--scenario`, `--agents`, `--seed` or `--hazard`.

## Project Structure

```
src/crowd_evac/
├── domain/        # Core NumPy simulation (agents, forces, spatial)
├── pathfinding/   # Grid flow field and navigation
├── application/   # Fixed-step loop, orchestration, injection API
├── scenarios/     # Default scenario data and schema
├── metrics/       # Per-tick measurement records
├── montecarlo/    # Monte Carlo batches (scripts/monte_carlo.py)
├── ports/         # Domain-level interfaces
└── adapters/
    ├── render/    # Arcade rendering backend
    └── io/        # Scenario loading and asset resolution

tests/             # Test suite mirroring src/ structure
assets/scenarios/  # Bundled scenario data (Lecture Hall)
docs/              # Documentation (PRD, plans)
```

## Gameplay

- **Launch:** `python -m crowd_evac` opens directly into the Lecture Hall scenario (3 agents; use `--agents 150` for the full crowd)
- **Controls:**
  - **Click and drag** to place/move a panic source (fire, smoke, etc.)
  - **Spacebar** to start, pause, or reset the simulation
  - **Resize or maximize** the window at any time; the room scales to fit, keeps its proportions, and is centred (clicks in the side bars are ignored)
- **Watch:** The crowd flees the panic source, re-routes around obstacles, and evacuates through exits
- **Goal:** Observe emergent crowd behavior and test evacuation effectiveness under a fixed-step seeded simulation

## Simulation Features

- **Deterministic behavior:** Same seed reproduces identical tick-by-tick results
- **Panic field:** Dynamic influence zone radiating from panic source; agents avoid it
- **Pathfinding:** Flow-field based navigation with real-time route updates
- **Exit capacity:** Exits queue agents and enforce realistic evacuation rates
- **Agent physics:** Collision avoidance, overlap resolution, velocity-based steering
- **Metrics:** Per-tick counts of active agents, evacuated agents, and panic-induced congestion

## Development

### Quality Gates (Run After Every Step)

```powershell
flake8 src/           # PEP 8 linting
mypy src/ --strict    # Strict type checking
pytest tests/ -v      # Run all tests (excludes perf, e2e, render)
pytest tests/ --cov=src --cov-branch --cov-report=term-missing  # Coverage report
```

### Running all tests except performance, render, e2e

```powershell
flake8 src/; mypy src/ --strict --no-error-summary; pytest tests/ -qq --no-header -m "not perf and not render and not e2e"
```

### Running Tests by Category

Tests are organized by marker. Default runs exclude performance, end-to-end, and render tests for speed:

```powershell
# All unit tests only (default, ~5 seconds)
pytest tests/ -v

# Include end-to-end headless tests
pytest tests/ -m e2e -v

# Include render tests (requires display)
pytest tests/ -m render -v

# Run a specific test file
pytest tests/crowd_evac/domain/test_forces.py -v
```

### Performance Testing

The project has two performance benchmarks targeting the simulation loop. These are excluded from default test runs to keep CI fast.

#### Headless load test (no display required)

Runs the full simulation pipeline (force composition, integration, exit resolution) for a range of agent counts and prints a timing table:

```powershell
pytest tests/ -m perf -v -s
```

The `-s` flag passes stdout through so the timing table prints to the console.

**Sample output:**

```
  agents   ticks   mean ms    min ms    p99 ms    max ms    ticks/s  agent·ticks/s
--------  ------  --------  --------  --------  --------  ---------  --------------
     100     200     0.xxx     0.xxx     0.xxx     0.xxx      x,xxx       x,xxx,xxx
     500     200     0.xxx     0.xxx     0.xxx     0.xxx      x,xxx       x,xxx,xxx
   1,000     200     0.xxx     0.xxx     0.xxx     0.xxx      x,xxx       x,xxx,xxx
   2,000     200     0.xxx     0.xxx     0.xxx     0.xxx      x,xxx       x,xxx,xxx
   5,000     200     0.xxx     0.xxx     0.xxx     0.xxx      x,xxx       x,xxx,xxx
```

Floor: 30 m × 30 m open room, single east exit, 5 force terms enabled, seed 42. Run on the target machine and record results in a performance log.

#### Simulation + render load test (requires display)

Opens an arcade window, runs `Simulation.step()` + sprite rendering each frame, and reports sim-step cost and draw cost separately:

```powershell
python scripts/bench_sim_render.py --agents 500
python scripts/bench_sim_render.py --agents 1000 --frames 300
```

**Sample output:**

```
sim + render benchmark @ 500 agents
  frames timed      :      300
  --- sim step (step() + snapshot()) ---
  mean step ms      :    x.xxx
  p99  step ms      :    x.xxx
  --- arcade draw (sprite pos update + draw call) ---
  mean draw ms      :    x.xxx
  p99  draw ms      :    x.xxx
  --- combined frame ---
  mean frame ms     :    x.xxx
  p99  frame ms     :    x.xxx
  mean FPS          :    xxx.x
  1% low FPS        :    xxx.x
```

The render spike (`python scripts/bench_render.py`) isolates pure arcade rendering cost with no simulation logic.

#### Adding load tests

**Any change to code on the critical path must update the performance
tests.** Critical path includes:

- `domain/forces.py` — force composition, spatial hash
- `domain/integrator.py` — semi-implicit Euler
- `domain/exit_model.py` — exit queuing and egress
- `domain/spatial_hash.py` — neighbour queries
- `application/simulation.py` — step pipeline orchestration
- `pathfinding/flow_field.py` — flow field sampling

For changes to these modules, either add a new parametrize case to
`tests/crowd_evac/performance/test_perf_headless.py`, add a targeted
test isolating the changed subsystem, or add a comment explaining why
the change does not affect per-tick cost.

### Architecture

Built on clean-architecture patterns:
- **Domain layer:** Pure NumPy, no framework dependencies
- **Ports:** Interface-based abstractions (Renderer, InputSource, etc.)
- **Adapters:** Concrete implementations (arcade rendering, file I/O)
- **Application:** Fixed-step orchestration and user interaction


## Git workflow — commit, push, PR, merge, close issue

Run after the quality gate passes, from the **project root** in PowerShell.
Update every `<...>` placeholder before running its command — nothing in
angle brackets is meant to run as-is.

```powershell
# --- 1. Commit ---
git add -A OR git add <file1> <file2>
git commit -m @'
<type>(<scope>): <short summary>

<body: what changed and why>

Refs: <issue-or-spec-reference>
'@
```

```powershell
# --- 2. Push ---
git push -u origin <branch-name>
```

```powershell
# --- 3. Open PR ---
gh pr create --title "<pr-title>" --body @'
## Summary
<bullet points of what changed>

## Test plan
<checklist of how it was verified>

Closes #<issue-number>
'@
```

```powershell
# --- 4. Merge to main ---
gh pr merge <pr-number> --squash --delete-branch
```

```powershell
# --- 5. Close issue ---
gh issue close <issue-number> --comment "Fixed in #<pr-number>: <one-line summary of the fix>."
```

```powershell
# --- 6. Push and merge into main without PR ---
git switch main
git pull origin main
git merge <local-branch>
git push origin main
```
