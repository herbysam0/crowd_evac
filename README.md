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

# Run the game (launches Lecture Hall scenario)
python -m crowd_evac

# Run tests
pytest tests/ -v

# Run linter and type checker
flake8 src/
mypy src/ --strict
```

## Project Structure

```
src/crowd_evac/
├── domain/        # Core NumPy simulation (agents, forces, spatial)
├── pathfinding/   # Grid flow field and navigation
├── application/   # Fixed-step loop, orchestration, injection API
├── scenarios/     # Default scenario data and schema
├── metrics/       # Per-tick measurement records
├── ports/         # Domain-level interfaces
└── adapters/
    ├── render/    # Arcade rendering backend
    └── io/        # Scenario loading and asset resolution

tests/             # Test suite mirroring src/ structure
assets/scenarios/  # Bundled scenario data (Lecture Hall)
docs/              # Documentation (PRD, plans)
```

## Gameplay

- **Launch:** `python -m crowd_evac` opens directly into the Lecture Hall scenario
- **Interact:** Click and drag to place/move a fire source
- **Watch:** The crowd flees the fire, re-routes around obstacles, and evacuates through exits
- **Goal:** Observe emergent crowd behavior under a fixed-step seeded simulation

## Development

### Quality Gates (Run After Every Step)

```powershell
flake8 src/           # PEP 8 linting
mypy src/ --strict    # Strict type checking
pytest tests/ -v      # Run all tests
pytest tests/ --cov=src --cov-branch --cov-report=term-missing  # Coverage report
```

### Performance Testing

The project has two performance benchmarks targeting the simulation loop.

#### Headless load test (no display required)

Runs the full 7-step pipeline (force composition, integration, exit
resolution) for a range of agent counts and prints a timing table:

```powershell
pytest tests/ -m perf -v -s
```

The `-s` flag passes stdout through so the timing table prints to the console.
Performance tests are excluded from the default `pytest` run to keep CI fast.

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

Floor: 30 m × 30 m open room, single east exit, 5 force terms enabled,
seed 42. Run on the target machine and record results in a performance log.

#### Simulation + render load test (requires display)

Opens an arcade window, runs `Simulation.step()` + sprite rendering each
frame, and reports sim-step cost and draw cost separately:

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

The render spike (`python scripts/bench_render.py`) remains available
to isolate pure arcade rendering cost with no simulation logic.

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

## License

MIT
