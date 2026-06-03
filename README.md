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

### Architecture

Built on clean-architecture patterns:
- **Domain layer:** Pure NumPy, no framework dependencies
- **Ports:** Interface-based abstractions (Renderer, InputSource, etc.)
- **Adapters:** Concrete implementations (arcade rendering, file I/O)
- **Application:** Fixed-step orchestration and user interaction

## License

MIT
