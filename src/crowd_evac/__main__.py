"""Package entry point: ``python -m crowd_evac`` launches the app."""
from __future__ import annotations

from crowd_evac.application.cli import main

# The guard keeps a spawned child process (a dashboard batch) from
# re-running ``main`` when it re-imports this module.
if __name__ == "__main__":
    main()
