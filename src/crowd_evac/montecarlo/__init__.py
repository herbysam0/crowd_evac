"""Monte Carlo batches of evacuation runs (issue #5).

Runs one room many times with different spawn positions and, optionally,
different hazards, then reports how long the last person took to leave.

Entry points:

* :func:`crowd_evac.montecarlo.runner.run_monte_carlo` -- the interface the
  future dashboard calls directly (config in, :class:`MonteCarloResult` out).
* :func:`crowd_evac.montecarlo.cli.main` -- the ``scripts/monte_carlo.py``
  command line.

Importing this package loads neither arcade nor the process pool; only
``--visual`` runs import :mod:`crowd_evac.montecarlo.visual`.
"""
from __future__ import annotations
