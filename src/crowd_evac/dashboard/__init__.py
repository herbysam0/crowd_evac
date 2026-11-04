"""Headless logic of the performance dashboard.

Everything here runs without a display: the form turns field text into a
:class:`~crowd_evac.montecarlo.config.MonteCarloConfig`, the batch process
runs it in a child process, and the result and chart models reduce outcomes
to display values.  Arcade code lives in ``adapters/render/dashboard_*``.
"""
