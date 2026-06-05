"""Search scenario suite and calibration micro-rigs for Phase-2 weight optimisation.

Provides two functions:

* :func:`search_suite` — the curated, down-scaled scenario list used during
  the NSGA-II search loop.  Each entry is small enough to evaluate cheaply
  (seconds of wall-clock time rather than minutes).
* :func:`calibration_rigs` — controlled synthetic micro-rigs designed to
  produce steady-state crowd statistics (specific flow, speed–density relation)
  under known geometry.  Step 2.4 uses these to anchor the realism metric.

No simulation is executed here.  Both functions return pure-data value objects;
callers pass the ``scenario_ref`` field to
:func:`~crowd_evac.optimization.harness.evaluate`.

All bundled scenario names resolve via
:func:`~crowd_evac.adapters.io.scenario_loader.load_bundled_scenario`; see
the ``assets/scenarios/`` package directory for the JSON files.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class SearchScenario:
    """One scenario entry in the calibration search suite.

    Each instance describes a small, cheap scenario used during the
    multi-objective optimiser's evaluation loop.  The ``scenario_ref``
    field is passed directly to
    :func:`~crowd_evac.optimization.harness.evaluate` — a ``str`` resolves
    as a bundled scenario name; a :class:`~pathlib.Path` resolves as a
    filesystem path.

    Attributes:
        name: Unique identifier, matching the bundled scenario name where
            applicable.
        scenario_ref: Bundled name (``str``) or filesystem path
            (:class:`~pathlib.Path`) forwarded to the harness.
        full_agent_count: Agent count in the corresponding full-scale
            scenario, for reporting the down-scale factor.  Set equal to
            the search count when the scenario has no full-scale sibling.
        description: Human-readable notes on floor plan and purpose.
    """

    name: str
    scenario_ref: str | Path
    full_agent_count: int
    description: str


@dataclasses.dataclass(frozen=True)
class CalibrationRig:
    """A controlled synthetic micro-rig for realism-metric extraction.

    Exposes a single, well-defined bottleneck so that Step 2.4 can extract
    specific flow (persons / m·s) and the speed–density curve under known
    geometry rather than inferred from a noisy full scenario.

    Attributes:
        name: Unique identifier, matching the bundled scenario name.
        scenario_ref: Bundled name (``str``) or filesystem path
            (:class:`~pathlib.Path`) forwarded to the harness.
        door_width_m: Width of the controlling bottleneck opening in metres.
            Used to normalise egress count to specific flow
            ``q = evacuated_count / (evac_time × door_width_m)``
            (persons / m·s).
        description: Human-readable notes on geometry and measurement intent.
    """

    name: str
    scenario_ref: str | Path
    door_width_m: float
    description: str


# ---------------------------------------------------------------------------
# Informational constant — full-scale agent count for logging the scale factor
# ---------------------------------------------------------------------------

_LECTURE_HALL_FULL_AGENTS: int = 150
"""Agent count in the unmodified full-scale ``lecture_hall`` bundled scenario."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_suite() -> list[SearchScenario]:
    """Return the curated scenario list used during optimiser search.

    The list contains two entries that cover distinct floor-plan topologies,
    preventing the optimiser from overfitting to a single geometry:

    1. **lecture_hall_small** — the lecture-hall floor plan (50 m × 30 m,
       five seating-row obstacles, three exits) with 30 agents — 20 % of
       the full 150-agent scenario.  Keeps the primary seating-row bottleneck
       at low evaluation cost.
    2. **open_room_search** — a 20 m × 15 m open room with no interior
       obstacles and two competing exits (north 3 m, east 2 m), 25 agents.
       Tests routing and herding without row-induced bottlenecks.

    Returns:
        List of :class:`SearchScenario` entries, always non-empty, in
        deterministic order.

    Example:
        >>> scenarios = search_suite()
        >>> len(scenarios) >= 2
        True
        >>> all(isinstance(s, SearchScenario) for s in scenarios)
        True
    """
    return [
        SearchScenario(
            name="lecture_hall_small",
            scenario_ref="lecture_hall_small",
            full_agent_count=_LECTURE_HALL_FULL_AGENTS,
            description=(
                "Down-scaled lecture hall: same 50 m × 30 m geometry "
                "with five seating-row obstacles and three exits, reduced "
                "to 30 agents (20 % of full scale) for cheap optimiser "
                "evaluation."
            ),
        ),
        SearchScenario(
            name="open_room_search",
            scenario_ref="open_room_search",
            full_agent_count=25,
            description=(
                "Contrasting open room: 20 m × 15 m with no interior "
                "obstacles and two competing exits (north 3 m, east 2 m), "
                "25 agents.  Tests routing and herding behaviour without "
                "seating-row bottlenecks."
            ),
        ),
    ]


def calibration_rigs() -> list[CalibrationRig]:
    """Return the controlled micro-rig list used for realism measurement.

    Currently contains one rig:

    * **bottleneck_corridor** — a 15 m × 10 m room with a 1.2 m bottleneck
      gap centred at mid-height in a dividing interior wall.  A generous 4 m
      outflow exit on the far (east) wall ensures the 1.2 m gap — not the
      exit — is the throughput constraint.  60 agents provide a sustained
      queue for steady-state specific-flow measurement.

    The ``door_width_m`` field carries the bottleneck width used by Step 2.4
    to compute specific flow:
    ``q_s = evacuated_count / (evac_time × door_width_m)`` (persons / m·s).

    Returns:
        List of :class:`CalibrationRig` entries, always non-empty, in
        deterministic order.

    Example:
        >>> rigs = calibration_rigs()
        >>> len(rigs) >= 1
        True
        >>> rigs[0].door_width_m > 0
        True
    """
    return [
        CalibrationRig(
            name="bottleneck_corridor",
            scenario_ref="bottleneck_corridor",
            door_width_m=1.2,
            description=(
                "15 m × 10 m room divided by an interior wall with a "
                "1.2 m bottleneck gap centred at mid-height (y = 4.4–5.6 m).  "
                "Outflow exit (4 m wide) on the east wall is intentionally "
                "generous so the 1.2 m gap controls throughput.  60 agents "
                "sustain a queue for steady-state specific-flow measurement "
                "of the Weidmann fundamental-diagram reference target."
            ),
        ),
    ]
