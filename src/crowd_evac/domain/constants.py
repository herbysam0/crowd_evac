"""Global constants for crowd evacuation simulation.

Defines physical parameters, force weights, and simulation tuning.
All values are in SI units (meters, seconds) unless noted.
"""
from __future__ import annotations

# -- Simulation timestep ---------------------------------------------------

DT: float = 0.05
"""Fixed simulation timestep in seconds."""

# -- Agent dynamics (FR-1: speed, accel limits) ----------------------------

MAX_SPEED: float = 2.5
"""Maximum agent speed in meters per second."""

MAX_ACCEL: float = 2.0
"""Maximum agent acceleration magnitude in meters per second squared."""

PANIC_SPEED_MULTIPLIER: float = 1.3
"""Speed boost factor when panicked (panic * MAX_SPEED * multiplier)."""

RELAXATION_TIME: float = 0.5
"""Characteristic time (s) for velocity relaxation in the exit-seeking force."""

AGENT_RADIUS: float = 0.55
"""Personal-space radius of a single agent in meters.

Two agents are considered overlapping when their centre distance drops below
``2 * AGENT_RADIUS``. Used by crowd-repulsion tuning and tests (FR-2 R2.1).
"""

OVERLAP_RESOLUTION_ITERATIONS: int = 4
"""Position-projection passes per tick for the hard no-overlap invariant.

Each pass separates overlapping agent pairs and pushes agents off walls
(:mod:`crowd_evac.domain.overlap`). A few Gauss-Seidel-style passes resolve
chained overlaps (an agent shoved by one neighbour into another) and the
agent-vs-wall corner case where separating from a wall re-overlaps a peer.
Higher values converge tighter at linear cost; ``4`` clears typical crowd
densities while staying cheap at Tier A counts (FR-2 R2.1 / FR-3 R3.2).
"""

# -- Spatial awareness (FR-2: crowd dynamics) --------------------------------

REPULSION_RADIUS: float = 0.5
"""Agent-agent collision detection and repulsion range in meters."""

REPULSION_STRENGTH: float = 1.0
"""Scaling factor for short-range repulsion force magnitude."""

REPULSION_MIN_DISTANCE: float = 0.05
"""Distance floor (m) for the repulsion kernel, preventing division blow-up.

Pair separations are clamped up to this value before the ``1/d`` repulsion
term is evaluated, so the unclamped acceleration stays finite even when two
agents are nearly coincident (the integrator clamps it to ``MAX_ACCEL``).
"""

# -- Density and flow (FR-2: throughput constraints) -------------------------

DENSITY_SENSING_RADIUS: float = 1.0
"""Radius (m) over which local crowd density is measured (R2.2).

The density-pressure term counts neighbours within this radius and divides by
the disc area to estimate agents per square metre.
"""

HIGH_DENSITY_THRESHOLD: float = 4.0
"""Agents per square meter above which density pressure applies."""

DENSITY_PRESSURE_STRENGTH: float = 0.3
"""Scaling factor for density-based speed reduction."""

HERD_PERCEPTION_RADIUS: float = 5.0
"""Radius (m) over which an agent averages neighbour velocity for herding.

Herd alignment (R2.5) draws an agent toward the mean velocity of everyone
within this radius; it is deliberately much wider than the density and
repulsion radii so panicked agents follow the broader crowd flow.
"""

HERD_ATTRACTION_STRENGTH: float = 0.1
"""Scaling factor for local velocity attraction (herd behavior)."""

# -- Panic dynamics (FR-11: panic source effects) ----------------------------

PANIC_RANGE: float = 10.0
"""Initial radius of panic source influence in meters."""

PANIC_DECAY_RATE: float = 0.02
"""Intensity reduction per second when panic source is active."""

AGENT_PANIC_DECAY_RATE: float = 0.05
"""Agent panic decay rate in units per second.

When the local panic field value drops below an agent's current panic level
(because the agent moved outside the influence radius or the source decayed),
the agent's panic falls toward the field value at this rate rather than
remaining locked at its peak.
"""

PANIC_REPULSION_STRENGTH: float = 1.5
"""Scaling factor for panic-gradient repulsion force."""

# -- Grid resolution (FR-4: pathfinding) ------------------------------------

GRID_CELL_SIZE: float = 0.25
"""Navigation grid cell size in meters."""

HAZARD_BLOCK_RADIUS: float = 1.5
"""Radius (m) of the navigation block a hazard punches in the flow field.

The *physical* footprint a hazard makes impassable, deliberately decoupled
from :data:`PANIC_RANGE` (the much larger radius over which the hazard merely
*frightens* agents).  A fire occupies roughly its drawn size
(:data:`FIRE_SYMBOL_SIZE_M` = 3 m diameter → ~1.5 m radius), not its 10 m fear
radius.  Blocking the full fear radius engulfs the surrounding crowd, stranding
agents whose every neighbouring cell becomes impassable and who therefore lose
all exit-seeking direction (the flow field returns a zero vector when every
neighbour is blocked).  Navigation blocking uses this radius; the panic
gradient uses :data:`PANIC_RANGE`.
"""

HAZARD_AVOIDANCE_COST: float = 50.0
"""Strength of hazard route-avoidance in the flow-field solve (R4.3).

Multiplier on traversal cost at the *centre* of a hazard's panic radius,
decaying to zero at the radius edge: entering a cell costs
``base_step * (1 + HAZARD_AVOIDANCE_COST * danger)`` where ``danger`` is the
normalised proximity to the hazard.  Unlike a hard block it keeps every cell
walkable (so no agent loses its exit-seeking direction), while making routes
through a hazard so expensive that agents divert to the next-best exit whenever
one is reachable.

Deliberately *overpowering* by default: with two exits and a fire by one of
them, the crowd routes to the other exit (the real-world response).  An agent
only walks through a hazard when it is the sole way out.  Exposed as the
:class:`~crowd_evac.domain.params.ForceParams` weight ``hazard_avoidance_cost``
so it is tunable per scenario and by the Phase-2 optimiser; ``0`` disables
avoidance (route by distance only).
"""

# -- Egress/exit dynamics (FR-5: evacuation flow) ----------------------------

EXIT_CAPACITY_PER_SECOND: int = 5
"""Maximum agents passing through an exit per fixed timestep."""

EXIT_CAPTURE_RADIUS: float = 1.0
"""Distance from exit opening segment (metres) that triggers queue entry.

Measured as perpendicular distance to the nearest point on the exit
segment (not the centre), so wide exits capture approaching agents
uniformly along their full lateral span.
"""

STALL_TICKS: int = 200
"""Consecutive no-egress ticks before declaring the evacuation stalled.

At the default DT of 0.05 s this equals 10 simulated seconds.  After this
many consecutive ticks with zero egress the remaining alive agents are
considered trapped and
:func:`~crowd_evac.application.termination.is_evacuation_complete` returns
``True`` (R5.3).
"""

# -- Rendering and visualization (FR-7) ------------------------------------

PIXELS_PER_METER: float = 40.0
"""Display scaling factor for rendering (pixels per meter)."""

FIRE_SYMBOL_SIZE_M: float = 3.0
"""World-space diameter (m) at which the fire emergency symbol is drawn.

Scales proportionally with pixels_per_meter, so a 3 m symbol occupies
3 × pixels_per_meter pixels on screen.  Adjust to match the visual footprint
of the hazard type.
"""
