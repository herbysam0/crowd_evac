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

AGENT_RADIUS: float = 0.2
"""Personal-space radius of a single agent in meters.

Two agents are considered overlapping when their centre distance drops below
``2 * AGENT_RADIUS``. Used by crowd-repulsion tuning and tests (FR-2 R2.1).
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

PANIC_REPULSION_STRENGTH: float = 1.5
"""Scaling factor for panic-gradient repulsion force."""

# -- Grid resolution (FR-4: pathfinding) ------------------------------------

GRID_CELL_SIZE: float = 0.25
"""Navigation grid cell size in meters."""

# -- Egress/exit dynamics (FR-5: evacuation flow) ----------------------------

EXIT_CAPACITY_PER_SECOND: int = 5
"""Maximum agents passing through an exit per fixed timestep."""

# -- Rendering and visualization (FR-7) ------------------------------------

PIXELS_PER_METER: float = 40.0
"""Display scaling factor for rendering (pixels per meter)."""
