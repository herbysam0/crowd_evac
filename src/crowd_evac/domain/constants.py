"""Global constants for crowd evacuation simulation.

Defines physical parameters, force weights, and simulation tuning.
All values are in SI units (meters, seconds) unless noted.

Release R0.3: the 13 behavioural weights below (all annotated "calibrated")
were set by the Phase-2 NSGA-II optimisation run documented in
``docs/calibration_report_r03.md``.  The §8 EB-1..6 thresholds (bottom of
this file) were derived from the same empirical reference targets used by the
realism metric (``optimization.realism``).
"""
from __future__ import annotations

# -- Simulation timestep ---------------------------------------------------

DT: float = 0.05
"""Fixed simulation timestep in seconds."""

# -- Agent dynamics (FR-1: speed, accel limits) — calibrated R0.3 ----------

MAX_SPEED: float = 2.429916430805863
"""Maximum agent speed in meters per second.

Calibrated: Phase-2 NSGA-II winner (realism_distance=0.266, stuck_count=0).
The Phase-1 hand-tuned value was 2.5; the realism metric anchors free-walking
speed to 1.20-1.40 m/s (Weidmann 1993) but allows the integrator cap to sit
above free-walking speed so panicking agents can overshoot briefly.
"""

MAX_ACCEL: float = 5.007946955156023
"""Maximum agent acceleration magnitude in meters per second squared.

Calibrated: Phase-2 NSGA-II winner.  Higher than the Phase-1 hand-tuned 2.0;
the optimiser found faster response to force changes produces better realism
at the small-crowd scales scored during search.
"""

PANIC_SPEED_MULTIPLIER: float = 1.9619922537090624
"""Speed boost factor when panicked (panic * MAX_SPEED * multiplier).

Calibrated: Phase-2 NSGA-II winner.  Search range [1.0, 2.5]; calibrated
value 1.96 sits toward the upper end, reflecting a strong speed boost that
drives the faster-is-slower congestion effect.
"""

RELAXATION_TIME: float = 0.21846420038476363
"""Characteristic time (s) for velocity relaxation in the exit-seeking force.

Calibrated: Phase-2 NSGA-II winner.  Short relaxation time (0.22 s) means
agents accelerate toward their desired velocity quickly — consistent with
a high-urgency evacuation context.
"""

AGENT_RADIUS: float = 0.55  # fixed physical property — not optimised
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

# -- Spatial awareness (FR-2: crowd dynamics) — calibrated R0.3 -------------

REPULSION_RADIUS: float = 1.563764341639853
"""Agent-agent collision detection and repulsion range in meters.

Calibrated: Phase-2 NSGA-II winner.  The wider radius (1.56 m vs Phase-1 0.5)
reflects longer-range personal-space anticipation seen in dense crowd flow.
"""

REPULSION_STRENGTH: float = 10.23159406176788
"""Scaling factor for short-range repulsion force magnitude.

Calibrated: Phase-2 NSGA-II winner.  Strong repulsion (10.2 vs Phase-1 1.0)
combined with the wider radius produces the arch-and-jam pattern at exits
(EB-1 and the faster-is-slower effect).
"""

REPULSION_MIN_DISTANCE: float = 0.05
"""Distance floor (m) for the repulsion kernel, preventing division blow-up.

Pair separations are clamped up to this value before the ``1/d`` repulsion
term is evaluated, so the unclamped acceleration stays finite even when two
agents are nearly coincident (the integrator clamps it to ``MAX_ACCEL``).
"""

# -- Density and flow (FR-2: throughput constraints) — calibrated R0.3 ------

DENSITY_SENSING_RADIUS: float = 2.792731466846792
"""Radius (m) over which local crowd density is measured (R2.2).

Calibrated: Phase-2 NSGA-II winner.  The density-pressure term counts
neighbours within this radius and divides by the disc area to estimate
agents per square metre.  Wider than Phase-1 (2.79 vs 1.0 m) to capture
density effects earlier as a crowd approaches a bottleneck.
"""

HIGH_DENSITY_THRESHOLD: float = 2.9709257535036206
"""Agents per square meter above which density pressure applies.

Calibrated: Phase-2 NSGA-II winner.  Lower threshold than Phase-1 (2.97 vs
4.0 /m²) means density drag activates sooner, matching the realism model's
``EB_CONGESTION_FLOOR_M2 = 0.8 /m²`` where jam formation begins.
"""

DENSITY_PRESSURE_STRENGTH: float = 4.1994076051570435
"""Scaling factor for density-based speed reduction.

Calibrated: Phase-2 NSGA-II winner.  Strong density drag (4.2 vs Phase-1
0.3) is needed to reproduce the bottleneck jam-density signature measured by
the realism metric (Seyfried et al. 2005 specific-flow target).
"""

HERD_PERCEPTION_RADIUS: float = 15.30757394634138
"""Radius (m) over which an agent averages neighbour velocity for herding.

Calibrated: Phase-2 NSGA-II winner.  Wide herd radius (15.3 m vs Phase-1
5.0 m) gives agents a global sense of crowd-flow direction in large rooms,
enabling the lane-formation and flow-splitting emergent behaviors.
"""

HERD_ATTRACTION_STRENGTH: float = 1.5326666533995992
"""Scaling factor for local velocity attraction (herd behavior).

Calibrated: Phase-2 NSGA-II winner.  Significantly higher than Phase-1 (1.53
vs 0.1) to produce visible herding during panic; the NSGA-II balanced this
against repulsion to avoid over-herding into a single exit.
"""

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

PANIC_REPULSION_STRENGTH: float = 1.7750644294194786
"""Scaling factor for panic-gradient repulsion force.

Calibrated: Phase-2 NSGA-II winner.  Slightly higher than Phase-1 (1.78 vs
1.5); the optimiser kept this near the prior, which already produced a clear
drift-away response without overwhelming the exit-seeking force.
"""

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

HAZARD_AVOIDANCE_COST: float = 78.70969415548011
"""Strength of hazard route-avoidance in the flow-field solve (R4.3).

Multiplier on traversal cost at the *centre* of a hazard's panic radius,
decaying to zero at the radius edge: entering a cell costs
``base_step * (1 + HAZARD_AVOIDANCE_COST * danger)`` where ``danger`` is the
normalised proximity to the hazard.  Unlike a hard block it keeps every cell
walkable (so no agent loses its exit-seeking direction), while making routes
through a hazard so expensive that agents divert to the next-best exit whenever
one is reachable.

Calibrated: Phase-2 NSGA-II winner.  Higher than Phase-1 (78.7 vs 50.0);
the optimiser reinforced the already-overpowering avoidance to ensure clean
crowd re-routing on hazard scenarios (validated by the hazard-check step in
``optimization.select``).  With two exits and a fire near one exit the crowd
routes entirely to the other; an agent crosses through a hazard only when it
is the sole way out.  Exposed as the
:class:`~crowd_evac.domain.params.ForceParams` weight ``hazard_avoidance_cost``
so it is tunable per scenario; ``0`` disables avoidance (route by distance
only).
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

# -- §8 Emergent-Behavior detection thresholds (empirical, set in Phase 2) --
#
# Derived from the same reference statistics used by ``optimization.realism``:
#   - EB_CONGESTION_FLOOR_M2 = 0.8  agents/m²  (realism.py)
#   - FREE_WALK_SPEED_BAND_MPS = (1.20, 1.40)  m/s  (Weidmann 1993; Fruin 1971)
#   - SPECIFIC_FLOW_BAND_PMS = (1.20, 1.30)    persons/(m·s)  (Seyfried et al.)
#   - WEIDMANN_JAM_DENSITY_M2 = 5.4            agents/m²
#
# These thresholds mark the minimum observable signal for each emergent behavior
# (PRD §8).  A run is credited with the behavior when the measured signal
# exceeds the threshold; below it the behavior is absent or too weak to detect.

EB1_UPSTREAM_DENSITY_THRESHOLD: float = 0.8
"""Minimum upstream local density (agents/m²) that constitutes detectable
cross-floor congestion propagation (EB-1).

Derived from ``optimization.realism.EB_CONGESTION_FLOOR_M2``: when local
density on an upper floor exceeds this floor, the bottleneck is real enough
that a time-lagged rise in downstream queue length is expected and measurable.
"""

EB2_COLLAPSE_PANIC_THRESHOLD: float = 0.7
"""Minimum normalised panic level [0, 1] at which stairwell collapse
probability rises detectably (EB-2).

Set empirically from Phase-1/Phase-2 runs: at panic ≥ 0.7 the combined
crowd-repulsion + density-pressure forces on a narrow stair edge are large
enough to exceed the modelled collapse criterion (R9.4).
"""

EB3_FLOW_SPLIT_FRACTION: float = 0.15
"""Minimum fractional change in exit share (relative to the no-signage
baseline) that constitutes detectable signage-induced flow splitting (EB-3).

Derived from the bottleneck specific-flow band width:
``(SPECIFIC_FLOW_BAND_PMS[1] - SPECIFIC_FLOW_BAND_PMS[0]) / target``
= (1.30 - 1.20) / 1.25 ≈ 0.08.  A 15 % shift is roughly 2× the reference-
band half-width, giving a comfortable signal-to-noise margin.
"""

EB4_PANIC_WAVE_MIN_SPEED_MPS: float = 0.50
"""Minimum panic-wave propagation speed (m/s) for detectable EB-4 diffusion.

Calibrated at roughly one-third of the lower free-walking-speed bound
(1.20 m/s / 3 ≈ 0.40 m/s) plus a margin; 0.50 m/s captures the observable
wavefront that forms when a panic source activates near a cluster of agents
(limited to line-of-sight diffusion until the Phase-10 cascade feature lands;
PRD §8 EB-4 caveat).
"""

EB5_INTERFERENCE_DEVIATION: float = 0.05
"""Minimum relative deviation (dimensionless) from the superposition of two
single-source panic fields required for detectable multi-source interference
(EB-5).

Set at 5 % of the combined field magnitude at the interference centroid.  At
this level the saddle / ridge structure is visible in the gradient field and
produces measurably different agent drift compared with the single-source
baseline.
"""

EB6_FALSE_ROUTE_FRACTION: float = 0.10
"""Minimum fraction of the crowd that must initially route toward the
false-optimal exit for detectable false-routing behavior (EB-6).

Set at 10 % of total agent count; below this the effect is indistinguishable
from stochastic position spread.  At 10 % a visible initial cluster forms near
the sign-directed exit before the crowd re-routes upon encountering the block
or hazard.
"""
