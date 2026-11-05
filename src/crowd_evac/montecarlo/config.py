"""Settings for a Monte Carlo batch.

Plain frozen dataclasses with no parsing: the CLI turns text into these, and
the future dashboard panel fills them in directly.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MC_SCENARIO: str = "lecture_hall"
"""Bundled room used when none is given."""

DEFAULT_MC_AGENTS: int = 1
"""Persons per run.  Separate from the game's ``DEFAULT_AGENT_COUNT`` (3)."""

DEFAULT_MC_RUNS: int = 20
"""Runs per batch when ``--runs`` is not given."""

MAX_HAZARD_SAMPLE_ATTEMPTS: int = 100
"""Redraws of a sampled hazard position that lands in a wall or obstacle."""

HAZARD_STREAM: int = 0x4841
"""Constant mixed into the hazard RNG key so that stream is independent of
the spawn-seed stream: adding a hazard never moves spawn positions."""


@dataclass(frozen=True)
class ValueRange:
    """Closed interval ``[lo, hi]`` sampled once per run.

    Attributes:
        lo: Lower bound.
        hi: Upper bound; equal to ``lo`` gives a fixed value.

    Raises:
        ValueError: If ``lo > hi``.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        """Reject a reversed interval."""
        if self.lo > self.hi:
            raise ValueError(
                f"range lower bound {self.lo:g} exceeds upper bound "
                f"{self.hi:g}"
            )


@dataclass(frozen=True)
class HazardSpec:
    """One hazard, fixed or randomized, for every run of a batch.

    ``x``, ``y`` and ``tick`` take a value or a :class:`ValueRange`; the
    other fields are fixed.  ``None`` means "use the simulation default".
    Hazard behaviour comes from ``intensity``, ``radius`` and
    ``decay_rate``; ``source_type`` only selects the icon.

    Attributes:
        x: Position in metres.
        y: Position in metres.
        tick: Simulation tick at which the hazard appears (``DT`` = 0.05 s).
        intensity: Initial intensity in ``[0, 1]``.
        radius: Panic influence radius in metres.
        decay_rate: Intensity lost per simulated second.
        block_radius: Navigation-block footprint radius in metres.
        blocks_navigation: Whether the hazard re-routes the crowd.
        source_type: Label such as ``"fire"``.
    """

    x: float | ValueRange
    y: float | ValueRange
    tick: int | ValueRange = 0
    intensity: float | None = None
    radius: float | None = None
    decay_rate: float | None = None
    block_radius: float | None = None
    blocks_navigation: bool | None = None
    source_type: str | None = None


@dataclass(frozen=True)
class MonteCarloConfig:
    """All settings of one batch.

    Attributes:
        scenario: Bundled room name.
        agents: Persons per run, ``>= 1``.
        runs: Number of runs, ``>= 1``.
        visual: Play the runs in a window instead of headless.
        seed: Base seed, ``>= 0``; ``None`` draws a random one.
        hazards: Hazards applied in every run.
        max_workers: Headless worker processes; ``None`` uses all cores and
            ``1`` runs in-process.  Results never depend on this.
        direct_seed: Use ``seed`` itself as the spawn seed of the single
            run, matching the game's ``--seed``, instead of deriving it.
        start_paused: A visual single run waits for SPACE before it starts;
            SPACE then pauses and resumes it.  Headless runs ignore it.

    Raises:
        ValueError: If a count is below its minimum, or ``direct_seed`` or
            ``start_paused`` is set with more than one run.
    """

    scenario: str = DEFAULT_MC_SCENARIO
    agents: int = DEFAULT_MC_AGENTS
    runs: int = DEFAULT_MC_RUNS
    visual: bool = False
    seed: int | None = None
    hazards: tuple[HazardSpec, ...] = ()
    max_workers: int | None = None
    direct_seed: bool = False
    start_paused: bool = False

    def __post_init__(self) -> None:
        """Validate counts and the seed."""
        if self.agents < 1:
            raise ValueError(f"agents must be >= 1, got {self.agents}")
        if self.runs < 1:
            raise ValueError(f"runs must be >= 1, got {self.runs}")
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.direct_seed and self.runs != 1:
            raise ValueError(
                f"direct_seed needs exactly one run, got {self.runs}"
            )
        if self.start_paused and self.runs != 1:
            raise ValueError(
                f"start_paused needs exactly one run, got {self.runs}"
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(
                f"max_workers must be >= 1, got {self.max_workers}"
            )
