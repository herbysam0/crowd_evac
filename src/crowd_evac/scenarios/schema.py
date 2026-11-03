"""TypedDicts describing the JSON schema for crowd_evac scenario files.

These types are the contract between the on-disk JSON format and the
scenario loader. All fields map 1:1 to JSON keys. Optional fields use
``NotRequired``; required fields are present in all valid scenario files.

Schema version: "2.0"
"""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class WallData(TypedDict):
    """JSON schema for a single rectangular wall segment."""

    x: float
    y: float
    width: float
    height: float
    label: NotRequired[str]


class ObstacleData(TypedDict):
    """JSON schema for a single interior obstacle."""

    x: float
    y: float
    width: float
    height: float
    label: NotRequired[str]


class ExitData(TypedDict):
    """JSON schema for an exit opening in a wall."""

    x: float
    y: float
    width_m: float
    side: Literal["north", "south", "east", "west"]
    capacity_per_second: int
    label: NotRequired[str]


class FloorPlanData(TypedDict):
    """JSON schema for the floor_plan section of a scenario file."""

    width_m: float
    height_m: float
    walls: list[WallData]
    obstacles: list[ObstacleData]
    exits: list[ExitData]


class EmergencyEventData(TypedDict):
    """One scripted emergency event injected mid-run (run-time input).

    Not part of the scenario file: built by the ``--hazard`` CLI flag and by
    the optimiser's hazard validation.

    Mirrors a runtime panic-source injection
    (:func:`crowd_evac.application.injection.add_panic_source`): at simulation
    tick ``tick`` a panic source is placed at ``pos``, driving the panic
    gradient and (when ``blocks_navigation``) re-routing the crowd around the
    hazard.  All fields beyond ``tick``, ``type`` and ``pos`` are optional and
    fall back to the :func:`~crowd_evac.application.injection.add_panic_source`
    defaults when absent.

    Fields:
        tick: Simulation tick at which the event fires.  Must be >= 0.
        type: Event kind.  Only ``"place_panic_source"`` is supported.
        pos: World position ``[x, y]`` in metres (a two-element list).
        intensity: Initial source intensity in ``[0, 1]``.
        radius: Panic-gradient influence radius in metres.
        decay_rate: Intensity reduction per simulated second.
        block_radius: Navigation-block footprint radius in metres.
        blocks_navigation: Whether the hazard blocks flow-field cells.
        source_type: Hazard type tag (e.g. ``"fire"``).
    """

    tick: int
    type: Literal["place_panic_source"]
    pos: list[float]
    intensity: NotRequired[float]
    radius: NotRequired[float]
    decay_rate: NotRequired[float]
    block_radius: NotRequired[float]
    blocks_navigation: NotRequired[bool]
    source_type: NotRequired[str]


class ScenarioData(TypedDict):
    """Root JSON schema for a crowd_evac scenario file (room only).

    Agents and hazards are run-time input (``--agents``, ``--seed``,
    ``--hazard``), not part of the file.

    Example structure::

        {
          "schema_version": "2.0",
          "name": "lecture_hall",
          "floor_plan": { ... }
        }
    """

    schema_version: str
    name: str
    floor_plan: FloorPlanData
