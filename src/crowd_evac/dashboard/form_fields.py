"""Form data: the text of every dashboard input and row helpers.

Pure: no arcade.  :class:`FormFields` keeps the text of every input;
``form.build_config`` validates it into a config.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from crowd_evac.adapters.io.hazard_catalog import HazardType
from crowd_evac.montecarlo.config import (
    DEFAULT_MC_AGENTS,
    DEFAULT_MC_RUNS,
    DEFAULT_MC_SCENARIO,
)

MAX_HAZARDS: int = 6
"""Most hazards the form holds; ``arcade.gui`` has no scroll widget here."""

DEFAULT_FIELD_SEED: str = ""
"""Blank seed draws a random one."""


@dataclass(frozen=True)
class HazardFields:
    """Text of one hazard row.  Max blank or equal to min means fixed.

    Attributes:
        source_type: Hazard type name, or blank for none.
        x_min: Position x in metres (range lower bound).
        x_max: Upper bound of x, or blank.
        y_min: Position y in metres (range lower bound).
        y_max: Upper bound of y, or blank.
        t_min: Appearance time in seconds (range lower bound).
        t_max: Upper bound of the time, or blank.
        intensity: Initial intensity in ``[0, 1]``, blank for the default.
        radius: Panic radius in metres, blank for the default.
        decay_rate: Intensity lost per second, blank for the default.
        block_radius: Navigation-block radius in metres, or blank.
        blocks_navigation: Whether the hazard re-routes the crowd.
    """

    source_type: str = ""
    x_min: str = ""
    x_max: str = ""
    y_min: str = ""
    y_max: str = ""
    t_min: str = "0"
    t_max: str = ""
    intensity: str = ""
    radius: str = ""
    decay_rate: str = ""
    block_radius: str = ""
    blocks_navigation: bool = True


@dataclass(frozen=True)
class FormFields:
    """Text and choices of the whole dashboard form.

    Attributes:
        scenario: Bundled room name.
        persons: Persons per run.
        visual: Play runs in a window instead of headless.
        seed: Blank or a non-negative integer.
        iterations: Runs of a Monte Carlo batch.
        hazards: One entry per hazard row.
    """

    scenario: str = DEFAULT_MC_SCENARIO
    persons: str = str(DEFAULT_MC_AGENTS)
    visual: bool = False
    seed: str = DEFAULT_FIELD_SEED
    iterations: str = str(DEFAULT_MC_RUNS)
    hazards: tuple[HazardFields, ...] = field(default_factory=tuple)


class FormError(ValueError):
    """One or more form fields are invalid.

    Attributes:
        errors: Message per field key, e.g. ``"persons"`` or
            ``"hazard.0.x"``.  ``"form"`` holds a message for the whole form.
    """

    def __init__(self, errors: dict[str, str]) -> None:
        """Keep the messages and summarize them."""
        super().__init__("; ".join(errors.values()))
        self.errors = errors


def hazard_from_type(hazard_type: HazardType) -> HazardFields:
    """Prefill a hazard row from a type's defaults; position stays blank.

    Args:
        hazard_type: Type selected in the dashboard.

    Returns:
        Fields carrying the type name and its numeric defaults.
    """
    return HazardFields(
        source_type=hazard_type.name,
        intensity=f"{hazard_type.intensity:g}",
        radius=f"{hazard_type.radius:g}",
        decay_rate=f"{hazard_type.decay_rate:g}",
        block_radius=f"{hazard_type.block_radius:g}",
        blocks_navigation=hazard_type.blocks_navigation,
    )


def add_hazard(fields: FormFields, row: HazardFields) -> FormFields:
    """Return ``fields`` with ``row`` appended, up to :data:`MAX_HAZARDS`.

    Args:
        fields: Current form.
        row: Hazard row to add.

    Returns:
        The same form if it is full, else the form with the extra row.
    """
    if len(fields.hazards) >= MAX_HAZARDS:
        return fields
    return replace(fields, hazards=(*fields.hazards, row))


def remove_hazard(fields: FormFields, index: int) -> FormFields:
    """Return ``fields`` without hazard row ``index`` (ignored if absent)."""
    if not 0 <= index < len(fields.hazards):
        return fields
    rows = fields.hazards[:index] + fields.hazards[index + 1:]
    return replace(fields, hazards=rows)
