"""Parse and validate ``--hazard`` values (scripted panic sources).

A hazard is run-time input, not part of a scenario file.  :func:`parse_hazard`
turns ``x=25,y=3.5,tick=100`` into the
:class:`~crowd_evac.scenarios.schema.EmergencyEventData` the simulation fires;
:func:`check_hazards_in_room` rejects positions outside the loaded room.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

from crowd_evac.domain.constants import (
    HAZARD_BLOCK_RADIUS,
    PANIC_DECAY_RATE,
    PANIC_RANGE,
)
from crowd_evac.domain.errors import ScenarioValidationError
from crowd_evac.domain.floor_plan import FloorPlan
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.scenarios.schema import EmergencyEventData

_REQUIRED_HAZARD_KEYS: tuple[str, ...] = ("x", "y")
_FLOAT_HAZARD_KEYS: tuple[str, ...] = (
    "intensity",
    "radius",
    "decay_rate",
    "block_radius",
)
_TRUE_WORDS: frozenset[str] = frozenset({"true", "1", "yes"})
_FALSE_WORDS: frozenset[str] = frozenset({"false", "0", "no"})


class HazardSpecError(ValueError):
    """A ``--hazard`` value is malformed."""


class HazardOutOfRoomError(ScenarioValidationError):
    """A ``--hazard`` position lies outside the loaded room."""


# ---------------------------------------------------------------------------
# --hazard parsing
# ---------------------------------------------------------------------------


def _to_float(key: str, text: str) -> float:
    """Parse a finite float for hazard key ``key``."""
    try:
        value = float(text)
    except ValueError:
        raise HazardSpecError(f"{key}={text!r} is not a number") from None
    if not math.isfinite(value):
        raise HazardSpecError(f"{key}={text!r} must be finite")
    return value


def _to_bool(key: str, text: str) -> bool:
    """Parse a boolean (true/false, 1/0, yes/no) for hazard key ``key``."""
    word = text.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise HazardSpecError(f"{key}={text!r} must be true or false")


def _split_pairs(spec: str) -> dict[str, str]:
    """Split ``k=v,k=v`` into a dict, rejecting duplicates and bad pairs."""
    pairs: dict[str, str] = {}
    for item in spec.split(","):
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise HazardSpecError(f"expected key=value, got {item!r}")
        if key in pairs:
            raise HazardSpecError(f"duplicate key {key!r}")
        pairs[key] = value.strip()
    return pairs


def _optional_fields(pairs: dict[str, str]) -> dict[str, Any]:
    """Convert the optional hazard keys present in ``pairs``."""
    fields: dict[str, Any] = {}
    for key in _FLOAT_HAZARD_KEYS:
        if key in pairs:
            fields[key] = _to_float(key, pairs[key])
    if "blocks_navigation" in pairs:
        fields["blocks_navigation"] = _to_bool(
            "blocks_navigation", pairs["blocks_navigation"]
        )
    if "source_type" in pairs:
        if not pairs["source_type"]:
            raise HazardSpecError("source_type must not be empty")
        fields["source_type"] = pairs["source_type"]
    return fields


def _check_ranges(event: dict[str, Any]) -> None:
    """Fail now, not mid-run, on out-of-range hazard values."""
    try:
        PanicSource(
            x=event["pos"][0],
            y=event["pos"][1],
            intensity=event.get("intensity", 1.0),
            radius=event.get("radius", PANIC_RANGE),
            decay_rate=event.get("decay_rate", PANIC_DECAY_RATE),
            block_radius=event.get("block_radius", HAZARD_BLOCK_RADIUS),
        )
    except ValueError as exc:
        raise HazardSpecError(str(exc)) from exc


def parse_hazard(spec: str) -> EmergencyEventData:
    """Parse one ``--hazard`` value into a scripted event.

    Format: comma-separated ``key=value`` pairs.  ``x`` and ``y`` (metres)
    are required.  Optional: ``tick`` (int >= 0, default 0), ``intensity``,
    ``radius``, ``decay_rate``, ``block_radius``, ``blocks_navigation``
    (true/false) and ``source_type``.  Unset keys fall back to the
    :func:`~crowd_evac.application.injection.add_panic_source` defaults.
    Value ranges (e.g. intensity in [0, 1]) are those of
    :class:`~crowd_evac.domain.panic_source.PanicSource`.

    Args:
        spec: Text such as ``x=25,y=3.5,tick=100,source_type=fire``.

    Returns:
        The event, carrying only the optional keys that were given.

    Raises:
        HazardSpecError: On an unknown or duplicate key, a missing ``x`` or
            ``y``, a non-numeric or out-of-range value, or a negative tick.

    Example:
        >>> parse_hazard("x=1,y=2,tick=5")["pos"]
        [1.0, 2.0]
    """
    pairs = _split_pairs(spec)
    known = {*_REQUIRED_HAZARD_KEYS, "tick", *_FLOAT_HAZARD_KEYS,
             "blocks_navigation", "source_type"}
    unknown = sorted(set(pairs) - known)
    if unknown:
        raise HazardSpecError(
            f"unknown key {unknown[0]!r}; valid keys: {', '.join(sorted(known))}"
        )
    for key in _REQUIRED_HAZARD_KEYS:
        if key not in pairs:
            raise HazardSpecError(f"missing required key {key!r}")
    try:
        tick = int(pairs.get("tick", "0"))
    except ValueError:
        raise HazardSpecError(
            f"tick={pairs['tick']!r} is not an integer"
        ) from None
    if tick < 0:
        raise HazardSpecError(f"tick must be >= 0, got {tick}")
    event: dict[str, Any] = {
        "tick": tick,
        "type": "place_panic_source",
        "pos": [_to_float("x", pairs["x"]), _to_float("y", pairs["y"])],
    }
    event.update(_optional_fields(pairs))
    _check_ranges(event)
    return cast(EmergencyEventData, event)


def check_hazards_in_room(
    events: Sequence[EmergencyEventData], floor_plan: FloorPlan
) -> None:
    """Reject hazards positioned outside the room.

    Positions inside a wall are accepted, exactly as for a mouse-placed
    source.

    Args:
        events: Scripted hazards from ``--hazard``.
        floor_plan: The loaded room.

    Raises:
        HazardOutOfRoomError: If any position is outside
            ``[0, width_m] x [0, height_m]``.
    """
    for event in events:
        x, y = event["pos"]
        if not (0.0 <= x <= floor_plan.width_m
                and 0.0 <= y <= floor_plan.height_m):
            raise HazardOutOfRoomError(
                f"--hazard position ({x:g}, {y:g}) is outside the "
                f"{floor_plan.width_m:g} x {floor_plan.height_m:g} m room"
            )
