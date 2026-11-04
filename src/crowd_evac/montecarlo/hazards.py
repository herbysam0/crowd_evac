"""Parse, validate and sample Monte Carlo hazards.

Extends the game's ``--hazard`` grammar: ``x``, ``y`` and ``tick`` each take
a value or ``lo..hi``.  A range is replaced by its lower bound so the
existing :func:`~crowd_evac.application.hazard_args.parse_hazard` validates
everything else, then the bounds are kept in a
:class:`~crowd_evac.montecarlo.config.HazardSpec`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from crowd_evac.application.hazard_args import (
    HazardSpecError,
    check_event_ranges,
    check_hazards_in_room,
    parse_hazard,
    split_pairs,
)
from crowd_evac.domain.constants import GRID_CELL_SIZE
from crowd_evac.domain.floor_plan import FloorPlan
from crowd_evac.montecarlo.config import (
    HAZARD_STREAM,
    MAX_HAZARD_SAMPLE_ATTEMPTS,
    HazardSpec,
    ValueRange,
)
from crowd_evac.scenarios.schema import EmergencyEventData

_RANGE_SEP: str = ".."
_RANGE_KEYS: tuple[str, ...] = ("x", "y", "tick")
_OPTIONAL_FIELDS: tuple[str, ...] = (
    "intensity", "radius", "decay_rate", "block_radius",
    "blocks_navigation", "source_type",
)


class HazardRangeError(ValueError):
    """A sampled hazard position never left the walls."""


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _parse_bound(key: str, text: str) -> float | int:
    """Parse one range bound: an int for ``tick``, else a finite float."""
    try:
        if key == "tick":
            return int(text)
        value = float(text)
    except ValueError:
        kind = "an integer" if key == "tick" else "a number"
        raise HazardSpecError(f"{key}={text!r} is not {kind}") from None
    if not math.isfinite(value):
        raise HazardSpecError(f"{key}={text!r} must be finite")
    return value


def _split_ranges(
    pairs: dict[str, str],
) -> tuple[dict[str, str], dict[str, ValueRange]]:
    """Replace each ``lo..hi`` value by ``lo``; return the bounds apart."""
    fixed = dict(pairs)
    ranges: dict[str, ValueRange] = {}
    for key, text in pairs.items():
        if _RANGE_SEP not in text:
            continue
        if key not in _RANGE_KEYS:
            raise HazardSpecError(
                f"{key} does not take a range; only x, y and tick do"
            )
        lo_text, _, hi_text = text.partition(_RANGE_SEP)
        lo = _parse_bound(key, lo_text)
        hi = _parse_bound(key, hi_text)
        try:
            ranges[key] = ValueRange(lo, hi)
        except ValueError as exc:
            raise HazardSpecError(f"{key}: {exc}") from None
        fixed[key] = lo_text
    return fixed, ranges


def parse_hazard_spec(text: str) -> HazardSpec:
    """Parse one Monte Carlo ``--hazard`` value.

    Args:
        text: Such as ``x=10..40,y=2..28,tick=0..400,source_type=fire``.
            ``x`` and ``y`` are required; times are ticks (0.05 s each).

    Returns:
        The spec, keeping ranges for ``x``, ``y`` and ``tick``.

    Raises:
        HazardSpecError: On any error
            :func:`~crowd_evac.application.hazard_args.parse_hazard` reports,
            a reversed range, or a range on any other key.
    """
    fixed, ranges = _split_ranges(split_pairs(text))
    event = parse_hazard(",".join(f"{k}={v}" for k, v in fixed.items()))
    values: dict[str, Any] = {
        "x": event["pos"][0],
        "y": event["pos"][1],
        "tick": event["tick"],
    }
    values.update(ranges)
    optional = cast(dict[str, Any], event)
    for name in _OPTIONAL_FIELDS:
        if name in optional:
            values[name] = optional[name]
    return HazardSpec(**values)


def _lower(value: float | ValueRange) -> float:
    """Return ``value``, or the lower bound of a range."""
    return value.lo if isinstance(value, ValueRange) else float(value)


def _upper(value: float | ValueRange) -> float:
    """Return ``value``, or the upper bound of a range."""
    return value.hi if isinstance(value, ValueRange) else float(value)


def _event(
    spec: HazardSpec, x: float, y: float, tick: int
) -> EmergencyEventData:
    """Build the scripted event for concrete ``x``, ``y`` and ``tick``."""
    event: dict[str, Any] = {
        "tick": tick, "type": "place_panic_source", "pos": [x, y],
    }
    for name in _OPTIONAL_FIELDS:
        value = getattr(spec, name)
        if value is not None:
            event[name] = value
    return cast(EmergencyEventData, event)


def validate_spec(spec: HazardSpec) -> None:
    """Reject values the simulation cannot accept, for a hand-built spec.

    Args:
        spec: Hazard to check; a range is checked at both bounds.

    Raises:
        HazardSpecError: On a negative or non-integer tick, a non-finite
            coordinate, or a value outside the ``PanicSource`` limits.
    """
    for pick in (_lower, _upper):
        tick = pick(spec.tick)
        if tick < 0 or tick != int(tick):
            raise HazardSpecError(f"tick must be an integer >= 0, got {tick:g}")
        x, y = pick(spec.x), pick(spec.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise HazardSpecError("x and y must be finite")
        event = _event(spec, x, y, int(tick))
        check_event_ranges(cast(dict[str, Any], event))


def check_specs_in_room(
    specs: Sequence[HazardSpec], floor_plan: FloorPlan
) -> None:
    """Reject any hazard bound outside the room.

    Args:
        specs: Hazards of the batch.
        floor_plan: The loaded room.

    Raises:
        HazardOutOfRoomError: If a coordinate or bound lies outside
            ``[0, width_m] x [0, height_m]``.
    """
    corners = [
        _event(s, pick(s.x), pick(s.y), 0)
        for s in specs
        for pick in (_lower, _upper)
    ]
    check_hazards_in_room(corners, floor_plan)


# ---------------------------------------------------------------------------
# Per-run sampling
# ---------------------------------------------------------------------------


def _draw(rng: np.random.Generator, value: float | ValueRange) -> float:
    """Return ``value``, or a uniform draw from a range."""
    if isinstance(value, ValueRange):
        return float(rng.uniform(value.lo, value.hi))
    return float(value)


def _walkable(mask: npt.NDArray[np.bool_], x: float, y: float) -> bool:
    """Whether ``(x, y)`` falls in a walkable grid cell."""
    rows, cols = mask.shape
    col = min(int(x // GRID_CELL_SIZE), cols - 1)
    row = min(int(y // GRID_CELL_SIZE), rows - 1)
    return bool(mask[row, col])


def _sample_one(
    spec: HazardSpec, mask: npt.NDArray[np.bool_], rng: np.random.Generator
) -> EmergencyEventData:
    """Draw one concrete event for ``spec`` (see :func:`sample_events`)."""
    sampled = isinstance(spec.x, ValueRange) or isinstance(spec.y, ValueRange)
    x, y = float(_lower(spec.x)), float(_lower(spec.y))
    if sampled:
        for _ in range(MAX_HAZARD_SAMPLE_ATTEMPTS):
            x, y = _draw(rng, spec.x), _draw(rng, spec.y)
            if _walkable(mask, x, y):
                break
        else:
            raise HazardRangeError(
                f"no walkable point in x={spec.x}, y={spec.y} after "
                f"{MAX_HAZARD_SAMPLE_ATTEMPTS} attempts"
            )
    tick = spec.tick
    if isinstance(tick, ValueRange):
        tick = int(rng.integers(int(tick.lo), int(tick.hi) + 1))
    return _event(spec, x, y, int(tick))


def sample_events(
    specs: Sequence[HazardSpec],
    mask: npt.NDArray[np.bool_],
    base_seed: int,
    run_index: int,
) -> tuple[EmergencyEventData, ...]:
    """Draw the concrete hazards of one run.

    The generator is keyed ``[base_seed, HAZARD_STREAM, run_index]``, so a
    run's hazards depend on nothing else: not on other runs, and not on the
    spawn seeds.  A position drawn inside a wall is redrawn; a fully fixed
    position is used as given, even inside a wall.

    Args:
        specs: Hazards of the batch.
        mask: Walkable mask from ``FloorPlan.walkable_mask(GRID_CELL_SIZE)``.
        base_seed: Batch base seed.
        run_index: Zero-based run number.

    Returns:
        One event per spec, in order.

    Raises:
        HazardRangeError: If a sampled position stays in a wall for
            :data:`~crowd_evac.montecarlo.config.MAX_HAZARD_SAMPLE_ATTEMPTS`
            draws.
    """
    rng = np.random.default_rng([base_seed, HAZARD_STREAM, run_index])
    return tuple(_sample_one(spec, mask, rng) for spec in specs)
