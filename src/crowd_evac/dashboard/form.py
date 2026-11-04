"""Turn dashboard field text into a Monte Carlo config, or field errors.

Pure: no arcade.  :func:`build_config` validates a
:class:`~crowd_evac.dashboard.form_fields.FormFields` all at once and raises
:class:`FormError` carrying one message per bad field, keyed so the window
can show each next to its input.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from crowd_evac.adapters.io.scenario_catalog import list_bundled_scenarios
from crowd_evac.adapters.io.scenario_loader import load_bundled_scenario
from crowd_evac.application.hazard_args import HazardSpecError
from crowd_evac.dashboard.form_fields import (
    DEFAULT_FIELD_SEED,
    MAX_HAZARDS,
    FormError,
    FormFields,
    HazardFields,
    add_hazard,
    hazard_from_type,
    remove_hazard,
)
from crowd_evac.domain.constants import DT
from crowd_evac.domain.errors import ScenarioError
from crowd_evac.montecarlo.config import (
    HazardSpec,
    MonteCarloConfig,
    ValueRange,
)
from crowd_evac.montecarlo.hazards import check_specs_in_room, validate_spec

__all__ = [
    "DEFAULT_FIELD_SEED",
    "MAX_HAZARDS",
    "FormError",
    "FormFields",
    "HazardFields",
    "add_hazard",
    "build_config",
    "describe_errors",
    "hazard_from_type",
    "remove_hazard",
    "room_errors",
]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _to_int(text: str, label: str, minimum: int, errors: dict[str, str],
            key: str) -> int | None:
    """Parse a whole number ``>= minimum``; record an error otherwise."""
    try:
        value = int(text.strip())
    except ValueError:
        errors[key] = f"{label} must be a whole number >= {minimum}"
        return None
    if value < minimum:
        errors[key] = f"{label} must be >= {minimum}"
        return None
    return value


def _to_float(text: str, label: str, errors: dict[str, str],
              key: str) -> float | None:
    """Parse a finite number; record an error otherwise."""
    try:
        value = float(text.strip())
    except ValueError:
        errors[key] = f"{label} must be a number"
        return None
    if not math.isfinite(value):
        errors[key] = f"{label} must be finite"
        return None
    return value


def _range(low: str, high: str, label: str, errors: dict[str, str],
           key: str) -> float | ValueRange | None:
    """Parse min/max text; blank or equal max gives a fixed value."""
    lo = _to_float(low, f"{label} min", errors, key)
    if lo is None:
        return None
    if not high.strip():
        return lo
    hi = _to_float(high, f"{label} max", errors, key)
    if hi is None:
        return None
    if hi < lo:
        errors[key] = f"{label} max must be >= min"
        return None
    return lo if hi == lo else ValueRange(lo, hi)


def _seconds_to_ticks(value: float | ValueRange) -> int | ValueRange:
    """Convert seconds to ticks with ``round(s / DT)``."""
    if isinstance(value, ValueRange):
        return ValueRange(round(value.lo / DT), round(value.hi / DT))
    return round(value / DT)


def _optional(text: str, label: str, errors: dict[str, str],
              key: str) -> float | None:
    """Parse an optional number; blank means the simulation default."""
    if not text.strip():
        return None
    return _to_float(text, label, errors, key)


# ---------------------------------------------------------------------------
# Building the config
# ---------------------------------------------------------------------------


def _build_hazard(
    index: int, row: HazardFields, errors: dict[str, str]
) -> HazardSpec | None:
    """Turn one hazard row into a spec, or record its errors."""
    pre = f"hazard.{index}."
    before = len(errors)
    x = _range(row.x_min, row.x_max, "x", errors, pre + "x")
    y = _range(row.y_min, row.y_max, "y", errors, pre + "y")
    seconds = _range(row.t_min, row.t_max, "time", errors, pre + "time")
    numbers = {
        name: _optional(getattr(row, name), name.replace("_", " "), errors,
                        pre + name)
        for name in ("intensity", "radius", "decay_rate", "block_radius")
    }
    if len(errors) > before or x is None or y is None or seconds is None:
        return None
    spec = HazardSpec(
        x=x, y=y, tick=_seconds_to_ticks(seconds),
        blocks_navigation=row.blocks_navigation,
        source_type=row.source_type.strip() or None,
        **numbers,
    )
    try:
        validate_spec(spec)
    except HazardSpecError as exc:
        errors[pre.rstrip(".")] = str(exc)
        return None
    return spec


def build_config(
    fields: FormFields,
    *,
    monte_carlo: bool,
    scenarios: Sequence[str] | None = None,
    max_workers: int | None = None,
) -> MonteCarloConfig:
    """Validate the whole form and build the batch settings.

    Single run: ``runs=1`` with ``direct_seed`` so the seed matches the
    game's ``--seed``.  Monte Carlo: the seed is the base seed.

    Args:
        fields: Form text.
        monte_carlo: ``True`` for a batch, ``False`` for one run.
        scenarios: Valid room names; defaults to the bundled list.
        max_workers: Headless worker count, ``None`` for all cores.

    Returns:
        The config to run.

    Raises:
        FormError: With a message per invalid field.
    """
    errors: dict[str, str] = {}
    known = list_bundled_scenarios() if scenarios is None else scenarios
    if fields.scenario not in known:
        errors["scenario"] = f"Unknown scenario {fields.scenario!r}"
    persons = _to_int(fields.persons, "Persons", 1, errors, "persons")
    runs = 1
    if monte_carlo:
        runs = _to_int(fields.iterations, "Iterations", 1, errors,
                       "iterations") or 1
    seed = None
    if fields.seed.strip():
        seed = _to_int(fields.seed, "Seed", 0, errors, "seed")
    if len(fields.hazards) > MAX_HAZARDS:
        errors["hazards"] = f"At most {MAX_HAZARDS} hazards"
    specs = [
        _build_hazard(i, row, errors) for i, row in enumerate(fields.hazards)
    ]
    if errors or persons is None:
        raise FormError(errors)
    return MonteCarloConfig(
        scenario=fields.scenario, agents=persons, runs=runs,
        visual=fields.visual, seed=seed,
        hazards=tuple(s for s in specs if s is not None),
        max_workers=max_workers, direct_seed=not monte_carlo,
    )


def room_errors(config: MonteCarloConfig) -> dict[str, str]:
    """Check hazard bounds against the room before any process starts.

    Args:
        config: Config from :func:`build_config`.

    Returns:
        Empty when fine; else ``{"form": message}`` for a hazard outside
        the room or a room that cannot be loaded.
    """
    try:
        floor_plan, _ = load_bundled_scenario(config.scenario)
        check_specs_in_room(config.hazards, floor_plan)
    except ScenarioError as exc:  # includes HazardOutOfRoomError
        return {"form": str(exc)}
    return {}


def describe_errors(errors: dict[str, str]) -> str:
    """Join field errors into the status-line text, one per line.

    Args:
        errors: Messages keyed as in :class:`FormError`.

    Returns:
        Lines in key order; hazard messages are prefixed ``Hazard N:``.
    """
    lines: list[str] = []
    for key, message in errors.items():
        parts = key.split(".")
        if parts[0] == "hazard" and len(parts) > 1 and parts[1].isdigit():
            lines.append(f"Hazard {int(parts[1]) + 1}: {message}")
        else:
            lines.append(message)
    return "\n".join(lines)
