"""Catalog of hazard types: one JSON file per type in ``assets/hazards``.

The file stem is the ``source_type`` (``smoke.json`` is ``smoke``).  A type
only supplies defaults and an optional icon; hazard behaviour comes from the
numeric parameters.  A bad file is logged and skipped so one typo cannot
stop the app.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crowd_evac.domain.constants import (
    HAZARD_BLOCK_RADIUS,
    PANIC_DECAY_RATE,
    PANIC_RANGE,
)
from crowd_evac.domain.panic_source import PanicSource

logger = logging.getLogger(__name__)

_HAZARD_SUFFIX = ".json"

HAZARDS_DIR: Path = Path(__file__).resolve().parents[4] / "assets" / "hazards"
"""Repo-root hazard type directory, independent of the working directory."""

DEFAULT_SYMBOL_SIZE_M: float = 1.0
"""Icon diameter (m) when a type file gives a symbol but no size."""


@dataclass(frozen=True)
class HazardType:
    """Defaults and icon of one hazard type.

    Attributes:
        name: ``source_type`` tag, the file stem.
        label: Name shown in the dashboard.
        symbol: Character drawn at the hazard, or ``None`` for no icon.
        symbol_size_m: Icon diameter in metres.
        intensity: Default initial intensity in ``[0, 1]``.
        radius: Default panic radius (m).
        decay_rate: Default intensity lost per second.
        block_radius: Default navigation-block radius (m).
        blocks_navigation: Whether the hazard re-routes the crowd.
    """

    name: str
    label: str
    symbol: str | None = None
    symbol_size_m: float = DEFAULT_SYMBOL_SIZE_M
    intensity: float = 1.0
    radius: float = PANIC_RANGE
    decay_rate: float = PANIC_DECAY_RATE
    block_radius: float = HAZARD_BLOCK_RADIUS
    blocks_navigation: bool = True


def _number(raw: dict[str, Any], key: str, default: float) -> float:
    """Read an optional numeric field, rejecting booleans and text."""
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return float(value)


def _parse(name: str, raw: object) -> HazardType:
    """Build and validate a type from decoded JSON.

    Raises:
        ValueError: On a wrong shape, wrong field type or out-of-range value.
    """
    if not isinstance(raw, dict):
        raise ValueError("top level must be a JSON object")
    symbol = raw.get("symbol")
    if symbol is not None and not (isinstance(symbol, str) and symbol):
        raise ValueError(f"symbol must be a non-empty string, got {symbol!r}")
    label = raw.get("label", name.capitalize())
    if not isinstance(label, str) or not label:
        raise ValueError(f"label must be a non-empty string, got {label!r}")
    blocks = raw.get("blocks_navigation", True)
    if not isinstance(blocks, bool):
        raise ValueError(
            f"blocks_navigation must be true/false, got {blocks!r}"
        )
    size = _number(raw, "symbol_size_m", DEFAULT_SYMBOL_SIZE_M)
    if size <= 0.0:
        raise ValueError(f"symbol_size_m must be positive, got {size:g}")
    hazard = HazardType(
        name=name,
        label=label,
        symbol=symbol,
        symbol_size_m=size,
        intensity=_number(raw, "intensity", 1.0),
        radius=_number(raw, "radius", PANIC_RANGE),
        decay_rate=_number(raw, "decay_rate", PANIC_DECAY_RATE),
        block_radius=_number(raw, "block_radius", HAZARD_BLOCK_RADIUS),
        blocks_navigation=blocks,
    )
    # PanicSource owns the numeric range rules; reuse them.
    PanicSource(
        0.0, 0.0, hazard.intensity, hazard.radius, hazard.decay_rate,
        name, hazard.block_radius, hazard.blocks_navigation,
    )
    return hazard


def load_hazard_types(
    directory: Path | None = None,
) -> dict[str, HazardType]:
    """Read every valid hazard type file.

    Args:
        directory: Folder to scan; defaults to ``assets/hazards``.

    Returns:
        Types keyed by name, sorted by name.  Empty if the folder is
        missing.  Unreadable or invalid files are logged and skipped.
    """
    folder = HAZARDS_DIR if directory is None else directory
    if not folder.is_dir():
        return {}
    types: dict[str, HazardType] = {}
    for entry in sorted(folder.iterdir()):
        if not (entry.is_file() and entry.name.endswith(_HAZARD_SUFFIX)):
            continue
        name = entry.name[: -len(_HAZARD_SUFFIX)]
        try:
            with entry.open(encoding="utf-8") as handle:
                types[name] = _parse(name, json.load(handle))
        except (OSError, ValueError) as exc:
            # JSONDecodeError and UnicodeDecodeError are ValueErrors.
            logger.warning("Skipping hazard type file %s: %s", entry.name, exc)
    return types
