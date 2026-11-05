"""Scenario loader: reads JSON scenario files and returns domain objects.

Two entry points:
- :func:`load_scenario_file` — load from an explicit filesystem path.
- :func:`load_bundled_scenario` — load by name from repo ``assets/scenarios``,
  independent of the current working directory (R0.4, R3.3).

Both return ``(FloorPlan, ScenarioData)``; the FloorPlan is the
pure-domain representation; ScenarioData carries the schema version and
room name.  Scenario files describe the room only; agent count, seed and
hazards are run-time input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crowd_evac.adapters.io.scenario_catalog import SCENARIOS_DIR
from crowd_evac.domain.errors import MalformedScenarioError
from crowd_evac.domain.floor_plan import (
    Exit,
    ExitSide,
    FloorPlan,
    Obstacle,
    Wall,
)
from crowd_evac.scenarios.schema import (
    ExitData,
    FloorPlanData,
    ObstacleData,
    ScenarioData,
    WallData,
)

SUPPORTED_VERSION = "2.0"
_LEGACY_VERSION = "1.0"
MIGRATION_SCRIPT = "scripts/migrate_scenarios.py"


def load_scenario_file(path: Path) -> tuple[FloorPlan, ScenarioData]:
    """Load a scenario from an explicit filesystem path.

    Args:
        path: Path to a .json scenario file (absolute or relative).

    Returns:
        Tuple of (FloorPlan, ScenarioData).

    Raises:
        FileNotFoundError: If the path does not exist.
        MalformedScenarioError: If the file is not valid JSON or is missing
            required fields.
        ScenarioValidationError: If the data fails semantic validation (e.g.
            negative room dimensions).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise MalformedScenarioError(
            f"Cannot read scenario file {path!r}: {exc}"
        ) from exc
    return _parse_and_build(text, source=str(path))


def load_bundled_scenario(name: str) -> tuple[FloorPlan, ScenarioData]:
    """Load a bundled scenario by name using importlib.resources.

    Bundled scenarios are JSON files in the repo-root
    ``assets/scenarios/<name>.json``. Resolution is anchored to this
    module's location, so the result is independent of the current
    working directory.

    Args:
        name: Scenario name without the .json extension
            (e.g. ``"fixture_minimal"``).

    Returns:
        Tuple of (FloorPlan, ScenarioData).

    Raises:
        MalformedScenarioError: If the bundled file is not found or is
            malformed.
        ScenarioValidationError: If the data fails semantic validation.
    """
    resource = SCENARIOS_DIR / f"{name}.json"
    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MalformedScenarioError(
            f"Bundled scenario not found: {name!r}"
        ) from exc
    return _parse_and_build(text, source=f"bundled:{name}")


# ---------------------------------------------------------------------------
# Internal parsing pipeline
# ---------------------------------------------------------------------------


def _parse_and_build(text: str, source: str) -> tuple[FloorPlan, ScenarioData]:
    """Parse raw JSON text and build domain objects.

    Args:
        text: Raw JSON string.
        source: Human-readable source label used in error messages.

    Returns:
        Tuple of (FloorPlan, ScenarioData).

    Raises:
        MalformedScenarioError: On JSON parse error or missing required field.
        ScenarioValidationError: On semantic validation failure.
    """
    try:
        # json.loads returns Any — this is the JSON boundary; typed below.
        raw: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedScenarioError(
            f"Invalid JSON in scenario {source!r}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise MalformedScenarioError(
            f"Scenario {source!r}: root must be a JSON object, "
            f"got {type(raw).__name__}"
        )

    data = _coerce_scenario(raw, source)
    floor_plan = _build_floor_plan(data["floor_plan"], source)
    return floor_plan, data


def _coerce_scenario(raw: Any, source: str) -> ScenarioData:
    """Validate top-level keys and cast to ScenarioData.

    Raises:
        MalformedScenarioError: On a missing field, a legacy ``"1.0"`` file
            (the message names the migration script) or any other
            unsupported ``schema_version``.
    """
    for field in ("schema_version", "name", "floor_plan"):
        if field not in raw:
            raise MalformedScenarioError(
                f"Scenario {source!r} missing required field: {field!r}"
            )

    version = raw["schema_version"]
    if version == _LEGACY_VERSION:
        raise MalformedScenarioError(
            f"Scenario {source!r}: schema_version {_LEGACY_VERSION!r} is no "
            f"longer supported; run {MIGRATION_SCRIPT} to convert it to "
            f"{SUPPORTED_VERSION!r}"
        )
    if version != SUPPORTED_VERSION:
        raise MalformedScenarioError(
            f"Scenario {source!r}: unsupported schema_version {version!r}; "
            f"expected {SUPPORTED_VERSION!r}"
        )

    return ScenarioData(
        schema_version=str(version),
        name=str(raw["name"]),
        floor_plan=_normalize_floor_plan(raw["floor_plan"], source),
    )


def _normalize_floor_plan(raw: Any, source: str) -> FloorPlanData:
    """Validate floor_plan section and return a normalised FloorPlanData."""
    if not isinstance(raw, dict):
        raise MalformedScenarioError(
            f"Scenario {source!r}: floor_plan must be a JSON object"
        )
    for field in ("width_m", "height_m", "exits"):
        if field not in raw:
            raise MalformedScenarioError(
                f"Scenario {source!r}: floor_plan missing required "
                f"field: {field!r}"
            )
    # walls and obstacles default to empty lists if absent.
    return FloorPlanData(
        width_m=float(raw["width_m"]),
        height_m=float(raw["height_m"]),
        walls=list(raw.get("walls", [])),
        obstacles=list(raw.get("obstacles", [])),
        exits=list(raw["exits"]),
    )


def _build_floor_plan(data: FloorPlanData, source: str) -> FloorPlan:
    """Convert a normalised FloorPlanData to a FloorPlan domain object.

    Raises:
        MalformedScenarioError: If any item in the lists is malformed.
        crowd_evac.domain.errors.ScenarioValidationError: Propagated from
            ``FloorPlan.__post_init__`` if geometry is semantically invalid.
    """
    try:
        walls = tuple(_build_wall(w) for w in data["walls"])
        obstacles = tuple(_build_obstacle(o) for o in data["obstacles"])
        exits = tuple(_build_exit(e, source) for e in data["exits"])
        return FloorPlan(
            width_m=data["width_m"],
            height_m=data["height_m"],
            walls=walls,
            obstacles=obstacles,
            exits=exits,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedScenarioError(
            f"Malformed floor_plan item in {source!r}: {exc}"
        ) from exc


def _build_wall(data: WallData) -> Wall:
    """Build a Wall domain object from a WallData TypedDict."""
    return Wall(
        x=float(data["x"]),
        y=float(data["y"]),
        width=float(data["width"]),
        height=float(data["height"]),
        label=str(data.get("label", "")),
    )


def _build_obstacle(data: ObstacleData) -> Obstacle:
    """Build an Obstacle domain object from an ObstacleData TypedDict."""
    return Obstacle(
        x=float(data["x"]),
        y=float(data["y"]),
        width=float(data["width"]),
        height=float(data["height"]),
        label=str(data.get("label", "")),
    )


def _build_exit(data: ExitData, source: str) -> Exit:
    """Build an Exit domain object from an ExitData TypedDict.

    Raises:
        MalformedScenarioError: If the side value is not a valid ExitSide.
    """
    try:
        side = ExitSide(data["side"])
    except ValueError as exc:
        valid = [s.value for s in ExitSide]
        raise MalformedScenarioError(
            f"Scenario {source!r}: invalid exit side {data['side']!r}; "
            f"expected one of {valid}"
        ) from exc
    return Exit(
        x=float(data["x"]),
        y=float(data["y"]),
        width_m=float(data["width_m"]),
        side=side,
        capacity_per_second=float(data["capacity_per_second"]),
        label=str(data.get("label", "")),
    )
