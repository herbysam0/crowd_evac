"""Safe scenario JSON writer used by the room designer.

Names are restricted to ``[a-z0-9_]+`` (no path traversal). The file is
first written to a temporary sibling, read back through the real loader,
and only then moved into place, so an invalid scenario never appears in
the scenarios directory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from crowd_evac.adapters.io.scenario_catalog import SCENARIOS_DIR
from crowd_evac.adapters.io.scenario_loader import load_scenario_file
from crowd_evac.domain.errors import (
    CrowdEvacError,
    DesignValidationError,
)
from crowd_evac.scenarios.schema import ScenarioData

_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
_JSON_INDENT = 2


def validate_name(name: str) -> None:
    """Check a scenario name is safe to use as a file stem.

    Args:
        name: Candidate scenario name.

    Raises:
        DesignValidationError: If the name is empty or has characters
            outside ``a-z``, ``0-9`` and ``_``.
    """
    if not _NAME_PATTERN.fullmatch(name):
        raise DesignValidationError(
            f"Invalid scenario name {name!r}: use only a-z, 0-9 and _"
        )


def scenario_path(name: str, directory: Path = SCENARIOS_DIR) -> Path:
    """Return the file path a named scenario is saved at.

    Raises:
        DesignValidationError: If ``name`` is invalid.
    """
    validate_name(name)
    return directory / f"{name}.json"


def scenario_exists(name: str, directory: Path = SCENARIOS_DIR) -> bool:
    """Return True if a scenario file with this name already exists."""
    return scenario_path(name, directory).exists()


def write_scenario(
    data: ScenarioData,
    directory: Path = SCENARIOS_DIR,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a scenario to ``<directory>/<name>.json`` after validation.

    Args:
        data: Scenario to save; ``data["name"]`` names the file.
        directory: Target directory (tests pass ``tmp_path``).
        overwrite: Replace an existing file when True.

    Returns:
        Path of the written file.

    Raises:
        DesignValidationError: If the name is invalid.
        FileExistsError: If the file exists and ``overwrite`` is False.
        CrowdEvacError: If the loader rejects the data; nothing is written.
    """
    target = scenario_path(data["name"], directory)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Scenario {data['name']!r} already exists")
    temp = target.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(data, indent=_JSON_INDENT) + "\n", encoding="utf-8"
    )
    try:
        load_scenario_file(temp)
    except CrowdEvacError:
        temp.unlink()
        raise
    temp.replace(target)
    return target
