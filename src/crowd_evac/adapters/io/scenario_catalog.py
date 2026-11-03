"""Catalog of scenarios bundled with the package.

Separate from :mod:`scenario_loader` so enumeration (directory listing)
stays independent of JSON parsing.
"""
from __future__ import annotations

from pathlib import Path

_SCENARIO_SUFFIX = ".json"

# Repo-root assets dir (src/crowd_evac/adapters/io/<this file> -> parents[4]).
# Resolved from __file__ so it is independent of the current working directory.
SCENARIOS_DIR: Path = Path(__file__).resolve().parents[4] / "assets" / "scenarios"


def list_bundled_scenarios() -> list[str]:
    """List bundled scenario names loadable by ``load_bundled_scenario``.

    Returns:
        Sorted scenario names without the ``.json`` extension. Empty list
        if the assets directory is missing.
    """
    directory = SCENARIOS_DIR
    if not directory.is_dir():
        return []
    return sorted(
        entry.name[: -len(_SCENARIO_SUFFIX)]
        for entry in directory.iterdir()
        if entry.is_file() and entry.name.endswith(_SCENARIO_SUFFIX)
    )
