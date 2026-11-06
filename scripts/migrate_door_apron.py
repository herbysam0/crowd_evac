"""Add a 1 m egress apron behind every door of schema-2.0 scenario files.

Rewrites each file in place as schema ``2.1`` (see
:func:`crowd_evac.scenarios.apron.add_aprons`): the floor plan grows on every
side that hosts a door, west / south growth shifts all geometry, and a walled
apron corridor is added behind each door. Files already on ``"2.1"`` are
skipped, so reruns are safe.

Agent positions are run-time input and not stored in scenario files; any
externally saved spawn positions for a room whose west or south side gained
an apron must be shifted by the printed ``dx`` / ``dy`` by hand.

Usage (PowerShell, .venv activated)::

    python scripts/migrate_door_apron.py (Get-ChildItem assets\\scenarios\\*.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from migrate_scenarios import _dump  # noqa: E402

from crowd_evac.scenarios.apron import APRON_VERSION, add_aprons  # noqa: E402
from crowd_evac.scenarios.schema import ScenarioData  # noqa: E402

SOURCE_VERSION = "2.0"
ENCODING = "utf-8"


class MigrationError(Exception):
    """A scenario file could not be migrated."""


def migrate_file(path: Path) -> tuple[float, float] | None:
    """Migrate one scenario file in place.

    Args:
        path: A ``.json`` schema-2.0 scenario file.

    Returns:
        ``(dx, dy)`` geometry shift applied, or ``None`` when the file is
        already on schema 2.1 and was left untouched.

    Raises:
        MigrationError: If the file is unreadable, not valid JSON, or not a
            schema 2.0 / 2.1 scenario.
    """
    try:
        raw: Any = json.loads(path.read_text(encoding=ENCODING))
    except OSError as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MigrationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MigrationError(f"{path}: root must be a JSON object")
    version = raw.get("schema_version")
    if version == APRON_VERSION:
        return None
    if version != SOURCE_VERSION:
        raise MigrationError(
            f"{path}: unsupported schema_version {version!r}; "
            "run scripts/migrate_scenarios.py first"
        )
    try:
        migrated = add_aprons(cast(ScenarioData, raw))
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError(f"{path}: malformed floor plan: {exc!r}") from exc
    path.write_text(_dump(migrated) + "\n", encoding=ENCODING)
    return _shift(raw, migrated)


def _shift(before: Any, after: Any) -> tuple[float, float]:
    """Return the (dx, dy) translation applied to the first wall or exit."""
    for group in ("walls", "exits"):
        old = before["floor_plan"].get(group, [])
        if old:
            new = after["floor_plan"][group][0]
            return (
                round(new["x"] - old[0]["x"], 6),
                round(new["y"] - old[0]["y"], 6),
            )
    return (0.0, 0.0)


def main(argv: Sequence[str] | None = None) -> int:
    """Migrate every path given; return a process exit code.

    Args:
        argv: Argument list excluding the program name.

    Returns:
        ``0`` when every file migrated or was already current, else ``1``.
    """
    parser = argparse.ArgumentParser(
        description="Add 1 m egress aprons (schema 2.0 -> 2.1)."
    )
    parser.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    args = parser.parse_args(argv)
    failed = False
    for path in args.paths:
        try:
            shift = migrate_file(path)
        except MigrationError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            failed = True
            continue
        if shift is None:
            print(f"skip   {path} (already {APRON_VERSION})")
        else:
            print(f"ok     {path} (shift dx={shift[0]}, dy={shift[1]})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
