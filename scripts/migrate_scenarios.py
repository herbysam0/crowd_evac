"""Convert schema-1.0 scenario files to schema 2.0 (room configuration only).

Schema 2.0 scenario files hold ``schema_version``, ``name`` and
``floor_plan``.  The ``agents``, ``simulation`` and ``events`` blocks of a 1.0
file are run-time input now, so this script strips them and prints the
equivalent ``python -m crowd_evac`` command for each file.  Nothing the old
file said is lost: copy the printed command.

Files already on ``"2.0"`` are skipped, so reruns are safe.

Usage (PowerShell, .venv activated)::

    python scripts/migrate_scenarios.py assets\\scenarios\\my_room.json
    python scripts/migrate_scenarios.py (Get-ChildItem assets\\scenarios\\*.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LEGACY_VERSION = "1.0"
TARGET_VERSION = "2.0"
REMOVED_BLOCKS: tuple[str, ...] = ("agents", "simulation", "events")
FALLBACK_AGENT_COUNT = 3
FALLBACK_SEED = 42
EVENT_OPTIONAL_KEYS: tuple[str, ...] = (
    "intensity",
    "radius",
    "decay_rate",
    "block_radius",
    "blocks_navigation",
    "source_type",
)
ENCODING = "utf-8"


class MigrationError(Exception):
    """A scenario file could not be migrated."""


def _fmt_value(value: Any) -> str:
    """Render one event value the way ``--hazard`` parses it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def hazard_flag(event: Mapping[str, Any]) -> str:
    """Return the ``--hazard`` argument equivalent to one scripted event.

    Args:
        event: One entry of a 1.0 file's ``events`` array.

    Returns:
        ``--hazard x=..,y=..,tick=..[,key=value ...]``.

    Raises:
        MigrationError: If the event has no usable ``pos``.
    """
    pos = event.get("pos")
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        raise MigrationError(f"event has no two-element pos: {event!r}")
    parts = [
        f"x={_fmt_value(pos[0])}",
        f"y={_fmt_value(pos[1])}",
        f"tick={_fmt_value(event.get('tick', 0))}",
    ]
    parts += [
        f"{key}={_fmt_value(event[key])}"
        for key in EVENT_OPTIONAL_KEYS
        if key in event
    ]
    return "--hazard " + ",".join(parts)


def run_command(name: str, data: Mapping[str, Any]) -> str:
    """Build the run command equivalent to a 1.0 file's removed blocks.

    Args:
        name: Scenario name (the file stem).
        data: Parsed 1.0 scenario.

    Returns:
        A ``python -m crowd_evac`` command line.
    """
    agents = data.get("agents", {})
    count = agents.get("count", FALLBACK_AGENT_COUNT)
    seed = agents.get("spawn_seed", FALLBACK_SEED)
    cmd = f"python -m crowd_evac -s {name} --agents {count} --seed {seed}"
    for event in data.get("events", []):
        cmd += " " + hazard_flag(event)
    return cmd


def _dump(value: Any, level: int = 0) -> str:
    """Serialise JSON in the repo's style: flat objects stay on one line."""
    if isinstance(value, dict):
        if all(not isinstance(v, (dict, list)) for v in value.values()):
            return json.dumps(value)
        pad = "  " * (level + 1)
        body = ",\n".join(
            f"{pad}{json.dumps(k)}: {_dump(v, level + 1)}"
            for k, v in value.items()
        )
        return "{\n" + body + "\n" + "  " * level + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        pad = "  " * (level + 1)
        body = ",\n".join(f"{pad}{_dump(v, level + 1)}" for v in value)
        return "[\n" + body + "\n" + "  " * level + "]"
    return json.dumps(value)


def migrate_file(path: Path) -> str | None:
    """Migrate one scenario file in place.

    Args:
        path: A ``.json`` scenario file.

    Returns:
        The equivalent run command, or ``None`` when the file is already on
        schema 2.0 and was left untouched.

    Raises:
        MigrationError: If the file is unreadable, not valid JSON, or not a
            schema-1.0 scenario.
    """
    try:
        raw = json.loads(path.read_text(encoding=ENCODING))
    except OSError as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MigrationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MigrationError(f"{path}: root must be a JSON object")
    version = raw.get("schema_version")
    if version == TARGET_VERSION:
        return None
    if version != LEGACY_VERSION:
        raise MigrationError(
            f"{path}: unsupported schema_version {version!r}"
        )
    command = run_command(path.stem, raw)
    migrated = {k: v for k, v in raw.items() if k not in REMOVED_BLOCKS}
    migrated["schema_version"] = TARGET_VERSION
    path.write_text(_dump(migrated) + "\n", encoding=ENCODING)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Migrate every path given; return a process exit code.

    Args:
        argv: Argument list excluding the program name.

    Returns:
        ``0`` when every file migrated or was already current, else ``1``.
    """
    parser = argparse.ArgumentParser(
        description="Convert schema-1.0 scenario files to schema 2.0."
    )
    parser.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    args = parser.parse_args(argv)
    failed = False
    for path in args.paths:
        try:
            command = migrate_file(path)
        except MigrationError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            failed = True
            continue
        if command is None:
            print(f"skip   {path} (already {TARGET_VERSION})")
        else:
            print(f"ok     {path}\n       {command}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
