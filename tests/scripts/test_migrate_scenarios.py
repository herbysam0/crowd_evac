"""Tests for scripts/migrate_scenarios.py (schema 1.0 -> 2.0).

Covers:
  - migrate_file: removes the three run blocks, sets "2.0", keeps the floor
    plan, prints the equivalent run command (including --hazard flags).
  - Idempotence: a 2.0 file is skipped and left byte-identical.
  - Failure: invalid JSON, wrong root type and unknown versions fail cleanly.
  - main: exit codes and the output the user sees.
"""
from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path
from types import ModuleType

import pytest

from crowd_evac.adapters.io.scenario_loader import load_scenario_file
from crowd_evac.application.hazard_args import parse_hazard

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_scenarios.py"


def _load_script() -> ModuleType:
    """Import scripts/migrate_scenarios.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("migrate_scenarios", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migrate = _load_script()

_FLOOR_PLAN = {
    "width_m": 10.0,
    "height_m": 8.0,
    "walls": [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 0.5}],
    "obstacles": [],
    "exits": [
        {"x": 5.0, "y": 0.25, "width_m": 2.0, "side": "south",
         "capacity_per_second": 5},
    ],
}
_EVENT = {
    "tick": 100, "type": "place_panic_source", "pos": [25.0, 3.5],
    "intensity": 1.0, "source_type": "fire",
}


def _legacy(**extra: object) -> dict[str, object]:
    """Return a 1.0 scenario dict with agents and simulation blocks."""
    return {
        "schema_version": "1.0",
        "name": "old_room",
        "floor_plan": _FLOOR_PLAN,
        "agents": {"count": 30, "spawn_seed": 9},
        "simulation": {"dt": 0.05, "max_ticks": 8000},
        **extra,
    }


def _write(tmp_path: Path, data: object, name: str = "old_room.json") -> Path:
    """Write ``data`` as JSON into tmp_path and return the path."""
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestMigrateFile:
    """migrate_file rewrites one scenario in place."""

    def test_removes_blocks_and_sets_version(self, tmp_path: Path) -> None:
        """Verify agents/simulation/events are gone and version is 2.0."""
        path = _write(tmp_path, _legacy(events=[_EVENT]))
        migrate.migrate_file(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert set(raw) == {"schema_version", "name", "floor_plan"}
        assert raw["schema_version"] == "2.0"
        assert raw["floor_plan"] == _FLOOR_PLAN

    def test_migrated_file_loads(self, tmp_path: Path) -> None:
        """Verify the result passes the real loader."""
        path = _write(tmp_path, _legacy())
        migrate.migrate_file(path)
        floor_plan, data = load_scenario_file(path)
        assert floor_plan.width_m == 10.0
        assert data["schema_version"] == "2.0"

    def test_returns_equivalent_run_command(self, tmp_path: Path) -> None:
        """Verify the printed command carries the old count and seed."""
        path = _write(tmp_path, _legacy())
        assert migrate.migrate_file(path) == (
            "python -m crowd_evac -s old_room --agents 30 --seed 9"
        )

    def test_command_includes_hazard_flag(self, tmp_path: Path) -> None:
        """Verify an old event becomes a --hazard flag."""
        path = _write(tmp_path, _legacy(events=[_EVENT]))
        command = migrate.migrate_file(path)
        assert command.endswith(
            "--hazard x=25.0,y=3.5,tick=100,intensity=1.0,source_type=fire"
        )

    def test_hazard_flag_round_trips_through_the_cli_parser(self) -> None:
        """Verify parse_hazard(flag) rebuilds the original event."""
        flag = migrate.hazard_flag(_EVENT)
        spec = shlex.split(flag)[1]
        assert parse_hazard(spec) == _EVENT

    def test_multiple_events_give_multiple_flags(self, tmp_path: Path) -> None:
        """Verify each event becomes its own --hazard."""
        second = {**_EVENT, "tick": 200, "pos": [1.0, 2.0]}
        path = _write(tmp_path, _legacy(events=[_EVENT, second]))
        assert migrate.migrate_file(path).count("--hazard") == 2

    def test_boolean_event_value_uses_lowercase(self) -> None:
        """Verify blocks_navigation is written true/false for the parser."""
        flag = migrate.hazard_flag({**_EVENT, "blocks_navigation": False})
        assert "blocks_navigation=false" in flag

    def test_missing_agents_block_uses_defaults(self, tmp_path: Path) -> None:
        """Verify a 1.0 file lacking agents still migrates."""
        data = _legacy()
        del data["agents"]
        path = _write(tmp_path, data)
        assert "--agents 3 --seed 42" in migrate.migrate_file(path)

    def test_keeps_flat_objects_on_one_line(self, tmp_path: Path) -> None:
        """Verify the repo's compact style survives (one wall per line)."""
        path = _write(tmp_path, _legacy())
        migrate.migrate_file(path)
        text = path.read_text(encoding="utf-8")
        assert '{"x": 0.0, "y": 0.0, "width": 10.0, "height": 0.5}' in text
        assert text.endswith("}\n")


class TestIdempotence:
    """Migrating twice is safe."""

    def test_already_2_0_is_skipped(self, tmp_path: Path) -> None:
        """Verify a 2.0 file returns None and is not rewritten."""
        path = _write(tmp_path, _legacy())
        migrate.migrate_file(path)
        before = path.read_bytes()
        assert migrate.migrate_file(path) is None
        assert path.read_bytes() == before


class TestFailures:
    """Bad input fails with MigrationError, never a traceback."""

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Verify unparsable JSON is reported."""
        path = tmp_path / "bad.json"
        path.write_text("{ nope", encoding="utf-8")
        with pytest.raises(migrate.MigrationError, match="invalid JSON"):
            migrate.migrate_file(path)

    def test_root_not_an_object(self, tmp_path: Path) -> None:
        """Verify a JSON array root is reported."""
        path = _write(tmp_path, [1, 2, 3])
        with pytest.raises(migrate.MigrationError, match="JSON object"):
            migrate.migrate_file(path)

    def test_unknown_version(self, tmp_path: Path) -> None:
        """Verify an unrecognised version is not touched."""
        path = _write(tmp_path, {**_legacy(), "schema_version": "9.9"})
        before = path.read_bytes()
        with pytest.raises(migrate.MigrationError, match="9.9"):
            migrate.migrate_file(path)
        assert path.read_bytes() == before

    def test_missing_file(self, tmp_path: Path) -> None:
        """Verify an absent file is reported."""
        with pytest.raises(migrate.MigrationError, match="cannot read"):
            migrate.migrate_file(tmp_path / "nope.json")

    def test_event_without_pos(self, tmp_path: Path) -> None:
        """Verify an unusable event aborts before the file is rewritten."""
        path = _write(tmp_path, _legacy(events=[{"tick": 1}]))
        before = path.read_bytes()
        with pytest.raises(migrate.MigrationError, match="pos"):
            migrate.migrate_file(path)
        assert path.read_bytes() == before


class TestMain:
    """main() processes many paths and returns an exit code."""

    def test_prints_command_and_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify the user sees the equivalent command."""
        path = _write(tmp_path, _legacy())
        assert migrate.main([str(path)]) == 0
        out = capsys.readouterr().out
        assert "python -m crowd_evac -s old_room --agents 30 --seed 9" in out

    def test_rerun_reports_skip(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify a second run says skip and still exits 0."""
        path = _write(tmp_path, _legacy())
        migrate.main([str(path)])
        capsys.readouterr()
        assert migrate.main([str(path)]) == 0
        assert "skip" in capsys.readouterr().out

    def test_bad_file_returns_one_and_continues(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify one bad file does not stop the others."""
        bad = tmp_path / "bad.json"
        bad.write_text("{ nope", encoding="utf-8")
        good = _write(tmp_path, _legacy(), "good.json")
        assert migrate.main([str(bad), str(good)]) == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert json.loads(good.read_text(encoding="utf-8"))[
            "schema_version"
        ] == "2.0"
