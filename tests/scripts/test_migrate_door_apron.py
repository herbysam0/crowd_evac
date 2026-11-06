"""Tests for scripts/migrate_door_apron.py (schema 2.0 -> 2.1).

Covers:
  - migrate_file: adds aprons, sets "2.1", reports the geometry shift.
  - Idempotence: a 2.1 file is skipped byte-identical.
  - Failure: invalid JSON, wrong root, unsupported version.
  - Every bundled scenario, migrated on a tmp copy, loads and runs 100 ticks.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from crowd_evac.adapters.io.scenario_catalog import SCENARIOS_DIR
from crowd_evac.adapters.io.scenario_loader import load_scenario_file
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.pathfinding.flow_field import FlowField

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_door_apron.py"
RUN_TICKS = 100
RUN_AGENTS = 10
RUN_SEED = 3


def _load_script() -> ModuleType:
    """Import scripts/migrate_door_apron.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("migrate_door_apron", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["migrate_door_apron"] = module
    spec.loader.exec_module(module)
    return module


migrate = _load_script()

_ROOM = {
    "schema_version": "2.0",
    "name": "r",
    "floor_plan": {
        "width_m": 10.0, "height_m": 8.0,
        "walls": [{"x": 0.0, "y": 0.0, "width": 0.5, "height": 8.0}],
        "obstacles": [],
        "exits": [{"x": 0.5, "y": 4.0, "width_m": 2.0, "side": "west",
                   "capacity_per_second": 1.05}],
    },
}


def _write(tmp_path: Path, raw: object) -> Path:
    """Write *raw* as JSON and return the path."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


class TestMigrateFile:
    """migrate_file behaviour."""

    def test_adds_apron_and_reports_shift(self, tmp_path: Path) -> None:
        """Verify version, width, shift and apron walls after migration."""
        path = _write(tmp_path, _ROOM)
        assert migrate.migrate_file(path) == (1.0, 0.0)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == "2.1"
        assert raw["floor_plan"]["width_m"] == 11.0
        assert any(
            w.get("label", "").startswith("apron_wall_")
            for w in raw["floor_plan"]["walls"]
        )

    def test_second_run_is_skipped_and_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """Verify idempotence."""
        path = _write(tmp_path, _ROOM)
        migrate.migrate_file(path)
        before = path.read_bytes()
        assert migrate.migrate_file(path) is None
        assert path.read_bytes() == before

    @pytest.mark.parametrize(
        "content, match",
        [
            ("{not json", "invalid JSON"),
            ("[1]", "root must be"),
            ('{"schema_version": "1.0"}', "unsupported"),
        ],
        ids=["bad-json", "bad-root", "legacy"],
    )
    def test_bad_input_raises(
        self, tmp_path: Path, content: str, match: str
    ) -> None:
        """Verify unusable files raise MigrationError."""
        path = tmp_path / "r.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(migrate.MigrationError, match=match):
            migrate.migrate_file(path)

    def test_main_exit_codes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify 0 on success and 1 when any file fails."""
        good = _write(tmp_path, _ROOM)
        assert migrate.main([str(good)]) == 0
        assert "ok" in capsys.readouterr().out
        bad = tmp_path / "bad.json"
        bad.write_text("{", encoding="utf-8")
        assert migrate.main([str(good), str(bad)]) == 1


def test_all_bundled_scenarios_migrate_load_and_run(tmp_path: Path) -> None:
    """Verify each bundled room migrates, loads and survives 100 ticks."""
    for src in SCENARIOS_DIR.glob("*.json"):
        dest = tmp_path / src.name
        shutil.copy(src, dest)
        migrate.migrate_file(dest)
        plan, data = load_scenario_file(dest)
        assert data["schema_version"] == "2.1"
        rng = SeededRNG(RUN_SEED)
        sim = Simulation(
            state=spawn(plan, RUN_AGENTS, rng.generator),
            flow_field=FlowField.build(plan),
            panic_field=PanicField(),
            exit_model=ExitModel(plan),
            rng=rng,
            collision_map=CollisionMap.from_floor_plan(plan),
        )
        for _ in range(RUN_TICKS):
            if sim.is_complete:
                break
            sim.step()
