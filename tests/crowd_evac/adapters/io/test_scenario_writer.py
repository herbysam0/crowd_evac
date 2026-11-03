"""Tests for crowd_evac.adapters.io.scenario_writer.

Covers name safety, overwrite guard, and no-write on failed validation.
All writes go to tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crowd_evac.adapters.io.scenario_writer import (
    scenario_exists,
    validate_name,
    write_scenario,
)
from crowd_evac.domain.errors import (
    DesignValidationError,
    ScenarioValidationError,
)
from crowd_evac.scenarios.schema import ScenarioData


def _data(name: str = "room_a", with_exit: bool = True) -> ScenarioData:
    """Return a minimal valid scenario."""
    exits: list[Any] = []
    if with_exit:
        exits = [
            {
                "x": 5.0,
                "y": 0.25,
                "width_m": 2.0,
                "side": "north",
                "capacity_per_second": 5,
                "label": "door_1",
            }
        ]
    data: Any = {
        "schema_version": "1.0",
        "name": name,
        "floor_plan": {
            "width_m": 10.0,
            "height_m": 8.0,
            "walls": [],
            "obstacles": [],
            "exits": exits,
        },
        "agents": {"count": 3, "spawn_seed": 1},
        "simulation": {"dt": 0.05, "max_ticks": 100},
    }
    return data  # type: ignore[no-any-return]


class TestValidateName:
    """Name pattern."""

    @pytest.mark.parametrize("name", ["room", "room_2", "a", "0_9"])
    def test_valid_names(self, name: str) -> None:
        """Verify safe names pass."""
        validate_name(name)

    @pytest.mark.parametrize(
        "name",
        ["../x", "", "Room", "my room", "a/b", "a\\b", "a.json", "x\n"],
        ids=[
            "traversal", "empty", "upper", "space", "slash",
            "backslash", "dot", "newline",
        ],
    )
    def test_invalid_names_rejected(self, name: str) -> None:
        """Verify unsafe names raise."""
        with pytest.raises(DesignValidationError):
            validate_name(name)


class TestWriteScenario:
    """write_scenario behaviour."""

    def test_writes_loadable_utf8_json(self, tmp_path: Path) -> None:
        """Verify file exists with indented JSON equal to the input."""
        path = write_scenario(_data(), tmp_path)
        assert path == tmp_path / "room_a.json"
        text = path.read_text(encoding="utf-8")
        assert json.loads(text)["name"] == "room_a"
        assert text.startswith('{\n  "schema_version"')

    def test_existing_name_requires_overwrite(self, tmp_path: Path) -> None:
        """Verify the overwrite guard."""
        write_scenario(_data(), tmp_path)
        with pytest.raises(FileExistsError):
            write_scenario(_data(), tmp_path)
        assert scenario_exists("room_a", tmp_path)
        write_scenario(_data(), tmp_path, overwrite=True)

    def test_invalid_scenario_writes_nothing(self, tmp_path: Path) -> None:
        """Verify no file (and no temp file) remains on rejection."""
        with pytest.raises(ScenarioValidationError):
            write_scenario(_data(with_exit=False), tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_failed_overwrite_keeps_old_file(self, tmp_path: Path) -> None:
        """Verify a rejected overwrite leaves the existing file intact."""
        path = write_scenario(_data(), tmp_path)
        before = path.read_text(encoding="utf-8")
        with pytest.raises(ScenarioValidationError):
            write_scenario(_data(with_exit=False), tmp_path, overwrite=True)
        assert path.read_text(encoding="utf-8") == before

    def test_traversal_name_writes_nothing(self, tmp_path: Path) -> None:
        """Verify an unsafe name never reaches the filesystem."""
        with pytest.raises(DesignValidationError):
            write_scenario(_data(name="../evil"), tmp_path)
        assert list(tmp_path.parent.glob("evil*")) == []
