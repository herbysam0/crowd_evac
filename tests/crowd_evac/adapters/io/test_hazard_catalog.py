"""Tests for crowd_evac.adapters.io.hazard_catalog.

Covers:
  - the shipped fire.json equals the domain constants
  - a new file is listed; the symbol is optional
  - malformed JSON and out-of-range values are skipped with a warning
  - a missing directory gives an empty catalog
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from crowd_evac.adapters.io.hazard_catalog import (
    HAZARDS_DIR,
    load_hazard_types,
)
from crowd_evac.domain.constants import (
    FIRE_SYMBOL_SIZE_M,
    HAZARD_BLOCK_RADIUS,
    PANIC_DECAY_RATE,
    PANIC_RANGE,
)


def _write(folder: Path, name: str, data: object) -> None:
    """Write ``data`` as JSON to ``folder/name``."""
    (folder / name).write_text(json.dumps(data), encoding="utf-8")


class TestShippedFire:
    """assets/hazards/fire.json must track the constants."""

    def test_fire_matches_constants(self) -> None:
        """Verify the shipped file equals today's defaults."""
        fire = load_hazard_types(HAZARDS_DIR)["fire"]
        assert fire.label == "Fire"
        assert fire.symbol == "\U0001f525"
        assert fire.symbol_size_m == pytest.approx(FIRE_SYMBOL_SIZE_M)
        assert fire.intensity == pytest.approx(1.0)
        assert fire.radius == pytest.approx(PANIC_RANGE)
        assert fire.decay_rate == pytest.approx(PANIC_DECAY_RATE)
        assert fire.block_radius == pytest.approx(HAZARD_BLOCK_RADIUS)
        assert fire.blocks_navigation is True


class TestLoad:
    """Discovery and validation."""

    def test_new_file_is_listed_by_stem(self, tmp_path: Path) -> None:
        """Verify smoke.json becomes the type ``smoke``, sorted by name."""
        _write(tmp_path, "smoke.json", {"label": "Smoke", "radius": 5.0})
        _write(tmp_path, "alarm.json", {"label": "Alarm"})
        types = load_hazard_types(tmp_path)
        assert list(types) == ["alarm", "smoke"]
        assert types["smoke"].radius == pytest.approx(5.0)
        assert types["smoke"].name == "smoke"

    def test_symbol_is_optional(self, tmp_path: Path) -> None:
        """Verify a file without a symbol loads with no icon."""
        _write(tmp_path, "gas.json", {"label": "Gas"})
        assert load_hazard_types(tmp_path)["gas"].symbol is None

    def test_label_defaults_to_stem(self, tmp_path: Path) -> None:
        """Verify a missing label falls back to the capitalised stem."""
        _write(tmp_path, "gas.json", {})
        assert load_hazard_types(tmp_path)["gas"].label == "Gas"

    def test_non_json_files_ignored(self, tmp_path: Path) -> None:
        """Verify other files and folders are not read."""
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        (tmp_path / "sub.json").mkdir()
        assert load_hazard_types(tmp_path) == {}

    def test_missing_directory_gives_empty(self, tmp_path: Path) -> None:
        """Verify an absent folder is an empty catalog, not an error."""
        assert load_hazard_types(tmp_path / "nope") == {}

    def test_malformed_json_skipped_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify a broken file is skipped and others still load."""
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        _write(tmp_path, "ok.json", {"label": "Ok"})
        with caplog.at_level(logging.WARNING):
            types = load_hazard_types(tmp_path)
        assert list(types) == ["ok"]
        assert "bad.json" in caplog.text

    @pytest.mark.parametrize(
        "data",
        [
            {"radius": -1.0},
            {"intensity": 2.0},
            {"decay_rate": -0.1},
            {"block_radius": 0},
            {"symbol_size_m": 0},
            {"radius": "big"},
            {"blocks_navigation": "yes"},
            {"symbol": ""},
            [1, 2],
        ],
        ids=[
            "radius", "intensity", "decay", "block", "size", "text",
            "bool", "symbol", "not-object",
        ],
    )
    def test_invalid_values_skipped_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        data: object,
    ) -> None:
        """Verify out-of-range or wrongly typed values skip the file."""
        _write(tmp_path, "bad.json", data)
        with caplog.at_level(logging.WARNING):
            assert load_hazard_types(tmp_path) == {}
        assert "bad.json" in caplog.text
