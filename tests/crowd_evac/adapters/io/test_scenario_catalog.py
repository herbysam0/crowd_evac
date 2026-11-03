"""Tests for crowd_evac.adapters.io.scenario_catalog."""
from __future__ import annotations

from pathlib import Path

import pytest

from crowd_evac.adapters.io.scenario_catalog import list_bundled_scenarios
from crowd_evac.adapters.io.scenario_loader import load_bundled_scenario


class TestListBundledScenarios:
    """Suite for list_bundled_scenarios."""

    def test_includes_default_scenario(self) -> None:
        """Verify the Lecture Hall default is listed."""
        assert "lecture_hall" in list_bundled_scenarios()

    def test_names_have_no_extension_and_are_sorted(self) -> None:
        """Verify names are bare and in sorted order."""
        names = list_bundled_scenarios()
        assert names == sorted(names)
        assert all(not n.endswith(".json") for n in names)

    @pytest.mark.parametrize("name", list_bundled_scenarios())
    def test_every_listed_name_is_loadable(self, name: str) -> None:
        """Verify each listed name round-trips through the loader."""
        floor_plan, _ = load_bundled_scenario(name)
        assert floor_plan.width_m > 0


def test_bundled_scenario_loads_from_foreign_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify bundled scenarios resolve regardless of the working directory."""
    monkeypatch.chdir(tmp_path)
    assert "lecture_hall" in list_bundled_scenarios()
    floor_plan, _ = load_bundled_scenario("lecture_hall")
    assert floor_plan is not None
