"""Tests for crowd_evac.designer.importer -- scenario to design layout.

Covers:
  - every bundled scenario imports, re-exports and reloads with identical
    geometry and a room-only file
  - exit-to-wall matching by position (mislabelled sides), wall thickness
  - unrepresentable scenarios fail with a clear error
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crowd_evac.adapters.io.scenario_catalog import (
    SCENARIOS_DIR,
    list_bundled_scenarios,
)
from crowd_evac.adapters.io.scenario_loader import load_scenario_file
from crowd_evac.adapters.io.scenario_writer import write_scenario
from crowd_evac.designer.export import build_validated_scenario
from crowd_evac.designer.importer import layout_from_scenario
from crowd_evac.designer.model import DesignModel
from crowd_evac.domain.constants import APRON_DEPTH_M
from crowd_evac.domain.errors import DesignValidationError
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.scenarios.apron import add_aprons
from crowd_evac.scenarios.schema import ScenarioData

BUNDLED = list_bundled_scenarios()


def _exit_key(plan: FloorPlan) -> list[tuple[float, float, int]]:
    """Return sorted (along-wall coordinate, width, capacity) per exit."""
    keys = []
    for e in plan.exits:
        along = e.x if e.side in (ExitSide.NORTH, ExitSide.SOUTH) else e.y
        keys.append((along, e.width_m, e.capacity_per_second))
    return sorted(keys)


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_scenario_round_trips(name: str, tmp_path: Path) -> None:
    """Import -> export -> reload keeps geometry and writes a room-only file."""
    raw = json.loads(
        (SCENARIOS_DIR / f"{name}.json").read_text(encoding="utf-8")
    )
    migrated = tmp_path / "expected"
    migrated.mkdir()
    expected = write_scenario(add_aprons(raw), migrated)
    plan, data = load_scenario_file(expected)
    layout = layout_from_scenario(plan, data)
    saved = write_scenario(build_validated_scenario(layout), tmp_path)
    plan2, data2 = load_scenario_file(saved)

    assert (plan2.width_m, plan2.height_m) == (plan.width_m, plan.height_m)
    rect = lambda r: (r.x, r.y, r.width, r.height)  # noqa: E731
    assert sorted(map(rect, plan2.walls)) == sorted(map(rect, plan.walls))
    assert sorted(map(rect, plan2.obstacles)) == sorted(
        map(rect, plan.obstacles)
    )
    assert _exit_key(plan2) == _exit_key(plan)
    assert data2["name"] == data["name"]
    assert set(data2) == {"schema_version", "name", "floor_plan"}


def test_wall_rect_becomes_centreline_with_thickness() -> None:
    """Verify a boundary wall rectangle maps to centreline + thickness."""
    plan, data = load_scenario_file(SCENARIOS_DIR / "fixture_minimal.json")
    layout = layout_from_scenario(plan, data)
    top = next(w for w in layout.walls.values() if w.y0 == 7.75)
    assert (top.x0, top.x1, top.thickness) == (0.0, 10.0, 0.5)


def test_mislabelled_exit_side_is_matched_by_position(tmp_path: Path) -> None:
    """open_room_search labels a top-wall exit 'north'; it must stay on top."""
    plan, data = load_scenario_file(SCENARIOS_DIR / "open_room_search.json")
    layout = layout_from_scenario(plan, data)
    out = build_validated_scenario(layout)
    # Top wall's inner face sits below the 1 m apron added above it.
    top_face = out["floor_plan"]["height_m"] - APRON_DEPTH_M - 1.0
    sides_at_top = [
        e["side"]
        for e in out["floor_plan"]["exits"]
        if e["y"] >= top_face
    ]
    assert sides_at_top == ["south"]  # south == top wall in this codebase


def test_model_continues_ids_after_import() -> None:
    """Verify new entities never reuse an imported id."""
    plan, data = load_scenario_file(SCENARIOS_DIR / "fixture_minimal.json")
    layout = layout_from_scenario(plan, data)
    model = DesignModel(layout=layout)
    used = {*layout.walls, *layout.obstacles, *layout.doorways}
    new = model.add_obstacle(1.0, 1.0, 0.5, 0.5)
    assert new.id not in used


def test_exit_without_boundary_wall_is_rejected() -> None:
    """Verify an exit with no wall to attach to gives a clear error."""
    plan = FloorPlan(
        width_m=10.0,
        height_m=8.0,
        walls=(),
        obstacles=(),
        exits=(Exit(5.0, 0.25, 2.0, ExitSide.NORTH, 5, "lonely"),),
    )
    data: ScenarioData = {  # type: ignore[typeddict-item]
        "schema_version": "2.0",
        "name": "x",
        "floor_plan": {},  # type: ignore[typeddict-item]
    }
    with pytest.raises(DesignValidationError, match="lonely"):
        layout_from_scenario(plan, data)
