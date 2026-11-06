"""Tests for crowd_evac.scenarios.apron.add_aprons (schema 2.0 -> 2.1).

Covers:
  - EAST / NORTH growth (no shift), WEST / SOUTH growth (shift all geometry).
  - Apron walls leave exactly the door span open and enclose it.
  - Several doors, idempotency, input immutability, bad depth.
  - The migrated result loads and builds a flow field.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crowd_evac.adapters.io.scenario_loader import load_scenario_file
from crowd_evac.domain.floor_plan import FloorPlan
from crowd_evac.pathfinding.flow_field import FlowField
from crowd_evac.scenarios.apron import add_aprons
from crowd_evac.scenarios.schema import ScenarioData


def _scenario(*exits: dict[str, Any]) -> ScenarioData:
    """Return a 10 x 8 room with perimeter walls and the given exits."""
    return {
        "schema_version": "2.0",
        "name": "unit",
        "floor_plan": {
            "width_m": 10.0,
            "height_m": 8.0,
            "walls": [
                {"x": 0.0, "y": 0.0, "width": 10.0, "height": 0.5},
                {"x": 0.0, "y": 7.5, "width": 10.0, "height": 0.5},
                {"x": 0.0, "y": 0.0, "width": 0.5, "height": 8.0},
                {"x": 9.5, "y": 0.0, "width": 0.5, "height": 8.0},
            ],
            "obstacles": [{"x": 3.0, "y": 3.0, "width": 1.0, "height": 1.0}],
            "exits": list(exits),
        },  # type: ignore[typeddict-item]
    }


def _exit(x: float, y: float, side: str, width: float = 2.0) -> dict[str, Any]:
    """Return one exit entry."""
    return {
        "x": x, "y": y, "width_m": width, "side": side,
        "capacity_per_second": 1.05,
    }


EAST = _exit(9.5, 4.0, "east")
WEST = _exit(0.5, 4.0, "west")
SOUTH_WALL_Y0 = _exit(5.0, 0.5, "south")  # centre nearer y = 0
TOP = _exit(5.0, 7.5, "north")  # centre nearer y = height


def _apron_walls(data: ScenarioData) -> list[dict[str, Any]]:
    """Return the walls added by add_aprons."""
    return [
        w for w in data["floor_plan"]["walls"]
        if w.get("label", "").startswith("apron_wall_")
    ]


class TestAddAprons:
    """Geometry produced by add_aprons."""

    def test_east_grows_width_only(self) -> None:
        """Verify an east door widens the plan and shifts nothing."""
        out = add_aprons(_scenario(EAST))
        fp = out["floor_plan"]
        assert out["schema_version"] == "2.1"
        assert (fp["width_m"], fp["height_m"]) == (11.0, 8.0)
        assert fp["exits"][0]["x"] == 9.5
        assert fp["walls"][0]["x"] == 0.0

    def test_east_apron_walls_leave_door_span_open(self) -> None:
        """Verify the strip is two walls with a 3.0-5.0 gap."""
        walls = _apron_walls(add_aprons(_scenario(EAST)))
        spans = sorted((w["y"], w["y"] + w["height"]) for w in walls)
        assert spans == [(0.0, 3.0), (5.0, 8.0)]
        assert all(w["x"] == 10.0 and w["width"] == 1.0 for w in walls)

    def test_west_shifts_all_geometry(self) -> None:
        """Verify a west door shifts walls, obstacles and exits by +1 x."""
        base = _scenario(WEST)
        out = add_aprons(base)["floor_plan"]
        assert out["width_m"] == 11.0
        assert out["exits"][0]["x"] == 1.5
        assert out["obstacles"][0]["x"] == 4.0
        assert out["walls"][0]["x"] == 1.0
        assert out["walls"][0]["y"] == 0.0

    def test_y_zero_side_shifts_y(self) -> None:
        """Verify a door nearer y = 0 grows and shifts the y axis."""
        out = add_aprons(_scenario(SOUTH_WALL_Y0))["floor_plan"]
        assert out["height_m"] == 9.0
        assert out["exits"][0]["y"] == 1.5
        assert out["obstacles"][0]["y"] == 4.0
        gaps = sorted(
            (w["x"], w["x"] + w["width"])
            for w in out["walls"] if w.get("label", "").startswith("apron_wall_")
        )
        assert gaps == [(0.0, 4.0), (6.0, 10.0)]

    def test_top_side_grows_height_without_shift(self) -> None:
        """Verify a door nearer y = height grows height only."""
        out = add_aprons(_scenario(TOP))["floor_plan"]
        assert out["height_m"] == 9.0
        assert out["exits"][0]["y"] == 7.5

    def test_two_sides_no_corner_overlap(self) -> None:
        """Verify east + west doors give four apron walls, one per piece."""
        out = add_aprons(_scenario(EAST, WEST))["floor_plan"]
        assert (out["width_m"], out["height_m"]) == (12.0, 8.0)
        assert len(_apron_walls(add_aprons(_scenario(EAST, WEST)))) == 4

    def test_same_side_doors_share_strip(self) -> None:
        """Verify two east doors cut two gaps from one strip."""
        second = _exit(9.5, 1.5, "east", width=1.0)
        walls = _apron_walls(add_aprons(_scenario(EAST, second)))
        spans = sorted((w["y"], w["y"] + w["height"]) for w in walls)
        assert spans == [(0.0, 1.0), (2.0, 3.0), (5.0, 8.0)]

    def test_input_not_mutated(self) -> None:
        """Verify the source dict is unchanged."""
        base = _scenario(WEST)
        snapshot = copy.deepcopy(base)
        add_aprons(base)
        assert base == snapshot

    def test_idempotent_on_2_1(self) -> None:
        """Verify applying twice equals applying once."""
        once = add_aprons(_scenario(EAST, WEST))
        assert add_aprons(once) == once

    def test_non_positive_depth_raises(self) -> None:
        """Verify depth must be positive."""
        with pytest.raises(ValueError, match="depth"):
            add_aprons(_scenario(EAST), depth=0.0)

    def test_no_exits_adds_no_geometry(self) -> None:
        """Verify an exit-less plan keeps its size."""
        out = add_aprons(_scenario())["floor_plan"]
        assert (out["width_m"], out["height_m"]) == (10.0, 8.0)
        assert len(out["walls"]) == 4


class TestMigratedScenarioRuns:
    """A migrated scenario loads and routes agents through the apron."""

    def test_flow_field_leads_out_through_apron(self, tmp_path: Path) -> None:
        """Verify load + flow field build with goals in the apron."""
        path = tmp_path / "s.json"
        path.write_text(json.dumps(add_aprons(_scenario(EAST, WEST))), "utf-8")
        plan, data = load_scenario_file(path)
        assert isinstance(plan, FloorPlan)
        assert data["schema_version"] == "2.1"
        field = FlowField.build(plan)
        gx, gy = field.sample(np.array([[8.0, 4.0]]))[0]
        assert gx > 0.5 and abs(gy) < 0.5
