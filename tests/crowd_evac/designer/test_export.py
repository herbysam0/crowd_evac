"""Tests for crowd_evac.designer.export -- model to ScenarioData.

Covers:
  - side mapping (north=y0) and inner-face exit centres
  - end-placed doorways, schema fields, headless validation
  - round trip: save -> load -> simulate -> agents evacuate
"""
from __future__ import annotations

from pathlib import Path

import pytest

from crowd_evac.adapters.io.scenario_loader import load_scenario_file
from crowd_evac.adapters.io.scenario_writer import write_scenario
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.designer.export import (
    build_validated_scenario,
    export_scenario,
)
from crowd_evac.designer.model import DesignModel
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.errors import ScenarioValidationError
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.params import ForceParams
from crowd_evac.pathfinding.flow_field import FlowField

ROOM_W = 15.0
ROOM_H = 10.0
MAX_STEPS = 6000


@pytest.fixture
def model() -> DesignModel:
    """Provide an empty named design."""
    m = DesignModel()
    m.update_room(name="unit_room")
    return m


def _closed_room(model: DesignModel) -> dict[str, int]:
    """Add four perimeter walls; return their ids by side."""
    return {
        "north": model.add_wall(0, 0, ROOM_W, 0).id,
        "south": model.add_wall(0, ROOM_H, ROOM_W, ROOM_H).id,
        "west": model.add_wall(0, 0, 0, ROOM_H).id,
        "east": model.add_wall(ROOM_W, 0, ROOM_W, ROOM_H).id,
    }


class TestExportFields:
    """Schema fields and constants."""

    def test_header_fields(self, model: DesignModel) -> None:
        """Verify version, name, agents and simulation block."""
        data = export_scenario(model.layout)
        assert data["schema_version"] == "1.0"
        assert data["name"] == "unit_room"
        assert data["agents"] == {"count": 50, "spawn_seed": 42}
        assert data["simulation"]["dt"] == 0.05
        assert data["simulation"]["max_ticks"] > 0

    def test_empty_design_exports_empty_lists(
        self, model: DesignModel
    ) -> None:
        """Verify an empty design exports (headless check rejects later)."""
        fp = export_scenario(model.layout)["floor_plan"]
        assert fp["walls"] == [] and fp["obstacles"] == []
        assert fp["exits"] == []

    def test_walls_and_obstacles_exported(self, model: DesignModel) -> None:
        """Verify rectangles and labels are written."""
        model.add_wall(0, 0, ROOM_W, 0)
        model.add_obstacle(5, 4, 2, 1)
        fp = export_scenario(model.layout)["floor_plan"]
        assert fp["walls"][0]["y"] == 0.0
        assert fp["walls"][0]["height"] == 0.5
        assert fp["obstacles"][0]["width"] == 2
        assert fp["obstacles"][0]["label"].startswith("obstacle_")


class TestThicknessExport:
    """Wall thickness reaches the JSON and exit centres follow it."""

    def test_wall_height_is_thickness(self, model: DesignModel) -> None:
        """Verify a thick bottom wall exports its thickness."""
        model.add_wall(0, 0, ROOM_W, 0, thickness=0.8)
        wall = export_scenario(model.layout)["floor_plan"]["walls"][0]
        assert wall["height"] == 0.8

    def test_exit_centre_on_inner_face_of_thick_wall(
        self, model: DesignModel
    ) -> None:
        """Verify the exit follows the wall's inner face."""
        wall = model.add_wall(0, 0, ROOM_W, 0, thickness=0.8)
        model.add_doorway(wall.id, 5.0, 2.0)
        exit_ = export_scenario(model.layout)["floor_plan"]["exits"][0]
        assert exit_["y"] == 0.8


class TestExitMapping:
    """north=y0 bottom wall; exit centre on the inner face."""

    @pytest.mark.parametrize(
        "side, expected_xy",
        [
            ("north", (5.0, 0.5)),
            ("south", (5.0, ROOM_H - 0.5)),
            ("west", (0.5, 5.0)),
            ("east", (ROOM_W - 0.5, 5.0)),
        ],
    )
    def test_exit_side_and_inner_face(
        self,
        model: DesignModel,
        side: str,
        expected_xy: tuple[float, float],
    ) -> None:
        """Verify side string and inner-face centre for all four walls."""
        walls = _closed_room(model)
        model.add_doorway(walls[side], 5.0, 2.0, 12)
        exit_ = export_scenario(model.layout)["floor_plan"]["exits"][0]
        assert exit_["side"] == side
        assert (exit_["x"], exit_["y"]) == expected_xy
        assert exit_["width_m"] == 2.0
        assert exit_["capacity_per_second"] == 12

    def test_doorway_at_wall_end_exports_outward_centre(
        self, model: DesignModel
    ) -> None:
        """Verify an opening starting at a wall end uses end + width/2."""
        wall = model.add_wall(0, 0, 6, 0)
        model.add_doorway(wall.id, 7.0, 2.0)
        exit_ = export_scenario(model.layout)["floor_plan"]["exits"][0]
        assert exit_["x"] == 7.0 and exit_["side"] == "north"


class TestHeadlessCheck:
    """build_validated_scenario failure paths."""

    def test_no_doorway_raises_validation_error(
        self, model: DesignModel
    ) -> None:
        """Verify a design without doorway cannot be validated."""
        _closed_room(model)
        with pytest.raises(ScenarioValidationError):
            build_validated_scenario(model.layout)

    def test_valid_design_passes(self, model: DesignModel) -> None:
        """Verify a closed room with a doorway validates."""
        walls = _closed_room(model)
        model.add_doorway(walls["east"], 5.0, 2.0)
        assert build_validated_scenario(model.layout)["name"] == "unit_room"


class TestRoundTrip:
    """Saved design loads and evacuates in the real simulator."""

    def test_saved_design_evacuates(
        self, model: DesignModel, tmp_path: Path
    ) -> None:
        """Verify designed room -> JSON -> loader -> simulation evacuates."""
        walls = _closed_room(model)
        model.add_doorway(walls["east"], 5.0, 3.0)
        model.add_obstacle(6, 4, 1, 2)
        path = write_scenario(build_validated_scenario(model.layout), tmp_path)

        plan, data = load_scenario_file(path)
        rng = SeededRNG(data["agents"]["spawn_seed"])
        state = spawn(plan, data["agents"]["count"], rng.generator)
        sim = Simulation(
            state=state,
            flow_field=FlowField.build(plan),
            panic_field=PanicField(),
            exit_model=ExitModel(plan),
            rng=rng,
            params=ForceParams(),
            collision_map=CollisionMap.from_floor_plan(plan),
        )
        for _ in range(MAX_STEPS):
            if sim.is_complete:
                break
            sim.step()
        assert sim.snapshot().evacuated_count == data["agents"]["count"]
