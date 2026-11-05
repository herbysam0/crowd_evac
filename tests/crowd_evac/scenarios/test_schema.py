"""Tests for crowd_evac.scenarios.schema TypedDicts.

Verifies that the schema module exports all required types and that
each TypedDict can be instantiated with the documented fields.
"""
from __future__ import annotations

from crowd_evac.scenarios.schema import (
    EmergencyEventData,
    ExitData,
    FloorPlanData,
    ObstacleData,
    ScenarioData,
    WallData,
)


class TestWallData:
    """Test WallData TypedDict structure."""

    def test_wall_data_has_required_fields(self) -> None:
        """Verify WallData can be constructed with all required fields."""
        wall: WallData = {"x": 0.0, "y": 0.0, "width": 5.0, "height": 0.5}
        assert wall["x"] == 0.0
        assert wall["width"] == 5.0

    def test_wall_data_accepts_optional_label(self) -> None:
        """Verify WallData accepts the optional label field."""
        wall: WallData = {
            "x": 0.0, "y": 0.0, "width": 5.0, "height": 0.5, "label": "south"
        }
        assert wall["label"] == "south"

    def test_wall_data_label_not_required(self) -> None:
        """Verify WallData is valid without a label field."""
        wall: WallData = {"x": 1.0, "y": 1.0, "width": 2.0, "height": 0.5}
        assert "label" not in wall


class TestExitData:
    """Test ExitData TypedDict structure."""

    def test_exit_data_has_required_fields(self) -> None:
        """Verify ExitData can be constructed with all required fields."""
        exit_: ExitData = {
            "x": 5.0,
            "y": 0.25,
            "width_m": 1.5,
            "side": "south",
            "capacity_per_second": 5,
        }
        assert exit_["capacity_per_second"] == 5
        assert exit_["side"] == "south"

    def test_exit_data_capacity_accepts_fraction(self) -> None:
        """Verify capacity_per_second may be fractional (persons/s)."""
        exit_: ExitData = {
            "x": 5.0, "y": 0.0, "width_m": 2.0,
            "side": "north", "capacity_per_second": 1.05,
        }
        assert exit_["capacity_per_second"] == 1.05

    def test_exit_data_accepts_all_sides(self) -> None:
        """Verify all four cardinal sides are accepted values."""
        for side in ("north", "south", "east", "west"):
            exit_: ExitData = {
                "x": 0.0, "y": 0.0, "width_m": 1.0,
                "side": side,  # type: ignore[typeddict-item]
                "capacity_per_second": 1,
            }
            assert exit_["side"] == side


class TestFloorPlanData:
    """Test FloorPlanData TypedDict structure."""

    def test_floor_plan_data_has_required_fields(self) -> None:
        """Verify FloorPlanData can be fully constructed."""
        fp: FloorPlanData = {
            "width_m": 10.0,
            "height_m": 8.0,
            "walls": [],
            "obstacles": [],
            "exits": [
                {"x": 5.0, "y": 0.0, "width_m": 2.0,
                 "side": "south", "capacity_per_second": 5}
            ],
        }
        assert fp["width_m"] == 10.0
        assert len(fp["exits"]) == 1

    def test_floor_plan_data_walls_list(self) -> None:
        """Verify walls field is a list type."""
        fp: FloorPlanData = {
            "width_m": 5.0,
            "height_m": 5.0,
            "walls": [{"x": 0.0, "y": 0.0, "width": 5.0, "height": 0.5}],
            "obstacles": [],
            "exits": [{"x": 2.5, "y": 0.0, "width_m": 1.0,
                        "side": "south", "capacity_per_second": 2}],
        }
        assert isinstance(fp["walls"], list)
        assert len(fp["walls"]) == 1


class TestScenarioData:
    """Test ScenarioData root TypedDict."""

    def test_scenario_data_holds_room_only(self) -> None:
        """Verify ScenarioData is version, name and floor plan, nothing else."""
        scenario: ScenarioData = {
            "schema_version": "2.0",
            "name": "test",
            "floor_plan": {
                "width_m": 5.0,
                "height_m": 5.0,
                "walls": [],
                "obstacles": [],
                "exits": [{"x": 2.5, "y": 0.0, "width_m": 1.0,
                            "side": "south", "capacity_per_second": 2}],
            },
        }
        assert scenario["schema_version"] == "2.0"
        assert set(scenario) == {"schema_version", "name", "floor_plan"}

    def test_agent_and_sim_types_are_gone(self) -> None:
        """Verify the removed run-setting types no longer exist."""
        from crowd_evac.scenarios import schema

        assert not hasattr(schema, "AgentConfigData")
        assert not hasattr(schema, "SimConfigData")

    def test_emergency_event_optional_fields(self) -> None:
        """Verify EmergencyEventData needs only tick, type and pos."""
        event: EmergencyEventData = {
            "tick": 100, "type": "place_panic_source", "pos": [25.0, 3.5],
        }
        assert event["pos"] == [25.0, 3.5]
        assert "intensity" not in event

    def test_obstacle_data_mirrors_wall_data(self) -> None:
        """Verify ObstacleData has the same required fields as WallData."""
        obs: ObstacleData = {"x": 2.0, "y": 2.0, "width": 1.0, "height": 0.5}
        wall: WallData = {"x": 2.0, "y": 2.0, "width": 1.0, "height": 0.5}
        assert dict(obs) == dict(wall)
