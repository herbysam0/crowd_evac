"""Component tests for run-time agents/seed/hazards in crowd_evac.application.cli.

Covers:
  - build_simulation_from_scenario: honours agent_count and seed, rejects
    hazards outside the room, accepts hazards inside a wall.
  - EvacWindow: scripted hazards fire by sim tick (not before), and Reset
    rebuilds the same run (count, seed, hazard replay).
  - main: forwards --agents/--seed/--hazard; an out-of-room hazard is a
    usage error (exit status 2).
"""
from __future__ import annotations

import numpy as np
import pytest
from pytest_mock import MockerFixture

from crowd_evac.application import cli
from crowd_evac.application.cli import (
    EvacWindow,
    _SimState,
    build_simulation_from_scenario,
    main,
)
from crowd_evac.application.hazard_args import HazardOutOfRoomError
from crowd_evac.scenarios.schema import EmergencyEventData

# Small room: these tests check wiring, and building lecture_hall costs ~2.5 s
# per test (fine 0.05 m grid).
_ROOM = "fixture_minimal"
_COUNT = 5
_SEED = 7
_FIRE_TICK = 3
_FIRE: EmergencyEventData = {
    "tick": _FIRE_TICK,
    "type": "place_panic_source",
    "pos": [5.0, 4.0],
}


class TestBuildSimulation:
    """build_simulation_from_scenario takes the population as input."""

    @pytest.mark.parametrize("count", [1, 4, 11])
    def test_agent_count_is_honoured(self, count: int) -> None:
        """Verify the requested number of agents is spawned."""
        sim, _ = build_simulation_from_scenario(_ROOM, count, _SEED)
        assert sim.snapshot().positions.shape == (count, 2)

    def test_same_seed_same_positions(self) -> None:
        """Verify one seed reproduces the spawn exactly."""
        a, _ = build_simulation_from_scenario(_ROOM, _COUNT, _SEED)
        b, _ = build_simulation_from_scenario(_ROOM, _COUNT, _SEED)
        assert np.array_equal(a.snapshot().positions, b.snapshot().positions)

    def test_different_seed_different_positions(self) -> None:
        """Verify another seed spawns elsewhere."""
        a, _ = build_simulation_from_scenario(_ROOM, _COUNT, 1)
        b, _ = build_simulation_from_scenario(_ROOM, _COUNT, 2)
        assert not np.array_equal(
            a.snapshot().positions, b.snapshot().positions
        )

    def test_hazard_outside_room_raises(self) -> None:
        """Verify a hazard beyond the room is rejected."""
        outside: EmergencyEventData = {
            "tick": 0, "type": "place_panic_source", "pos": [500.0, 1.0],
        }
        with pytest.raises(HazardOutOfRoomError, match="outside"):
            build_simulation_from_scenario(_ROOM, _COUNT, _SEED, [outside])

    def test_hazard_inside_a_wall_is_accepted(self) -> None:
        """Verify a wall position is accepted, as for a mouse-placed source."""
        in_wall: EmergencyEventData = {
            "tick": 0, "type": "place_panic_source", "pos": [5.0, 0.25],
        }
        sim, _ = build_simulation_from_scenario(
            "fixture_minimal", _COUNT, _SEED, [in_wall]
        )
        assert sim.tick == 0

    def test_hazards_do_not_fire_at_build_time(self) -> None:
        """Verify events are armed later, not injected during build."""
        sim, _ = build_simulation_from_scenario(_ROOM, _COUNT, _SEED, [_FIRE])
        assert len(sim.panic_field.sources) == 0


@pytest.fixture
def window(mocker: MockerFixture) -> EvacWindow:
    """Return a headless EvacWindow running with a scripted fire."""
    mocker.patch("crowd_evac.application.cli.compute_fit_ppm", return_value=25.0)
    mocker.patch("crowd_evac.application.cli.ArcadeRenderer")
    mocker.patch("arcade.Window.__init__", return_value=None)
    sim, floor_plan = build_simulation_from_scenario(
        _ROOM, _COUNT, _SEED, [_FIRE]
    )
    return EvacWindow(sim, floor_plan, _ROOM, _COUNT, _SEED, [_FIRE])


def _run_updates(win: EvacWindow, n: int) -> None:
    """Advance a running window by ``n`` updates."""
    win._sim_state = _SimState.RUNNING
    for _ in range(n):
        win.on_update(0.0)


class TestScriptedHazardFiring:
    """Hazards fire by simulation tick, before that tick's step."""

    def test_not_fired_before_its_tick(self, window: EvacWindow) -> None:
        """Verify nothing fires on ticks 0..tick-1."""
        _run_updates(window, _FIRE_TICK)
        assert window._sim.tick == _FIRE_TICK
        assert len(window._sim.panic_field.sources) == 0

    def test_fires_at_its_tick(self, window: EvacWindow) -> None:
        """Verify the source exists once the sim reaches the tick."""
        _run_updates(window, _FIRE_TICK + 1)
        assert len(window._sim.panic_field.sources) == 1

    def test_fires_only_once(self, window: EvacWindow) -> None:
        """Verify later updates do not re-fire the event."""
        _run_updates(window, _FIRE_TICK + 5)
        assert len(window._sim.panic_field.sources) == 1

    def test_paused_window_does_not_fire(self, window: EvacWindow) -> None:
        """Verify nothing fires while paused at the start."""
        for _ in range(_FIRE_TICK + 2):
            window.on_update(0.0)
        assert window._sim.tick == 0
        assert len(window._sim.panic_field.sources) == 0

    def test_no_events_means_no_sources(self, mocker: MockerFixture) -> None:
        """Verify a plain run never gains a panic source by itself."""
        mocker.patch("crowd_evac.application.cli.ArcadeRenderer")
        mocker.patch("arcade.Window.__init__", return_value=None)
        sim, floor_plan = build_simulation_from_scenario(_ROOM, _COUNT, _SEED)
        win = EvacWindow(sim, floor_plan, _ROOM, _COUNT, _SEED)
        _run_updates(win, _FIRE_TICK + 2)
        assert len(win._sim.panic_field.sources) == 0


class TestReset:
    """Reset rebuilds the same run: count, seed and scripted hazard."""

    def test_reset_restores_count_and_seed(self, window: EvacWindow) -> None:
        """Verify the respawned crowd matches the original spawn."""
        initial = window._sim.snapshot().positions.copy()
        _run_updates(window, 10)
        window._reset()
        assert window._sim.tick == 0
        assert np.array_equal(window._sim.snapshot().positions, initial)

    def test_reset_clears_the_fired_hazard(self, window: EvacWindow) -> None:
        """Verify the fire is gone right after reset."""
        _run_updates(window, _FIRE_TICK + 2)
        window._reset()
        assert len(window._sim.panic_field.sources) == 0
        assert window._sim_state is _SimState.PAUSED_INITIAL

    def test_reset_replays_the_hazard(self, window: EvacWindow) -> None:
        """Verify the fire fires again at its tick after a reset."""
        _run_updates(window, _FIRE_TICK + 2)
        window._reset()
        _run_updates(window, _FIRE_TICK)
        assert len(window._sim.panic_field.sources) == 0
        _run_updates(window, 1)
        assert len(window._sim.panic_field.sources) == 1


class TestMainForwarding:
    """main() hands the run settings to the builder and the window."""

    def test_flags_reach_builder_and_window(self, mocker: MockerFixture) -> None:
        """Verify --agents, --seed and --hazard are forwarded intact."""
        build = mocker.patch.object(
            cli, "build_simulation_from_scenario",
            return_value=(object(), object()),
        )
        win = mocker.patch.object(cli, "EvacWindow")
        mocker.patch.object(cli.arcade, "run")
        main([
            "-s", "fixture_minimal", "--agents", "7", "--seed", "3",
            "--hazard", "x=1,y=1,tick=5",
        ])
        event = {"tick": 5, "type": "place_panic_source", "pos": [1.0, 1.0]}
        build.assert_called_once_with("fixture_minimal", 7, 3, [event])
        assert win.call_args.args[3:] == (7, 3, [event])

    def test_out_of_room_hazard_is_a_usage_error(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify an out-of-room hazard exits with status 2, no window."""
        win = mocker.patch.object(cli, "EvacWindow")
        with pytest.raises(SystemExit) as exc:
            main(["-s", "fixture_minimal", "--hazard", "x=999,y=1"])
        assert exc.value.code == 2
        assert "outside" in capsys.readouterr().err
        win.assert_not_called()
