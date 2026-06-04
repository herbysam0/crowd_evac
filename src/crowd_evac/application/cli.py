"""Application entry point: load scenario, wire components, start game loop.

Usage::

    python -m crowd_evac       # launches the Lecture Hall default scenario
    crowd-evac                 # pip-installed console script (same entry point)

:func:`build_simulation_from_scenario` performs all domain + application
wiring without touching arcade or opening a window, so the full stack is
testable headless.  :func:`main` takes that result, opens an
:class:`EvacWindow`, and delegates to ``arcade.run()``.
"""
from __future__ import annotations

import logging

import arcade

from crowd_evac.adapters.io.scenario_loader import load_bundled_scenario
from crowd_evac.adapters.render.arcade_input import ArcadeInputSource
from crowd_evac.adapters.render.arcade_renderer import ArcadeRenderer
from crowd_evac.application.injection import process_input_events
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.constants import PIXELS_PER_METER
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.pathfinding.flow_field import FlowField

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO: str = "lecture_hall"
"""Bundled scenario name loaded when no user file is specified (FR-0 R0.1)."""

WINDOW_TITLE: str = "Crowd Evac"
"""Title bar text for the arcade window."""


def build_simulation_from_scenario(
    scenario_name: str = DEFAULT_SCENARIO,
) -> tuple[Simulation, FloorPlan]:
    """Load a bundled scenario and construct a fully-wired Simulation.

    No GL context is required; this function is safe to call headless.  The
    returned simulation is at tick 0 with agents spawned inside the walkable
    region using the scenario's ``spawn_seed``.

    Args:
        scenario_name: Bundled scenario name without the ``.json`` extension.
            Defaults to :data:`DEFAULT_SCENARIO` (Lecture Hall).

    Returns:
        A ``(sim, floor_plan)`` tuple.  The simulation owns all domain
        components (flow field, panic field, exit model, RNG) and is ready to
        :meth:`~Simulation.step` headless or to be handed to an
        :class:`EvacWindow`.

    Raises:
        MalformedScenarioError: If the bundled file is missing or malformed.
        ScenarioValidationError: If the scenario fails semantic validation.
        PathfindingError: If the floor plan has no walkable exit cell.

    Example:
        >>> sim, fp = build_simulation_from_scenario()
        >>> sim.tick
        0
    """
    floor_plan, scenario_data = load_bundled_scenario(scenario_name)
    seed: int = scenario_data["agents"]["spawn_seed"]
    count: int = scenario_data["agents"]["count"]
    rng = SeededRNG(seed)
    state = spawn(floor_plan, count, rng.generator)
    flow_field = FlowField.build(floor_plan)
    panic_field = PanicField()
    exit_model = ExitModel(floor_plan)
    collision_map = CollisionMap.from_floor_plan(floor_plan)
    sim = Simulation(
        state,
        flow_field,
        panic_field,
        exit_model,
        rng,
        collision_map=collision_map,
    )
    logger.info(
        "Simulation built: scenario=%r agents=%d seed=%d",
        scenario_name,
        count,
        seed,
    )
    return sim, floor_plan


class EvacWindow(arcade.Window):
    """Arcade window driving the fixed-step render/input loop (FR-0 R0.1).

    ``on_update`` advances the simulation by one tick and drains the mouse
    input queue each frame.  ``on_draw`` renders the current snapshot via
    :class:`~crowd_evac.adapters.render.arcade_renderer.ArcadeRenderer` and
    overlays a completion message once the evacuation ends.

    The window stays open after evacuation completes so the player can see
    the final state; close it manually to exit.

    Args:
        sim: Fully-wired simulation at tick 0.
        floor_plan: Static geometry for this scenario.  Used to size the window
            (``width_m × height_m × pixels_per_meter``) and passed to the
            renderer.
    """

    def __init__(self, sim: Simulation, floor_plan: FloorPlan) -> None:
        """Open a window sized to the floor plan and wire adapters.

        Args:
            sim: Simulation to drive.
            floor_plan: Provides physical dimensions and geometry.
        """
        width_px = int(floor_plan.width_m * PIXELS_PER_METER)
        height_px = int(floor_plan.height_m * PIXELS_PER_METER)
        super().__init__(width_px, height_px, WINDOW_TITLE)
        self._sim = sim
        self._renderer = ArcadeRenderer(floor_plan)
        self._input_source = ArcadeInputSource()
        self._input_source.register(self)
        self._current_source: PanicSource | None = None
        logger.info(
            "EvacWindow opened: %d×%d px (%.1f×%.1f m).",
            width_px,
            height_px,
            floor_plan.width_m,
            floor_plan.height_m,
        )

    def on_update(self, delta_time: float) -> None:
        """Advance the simulation by one tick and process pending mouse input.

        Drains :meth:`~ArcadeInputSource.poll` first so each tick sees the
        input events that arrived during the *previous* frame.  If the
        simulation has already completed no step is taken and the display
        remains frozen on the final state.

        Args:
            delta_time: Seconds since last frame (provided by arcade; the
                simulation ignores it and always advances by the fixed
                :data:`~crowd_evac.domain.constants.DT`).
        """
        if self._sim.is_complete:
            return
        events = self._input_source.poll()
        self._current_source = process_input_events(
            self._sim,
            events,
            self._current_source,
        )
        self._sim.step()

    def on_draw(self) -> None:
        """Render the current snapshot and overlay the completion message.

        Clears the framebuffer, draws all geometry and agents via the
        :class:`ArcadeRenderer`, then overlays centred white text once
        :attr:`~Simulation.is_complete` is ``True``.
        """
        self.clear()
        snapshot = self._sim.snapshot()
        self._renderer.render(snapshot)
        if self._sim.is_complete:
            arcade.draw_text(
                f"Evacuation complete — "
                f"{snapshot.evacuated_count} evacuated "
                f"in {snapshot.sim_time:.1f} s",
                self.width / 2,
                self.height / 2,
                arcade.color.WHITE,
                font_size=18,
                anchor_x="center",
            )


def main() -> None:
    """Launch the Lecture Hall default scenario and enter the event loop.

    Loads the bundled Lecture Hall, wires the simulation, opens
    :class:`EvacWindow`, and calls ``arcade.run()``.  Returns when the
    player closes the window.

    Raises:
        MalformedScenarioError: If the bundled Lecture Hall file is missing.
        PathfindingError: If the floor plan has no walkable exit.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    sim, floor_plan = build_simulation_from_scenario()
    EvacWindow(sim, floor_plan)
    arcade.run()
