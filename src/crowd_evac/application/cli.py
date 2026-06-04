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

_SCREEN_MARGIN: float = 0.90
"""Fraction of each screen dimension available for the floor plan window.

The remaining 10 % is reserved for the OS taskbar and window chrome so the
window does not clip against screen edges.
"""

_MIN_PPM: float = 5.0
"""Hard lower bound on pixels-per-meter to keep geometry legible."""


def _get_logical_screen_size() -> tuple[float, float] | None:
    """Return the primary monitor size in logical (DPI-scaled) pixels.

    pyglet (and therefore arcade) sizes windows in *logical* coordinates, while
    :attr:`Screen.width`/``height`` report *physical* pixels.  On a display with
    DPI scaling these differ by :meth:`Screen.get_scale` (e.g. 1.5 at 150 %), so
    a window requested at the physical width would overflow the desktop.  This
    helper divides the physical size by the scale to give the usable logical
    extent a window can occupy.

    Returns:
        ``(logical_width, logical_height)`` in pixels, or ``None`` when no
        screen can be queried (headless CI, virtual framebuffers).
    """
    try:
        screen = arcade.get_screens()[0]
        scale = screen.get_scale()
        if scale <= 0.0:
            scale = 1.0
        return screen.width / scale, screen.height / scale
    except Exception:
        return None


def compute_fit_ppm(
    floor_plan: FloorPlan,
    *,
    default_ppm: float = PIXELS_PER_METER,
    margin: float = _SCREEN_MARGIN,
) -> float:
    """Compute pixels-per-meter so the entire floor plan fits on screen.

    Queries the primary monitor's *logical* (DPI-scaled) size via
    :func:`_get_logical_screen_size` and returns the largest ppm satisfying::

        floor_plan.width_m  * ppm <= logical_width  * margin
        floor_plan.height_m * ppm <= logical_height * margin

    Logical pixels are used because pyglet/arcade size windows and map their
    default camera in logical coordinates; sizing against physical pixels would
    overflow the desktop on any display with DPI scaling > 100 %.

    The result is capped at ``default_ppm`` so small floor plans are never
    upscaled beyond the configured default, and floored at :data:`_MIN_PPM` so
    geometry stays legible even on very small screens.

    Falls back to ``default_ppm`` silently when the display cannot be queried
    (headless CI, virtual framebuffers, etc.).

    Args:
        floor_plan: Floor plan whose physical dimensions drive the fit.
        default_ppm: Upper bound for the returned ppm; defaults to the
            module-level :data:`~crowd_evac.domain.constants.PIXELS_PER_METER`.
        margin: Fraction of each screen dimension available for the window
            (default :data:`_SCREEN_MARGIN`).

    Returns:
        Effective pixels-per-meter in ``[_MIN_PPM, default_ppm]``.

    Example:
        >>> ppm = compute_fit_ppm(floor_plan, default_ppm=40.0)  # doctest: +SKIP
        >>> ppm <= 40.0
        True
    """
    size = _get_logical_screen_size()
    if size is None:
        logger.debug("Cannot determine screen size; using default ppm %.1f.", default_ppm)
        return default_ppm
    screen_w, screen_h = size

    ppm_x = (screen_w * margin) / floor_plan.width_m
    ppm_y = (screen_h * margin) / floor_plan.height_m
    fit = min(ppm_x, ppm_y, default_ppm)
    result = max(fit, _MIN_PPM)
    if result < default_ppm:
        logger.info(
            "Floor plan %.1f×%.1f m exceeds screen at %.0f px/m on %.0f×%.0f "
            "logical display; scaling to %.2f px/m.",
            floor_plan.width_m,
            floor_plan.height_m,
            default_ppm,
            screen_w,
            screen_h,
            result,
        )
    return result


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

        The window size is computed via :func:`compute_fit_ppm` so the entire
        floor plan — including perimeter walls — is always fully visible within
        the primary monitor's bounds.

        Args:
            sim: Simulation to drive.
            floor_plan: Provides physical dimensions and geometry.
        """
        ppm = compute_fit_ppm(floor_plan)
        width_px = int(floor_plan.width_m * ppm)
        height_px = int(floor_plan.height_m * ppm)
        super().__init__(width_px, height_px, WINDOW_TITLE)
        self._sim = sim
        # Agent visual size is now proportional to pixels_per_meter, so it
        # automatically scales when ppm is adjusted to fit the floor plan on screen
        self._renderer = ArcadeRenderer(floor_plan, pixels_per_meter=ppm)
        self._input_source = ArcadeInputSource()
        self._input_source.register(self)
        self._current_source: PanicSource | None = None
        logger.info(
            "EvacWindow opened: %d×%d px at %.2f px/m (%.1f×%.1f m).",
            width_px,
            height_px,
            ppm,
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
