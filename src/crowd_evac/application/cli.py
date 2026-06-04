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

import enum
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


class _SimState(enum.Enum):
    """State machine governing simulation playback via the spacebar.

    State transitions:
      PAUSED_INITIAL → RUNNING     (player presses SPACE — "start")
      RUNNING        → PAUSED_MID  (player presses SPACE — "pause")
      PAUSED_MID     → RUNNING     (player presses SPACE — "continue")
      RUNNING        → COMPLETE    (evacuation finishes automatically)
      COMPLETE       → PAUSED_INITIAL (player presses SPACE — "reset")
    """

    PAUSED_INITIAL = "paused_initial"
    RUNNING = "running"
    PAUSED_MID = "paused_mid"
    COMPLETE = "complete"


_STATE_HINT: dict[_SimState, str] = {
    _SimState.PAUSED_INITIAL: "Press SPACE to start",
    _SimState.PAUSED_MID: "Press SPACE to continue",
    _SimState.COMPLETE: "Press SPACE to reset",
}
"""On-screen hint shown when the simulation is not running."""


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

    The window starts **paused**: the simulation does not advance until the
    player presses **SPACE**.  The spacebar cycles through the states defined
    by :class:`_SimState`:

    * ``SPACE`` (initial) → starts the simulation (PAUSED_INITIAL → RUNNING).
    * ``SPACE`` (running) → freezes it mid-run (RUNNING → PAUSED_MID).
    * ``SPACE`` (paused)  → resumes from mid-run pause (PAUSED_MID → RUNNING).
    * ``SPACE`` (complete) → rebuilds the simulation and returns to the paused
      initial state (COMPLETE → PAUSED_INITIAL).

    An on-screen hint (``"Press SPACE to …"``) is shown in all non-running
    states and hidden while the simulation advances.

    Args:
        sim: Fully-wired simulation at tick 0.
        floor_plan: Static geometry for this scenario.  Used to size the window
            (``width_m × height_m × pixels_per_meter``).
        scenario_name: Bundled scenario name used to rebuild the simulation on
            Reset.  Defaults to :data:`DEFAULT_SCENARIO`.
    """

    def __init__(
        self,
        sim: Simulation,
        floor_plan: FloorPlan,
        scenario_name: str = DEFAULT_SCENARIO,
    ) -> None:
        """Open a window sized to the floor plan and wire adapters.

        Args:
            sim: Simulation to drive.
            floor_plan: Provides physical dimensions and geometry.
            scenario_name: Scenario to reload on Reset.
        """
        ppm = compute_fit_ppm(floor_plan)
        width_px = int(floor_plan.width_m * ppm)
        height_px = int(floor_plan.height_m * ppm)
        super().__init__(width_px, height_px, WINDOW_TITLE)

        self._sim = sim
        self._scenario_name = scenario_name
        self._ppm = ppm
        self._floor_plan = floor_plan
        self._sim_state: _SimState = _SimState.PAUSED_INITIAL
        self._current_source: PanicSource | None = None

        self._renderer = ArcadeRenderer(floor_plan, pixels_per_meter=ppm)
        self._input_source = ArcadeInputSource(pixels_per_meter=ppm)
        self._input_source.register(self)

        logger.info(
            "EvacWindow opened: %d×%d px at %.2f px/m (%.1f×%.1f m).",
            width_px,
            height_px,
            ppm,
            floor_plan.width_m,
            floor_plan.height_m,
        )

    # ------------------------------------------------------------------
    # State-machine helpers
    # ------------------------------------------------------------------

    def _handle_spacebar(self) -> None:
        """Advance the simulation state machine on spacebar press.

        Transitions:
          PAUSED_INITIAL → RUNNING     (start)
          RUNNING        → PAUSED_MID  (pause)
          PAUSED_MID     → RUNNING     (continue)
          COMPLETE       → PAUSED_INITIAL (reset — rebuilds sim)
        """
        if self._sim_state == _SimState.PAUSED_INITIAL:
            self._sim_state = _SimState.RUNNING
        elif self._sim_state == _SimState.RUNNING:
            self._sim_state = _SimState.PAUSED_MID
        elif self._sim_state == _SimState.PAUSED_MID:
            self._sim_state = _SimState.RUNNING
        elif self._sim_state == _SimState.COMPLETE:
            self._reset()
            return
        logger.info("EvacWindow: state → %s", self._sim_state.value)

    def _reset(self) -> None:
        """Rebuild the simulation from the original scenario.

        Creates a fresh :class:`~crowd_evac.application.simulation.Simulation`
        loaded from :attr:`_scenario_name`, clears any active panic source,
        re-registers the input source, and returns the state machine to
        :attr:`_SimState.PAUSED_INITIAL`.
        """
        sim, _ = build_simulation_from_scenario(self._scenario_name)
        self._sim = sim
        self._current_source = None
        self._input_source = ArcadeInputSource(pixels_per_meter=self._ppm)
        self._input_source.register(self)
        self._sim_state = _SimState.PAUSED_INITIAL
        logger.info(
            "EvacWindow: simulation reset from scenario=%r.", self._scenario_name
        )

    # ------------------------------------------------------------------
    # Arcade callbacks
    # ------------------------------------------------------------------

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """Handle spacebar to start, pause, continue, or reset the simulation.

        Args:
            symbol: Key symbol constant from ``arcade.key``.
            modifiers: Keyboard modifier bitmask (unused).
        """
        if symbol == arcade.key.SPACE:
            self._handle_spacebar()

    def on_update(self, delta_time: float) -> None:
        """Advance the simulation when running; detect evacuation completion.

        Does nothing unless the state is :attr:`_SimState.RUNNING`.  When
        :attr:`~Simulation.is_complete` becomes ``True`` mid-run, transitions
        automatically to :attr:`_SimState.COMPLETE`.

        Args:
            delta_time: Seconds since last frame (provided by arcade; the
                simulation ignores it and always advances by the fixed
                :data:`~crowd_evac.domain.constants.DT`).
        """
        if self._sim_state != _SimState.RUNNING:
            return
        if self._sim.is_complete:
            self._sim_state = _SimState.COMPLETE
            logger.info("EvacWindow: evacuation complete — state → complete.")
            return
        events = self._input_source.poll()
        self._current_source = process_input_events(
            self._sim,
            events,
            self._current_source,
        )
        self._sim.step()

    def on_draw(self) -> None:
        """Render the simulation snapshot and on-screen state hint.

        Drawing order:
          1. Clear framebuffer.
          2. Simulation snapshot.
          3. Completion overlay text (only when ``COMPLETE``).
          4. State hint at window bottom (hidden while running).
        """
        self.clear()
        snapshot = self._sim.snapshot()
        self._renderer.render(snapshot)
        if self._sim_state == _SimState.COMPLETE:
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
        hint = _STATE_HINT.get(self._sim_state)
        if hint is not None:
            arcade.draw_text(
                hint,
                self.width / 2,
                20,
                arcade.color.LIGHT_GRAY,
                font_size=14,
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
    sim, floor_plan = build_simulation_from_scenario(DEFAULT_SCENARIO)
    EvacWindow(sim, floor_plan, DEFAULT_SCENARIO)
    arcade.run()
