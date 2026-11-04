"""Application entry point: load scenario, wire components, start game loop.

Usage::

    python -m crowd_evac                        # Lecture Hall, 3 agents, seed 42
    python -m crowd_evac -s NAME --agents N --seed S   # pick room + population
    python -m crowd_evac --hazard x=25,y=3.5,tick=100  # scripted fire (repeatable)
    python -m crowd_evac --list                 # show bundled scenario names
    crowd-evac                                  # pip-installed console script

:func:`build_simulation_from_scenario` performs all domain + application
wiring without touching arcade or opening a window, so the full stack is
testable headless.  :func:`main` takes that result, opens an
:class:`EvacWindow`, and delegates to ``arcade.run()``.
"""
from __future__ import annotations

import enum
import logging
from collections.abc import Sequence
from typing import Literal

import arcade

from crowd_evac.adapters.io.scenario_catalog import list_bundled_scenarios
from crowd_evac.adapters.io.scenario_loader import load_bundled_scenario
from crowd_evac.adapters.render.arcade_input import ArcadeInputSource
from crowd_evac.adapters.render.arcade_renderer import ArcadeRenderer
from crowd_evac.adapters.render.viewport import Viewport, fit_viewport
from crowd_evac.application.cli_args import (
    DEFAULT_SCENARIO,
    build_parser,
    parse_args,
)
from crowd_evac.application.hazard_args import (
    HazardOutOfRoomError,
    check_hazards_in_room,
)
from crowd_evac.application.injection import (
    apply_due_events,
    events_by_tick,
    process_input_events,
)
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.constants import (
    DEFAULT_AGENT_COUNT,
    DEFAULT_SPAWN_SEED,
    PIXELS_PER_METER,
)
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.pathfinding.flow_field import FlowField
from crowd_evac.scenarios.schema import EmergencyEventData

logger = logging.getLogger(__name__)

WINDOW_TITLE: str = "Crowd Evac"
"""Title bar text for the arcade window."""

_SCREEN_MARGIN: float = 0.90
"""Fraction of each screen dimension available for the floor plan window.

The remaining 10 % is reserved for the OS taskbar and window chrome so the
window does not clip against screen edges.
"""

_MIN_PPM: float = 5.0
"""Hard lower bound on pixels-per-meter to keep geometry legible."""

MIN_WINDOW_W_PX: int = 320
MIN_WINDOW_H_PX: int = 240
"""Smallest window the user can resize to; keeps hints and room legible."""


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
    agent_count: int = DEFAULT_AGENT_COUNT,
    seed: int = DEFAULT_SPAWN_SEED,
    events: Sequence[EmergencyEventData] = (),
) -> tuple[Simulation, FloorPlan]:
    """Load a bundled room and construct a fully-wired Simulation.

    No GL context is required; this function is safe to call headless.  The
    returned simulation is at tick 0 with ``agent_count`` agents spawned
    inside the walkable region from ``seed``.  ``events`` are validated here
    but fire later, by tick, in :meth:`EvacWindow.on_update`.

    Args:
        scenario_name: Bundled scenario name without the ``.json`` extension.
            Defaults to :data:`DEFAULT_SCENARIO` (Lecture Hall).
        agent_count: Number of agents to spawn.
        seed: Spawn RNG seed.
        events: Scripted hazards to validate against the room.

    Returns:
        A ``(sim, floor_plan)`` tuple.  The simulation owns all domain
        components (flow field, panic field, exit model, RNG) and is ready to
        :meth:`~Simulation.step` headless or to be handed to an
        :class:`EvacWindow`.

    Raises:
        MalformedScenarioError: If the bundled file is missing or malformed.
        ScenarioValidationError: If the scenario fails semantic validation.
        HazardOutOfRoomError: If a hazard lies outside the room.
        PathfindingError: If the floor plan has no walkable exit cell.

    Example:
        >>> sim, fp = build_simulation_from_scenario()
        >>> sim.tick
        0
    """
    floor_plan, _ = load_bundled_scenario(scenario_name)
    check_hazards_in_room(events, floor_plan)
    rng = SeededRNG(seed)
    state = spawn(floor_plan, agent_count, rng.generator)
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
        "Simulation built: scenario=%r agents=%d seed=%d hazards=%d",
        scenario_name,
        agent_count,
        seed,
        len(events),
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
        agent_count: Population to respawn on Reset.
        seed: Spawn seed to reuse on Reset, so the run repeats exactly.
        events: Scripted hazards fired by simulation tick; replayed after
            Reset.
    """

    def __init__(
        self,
        sim: Simulation,
        floor_plan: FloorPlan,
        scenario_name: str = DEFAULT_SCENARIO,
        agent_count: int = DEFAULT_AGENT_COUNT,
        seed: int = DEFAULT_SPAWN_SEED,
        events: Sequence[EmergencyEventData] = (),
    ) -> None:
        """Open a window sized to the floor plan and wire adapters.

        Args:
            sim: Simulation to drive.
            floor_plan: Provides physical dimensions and geometry.
            scenario_name: Scenario to reload on Reset.
            agent_count: Population to respawn on Reset.
            seed: Spawn seed to reuse on Reset.
            events: Scripted hazards to fire by tick.
        """
        ppm = compute_fit_ppm(floor_plan)
        width_px = int(floor_plan.width_m * ppm)
        height_px = int(floor_plan.height_m * ppm)
        # on_resize can fire inside super().__init__, before the adapters
        # below exist; it checks for them and returns early.
        super().__init__(width_px, height_px, WINDOW_TITLE, resizable=True)
        self.set_minimum_size(MIN_WINDOW_W_PX, MIN_WINDOW_H_PX)

        self._sim = sim
        self._scenario_name = scenario_name
        self._agent_count = agent_count
        self._seed = seed
        self._events = tuple(events)
        self._schedule = events_by_tick(self._events)
        self._ppm = ppm
        self._floor_plan = floor_plan
        self._sim_state: _SimState = _SimState.PAUSED_INITIAL
        self._current_source: PanicSource | None = None

        self._renderer = ArcadeRenderer(floor_plan, pixels_per_meter=ppm)
        self._input_source = ArcadeInputSource(pixels_per_meter=ppm)
        self._input_source.register(self)
        self._viewport = Viewport(ppm, 0.0, 0.0)

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
        from :attr:`_scenario_name` with the same agent count and seed (so the
        scripted hazards replay identically), clears any active panic source,
        re-registers the input source, and returns the state machine to
        :attr:`_SimState.PAUSED_INITIAL`.
        """
        sim, _ = build_simulation_from_scenario(
            self._scenario_name, self._agent_count, self._seed, self._events
        )
        self._sim = sim
        self._current_source = None
        self._input_source = ArcadeInputSource(pixels_per_meter=self._ppm)
        self._input_source.register(self)
        # The new source starts at the startup scale; re-apply the current one.
        self._apply_viewport()
        self._sim_state = _SimState.PAUSED_INITIAL
        logger.info(
            "EvacWindow: simulation reset from scenario=%r.", self._scenario_name
        )

    def _apply_viewport(self) -> None:
        """Push the current viewport to the renderer and input source."""
        vp = self._viewport
        fp = self._floor_plan
        self._ppm = vp.ppm
        self._renderer.set_viewport(vp.ppm, vp.ox, vp.oy)
        self._input_source.set_viewport(vp, fp.width_m, fp.height_m)

    # ------------------------------------------------------------------
    # Arcade callbacks
    # ------------------------------------------------------------------

    def on_resize(self, width: int, height: int) -> Literal[True] | None:
        """Refit the room to the new window size (presentation only).

        Uses a uniform scale so agents stay round and the room keeps its
        proportions; spare space becomes background bars. Simulation state
        is never touched.

        Args:
            width: New window width in pixels.
            height: New window height in pixels.

        Returns:
            Whatever the base class returns.
        """
        result = super().on_resize(width, height)
        if not hasattr(self, "_input_source"):
            return result
        fp = self._floor_plan
        self._viewport = fit_viewport(
            fp.width_m, fp.height_m, float(width), float(height)
        )
        self._apply_viewport()
        return result

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
        # Scripted hazards fire before step() so the source is active for this
        # tick's panic phase, matching the headless harness.
        apply_due_events(self._sim, self._schedule)
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


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the selected scenario and enter the event loop.

    Loads the bundled room chosen via ``--scenario`` (default Lecture Hall),
    spawns ``--agents`` agents from ``--seed``, arms any ``--hazard`` events,
    wires the simulation, opens :class:`EvacWindow`, and calls
    ``arcade.run()``.  With ``--list``, prints scenario names and returns
    without opening a window.  With no flags at all, opens the performance
    dashboard (imported lazily too).  With ``--design``, opens the room designer
    (imported lazily so normal runs never load it) and returns when it is
    closed.  Returns when the player closes the window.

    Args:
        argv: Argument list excluding the program name; ``None`` reads
            ``sys.argv``.

    Raises:
        MalformedScenarioError: If the bundled scenario file is malformed.
        PathfindingError: If the floor plan has no walkable exit.
    """
    args = parse_args(argv)
    if args.dashboard:
        # Lazy import, like the designer: the game must not load it.
        from crowd_evac.adapters.render.dashboard_window import run_dashboard

        run_dashboard()
        return
    if args.design:
        # Lazy import: the designer is a separate tool and must not slow or
        # break normal scenario runs.
        from crowd_evac.adapters.render.designer_window import run_designer

        run_designer()
        return
    if args.list:
        # CLI output is user-facing, not diagnostic logging.
        print("\n".join(list_bundled_scenarios()))
        return
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        sim, floor_plan = build_simulation_from_scenario(
            args.scenario, args.agents, args.seed, args.hazard
        )
    except HazardOutOfRoomError as exc:
        build_parser().error(str(exc))
    EvacWindow(
        sim, floor_plan, args.scenario, args.agents, args.seed, args.hazard
    )
    arcade.run()
