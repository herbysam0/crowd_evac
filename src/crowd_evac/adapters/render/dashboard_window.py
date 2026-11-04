"""Arcade window of the performance dashboard (``python -m crowd_evac``).

Holds the settings panel, hazard list, results text and chart.  A run
starts a :class:`~crowd_evac.dashboard.batch_process.BatchProcess` child;
``on_update`` drains its events, so the window never blocks, also while a
visual batch plays in the child's own window.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import arcade
import arcade.gui as gui

from crowd_evac.adapters.io.hazard_catalog import load_hazard_types
from crowd_evac.adapters.io.scenario_catalog import list_bundled_scenarios
from crowd_evac.adapters.io.scenario_loader import load_bundled_scenario
from crowd_evac.adapters.render.dashboard_chart import ChartView
from crowd_evac.adapters.render.dashboard_hazards import (
    LIST_WIDTH_PX,
    HazardList,
)
from crowd_evac.adapters.render.dashboard_panel import DashboardPanel
from crowd_evac.adapters.render.dashboard_results import (
    HEIGHT_PX as RESULTS_HEIGHT_PX,
)
from crowd_evac.adapters.render.dashboard_results import ResultsPanel
from crowd_evac.adapters.render.designer_view import fit_window_size
from crowd_evac.adapters.render.designer_widgets import CONTENT_WIDTH_PX
from crowd_evac.dashboard.batch_process import (
    BatchEvent,
    BatchFailed,
    BatchFinished,
    BatchProcess,
    RunFinished,
)
from crowd_evac.dashboard.form import (
    FormError,
    build_config,
    describe_errors,
    room_errors,
)
from crowd_evac.dashboard.results_model import build_view
from crowd_evac.domain.errors import ScenarioError
from crowd_evac.montecarlo.config import DEFAULT_MC_SCENARIO, MonteCarloConfig
from crowd_evac.montecarlo.results import RunOutcome, build_curves
from crowd_evac.montecarlo.runner import resolve_base_seed

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Crowd Evac - Performance Dashboard"
BACKGROUND_COLOR = (24, 24, 30, 255)
DESIRED_SIZE = (1400, 800)
MIN_SIZE = (1100, 600)
MARGIN_PX = 12
COLUMN_GAP_PX = 16
CANCELLING_TEXT = "Cancelling..."
CHART_X_PX = MARGIN_PX + CONTENT_WIDTH_PX + COLUMN_GAP_PX + LIST_WIDTH_PX + (
    COLUMN_GAP_PX
)


class DashboardWindow(arcade.Window):
    """Home screen: configure and run one scenario or a Monte Carlo batch."""

    def __init__(self) -> None:
        """Open the window with default settings."""
        super().__init__(*DESIRED_SIZE, WINDOW_TITLE, resizable=True)
        self._fit_to_screen()
        self.set_minimum_size(*MIN_SIZE)
        self.background_color = BACKGROUND_COLOR
        self.scenarios = list_bundled_scenarios()
        self.hazards = HazardList(load_hazard_types())
        self.results = ResultsPanel()
        self.chart = ChartView()
        self.panel = DashboardPanel(
            self.scenarios, DEFAULT_MC_SCENARIO, self.run_single,
            self.run_batch, self.cancel, self._on_scenario,
        )
        self._batch: BatchProcess | None = None
        self._outcomes: list[RunOutcome] = []
        self._total = 1
        self._single = True
        self._seed = (0, False)
        self._cancelling = False
        self.manager = gui.UIManager()
        self.manager.add(self._layout())
        self.manager.enable()
        self._on_scenario(self.panel.scenario)

    def _fit_to_screen(self) -> None:
        """Shrink the window so it lies fully on the display."""
        screen = self.screen
        if screen is None:
            return
        size = fit_window_size(
            (screen.width, screen.height), self.get_pixel_ratio(),
            DESIRED_SIZE,
        )
        self.set_size(*size)
        self.dispatch_events()

    def _layout(self) -> gui.UIAnchorLayout:
        """Place the settings column and the hazard list; chart is drawn."""
        root = gui.UIAnchorLayout()
        root.add(self.panel.widget, anchor_x="left", anchor_y="top",
                 align_x=MARGIN_PX, align_y=-MARGIN_PX)
        root.add(self.results.widget, anchor_x="left", anchor_y="top",
                 align_x=CHART_X_PX, align_y=-MARGIN_PX)
        root.add(
            self.hazards.widget, anchor_x="left", anchor_y="top",
            align_x=MARGIN_PX + CONTENT_WIDTH_PX + COLUMN_GAP_PX,
            align_y=-MARGIN_PX,
        )
        return root

    # -- Room hint ---------------------------------------------------------

    def _on_scenario(self, name: str) -> None:
        """Show the room size and centre new hazards in it."""
        try:
            plan, _ = load_bundled_scenario(name)
        except ScenarioError as exc:
            self.panel.set_hint("Room cannot be loaded")
            logger.warning("Cannot load %s: %s", name, exc)
            return
        self.panel.set_hint(f"Room: {plan.width_m:g} x {plan.height_m:g} m")
        self.hazards.set_room_center(plan.width_m / 2, plan.height_m / 2)

    # -- Starting and stopping --------------------------------------------

    @property
    def running(self) -> bool:
        """Whether a batch is active."""
        return self._batch is not None and self._batch.active

    def run_single(self) -> None:
        """Run the form's scenario once, with the exact seed."""
        self._start(monte_carlo=False)

    def run_batch(self) -> None:
        """Run the form's scenario as a Monte Carlo batch."""
        self._start(monte_carlo=True)

    def _start(self, *, monte_carlo: bool) -> None:
        """Validate the form and start a child process."""
        if self.running:
            return
        config = self._form_config(monte_carlo)
        if config is None:
            return
        seed, was_random = resolve_base_seed(config.seed)
        config = replace(config, seed=seed)
        self._seed = (seed, was_random)
        self._single = not monte_carlo
        self._total = config.runs
        self._outcomes = []
        self._cancelling = False
        self.chart.set_curves(build_curves([]))
        self._refresh_results(finished=False)
        self.panel.set_status("")
        self.panel.set_running(True)
        self._batch = BatchProcess(config)
        self._batch.start()

    def _form_config(self, monte_carlo: bool) -> MonteCarloConfig | None:
        """Build the config or show why the form is invalid."""
        fields = self.panel.read_fields(self.hazards.read())
        try:
            config = build_config(
                fields, monte_carlo=monte_carlo, scenarios=self.scenarios
            )
        except FormError as exc:
            self.panel.set_status(describe_errors(exc.errors))
            return None
        errors = room_errors(config)
        if errors:
            self.panel.set_status(describe_errors(errors))
            return None
        return config

    def cancel(self) -> None:
        """Ask the batch to stop; finished runs are kept."""
        if self._batch is None or not self.running:
            return
        self._batch.cancel()
        self._cancelling = True
        self.results.set_status(CANCELLING_TEXT)

    # -- Events ------------------------------------------------------------

    def on_update(self, delta_time: float) -> None:
        """Apply the events the child process sent since the last frame."""
        if self._batch is None:
            return
        for event in self._batch.poll():
            self._apply(event)

    def _apply(self, event: BatchEvent) -> None:
        """Update results, chart and buttons for one event."""
        if isinstance(event, RunFinished):
            self._outcomes.append(event.outcome)
            self._outcomes.sort(key=lambda o: o.run_index)
            self.chart.set_curves(build_curves(self._outcomes))
            self._refresh_results(finished=False)
        elif isinstance(event, BatchFinished):
            result = event.result
            self._outcomes = list(result.outcomes)
            self.chart.set_curves(result.curves)
            self._refresh_results(finished=True, cancelled=result.cancelled)
            self.panel.set_running(False)
        elif isinstance(event, BatchFailed):
            self.panel.set_status(f"Batch failed: {event.message}")
            self.panel.set_running(False)

    def _refresh_results(
        self, *, finished: bool, cancelled: bool = False
    ) -> None:
        """Rebuild the results text from the outcomes so far."""
        seed, was_random = self._seed
        self.results.show(build_view(
            self._outcomes, total_runs=self._total, single=self._single,
            base_seed=seed, seed_was_random=was_random,
            finished=finished, cancelled=cancelled,
        ))
        if self._cancelling and not finished:
            self.results.set_status(CANCELLING_TEXT)

    # -- Window ------------------------------------------------------------

    def on_draw(self) -> None:
        """Draw the chart and the widgets."""
        self.clear()
        self.chart.set_rect(
            CHART_X_PX, MARGIN_PX,
            self.width - CHART_X_PX - MARGIN_PX,
            self.height - 2 * MARGIN_PX - RESULTS_HEIGHT_PX,
        )
        self.chart.draw()
        self.manager.draw()

    def on_close(self) -> None:
        """Stop the batch and close at once; the child ends by itself."""
        if self._batch is not None and self.running:
            self._batch.cancel()
        super().on_close()


def run_dashboard() -> None:
    """Open the dashboard and block until its window is closed.

    Ctrl+C in the terminal closes the window the same way the close button
    does: the batch is cancelled and no traceback is printed.
    """
    window = DashboardWindow()
    try:
        arcade.run()
    except KeyboardInterrupt:
        logger.info("Dashboard interrupted; closing.")
        window.on_close()
