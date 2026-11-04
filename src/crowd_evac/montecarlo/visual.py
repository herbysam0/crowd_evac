"""Play a Monte Carlo batch in one arcade window (``--visual``).

Imported on demand by the runner so headless batches never load arcade.
The window starts each run itself, ignores the mouse and SPACE, and records
the same exits-per-tick series the headless worker does, so a visual batch
returns the same outcomes as a headless one.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import arcade

from crowd_evac.application.cli import (
    EvacWindow,
    build_simulation_from_scenario,
)
from crowd_evac.application.injection import apply_due_events, events_by_tick
from crowd_evac.metrics.records import make_record
from crowd_evac.metrics.speed import MovingSpeedTally
from crowd_evac.montecarlo.results import RunOutcome, make_outcome
from crowd_evac.montecarlo.runner import RunTask
from crowd_evac.optimization.harness import DEFAULT_MAX_TICKS

_LABEL_MARGIN_PX: int = 10
_LABEL_FONT_SIZE: int = 14


def _bundled_name(task: RunTask) -> str:
    """Return the task's room name; visual runs use bundled rooms only."""
    if not isinstance(task.scenario, str):
        raise TypeError("visual runs need a bundled scenario name")
    return task.scenario


class MonteCarloWindow(EvacWindow):
    """Window that plays every run of a batch in turn, automatically.

    Attributes:
        outcomes: Runs finished so far, in order.
        finished: ``True`` once the window has closed itself.
    """

    def __init__(
        self,
        tasks: Sequence[RunTask],
        notify: Callable[[RunOutcome], None],
        should_cancel: Callable[[], bool],
    ) -> None:
        """Open the window on the first run and start it.

        Args:
            tasks: Runs to play; must not be empty.
            notify: Called with each finished run.
            should_cancel: Polled every tick; ``True`` closes the window and
                drops the unfinished run.
        """
        first = tasks[0]
        sim, floor_plan = build_simulation_from_scenario(
            _bundled_name(first), first.agents, first.seed, first.events
        )
        super().__init__(
            sim, floor_plan, _bundled_name(first), first.agents, first.seed,
            first.events,
        )
        self._tasks = tuple(tasks)
        self._notify = notify
        self._should_cancel = should_cancel
        self._index = 0
        self._exits: list[int] = []
        self._speed = MovingSpeedTally()
        self.outcomes: list[RunOutcome] = []
        self.finished = False
        self._label: arcade.Text | None = None
        self._handle_spacebar()  # start without waiting for SPACE

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """Ignore keys: pausing or resetting would break the batch."""

    def on_update(self, delta_time: float) -> None:
        """Advance the current run one tick, or finish it.

        Cancel is checked first.  Mirrors the headless loop: a run ends
        when the simulation reports completion or the tick cap is hit;
        hazards fire before each step.

        Args:
            delta_time: Ignored; the simulation steps at a fixed ``DT``.
        """
        if self.finished:
            return
        if self._should_cancel():
            # Cancel must not wait for the current run to play out; the
            # unfinished run is discarded.
            self.finished = True
            self.close()
            return
        sim = self._sim
        if sim.is_complete or sim.tick >= DEFAULT_MAX_TICKS:
            self._finish_run()
            return
        apply_due_events(sim, self._schedule)
        # Mouse hazards would make the batch irreproducible; drop them.
        self._input_source.poll()
        sim.step()
        snap = sim.snapshot()
        record = make_record(snap, self._tasks[self._index].agents)
        self._speed.add(snap)
        self._exits.append(record["egressed_this_tick"])

    def on_draw(self) -> None:
        """Draw the simulation and which run is playing."""
        super().on_draw()
        task = self._tasks[self._index]
        text = f"Run {self._index + 1}/{len(self._tasks)}  seed {task.seed}"
        y = self.height - _LABEL_MARGIN_PX - _LABEL_FONT_SIZE
        if self._label is None:
            # Built on first draw (needs a GL context); a Text object is
            # much cheaper per frame than arcade.draw_text.
            self._label = arcade.Text(
                text, _LABEL_MARGIN_PX, y, arcade.color.WHITE,
                font_size=_LABEL_FONT_SIZE,
            )
        self._label.text = text
        self._label.y = y
        self._label.draw()

    def _finish_run(self) -> None:
        """Record the finished run, then start the next or close."""
        task = self._tasks[self._index]
        outcome = make_outcome(
            task.run_index, task.seed, task.events, task.agents, self._exits,
            self._speed.mean_mps,
        )
        self.outcomes.append(outcome)
        self._notify(outcome)
        if self._index + 1 >= len(self._tasks) or self._should_cancel():
            self.finished = True
            self.close()
            return
        self._index += 1
        self._start_run(self._tasks[self._index])

    def _start_run(self, task: RunTask) -> None:
        """Replace the simulation with the one for ``task``."""
        self._sim, _ = build_simulation_from_scenario(
            _bundled_name(task), task.agents, task.seed, task.events
        )
        self._seed = task.seed
        self._events = task.events
        self._schedule = events_by_tick(task.events)
        self._current_source = None
        self._exits = []
        self._speed = MovingSpeedTally()


def run_visual(
    tasks: Sequence[RunTask],
    notify: Callable[[RunOutcome], None],
    should_cancel: Callable[[], bool],
) -> list[RunOutcome]:
    """Play the tasks in a window and return the finished runs.

    Args:
        tasks: Runs to play, in order.
        notify: Called with each finished run.
        should_cancel: Polled every tick.

    Returns:
        Runs finished when the batch ended; fewer than ``tasks`` if the
        window was closed or the batch cancelled early.
    """
    if not tasks or should_cancel():
        return []
    window = MonteCarloWindow(tasks, notify, should_cancel)
    arcade.run()
    return window.outcomes
