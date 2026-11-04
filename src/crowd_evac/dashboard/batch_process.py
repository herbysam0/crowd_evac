"""Run one batch in a child process and report it through a queue.

Every batch, headless or visual, runs in a child so the dashboard never
blocks and a visual batch can own its own main thread.  The child calls
:func:`~crowd_evac.montecarlo.runner.run_monte_carlo` and puts events on a
queue; the dashboard drains it with :meth:`BatchProcess.poll`.  This module
never imports arcade, so a headless child stays display-free.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import signal
from dataclasses import dataclass, replace
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event as EventType
from typing import TYPE_CHECKING

from crowd_evac.montecarlo.config import MonteCarloConfig
from crowd_evac.montecarlo.results import MonteCarloResult, RunOutcome
from crowd_evac.montecarlo.runner import run_monte_carlo

if TYPE_CHECKING:
    from multiprocessing.queues import Queue as MpQueue

logger = logging.getLogger(__name__)

LOST_CHILD_MESSAGE: str = (
    "The batch process ended unexpectedly; see the log for details."
)
"""Shown when the child dies without a final message."""


@dataclass(frozen=True)
class RunFinished:
    """One run of the batch finished.

    Attributes:
        outcome: What the run produced.
        done: Runs finished so far.
        total: Runs in the batch.
    """

    outcome: RunOutcome
    done: int
    total: int


@dataclass(frozen=True)
class BatchFinished:
    """The batch ended, finished or cancelled; always the last event."""

    result: MonteCarloResult


@dataclass(frozen=True)
class BatchFailed:
    """The batch could not run or its process died; always the last event."""

    message: str


BatchEvent = RunFinished | BatchFinished | BatchFailed


def headless_workers() -> int:
    """Return the headless worker count, keeping one core for the UI."""
    return max((os.cpu_count() or 1) - 1, 1)


class _QueueListener:
    """Forward finished runs to the queue."""

    def __init__(self, events: MpQueue[BatchEvent]) -> None:
        """Remember the queue."""
        self._events = events

    def on_run_complete(
        self, outcome: RunOutcome, done: int, total: int
    ) -> None:
        """Queue a :class:`RunFinished`."""
        self._events.put(RunFinished(outcome, done, total))


def _child_main(
    config: MonteCarloConfig,
    events: MpQueue[BatchEvent],
    cancel: EventType,
) -> None:
    """Child entry point: run the batch, then queue exactly one last event."""
    # Ctrl+C in the terminal reaches every process on the console.  The
    # parent owns shutdown and sets the cancel event, so the child must not
    # die with its own traceback and lose the runs already finished.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        result = run_monte_carlo(config, _QueueListener(events), cancel.is_set)
    except Exception as exc:  # noqa: BLE001
        # Process boundary: every failure (bad hazard, too many agents for
        # the room) must reach the parent as a message, not a lost child.
        logger.exception("Batch failed")
        events.put(BatchFailed(str(exc) or type(exc).__name__))
        return
    events.put(BatchFinished(result))


class BatchProcess:
    """One batch running in a spawned child process.

    Attributes:
        config: Settings the child runs with.
    """

    def __init__(self, config: MonteCarloConfig) -> None:
        """Prepare, but do not start, the child.

        Args:
            config: Batch settings.  A headless config without a worker
                count gets :func:`headless_workers`.
        """
        if config.max_workers is None and not config.visual:
            config = replace(config, max_workers=headless_workers())
        self.config = config
        ctx = multiprocessing.get_context("spawn")
        self._events: MpQueue[BatchEvent] = ctx.Queue()
        self._cancel = ctx.Event()
        # Non-daemon: a daemon cannot start the headless worker pool.
        self._process: SpawnProcess = ctx.Process(
            target=_child_main,
            args=(config, self._events, self._cancel),
            daemon=False,
        )
        self._finished = False

    def start(self) -> None:
        """Start the child."""
        self._process.start()

    def cancel(self) -> None:
        """Ask the child to stop; finished runs are kept."""
        self._cancel.set()

    @property
    def active(self) -> bool:
        """Whether a final event is still to come."""
        return self._process.pid is not None and not self._finished

    def poll(self) -> list[BatchEvent]:
        """Return the events that arrived since the last call.

        Never blocks.  If the child ended with no final event, a
        :class:`BatchFailed` ends the list.

        Returns:
            Events in arrival order; empty when nothing is new.
        """
        if self._finished or self._process.pid is None:
            return []
        events = self._drain()
        if not self._ended(events) and not self._process.is_alive():
            # The child may have queued its last event just before exiting.
            events.extend(self._drain())
            if not self._ended(events):
                logger.error(
                    "Batch process exited with code %s and no result",
                    self._process.exitcode,
                )
                events.append(BatchFailed(LOST_CHILD_MESSAGE))
        if self._ended(events):
            self._finished = True
        return events

    def _drain(self) -> list[BatchEvent]:
        """Take everything currently on the queue."""
        events: list[BatchEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    @staticmethod
    def _ended(events: list[BatchEvent]) -> bool:
        """Whether ``events`` holds the batch's last event."""
        return any(isinstance(e, (BatchFinished, BatchFailed)) for e in events)
