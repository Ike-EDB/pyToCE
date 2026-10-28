"""Evaluate optimization candidates with a fixed number of worker processes.

Numerical-library thread limits are set before workers are spawned and import
scientific dependencies. The coordinator saves completed results in their assigned cache
slots and returns compact summaries in request order.
"""

import multiprocessing
import os
import sys
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Protocol, TypeVar, cast

from pytoce.run_optimization.runtime.candidate_results import (
    CandidateResultStore,
    CandidateTask,
    CandidateTaskDispatch,
)

_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class CandidateWorker(Protocol):
    """One worker retaining its scientific inputs across candidate evaluations."""

    def __call__(self, payload: bytes, repair_reference: bytes | None = None) -> bytes:
        """Evaluate one candidate and return its serialized result bytes."""

        ...


ResultSummary = TypeVar("ResultSummary")
CandidateResultReader = Callable[
    [bytes],
    ResultSummary,
]


_WORKER: CandidateWorker | None = None


class SchedulerError(RuntimeError):
    """A candidate worker could not be started or used."""


class _CandidateBatchProgress:
    """Display completed candidate evaluations for the current batch."""

    def __init__(self, *, total: int, reused: int) -> None:
        """Group cache hits and select terminal, notebook or separate log lines."""
        self._total = total
        self._reused = reused
        self._computed = 0
        self._stream = sys.stdout
        self._terminal = self._stream.isatty()
        self._display_update: Callable[[str], object] | None = None
        message = self._message()
        if not self._terminal and "ipykernel.iostream" in sys.modules:
            from IPython import get_ipython

            # An active notebook can redirect stdout to a file or captured stream.
            # Update a display only when output still goes to the notebook.
            if (
                isinstance(self._stream, sys.modules["ipykernel.iostream"].OutStream)
                and get_ipython() is not None
            ):
                from IPython.display import DisplayHandle

                handle = DisplayHandle()
                handle.display(message)
                self._display_update = handle.update
        if self._display_update is None:
            self._write(message)

    def _message(self) -> str:
        """Label the batch denominator and distinguish computed evaluations from reuse."""
        completed = self._computed + self._reused
        return (
            f"run_optimization: candidate batch {completed}/{self._total} "
            f"({100 * completed / self._total:.1f}%) evaluations completed; "
            f"computed={self._computed} reused={self._reused}"
        )

    def _write(self, message: str) -> None:
        """Replace terminal/notebook text while preserving readable redirected snapshots."""
        if self._display_update is not None:
            self._display_update(message)
        else:
            print(
                ("\r" if self._terminal else "") + message,
                end="" if self._terminal else "\n",
                file=self._stream,
                flush=True,
            )

    def advance(self) -> None:
        """Advance only after a new result has been saved and decoded successfully."""
        self._computed += 1
        self._write(self._message())

    def close(self) -> None:
        """Leave subsequent phase messages or error tracebacks on a fresh terminal line."""
        if self._terminal:
            print(file=self._stream, flush=True)


def _configure_worker_environment() -> None:
    """Set numerical-library thread limits inherited by spawned candidate workers."""
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"


def _initialize_worker(
    initializer_payload: bytes,
) -> None:
    """Construct the package candidate worker after setting thread limits."""
    global _WORKER

    from pytoce.run_optimization.calculation.candidate_worker import (
        initialize_candidate_worker,
    )

    _WORKER = initialize_candidate_worker(initializer_payload)


def _evaluate_in_worker(payload: bytes, repair_reference: bytes | None = None) -> bytes:
    """Evaluate one candidate and return its serialized result."""
    return cast(CandidateWorker, _WORKER)(payload, repair_reference)


class CandidateProcessPool:
    """Persistent spawned workers that evaluate deterministic candidate batches."""

    def __init__(
        self,
        *,
        worker_initializer_payload: bytes,
        candidate_workers: int = 1,
    ) -> None:
        """Retain the validated inputs for a spawn-based process pool."""
        self._worker_initializer_payload = worker_initializer_payload
        self._candidate_workers = candidate_workers
        self._executor: ProcessPoolExecutor | None = None
        self._parent_thread_environment: dict[str, str | None] | None = None
        self._computed_candidate_identities: set[str] = set()

    @property
    def computed_candidate_identities(self) -> frozenset[str]:
        """Identify successful worker calculations performed by this pool, excluding cache hits."""
        return frozenset(self._computed_candidate_identities)

    def __enter__(self) -> "CandidateProcessPool":
        """Enter the owned context and return its active value."""
        return self

    def __exit__(self, _kind: object, failure: BaseException | None, _traceback: object) -> None:
        """Release workers while preserving calculation and result-storage failures."""
        try:
            self.close()
        except BaseException as cleanup_error:
            if failure is None:
                raise
            failure.add_note(
                f"Worker-pool shutdown also failed: {type(cleanup_error).__name__}: {cleanup_error}"
            )

    def close(self) -> None:
        """Release workers before selected-result loading; repeated closure is harmless."""

        executor = self._executor
        self._executor = None
        try:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self._restore_parent_thread_environment()

    def evaluate_batch(
        self,
        tasks: Sequence[CandidateTask],
        *,
        result_store: CandidateResultStore,
        read_result: CandidateResultReader[ResultSummary],
        prepare_dispatch: Callable[[CandidateTask], CandidateTaskDispatch | None] | None = None,
    ) -> tuple[ResultSummary, ...]:
        """Evaluate one candidate batch and return its results in task order.

        The callback receives the same saved result bytes for new and reused
        candidates. At most the requested number of tasks are outstanding. The coordinator saves
        each completed result and releases its bytes before refilling the pool.
        Live completion counts describe the current batch. MFA and environmental
        outcomes are recorded for each completed candidate. Cache hits are reported together.
        An optional MFA group dispatcher holds only siblings whose one
        representative is unfinished; unrelated families fill the other slots.
        """
        ordered = tuple(sorted(tasks, key=lambda task: task.sequence))
        if not ordered:
            return ()

        completed: dict[int, ResultSummary] = {}
        pending: deque[CandidateTask] = deque()
        for task in ordered:
            saved = result_store.read_record(task.record)
            if saved is not None:
                completed[task.sequence] = read_result(saved)
                del saved
                continue
            pending.append(task)

        progress = _CandidateBatchProgress(total=len(ordered), reused=len(completed))
        running: dict[Future[bytes], CandidateTaskDispatch] = {}
        batch_failure: BaseException | None = None
        try:
            while pending or running:
                while pending and len(running) < self._candidate_workers:
                    for _ in range(len(pending)):
                        task = pending.popleft()
                        dispatch = (
                            CandidateTaskDispatch(task)
                            if prepare_dispatch is None
                            else prepare_dispatch(task)
                        )
                        if dispatch is not None:
                            break
                        pending.append(task)
                    else:
                        break
                    if dispatch.repair_reference is None:
                        future = self._get_executor().submit(_evaluate_in_worker, task.payload)
                    else:
                        future = self._get_executor().submit(
                            _evaluate_in_worker, task.payload, dispatch.repair_reference
                        )
                    running[future] = dispatch
                # The bridge defers only siblings of a requested representative;
                # a nonempty pending batch therefore has work to await here.
                finished, waiting = wait(running, return_when=FIRST_COMPLETED)
                del waiting
                while finished:
                    future = min(finished, key=lambda item: running[item].task.sequence)
                    finished.remove(future)
                    dispatch = running.pop(future)
                    task = dispatch.task
                    worker_result = future.result()
                    saved = result_store.save(
                        task, worker_result, repair_provenance=dispatch.repair_provenance
                    )
                    completed[task.sequence] = read_result(saved)
                    self._computed_candidate_identities.add(task.candidate_identity)
                    progress.advance()
                    # Futures retain their full result after collection. Drop
                    # them before another candidate can fill the free slot.
                    del future, worker_result, saved
        except BaseException as failure:
            batch_failure = failure
            try:
                self.close()
            except BaseException as cleanup_error:
                failure.add_note(
                    "Worker-pool shutdown also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            if isinstance(failure, BrokenProcessPool):
                batch_failure = SchedulerError(
                    "A candidate worker could not start or stopped unexpectedly. "
                    "Keep the candidate cache and completed results unchanged. "
                    "Diagnose the failure before choosing another run; report the "
                    "error if the cause is unclear."
                )
                raise batch_failure from failure
            raise
        finally:
            try:
                progress.close()
            except BaseException as cleanup_error:
                if batch_failure is None:
                    raise
                batch_failure.add_note(
                    "Progress-line cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        return tuple(completed[task.sequence] for task in ordered)

    def _get_executor(self) -> ProcessPoolExecutor:
        """Create the process executor on first use and handle startup failure."""
        if self._executor is None:
            self._parent_thread_environment = {
                name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
            }
            _configure_worker_environment()
            try:
                self._executor = ProcessPoolExecutor(
                    max_workers=self._candidate_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker,
                    initargs=(self._worker_initializer_payload,),
                )
            except BaseException:
                self._restore_parent_thread_environment()
                raise
        return self._executor

    def _restore_parent_thread_environment(self) -> None:
        """Restore numerical-library thread variables changed before startup."""
        if self._parent_thread_environment is None:
            return
        for name, value in self._parent_thread_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._parent_thread_environment = None
