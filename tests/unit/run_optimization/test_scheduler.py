"""Candidate process scheduling at the external executor boundary."""

import os
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from threading import Lock, Timer
from typing import TypeVar

import pytest

from pytoce.run_optimization.runtime import scheduler
from pytoce.run_optimization.runtime.candidate_results import (
    CandidateResultStore,
    CandidateTask,
    CandidateTaskDispatch,
)

T = TypeVar("T")


class _Future(Future[T]):
    """Return or raise one result supplied by the executor test double."""

    def __init__(self, result: T | BaseException) -> None:
        """Complete a real future so the ordinary wait boundary remains exercised."""
        super().__init__()
        if isinstance(result, BaseException):
            self.set_exception(result)
        else:
            self.set_result(result)


@pytest.fixture
def result_store(tmp_path: Path) -> Iterator[CandidateResultStore]:
    """Use the real locked candidate cache with disposable scheduler results."""
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::scheduler",
        worker_authority_identity="worker::fixture",
    ) as store:
        yield store


def _task(sequence: int) -> CandidateTask:
    """Build one deterministic scheduler task."""

    return CandidateTask(sequence, f"candidate::{sequence}", f"task-{sequence}".encode())


def test_scheduler_reuses_saved_results_and_restores_thread_limits(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
) -> None:
    """Reuse completed candidates and process missing candidates in request order."""

    original = {
        name: f"original-{index}" for index, name in enumerate(scheduler._THREAD_ENVIRONMENT)
    }
    for name, value in original.items():
        monkeypatch.setenv(name, value)
    observed: dict[str, object] = {}

    class Executor:
        """Record the process-pool configuration and return deterministic bytes."""

        def __init__(self, **options: object) -> None:
            observed["options"] = options
            observed["initializations"] = int(observed.get("initializations", 0)) + 1
            observed["worker_environment"] = {
                name: scheduler.os.environ.get(name) for name in scheduler._THREAD_ENVIRONMENT
            }

        def submit(
            self,
            function: Callable[[bytes], bytes],
            task: bytes,
        ) -> _Future[bytes]:
            """Return the bytes that a worker would send to the parent."""

            assert function is scheduler._evaluate_in_worker
            return _Future(task.replace(b"task-", b"result-"))

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            """Record orderly worker shutdown."""

            observed["shutdown"] = (wait, cancel_futures)

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    result_store.save(_task(0), b"saved-0")
    with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
        results = pool.evaluate_batch(
            (_task(1), _task(0)),
            result_store=result_store,
            read_result=bytes.decode,
        )

    assert results == ("saved-0", "result-1")
    assert pool.computed_candidate_identities == frozenset({"candidate::1"})
    assert result_store.read_record(_task(1).record) == b"result-1"
    assert observed["worker_environment"] == {name: "1" for name in scheduler._THREAD_ENVIRONMENT}
    assert observed["shutdown"] == (True, True)
    assert observed["initializations"] == 1
    assert len(tuple(result_store.directory.glob("[0-9]*"))) == 2
    restored = {name: scheduler.os.environ.get(name) for name in scheduler._THREAD_ENVIRONMENT}
    assert restored == original


@pytest.mark.parametrize("candidate_workers", (1, 3))
def test_scheduler_bounds_pending_work_and_preserves_order_after_early_completion(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
    candidate_workers: int,
) -> None:
    """Refill after an out-of-order result with each candidate submitted once."""
    submitted: list[int] = []
    consumed: set[int] = set()
    futures: dict[int, Future[bytes]] = {}
    completion_lock = Lock()
    refill_timeout: Timer | None = None

    def fail_stalled_refill() -> None:
        """Release pending futures with failure if a scheduler regression would hang the test."""
        with completion_lock:
            for future in futures.values():
                if not future.done():
                    future.set_exception(AssertionError("scheduler did not refill the free slot"))

    class Executor:
        """Complete actual futures in a deterministic external-worker completion order."""

        def __init__(self, **options: object) -> None:
            """Check the requested worker bound at the external executor boundary."""
            assert options["max_workers"] == candidate_workers

        def submit(self, _function: Callable[[bytes], bytes], payload: bytes) -> Future[bytes]:
            """Keep earlier work pending until the scheduler refills task one's free slot."""
            nonlocal refill_timeout
            with completion_lock:
                number = int(payload.removeprefix(b"task-"))
                submitted.append(number)
                assert len(submitted) - len(consumed) <= candidate_workers
                future: Future[bytes] = Future()
                futures[number] = future
                if candidate_workers == 1:
                    future.set_result(f"result-{number}".encode())
                elif number == 1:
                    future.set_result(b"result-1")
                elif number == 2:
                    assert consumed == set()
                    # Test safety only; successful completion depends on submission order.
                    refill_timeout = Timer(10.0, fail_stalled_refill)
                    refill_timeout.start()
                elif number == 3:
                    assert consumed == {1}
                    assert not futures[0].done() and not futures[2].done()
                    futures[0].set_result(b"result-0")
                    futures[2].set_result(b"result-2")
                    future.set_result(b"result-3")
                return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            """Require ordinary joined cleanup after all returned records are saved."""
            assert wait and cancel_futures

    def read_saved(payload: bytes) -> str:
        """Read compact summaries only after the parent has published each result."""
        if payload.startswith(b"result-"):
            number = int(payload.removeprefix(b"result-"))
            assert result_store.read_record(_task(number).record) == payload
            consumed.add(number)
        return payload.decode()

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    result_store.save(_task(4), b"saved-4")
    tasks = tuple(_task(number) for number in reversed(range(5)))
    try:
        with scheduler.CandidateProcessPool(
            worker_initializer_payload=b"initializer", candidate_workers=candidate_workers
        ) as pool:
            result = pool.evaluate_batch(tasks, result_store=result_store, read_result=read_saved)
            pool.close()
            reused = pool.evaluate_batch(tasks, result_store=result_store, read_result=bytes.decode)
    finally:
        if refill_timeout is not None:
            refill_timeout.cancel()
            refill_timeout.join()

    assert result == reused == ("result-0", "result-1", "result-2", "result-3", "saved-4")
    assert submitted == [0, 1, 2, 3]
    assert pool.computed_candidate_identities == frozenset(
        f"candidate::{index}" for index in range(4)
    )
    assert consumed == {0, 1, 2, 3}
    assert len(tuple(result_store.directory.glob("[0-9]*"))) == 5


@pytest.mark.parametrize("single_family", (False, True))
def test_scheduler_holds_only_unknown_family_siblings_and_saves_each_result(
    monkeypatch: pytest.MonkeyPatch, result_store: CandidateResultStore, single_family: bool
) -> None:
    """Allow unrelated families to progress while awaiting a completed repair proof."""
    groups = (
        dict.fromkeys(range(5), "a") if single_family else {0: "a", 1: "a", 2: "b", 3: "b", 4: "c"}
    )
    leaders = {
        group: min(index for index in groups if groups[index] == group) for group in groups.values()
    }
    known: dict[str, bytes] = {}
    submitted: list[tuple[int, bytes | None]] = []
    futures: dict[int, Future[bytes]] = {}

    class Executor:
        """Represent independently completing workers at the actual executor boundary."""

        def __init__(self, **options: object) -> None:
            assert options["max_workers"] == 3

        def submit(self, _function, payload: bytes, reference: bytes | None = None):
            number = int(payload.removeprefix(b"task-"))
            submitted.append((number, reference))
            future: Future[bytes] = Future()
            futures[number] = future
            if single_family:
                assert reference is None if number == 0 else reference == known["a"]
                future.set_result(f"result-{number}".encode())
            elif number == 2:
                future.set_result(b"result-2")
            elif number == 4:
                assert [item[0] for item in submitted] == [0, 2, 4]
                futures[0].set_result(b"result-0")
                future.set_result(b"result-4")
            elif number in (1, 3):
                assert reference == known[groups[number]]
                future.set_result(f"result-{number}".encode())
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait and cancel_futures

    def dispatch(task: CandidateTask) -> CandidateTaskDispatch | None:
        group = groups[task.sequence]
        if group in known:
            return CandidateTaskDispatch(task, known[group])
        return CandidateTaskDispatch(task) if leaders[group] == task.sequence else None

    def read(payload: bytes) -> str:
        number = int(payload.removeprefix(b"result-"))
        assert result_store.read_record(_task(number).record) == payload
        known.setdefault(groups[number], payload)
        return payload.decode()

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    with scheduler.CandidateProcessPool(
        worker_initializer_payload=b"fixture", candidate_workers=3
    ) as pool:
        results = pool.evaluate_batch(
            tuple(_task(number) for number in range(5)),
            result_store=result_store,
            read_result=read,
            prepare_dispatch=dispatch,
        )
    assert results == tuple(f"result-{number}" for number in range(5))
    assert [item[0] for item in submitted] == (
        [0, 1, 2, 3, 4] if single_family else [0, 2, 4, 1, 3]
    )
    assert len(tuple(result_store.directory.glob("[0-9]*"))) == 5


@pytest.mark.parametrize("cached_count", (1, 10))
def test_scheduler_does_not_start_a_process_for_a_completed_batch(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
    capsys: pytest.CaptureFixture[str],
    cached_count: int,
) -> None:
    """Return cached results directly."""

    class UnexpectedExecutor:
        """Fail if the scheduler tries to start an unnecessary process."""

        def __init__(self, **_options: object) -> None:
            raise AssertionError("executor was not needed")

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", UnexpectedExecutor)
    tasks = tuple(_task(index) for index in range(cached_count))
    for task in tasks:
        result_store.save(task, b"saved")
    with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
        results = pool.evaluate_batch(
            tasks,
            result_store=result_store,
            read_result=bytes.decode,
        )
    assert results == ("saved",) * cached_count
    assert pool.computed_candidate_identities == frozenset()
    assert capsys.readouterr().out == (
        f"run_optimization: candidate batch {cached_count}/{cached_count} "
        f"(100.0%) evaluations completed; computed=0 reused={cached_count}\n"
    )


@pytest.mark.parametrize(
    ("failure", "expected"),
    ((BrokenProcessPool("stopped"), scheduler.SchedulerError), (ValueError("failed"), ValueError)),
)
def test_scheduler_closes_the_pool_after_a_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
    failure: BaseException,
    expected: type[BaseException],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Translate a broken process and preserve an ordinary calculation error."""

    class FailingExecutor:
        """Raise the requested failure when the worker result is collected."""

        def __init__(self, **_options: object) -> None:
            pass

        def submit(
            self,
            _function: Callable[[bytes], bytes],
            _task: bytes,
        ) -> _Future[bytes]:
            return _Future(failure)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait and cancel_futures

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", FailingExecutor)
    pool = scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer")
    with pytest.raises(expected):
        pool.evaluate_batch(
            (_task(0),),
            result_store=result_store,
            read_result=bytes.decode,
        )
    assert capsys.readouterr().out == (
        "run_optimization: candidate batch 0/1 (0.0%) evaluations completed; computed=0 reused=0\n"
    )


def test_scheduler_restores_the_environment_when_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
) -> None:
    """Restore the parent environment if the operating system rejects startup."""

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    class FailingExecutor:
        """Represent an executor that cannot be created."""

        def __init__(self, **_options: object) -> None:
            raise OSError("cannot start")

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", FailingExecutor)
    pool = scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer")
    with pytest.raises(OSError, match="cannot start"):
        pool.evaluate_batch(
            (_task(0),),
            result_store=result_store,
            read_result=bytes.decode,
        )
    assert "OMP_NUM_THREADS" not in scheduler.os.environ


def test_scheduler_adds_a_note_when_shutdown_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
) -> None:
    """Retain both a worker failure and its shutdown failure."""

    failure = ValueError("calculation failed")

    class FailingExecutor:
        """Raise once from the worker and once during shutdown."""

        def __init__(self, **_options: object) -> None:
            pass

        def submit(
            self,
            _function: Callable[[bytes], bytes],
            _task: bytes,
        ) -> _Future[bytes]:
            return _Future(failure)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            raise RuntimeError("shutdown failed")

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", FailingExecutor)
    pool = scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer")
    with pytest.raises(ValueError, match="calculation failed") as caught:
        pool.evaluate_batch(
            (_task(0),),
            result_store=result_store,
            read_result=bytes.decode,
        )
    assert caught.value.__notes__ == [
        "Worker-pool shutdown also failed: RuntimeError: shutdown failed"
    ]


@pytest.mark.parametrize("interrupt_search", (False, True))
def test_successful_candidate_remains_reusable_after_shutdown_failure(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
    interrupt_search: bool,
) -> None:
    """Preserve saved results and an interrupted search when worker cleanup fails."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    original = {name: os.environ.get(name) for name in scheduler._THREAD_ENVIRONMENT}
    submitted: list[bytes] = []

    class Executor:
        """Return a successful worker result and fail at external executor shutdown."""

        def __init__(self, **_options: object) -> None:
            """Accept the real executor's initialization options."""

        def submit(self, _function: Callable[[bytes], bytes], task: bytes) -> _Future[bytes]:
            """Record scientific work requested across the original and reused batch."""
            submitted.append(task)
            return _Future(b"completed-worker-output")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            """Represent an operating-system failure while closing executor resources."""
            assert wait and cancel_futures
            raise OSError("executor shutdown failed")

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    task = _task(0)
    expected_failure = KeyboardInterrupt if interrupt_search else OSError
    message = "search interrupted" if interrupt_search else "executor shutdown failed"
    with pytest.raises(expected_failure, match=message) as caught:
        with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
            result = pool.evaluate_batch(
                (task,), result_store=result_store, read_result=bytes.decode
            )
            assert result == ("completed-worker-output",)
            if interrupt_search:
                raise KeyboardInterrupt("search interrupted")

    if interrupt_search:
        assert caught.value.__notes__ == [
            "Worker-pool shutdown also failed: OSError: executor shutdown failed"
        ]

    assert result_store.read_record(task.record) == b"completed-worker-output"
    assert {name: os.environ.get(name) for name in scheduler._THREAD_ENVIRONMENT} == original
    with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
        reused = pool.evaluate_batch((task,), result_store=result_store, read_result=bytes.decode)
    assert reused == result
    assert submitted == [task.payload]
    assert {name: os.environ.get(name) for name in scheduler._THREAD_ENVIRONMENT} == original


def test_storage_failure_is_not_hidden_by_worker_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
) -> None:
    """Preserve the actual filesystem error after a worker has returned its result."""

    class Executor:
        def __init__(self, **_options: object) -> None:
            pass

        def submit(self, _function: Callable[[bytes], bytes], _task: bytes) -> _Future[bytes]:
            return _Future(b"completed-worker-output")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            raise RuntimeError("shutdown failed")

    def unavailable_destination(_source: object, _destination: object) -> None:
        """Reproduce a filesystem publication failure at the real atomic-write boundary."""
        raise OSError("destination unavailable")

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(os, "replace", unavailable_destination)
    with pytest.raises(OSError, match="destination unavailable") as caught:
        with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
            pool.evaluate_batch((_task(0),), result_store=result_store, read_result=bytes.decode)
    assert caught.value.__notes__ == [
        "Worker-pool shutdown also failed: RuntimeError: shutdown failed"
    ]
    assert not tuple(result_store.directory.glob("[0-9]*"))


@pytest.mark.parametrize("output_mode", ("terminal", "redirected"))
def test_batch_progress_advances_after_saved_results_are_decoded(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
    output_mode: str,
) -> None:
    """Report batch completion through the active output and retain stored results."""
    messages: list[str] = []
    decoded: list[int] = []

    class Output(StringIO):
        """Represent a terminal or redirected stream at the public stdout boundary."""

        def isatty(self) -> bool:
            """Report the output device selected for this behavior case."""
            return output_mode == "terminal"

        def write(self, value: str) -> int:
            """Observe output only after the corresponding persisted summaries were consumed."""
            if "evaluations completed" in value:
                assert f"computed={len(decoded)} reused=1" in value
                messages.append(value.lstrip("\r"))
            return super().write(value)

    class Executor:
        """Complete real futures with opaque worker payloads at the executor boundary."""

        def __init__(self, **_options: object) -> None:
            """Accept the scheduler's unchanged pool configuration."""

        def submit(self, _function: Callable[[bytes], bytes], payload: bytes) -> _Future[bytes]:
            """Return worker output through production parsing and persistence."""
            return _Future(payload.replace(b"task-", b"result-"))

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            """Preserve normal executor cleanup."""
            assert wait and cancel_futures

    def read_saved(payload: bytes) -> str:
        """Confirm progress has not advanced before this stored result is decoded."""
        if payload.startswith(b"result-"):
            number = int(payload.removeprefix(b"result-"))
            assert result_store.read_record(_task(number).record) == payload
            assert f"computed={len(decoded)} reused=1" in messages[-1]
            decoded.append(number)
        return payload.decode()

    output = Output()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    result_store.save(_task(2), b"cached-2")
    with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
        result = pool.evaluate_batch(
            (_task(2), _task(1), _task(0)), result_store=result_store, read_result=read_saved
        )
    assert result == ("result-0", "result-1", "cached-2")
    assert len(messages) == 3
    assert messages[0].endswith("1/3 (33.3%) evaluations completed; computed=0 reused=1")
    assert messages[-1].endswith("3/3 (100.0%) evaluations completed; computed=2 reused=1")
    if output_mode == "terminal":
        assert output.getvalue().count("\r") == 3
        assert output.getvalue().count("\n") == 1
    else:
        assert "\r" not in output.getvalue()
        assert len(output.getvalue().splitlines()) == 3


def test_empty_scheduler_batch_emits_no_completion_claim(
    result_store: CandidateResultStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Return the specified empty-batch result."""
    with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
        assert pool.evaluate_batch((), result_store=result_store, read_result=bytes.decode) == ()
    assert capsys.readouterr().out == ""


def test_redirected_progress_stays_in_stdout_with_active_ipython(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
    tmp_path: Path,
) -> None:
    """Respect an explicit output redirection even with an active rich-display publisher."""
    pytest.importorskip("ipykernel.iostream")
    from IPython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    shell = InteractiveShell.instance()
    result_store.save(_task(0), b"completed")
    output = StringIO()
    try:
        assert get_ipython() is shell
        with capture_output(display=True) as published, redirect_stdout(output):
            with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
                result = pool.evaluate_batch(
                    (_task(0),), result_store=result_store, read_result=bytes.decode
                )
        assert result == ("completed",)
        assert output.getvalue() == (
            "run_optimization: candidate batch 1/1 (100.0%) evaluations completed; "
            "computed=0 reused=1\n"
        )
        assert published.outputs == []
    finally:
        InteractiveShell.clear_instance()


def test_notebook_progress_updates_one_display_on_the_actual_kernel_stream(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
    tmp_path: Path,
) -> None:
    """Use a real kernel output stream and retain one display identity across completions."""
    iostream = pytest.importorskip("ipykernel.iostream")
    import zmq
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from jupyter_client.session import Session

    class Executor:
        """Return opaque result bytes at the external executor boundary."""

        def __init__(self, **_options: object) -> None:
            """Accept the scheduler's unchanged process-pool configuration."""

        def submit(self, _function: Callable[[bytes], bytes], _payload: bytes) -> _Future[bytes]:
            """Complete a future and publish its cache record."""
            return _Future(b"completed")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            """Require the usual executor cleanup."""
            assert wait and cancel_futures

    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    InteractiveShell.instance()
    try:
        with zmq.Context() as context:
            publisher = iostream.IOPubThread(context.socket(zmq.PUB))
            output = iostream.OutStream(Session(), publisher, "stdout", watchfd=False)
            try:
                with capture_output(display=True) as published, redirect_stdout(output):
                    with scheduler.CandidateProcessPool(
                        worker_initializer_payload=b"initializer"
                    ) as pool:
                        result = pool.evaluate_batch(
                            (_task(0),), result_store=result_store, read_result=bytes.decode
                        )
            finally:
                output.close()
                publisher.close()
                publisher.io_loop.close(all_fds=True)
        assert result == ("completed",)
        assert result_store.read_record(_task(0).record) == b"completed"
        displays = published.outputs
        assert len(displays) == 2
        assert displays[0].data["text/plain"] == repr(
            "run_optimization: candidate batch 0/1 (0.0%) evaluations completed; "
            "computed=0 reused=0"
        )
        assert displays[1].data["text/plain"] == repr(
            "run_optimization: candidate batch 1/1 (100.0%) evaluations completed; "
            "computed=1 reused=0"
        )
        assert displays[0].transient["display_id"] == displays[1].transient["display_id"]
        assert not displays[0].update
        assert displays[1].update
        assert published.stdout == ""
    finally:
        InteractiveShell.clear_instance()


@pytest.mark.parametrize(
    "worker_failure", (ValueError("worker failed"), BrokenProcessPool("lost"), None)
)
def test_progress_cleanup_preserves_the_primary_failure_and_completed_cache(
    monkeypatch: pytest.MonkeyPatch,
    result_store: CandidateResultStore,
    worker_failure: BaseException | None,
) -> None:
    """Keep the worker error when terminal cleanup also fails, or expose cleanup alone."""

    class FailingTerminal(StringIO):
        """Represent a terminal becoming unavailable only as its progress line closes."""

        def isatty(self) -> bool:
            """Enable the ordinary terminal completion newline."""
            return True

        def write(self, value: str) -> int:
            """Fail the real stdout boundary after accepting live progress text."""
            if value == "\n":
                raise OSError("terminal unavailable")
            return super().write(value)

    class Executor:
        """Return or fail one actual future at the external executor boundary."""

        def __init__(self, **_options: object) -> None:
            """Accept the unchanged process-pool configuration."""

        def submit(self, _function: Callable[[bytes], bytes], _payload: bytes) -> _Future[bytes]:
            """Keep calculation failure distinct from the later terminal failure."""
            return _Future(worker_failure if worker_failure is not None else b"completed")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            """Close the executor and preserve the original failure."""
            assert wait and cancel_futures

    monkeypatch.setattr(sys, "stdout", FailingTerminal())
    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    result_store.save(_task(1), b"already-completed")
    expected = (
        scheduler.SchedulerError
        if isinstance(worker_failure, BrokenProcessPool)
        else ValueError
        if worker_failure is not None
        else OSError
    )
    with pytest.raises(expected) as caught:
        with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
            pool.evaluate_batch(
                (_task(0), _task(1)), result_store=result_store, read_result=bytes.decode
            )
    if worker_failure is not None:
        assert caught.value.__notes__ == [
            "Progress-line cleanup also failed: OSError: terminal unavailable"
        ]
        if isinstance(worker_failure, BrokenProcessPool):
            assert caught.value.__cause__ is worker_failure
        else:
            assert caught.value is worker_failure
        assert result_store.read_record(_task(0).record) is None
    else:
        assert str(caught.value) == "terminal unavailable"
        assert result_store.read_record(_task(0).record) == b"completed"
    assert result_store.read_record(_task(1).record) == b"already-completed"
