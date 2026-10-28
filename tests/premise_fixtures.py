"""Isolate premise imports and run the preparation worker with small test inputs."""

import json
import logging.config
import multiprocessing
import os
from collections.abc import Callable, Iterator
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, TextIOWrapper
from pathlib import Path
from types import ModuleType

import pytest

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)


class RecordingPremiseConnection:
    """Record one end of the worker's single-result pipe."""

    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        self.messages = [] if messages is None else messages
        self.closed = False

    @property
    def payload(self) -> dict[str, object] | None:
        """Expose the message for assertions on the real worker output."""
        return self.messages[0] if self.messages else None

    def send(self, payload: dict[str, object]) -> None:
        """Record one worker payload on an open endpoint."""
        assert not self.closed
        self.messages.append(payload)

    def recv(self) -> dict[str, object]:
        """Read the worker result or report that the child closed before returning it."""
        assert not self.closed
        if not self.messages:
            raise EOFError
        return self.messages[0]

    def close(self) -> None:
        """Record endpoint closure."""
        self.closed = True


@pytest.fixture(scope="session")
def isolated_premise(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """Import premise once with all writable paths in the pytest session directory."""
    root = tmp_path_factory.mktemp("premise_import")
    names = BrightwayNames(root / "brightway", "premise_import", "biosphere", "source")
    names.store.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("BRIGHTWAY2_DIR", str(names.store))
        patch.setenv("WIN_PD_OVERRIDE_LOCAL_APPDATA", str(root / "appdata"))
        patch.setenv("XDG_STATE_HOME", str(root / "state"))
        patch.setenv("OPENPYXL_LXML", os.environ.get("OPENPYXL_LXML", "False"))
        patch.chdir(root)
        (root / "variables.yaml").write_text(
            f"USER_DATA_BASE_DIR: {json.dumps(str(root / 'cache'))}\n", encoding="utf-8"
        )
        # premise configures logging at import; preserve pytest's handlers and loggers.
        patch.setattr(logging.config, "dictConfig", lambda _configuration: None)
        with selected_brightway_project(names):
            import premise

    return premise


@pytest.fixture
def premise_worker_streams() -> Iterator[None]:
    """Give the worker reconfigurable streams and preserve pytest capture."""
    with (
        TextIOWrapper(BytesIO(), encoding="utf-8") as stdout,
        TextIOWrapper(BytesIO(), encoding="utf-8") as stderr,
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        yield


@pytest.fixture
def inline_premise_process(
    isolated_premise: ModuleType,
    premise_worker_streams: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the actual worker in a temporary project through the external process API."""

    class InlineProcess:
        """Execute the worker while restoring the parent cwd, environment and project."""

        def __init__(self, *, target: Callable, kwargs: dict[str, object]) -> None:
            self.target = target
            self.kwargs = kwargs
            self.exitcode = 0

        def start(self) -> None:
            """Use the supplied real temporary project for one package worker call."""
            names = BrightwayNames(
                Path(str(self.kwargs["brightway_store"])),
                str(self.kwargs["project_name"]),
                str(self.kwargs["biosphere_database"]),
                str(self.kwargs["source_database"]),
            )
            with pytest.MonkeyPatch.context() as patch:
                patch.chdir(Path.cwd())
                patch.setenv("BRIGHTWAY2_DIR", str(names.store))
                with selected_brightway_project(names):
                    self.target(**self.kwargs)

        def join(self) -> None:
            """The inline worker has already completed when start returns."""

        def is_alive(self) -> bool:
            """The synchronous fixture has no live child after start returns."""
            return False

        def close(self) -> None:
            """The synchronous fixture has no process handle to release."""

    class InlineContext:
        """Implement the pipe and process factories used by premise preparation."""

        def Pipe(  # noqa: N802
            self, *, duplex: bool
        ) -> tuple[RecordingPremiseConnection, RecordingPremiseConnection]:
            """Connect distinct endpoints through their one recorded message."""
            assert duplex is False
            messages: list[dict[str, object]] = []
            return RecordingPremiseConnection(messages), RecordingPremiseConnection(messages)

        def Process(self, *, target: Callable, kwargs: dict[str, object]) -> InlineProcess:  # noqa: N802
            """Pass the real worker and arguments to the inline process."""
            return InlineProcess(target=target, kwargs=kwargs)

    monkeypatch.setattr(multiprocessing, "get_context", lambda _method: InlineContext())
