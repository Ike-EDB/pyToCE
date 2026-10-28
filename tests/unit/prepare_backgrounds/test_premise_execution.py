"""Premise process-boundary and IAM custody contracts."""

import json
import multiprocessing
import os
import time
from hashlib import sha256
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from premise_fixtures import RecordingPremiseConnection

from pytoce.prepare_backgrounds.preparation import premise as owner


def _synthetic_scenario(content: bytes) -> owner.BackgroundScenario:
    """Return a hash-bound synthetic IAM member with no licensed content."""
    return owner.BackgroundScenario(
        background_scenario="synthetic_ssp2",
        premise_model="remind",
        iam_model="REMIND",
        iam_scenario="SSP2-NPi",
        gmst="test",
        iam_filename="remind_SSP2-NPi.csv",
        iam_file_size=len(content),
        iam_file_sha256=sha256(content).hexdigest(),
    )


def _iam_input(tmp_path: Path, scenario: owner.BackgroundScenario) -> owner.PremiseIAMInput:
    """Return the recorded IAM reference consumed by the runner."""
    directory = tmp_path / "iam"
    directory.mkdir()
    payload = directory / scenario.iam_filename
    payload.write_bytes(b"bounded")
    manifest = directory / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return owner.PremiseIAMInput(
        identity="iam::bounded",
        payload=payload,
        manifest_sha256=sha256(b"{}\n").hexdigest(),
        payload_sha256=scenario.iam_file_sha256,
        premise_authority_identity="premise::2.4.6",
    )


def _runner_kwargs(tmp_path: Path) -> dict[str, object]:
    """Use disposable paths and an admitted IAM reference at the process boundary."""
    scenario = _synthetic_scenario(b"bounded")

    return {
        "brightway_store": tmp_path / "brightway",
        "project_name": "bounded_project",
        "cache_root": tmp_path / "cache",
        "report_root": tmp_path / "reports",
        "iam_input": _iam_input(tmp_path, scenario),
        "source_database": "source",
        "biosphere_database": "biosphere",
        "scenario": scenario,
        "years": (2030,),
        "output_names": ("background_2030",),
        "premise_key": None,
    }


@pytest.mark.usefixtures("isolated_premise")
@pytest.mark.parametrize("obstruct_guide", [False, True])
def test_verified_downloader_result_is_saved_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    obstruct_guide: bool,
) -> None:
    """Publish and reuse admitted IAM bytes only after their guides are writable."""
    import premise.scenario_downloader as downloader

    content = b"gAAAA-synthetic-encrypted-iam"
    scenario = _synthetic_scenario(content)
    calls: list[tuple[str, str, Path]] = []

    def download_csv(file_name: str, url: str, download_folder: Path) -> Path:
        """Implement premise 2.4.6's public downloader result contract."""
        destination = Path(download_folder) / file_name
        destination.write_bytes(content)
        calls.append((file_name, url, Path(download_folder)))
        return destination

    monkeypatch.setattr(downloader, "download_csv", download_csv)
    workspace = tmp_path / "workspace"
    source_root = workspace / "data_raw" / "premise" / scenario.background_scenario
    if obstruct_guide:
        obstacle = source_root.parent / "README.md"
        obstacle.mkdir(parents=True)
        with pytest.raises(OSError):
            owner.acquire_iam_input(
                workspace,
                scenario,
                premise_authority={"identity": "premise::2.4.6"},
            )
        assert obstacle.is_dir()
        assert not source_root.exists()
        assert not any(path.name.startswith(".iam_input.") for path in source_root.parent.iterdir())
        return
    accepted = owner.acquire_iam_input(
        workspace,
        scenario,
        premise_authority={"identity": "premise::2.4.6"},
    )

    assert accepted.payload.read_bytes() == content
    manifest = json.loads((source_root / "manifest.json").read_bytes())
    assert accepted.payload == source_root / scenario.iam_filename
    assert calls[0][:2] == (
        scenario.iam_filename,
        manifest["identity_payload"]["source_url"],
    )
    assert not any(path.name.startswith(".iam_input.") for path in source_root.parent.iterdir())

    reused = owner.acquire_iam_input(
        workspace,
        scenario,
        premise_authority={"identity": "premise::2.4.6"},
    )
    assert reused == accepted
    assert len(calls) == 1


@pytest.mark.usefixtures("isolated_premise", "premise_worker_streams")
def test_premise_worker_maps_exact_246_arguments_and_records_database_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve scenario isolation and all configured public premise arguments."""
    import bw2data as bd
    import premise

    output_names = ("background_2030", "background_2035")
    recorded: dict[str, object] = {}
    iam_payload = tmp_path / "iam" / "remind_SSP2-NPi.csv"
    iam_payload.parent.mkdir()
    iam_payload.write_bytes(b"bounded")

    class FakeNewDatabase:
        """Implement the premise 2.4.6 constructor/update/write boundary used here."""

        def __init__(self, **kwargs: object) -> None:
            recorded["constructor"] = kwargs
            for scenario in kwargs["scenarios"]:
                payload = Path(scenario["filepath"]) / "remind_SSP2-NPi.csv"
                assert payload.read_bytes() == b"bounded"

        def update(self, sectors: list[str]) -> None:
            recorded["sectors"] = sectors

        def write_db_to_brightway(self, name: list[str] | None = None) -> None:
            recorded["output_names"] = name

    def set_current(project_name: str) -> None:
        """Represent project selection with a test double."""
        recorded["project"] = project_name

    monkeypatch.setattr(premise, "NewDatabase", FakeNewDatabase)
    monkeypatch.setattr(bd.projects, "set_current", set_current)
    monkeypatch.setattr(
        bd,
        "databases",
        {name: {"backend": "sqlite", "number": index} for index, name in enumerate(output_names)},
    )
    monkeypatch.setenv("BRIGHTWAY2_DIR", "restored-by-monkeypatch")
    monkeypatch.chdir(tmp_path)
    result_connection = RecordingPremiseConnection()
    reports = tmp_path / "reports"

    owner._premise_worker(
        result_connection=result_connection,  # type: ignore[arg-type]
        brightway_store=str(tmp_path / "brightway"),
        project_name="bounded_project",
        cache_root=str(tmp_path / "cache"),
        report_root=str(reports),
        iam_directory=str(tmp_path / "iam"),
        iam_input_identity="iam::bounded",
        iam_payload_sha256="a" * 64,
        premise_authority_identity="premise::2.4.6",
        source_database="ecoinvent_3_12_cutoff",
        biosphere_database="biosphere_3_12",
        scenario_payload={
            "premise_model": "remind",
            "background_scenario": "remind_ssp2_npi",
            "iam_scenario": "SSP2-NPi",
        },
        years=(2030, 2035),
        output_names=output_names,
        premise_key="bounded-key",
    )

    constructor = recorded["constructor"]
    assert isinstance(constructor, dict)
    assert constructor["scenarios"] == [
        {
            "model": "remind",
            "pathway": "SSP2-NPi",
            "year": 2030,
            "filepath": str(tmp_path / "iam"),
        },
        {
            "model": "remind",
            "pathway": "SSP2-NPi",
            "year": 2035,
            "filepath": str(tmp_path / "iam"),
        },
    ]
    assert constructor["source_version"] == "3.12"
    assert constructor["source_type"] == "brightway"
    assert constructor["system_model"] == "cutoff"
    assert constructor["source_db"] == "ecoinvent_3_12_cutoff"
    assert constructor["biosphere_name"] == "biosphere_3_12"
    assert recorded["sectors"] == list(owner.PREMISE_UPDATE_SECTORS)
    assert recorded["output_names"] == list(output_names)
    assert recorded["project"] == "bounded_project"
    assert (
        (reports / "variables.yaml").read_text(encoding="utf-8").startswith("USER_DATA_BASE_DIR:")
    )
    assert result_connection.payload is not None
    assert result_connection.payload["ok"] is True
    rows = result_connection.payload["database_rows"]
    assert isinstance(rows, list)
    assert [(row["database_name"], row["anchor_year"]) for row in rows] == [
        ("background_2030", 2030),
        ("background_2035", 2035),
    ]
    assert all(
        str(row["premise_identity"]).startswith("pytoce_premise_database_schema_v1::")
        for row in rows
    )


@pytest.mark.usefixtures("isolated_premise", "premise_worker_streams")
def test_premise_worker_redacts_credentials_from_dependency_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep both the supplied key and a Fernet-looking token out of worker errors."""
    import bw2data as bd
    import premise

    secret = "private-key"
    token = "gAAAAABoundedToken_123"

    class FailingNewDatabase:
        """Implement a constructor failure from the external premise boundary."""

        def __init__(self, **_kwargs: object) -> None:
            raise ValueError(f"credentials {secret} and {token}")

    monkeypatch.setattr(premise, "NewDatabase", FailingNewDatabase)
    monkeypatch.setattr(bd.projects, "set_current", lambda _name: None)
    monkeypatch.setenv("BRIGHTWAY2_DIR", str(tmp_path / "original_brightway"))
    monkeypatch.chdir(tmp_path)
    result_connection = RecordingPremiseConnection()

    owner._premise_worker(
        result_connection=result_connection,  # type: ignore[arg-type]
        brightway_store=str(tmp_path / "brightway"),
        project_name="bounded_project",
        cache_root=str(tmp_path / "cache"),
        report_root=str(tmp_path / "reports"),
        iam_directory=str(tmp_path / "iam"),
        iam_input_identity="iam::bounded",
        iam_payload_sha256="a" * 64,
        premise_authority_identity="premise::2.4.6",
        source_database="source",
        biosphere_database="biosphere",
        scenario_payload={
            "premise_model": "remind",
            "background_scenario": "remind_ssp2_npi",
            "iam_scenario": "SSP2-NPi",
        },
        years=(2030,),
        output_names=("background_2030",),
        premise_key=secret,
    )

    assert result_connection.payload == {
        "schema_version": "pytoce_premise_result_schema_v1",
        "ok": False,
        "error_type": "ValueError",
        "error": "credentials <redacted> and <redacted-fernet-token>",
    }


@pytest.mark.parametrize(
    ("payload", "exitcode", "message"),
    [
        (None, 9, "exit code 9 before it returned"),
        (
            {"ok": False, "error_type": "ValueError", "error": "bounded failure"},
            0,
            "ValueError: bounded failure",
        ),
        ({"ok": True, "databases": ["background_2030"]}, 0, None),
    ],
)
def test_premise_runner_translates_child_process_failures_and_closes_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object] | None,
    exitcode: int,
    message: str | None,
) -> None:
    """Give users actionable errors for a crashed or rejected premise child."""
    result_connection = RecordingPremiseConnection([] if payload is None else [payload])
    sender = RecordingPremiseConnection()

    class FakeProcess:
        """Implement the multiprocessing Process lifecycle observed by the runner."""

        def __init__(self, *, target: object, kwargs: dict[str, object]) -> None:
            self.target = target
            self.kwargs = kwargs
            self.exitcode = exitcode

        def start(self) -> None:
            """Represent starting the isolated worker."""

        def join(self) -> None:
            """Represent waiting for the isolated worker."""

        def is_alive(self) -> bool:
            """This fixture represents a child that has already exited."""
            return False

        def close(self) -> None:
            """Represent releasing the stopped child's handle."""

    class SpawnContext:
        """Implement the two public spawn-context factories used by the runner."""

        def Pipe(  # noqa: N802
            self, *, duplex: bool
        ) -> tuple[RecordingPremiseConnection, RecordingPremiseConnection]:
            assert duplex is False
            return result_connection, sender

        def Process(self, *, target: object, kwargs: dict[str, object]) -> object:  # noqa: N802
            return FakeProcess(target=target, kwargs=kwargs)

    monkeypatch.setattr(multiprocessing, "get_context", lambda method: SpawnContext())
    kwargs = _runner_kwargs(tmp_path)
    if message is None:
        assert owner.run_premise_member(**kwargs) == payload  # type: ignore[arg-type]
    else:
        with pytest.raises(RuntimeError, match=message):
            owner.run_premise_member(**kwargs)  # type: ignore[arg-type]

    assert result_connection.closed is True
    assert sender.closed is True


def test_background_selection_rejects_empty_public_values() -> None:
    """Reject empty scalar and sequence selections before source preparation."""
    for requested in ("", "   ", (), ("remind_ssp2_ndc", "")):
        with pytest.raises(ValueError):
            owner.select_background_scenarios(requested)

    assert owner.select_background_scenarios(("all",)) == owner.ACTIVE_BACKGROUND_SCENARIOS


def test_database_names_keep_scenario_year_and_identity_separate() -> None:
    """Keep general anchors identity-bound and provider anchors early-year."""
    scenario = owner.BACKGROUND_BY_ID["remind_ssp2_ndc"]
    assert (
        owner.database_name(
            scenario,
            2030,
            background_identity="background::abcdef0123456789",
        )
        == "pytoce__remind_ssp2_ndc__2030__abcdef012345"
    )
    assert (
        owner.provider_technology_database_name(
            scenario,
            2025,
        )
        == "pytoce__remind_ssp2_ndc__provider_technology__2025"
    )


def _transport_reply(connection: Connection, mode: str) -> None:
    """Supply a bounded synthetic external-process outcome."""
    if mode == "crash":
        os._exit(17)
    if mode == "interrupt":
        time.sleep(30)
    try:
        if mode == "large_error":
            # Retain complete external dependency diagnostics across OS pipe boundaries.
            connection.send({"ok": False, "error_type": "ValueError", "error": "x" * 1_048_576})
        else:
            connection.send({"ok": True, "databases": ["background_2030"]})
    finally:
        connection.close()


@pytest.mark.parametrize("mode", ["success", "large_error", "crash", "interrupt", "start_failure"])
def test_real_premise_transport_drains_results_and_stops_interrupted_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Receive a real spawned reply and leave no writer alive after failure."""
    spawn = multiprocessing.get_context("spawn")
    receiver, sender = spawn.Pipe(duplex=False)
    closed_processes: list[int | None] = []

    class ReceivingBoundary:
        """Inject interruption at the real external receive operation."""

        def recv(self) -> object:
            if mode == "interrupt":
                raise KeyboardInterrupt
            return receiver.recv()

        def close(self) -> None:
            receiver.close()

    class ProcessBoundary:
        """Delegate the lifecycle to an actual child process with a harmless reply."""

        def __init__(self, connection: Connection) -> None:
            self.process = spawn.Process(target=_transport_reply, args=(connection, mode))

        @property
        def exitcode(self) -> int | None:
            return self.process.exitcode

        def start(self) -> None:
            if mode == "start_failure":
                raise OSError("spawn unavailable")
            self.process.start()

        def join(self) -> None:
            self.process.join(timeout=10)
            assert not self.process.is_alive(), "The owner joined before draining its result"

        def is_alive(self) -> bool:
            return self.process.is_alive()

        def terminate(self) -> None:
            self.process.terminate()

        def close(self) -> None:
            assert not self.process.is_alive()
            closed_processes.append(self.process.exitcode)
            self.process.close()

    class SpawnBoundary:
        """Use real pipe endpoints and the real child lifecycle at the OS boundary."""

        def Pipe(self, *, duplex: bool) -> tuple[ReceivingBoundary, Connection]:  # noqa: N802
            assert duplex is False
            return ReceivingBoundary(), sender

        def Process(self, *, target: object, kwargs: dict[str, object]) -> ProcessBoundary:  # noqa: N802
            return ProcessBoundary(kwargs["result_connection"])  # type: ignore[arg-type]

    monkeypatch.setattr(multiprocessing, "get_context", lambda _method: SpawnBoundary())
    kwargs = _runner_kwargs(tmp_path)
    if mode == "success":
        assert owner.run_premise_member(**kwargs) == {  # type: ignore[arg-type]
            "ok": True,
            "databases": ["background_2030"],
        }
    elif mode == "large_error":
        with pytest.raises(RuntimeError, match="ValueError: x") as error:
            owner.run_premise_member(**kwargs)  # type: ignore[arg-type]
        assert len(str(error.value)) > 1_048_576
    elif mode == "crash":
        with pytest.raises(RuntimeError, match="exit code 17 before it returned"):
            owner.run_premise_member(**kwargs)  # type: ignore[arg-type]
    elif mode == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            owner.run_premise_member(**kwargs)  # type: ignore[arg-type]
    else:
        with pytest.raises(OSError, match="spawn unavailable"):
            owner.run_premise_member(**kwargs)  # type: ignore[arg-type]
    assert receiver.closed and sender.closed
    assert len(closed_processes) == 1
    assert (
        closed_processes[0] is None if mode == "start_failure" else closed_processes[0] is not None
    )
