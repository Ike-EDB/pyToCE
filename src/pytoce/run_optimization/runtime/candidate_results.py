"""Persist long-running optimization candidate results between runs."""

import json
import os
import shutil
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self, cast

from pytoce.run_optimization.calculation.worker_protocol import (
    CANDIDATE_REPAIR_PROVENANCE_SCHEMA,
    CANDIDATE_WORKER_RESULT_SCHEMA,
    CANDIDATE_WORKER_TASK_SCHEMA,
)
from pytoce.shared.runtime.file_lock import (
    FileLockError,
    acquire_writer_lock,
    release_writer_lock,
)
from pytoce.shared.runtime.filesystem import atomic_write, canonical_json_bytes

_OWNER_SCHEMA = "pytoce_candidate_calculation_owner_schema_v1"
_RESULT_SCHEMA = "pytoce_candidate_result_schema_v1"


class CandidateResultStoreError(RuntimeError):
    """A saved optimization candidate result cannot be used."""


@dataclass(frozen=True, slots=True)
class CandidateTask:
    """One serialized candidate in deterministic request order."""

    sequence: int
    candidate_identity: str
    payload: bytes
    repair_family_identity: str | None = None

    @property
    def record(self) -> "CandidateTaskRecord":
        """Return request order and intervention identity."""

        return CandidateTaskRecord(
            self.sequence,
            self.candidate_identity,
            self.repair_family_identity,
        )


@dataclass(frozen=True, slots=True)
class CandidateTaskRecord:
    """Request sequence and intervention identity retained as a compact record."""

    sequence: int
    candidate_identity: str
    repair_family_identity: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateRepairProvenance:
    """Execution-only basis stored with a new candidate row, outside result identity."""

    family_identity: str
    requested_intervention_identity: str
    worker_result_identity: str


@dataclass(frozen=True, slots=True)
class CandidateTaskDispatch:
    """One unchanged task plus optional compact repair execution reference."""

    task: CandidateTask
    repair_reference: bytes | None = None
    repair_provenance: CandidateRepairProvenance | None = None


class CandidateResultStore(AbstractContextManager["CandidateResultStore"]):
    """Own completed candidate calculations shared by compatible search schedules."""

    def __init__(
        self,
        *,
        directory: Path,
        lock_stream: BinaryIO,
        rows: dict[str, Path],
        next_row: int,
    ) -> None:
        """Attach the checked cache directory and its held writer lock."""

        self.directory = directory
        self._lock_stream = lock_stream
        self._rows = rows
        self._next_row = next_row

    @classmethod
    def open(
        cls,
        parent: Path,
        *,
        candidate_input_identity: str,
        worker_authority_identity: str,
    ) -> "CandidateResultStore":
        """Open or create the candidate result cache under an exclusive lock."""

        parent.mkdir(parents=True, exist_ok=True)
        directory = parent / ".candidate_results"
        try:
            lock_stream = acquire_writer_lock(parent / ".candidate_results.lock")
        except FileLockError as exc:
            raise CandidateResultStoreError(str(exc)) from exc
        try:
            if directory.exists() or directory.is_symlink():
                if directory.is_symlink() or not directory.is_dir():
                    raise CandidateResultStoreError(
                        f"The candidate result cache is not a directory: {directory}. "
                        "Keep this path unchanged. Use a new workspace or investigate "
                        "the existing path before attempting recovery."
                    )
                manifest = _read_json(directory / "manifest.json")
                if (
                    manifest.get("schema_version") != _OWNER_SCHEMA
                    or manifest["candidate_input_identity"] != candidate_input_identity
                    or manifest["worker_authority_identity"] != worker_authority_identity
                ):
                    raise CandidateResultStoreError(
                        f"The candidate result cache belongs to different inputs: {directory}. "
                        "Keep this cache unchanged. Use a new workspace for the changed inputs."
                    )
            else:
                staging = Path(
                    tempfile.mkdtemp(prefix=".candidate_results.", suffix=".staging", dir=parent)
                )
                try:
                    atomic_write(
                        staging / "manifest.json",
                        canonical_json_bytes(
                            {
                                "schema_version": _OWNER_SCHEMA,
                                "task_schema_version": CANDIDATE_WORKER_TASK_SCHEMA,
                                "result_schema_version": CANDIDATE_WORKER_RESULT_SCHEMA,
                                "candidate_input_identity": candidate_input_identity,
                                "worker_authority_identity": worker_authority_identity,
                            }
                        ),
                    )
                    os.replace(staging, directory)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
            # Numbered rows retain their immutable storage locations. Rebuild only
            # the lookup needed for schedule-independent reuse
            # or an integrity scan of completed scientific payloads.
            rows: dict[str, Path] = {}
            next_row = 0
            for row in sorted(directory.glob("[0-9]*")):
                if row.is_symlink() or not row.is_dir():
                    cls._invalid(row, "is not a directory")
                row_manifest = _read_json(row / "manifest.json")
                rows[cast(str, row_manifest["candidate_identity"])] = row
                next_row = max(next_row, int(row.name) + 1)
        except BaseException:
            release_writer_lock(lock_stream)
            raise
        return cls(
            directory=directory,
            lock_stream=lock_stream,
            rows=rows,
            next_row=next_row,
        )

    def __enter__(self) -> Self:
        """Return the locked candidate result store."""

        return self

    def __exit__(self, *_args: object) -> None:
        """Release the candidate result store lock."""

        release_writer_lock(self._lock_stream)

    def read_record(self, task: CandidateTaskRecord) -> bytes | None:
        """Read a completed intervention independently of this search's task order."""

        row = self._rows.get(task.candidate_identity)
        if row is None:
            return None
        return (row / "payload.bin").read_bytes()

    def save(
        self,
        task: CandidateTask,
        payload: bytes,
        *,
        repair_provenance: CandidateRepairProvenance | None = None,
    ) -> bytes:
        """Save a complete result, optionally binding its execution-only repair basis.

        New rows may include a named provenance object. Existing rows and manifests are immutable;
        lookup uses the candidate identity. Provenance records the calculation's execution context.
        """

        row = self.directory / f"{self._next_row:08d}"
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self._next_row:08d}.",
                suffix=".staging",
                dir=self.directory,
            )
        )
        try:
            atomic_write(staging / "payload.bin", payload)
            manifest: dict[str, object] = {
                "schema_version": _RESULT_SCHEMA,
                "candidate_identity": task.candidate_identity,
            }
            if repair_provenance is not None:
                manifest["repair_reference"] = {
                    "schema_version": CANDIDATE_REPAIR_PROVENANCE_SCHEMA,
                    "family_identity": repair_provenance.family_identity,
                    "requested_intervention_identity": (
                        repair_provenance.requested_intervention_identity
                    ),
                    "worker_result_identity": repair_provenance.worker_result_identity,
                }
            atomic_write(
                staging / "manifest.json",
                canonical_json_bytes(manifest),
            )
            os.replace(staging, row)
            self._rows[task.candidate_identity] = row
            self._next_row += 1
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return payload

    @staticmethod
    def _invalid(path: Path, detail: str) -> None:
        """Identify a cache row that cannot be read as a completed result."""

        raise CandidateResultStoreError(
            f"The saved candidate result at {path} {detail}. Keep this result and "
            "the remaining cache unchanged. Diagnose the result before choosing "
            "recovery or a new workspace."
        )


def _read_json(path: Path) -> dict[str, object]:
    """Read one package-written JSON manifest from the candidate result cache."""

    if not path.is_file():
        raise CandidateResultStoreError(
            f"The candidate result manifest is missing at {path}. Keep the cache unchanged. "
            "Diagnose the incomplete product before choosing recovery or a new workspace."
        )
    return json.loads(path.read_bytes())
