"""Write workflow files and calculate reproducible identifiers."""

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import uuid4

import numpy as np
import pandas as pd


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize a JSON-compatible payload deterministically as UTF-8."""

    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    """Return the SHA-256 digest of canonical JSON content."""

    return sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest by reading the file in chunks."""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_table_sha256(frame: pd.DataFrame) -> str:
    """Hash table values, types, ordering, multiplicity, and null placement."""

    schema = _canonical_table_schema(frame)
    digest = sha256()
    digest.update(_canonical_table_header(schema, len(frame)))
    _update_canonical_table_rows(digest, frame)
    return digest.hexdigest()


class CanonicalTableHashSpool:
    """Hash one logical table incrementally from its row frames."""

    def __init__(self, path: Path) -> None:
        """Open a new spool that records canonical row hashes in table order."""
        self._path = Path(path)
        self._stream: BinaryIO | None = self._path.open("xb")
        self._schema: tuple[tuple[str, str], ...] | None = None
        self._row_count = 0

    @property
    def row_count(self) -> int:
        """Return the number of rows written so far."""

        return self._row_count

    def update(self, frame: pd.DataFrame) -> None:
        """Append rows with the same columns and dtypes as the first table."""

        if self._schema is None:
            self._schema = _canonical_table_schema(frame)
        stream = cast(BinaryIO, self._stream)
        for start in range(0, len(frame), 100_000):
            hashes = pd.util.hash_pandas_object(
                frame.iloc[start : start + 100_000],
                index=False,
                categorize=False,
            ).to_numpy(dtype=np.uint64)
            stream.write(hashes.tobytes())
        self._row_count += len(frame)

    def finalize(self) -> str:
        """Return the combined table hash and remove the temporary row file."""

        stream = cast(BinaryIO, self._stream)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        self._stream = None
        digest = sha256()
        digest.update(
            _canonical_table_header(
                cast(tuple[tuple[str, str], ...], self._schema),
                self._row_count,
            )
        )
        with self._path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self._path.unlink()
        return digest.hexdigest()

    def abort(self) -> None:
        """Close and remove an unfinished private row spool."""

        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._path.unlink(missing_ok=True)


def _canonical_table_schema(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    """Encode table column names and dtypes for canonical hashing."""
    return tuple((str(column), str(frame[column].dtype)) for column in frame.columns)


def _canonical_table_header(
    schema: tuple[tuple[str, str], ...],
    row_count: int,
) -> bytes:
    """Encode the table schema and row count for canonical hashing."""
    return canonical_json_bytes(
        {
            "contract": "pytoce_canonical_table_schema_v1",
            "schema": list(schema),
            "row_count": row_count,
        }
    )


def _update_canonical_table_rows(digest: Any, frame: pd.DataFrame) -> None:
    """Add the frame's canonical row hashes to ``digest`` in fixed-size batches."""
    for start in range(0, len(frame), 100_000):
        hashes = pd.util.hash_pandas_object(
            frame.iloc[start : start + 100_000],
            index=False,
            categorize=False,
        ).to_numpy(dtype=np.uint64)
        digest.update(hashes.tobytes())


def atomic_write(path: Path, content: bytes) -> None:
    """Write one file atomically in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_directory(
    staging: Path,
    destination: Path,
    *,
    replace_existing: bool = False,
) -> None:
    """Move a completed directory into place and restore the previous one on failure."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = destination.parent / (f".{destination.name}.previous.{uuid4().hex}")
    had_previous = destination.exists() or destination.is_symlink()
    if had_previous:
        if destination.is_symlink() or not destination.is_dir():
            raise FileExistsError(
                f"Cannot replace {destination} because it is not a package result directory"
            )
        if not replace_existing:
            raise FileExistsError(
                f"Cannot write {destination} because that directory already exists. Keep "
                "it unchanged and use the workflow's refresh option only for a completed "
                "package result."
            )
        os.replace(destination, previous)
    attached = False
    try:
        os.replace(staging, destination)
        attached = True
        _sync_directory(destination.parent)
    except BaseException:
        if attached and destination.exists():
            os.replace(destination, staging)
        if had_previous and previous.exists() and not destination.exists():
            os.replace(previous, destination)
        raise
    if previous.exists():
        shutil.rmtree(previous, ignore_errors=True)


def _sync_directory(directory: Path) -> None:
    """Synchronize directory entries where directory handles are supported."""

    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
