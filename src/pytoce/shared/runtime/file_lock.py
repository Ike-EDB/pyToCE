"""Cross-platform file locks used by package workflows."""

import errno
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Literal, cast


class FileLockError(RuntimeError):
    """A package operation lock cannot be opened, locked, or used."""


@contextmanager
def writer_lock(path: Path) -> Iterator[None]:
    """Hold one package writer lock for the duration of an operation."""

    stream = acquire_writer_lock(path)
    try:
        yield
    finally:
        release_writer_lock(stream)


def acquire_writer_lock(path: Path) -> BinaryIO:
    """Obtain a process lock that the operating system releases after a crash."""

    try:
        stream = path.open("a+b")
    except OSError as exc:
        raise FileLockError(
            f"The package cannot open its operation lock at {path}. Check that the parent "
            "directory exists and is writable, then run the workflow again."
        ) from exc

    try:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
    except OSError as exc:
        stream.close()
        raise FileLockError(
            f"The package cannot write its operation lock at {path}. Check the available "
            "disk space and directory permissions, then run the workflow again."
        ) from exc

    try:
        if os.name == "nt":
            _windows_lock(stream, mode_name="LK_NBLCK")
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
            )
        return stream
    except OSError as exc:
        stream.close()
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise FileLockError(
                f"Another package process is using {path.parent}. Wait for it to finish, then "
                "run the workflow again."
            ) from exc
        raise FileLockError(
            f"The operating system cannot lock {path}. Check that file locking is supported "
            "and that the directory is writable, then run the workflow again."
        ) from exc


def release_writer_lock(stream: BinaryIO) -> None:
    """Release and close a writer lock."""

    try:
        stream.seek(0)
        if os.name == "nt":
            _windows_lock(stream, mode_name="LK_UNLCK")
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                stream.fileno(),
                fcntl.LOCK_UN,  # type: ignore[attr-defined]
            )
    finally:
        stream.close()


def _windows_lock(
    stream: BinaryIO,
    *,
    mode_name: Literal["LK_NBLCK", "LK_UNLCK"],
) -> None:
    """Apply the requested Windows byte lock."""

    import msvcrt

    locking = cast(Callable[[int, int, int], None], getattr(msvcrt, "locking"))
    mode = cast(int, getattr(msvcrt, mode_name))
    locking(stream.fileno(), mode, 1)
