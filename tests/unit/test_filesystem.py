import errno
import os
from pathlib import Path

import pytest

from pytoce.shared.runtime.file_lock import (
    FileLockError,
    acquire_writer_lock,
    writer_lock,
)
from pytoce.shared.runtime.filesystem import replace_directory


def test_directory_replacement_preserves_an_incomplete_destination(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    destination.mkdir()
    note = destination / "unrecognized.txt"
    note.write_text("keep", encoding="utf-8")
    staging = tmp_path / ".result.staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("complete", encoding="utf-8")

    with pytest.raises(FileExistsError, match="directory already exists"):
        replace_directory(staging, destination)

    assert note.read_text(encoding="utf-8") == "keep"
    assert staging.is_dir()


def test_explicit_refresh_replaces_a_completed_destination(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    destination.mkdir()
    (destination / "owner_file.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".result.staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("new", encoding="utf-8")

    replace_directory(staging, destination, replace_existing=True)

    assert (destination / "manifest.json").read_text(encoding="utf-8") == "new"
    assert not staging.exists()


@pytest.mark.skipif(os.name == "nt", reason="Directory fsync requires a native POSIX handle")
def test_failed_directory_sync_restores_the_completed_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result"
    destination.mkdir()
    (destination / "manifest.json").write_text("old", encoding="utf-8")
    staging = tmp_path / ".result.staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("new", encoding="utf-8")

    def fail_sync(_descriptor: int) -> None:
        raise OSError("directory sync failed")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="directory sync failed"):
        replace_directory(staging, destination, replace_existing=True)

    assert (destination / "manifest.json").read_text(encoding="utf-8") == "old"
    assert (staging / "manifest.json").read_text(encoding="utf-8") == "new"
    assert not tuple(tmp_path.glob(".result.previous.*"))


@pytest.mark.skipif(os.name == "nt", reason="Directory fsync requires a native POSIX handle")
def test_failed_directory_sync_restores_new_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a new staged result available when directory synchronization fails."""

    destination = tmp_path / "result"
    staging = tmp_path / ".result.staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("new", encoding="utf-8")

    def fail_sync(_descriptor: int) -> None:
        raise OSError("directory sync failed")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="directory sync failed"):
        replace_directory(staging, destination)

    assert not destination.exists()
    assert (staging / "manifest.json").read_text(encoding="utf-8") == "new"


@pytest.mark.parametrize("replace_existing", [False, True], ids=["fresh", "refresh"])
def test_failed_attachment_preserves_staging_and_prior_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_existing: bool,
) -> None:
    """Preserve staged content and any prior result when attachment is rejected."""

    destination = tmp_path / "result"
    if replace_existing:
        destination.mkdir()
        (destination / "manifest.json").write_text("old", encoding="utf-8")
    staging = tmp_path / ".result.staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("new", encoding="utf-8")
    original_replace = os.replace

    def fail_staging_attachment(source: str | Path, target: str | Path) -> None:
        """Reject only the staged-directory move at the operating-system boundary."""
        if Path(source) == staging:
            raise OSError("directory attachment failed")
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_staging_attachment)
    with pytest.raises(OSError, match="directory attachment failed"):
        replace_directory(staging, destination, replace_existing=replace_existing)

    if replace_existing:
        assert (destination / "manifest.json").read_text(encoding="utf-8") == "old"
    else:
        assert not destination.exists()
    assert (staging / "manifest.json").read_text(encoding="utf-8") == "new"
    assert not tuple(tmp_path.glob(".result.previous.*"))


def test_directory_replacement_rejects_a_file_destination(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    destination.write_text("not a package result", encoding="utf-8")
    staging = tmp_path / ".result.staging"
    staging.mkdir()

    with pytest.raises(FileExistsError, match="not a package result directory"):
        replace_directory(staging, destination, replace_existing=True)

    assert destination.read_text(encoding="utf-8") == "not a package result"
    assert staging.is_dir()


def test_writer_lock_reports_a_filesystem_error_as_a_filesystem_error(tmp_path: Path) -> None:
    parent = tmp_path / "not_a_directory"
    parent.write_text("content", encoding="utf-8")

    with pytest.raises(FileLockError, match="cannot open.*Check that the parent"):
        acquire_writer_lock(parent / "writer.lock")


def test_writer_lock_reports_a_write_error_and_closes_the_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingStream:
        """Minimal stream that reports a disk write failure."""

        closed = False

        def tell(self) -> int:
            """Report a newly created empty lock file."""

            return 0

        def write(self, _content: bytes) -> None:
            """Represent a failed operating-system write."""

            raise OSError("disk full")

        def close(self) -> None:
            """Record resource cleanup after the failed write."""

            self.closed = True

    stream = FailingStream()
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: stream)

    with pytest.raises(FileLockError, match="cannot write.*disk space"):
        acquire_writer_lock(tmp_path / "writer.lock")

    assert stream.closed


def test_writer_lock_prevents_two_writers_and_releases_after_use(tmp_path: Path) -> None:
    path = tmp_path / "writer.lock"

    with writer_lock(path):
        with pytest.raises(FileLockError, match="Another package process"):
            acquire_writer_lock(path)

    with writer_lock(path):
        assert path.is_file()


def test_writer_lock_reports_an_unsupported_native_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate a native lock failure on the executing platform."""

    def fail_lock(_descriptor: int, _mode: int, *_length: int) -> None:
        """Represent a filesystem that rejects the native lock operation."""
        raise OSError(errno.EINVAL, "unsupported lock")

    if os.name == "nt":
        import msvcrt

        monkeypatch.setattr(msvcrt, "locking", fail_lock)
    else:
        import fcntl

        monkeypatch.setattr(fcntl, "flock", fail_lock)

    path = tmp_path / "writer.lock"
    with pytest.raises(FileLockError, match="operating system cannot lock"):
        acquire_writer_lock(path)

    path.unlink()
