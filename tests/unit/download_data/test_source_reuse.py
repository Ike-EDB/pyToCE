import errno
import shutil
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from pytoce.download_data.download_data import _download_from_catalogue
from pytoce.download_data.sources.catalogue import (
    SourceSpec,
    manifest_bytes,
    source_directory,
)
from pytoce.download_data.sources.ecoinvent import acquire_ecoinvent
from pytoce.download_data.sources.network import (
    DownloadTransport,
    acquire_open_owner,
    download_source,
)
from pytoce.set_workspace.set_workspace import set_workspace


def _local_file_url(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc:
        raise AssertionError("The deterministic source fixture must use a local file URL")
    return Path(url2pathname(parsed.path))


def _place_licensed_archives(
    workspace: Path,
    specs: tuple[SourceSpec, ...],
    sources: tuple[Path, ...],
) -> None:
    root = workspace / "data_raw" / "ecoinvent"
    root.mkdir(parents=True, exist_ok=True)
    for spec, source in zip(specs, sources, strict=True):
        shutil.copyfile(source, root / spec.filename)


def _write_open_source_owner(root: Path, specs: tuple[SourceSpec, ...]) -> None:
    """Write the exact files recognized as one completed open-source owner."""

    root.mkdir(parents=True)
    (root / "README.md").write_text("test source owner\n", encoding="utf-8")
    (root / "manifest.json").write_bytes(manifest_bytes(specs))
    for spec in specs:
        assert spec.url is not None
        shutil.copyfile(_local_file_url(spec.url), root / spec.filename)


def test_download_transport_streams_a_catalogued_file_url(
    tmp_path: Path,
    acquisition_catalogue: tuple[
        tuple[SourceSpec, ...],
        tuple[SourceSpec, ...],
        tuple[Path, ...],
    ],
) -> None:
    """Exercise the real streaming transport against a local immutable source."""

    open_sources, _licensed, _archives = acquisition_catalogue
    source = open_sources[0]
    destination = tmp_path / source.filename

    download_source(source, destination)

    assert destination.read_bytes() == _local_file_url(source.url or "").read_bytes()


def test_reuses_readable_source_manifest_without_rehashing(
    tmp_path: Path,
    acquisition_catalogue: tuple[
        tuple[SourceSpec, ...],
        tuple[SourceSpec, ...],
        tuple[Path, ...],
    ],
    acquisition_transport: DownloadTransport,
) -> None:
    open_sources, licensed, archives = acquisition_catalogue
    workspace = tmp_path / "workspace"
    set_workspace(workspace)
    _place_licensed_archives(workspace, licensed, archives)

    initial = _download_from_catalogue(
        workspace,
        open_sources=open_sources,
        ecoinvent_specs=licensed,
        refresh=False,
        started=perf_counter(),
        transport=acquisition_transport,
    )
    initial.sources[0].artifact_path.write_bytes(b"user-managed same-name bytes")
    initial.sources[0].manifest_path.write_bytes(b"ordinary reuse does not reopen this file\n")
    for source in open_sources:
        assert source.url is not None
        _local_file_url(source.url).unlink(missing_ok=True)

    reused = _download_from_catalogue(
        workspace,
        open_sources=open_sources,
        ecoinvent_specs=licensed,
        refresh=False,
        started=perf_counter(),
        transport=acquisition_transport,
    )

    assert reused.status == "reused"
    assert {source.status for source in reused.sources} == {"reused"}


def test_registered_sources_create_the_missing_physical_recipe(
    tmp_path: Path,
    acquisition_catalogue: tuple[tuple[SourceSpec, ...], tuple[SourceSpec, ...], tuple[Path, ...]],
    acquisition_transport: DownloadTransport,
) -> None:
    """Create a missing recipe from registered local sources."""
    open_sources, licensed, archives = acquisition_catalogue
    workspace = tmp_path / "workspace"
    set_workspace(workspace)
    _place_licensed_archives(workspace, licensed, archives)
    for specs in (open_sources[:2], open_sources[2:]):
        acquire_open_owner(workspace, specs, refresh=False, transport=acquisition_transport)
    sources = acquire_ecoinvent(workspace, licensed)
    before = {source.artifact_path: source.artifact_path.read_bytes() for source in sources}
    for source in open_sources:
        assert source.url is not None
        _local_file_url(source.url).unlink()

    report = _download_from_catalogue(
        workspace,
        open_sources=open_sources,
        ecoinvent_specs=licensed,
        refresh=False,
        started=perf_counter(),
        transport=acquisition_transport,
    )

    assert report.status == "updated"
    assert {source.status for source in report.sources} == {"reused"}
    assert (workspace / "data_processed" / "ecoinvent" / "material_recipe.json").is_file()
    assert {path: path.read_bytes() for path in before} == before


def test_missing_licensed_archive_fails_after_open_source_reuse(
    tmp_path: Path,
    acquisition_catalogue: tuple[
        tuple[SourceSpec, ...],
        tuple[SourceSpec, ...],
        tuple[Path, ...],
    ],
    acquisition_transport: DownloadTransport,
) -> None:
    open_sources, licensed, _archives = acquisition_catalogue
    workspace = tmp_path / "workspace"
    set_workspace(workspace)

    with pytest.raises(FileNotFoundError, match="Place the licensed ecoinvent archive"):
        _download_from_catalogue(
            workspace,
            open_sources=open_sources,
            ecoinvent_specs=licensed,
            refresh=False,
            started=perf_counter(),
            transport=acquisition_transport,
        )


def test_failed_refresh_preserves_completed_source_owner(
    tmp_path: Path,
    acquisition_catalogue: tuple[
        tuple[SourceSpec, ...],
        tuple[SourceSpec, ...],
        tuple[Path, ...],
    ],
    acquisition_transport: DownloadTransport,
) -> None:
    open_sources, licensed, archives = acquisition_catalogue
    workspace = tmp_path / "workspace"
    set_workspace(workspace)
    _place_licensed_archives(workspace, licensed, archives)
    initial = _download_from_catalogue(
        workspace,
        open_sources=open_sources,
        ecoinvent_specs=licensed,
        refresh=False,
        started=perf_counter(),
        transport=acquisition_transport,
    )
    remind = tuple(source for source in open_sources if source.source_id.startswith("remind_mfa_"))
    before = {
        source.artifact_path: source.artifact_path.read_bytes() for source in initial.sources[:2]
    }
    manifest = initial.sources[0].manifest_path
    manifest_bytes = manifest.read_bytes()

    def fail_second(source: SourceSpec, destination: Path) -> None:
        if source == remind[1]:
            raise OSError("bounded transport failure")
        assert source.url is not None
        shutil.copyfile(_local_file_url(source.url), destination)

    with pytest.raises(OSError, match="bounded transport failure"):
        acquire_open_owner(
            workspace,
            remind,
            refresh=True,
            transport=fail_second,
        )

    assert manifest.read_bytes() == manifest_bytes
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("not_directory", "exists without a completed package download"),
        ("owner_symlink", "symbolic link"),
        ("unexpected_file", "unexpected.txt"),
        ("artifact_directory", "not a regular file"),
        ("guide_directory", "not a regular file"),
    ),
)
def test_refresh_rejects_changed_source_owners(
    tmp_path: Path,
    acquisition_catalogue: tuple[
        tuple[SourceSpec, ...],
        tuple[SourceSpec, ...],
        tuple[Path, ...],
    ],
    acquisition_transport: DownloadTransport,
    corruption: str,
    message: str,
) -> None:
    """Preserve unrelated files and directories when source refresh is requested."""

    open_sources, _licensed, _archives = acquisition_catalogue
    specs = open_sources[:2]
    root = source_directory(tmp_path, specs[0])
    if corruption == "not_directory":
        root.parent.mkdir(parents=True)
        root.write_text("not a directory\n", encoding="utf-8")
    elif corruption == "owner_symlink":
        target = tmp_path / "external_source"
        _write_open_source_owner(target, specs)
        root.parent.mkdir(parents=True)
        try:
            root.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            if (
                exc.errno in {errno.EACCES, errno.EPERM, errno.ENOTSUP}
                or getattr(exc, "winerror", None) == 1314
            ):
                pytest.skip(f"Directory symbolic links are unavailable: {exc}")
            raise
    else:
        _write_open_source_owner(root, specs)
        if corruption == "unexpected_file":
            (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
        else:
            artifact = root / (
                "README.md" if corruption == "guide_directory" else specs[0].filename
            )
            artifact.unlink()
            artifact.mkdir()
            (artifact / "user-note.txt").write_text("keep this note\n", encoding="utf-8")

    before = (
        {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        if root.is_dir()
        else {Path("."): root.read_bytes()}
    )

    with pytest.raises(ValueError, match=message):
        acquire_open_owner(tmp_path, specs, refresh=True, transport=acquisition_transport)

    after = (
        {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        if root.is_dir()
        else {Path("."): root.read_bytes()}
    )
    assert after == before
    if corruption == "owner_symlink":
        assert root.is_symlink()
        assert root.resolve() == tmp_path / "external_source"


@pytest.mark.parametrize("refresh", (False, True))
def test_incomplete_source_owner_requires_a_new_workspace(
    tmp_path: Path,
    acquisition_catalogue: tuple[
        tuple[SourceSpec, ...],
        tuple[SourceSpec, ...],
        tuple[Path, ...],
    ],
    refresh: bool,
) -> None:
    """Preserve incomplete source directories under either refresh setting."""

    open_sources, _licensed, _archives = acquisition_catalogue
    specs = open_sources[:2]
    root = source_directory(tmp_path, specs[0])
    root.mkdir(parents=True)

    with pytest.raises(ValueError, match="exists without a completed package download"):
        acquire_open_owner(
            tmp_path,
            specs,
            refresh=refresh,
            transport=lambda _source, _destination: None,
        )
