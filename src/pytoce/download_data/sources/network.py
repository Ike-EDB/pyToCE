"""Download open source files into their workspace directories."""

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

from pytoce.shared.runtime.file_lock import writer_lock
from pytoce.shared.runtime.filesystem import atomic_write, replace_directory
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

from .catalogue import (
    DOWNLOAD_BLOCK_BYTES,
    AcquiredSource,
    SourceSpec,
    manifest_bytes,
    owner_complete,
    read_verified_source,
    source_directory,
    source_reports,
    validate_source_owner_for_refresh,
)

_USER_AGENT = "pytoce/0.1 source-acquisition"
DownloadTransport = Callable[[SourceSpec, Path], None]


def download_source(source: SourceSpec, destination: Path) -> None:
    """Stream one cataloged source through HTTP."""

    request = Request(cast(str, source.url), headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=120) as response, destination.open("wb") as stream:
        while block := response.read(DOWNLOAD_BLOCK_BYTES):
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())


def acquire_open_owner(
    workspace: Path,
    specs: tuple[SourceSpec, ...],
    *,
    refresh: bool,
    transport: DownloadTransport,
) -> tuple[AcquiredSource, ...]:
    """Download or reuse all files owned by one readable source directory."""

    root = source_directory(workspace, specs[0])
    root.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(root.parent / f".{root.name}.writer.lock"):
        completed = owner_complete(root)
        if completed and not refresh:
            return source_reports(root, specs, "reused")
        if root.exists() or root.is_symlink():
            if not completed:
                raise ValueError(
                    f"The source directory {root} exists without a completed package download. "
                    "Keep it unchanged and create a new workspace."
                )
            validate_source_owner_for_refresh(root, specs)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
        try:
            for spec in specs:
                artifact = staging / spec.filename
                transport(spec, artifact)
                read_verified_source(
                    artifact,
                    expected_bytes=spec.size_bytes,
                    expected_sha256=spec.sha256,
                )
            guide_id = (
                "remind_mfa_source"
                if specs[0].source_id.startswith("remind_mfa_")
                else "pb_lcia_source"
            )
            write_workspace_guide(
                staging,
                "pytoce.download_data",
                guide_id,
            )
            atomic_write(staging / "manifest.json", manifest_bytes(specs))
            replace_directory(staging, root, replace_existing=refresh)
            return source_reports(root, specs, "updated" if completed else "computed")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
