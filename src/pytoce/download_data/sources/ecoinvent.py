"""Licensed ecoinvent verification in its readable workspace folder."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import py7zr

from pytoce.shared.runtime.filesystem import atomic_write
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

from .catalogue import (
    AcquiredSource,
    SourceSpec,
    manifest_bytes,
    owner_complete,
    read_verified_source,
    source_directory,
    source_reports,
)


def _extract_lcia_workbook(
    archive_path: Path,
    spec: SourceSpec,
    *,
    destination: Path,
) -> None:
    """Extract and publish the workbook from the same authenticated source bytes."""

    archive_bytes = read_verified_source(
        archive_path,
        expected_bytes=spec.size_bytes,
        expected_sha256=spec.sha256,
    )
    workbook_member = str(spec.metadata["workbook_member"])
    with TemporaryDirectory(prefix=".ecoinvent_lcia.", dir=destination.parent) as staging:
        with BytesIO(archive_bytes) as stream, py7zr.SevenZipFile(stream, mode="r") as archive:
            archive.extract(path=staging, targets=[workbook_member])
        extracted = Path(staging) / workbook_member
        atomic_write(destination, extracted.read_bytes())


def acquire_ecoinvent(
    workspace: Path,
    specs: tuple[SourceSpec, ...],
) -> tuple[AcquiredSource, ...]:
    """Verify both documented licensed archives and write their manifest."""

    by_id = {spec.source_id: spec for spec in specs}
    ecospold = by_id["ecoinvent_3_12_cutoff_ecospold02"]
    lcia = by_id["ecoinvent_3_12_lcia_implementation"]
    root = source_directory(workspace, ecospold)
    if owner_complete(root):
        return source_reports(root, specs, "reused")
    missing = tuple(spec.filename for spec in specs if not (root / spec.filename).is_file())
    if missing:
        raise FileNotFoundError(
            "Place the licensed ecoinvent archives under data_raw/ecoinvent/: " + ", ".join(missing)
        )
    read_verified_source(
        root / ecospold.filename,
        expected_bytes=ecospold.size_bytes,
        expected_sha256=ecospold.sha256,
    )
    workbook = root / str(lcia.metadata["workbook_filename"])
    _extract_lcia_workbook(
        root / lcia.filename,
        lcia,
        destination=workbook,
    )
    write_workspace_guide(
        root,
        "pytoce.download_data",
        "ecoinvent_source",
    )
    atomic_write(root / "manifest.json", manifest_bytes(specs))
    return source_reports(root, specs, "computed")
