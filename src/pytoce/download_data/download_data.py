"""Public coordinator for source downloads and licensed-input registration."""

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from pytoce.shared.runtime.manifests import OperationStatus

if TYPE_CHECKING:
    from .sources.catalogue import AcquiredSource, SourceSpec
    from .sources.network import DownloadTransport


@dataclass(frozen=True, slots=True)
class DownloadDataReport:
    """Result of :func:`pytoce.download_data`.

    from the packaged source catalog and completion status."""

    status: OperationStatus
    workspace: Path
    elapsed_seconds: float
    sources: tuple["AcquiredSource", ...]

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary with paths converted to text."""

        return {
            "status": self.status,
            "workspace": str(self.workspace),
            "elapsed_seconds": self.elapsed_seconds,
            "sources": [source.as_dict() for source in self.sources],
        }


def _aggregate_status(sources: tuple["AcquiredSource", ...]) -> OperationStatus:
    """Return the aggregate status for the supplied records."""
    statuses = {source.status for source in sources}
    if statuses == {"computed"}:
        return "computed"
    if statuses == {"reused"}:
        return "reused"
    return "updated"


def _download_from_catalogue(
    root: Path,
    *,
    open_sources: tuple["SourceSpec", ...],
    ecoinvent_specs: tuple["SourceSpec", ...],
    refresh: bool,
    started: float,
    transport: "DownloadTransport",
) -> DownloadDataReport:
    """Download public sources and register licensed inputs specified by the catalog."""

    from .sources.ecoinvent import acquire_ecoinvent
    from .sources.ecoinvent_materials import prepare_material_recipe
    from .sources.network import acquire_open_owner

    remind = tuple(source for source in open_sources if source.source_id.startswith("remind_mfa_"))
    pb_lcia = tuple(source for source in open_sources if source.source_id == "aesa_pb_lcia")
    sources = (
        *acquire_open_owner(root, remind, refresh=refresh, transport=transport),
        *acquire_open_owner(root, pb_lcia, refresh=refresh, transport=transport),
        *acquire_ecoinvent(root, ecoinvent_specs),
    )
    ecospold = next(
        source
        for source in ecoinvent_specs
        if source.source_id == "ecoinvent_3_12_cutoff_ecospold02"
    )
    recipe_created = prepare_material_recipe(root, ecospold)
    status = _aggregate_status(sources)
    if recipe_created and status == "reused":
        status = "updated"
    report = DownloadDataReport(
        status=status,
        workspace=root,
        elapsed_seconds=perf_counter() - started,
        sources=sources,
    )
    print(
        "[pyToCE download_data]\n"
        f"status: {report.status}\n"
        + "\n".join(f"{source.source_id}: {source.status}" for source in report.sources),
        file=sys.stderr,
    )
    return report


def download_data(
    *,
    refresh: bool = False,
) -> DownloadDataReport:
    """Download public sources and register local licensed archives.

    The function downloads REMIND-MFA 0.2.0, its input data 1.0.0, and the
    AESA PB-LCIA 0.9.2 characterization factors archive. It verifies and registers the licensed
    ecoinvent archives supplied by the user, and derives monomer requirements
    and chemical-recovery yields from selected EcoSpold records. The derived
    material recipe is stored under ``data_processed/ecoinvent/``.

    Before the first call, place ``ecoinvent_3.12_cutoff_ecoSpold02.7z`` and
    ``ecoinvent 3.12_LCIA_implementation.7z`` under ``data_raw/ecoinvent/``
    in the selected workspace.

    Args:
        refresh: If ``True``, download the cataloged public sources again.
            Existing public source directories must have a completion
            manifest and contain the cataloged files, their guide, and
            manifest. Defaults to ``False``, which reuses completed sources.
            Licensed ecoinvent archives are verified in place.

    Returns:
        ``DownloadDataReport`` describing completion status, elapsed time in
        seconds, and the version, revision, license, path, and file hash of
        each public or licensed input.

    Raises:
        RuntimeError: If no active workspace has been selected.
        FileNotFoundError: If a required user-supplied archive is absent.
        ValueError: If ``refresh`` is not a Boolean, source contents differ
            from the expected version, file hash, or structure, or an existing
            public source directory is incomplete or contains unexpected files.
        OSError: If a source cannot be downloaded, verified, or recorded.

    Notes:
        Call ``set_workspace()`` before invoking this function.
        Source manifests record version, revision, URL, license, and file
        hash. Completed sources are reused from their manifests. If the source
        archives are reused and a material recipe is newly created, the report
        status is ``updated`` and the source reports remain ``reused``.

    Example:
        Download public inputs after placing the licensed archives in the
        selected workspace::

            from pytoce import download_data

            report = download_data()
    """

    if not isinstance(refresh, bool):
        raise ValueError(
            f"refresh={refresh!r} is invalid. Use refresh=True to download public "
            "sources or refresh=False to reuse completed sources."
        )

    from pytoce.set_workspace.runtime.workspace import get_active_workspace

    from .sources import catalogue
    from .sources.network import download_source

    started = perf_counter()
    root = get_active_workspace()
    open_sources, ecoinvent_specs = catalogue.load_catalogue()
    return _download_from_catalogue(
        root,
        open_sources=open_sources,
        ecoinvent_specs=ecoinvent_specs,
        refresh=refresh,
        started=started,
        transport=download_source,
    )
