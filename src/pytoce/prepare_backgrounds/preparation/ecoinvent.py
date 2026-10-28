"""Licensed ecoinvent 3.12 import and Brightway project ownership."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import py7zr

from pytoce.download_data.sources.catalogue import (
    SourceSpec,
    owner_complete,
    source_directory,
)

ECOINVENT_IMPORT_METHOD = "pytoce_ecoinvent_native_import_v1"


@dataclass(frozen=True, slots=True)
class EcoinventInputs:
    """Licensed paths and file hashes recorded for fresh preparation."""

    archive: Path
    archive_sha256: str
    archive_manifest: Path
    ef_workbook: Path
    ef_workbook_sha256: str


@dataclass(frozen=True, slots=True)
class BrightwayNames:
    """Brightway store, project, and source database names for one source."""

    store: Path
    project: str
    biosphere_database: str
    source_database: str

    @property
    def writer_lock_path(self) -> Path:
        """Coordinate all package writers using this project across workspaces."""

        return self.store / f".{self.project}.writer.lock"


def resolve_licensed_inputs(
    workspace: Path,
    specs: tuple[SourceSpec, ...],
) -> EcoinventInputs:
    """Resolve completed source paths and recorded hashes from the catalog."""

    by_id = {spec.source_id: spec for spec in specs}
    ecospold = by_id["ecoinvent_3_12_cutoff_ecospold02"]
    lcia = by_id["ecoinvent_3_12_lcia_implementation"]
    root = source_directory(workspace, ecospold)
    if not owner_complete(root):
        raise FileNotFoundError(
            f"The required ecoinvent 3.12 files are not complete in {root}. Place the "
            "licensed EcoSpold and LCIA archives in data_raw/ecoinvent, then run "
            "download_data()."
        )
    return EcoinventInputs(
        archive=root / ecospold.filename,
        archive_sha256=ecospold.sha256,
        archive_manifest=root / "manifest.json",
        ef_workbook=root / str(lcia.metadata["workbook_filename"]),
        ef_workbook_sha256=str(lcia.metadata["workbook_sha256"]),
    )


def brightway_names(workspace: Path, *, source_identity: str) -> BrightwayNames:
    """Return deterministic Brightway project and database names."""

    digest = source_identity.split("::", maxsplit=1)[-1]
    suffix = digest[:12]
    return BrightwayNames(
        store=workspace / "data_processed" / "brightway",
        project=f"pytoce_{suffix}",
        biosphere_database=f"pytoce__biosphere_3_12__{suffix}",
        source_database=f"pytoce__ecoinvent_3_12_cutoff__{suffix}",
    )


def extract_ecoinvent(archive: Path, destination: Path) -> None:
    """Extract the admitted EcoSpold archive into the caller's working directory."""
    with py7zr.SevenZipFile(archive, mode="r") as handle:
        destination.mkdir(parents=True, exist_ok=False)
        handle.extractall(path=destination)


@contextmanager
def selected_brightway_project(names: BrightwayNames) -> Iterator[Any]:
    """Select an owned Brightway project and restore the caller's selection."""

    import bw2data as bd  # type: ignore[import-untyped]

    previous_base = bd.projects.dir.parent
    previous_logs = bd.projects.logs_dir.parent
    previous_project = bd.projects.current
    names.store.mkdir(parents=True, exist_ok=True)
    (names.store / "logs").mkdir(parents=True, exist_ok=True)
    try:
        bd.projects.change_base_directories(
            names.store,
            base_logs_dir=names.store / "logs",
            project_name=names.project,
            update=False,
        )
        yield bd
    finally:
        bd.projects.change_base_directories(
            previous_base,
            base_logs_dir=previous_logs,
            project_name=previous_project,
            update=False,
        )


def import_ecoinvent(
    extracted: Path,
    names: BrightwayNames,
) -> dict[str, object]:
    """Import native inventories using the public bw2io 0.9.17 strategies.

    Licensed source files remain unchanged. Quantities, links and uncertainty
    follow bw2io's EcoSpold conversion; study-specific inventory adaptations
    belong to their provider owners after this import.
    """

    import bw2data as bd  # type: ignore[import-untyped]
    from bw2io.importers import (  # type: ignore[import-untyped]
        Ecospold2BiosphereImporter,
        SingleOutputEcospold2Importer,
    )

    biosphere = cast(Any, Ecospold2BiosphereImporter)(
        name=names.biosphere_database,
        version="3.12",
        filepath=str(extracted / "MasterData" / "ElementaryExchanges.xml"),
    )
    biosphere.apply_strategies(verbose=False)
    biosphere.write_database()
    bd.preferences["biosphere_database"] = names.biosphere_database

    importer = cast(Any, SingleOutputEcospold2Importer)(
        dirpath=str(extracted / "datasets"),
        db_name=names.source_database,
        biosphere_database_name=names.biosphere_database,
        use_mp=True,
        reparametrize_lognormals=False,
        add_product_information=True,
        separate_products=False,
        cache=False,
    )
    importer.apply_strategies(verbose=False)
    importer.write_database(delete_existing=False, check_typos=False)
    return {"use_mp": True, "method": ECOINVENT_IMPORT_METHOD}
