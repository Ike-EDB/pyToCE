"""Prepare and save shared Brightway, LCIA, and provider data."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import resources
from math import isnan
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq

from pytoce.shared.runtime.file_lock import acquire_writer_lock, release_writer_lock
from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    payload_sha256,
    replace_directory,
)
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

from ..ecoinvent import (
    BrightwayNames,
    brightway_names,
    extract_ecoinvent,
    import_ecoinvent,
    selected_brightway_project,
)
from ..project_biosphere import (
    lcia_authority_identity,
    load_lcia_catalogue,
    register_methods,
    resolve_ef_methods,
    resolve_pb_methods,
)
from ..providers.catalogue import (
    PreparedProviderAuthority,
    _service_source_closures,
    provider_timeline_database_name,
)
from ..providers.interface_database import (
    PROVIDER_TIMELINE_YEARS,
    provider_timeline_databases,
    provider_timeline_identity,
    write_provider_interface,
)
from ..providers.material_carbon import prepare_material_carbon
from ..providers.native_resolution import _ActivityResolver, _FlowResolver
from ..providers.pa_receiving_sources import (
    PaReceivingReference,
    admit_pa_receiving_reference,
    reference_from_payload,
    reference_to_payload,
)
from ..providers.recipe_context import _RecipeContext
from ..providers.route_authorities import resolve_open_dump_routes
from .authority_tables import SHARED_SCHEMA, database_row, read_prepared_manifest, write_table
from .source_authority import Sources


@dataclass(frozen=True, slots=True)
class SharedAuthority:
    """Prepared data shared by every background scenario.

    Attributes:
        names: Names and local storage path for the workspace's Brightway project.
        identity: Content identifier for the shared sources, methods, providers
            and scientific table contents.
        source_identity: Identifier of the admitted input hashes, source manifests,
            material and provider recipes, and dependency versions.
        preparation_method_identity: Version identifier for the preparation method.
        lcia_identity: Content identifier calculated from the 26 registered LCIA categories.
        provider_identity: Content identifier calculated from the provider catalog and sources.
        provider_interface_database: Brightway database containing the provider services.
        provider_timeline_identity: Content identifier for the dated provider database series.
        provider_timeline_database_by_year: Provider database name for each supported year.
        provider_routes: Rows mapping plastic demands to provider services.
        impact_categories: Ordered EF v3.1 and PB-LCIA category records.
        database_rows: Prepared Brightway database records.
        matrix_rows: Records for the processed Brightway matrices.
        biosphere_rows: Records identifying the ecoinvent 3.12 biosphere database.
        pa_receiving_reference: Original thermal, sanitary/water-chain and fuel calibration for PA
            fiber-residue receiving.
    """

    names: BrightwayNames
    identity: str
    source_identity: str
    preparation_method_identity: str
    lcia_identity: str
    provider_identity: str
    provider_interface_database: str
    provider_timeline_identity: str
    provider_timeline_database_by_year: dict[int, str]
    provider_routes: tuple[dict[str, object], ...]
    impact_categories: tuple[dict[str, object], ...]
    database_rows: tuple[dict[str, object], ...]
    matrix_rows: tuple[dict[str, object], ...]
    biosphere_rows: tuple[dict[str, object], ...]
    pa_receiving_reference: PaReceivingReference


def shared_root(workspace: Path) -> Path:
    """Return the readable shared prepared-background directory."""

    return workspace / "data_processed" / "prepared_backgrounds" / "shared"


def load_shared(
    workspace: Path,
    *,
    names: BrightwayNames,
) -> SharedAuthority | None:
    """Read completed shared data for preparation."""

    root = shared_root(workspace)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    reference = read_prepared_manifest(manifest)
    payload = reference.payload
    identity_payload = cast(dict[str, object], payload["identity_payload"])
    method = cast(
        dict[str, object],
        identity_payload["preparation_method"],
    )
    provider_payload = json.loads(
        (reference.path.parent / "provider_service_manifest.json").read_bytes()
    )
    biosphere_rows = tuple(
        pq.read_table(reference.path.parent / "biosphere_authority.parquet").to_pylist()
    )
    provider_routes = tuple(
        pq.read_table(reference.path.parent / "provider_service_routes.parquet").to_pylist()
    )
    impact_categories = tuple(
        pq.read_table(reference.path.parent / "impact_categories.parquet").to_pylist()
    )
    database_rows = tuple(
        pq.read_table(reference.path.parent / "database_authority.parquet").to_pylist()
    )
    matrix_rows = tuple(
        pq.read_table(reference.path.parent / "matrix_authority.parquet").to_pylist()
    )
    return SharedAuthority(
        names=names,
        identity=str(payload["identity"]),
        source_identity=str(identity_payload["source_identity"]),
        preparation_method_identity=str(method["identity"]),
        lcia_identity=str(payload["lcia_authority_identity"]),
        provider_identity=str(provider_payload["identity"]),
        provider_interface_database=str(provider_payload["interface_database"]),
        provider_timeline_identity=str(provider_payload["timeline_identity"]),
        provider_timeline_database_by_year={
            int(year): str(database)
            for year, database in cast(
                dict[str, str],
                provider_payload["timeline_database_by_year"],
            ).items()
        },
        provider_routes=provider_routes,
        impact_categories=impact_categories,
        database_rows=database_rows,
        matrix_rows=matrix_rows,
        biosphere_rows=biosphere_rows,
        pa_receiving_reference=reference_from_payload(
            json.loads((root / "pa_receiving_reference.json").read_bytes())
        ),
    )


def write_shared_data(
    workspace: Path,
    sources: Sources,
    names: BrightwayNames,
    *,
    provider: PreparedProviderAuthority,
    pa_receiving_reference: PaReceivingReference,
    material_carbon_rows: tuple[dict[str, object], ...],
    impact_rows: tuple[dict[str, object], ...],
    database_rows: tuple[dict[str, object], ...],
    matrix_rows: tuple[dict[str, object], ...],
    biosphere_rows: tuple[dict[str, object], ...],
    import_report: dict[str, object],
) -> SharedAuthority:
    """Write background data shared by every REMIND scenario.

    The result contains LCIA categories, Brightway database and matrix records,
    provider routes, provider source and method records, the U.S. LCI notice,
    and the source-import report. File and table digests are included in an
    identity derived from the downloaded sources, dependency versions,
    Brightway project names, LCIA methods, and provider catalog. The completion
    manifest is written last before the complete directory is moved into place.

    Args:
        workspace: Active pyToCE workspace.
        sources: Recorded source paths, checksums, and preparation authorities.
        names: Brightway project, database, and store names for this workspace.
        provider: Compiled provider routes and their source and method records.
        pa_receiving_reference: Original HWI, water and sanitary landfill calibration,
            including gas heating values shared by PA receiving anchors.
        material_carbon_rows: Constituent carbon by virgin route and ultimate origin.
        impact_rows: Installed LCIA method records.
        database_rows: Shared Brightway database records.
        matrix_rows: Processed Brightway matrix records.
        biosphere_rows: Record identifying the shared ecoinvent biosphere database.
        import_report: Import method and multiprocessing setting recorded by
            the ecoinvent importer.

    Returns:
        Paths, identities, provider routes, and LCIA records needed to prepare
        each background scenario.
    """

    timeline_databases = provider_timeline_databases(provider.interface_database)
    timeline_identity = provider_timeline_identity(
        provider.identity, interface_database=provider.interface_database
    )
    root = shared_root(workspace)
    staging = Path(tempfile.mkdtemp(prefix=".shared_background.", dir=root.parent))
    try:
        provider_routes: tuple[dict[str, object], ...] = tuple(
            cast(
                dict[str, object],
                {
                    str(field): (None if isinstance(value, float) and isnan(value) else value)
                    for field, value in record.items()
                },
            )
            for record in provider.routes.to_dict(orient="records")
        )
        artifacts = {
            "impact_categories.parquet": write_table(
                staging / "impact_categories.parquet", impact_rows
            ),
            "database_authority.parquet": write_table(
                staging / "database_authority.parquet", database_rows
            ),
            "matrix_authority.parquet": write_table(
                staging / "matrix_authority.parquet", matrix_rows
            ),
            "biosphere_authority.parquet": write_table(
                staging / "biosphere_authority.parquet", biosphere_rows
            ),
            "provider_service_routes.parquet": write_table(
                staging / "provider_service_routes.parquet", provider_routes
            ),
            "material_carbon.parquet": write_table(
                staging / "material_carbon.parquet", material_carbon_rows
            ),
            "provider_source_authorities.parquet": write_table(
                staging / "provider_source_authorities.parquet",
                provider.source_authorities,
            ),
            "provider_method_authorities.parquet": write_table(
                staging / "provider_method_authorities.parquet",
                provider.method_authorities,
            ),
        }
        lcia_identity = lcia_authority_identity(impact_rows)
        pa_reference_bytes = canonical_json_bytes(reference_to_payload(pa_receiving_reference))
        atomic_write(staging / "pa_receiving_reference.json", pa_reference_bytes)
        pa_reference_identity = sha256(pa_reference_bytes).hexdigest()
        artifacts["pa_receiving_reference.json"] = {
            "path": "pa_receiving_reference.json",
            "artifact_sha256": pa_reference_identity,
            "scientific_identity": pa_reference_identity,
        }
        identity_payload = {
            "schema_version": SHARED_SCHEMA,
            "source_identity": sources.source_identity,
            "source_identity_payload": sources.source_identity_payload,
            "dependency_authorities": sources.dependency_authorities,
            "preparation_method": sources.preparation_method,
            "brightway_project": names.project,
            "biosphere_database": names.biosphere_database,
            "source_database": names.source_database,
            "lcia_authority_identity": lcia_identity,
            "provider_authority_identity": provider.identity,
            "provider_interface_database": provider.interface_database,
            "provider_timeline_identity": timeline_identity,
            "provider_timeline_database_by_year": timeline_databases,
            "table_identities": {
                name: descriptor["scientific_identity"] for name, descriptor in artifacts.items()
            },
        }
        identity = f"{SHARED_SCHEMA}::{payload_sha256(identity_payload)}"
        notice = (
            resources.files("pytoce.prepare_backgrounds.data")
            .joinpath("USLCI_DATA_USE_NOTICE.txt")
            .read_bytes()
        )
        atomic_write(staging / "USLCI_DATA_USE_NOTICE.txt", notice)
        artifacts["USLCI_DATA_USE_NOTICE.txt"] = {
            "path": "USLCI_DATA_USE_NOTICE.txt",
            "artifact_sha256": sha256(notice).hexdigest(),
        }
        report = {
            "schema_version": "pytoce_background_preparation_report_schema_v1",
            "source_import": import_report,
            "lcia_authority_identity": lcia_identity,
        }
        atomic_write(staging / "preparation_report.json", canonical_json_bytes(report))
        artifacts["preparation_report.json"] = {
            "path": "preparation_report.json",
            "artifact_sha256": file_sha256(staging / "preparation_report.json"),
        }
        provider_manifest = {
            "schema_version": "pytoce_provider_service_authority_schema_v1",
            "identity": provider.identity,
            "interface_database": provider.interface_database,
            "timeline_identity": timeline_identity,
            "timeline_database_by_year": timeline_databases,
            "routes": artifacts["provider_service_routes.parquet"],
            "material_carbon": artifacts["material_carbon.parquet"],
            "sources": artifacts["provider_source_authorities.parquet"],
            "methods": artifacts["provider_method_authorities.parquet"],
            "complete": True,
        }
        atomic_write(
            staging / "provider_service_manifest.json",
            canonical_json_bytes(provider_manifest),
        )
        artifacts["provider_service_manifest.json"] = {
            "path": "provider_service_manifest.json",
            "artifact_sha256": file_sha256(staging / "provider_service_manifest.json"),
        }
        write_workspace_guide(
            staging,
            "pytoce.prepare_backgrounds",
            "prepared_shared",
        )
        atomic_write(
            staging / "manifest.json",
            canonical_json_bytes(
                {
                    "schema_version": SHARED_SCHEMA,
                    "identity": identity,
                    "identity_payload": identity_payload,
                    "lcia_authority_identity": lcia_identity,
                    "provider_authority_identity": provider.identity,
                    "provider_interface_database": provider.interface_database,
                    "provider_timeline_identity": timeline_identity,
                    "provider_timeline_database_by_year": timeline_databases,
                    "brightway_project": names.project,
                    "brightway_store_relative_path": Path(
                        os.path.relpath(names.store, workspace)
                    ).as_posix(),
                    "biosphere_database": names.biosphere_database,
                    "source_database": names.source_database,
                    "artifacts": artifacts,
                    "complete": True,
                }
            ),
        )
        write_workspace_guide(
            root.parent,
            "pytoce.prepare_backgrounds",
            "prepared_backgrounds",
        )
        replace_directory(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return SharedAuthority(
        names=names,
        identity=identity,
        source_identity=sources.source_identity,
        preparation_method_identity=str(sources.preparation_method["identity"]),
        lcia_identity=lcia_identity,
        provider_identity=provider.identity,
        provider_interface_database=provider.interface_database,
        provider_timeline_identity=timeline_identity,
        provider_timeline_database_by_year=timeline_databases,
        provider_routes=provider_routes,
        impact_categories=impact_rows,
        database_rows=database_rows,
        matrix_rows=matrix_rows,
        biosphere_rows=biosphere_rows,
        pa_receiving_reference=pa_receiving_reference,
    )


def _ensure_shared_authority_unlocked(
    workspace: Path, sources: Sources, names: BrightwayNames
) -> SharedAuthority:
    """Create or reuse the shared Brightway data while the writer lock is held.

    Args:
        workspace: Active pyToCE workspace.
        sources: Recorded ecoinvent, EF v3.1, and PB-LCIA paths and file hashes.
        names: Selected project names whose writer lock is held.

    Returns:
        Prepared Brightway names, identities, LCIA categories, and provider routes.

    Raises:
        RuntimeError: If existing data have a different identity, or a partial output
            or Brightway name collision is present.

    Notes:
        A new result imports ecoinvent 3.12, registers the 26 selected LCIA categories,
        and writes the dated inventory-neutral provider interfaces. Existing completed
        data are reused only when their source, method, provider, and provider
        timeline identities match.
    """

    root = shared_root(workspace)
    root.parent.mkdir(parents=True, exist_ok=True)
    provider = sources.provider
    existing = load_shared(workspace, names=names)
    if existing is not None:
        if (
            existing.source_identity != sources.source_identity
            or existing.preparation_method_identity != sources.preparation_method["identity"]
            or existing.provider_identity != provider.identity
            or existing.provider_timeline_identity
            != provider_timeline_identity(
                provider.identity, interface_database=provider.interface_database
            )
            or existing.provider_timeline_database_by_year
            != provider_timeline_databases(provider.interface_database)
        ):
            raise RuntimeError(
                f"The completed shared background data at {root} were created from different "
                "source files, package code, or provider data. Leave that directory unchanged, "
                "create a new workspace, and run prepare_backgrounds there."
            )
        return existing
    if root.exists() or root.is_symlink():
        raise RuntimeError(
            f"The shared prepared-background directory {root} has no completion manifest. "
            "It was left unchanged. Create a new workspace and run the workflows there."
        )

    database_names = (
        names.biosphere_database,
        names.source_database,
        *(
            provider_timeline_database_name(year, interface_database=provider.interface_database)
            for year in PROVIDER_TIMELINE_YEARS
        ),
    )
    specs, _catalogue = load_lcia_catalogue()
    with selected_brightway_project(names) as bd:
        database_collisions = sorted(set(database_names).intersection(bd.databases))
        method_collisions = sorted(
            spec.method_id for spec in specs if spec.impact_category_tuple in bd.methods
        )
    if database_collisions or method_collisions:
        details = [
            *(f"database {name!r}" for name in database_collisions),
            *(f"LCIA method {name!r}" for name in method_collisions),
        ]
        raise RuntimeError(
            f"The Brightway project for {workspace} already contains "
            f"{', '.join(details)} needed by prepare_backgrounds. Create a new workspace "
            "and run the workflows there; the existing project has not been changed."
        )
    work = Path(tempfile.mkdtemp(prefix=".shared_background_work.", dir=root.parent))
    try:
        extracted = work / "ecoinvent"
        extract_ecoinvent(sources.ecoinvent_archive, extracted)
        with selected_brightway_project(names) as bd:
            import_report = import_ecoinvent(extracted, names)
            # Admit original receiving coefficients once. Study
            # inventories are written only to the dated provider frontiers;
            # the imported ecoinvent database remains the native source.
            native = _ActivityResolver(bd, names.source_database)
            context = _RecipeContext(
                background_scenario=None,
                year=2025,
                provider_database=provider.interface_database,
                source=native,
                ordinary=native,
                technology=native,
                biosphere=_FlowResolver(bd, names.biosphere_database),
                catalogue=provider.catalogue,
                material_recipe=sources.material_recipe,
                method_authorities={
                    str(row["method_authority_id"]): row for row in provider.method_authorities
                },
                complete_source_ids=_service_source_closures(provider.catalogue),
                semantic_rows=[],
                technology_rows=[],
                traced_services=set(),
            )
            pa_receiving_reference = admit_pa_receiving_reference(context)
            routes, route_source = resolve_open_dump_routes(names.source_database, provider.routes)
            provider = replace(
                provider,
                routes=routes,
                source_authorities=tuple(
                    {**row, **route_source}
                    if row["source_id"] == route_source["source_id"]
                    else row
                    for row in provider.source_authorities
                ),
            )
            material_carbon_rows = prepare_material_carbon(
                names.source_database, provider.catalogue
            )
            ef_methods = resolve_ef_methods(
                sources.ef_workbook,
                biosphere_database=names.biosphere_database,
                specs=specs,
            )
            pb_methods = resolve_pb_methods(
                sources.pb_archive,
                biosphere_database=names.biosphere_database,
                specs=specs,
            )
            resolved = tuple(sorted((*ef_methods, *pb_methods), key=lambda item: item.spec.order))
            impact_rows = register_methods(
                resolved,
                source_identity=sources.source_identity,
            )
            write_provider_interface(bd, names.biosphere_database, provider)
            database_rows = (
                database_row(names.biosphere_database, role="ecoinvent_biosphere"),
                database_row(names.source_database, role="ecoinvent_3_12_cutoff_base"),
                *(
                    database_row(
                        provider_timeline_database_name(
                            year, interface_database=provider.interface_database
                        ),
                        role=(
                            "provider_service_interface"
                            if year == PROVIDER_TIMELINE_YEARS[0]
                            else "provider_timeline_interpolation"
                        ),
                    )
                    for year in PROVIDER_TIMELINE_YEARS
                ),
            )
            matrix_rows = tuple(
                {
                    "database_name": row["database_name"],
                }
                for row in database_rows
            )
            biosphere_rows: tuple[dict[str, object], ...] = (
                {
                    "authority_role": "ecoinvent_biosphere",
                    "ecoinvent_biosphere_database": names.biosphere_database,
                },
            )
        shared = write_shared_data(
            workspace,
            sources,
            names,
            provider=provider,
            pa_receiving_reference=pa_receiving_reference,
            material_carbon_rows=material_carbon_rows,
            impact_rows=impact_rows,
            database_rows=database_rows,
            matrix_rows=matrix_rows,
            biosphere_rows=biosphere_rows,
            import_report=import_report,
        )
        return shared
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ensure_shared_authority(workspace: Path, sources: Sources) -> SharedAuthority:
    """Create or reuse shared Brightway, LCIA, and provider data under one writer lock."""

    names = brightway_names(workspace, source_identity=sources.source_identity)
    names.store.mkdir(parents=True, exist_ok=True)
    stream = acquire_writer_lock(names.writer_lock_path)
    try:
        return _ensure_shared_authority_unlocked(workspace, sources, names)
    finally:
        release_writer_lock(stream)
