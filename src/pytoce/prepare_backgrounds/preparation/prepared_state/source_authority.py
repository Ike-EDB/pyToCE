"""Resolve the source files and methods used for background preparation."""

from dataclasses import dataclass, replace
from importlib.metadata import version as distribution_version
from pathlib import Path

from pytoce.download_data.sources.catalogue import (
    load_catalogue,
    owner_complete,
    source_directory,
)
from pytoce.download_data.sources.ecoinvent_materials import (
    EcoinventMaterialRecipe,
    load_material_recipe,
)
from pytoce.shared.runtime.filesystem import file_sha256, payload_sha256

from ..ecoinvent import ECOINVENT_IMPORT_METHOD, resolve_licensed_inputs
from ..premise import installed_preparation_dependency_authorities
from ..providers.catalogue import (
    PreparedProviderAuthority,
    prepared_provider_authority,
    provider_interface_database_name,
)

_PREPARATION_METHOD = "pytoce_background_preparation_with_constituent_carbon_v50"
_PREMISE_TRANSFORMATION_METHOD = "pytoce_premise_transformation_method_v1"


@dataclass(frozen=True, slots=True)
class Sources:
    """Admitted source paths, provenance and methods for background preparation.

    ``download_data`` owns source verification; preparation reuses those files directly.
    ``source_identity`` hashes the explicit ``source_identity_payload``, including
    the source-manifest hashes and exact ecoinvent/dependency versions. Manifest
    contents are opaque provenance; scientific payload digests come from the
    packaged catalog. Dependency and preparation authorities identify the
    scientific libraries and pyToCE transformation. The completed material
    recipe supplies manufacturing, recovery and native dry-operation bases;
    its identity is a source parent. The packaged provider catalog and its
    compilation method define the separate study inventories. Their identity
    is independent of generated databases; the admitted provider object passes
    to the shared interface and dated frontiers.
    """

    ecoinvent_archive: Path
    ef_workbook: Path
    pb_archive: Path
    source_identity: str
    source_identity_payload: dict[str, object]
    dependency_authorities: tuple[dict[str, object], ...]
    premise_authority: dict[str, object]
    premise_authority_identity: str
    preparation_method: dict[str, object]
    material_recipe: EcoinventMaterialRecipe
    provider: PreparedProviderAuthority


def preparation_method() -> dict[str, object]:
    """Identify the pyToCE method and package version used for preparation."""

    payload: dict[str, object] = {
        "schema_version": "pytoce_background_preparation_method_schema_v1",
        "method": _PREPARATION_METHOD,
        "package_version": distribution_version("pytoce"),
    }
    return {**payload, "identity": _PREPARATION_METHOD + "::" + payload_sha256(payload)}


def premise_transformation_method() -> dict[str, object]:
    """Identify the pyToCE transformation applied around premise."""

    payload: dict[str, object] = {
        "schema_version": "pytoce_premise_transformation_method_schema_v1",
        "method": _PREMISE_TRANSFORMATION_METHOD,
    }
    return {**payload, "identity": _PREMISE_TRANSFORMATION_METHOD + "::" + payload_sha256(payload)}


def resolve_sources(workspace: Path) -> Sources:
    """Resolve sources only when a requested background is not complete."""

    open_sources, licensed_specs = load_catalogue()
    licensed = resolve_licensed_inputs(workspace, licensed_specs)
    pb_spec = next(item for item in open_sources if item.source_id == "aesa_pb_lcia")
    pb_root = source_directory(workspace, pb_spec)
    if not owner_complete(pb_root):
        if not pb_root.exists() and not pb_root.is_symlink():
            raise FileNotFoundError(
                f"The required PB-LCIA files are missing from {pb_root}. Run "
                "download_data() in this workspace, then run prepare_backgrounds() again."
            )
        raise FileNotFoundError(
            f"The required PB-LCIA files are not complete in {pb_root}. Leave this "
            "workspace unchanged. Create a new workspace and run download_data() there."
        )
    dependency_authorities = installed_preparation_dependency_authorities()
    premise_authority = next(
        item for item in dependency_authorities if item["distribution"] == "premise"
    )
    material_recipe = load_material_recipe(workspace)
    provider = prepared_provider_authority()
    identity_payload: dict[str, object] = {
        "schema_version": "pytoce_background_sources_schema_v1",
        "ecoinvent_archive_sha256": licensed.archive_sha256,
        "ecoinvent_manifest_sha256": file_sha256(licensed.archive_manifest),
        "ef_workbook_sha256": licensed.ef_workbook_sha256,
        "pb_archive_sha256": pb_spec.sha256,
        "pb_manifest_sha256": file_sha256(pb_root / "manifest.json"),
        "material_recipe_identity": material_recipe.identity,
        "provider_recipe_identity": provider.identity,
        "ecoinvent_version": "3.12",
        "ecoinvent_import_method": ECOINVENT_IMPORT_METHOD,
        "system_model": "cutoff",
        "premise_version": "2.4.6",
        "bw2data_version": "4.7",
        "bw2io_version": "0.9.17",
    }
    identity = "pytoce_background_sources_schema_v1::" + payload_sha256(identity_payload)
    method = preparation_method()
    provider = replace(
        provider,
        interface_database=provider_interface_database_name(
            source_identity=identity,
            preparation_method_identity=str(method["identity"]),
            dependency_authorities=dependency_authorities,
        ),
    )
    return Sources(
        ecoinvent_archive=licensed.archive,
        ef_workbook=licensed.ef_workbook,
        pb_archive=pb_root / pb_spec.filename,
        source_identity=identity,
        source_identity_payload=identity_payload,
        dependency_authorities=dependency_authorities,
        premise_authority=premise_authority,
        premise_authority_identity=str(premise_authority["identity"]),
        preparation_method=method,
        material_recipe=material_recipe,
        provider=provider,
    )
