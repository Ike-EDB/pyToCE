from pathlib import Path

import pytest

from pytoce.download_data.sources.catalogue import (
    load_catalogue,
    owner_manifest,
    source_directory,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.source_authority import (
    premise_transformation_method,
    resolve_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    provider_interface_database_name,
)
from pytoce.shared.runtime.filesystem import canonical_json_bytes, payload_sha256


def test_missing_and_incomplete_pb_lcia_sources_give_different_actions(
    tmp_path: Path,
) -> None:
    """Direct missing PB-LCIA inputs to download_data and incomplete inputs to a new workspace."""

    ecoinvent = tmp_path / "data_raw" / "ecoinvent"
    ecoinvent.mkdir(parents=True)
    (ecoinvent / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.*Run download_data.*this workspace"):
        resolve_sources(tmp_path)

    pb_lcia = tmp_path / "data_raw" / "pb_lcia"
    pb_lcia.mkdir()
    with pytest.raises(FileNotFoundError, match="not complete.*new workspace"):
        resolve_sources(tmp_path)


def test_premise_transformation_identity_is_versioned_and_path_neutral() -> None:
    """Bind transformation identity to its declared inputs."""

    method = premise_transformation_method()

    assert method["schema_version"] == "pytoce_premise_transformation_method_schema_v1"
    assert method["method"] == "pytoce_premise_transformation_method_v1"
    assert str(method["identity"]).startswith("pytoce_premise_transformation_method_v1::")
    assert set(method) == {"schema_version", "method", "identity"}


def test_completed_source_owner_manifests_resolve_without_reopening_archives(
    tmp_path: Path,
    stored_material_recipe: tuple[Path, dict[str, object]],
) -> None:
    """Use manifest-last parent custody while retaining every exact source identity."""

    open_sources, licensed_sources = load_catalogue()
    pb_spec = next(spec for spec in open_sources if spec.source_id == "aesa_pb_lcia")
    for specs in ((pb_spec,), licensed_sources):
        root = source_directory(tmp_path, specs[0])
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_bytes(canonical_json_bytes(owner_manifest(specs)))

    sources = resolve_sources(tmp_path)

    _, material_payload = stored_material_recipe
    assert (
        sources.source_identity_payload["material_recipe_identity"] == material_payload["identity"]
    )
    assert sources.source_identity_payload["provider_recipe_identity"] == sources.provider.identity
    assert canonical_json_bytes(sources.material_recipe.as_dict()) == canonical_json_bytes(
        material_payload
    )
    assert sources.ecoinvent_archive.name == licensed_sources[0].filename
    assert sources.pb_archive.name == pb_spec.filename
    assert sources.source_identity_payload["pb_archive_sha256"] == pb_spec.sha256
    assert sources.source_identity_payload["ecoinvent_import_method"] == (
        "pytoce_ecoinvent_native_import_v1"
    )
    predecessor_payload = dict(sources.source_identity_payload)
    predecessor_payload["ecoinvent_import_method"] = (
        "pytoce_ecoinvent_manufacturing_supplements_v11"
    )
    assert sources.source_identity != (
        "pytoce_background_sources_schema_v1::" + payload_sha256(predecessor_payload)
    )
    assert tuple(
        (row["distribution"], row["version"]) for row in sources.dependency_authorities
    ) == (("bw2data", "4.7"), ("bw2io", "0.9.17"), ("premise", "2.4.6"))
    assert sources.premise_authority["identity"] == sources.premise_authority_identity
    assert sources.source_identity.startswith("pytoce_background_sources_schema_v1::")
    namespace_inputs = {
        "source_identity": sources.source_identity,
        "preparation_method_identity": str(sources.preparation_method["identity"]),
        "dependency_authorities": sources.dependency_authorities,
    }
    assert sources.provider.interface_database == provider_interface_database_name(
        **namespace_inputs
    )
    # Input authorities select predecessor inputs; generated products are written into their own
    # result directories.
    successors = (
        {**namespace_inputs, "source_identity": "sources::different_recipe_or_native_parent"},
        {**namespace_inputs, "preparation_method_identity": "preparation::different_method"},
        {
            **namespace_inputs,
            "dependency_authorities": ({"distribution": "premise", "identity": "different"},),
        },
    )
    names = {provider_interface_database_name(**inputs) for inputs in successors}
    assert len(names) == 3
    assert sources.provider.interface_database not in names
