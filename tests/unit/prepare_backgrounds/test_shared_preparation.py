"""Shared preparation through compact source files and real Brightway persistence."""

import json
import tarfile
from dataclasses import replace
from importlib import metadata
from io import BytesIO
from pathlib import Path
from threading import Event, Thread

import py7zr
import pytest
import yaml
from infiltration_fixtures import infiltration_source_data, resolved_infiltration_provider
from lxml.etree import XMLSyntaxError
from material_carbon_fixtures import material_carbon_sources, prepared_material_carbon
from material_recipe_fixtures import physical_material_recipe
from openpyxl import Workbook
from pa_receiving_fixtures import pa_receiving_reference
from shared_chemical_fixtures import shared_chemical_inputs

from pytoce import prepare_backgrounds, set_workspace
from pytoce.download_data.sources.catalogue import load_catalogue, source_directory
from pytoce.prepare_backgrounds.preparation import project_biosphere
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    brightway_names,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.premise import BACKGROUND_BY_ID
from pytoce.prepare_backgrounds.preparation.prepared_state import (
    shared_preparation,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.source_authority import (
    Sources,
    resolve_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
)
from pytoce.shared.runtime.filesystem import atomic_write


def _write_ecospold_archive(path: Path, *, invalid_biosphere: bool) -> None:
    """Archive invented valid inventories, optionally with a malformed biosphere."""

    source = Path(__file__).resolve().parents[2] / "fixtures" / "ecoinvent_import"
    with py7zr.SevenZipFile(path, mode="w") as handle:
        for directory in ("datasets", "MasterData"):
            for member in sorted((source / directory).iterdir()):
                destination = member.relative_to(source).as_posix()
                if invalid_biosphere and member.name == "ElementaryExchanges.xml":
                    handle.writestr("<invalid", destination)
                else:
                    handle.write(member, destination)


def _write_lcia_files(workbook_path: Path, archive_path: Path) -> None:
    """Write every selected EF and PB category against one compact biosphere flow."""

    specs, catalogue = project_biosphere.load_lcia_catalogue()
    workbook = Workbook()
    indicators = workbook.active
    indicators.title = "Indicators"
    indicators.append(("method", "category", "indicator", "unit"))
    factors = workbook.create_sheet("CFs")
    factors.append(
        ("method", "category", "indicator", "name", "compartment", "subcompartment", "amount")
    )
    for spec in specs:
        if spec.method_id != "ef31":
            continue
        indicators.append((*spec.impact_category_tuple, spec.unit))
        factors.append((*spec.impact_category_tuple, "fixture emission", "air", None, 1.0))
    workbook.save(workbook_path)
    workbook.close()

    members: dict[str, list[dict[str, object]]] = {
        str(name): [] for name in catalogue["excluded_pb_source_members"]
    }
    for spec in specs:
        if spec.method_id == "pb_lcia":
            members[spec.source_member] = [
                {
                    "name": "fixture emission",
                    "categories": "air",
                    "amount": 1.0,
                }
            ]
    with tarfile.open(archive_path, mode="w:gz") as handle:
        for filename, rows in sorted(members.items()):
            content = yaml.safe_dump(rows).encode("utf-8")
            member = tarfile.TarInfo(filename)
            member.size = len(content)
            handle.addfile(member, BytesIO(content))


def _sources(workspace: Path, *, invalid_biosphere: bool = False) -> Sources:
    """Return source records backed by compact files in the temporary workspace."""

    source_root = workspace / "source_files"
    source_root.mkdir(parents=True)
    archive = source_root / "ecoinvent.7z"
    workbook = source_root / "ef.xlsx"
    pb_archive = source_root / "pb.tar.gz"
    _write_ecospold_archive(archive, invalid_biosphere=invalid_biosphere)
    _write_lcia_files(workbook, pb_archive)
    return Sources(
        ecoinvent_archive=archive,
        ef_workbook=workbook,
        pb_archive=pb_archive,
        source_identity="sources::compact_shared_preparation",
        source_identity_payload={"schema_version": "compact_sources_schema_v1"},
        dependency_authorities=(),
        premise_authority={"distribution": "premise", "version": "2.4.6"},
        premise_authority_identity="premise::2.4.6",
        material_recipe=physical_material_recipe(),
        provider=prepared_provider_authority(),
        preparation_method={"identity": "preparation::compact_shared"},
    )


def _write_public_source_manifests(workspace: Path) -> None:
    """Mark compact source directories complete for the public reuse-path test."""

    open_sources, licensed_sources = load_catalogue()
    pb_spec = next(spec for spec in open_sources if spec.source_id == "aesa_pb_lcia")
    for spec in (licensed_sources[0], pb_spec):
        root = source_directory(workspace, spec)
        root.mkdir(parents=True, exist_ok=True)
        atomic_write(root / "manifest.json", b'{"complete":true}\n')


def _write_shared_reuse_fixture(workspace: Path, sources: Sources) -> None:
    """Write the shared manifest and tables consumed by member-level reuse."""

    provider = replace(
        resolved_infiltration_provider(workspace.parent / "infiltration_fixture"),
        interface_database=sources.provider.interface_database,
    )
    (workspace / "data_processed" / "prepared_backgrounds").mkdir(parents=True)
    shared_preparation.write_shared_data(
        workspace,
        sources,
        brightway_names(workspace, source_identity=sources.source_identity),
        pa_receiving_reference=pa_receiving_reference(
            brightway_names(workspace, source_identity=sources.source_identity).source_database
        ),
        provider=provider,
        material_carbon_rows=prepared_material_carbon(workspace.parent / "carbon_fixture"),
        impact_rows=(
            {
                "selection_order": 1,
                "method_id": "ef31",
                "factor_identity": "factor::compact_public_reuse",
            },
        ),
        database_rows=(
            {
                "database_name": "compact_biosphere",
                "database_role": "ecoinvent_biosphere",
            },
        ),
        matrix_rows=({"database_name": "compact_biosphere"},),
        biosphere_rows=(
            {
                "authority_role": "ecoinvent_biosphere",
                "ecoinvent_biosphere_database": "compact_biosphere",
            },
        ),
        import_report={"source_activity_count": 1},
    )


def _include_shared_scientific_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add invented scientific records at the real importer's external boundary."""

    import bw2data as bd
    import bw2io.importers as importers

    class InventoryImporter(importers.SingleOutputEcospold2Importer):
        """Parse XML, apply real strategies and persist one augmented data buffer."""

        def apply_strategies(self, *, verbose: bool) -> None:
            """Retain parsed data and add the shared owner's scientific fixtures."""

            super().apply_strategies(verbose=verbose)
            catalogue = prepared_provider_authority().catalogue
            services = catalogue["services"]
            bd.geomapping.add({"GLO", *(str(service["location"]) for service in services)})
            biosphere = bd.preferences["biosphere_database"]
            carbon_data, carbon_flows = material_carbon_sources(self.db_name, biosphere)
            chemical_data, chemical_flows = shared_chemical_inputs(
                catalogue, self.db_name, biosphere
            )
            bd.Database(biosphere).write(
                {**bd.Database(biosphere).load(), **carbon_flows, **chemical_flows},
                searchable=False,
            )
            data = {
                **carbon_data,
                **infiltration_source_data(self.db_name),
                **chemical_data,
            }
            self.data.extend(
                {**node, "database": database, "code": code}
                for (database, code), node in data.items()
            )

    monkeypatch.setattr(importers, "SingleOutputEcospold2Importer", InventoryImporter)


def test_shared_preparation_writes_methods_providers_and_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare the complete shared product from compact external source boundaries."""

    workspace = tmp_path / "workspace"
    sources = _sources(workspace)
    _include_shared_scientific_fixtures(monkeypatch)

    prepared = shared_preparation.ensure_shared_authority(workspace, sources)

    root = shared_preparation.shared_root(workspace)
    assert prepared.source_identity == sources.source_identity
    assert (root / "manifest.json").is_file()
    assert len(prepared.impact_categories) == 26
    with selected_brightway_project(prepared.names) as bd:
        native, _ = shared_chemical_inputs(
            sources.provider.catalogue,
            prepared.names.source_database,
            prepared.names.biosphere_database,
        )
        # Shared preparation preserves the complete native source. Study
        # manufacture and receiving are owned by later dated frontiers.
        for code in (
            "tea_RER",
            "tea_RoW",
            "mibk_rer",
            "mibk_row",
            "steam_rer",
            "steam_row",
            "steam_rer_market",
            "steam_row_market",
        ):
            key = prepared.names.source_database, code
            node = bd.get_node(database=key[0], code=key[1])
            for field in ("name", "reference product", "unit", "location"):
                assert node[field] == native[key][field]
            assert [
                {field: value for field, value in dict(row).items() if field != "output"}
                for row in node.exchanges()
            ] == [
                {field: value for field, value in row.items() if field != "output"}
                for row in native[key]["exchanges"]
            ]
        assert all(
            not node.get("provider_service_id")
            for node in bd.Database(prepared.names.source_database)
        )
    dumping = [
        row
        for row in prepared.provider_routes
        if row["provider_demand_type"] == "mismanaged_open_dump_treatment"
    ]
    assert {row["infiltration_share"] for row in dumping} == {
        0.0588235294117647,
        0.117647058823529,
        0.176470588235294,
        0.235294117647059,
        0.411764705882353,
    }
    import pandas as pd

    saved_sources = pd.read_parquet(root / "provider_source_authorities.parquet")
    source = saved_sources.loc[
        saved_sources["source_id"] == "ecoinvent_3_12_row_open_dump_portfolio"
    ].iloc[0]
    assert json.loads(source["native_source_key_json"]) == [
        prepared.names.source_database,
        "pe_waste_market",
    ]
    assert json.loads(source["derivation_json"])["conditional_denominator"] == 17.0
    sources.ecoinvent_archive.unlink()
    reused = shared_preparation.ensure_shared_authority(workspace, sources)
    assert reused.identity == prepared.identity
    pd.testing.assert_frame_equal(
        pd.DataFrame(reused.provider_routes), pd.DataFrame(prepared.provider_routes)
    )
    assert not any(
        path.name.startswith(".shared_background_work.") for path in root.parent.iterdir()
    )


def test_shared_preparation_preserves_import_failure_and_removes_working_files(
    tmp_path: Path,
) -> None:
    """Report a bw2io failure and remove only the hidden temporary extraction."""

    workspace = tmp_path / "workspace"
    sources = _sources(workspace, invalid_biosphere=True)

    with pytest.raises(XMLSyntaxError, match="invalid"):
        shared_preparation.ensure_shared_authority(workspace, sources)

    prepared = workspace / "data_processed" / "prepared_backgrounds"
    assert not any(path.name.startswith(".shared_background_work.") for path in prepared.iterdir())


def test_public_preparation_reuses_a_member_completed_after_initial_selection(
    tmp_path: Path,
    stored_material_recipe: tuple[Path, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse a background completed concurrently after the public pending check."""

    workspace = set_workspace(tmp_path).workspace
    _write_public_source_manifests(workspace)
    sources = resolve_sources(workspace)
    _write_shared_reuse_fixture(workspace, sources)

    scenario = BACKGROUND_BY_ID["remind_ssp2_npi"]
    iam_root = workspace / "data_raw" / "premise" / scenario.background_scenario
    iam_root.mkdir(parents=True)
    atomic_write(iam_root / "manifest.json", b'{"complete":true}\n')

    member_root = (
        workspace / "data_processed" / "prepared_backgrounds" / scenario.background_scenario
    )
    metadata_started = Event()
    member_completed = Event()
    thread_errors: list[BaseException] = []
    distribution = metadata.distribution

    def wait_for_concurrent_member(distribution_name: str) -> metadata.Distribution:
        """Pause the exact dependency-metadata boundary until the member is complete."""

        if not metadata_started.is_set():
            metadata_started.set()
            assert member_completed.wait(timeout=5)
        return distribution(distribution_name)

    def complete_member() -> None:
        """Represent another process publishing its manifest-last member result."""

        try:
            assert metadata_started.wait(timeout=5)
            member_root.mkdir(parents=True)
            atomic_write(member_root / "manifest.json", b'{"complete":true}\n')
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            member_completed.set()

    monkeypatch.setattr(metadata, "distribution", wait_for_concurrent_member)
    writer = Thread(target=complete_member)
    writer.start()
    try:
        report = prepare_backgrounds(background_scenario=scenario.background_scenario)
    finally:
        writer.join(timeout=5)

    assert not writer.is_alive()
    assert thread_errors == []
    assert report.status == "reused"
    assert report.results[0].manifest_path == member_root / "manifest.json"
