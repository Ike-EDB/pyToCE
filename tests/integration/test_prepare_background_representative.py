"""Bounded background preparation and file-writing contracts."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from material_carbon_fixtures import prepared_material_carbon
from material_recipe_fixtures import physical_material_recipe
from openpyxl import Workbook
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources

from pytoce import prepare_backgrounds, set_workspace
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.premise import (
    BACKGROUND_BY_ID,
    BackgroundScenario,
    PremiseIAMInput,
    database_name,
    provider_technology_database_name,
    run_premise_member,
    select_background_scenarios,
)
from pytoce.prepare_backgrounds.preparation.prepared_state import (
    batch_preparation as batch_owner,
)
from pytoce.prepare_backgrounds.preparation.prepared_state import (
    member_preparation as member_owner,
)
from pytoce.prepare_backgrounds.preparation.prepared_state import (
    shared_preparation as shared_owner,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.source_authority import (
    Sources,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.interface_database import (
    provider_timeline_databases,
    provider_timeline_identity,
    write_provider_interface,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    sources_from_payload,
)
from pytoce.shared.runtime.filesystem import atomic_write
from tests.integration.pa_background_fixtures import (
    bounded_pa_premise,
    pa_background_provider,
    prepare_pa_background_inputs,
)

__all__ = ("bounded_pa_premise",)


def _sources(tmp_path: Path) -> Sources:
    return Sources(
        ecoinvent_archive=tmp_path / "ecoinvent.7z",
        ef_workbook=tmp_path / "lcia.xlsx",
        pb_archive=tmp_path / "pb.tar.gz",
        source_identity="source::" + "d" * 64,
        source_identity_payload={"schema_version": "source_v1"},
        dependency_authorities=(
            {
                "distribution": "premise",
                "version": "2.4.6",
                "identity": "premise::synthetic",
            },
        ),
        premise_authority={
            "distribution": "premise",
            "version": "2.4.6",
            "identity": "premise::synthetic",
        },
        premise_authority_identity="premise::synthetic",
        material_recipe=physical_material_recipe(),
        provider=prepared_provider_authority(),
        preparation_method={
            "schema_version": "software_v1",
            "identity": "software::synthetic",
        },
    )


def _iam_input(
    tmp_path: Path,
    scenario: BackgroundScenario,
) -> PremiseIAMInput:
    directory = tmp_path / "data_raw" / "premise" / scenario.background_scenario
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / scenario.iam_filename
    payload.write_bytes(b"synthetic")
    manifest = directory / "manifest.json"
    atomic_write(
        manifest,
        b'{"complete":true,"identity":"iam::synthetic"}\n',
    )
    return PremiseIAMInput(
        identity=f"iam::{scenario.background_scenario}",
        payload=payload,
        manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
        payload_sha256=sha256(payload.read_bytes()).hexdigest(),
        premise_authority_identity="premise::synthetic",
    )


def _shared(tmp_path: Path) -> shared_owner.SharedAuthority:
    impact_categories: tuple[dict[str, object], ...] = tuple(
        {
            "selection_order": order,
            "method_id": "ef31" if order <= 16 else "pb_lcia",
            "method_version": "3.1" if order <= 16 else "0.9.2",
            "impact_category_id": f"category_{order}",
            "impact_category_tuple_json": json.dumps(["method", f"category_{order}"]),
            "unit": "point",
            "factor_identity": f"factor_{order}",
        }
        for order in range(1, 27)
    )
    return shared_owner.SharedAuthority(
        pa_receiving_reference=pa_receiving_reference("ecoinvent"),
        names=BrightwayNames(
            store=tmp_path / "data_processed" / "brightway",
            project="pytoce_test",
            biosphere_database="biosphere",
            source_database="ecoinvent",
        ),
        identity="shared::" + "1" * 64,
        source_identity="source::" + "d" * 64,
        preparation_method_identity="software::synthetic",
        lcia_identity="lcia::" + "2" * 64,
        provider_identity="provider::" + "4" * 64,
        provider_interface_database="pytoce__provider_interface",
        provider_timeline_identity="provider_timeline::" + "5" * 64,
        provider_timeline_database_by_year={
            2020: "pytoce__provider_interface",
            2025: "pytoce__provider_timeline__2025",
        },
        provider_routes=(
            {
                "provider_service_id": "synthetic_service",
                "provider_service_code": "provider_service__synthetic_service",
            },
        ),
        impact_categories=impact_categories,
        database_rows=(
            {
                "database_name": "biosphere",
                "database_role": "ecoinvent_biosphere",
            },
            {
                "database_name": "ecoinvent",
                "database_role": "ecoinvent_3_12_cutoff_base",
            },
        ),
        matrix_rows=(
            {
                "database_name": "ecoinvent",
            },
        ),
        biosphere_rows=(
            {
                "authority_role": "ecoinvent_biosphere",
                "ecoinvent_biosphere_database": "biosphere",
            },
        ),
    )


@pytest.fixture
def bounded_premise(
    isolated_premise: ModuleType,
    inline_premise_process: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supply tiny databases and valid reports at premise's public write boundary."""

    class BoundedNewDatabase:
        """Implement the constructor, update and writer used by premise 2.4.6."""

        def __init__(self, **kwargs: object) -> None:
            """Accept the configuration verified by the dedicated integration test."""

        def update(self, sectors: list[str]) -> None:
            """The bounded inventory is already transformed."""

        def write_db_to_brightway(self, name: list[str] | None = None) -> None:
            """Write the requested real databases; the package worker owns provenance."""
            import bw2data as bd

            assert name is not None
            for database in name:
                bd.Database(database).write(
                    {
                        (database, "activity"): {
                            "name": "bounded provider",
                            "reference product": "bounded product",
                            "location": "GLO",
                            "unit": "kilogram",
                            "type": "process",
                            "exchanges": [
                                {
                                    "input": (database, "activity"),
                                    "amount": 1.0,
                                    "type": "production",
                                }
                            ],
                        }
                    }
                )
            for report_kind in ("scenario_report", "change reports"):
                directory = Path.cwd() / "export" / report_kind
                directory.mkdir(parents=True)
                workbook = Workbook()
                workbook.active.append(["bounded premise report"])
                workbook.save(directory / "report.xlsx")
                workbook.close()

    monkeypatch.setattr(isolated_premise, "NewDatabase", BoundedNewDatabase)


def test_public_reuse_uses_only_the_direct_completion_manifest(
    tmp_path: Path,
) -> None:
    workspace = set_workspace(tmp_path).workspace
    result = workspace / "data_processed" / "prepared_backgrounds" / "remind_ssp2_npi"
    atomic_write(
        result / "manifest.json",
        b"ordinary reuse does not reopen this file\n",
    )

    report = prepare_backgrounds(
        background_scenario="remind_ssp2_npi",
    )

    assert report.status == "reused"
    assert report.results[0].manifest_path == result / "manifest.json"
    assert report.results[0].result_directory == result


def test_remind_eu_preparation_is_reuse_only(tmp_path: Path) -> None:
    workspace = set_workspace(tmp_path).workspace
    result = workspace / "data_processed" / "prepared_backgrounds" / "remind_eu_ssp2_npi"
    atomic_write(
        result / "manifest.json",
        b"ordinary legacy reuse does not reopen this file\n",
    )

    reused = prepare_backgrounds(background_scenario="remind_eu_ssp2_npi")
    assert reused.status == "reused"
    assert reused.results[0].manifest_path == result / "manifest.json"

    (result / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="Choose a completed historical scenario"):
        prepare_backgrounds(background_scenario="remind_eu_ssp2_npi")


def test_shared_result_owns_the_source_neutral_provider_authority(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    shared = _shared(tmp_path)
    root = tmp_path / "data_processed" / "prepared_backgrounds"
    root.mkdir(parents=True)
    provider = PreparedProviderAuthority(
        identity=shared.provider_identity,
        interface_database=shared.provider_interface_database,
        catalogue={},
        routes=pd.DataFrame(
            [
                {**shared.provider_routes[0], "lca_fate": None},
                {
                    "provider_service_id": "open_dump_service",
                    "provider_service_code": "provider_service__open_dump_service",
                    "lca_fate": "open_dump",
                },
            ]
        ),
    )

    written = shared_owner.write_shared_data(
        tmp_path,
        sources,
        shared.names,
        pa_receiving_reference=pa_receiving_reference("ecoinvent"),
        provider=provider,
        material_carbon_rows=prepared_material_carbon(tmp_path / "carbon_fixture"),
        impact_rows=shared.impact_categories,
        database_rows=shared.database_rows,
        matrix_rows=shared.matrix_rows,
        biosphere_rows=shared.biosphere_rows,
        import_report={"source_activity_count": 1},
    )

    provider_manifest = json.loads(
        (root / "shared" / "provider_service_manifest.json").read_text(encoding="utf-8")
    )
    assert provider_manifest["identity"] == shared.provider_identity
    assert provider_manifest["interface_database"] == shared.provider_interface_database
    route_rows = pd.read_parquet(root / "shared" / "provider_service_routes.parquet").to_dict(
        orient="records"
    )
    assert route_rows == [
        {**shared.provider_routes[0], "lca_fate": None},
        {
            "provider_service_id": "open_dump_service",
            "provider_service_code": "provider_service__open_dump_service",
            "lca_fate": "open_dump",
        },
    ]
    assert list(written.provider_routes) == route_rows
    obsolete = root / "shared" / "obsolete_provider_authority"
    obsolete.write_text("superseded", encoding="utf-8")
    reloaded = shared_owner.load_shared(tmp_path, names=shared.names)
    assert reloaded is not None
    assert reloaded.provider_identity == shared.provider_identity
    assert obsolete.read_text(encoding="utf-8") == "superseded"
    with pytest.raises(RuntimeError, match="different source files.*new workspace"):
        shared_owner.ensure_shared_authority(tmp_path, sources)


def test_incomplete_shared_directory_is_rejected_before_brightway_changes(
    tmp_path: Path,
) -> None:
    """Stop before importing ecoinvent when the shared result is incomplete."""

    incomplete = tmp_path / "data_processed" / "prepared_backgrounds" / "shared"
    incomplete.mkdir(parents=True)
    note = incomplete / "user_note.txt"
    note.write_text("preserve", encoding="utf-8")

    with pytest.raises(RuntimeError, match="has no completion manifest"):
        shared_owner.ensure_shared_authority(tmp_path, _sources(tmp_path))

    assert note.read_text(encoding="utf-8") == "preserve"


@pytest.mark.scientific_integration
@pytest.mark.usefixtures("bounded_premise")
def test_premise_anchor_custody_drives_the_temporal_background_crosswalk(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    shared = _shared(tmp_path)
    scenario = BACKGROUND_BY_ID["remind_ssp2_npi"]
    iam_input = _iam_input(tmp_path, scenario)
    target = member_owner.target(
        tmp_path,
        sources,
        shared,
        iam_input,
        scenario,
        years=(2030, 2040),
    )
    premise_years = (2020, 2025, *target.years)
    output_names = (
        provider_technology_database_name(scenario, 2020),
        provider_technology_database_name(scenario, 2025),
        *(
            database_name(
                scenario,
                year,
                background_identity=target.background_identity,
            )
            for year in target.years
        ),
    )
    premise_report = run_premise_member(
        brightway_store=shared.names.store,
        project_name=shared.names.project,
        cache_root=tmp_path / "premise_cache",
        report_root=tmp_path / "premise_reports",
        iam_input=iam_input,
        source_database=shared.names.source_database,
        biosphere_database=shared.names.biosphere_database,
        scenario=scenario,
        years=premise_years,
        output_names=output_names,
        premise_key=None,
    )
    anchor_rows = member_owner.anchor_database_rows(shared, target, premise_report)
    tables = member_owner.item_tables(
        shared,
        target,
        anchor_rows,
        {
            "provider_technology_crosswalk.parquet": (),
            "provider_semantic_ledger.parquet": (),
            "provider_database_rows": (),
            "provider_matrix_rows": (),
        },
    )
    crosswalk = pd.DataFrame(tables["provider_crosswalk.parquet"])

    assert tuple(crosswalk["stage_year"]) == tuple(range(2019, 2041))
    assert (
        crosswalk.loc[
            crosswalk["stage_year"] <= 2025,
            ["lower_anchor_year", "upper_anchor_year", "lower_weight", "upper_weight"],
        ]
        == [2025, 2025, 1.0, 0.0]
    ).all(axis=None)
    assert crosswalk.loc[
        crosswalk["stage_year"] == 2035,
        [
            "lower_anchor_year",
            "upper_anchor_year",
            "lower_weight",
            "upper_weight",
        ],
    ].iloc[0].tolist() == [2030, 2040, 0.5, 0.5]
    assert [(row["anchor_year"], row["database_role"]) for row in anchor_rows] == [
        (2020, "early_premise_only_provider_anchor"),
        (2025, "early_premise_only_provider_anchor"),
        (2030, "premise_anchor"),
        (2040, "premise_anchor"),
    ]


def test_member_result_binds_the_provider_timeline_authority(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    shared = _shared(tmp_path)
    scenario = BACKGROUND_BY_ID["remind_ssp2_npi"]
    prepared_target = member_owner.target(
        tmp_path,
        sources,
        shared,
        _iam_input(tmp_path, scenario),
        scenario,
        years=(2030,),
    )
    anchor_rows: tuple[dict[str, object], ...] = (
        {
            "database_name": "premise_2030",
            "database_role": "premise_anchor",
            "anchor_year": 2030,
        },
    )
    tables = member_owner.item_tables(
        shared,
        prepared_target,
        anchor_rows,
        {
            "provider_technology_crosswalk.parquet": (),
            "provider_semantic_ledger.parquet": (),
            "provider_database_rows": (),
            "provider_matrix_rows": (),
        },
    )
    staging = tmp_path / "member_staging"
    staging.mkdir()
    member_owner.write_prepared_background(
        prepared_target,
        shared,
        staging,
        tables,
        pa_receiving_sources_by_year={
            year: pa_receiving_sources(
                year,
                background_scenario=scenario.background_scenario,
                source_database=shared.names.source_database,
            )
            for year in (2020, 2025, *prepared_target.years)
        },
        premise_report={"database_rows": []},
        report_artifacts=(),
    )

    background_row = pd.read_parquet(staging / "background_scenarios.parquet").to_dict(
        orient="records"
    )[0]
    report = json.loads((staging / "preparation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    expected_json_map = {
        str(year): database for year, database in shared.provider_timeline_database_by_year.items()
    }

    assert prepared_target.identity_payload["provider_timeline_identity"] == (
        shared.provider_timeline_identity
    )
    assert prepared_target.identity_payload["provider_timeline_database_by_year"] == (
        shared.provider_timeline_database_by_year
    )
    assert background_row["provider_timeline_identity"] == shared.provider_timeline_identity
    assert json.loads(background_row["provider_timeline_database_by_year_json"]) == (
        expected_json_map
    )
    assert report["provider_timeline_identity"] == shared.provider_timeline_identity
    assert report["provider_timeline_database_by_year"] == expected_json_map
    assert manifest["provider_timeline_identity"] == shared.provider_timeline_identity
    assert manifest["provider_timeline_database_by_year"] == expected_json_map
    assert manifest["identity_payload"]["provider_timeline_identity"] == (
        shared.provider_timeline_identity
    )
    assert manifest["identity_payload"]["provider_timeline_database_by_year"] == (expected_json_map)


def test_member_refuses_an_existing_database_without_deleting_it(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    shared = _shared(tmp_path)
    scenario = BACKGROUND_BY_ID["remind_ssp2_npi"]
    iam_input = _iam_input(tmp_path, scenario)
    target = member_owner.target(
        tmp_path,
        sources,
        shared,
        iam_input,
        scenario,
        years=(2030,),
    )
    owned = database_name(
        scenario,
        2030,
        background_identity=target.background_identity,
    )
    with selected_brightway_project(shared.names) as bd:
        for name in (owned, "unrelated_database"):
            bd.Database(name).write(
                {
                    (name, "activity"): {
                        "name": name,
                        "reference product": "product",
                        "location": "GLO",
                        "unit": "kilogram",
                        "type": "process",
                        "exchanges": [
                            {
                                "input": (name, "activity"),
                                "amount": 1.0,
                                "type": "production",
                            }
                        ],
                    }
                }
            )
    with pytest.raises(RuntimeError, match="already contains.*Create a new workspace"):
        provider = prepared_provider_authority()
        member_owner.prepare_member(
            tmp_path,
            sources,
            replace(shared, provider_identity=provider.identity),
            provider,
            iam_input,
            scenario,
            years=(2030,),
            premise_key=None,
            progress=lambda _scenario, _phase: None,
            provider_resolvers=_NativeResolverCache(
                shared.names.source_database,
                shared.names.biosphere_database,
            ),
        )

    with selected_brightway_project(shared.names) as bd:
        assert owned in bd.databases
        assert "unrelated_database" in bd.databases


def test_batch_failure_keeps_completed_sibling_reusable(
    tmp_path: Path,
    isolated_premise: ModuleType,
    inline_premise_process: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate a simulated dependency failure and preserve the completed sibling."""

    class FailingNewDatabase:
        """Fail at the dependency constructor reached only for the unfinished member."""

        def __init__(self, **kwargs: object) -> None:
            """Keep the completed member out of premise execution."""
            assert kwargs["scenarios"][0]["pathway"] == "SSP2-NPi"
            raise RuntimeError("second member failed")

    monkeypatch.setattr(isolated_premise, "NewDatabase", FailingNewDatabase)
    sources = _sources(tmp_path)
    provider = prepared_provider_authority()
    shared = replace(
        _shared(tmp_path),
        provider_identity=provider.identity,
        provider_interface_database=provider.interface_database,
        provider_timeline_identity=provider_timeline_identity(
            provider.identity, interface_database=provider.interface_database
        ),
        provider_timeline_database_by_year=provider_timeline_databases(provider.interface_database),
    )
    scenarios = select_background_scenarios(("remind_ssp2_ndc", "remind_ssp2_npi"))
    iam_inputs = {
        scenario.background_scenario: _iam_input(tmp_path, scenario) for scenario in scenarios
    }
    completed_directory = (
        tmp_path / "data_processed" / "prepared_backgrounds" / scenarios[0].background_scenario
    )
    atomic_write(
        completed_directory / "manifest.json",
        b'{"complete":true}\n',
    )
    iam_inputs[scenarios[0].background_scenario].payload.unlink()
    with pytest.raises(RuntimeError, match="second member failed"):
        batch_owner.prepare_members(
            tmp_path,
            sources,
            shared,
            scenarios,
            iam_inputs,
            years=(2030, 2040),
            premise_key=None,
            progress=lambda _scenario, _phase: None,
        )
    completed = member_owner.selected_prepared_background(
        tmp_path,
        "remind_ssp2_ndc",
    )
    assert completed is not None
    assert completed.status == "reused"


@pytest.mark.scientific_integration
@pytest.mark.usefixtures("bounded_pa_premise")
def test_member_writes_bounded_provider_frontiers(tmp_path: Path) -> None:
    """Run the member owner through premise, provider, and manifest writing."""

    receiving = prepare_pa_background_inputs(tmp_path)
    sources = _sources(tmp_path)
    provider = pa_background_provider(sources.provider)
    sources = replace(sources, provider=provider)
    shared = replace(
        _shared(tmp_path),
        names=receiving.names,
        pa_receiving_reference=receiving.reference,
        provider_identity=provider.identity,
        provider_interface_database=provider.interface_database,
        provider_timeline_identity=provider_timeline_identity(
            provider.identity, interface_database=provider.interface_database
        ),
        provider_timeline_database_by_year=provider_timeline_databases(provider.interface_database),
        provider_routes=(),
        database_rows=(
            {"database_name": "biosphere", "database_role": "ecoinvent_biosphere"},
            {"database_name": "original", "database_role": "ecoinvent_3_12_cutoff_base"},
        ),
        matrix_rows=({"database_name": "original"},),
    )
    scenario = BACKGROUND_BY_ID["remind_ssp2_npi"]
    iam_input = _iam_input(tmp_path, scenario)
    with selected_brightway_project(shared.names) as bd:
        bd.geomapping.add({"GLO"})
        write_provider_interface(bd, shared.names.biosphere_database, provider)

    result = member_owner.prepare_member(
        tmp_path,
        sources,
        shared,
        provider,
        iam_input,
        scenario,
        years=(2030,),
        premise_key=None,
        progress=lambda _scenario, _phase: None,
        provider_resolvers=_NativeResolverCache(
            shared.names.source_database,
            shared.names.biosphere_database,
        ),
    )

    assert result.status == "computed"
    assert result.manifest_path.is_file()
    provider_rows = pd.read_parquet(
        result.result_directory / "provider_technology_crosswalk.parquet"
    )
    assert set(provider_rows["anchor_year"]) == {2020, 2025, 2030}
    databases = pd.read_parquet(result.result_directory / "database_authority.parquet")
    assert databases.loc[
        databases["database_role"].eq("ecoinvent_3_12_cutoff_base"), "database_name"
    ].tolist() == ["original"]
    admitted = {
        source.provider_anchor_year: source
        for source in map(
            sources_from_payload,
            json.loads((result.result_directory / "pa_receiving_sources.json").read_text()),
        )
    }
    assert set(admitted) == {2020, 2025, 2030}
    # Original HWI is -3 kg with 0.004 kg peroxide/kg. The future fixture doubles the production
    # basis and raises peroxide intensity by 50%. Ordinary receiving retains the original
    # general-background technology.
    for year, production, dose in ((2020, -3.0, 0.004), (2025, -3.0, 0.004), (2030, -6.0, 0.006)):
        source = admitted[year]
        assert source.hwi.production_amount == production
        peroxide = next(row for row in source.hwi.rows if row.endpoint.key[1] == "cr_peroxide")
        assert peroxide.amount == pytest.approx(dose)
        assert peroxide.endpoint.unit == "kilogram"
        assert peroxide.metadata["allocation"] == "invented chemical source"
        assert source.general_background_anchor_year == (2025 if year <= 2025 else year)
        assert source.general_background_database == (
            "original"
            if year <= 2025
            else provider_rows.set_index("anchor_year").loc[year, "ordinary_database_name"]
        )
    with selected_brightway_project(shared.names) as bd:
        for database in provider_rows["database_name"]:
            assert {node["provider_service_id"] for node in bd.Database(database)} == {
                "hydraulic_water_collection_row",
                "hydraulic_water_treatment_row",
                "recycling_weee_preparation",
                "recycling_pet_finishing",
                "recycling_pet_pellet_delivery",
            }
