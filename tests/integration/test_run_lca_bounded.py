import json
from contextlib import nullcontext
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from infiltration_fixtures import resolved_infiltration_provider
from material_carbon_fixtures import prepared_material_carbon
from material_recipe_fixtures import material_source_members
from mfa_workspace_fixtures import prepare_mfa_workspace

from pytoce import run_lca, set_workspace
from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.premise import (
    GENERAL_BACKGROUND_ANCHOR_YEARS,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    PreparedManifest,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PROVIDER_INTERFACE_DATABASE,
    provider_frontier_database_name,
    provider_timeline_database_name,
)
from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    create_provider_response_state,
)
from pytoce.run_lca.calculation.flow_coverage import (
    FLOW_OWNERSHIP_METHOD_IDENTITY,
)
from pytoce.run_lca.calculation.foreground import (
    H12_REGIONS,
    ForegroundProduct,
    build_foreground_database,
    compile_foreground_activations,
    optimization_activation_delta,
)
from pytoce.run_lca.calculation.lcia import (
    ImpactCategory,
    normalized_lcia_method,
)
from pytoce.run_lca.calculation.providers import (
    PreparedProviderProduct,
    ProviderRoutes,
    load_prepared_provider_product,
    load_provider_routes,
    prepared_brightway_names,
)
from pytoce.run_lca.calculation.results import write_lca_result
from pytoce.run_lca.calculation.timex import build_provider_inventory_bank
from pytoce.run_lca.run_lca import _background_dimensions
from pytoce.run_optimization.calculation.response_bank import (
    project_optimization_response_bank,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    LoadedOptimizationResponseBank,
)
from pytoce.run_optimization.calculation.response_kernel import (
    ResponseMethodSelection,
    _method_registry,
    _provider_rows,
    _resolved_response_methods,
    _service_responses,
    _timex_weights,
)
from pytoce.run_optimization.calculation.selected_reporting import (
    _effective_responses,
    _stable_group,
    _stage_response_adjustments,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256, file_sha256
from tests.integration.brightway_lca_fixtures import (
    _PROVIDER_SERVICE_CODE,
    _PROVIDER_SERVICE_ID,
    PROVIDER_TECHNICAL_YEARS,
    _process,
    write_prepared_inputs,
)


def _fixed_service_response(
    names: BrightwayNames,
    background_scenario: str,
    methods: tuple[ImpactCategory, ...],
    registry: pd.DataFrame,
    services: pd.DataFrame,
    providers: PreparedProviderProduct,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exercise fixed-service equivalence on this compact non-PA native graph.

    The complete actual-PA member boundary has its own real source fixture;
    these service-only tests use the same native solve and service reduction.
    """
    with selected_brightway_project(names) as bd:
        rows = _provider_rows(bd, providers, services)
        state = create_provider_response_state(
            bd,
            tuple(
                ContributionMethod(
                    method.method_id,
                    method.method_version,
                    method.impact_category_id,
                    method.method_tuple,
                    method.unit,
                )
                for method in methods
            ),
        )
        bank = build_provider_inventory_bank(bd, rows, state)
    return (
        _service_responses(bank.unit_responses, rows, registry, background_scenario),
        _timex_weights(providers, background_scenario),
    )


def _write_prepared(
    workspace: Path,
    names: BrightwayNames,
    *,
    background_scenario: str,
) -> tuple[PreparedManifest, PreparedManifest]:
    """Publish complete prepared metadata."""
    methods = pd.DataFrame(
        [
            {
                "selection_order": order,
                "method_id": "ef31",
                "method_version": "3.1",
                "impact_category_id": method[-1],
                "impact_category_tuple_json": json.dumps(method, separators=(",", ":")),
                "unit": "kilogram carbon dioxide",
                "factor_identity": f"bounded_factor_{order}",
            }
            for order, method in enumerate(
                (
                    ("bounded public", "climate change"),
                    ("bounded public", "climate change doubled"),
                ),
                start=1,
            )
        ]
    )
    shared, members = write_prepared_inputs(
        workspace,
        names,
        methods=methods,
        provider=resolved_infiltration_provider(workspace / "native_infiltration"),
        carbon_rows=prepared_material_carbon(workspace / "native_carbon"),
        backgrounds=(background_scenario,),
    )
    return shared, members[background_scenario]


def _write_brightway_fixture(
    names: BrightwayNames,
    *,
    service_codes: tuple[str, ...] = (_PROVIDER_SERVICE_CODE,),
) -> None:
    """Install unit-emission services and dated frontiers, using the actual foreground calculation.

    Each kilogram of any service emits one invented kilogram of CO2 and an
    uncharacterized flow. This tests LCA arithmetic; the emissions use the stated synthetic unit
    factors.
    """
    biosphere = (names.biosphere_database, "co2")
    unmarked = (names.biosphere_database, "ordinary_unmarked_flow")
    with selected_brightway_project(names) as bd:
        bd.geomapping.add(H12_REGIONS)
        bd.Database(names.biosphere_database).write(
            {
                biosphere: {
                    "name": "carbon dioxide",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                    "CAS number": "000124-38-9",
                },
                unmarked: {
                    "name": "ordinary unmarked flow",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("water",),
                },
            },
            searchable=False,
        )
        bd.preferences["biosphere_database"] = names.biosphere_database
        for database in (
            names.source_database,
            *(f"remind_ssp2_npi_anchor_{year}" for year in GENERAL_BACKGROUND_ANCHOR_YEARS),
        ):
            bd.Database(database).write(
                {(database, "unused_native_process"): _process(database, "unused_native_process")},
                searchable=False,
            )
        for database in (
            PROVIDER_INTERFACE_DATABASE,
            *(
                provider_timeline_database_name(
                    year, interface_database=PROVIDER_INTERFACE_DATABASE
                )
                for year in PROVIDER_TECHNICAL_YEARS[1:]
            ),
        ):
            bd.Database(database).write(
                {
                    (database, code): {
                        **_process(database, code),
                        "exchanges": [
                            {"input": (database, code), "amount": 1.0, "type": "production"},
                            {"input": biosphere, "amount": 0.0, "type": "biosphere"},
                        ],
                    }
                    for code in service_codes
                },
                searchable=False,
            )
        for year in PROVIDER_TECHNICAL_YEARS:
            database = provider_frontier_database_name(
                "remind_ssp2_npi", year, interface_database=PROVIDER_INTERFACE_DATABASE
            )
            native = (database, "unit_emission_operation")
            general = names.source_database if year <= 2025 else f"remind_ssp2_npi_anchor_{year}"
            bd.Database(database).write(
                {
                    native: _process(database, native[1], biosphere=biosphere, unmarked=unmarked),
                    **{
                        (database, code): {
                            **_process(database, code, provider=native),
                            "provider_source_database": general,
                            "provider_source_role": (
                                "native_ecoinvent_provider"
                                if year <= 2025
                                else "matching_member_premise_provider"
                            ),
                            "general_background_database": general,
                            "general_background_anchor_year": max(2025, year),
                            "general_background_role": (
                                "ordinary_ecoinvent"
                                if year <= 2025
                                else "matching_member_premise_general_background"
                            ),
                            "selected_background_member": "remind_ssp2_npi",
                            "direct_treatment_carbon_owners_json": json.dumps(
                                [
                                    {
                                        "owner_database": database,
                                        "owner_code": native[1],
                                        "owner_scale_per_provider_unit": 1.0,
                                        "owner_scope": "direct_biosphere_only",
                                        "source_role": "native_treatment_operation",
                                        "origin_mapping_status": "native_identity",
                                    }
                                ]
                            ),
                        }
                        for code in service_codes
                    },
                },
                searchable=False,
            )
        for method, factor in (
            (("bounded public", "climate change"), 1.0),
            (("bounded public", "climate change doubled"), 2.0),
        ):
            bd.Method(method).register(unit="kilogram carbon dioxide")
            bd.Method(method).write([(biosphere, factor)])


def _physical_foreground(names: BrightwayNames, routes: ProviderRoutes) -> ForegroundProduct:
    """Build twelve actual roots from a small cohort, recycling and virgin-input account."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    state, inputs = empty_model("HDPE")
    good = state.goods.index("Packaging")
    # Each region receives 2 kg in 2020 and 1 kg in 2021, with 1 kg discarded in 2021.
    state.annual_inflow[2, :, good] = 2e-9
    state.stock_by_cohort[2, 2, :, good] = 2e-9
    state.annual_inflow[3, :, good] = 1e-9
    state.stock_by_cohort[3, 2, :, good] = 1e-9
    state.stock_by_cohort[3, 3, :, good] = 1e-9
    state.outflow_by_cohort[3, 2, :, good] = 1e-9
    inputs.mechanical_recycling_rate[3] = 0.5
    inputs.incineration_rate[3] = 0.5
    mfa = evaluate_mfa(state, inputs, recipe)
    activations = compile_foreground_activations(mfa, routes, recipe).activations
    _write_brightway_fixture(
        names, service_codes=tuple(sorted(activations.provider_service_code.unique()))
    )
    return build_foreground_database(
        "constituent_fixture",
        names=names,
        routes=routes,
        mfa_result_identity="bounded_mfa_identity",
        mfa=mfa,
        recipe=recipe,
    )


@pytest.mark.scientific_integration
def test_response_builder_uses_one_multi_rhs_provider_solve(tmp_path: Path) -> None:
    workspace = set_workspace(tmp_path).workspace
    background_scenario = "remind_ssp2_npi"
    names = BrightwayNames(
        store=workspace / "data_processed" / "brightway",
        project="bounded_response_project",
        biosphere_database="bounded_response_biosphere",
        source_database="bounded_response_ecoinvent",
    )
    shared_reference, member_reference = _write_prepared(
        workspace,
        names,
        background_scenario=background_scenario,
    )
    _write_brightway_fixture(names)
    member = workspace / "data_processed" / "prepared_backgrounds" / background_scenario
    exact_method = normalized_lcia_method(("bounded public", "climate change"))
    methods, registry = _resolved_response_methods(
        names,
        ResponseMethodSelection(
            readable_name="bounded_climate",
            selections=(exact_method,),
            groups=(),
        ),
        pd.read_parquet(member / "impact_categories.parquet"),
    )
    responses, weights = _fixed_service_response(
        names,
        background_scenario,
        methods,
        registry,
        pd.DataFrame.from_records(
            [
                {
                    "provider_service_id": _PROVIDER_SERVICE_ID,
                    "provider_service_code": _PROVIDER_SERVICE_CODE,
                }
            ]
        ),
        load_prepared_provider_product(shared_reference, member_reference),
    )

    assert len(responses) == len(PROVIDER_TECHNICAL_YEARS)
    assert set(responses["provider_service_id"]) == {_PROVIDER_SERVICE_ID}
    # One unit of native treatment emits one characterized kilogram; the
    # separate negative elementary flow has no factor in this method.
    assert responses["unit_score"].eq(1.0).all()
    assert responses["unit_contribution_norm"].eq(1.0).all()
    assert set(weights["occurrence_year"]) == set(range(2019, 2101))
    assert weights.groupby("occurrence_year")["temporal_provider_share"].sum().eq(1.0).all()
    assert len(registry) == 1


@pytest.mark.scientific_integration
def test_result_owner_writes_twelve_regional_products(tmp_path: Path) -> None:
    workspace = set_workspace(tmp_path).workspace
    foreground_scenario = "constituent_fixture"
    background_scenario = "remind_ssp2_npi"
    names = BrightwayNames(
        store=workspace / "data_processed" / "brightway",
        project="bounded_public_project",
        biosphere_database="bounded_biosphere",
        source_database="bounded_ecoinvent",
    )
    shared_reference, member_reference = _write_prepared(
        workspace,
        names,
        background_scenario=background_scenario,
    )
    foreground = _physical_foreground(names, load_provider_routes(shared_reference))
    result_directory = (
        workspace / "results" / foreground_scenario / "lca_results" / "bounded_result_owner"
    )
    mfa_manifest_path = (
        workspace / "results" / foreground_scenario / "mfa_results" / "manifest.json"
    )
    background_manifest_path = (
        workspace
        / "data_processed"
        / "prepared_backgrounds"
        / background_scenario
        / "manifest.json"
    )
    mfa_manifest_path.parent.mkdir(parents=True)
    mfa_manifest_path.write_text(
        json.dumps(
            {
                "identity": "bounded_mfa_identity",
                "foreground_scenario": foreground_scenario,
                "ssp_scenario": "SSP2",
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    method_specs = (
        (
            "climate change",
            ("bounded public", "climate change"),
            "bounded_factor_1",
        ),
        (
            "climate change doubled",
            ("bounded public", "climate change doubled"),
            "bounded_factor_2",
        ),
    )
    methods = tuple(
        ImpactCategory(
            selection_order=order,
            method_id="ef31",
            method_version="3.1",
            impact_category_id=category,
            method_tuple=method,
            unit="kilogram carbon dioxide",
            factor_identity=factor,
        )
        for order, (category, method, factor) in enumerate(method_specs, start=1)
    )
    blocked_guide = result_directory.parent / "README.md"
    blocked_guide.mkdir(parents=True)
    for guide_blocked in (True, False):
        with pytest.raises(OSError) if guide_blocked else nullcontext():
            status = write_lca_result(
                result_directory,
                foreground_scenario,
                background_scenario,
                foreground,
                methods,
                (2020, 2021),
                json.loads(mfa_manifest_path.read_text(encoding="utf-8")),
                file_sha256(mfa_manifest_path),
                json.loads(background_manifest_path.read_text(encoding="utf-8")),
                file_sha256(background_manifest_path),
                names=prepared_brightway_names(workspace, shared_reference),
                providers=load_prepared_provider_product(shared_reference, member_reference),
                pa_dump_shares=load_provider_routes(shared_reference).open_dump_shares(),
                background_dimensions=_background_dimensions(member_reference),
                replacing=False,
            )
        if guide_blocked:
            assert blocked_guide.is_dir()
            assert not result_directory.exists()
            assert not list(result_directory.parent.glob(".lca_result.*"))
            blocked_guide.rmdir()

    assert status == "computed"
    scores = pd.read_parquet(result_directory / "scores.parquet")
    annual = pd.read_parquet(result_directory / "annual_scores.parquet")
    annual_stages = pd.read_parquet(result_directory / "annual_life_cycle_stage_scores.parquet")
    cumulative_stages = pd.read_parquet(result_directory / "life_cycle_stage_scores.parquet")
    authority = pd.read_parquet(result_directory / "inventory_authority.parquet")
    contributions = pd.read_parquet(result_directory / "contributions.parquet")
    stage_contributions = pd.read_parquet(result_directory / "stage_contributions.parquet")
    carbon_consistency = pd.read_csv(result_directory / "carbon_consistency.csv")
    direct_treatment_carbon = pd.read_parquet(
        result_directory / "direct_treatment_carbon_flows.parquet"
    )
    activation_lineage = pd.read_parquet(result_directory / "activation_lineage.parquet")
    flow_coverage = pd.read_parquet(result_directory / "mfa_to_lca_flow_coverage.parquet")
    structural_paths = pd.read_parquet(result_directory / "structural_paths.parquet")
    structural_edges = pd.read_parquet(result_directory / "structural_edges.parquet")
    structural_attribution = pd.read_parquet(result_directory / "structural_attribution.parquet")
    manifest = json.loads((result_directory / "manifest.json").read_text(encoding="utf-8"))
    execution_report = json.loads(
        (result_directory / "execution_report.json").read_text(encoding="utf-8")
    )
    assert execution_report["schema_version"] == "pytoce_lca_execution_report_schema_v1"

    identity = manifest["identity_payload"]
    assert identity["mfa_result_identity"] == "bounded_mfa_identity"
    assert identity["mfa_manifest_sha256"] == file_sha256(mfa_manifest_path)
    assert identity["prepared_background_identity"] == member_reference.payload["identity"]
    assert identity["prepared_background_manifest_sha256"] == file_sha256(background_manifest_path)
    assert identity["lcia_methods"] == [
        {
            "method_id": "ef31",
            "method_version": "3.1",
            "impact_category_id": "climate change",
            "method_tuple": ["bounded public", "climate change"],
            "factor_identity": "bounded_factor_1",
        },
        {
            "method_id": "ef31",
            "method_version": "3.1",
            "impact_category_id": "climate change doubled",
            "method_tuple": ["bounded public", "climate change doubled"],
            "factor_identity": "bounded_factor_2",
        },
    ]
    assert identity["flow_ownership_method_identity"] == FLOW_OWNERSHIP_METHOD_IDENTITY

    assert set(scores["region"]) == {*H12_REGIONS, "global_derived"}
    regional_scores = scores.loc[scores["region"].isin(H12_REGIONS)]
    expected_scores = foreground.activations.groupby("region").amount.sum()
    for category, factor in (("climate change", 1.0), ("climate change doubled", 2.0)):
        observed = (
            regional_scores.loc[regional_scores.impact_category_id.eq(category)]
            .set_index("region")
            .score
        )
        pd.testing.assert_series_equal(
            observed.sort_index(),
            (expected_scores * factor).sort_index(),
            check_names=False,
            rtol=1e-13,
            atol=0,
        )
    # Ordinary LCA and optimization characterize the same provider inventory.
    # Compare their actual owners here instead of repeating LCA during every bank build.
    registry = _method_registry(methods, ("ef31",))
    services = (
        foreground.activations.loc[:, ["provider_service_id", "provider_service_code"]]
        .drop_duplicates(ignore_index=True)
        .astype("string")
    )
    unit_responses, weights = _fixed_service_response(
        names,
        background_scenario,
        methods,
        registry,
        services,
        load_prepared_provider_product(shared_reference, member_reference),
    )
    services = services.assign(provider_unit="kilogram").astype("string")
    response_bank = LoadedOptimizationResponseBank(
        identity="bounded_response_bank",
        response_method_union="ef31",
        provider_route_identity="bounded_provider_routes",
        provider_service_registry_identity=canonical_table_sha256(services),
        method_registry=registry,
        provider_service_registry=services,
        provider_unit_responses=unit_responses,
        timex_year_weights=weights,
        native_unit_responses=pd.DataFrame(columns=_NATIVE_COLUMNS),
        direct_characterization_factors=pd.DataFrame(columns=_DIRECT_COLUMNS),
    )
    projection = project_optimization_response_bank(response_bank, registry, role="result")
    _method_coordinates, effective = _effective_responses(projection)
    delta = optimization_activation_delta(
        foreground.activations.assign(amount=0.0), foreground.activations
    )
    projected_stage = _stage_response_adjustments(delta, effective, background=background_scenario)
    value_columns = ["score", "contribution_norm"]
    rename_values = {"score_adjustment": "score", "norm_adjustment": "contribution_norm"}
    projected_stage = projected_stage.rename(columns=rename_values)
    annual_keys = [
        "region",
        "year",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
    ]
    stage_keys = [
        *annual_keys,
        "life_cycle_stage",
        "stage_group",
        "cutoff_cycle_role",
        "material_cycle_source",
        "physical_event_basis",
        "physical_event_year",
    ]
    total_keys = [key for key in annual_keys if key != "year"]
    projected_annual = (
        projected_stage.groupby(annual_keys, observed=True, sort=True)[value_columns]
        .agg(fsum)
        .reset_index()
    )
    cumulative_stage_keys = [
        key for key in stage_keys if key not in {"year", "physical_event_year"}
    ]
    for expected_table, projected, keys in (
        (annual, projected_annual, annual_keys),
        (annual_stages, projected_stage, stage_keys),
        (scores, _stable_group(projected_annual, total_keys, value_columns), total_keys),
        (
            cumulative_stages,
            _stable_group(projected_stage, cumulative_stage_keys, value_columns),
            cumulative_stage_keys,
        ),
    ):
        expected = (
            expected_table.loc[expected_table.region.isin(H12_REGIONS)]
            .set_index(keys)[value_columns]
            .sort_index()
        )
        observed = projected.set_index(keys)[value_columns].sort_index()
        assert observed.index.difference(expected.index).empty
        pd.testing.assert_frame_equal(
            observed.reindex(expected.index, fill_value=0.0),
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-13,
            atol=1e-12,
        )
    assert annual.loc[~annual.year.isin((2020, 2021)), "score"].eq(0.0).all()
    assert annual.loc[~annual.year.isin((2020, 2021)), "contribution_norm"].eq(0.0).all()
    assert authority["principal_inventory_identity"].nunique() == 12
    assert authority["timeline_identity"].nunique() == 12
    loss = pd.read_parquet(result_directory / "mechanical_recycling_loss.parquet")
    expected_loss = foreground.mechanical_recycling_loss.assign(
        background_scenario=background_scenario
    )
    pd.testing.assert_frame_equal(loss, expected_loss.loc[:, loss.columns])
    hdpe_loss = loss.loc[
        loss.year.eq(2021)
        & loss.material.eq("HDPE")
        & loss.good.eq("all")
        & loss.region.isin(H12_REGIONS)
    ]
    assert sorted(hdpe_loss.region.tolist()) == sorted(H12_REGIONS)
    np.testing.assert_allclose(hdpe_loss.mechanical_recycling_input_kg, 0.5, rtol=1e-13, atol=0)
    # This native HDPE recipe sends managed rejects to incineration. Its
    # wastewater and airborne principal losses remain separate exits.
    assert hdpe_loss.mechanical_recycling_managed_reject_landfilled_kg.eq(0.0).all()
    assert hdpe_loss.mechanical_recycling_managed_reject_incinerated_kg.gt(0.0).all()
    assert hdpe_loss.mechanical_recycling_waterborne_loss_kg.gt(0.0).all()
    assert hdpe_loss.mechanical_recycling_airborne_loss_kg.gt(0.0).all()
    np.testing.assert_allclose(
        hdpe_loss.mechanical_recycling_output_kg + hdpe_loss.mechanical_recycling_total_loss_kg,
        hdpe_loss.mechanical_recycling_input_kg,
        rtol=1e-13,
        atol=0,
    )
    assert manifest["schema_version"] == "pytoce_lca_result_schema_v1"
    assert (
        manifest["identity_payload"]["contribution_method_identity"]
        == "timex_frontier_native_flow_stage_contributions_v10"
    )
    assert (
        manifest["identity_payload"]["carbon_consistency_method_identity"]
        == "mfa_native_direct_treatment_carbon_v11"
    )
    assert (
        manifest["identity_payload"]["direct_treatment_carbon_flows_identity"]
        == "pytoce_direct_treatment_carbon_flows_v5"
    )
    assert {
        "contributions.parquet",
        "annual_life_cycle_stage_scores.parquet",
        "life_cycle_stage_scores.parquet",
        "stage_contributions.parquet",
        "carbon_consistency.csv",
        "direct_treatment_carbon_flows.parquet",
        "mechanical_recycling_loss.parquet",
        "pa_receiving_inventory.parquet",
        "pa_receiving_physical_events.parquet",
        "activation_lineage.parquet",
        "mismanaged_waste_fate_closure.parquet",
        "mfa_to_lca_flow_coverage.parquet",
        "structural_paths.parquet",
        "structural_edges.parquet",
        "structural_attribution.parquet",
    }.issubset(manifest["artifacts"])
    for artifact_name, descriptor in manifest["artifacts"].items():
        artifact_path = result_directory / artifact_name
        assert descriptor["path"] == artifact_name
        assert descriptor["artifact_sha256"] == file_sha256(artifact_path)
    assert activation_lineage["provider_member"].eq(background_scenario).all()
    assert activation_lineage["provider_demand_unit"].eq("kilogram").all()
    assert activation_lineage["provider_source_role"].eq("native_ecoinvent_provider").all()
    assert activation_lineage["general_background_role"].eq("ordinary_ecoinvent").all()
    assert activation_lineage["general_background_anchor_year"].eq(2025).all()
    assert activation_lineage["selected_background_member"].eq(background_scenario).all()
    assert not direct_treatment_carbon.empty
    expected_coverage = foreground.mfa_to_lca_flow_coverage.assign(
        background_scenario=background_scenario
    )
    pd.testing.assert_frame_equal(flow_coverage, expected_coverage.loc[:, flow_coverage.columns])
    stage_total_keys = [
        "foreground_scenario",
        "background_scenario",
        "region",
        "year",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
    ]
    observed_annual = annual_stages.groupby(stage_total_keys, sort=True)[
        ["score", "contribution_norm"]
    ].sum()
    expected_annual = annual.set_index(stage_total_keys)[["score", "contribution_norm"]]
    pd.testing.assert_frame_equal(
        observed_annual,
        expected_annual.sort_index(),
        check_exact=False,
        atol=1e-12,
        rtol=0.0,
    )
    global_stage_keys = [
        column
        for column in annual_stages.columns
        if column
        not in {
            "foreground_scenario",
            "background_scenario",
            "region",
            "score",
            "contribution_norm",
            "share",
            "norm_share",
            "share_status",
        }
    ]
    regional_stage_sum = (
        annual_stages.loc[annual_stages["region"].isin(H12_REGIONS)]
        .groupby(global_stage_keys, sort=True)[["score", "contribution_norm"]]
        .sum()
    )
    global_stage = annual_stages.loc[annual_stages["region"].eq("global_derived")].set_index(
        global_stage_keys
    )[["score", "contribution_norm"]]
    pd.testing.assert_frame_equal(
        regional_stage_sum,
        global_stage.sort_index(),
        check_exact=False,
        atol=1e-12,
        rtol=0.0,
    )
    cumulative_keys = [column for column in stage_total_keys if column != "year"]
    observed_cumulative = cumulative_stages.groupby(cumulative_keys, sort=True)[
        ["score", "contribution_norm"]
    ].sum()
    expected_cumulative = scores.set_index(cumulative_keys)[["score", "contribution_norm"]]
    pd.testing.assert_frame_equal(
        observed_cumulative,
        expected_cumulative.sort_index(),
        check_exact=False,
        atol=1e-12,
        rtol=0.0,
    )
    assert annual_stages["physical_event_year"].equals(annual_stages["year"])
    assert cumulative_stages["year"].isna().all()
    assert cumulative_stages["physical_event_year"].isna().all()
    assert cumulative_stages["period_start_year"].eq(2019).all()
    assert cumulative_stages["period_end_year"].eq(2100).all()
    managed_reject = carbon_consistency.loc[
        carbon_consistency["life_cycle_stage"].eq(
            "mechanical_recycling_reject_incineration_treatment"
        )
    ]
    assert set(managed_reject.h12_region) == set(H12_REGIONS)
    assert managed_reject["fate_origin"].eq("remind_mfa_physical_route").all()
    expected_carbon = (
        foreground.activations.loc[
            foreground.activations.demand_type.eq(
                "mechanical_recycling_reject_incineration_treatment"
            )
        ]
        .groupby("region")
        .mfa_carbon_amount_kg.sum()
    )
    pd.testing.assert_series_equal(
        managed_reject.groupby("h12_region").mfa_treatment_input_carbon_kg.sum().sort_index(),
        expected_carbon.sort_index(),
        check_names=False,
    )
    np.testing.assert_allclose(
        managed_reject.groupby("h12_region").native_direct_selected_carbon_kg.sum().sort_index(),
        # The fixture emits one kg CO2 per kg service; 12/44 converts molecular mass to carbon.
        hdpe_loss.set_index(
            "region"
        ).mechanical_recycling_managed_reject_incinerated_kg.sort_index()
        * 12.0
        / 44.0,
        rtol=1e-13,
        atol=0,
    )
    np.testing.assert_allclose(
        managed_reject.selected_native_minus_mfa_carbon_kg,
        managed_reject.native_direct_selected_carbon_kg
        - managed_reject.mfa_treatment_input_carbon_kg,
        rtol=1e-13,
        atol=0,
    )
    for view in (
        "life_cycle_stage",
        "provider_frontier",
        "native_activity",
        "elementary_flow",
        "classification",
    ):
        selected = contributions.loc[
            contributions["contribution_view"].eq(view)
            & contributions["region"].eq("EUR")
            & contributions["impact_category_id"].eq("climate change")
        ]
        assert selected["score"].sum() == pytest.approx(expected_scores.loc["EUR"])
        assert selected["contribution_norm"].sum() == pytest.approx(expected_scores.loc["EUR"])
        stage_selected = stage_contributions.loc[
            stage_contributions["contribution_view"].eq(view)
            & stage_contributions["region"].eq("EUR")
            & stage_contributions["impact_category_id"].eq("climate change")
        ]
        assert stage_selected["score"].sum() == pytest.approx(expected_scores.loc["EUR"])
        assert stage_selected["contribution_norm"].sum() == pytest.approx(
            expected_scores.loc["EUR"]
        )
    assert structural_paths["additivity"].eq("non_additive_explanation").all()
    assert structural_edges["additivity"].eq("non_additive_explanation").all()
    attribution_scores = structural_attribution.groupby(
        ["region", "impact_category_id"],
        sort=True,
    )["score"].sum()
    expected_scores = scores.set_index(["region", "impact_category_id"])["score"]
    pd.testing.assert_series_equal(
        attribution_scores,
        expected_scores.sort_index(),
        check_names=False,
    )
    dimensions = {
        "background_scenario": background_scenario,
        "ssp_scenario": "SSP2",
        "iam_model": "REMIND",
        "iam_scenario": member_reference.payload["iam_scenario"],
        "gmst": member_reference.payload["gmst"],
        "gmst_source": member_reference.payload["gmst_source"],
        "premise_version": "2.4.6",
        "ecoinvent_version": "3.12",
        "system_model": "cutoff",
    }
    for table in (scores, annual, authority):
        assert (table[list(dimensions)].drop_duplicates().to_dict(orient="records")) == [dimensions]
    assert {key: manifest["identity_payload"][key] for key in dimensions} == dimensions
    timex_identity = manifest["identity_payload"]["timex"]
    assert timex_identity["principal_inventory_method"] == (
        "timex_multi_root_timeline_brightway_multi_rhs_v48"
    )
    assert timex_identity["root_count"] == 12
    assert timex_identity["cutoff"] == pytest.approx(1e-15 / 12)
    assert timex_identity["max_calc"] == 24_000
    assert timex_identity["unit_root_cutoff"] == pytest.approx(1e-15)
    assert timex_identity["unit_root_max_calc"] == 2_000


@pytest.mark.scientific_integration
def test_public_lca_calculates_from_completed_mfa_and_reuses_result(
    tmp_path: Path, remind_mfa_source_root: Path, external_remind_mfa_import_state: None
) -> None:
    """Carry a full-coordinate published MFA through public LCA and completed reuse."""
    scenario = "bounded_public_parent"
    parent = prepare_mfa_workspace(tmp_path, remind_mfa_source_root, foreground_scenario=scenario)
    background = "remind_ssp2_npi"
    names = BrightwayNames(
        store=tmp_path / "data_processed/brightway",
        project="bounded_mfa_lca_project",
        biosphere_database="bounded_mfa_lca_biosphere",
        source_database="bounded_mfa_lca_ecoinvent",
    )
    shared, member = _write_prepared(tmp_path, names, background_scenario=background)
    parent_manifest = parent.reference.path
    activations = compile_foreground_activations(
        parent.mfa, load_provider_routes(shared), parent.recipe
    ).activations
    _write_brightway_fixture(
        names, service_codes=tuple(sorted(activations.provider_service_code.unique()))
    )
    parent_bytes = parent_manifest.read_bytes()

    report = run_lca(
        foreground_scenario=scenario,
        background_scenario=background,
        lcia_method=("bounded public", "climate change doubled"),
        upstream_analysis_year=2019,
    )

    assert report.status == "computed"
    assert len(report.results) == 1
    item = report.results[0]
    assert item.status == "computed"
    manifest_bytes = item.manifest_path.read_bytes()
    identity = json.loads(manifest_bytes)["identity_payload"]
    assert identity["mfa_result_identity"] == json.loads(parent_bytes)["identity"]
    assert identity["prepared_background_identity"] == member.payload["identity"]
    annual = pd.read_parquet(item.annual_scores)
    coordinates = pd.MultiIndex.from_product(
        [H12_REGIONS, range(2019, 2101)], names=["region", "year"]
    )
    # Each synthetic service emits one kg CO2 per unit. Select the fixture method's exact tuple with
    # factor 2.
    expected = activations.groupby(["region", "year"]).amount.agg(fsum)
    expected = expected.reindex(coordinates, fill_value=0.0).sort_index() * 2.0
    assert annual.impact_category_id.eq("climate change doubled").all()
    observed = annual.loc[annual.region.isin(H12_REGIONS)].set_index(["region", "year"])
    pd.testing.assert_series_equal(
        observed.score.sort_index(), expected, check_names=False, rtol=1e-12, atol=0.0
    )
    assert annual.unit.eq("kilogram carbon dioxide").all()
    global_values = annual.loc[annual.region.eq("global_derived")].set_index("year")
    pd.testing.assert_series_equal(
        global_values.score.sort_index(),
        expected.groupby("year").agg(fsum),
        check_names=False,
        rtol=1e-12,
        atol=0.0,
    )

    reused = run_lca(
        foreground_scenario=scenario,
        background_scenario=background,
        lcia_method=("bounded public", "climate change doubled"),
        upstream_analysis_year=2019,
    )
    assert reused.status == "reused"
    assert reused.results[0].manifest_path == item.manifest_path
    assert item.manifest_path.read_bytes() == manifest_bytes
    assert parent_manifest.read_bytes() == parent_bytes
