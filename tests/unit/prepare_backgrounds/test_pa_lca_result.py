"""Write polyamide receiving results with fixed suppliers whose emissions are known."""

import json
from math import fsum
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from infiltration_fixtures import resolved_infiltration_provider
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.run_lca.calculation.carbon import TREATMENT_STAGES, pa_direct_carbon_flows
from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    create_contribution_state,
)
from pytoce.run_lca.calculation.direct_contributions import (
    add_direct_contributions,
    admit_direct_characterization_factors,
)
from pytoce.run_lca.calculation.foreground import (
    H12_REGIONS,
    build_foreground_database,
    compile_foreground_activations,
)
from pytoce.run_lca.calculation.lcia import ImpactCategory
from pytoce.run_lca.calculation.pa_inventory import build_pa_receiving_inventory
from pytoce.run_lca.calculation.pa_projection import iter_pa_projection
from pytoce.run_lca.calculation.providers import (
    MaterialCarbon,
    PreparedProviderProduct,
    ProviderRoutes,
)
from pytoce.run_lca.calculation.results import write_lca_result
from pytoce.run_lca.calculation.timex import build_provider_inventory_bank


def _fixed_services(bd: Any, codes: tuple[str, ...], interface: str, flow: tuple[str, str]) -> None:
    """Register suppliers with unit emissions and Timex nodes with zero emissions.

    Each fixed supplier emits one kg of the selected flow per kg of service.
    These synthetic emissions test result assembly, not resin manufacture or
    conversion inventories. Separate databases preserve the original and dated
    inventories used to calculate polyamide receiving quantities.
    """
    for database in (interface, "pa_result_timeline_2025", "pa_result_fixed"):
        rows = {}
        for code in codes:
            key = database, code
            rows[key] = {
                "name": "Invented fixed service " + code,
                "reference product": "invented fixed service",
                "unit": "kilogram",
                "location": "GLO",
                "provider_source_database": "original",
                "provider_source_role": "native_ecoinvent_provider",
                "general_background_database": "original",
                "general_background_anchor_year": 2025,
                "general_background_role": "ordinary_ecoinvent",
                "selected_background_member": "remind_ssp2_npi",
                "direct_treatment_carbon_owners_json": json.dumps(
                    [
                        {
                            "owner_database": database,
                            "owner_code": code,
                            "owner_scale_per_provider_unit": 1.0,
                            "owner_scope": "direct_biosphere_only",
                            "source_role": "native_treatment_operation",
                            "origin_mapping_status": "native_identity",
                        }
                    ]
                ),
                "exchanges": [
                    {"input": key, "amount": 1.0, "type": "production"},
                    {
                        "input": flow,
                        "amount": 1.0 if database == "pa_result_fixed" else 0.0,
                        "type": "biosphere",
                    },
                ],
            }
        bd.Database(database).write(rows, searchable=False)


@pytest.mark.parametrize(
    "fixed_flow_name", ["Carbon dioxide, fossil", "Sulfur dioxide"], ids=["co2", "so2"]
)
def test_pa_result_owner_preserves_actual_receiving_and_complete_stage_totals(
    tmp_path: Path,
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
    fixed_flow_name: str,
) -> None:
    """Write fiber MFA, foreground, Timex and LCA results using synthetic inventories.

    Polyamide receiving quantities use the existing synthetic source fixture.
    Fixed suppliers each emit one kg of a selected flow per kg of service.
    Their scaled impacts are added to independently accumulated receiving
    impacts, then compared with annual and stage totals and material/carbon
    records. The synthetic suppliers provide controlled test inventories.
    A fixed non-carbon emission contributes to impacts while leaving only
    polyamide receiving emissions in the carbon records.
    """
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    # One kg per cohort: MFA uses Mt, so 1e-9 Mt is one kg. Three synthetic
    # annual cohorts produce a second recycling cycle and mixed-origin EOL.
    for time in (0, 1, 2):
        state.annual_inflow[time, region, good] = 1e-9
        state.stock_by_cohort[time, time, region, good] = 1e-9
        state.outflow_by_cohort[time + 1, time, region, good] = 1e-9
    for time in (1, 2):
        inputs.mechanical_recycling_rate[time, region] = 0.5
        inputs.incineration_rate[time, region] = 0.25
    inputs.bio_production_rate[2, region] = 0.35
    inputs.daccu_production_rate[2, region] = 0.25
    inputs.collection_rate[3, region] = 0.6
    mfa = evaluate_mfa(state, inputs, recipe)
    authority = resolved_infiltration_provider(tmp_path / "infiltration")
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    root = tmp_path / "native"
    names = BrightwayNames(root / "brightway", "pa_receiving", "biosphere", "original")
    result_directory = tmp_path / "results" / "pa_fixture" / "lca_results" / "original"
    methods = tuple(
        ImpactCategory(index, "ef31", "3.1", label, ("PA result", label), "test eq", label)
        for index, label in enumerate(("unit", "double"))
    )
    with pa_event_receiving_boundary(root, original=True) as boundary:
        bd = boundary.bd
        bd.geomapping.add(H12_REGIONS)
        bd.preferences["biosphere_database"] = "biosphere"
        flows = tuple(bd.Database("biosphere"))
        fixed_flow = next(flow.key for flow in flows if flow["name"] == fixed_flow_name)
        for method, factor in zip(methods, (1.0, 2.0), strict=True):
            native_method = bd.Method(method.method_tuple)
            native_method.register(unit=method.unit)
            native_method.write([(flow.key, factor) for flow in flows])
        compiled = compile_foreground_activations(mfa, routes, recipe)
        _fixed_services(
            bd,
            tuple(sorted(compiled.activations.provider_service_code.unique())),
            authority.interface_database,
            fixed_flow,
        )
        foreground = build_foreground_database(
            "pa_fixture",
            names=names,
            routes=routes,
            mfa_result_identity="mfa::pa_fixture",
            mfa=mfa,
            recipe=recipe,
        )
        assert len(foreground.root_keys) == 12
        assert foreground.pa_spinning_events and foreground.pa_recycling_events
        assert foreground.pa_disposal_events
        providers = PreparedProviderProduct(
            {2020: "pa_result_fixed", 2025: "pa_result_fixed"},
            {2020: authority.interface_database, 2025: "pa_result_timeline_2025"},
            boundary.reference,
            {2025: boundary.sources},
        )
        projected = tuple(
            iter_pa_projection(
                spinning_events=foreground.pa_spinning_events,
                recycling_events=foreground.pa_recycling_events,
                disposal_events=foreground.pa_disposal_events,
                reference=boundary.reference,
                sources_by_year={2025: boundary.sources},
                matrix_by_year=providers.database_by_year,
                dump_shares=routes.open_dump_shares(),
            )
        )
        roots: dict[str, tuple[str, str]] = dict(
            zip(H12_REGIONS, foreground.root_keys, strict=True)
        )
        actual = build_pa_receiving_inventory(bd, projected, roots)
        assert actual.activations.amount.lt(0).any()
        assert not actual.direct_inventory.empty
        contribution_state = create_contribution_state(
            bd,
            {"EUR": actual.activations},
            roots,
            tuple(
                ContributionMethod(
                    m.method_id, m.method_version, m.impact_category_id, m.method_tuple, m.unit
                )
                for m in methods
            ),
            (2019, 2020, 2021, 2022),
        )
        admit_direct_characterization_factors(
            contribution_state,
            sorted(set(int(x) for x in actual.direct_inventory.flow_id)),
        )
        add_direct_contributions(contribution_state, actual.direct_contributions)
        build_provider_inventory_bank(bd, actual.activations, contribution_state)
        expected_annual = dict(contribution_state.annual_totals)
        expected_carbon = pa_direct_carbon_flows(projected)
        expected_records = actual.records.copy()
        status = write_lca_result(
            result_directory,
            "pa_fixture",
            "remind_ssp2_npi",
            foreground,
            methods,
            (2019, 2020, 2021, 2022),
            {"identity": "mfa::pa_fixture", "foreground_scenario": "pa_fixture"},
            "mfa-fixture",
            {"identity": "background::pa_fixture"},
            "background-fixture",
            names=names,
            providers=providers,
            pa_dump_shares=routes.open_dump_shares(),
            background_dimensions={"ssp_scenario": "SSP2"},
            replacing=False,
        )
    assert status == "computed"
    annual = pd.read_parquet(result_directory / "annual_scores.parquet")
    stages = pd.read_parquet(result_directory / "annual_life_cycle_stage_scores.parquet")
    fixed = foreground.activations.groupby(["region", "year"]).amount.agg(fsum)
    for (region_name, year, index), (score, norm) in expected_annual.items():
        selected = annual.loc[
            annual.region.eq(region_name)
            & annual.year.eq(year)
            & annual.impact_category_id.eq(methods[index].impact_category_id)
        ]
        fixed_score = float(fixed.get((region_name, year), 0.0)) * (index + 1)
        assert selected.iloc[0].score == pytest.approx(score + fixed_score, rel=3e-7)
        assert selected.iloc[0].contribution_norm == pytest.approx(norm + fixed_score, rel=3e-7)
    for category in ("unit", "double"):
        regional = annual.loc[
            annual.impact_category_id.eq(category) & annual.region.isin(H12_REGIONS)
        ]
        global_rows = annual.loc[
            annual.impact_category_id.eq(category) & annual.region.eq("global_derived")
        ]
        assert fsum(global_rows.score) == pytest.approx(fsum(regional.score))
    grouped = stages.groupby(["region", "year", "impact_category_id"])[
        ["score", "contribution_norm"]
    ].sum()
    observed = annual.set_index(["region", "year", "impact_category_id"])[
        ["score", "contribution_norm"]
    ]
    pd.testing.assert_frame_equal(
        grouped.sort_index(), observed.sort_index(), check_dtype=False, rtol=3e-7
    )
    published = pd.read_parquet(result_directory / "pa_receiving_inventory.parquet")
    pd.testing.assert_frame_equal(
        published.drop(columns=["foreground_scenario", "background_scenario"]),
        expected_records,
        check_dtype=False,
    )
    physical = pd.read_parquet(result_directory / "pa_receiving_physical_events.parquet")
    assert len(physical) == len(projected)
    assert set(physical.constituent) == {"polyamide_6_resin", "polyamide_66_resin"}
    assert physical.temporal_share.eq(1.0).all()
    assert set(physical.infiltration_class) - {""} == set(routes.open_dump_shares())
    for row in physical.itertuples(index=False):
        payload = json.loads(str(row.physical_receiving_json))
        assert '"amounts":' not in str(row.physical_receiving_json)
        event = json.loads(str(row.physical_event_json))
        if row.life_cycle_stage == "processing_conversion":
            assert event["finished_principal_kg"] > 0
            assert payload["water"]["gate_water_kg"] > 0
    carbon_rows = pd.read_parquet(result_directory / "direct_treatment_carbon_flows.parquet")
    ordinary_carbon = carbon_rows.loc[carbon_rows.source_role.eq("native_treatment_operation")]
    if fixed_flow_name == "Carbon dioxide, fossil":
        assert not ordinary_carbon.empty
        assert ordinary_carbon.owner_database.eq("pa_result_fixed").all()
        assert ordinary_carbon.flow_name.eq(fixed_flow_name).all()
        assert ordinary_carbon.flow_unit.eq("kilogram").all()
        assert ordinary_carbon.flow_amount_per_provider_unit.eq(1.0).all()
        assert ordinary_carbon.carbon_unit.eq("kilogram carbon").all()
        coordinates = [
            "year",
            "region",
            "life_cycle_stage",
            "constituent",
            "ultimate_carbon_origin",
            "provider_code",
        ]
        expected_treatment = foreground.activations.loc[
            foreground.activations.demand_type.isin(TREATMENT_STAGES)
        ].rename(
            columns={"demand_type": "life_cycle_stage", "provider_service_code": "provider_code"}
        )
        published_treatment = ordinary_carbon.rename(
            columns={"h12_region": "region", "mfa_carbon_origin": "ultimate_carbon_origin"}
        )
        pd.testing.assert_series_equal(
            published_treatment.groupby(coordinates).provider_demand_amount.agg(fsum),
            expected_treatment.groupby(coordinates).amount.agg(fsum),
            check_names=False,
        )
        # Fixed treatment services each emit one kg CO2 per kg service. Converting
        # its molecular mass to carbon gives 12/44 kg C per kg CO2; PA gas releases
        # remain in their independently calculated receiving stream below.
        pd.testing.assert_series_equal(
            ordinary_carbon.carbon_amount,
            ordinary_carbon.provider_demand_amount * (12.0 / 44.0),
            check_names=False,
        )
    else:
        assert ordinary_carbon.empty
    pa_carbon = carbon_rows.loc[carbon_rows.source_role.eq("modelled_direct_pa_receiving")]
    assert len(carbon_rows) == len(ordinary_carbon) + len(pa_carbon)
    # Parquet stores unset native-producer fields in nullable columns. Compare the stored null
    # representation.
    observed_carbon = (
        pa_carbon.drop(columns=["foreground_scenario", "background_scenario"])
        .reset_index(drop=True)
        .astype(object)
    )
    expected_carbon = expected_carbon.reset_index(drop=True).astype(object)
    pd.testing.assert_frame_equal(
        observed_carbon.where(observed_carbon.notna()),
        expected_carbon.where(expected_carbon.notna()),
        check_dtype=False,
    )
    assert set(pa_carbon.mfa_carbon_origin) == {"fossil", "biogenic", "atmospheric_dac"}
    assert (result_directory / "manifest.json").is_file()
