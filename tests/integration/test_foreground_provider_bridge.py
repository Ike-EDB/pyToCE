from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from bw_temporalis import TemporalDistribution
from bw_timex import TimexLCA

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    create_contribution_state,
    create_provider_response_state,
)
from pytoce.run_lca.calculation.lcia import (
    characterize_principal_inventory,
    normalized_lcia_method,
    resolve_lcia_methods_from_authority,
)
from pytoce.run_lca.calculation.timex import (
    GRAPH_TRAVERSAL,
    INTERPOLATION_TYPE,
    MAX_CALCULATIONS,
    STARTING_DATETIME,
    TEMPORAL_GROUPING,
    TRAVERSAL_CUTOFF,
    CharacterizationFactorBank,
    PrincipalTimeline,
    _provider_activations,
    assemble_principal_inventory,
    build_principal_timelines,
    build_provider_inventory_bank,
    compact_principal_timelines,
    database_dates,
)


def _temporal_distribution(
    years: tuple[int, ...],
    amounts: tuple[float, ...],
) -> TemporalDistribution:
    return TemporalDistribution(
        date=np.array([np.datetime64(f"{year}-01-01T00:00:00") for year in years]),
        amount=np.array(amounts, dtype=np.float64),
    )


def _provider_dataset(
    database: str,
    code: str,
    emission: float,
) -> dict[str, object]:
    anchor_year = 2025 if database == "provider_2025" else 2030
    general_background_role = (
        "ordinary_ecoinvent"
        if anchor_year == 2025
        else "matching_member_premise_general_background"
    )
    return {
        "name": "provider service",
        "reference product": "provider service",
        "unit": "kilogram",
        "location": "GLO",
        "type": "process",
        "provider_source_database": database,
        "provider_source_role": (
            "native_ecoinvent_provider"
            if anchor_year == 2025
            else "matching_member_premise_provider"
        ),
        "general_background_database": database,
        "general_background_anchor_year": anchor_year,
        "general_background_role": general_background_role,
        "selected_background_member": "remind_ssp2_npi",
        "exchanges": [
            {
                "input": (database, code),
                "amount": 1.0,
                "type": "production",
            },
            {
                "input": ("biosphere", "co2"),
                "amount": emission,
                "type": "biosphere",
            },
        ],
    }


def _timex_result(
    root: object,
    method: tuple[str, str],
    dates: dict[str, str | datetime],
) -> tuple[pd.DataFrame, float]:
    calculation = TimexLCA(
        {root: 1.0},
        method,
        dates,
        use_global_lci_cache=False,
    )
    calculation.build_timeline(
        starting_datetime=STARTING_DATETIME,
        temporal_grouping=TEMPORAL_GROUPING,
        interpolation_type=INTERPOLATION_TYPE,
        edge_filter_function=None,
        cutoff=TRAVERSAL_CUTOFF,
        max_calc=MAX_CALCULATIONS,
        graph_traversal=GRAPH_TRAVERSAL,
        traverse_background=False,
    )
    calculation.lci(
        build_dynamic_biosphere=True,
        expand_technosphere=True,
    )
    calculation.static_lcia()
    inventory = (
        calculation.dynamic_inventory_df.groupby("date", as_index=False, sort=True)["amount"]
        .sum()
        .reset_index(drop=True)
    )
    inventory["year"] = pd.to_datetime(inventory.pop("date")).dt.year
    return inventory.loc[:, ["year", "amount"]], float(calculation.static_score)


def _single_root_principal_timeline(
    root: object,
    method: tuple[str, ...],
    dates: dict[str, str | datetime],
    bd: Any,
    activations: pd.DataFrame,
    provider_databases: dict[int, str],
    timeline_databases: dict[int, str],
) -> PrincipalTimeline:
    """Build an independent installed-API reference for batched timeline comparison."""

    calculation = TimexLCA(
        {root: 1.0},
        method,
        dates,
        use_global_lci_cache=False,
    )
    timeline = calculation.build_timeline(
        starting_datetime=STARTING_DATETIME,
        temporal_grouping=TEMPORAL_GROUPING,
        interpolation_type=INTERPOLATION_TYPE,
        edge_filter_function=None,
        cutoff=TRAVERSAL_CUTOFF,
        max_calc=MAX_CALCULATIONS,
        graph_traversal=GRAPH_TRAVERSAL,
        traverse_background=False,
    )
    return PrincipalTimeline(
        timeline=timeline,
        activations=_provider_activations(
            calculation.timeline,
            bd,
            activations.loc[activations["region"].eq(str(root["region"]))],
            provider_databases,
            timeline_databases,
            activation_keys={
                (str(region), int(year), str(service_code))
                for region, year, service_code in activations[
                    ["region", "year", "provider_service_code"]
                ].itertuples(index=False, name=None)
            },
            producer_codes={},
            targets={},
        ),
        base_static_score=float(calculation.base_lca.score),
        direct_inventory=pd.DataFrame(columns=("year", "flow_id", "amount")),
    )


@pytest.mark.scientific_integration
def test_compact_root_is_exactly_equivalent_to_bounded_annual_stage_reference(
    tmp_path: Path,
) -> None:
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="compact_foreground",
        biosphere_database="biosphere",
        source_database="provider_2025",
    )
    method = ("bounded equivalence", "climate change")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "co2"): {
                    "name": "carbon dioxide",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                }
            },
            searchable=False,
        )
        bd.Database("provider_interface").write(
            {
                ("provider_interface", "polymer_service"): {
                    "name": "provider service",
                    "reference product": "provider service",
                    "unit": "kilogram",
                    "location": "GLO",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": ("provider_interface", "polymer_service"),
                            "amount": 1.0,
                            "type": "production",
                        },
                        {
                            "input": ("biosphere", "co2"),
                            "amount": 0.0,
                            "type": "biosphere",
                        },
                    ],
                }
            },
            searchable=False,
        )
        for database, emission in (("provider_2025", 10.0), ("provider_2030", 20.0)):
            bd.Database(database).write(
                {
                    (database, "polymer_service"): _provider_dataset(
                        database,
                        "polymer_service",
                        emission,
                    )
                },
                searchable=False,
            )
        for database in ("provider_timeline_2025", "provider_timeline_2030"):
            bd.Database(database).write(
                {
                    (database, "polymer_service"): {
                        "name": "provider service",
                        "reference product": "provider service",
                        "unit": "kilogram",
                        "location": "GLO",
                        "type": "process",
                        "exchanges": [
                            {
                                "input": (database, "polymer_service"),
                                "amount": 1.0,
                                "type": "production",
                            },
                            {
                                "input": ("biosphere", "co2"),
                                "amount": 0.0,
                                "type": "biosphere",
                            },
                        ],
                    }
                },
                searchable=False,
            )
        bd.Database("foreground").write(
            {
                ("foreground", "annual_root"): {
                    "name": "bounded annual-stage reference",
                    "reference product": "regional plastics system",
                    "unit": "unit",
                    "location": "GLO",
                    "type": "process",
                    "region": "EUR",
                    "exchanges": [
                        {
                            "input": ("foreground", "annual_root"),
                            "amount": 1.0,
                            "type": "production",
                        },
                        *[
                            {
                                "input": ("foreground", f"stage_{year}"),
                                "amount": amount,
                                "type": "technosphere",
                                "temporal_distribution": _temporal_distribution(
                                    (year,),
                                    (1.0,),
                                ),
                            }
                            for year, amount in ((2025, 3.0), (2027, 5.0), (2028, 2.0))
                        ],
                    ],
                },
                **{
                    ("foreground", f"stage_{year}"): {
                        "name": f"bounded stage {year}",
                        "reference product": "provider service",
                        "unit": "kilogram",
                        "location": "GLO",
                        "type": "process",
                        "exchanges": [
                            {
                                "input": ("foreground", f"stage_{year}"),
                                "amount": 1.0,
                                "type": "production",
                            },
                            {
                                "input": ("provider_interface", "polymer_service"),
                                "amount": coefficient,
                                "type": "technosphere",
                            },
                        ],
                    }
                    for year, coefficient in ((2025, 2.0), (2027, 3.0), (2028, 4.0))
                },
                ("foreground", "compact_root"): {
                    "name": "compact regional plastics system",
                    "reference product": "regional plastics system",
                    "unit": "unit",
                    "location": "GLO",
                    "type": "process",
                    "region": "EUR",
                    "exchanges": [
                        {
                            "input": ("foreground", "compact_root"),
                            "amount": 1.0,
                            "type": "production",
                        },
                        {
                            "input": ("provider_interface", "polymer_service"),
                            "amount": 1.0,
                            "type": "technosphere",
                            "temporal_distribution": _temporal_distribution(
                                (2025, 2027, 2028),
                                (6.0, 15.0, 8.0),
                            ),
                        },
                    ],
                },
                ("foreground", "compact_root_usa"): {
                    "name": "compact regional plastics system, USA",
                    "reference product": "regional plastics system",
                    "unit": "unit",
                    "location": "GLO",
                    "type": "process",
                    "region": "USA",
                    "exchanges": [
                        {
                            "input": ("foreground", "compact_root_usa"),
                            "amount": 1.0,
                            "type": "production",
                        },
                        {
                            "input": ("provider_interface", "polymer_service"),
                            "amount": 1.0,
                            "type": "technosphere",
                            "temporal_distribution": _temporal_distribution(
                                (2025, 2027, 2028),
                                (4.0, 9.0, 2.0),
                            ),
                        },
                    ],
                },
            },
            searchable=False,
        )
        bd.Database("foreground").process()
        bd.Method(method).register(unit="kilogram carbon dioxide")
        bd.Method(method).write([(("biosphere", "co2"), 1.0)])
        authority = pd.DataFrame(
            [
                {
                    "selection_order": 1,
                    "method_id": "ef31",
                    "method_version": "3.1",
                    "impact_category_id": "climate change",
                    "impact_category_tuple_json": ('["bounded equivalence","climate change"]'),
                    "unit": "kilogram carbon dioxide",
                    "factor_identity": "bounded_factor",
                }
            ]
        )

        full_dates = {
            "foreground": "dynamic",
            "provider_interface": datetime(2025, 1, 1),
            "provider_2025": datetime(2025, 1, 1),
            "provider_2030": datetime(2030, 1, 1),
        }
        compact_dates = database_dates(
            "foreground",
            {
                2020: "provider_interface",
                2025: "provider_timeline_2025",
                2030: "provider_timeline_2030",
            },
        )
        annual_inventory, annual_score = _timex_result(
            bd.get_node(database="foreground", code="annual_root"),
            method,
            full_dates,
        )
        activations = pd.DataFrame(
            [
                {
                    "region": region,
                    "year": year,
                    "provider_service_code": "polymer_service",
                    "demand_type": "processing_conversion",
                    "provider_demand_type": "processing_conversion",
                    "mapping_basis": "kg_processed_plastic",
                    "lca_boundary": "processing_conversion",
                    "burden_attribution_basis": "complete_processed_material_output",
                    "stage_group": "processing_conversion",
                    "cutoff_cycle_role": "material_conversion",
                    "physical_event_basis": "year processed plastic leaves the processing node",
                    "physical_event_year": year,
                    "material": "PE",
                    "constituent": "all",
                    "product": "polymer",
                    "virgin_feedstock_route": "all",
                    "good": "Packaging",
                    "custody_status": "transferred_material",
                    "source_parent_amount_kg": amount,
                    "source_allocation_share": 1.0,
                    "material_cycle_source": "not_applicable",
                    "ultimate_carbon_origin": "fossil",
                    "origin_fraction": 1.0,
                    "branch_lineage_id": "processing_conversion",
                    "basis_flow_names": (
                        "processing => fabrication + processing => intermediate_market"
                    ),
                    "lca_fate": "",
                    "fate_origin": "",
                    "route_id": "processing_conversion",
                    "route_component_id": "processing_conversion",
                    "provider_service_id": "polymer_service",
                    "provider_role": "processing_conversion",
                    "provider_material": "PE",
                    "infiltration_class": "",
                    "material_component_share": 1.0,
                    "infiltration_share": 1.0,
                    "provider_class": "direct_match",
                    "analogue_status": "direct_match",
                    "amount": amount,
                    "mfa_carbon_amount_kg": amount * 0.8,
                    "throughput_multiplier": 1.0,
                    "resolution_status": "direct_match",
                    "material_functional_unit": "",
                    "styrene_component_share": 0.0,
                    "butadiene_analogue_component_share": 0.0,
                    "sbr_share_in_material": 0.0,
                    "sbr_proxy_component_role": "",
                    "whole_tyre_constituents_excluded": False,
                    "no_treatment_sensitivity_executed": False,
                    "declared_provider_source_role": "native_ecoinvent_provider",
                }
                for region, annual_amounts in (
                    ("EUR", ((2025, 6.0), (2027, 15.0), (2028, 8.0))),
                    ("USA", ((2025, 4.0), (2027, 9.0), (2028, 2.0))),
                )
                for year, amount in annual_amounts
            ]
        )
        selected_methods = resolve_lcia_methods_from_authority(
            authority,
            normalized_lcia_method(method),
            bd,
        )
        provider_databases = {
            2020: "provider_2025",
            2025: "provider_2025",
            2030: "provider_2030",
        }
        timeline_databases = {
            2020: "provider_interface",
            2025: "provider_timeline_2025",
            2030: "provider_timeline_2030",
        }
        root_by_region = {
            "EUR": bd.get_node(database="foreground", code="compact_root"),
            "USA": bd.get_node(database="foreground", code="compact_root_usa"),
        }
        serial_timelines = {
            region: _single_root_principal_timeline(
                root,
                method,
                compact_dates,
                bd,
                activations,
                provider_databases,
                timeline_databases,
            )
            for region, root in root_by_region.items()
        }
        full_timeline = serial_timelines["EUR"]
        compact_timeline = compact_principal_timelines({"EUR": full_timeline})["EUR"]
        regional_timelines = build_principal_timelines(
            root_by_region,
            method,
            compact_dates,
            bd,
            activations,
            provider_databases,
            timeline_databases,
            use_global_lci_cache=False,
        )
        for region in ("EUR", "USA"):
            pd.testing.assert_frame_equal(
                regional_timelines[region].activations,
                serial_timelines[region].activations,
            )
            pd.testing.assert_frame_equal(
                regional_timelines[region].timeline.reset_index(drop=True),
                serial_timelines[region].timeline.reset_index(drop=True),
            )
        contribution_state = create_contribution_state(
            bd,
            {"EUR": full_timeline.activations},
            {"EUR": ("foreground", "compact_root")},
            tuple(
                ContributionMethod(
                    method_id=item.method_id,
                    method_version=item.method_version,
                    impact_category_id=item.impact_category_id,
                    method_tuple=item.method_tuple,
                    unit=item.unit,
                )
                for item in selected_methods
            ),
            (2025, 2027, 2028),
        )
        bank = build_provider_inventory_bank(
            bd,
            compact_timeline.activations,
            contribution_state,
        )
        assert bank.unit_responses.empty
        response_state = create_provider_response_state(
            bd,
            tuple(
                ContributionMethod(
                    method_id=item.method_id,
                    method_version=item.method_version,
                    impact_category_id=item.impact_category_id,
                    method_tuple=item.method_tuple,
                    unit=item.unit,
                )
                for item in selected_methods
            ),
        )
        response_bank = build_provider_inventory_bank(
            bd,
            serial_timelines["EUR"].activations,
            response_state,
        )
        assert response_bank.inventories == {}
        assert len(response_bank.unit_responses) == (
            response_bank.provider_count * len(selected_methods)
        )
        assert not response_bank.unit_responses.duplicated(
            [
                "provider_database",
                "provider_code",
                "method_id",
                "method_version",
                "impact_category_id",
            ]
        ).any()
        assert response_bank.unit_responses["provider_unit"].eq("kilogram").all()
        assert response_bank.unit_responses["unit_contribution_norm"].ge(0.0).all()
        assert contribution_state.activations.empty
        assert response_state.activations.empty
        import bw2calc as bc

        if bc.PYPARDISO:
            from pypardiso import ps

            assert ps.factorized_A.shape == (0, 0)
        factors = CharacterizationFactorBank(
            factors=contribution_state.characterization_factors,
        )
        compact_principal = assemble_principal_inventory(compact_timeline, bank)
        compact_inventory = (
            compact_principal.inventory.groupby("year", as_index=False, sort=True)["amount"]
            .sum()
            .reset_index(drop=True)
        )
        characterized = characterize_principal_inventory(
            compact_principal,
            selected_methods,
            factors,
            foreground_scenario="bounded_equivalence",
            background_scenario="bounded_background",
            background_dimensions={
                "ssp_scenario": "SSP2",
                "iam_model": "bounded",
                "iam_scenario": "bounded",
                "gmst": "bounded",
                "gmst_source": "bounded",
                "premise_version": "2.4.6",
                "ecoinvent_version": "3.12",
                "system_model": "Allocation, cut-off by classification",
            },
            region="EUR",
        )

        regional_contribution_state = create_contribution_state(
            bd,
            {region: regional_timelines[region].activations for region in ("EUR", "USA")},
            {
                "EUR": ("foreground", "compact_root"),
                "USA": ("foreground", "compact_root_usa"),
            },
            tuple(
                ContributionMethod(
                    method_id=item.method_id,
                    method_version=item.method_version,
                    impact_category_id=item.impact_category_id,
                    method_tuple=item.method_tuple,
                    unit=item.unit,
                )
                for item in selected_methods
            ),
            (2025, 2027, 2028),
        )
        regional_bank = build_provider_inventory_bank(
            bd,
            pd.concat(
                [item.activations for item in regional_timelines.values()], ignore_index=True
            ),
            regional_contribution_state,
        )
        assert regional_contribution_state.activations.empty
        regional_inventories = {
            region: (
                assemble_principal_inventory(timeline, regional_bank)
                .inventory.groupby("year", as_index=False, sort=True)["amount"]
                .sum()
                .reset_index(drop=True)
            )
            for region, timeline in regional_timelines.items()
        }

    # The 2025/2030 anchors emit 10/20 kg per kg service. Linear interpolation
    # gives 14 in 2027 and 16 in 2028, multiplied by positive regional demand.
    expected = pd.DataFrame(
        {
            "year": (2025, 2027, 2028),
            "amount": (60.0, 210.0, 128.0),
        }
    )
    pd.testing.assert_frame_equal(annual_inventory, expected, check_dtype=False)
    pd.testing.assert_frame_equal(compact_inventory, expected, check_dtype=False)
    pd.testing.assert_frame_equal(
        regional_inventories["EUR"],
        expected,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        regional_inventories["USA"],
        pd.DataFrame(
            {
                "year": (2025, 2027, 2028),
                "amount": (40.0, 126.0, 32.0),
            }
        ),
        check_dtype=False,
    )
    assert annual_score == pytest.approx(398.0, rel=0.0, abs=1e-12)
    assert characterized.scores.iloc[0]["score"] == annual_score
