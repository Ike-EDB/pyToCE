"""Dated provider-frontier database contracts."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    frontier_database as provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    native_resolution,
)


def test_treatment_carbon_applicability_retains_one_origin_lineage() -> None:
    rows = [
        {
            "provider_service_id": "shared_non_fossil_treatment",
            "ultimate_carbon_origin": "biogenic",
            "ledger_entry_type": "treatment_carbon_mapping",
        }
    ]
    contexts = {
        "shared_non_fossil_treatment": [
            {
                "route_id": f"route_{origin}",
                "route_component_id": f"component_{origin}",
                "provider_demand_type": "incineration_treatment",
                "ultimate_carbon_origin": origin,
                "material": "HDPE",
                "constituent": "hdpe",
                "product": "polymer",
                "virgin_feedstock_route": "all",
                "good": "all",
                "region": "all",
                "valid_from_year": 2020,
                "valid_to_year": 2100,
            }
            for origin in ("biogenic", "atmospheric_dac")
        ]
    }

    attached = provider_frontier._attach_applicability(rows, contexts)

    assert json.loads(attached[0]["applicability_json"]) == [
        {
            "good": "all",
            "h12_applicability": "all",
            "infiltration_class": "",
            "lca_fate": "",
            "material": "HDPE",
            "constituent": "hdpe",
            "product": "polymer",
            "virgin_feedstock_route": "all",
            "provider_demand_type": "incineration_treatment",
            "route_component_id": "component_biogenic",
            "route_id": "route_biogenic",
            "ultimate_carbon_origin": "biogenic",
            "valid_from_year": 2020,
            "valid_to_year": 2100,
        }
    ]


def _selector(
    name: str,
    reference_product: str,
    unit: str,
    location: str,
) -> dict[str, object]:
    return {
        "name": name,
        "reference_product": reference_product,
        "unit": unit,
        "exact_location": location,
    }


def _catalogue() -> dict[str, object]:
    precursor = _selector(
        "conventional precursor production",
        "conventional precursor",
        "kilogram",
        "RER",
    )
    return {
        "schema_version": "provider_catalogue_test_v1",
        "source_authorities": [
            {
                "source_id": "bounded_source",
                "role": "bounded exact provider-equivalence fixture",
            },
            {
                "source_id": "ecoinvent_3_12_cutoff",
                "role": "bounded native ecoinvent activity authority",
            },
            {
                "source_id": "premise_2_4_6",
                "role": "bounded native premise activity authority",
            },
        ],
        "method_authority_registry": {
            "prefix_rules": [],
            "exact_source_bindings": {
                "bounded_operation_only": ["bounded_source"],
                "bounded_premise_activity": ["bounded_source"],
                "bounded_treatment": ["bounded_source"],
                "bounded_nested_treatment": ["bounded_source"],
                "bounded_project_recipe": ["bounded_source"],
            },
            "project_recipe_defaults": {
                "technology_boundary": "bounded fixture",
                "allocation_boundary": "bounded fixture",
                "geography_year_boundary": "bounded fixture",
                "limitation": "bounded fixture",
            },
        },
        "semantic_contract": {
            "explicit_cross_material_route_ids": ["alternative_operation"],
            "service_normalization_route_ids": [],
        },
        "services": [
            {
                "provider_service_id": "alternative_operation",
                "name": "alternative operation",
                "reference_product": "alternative product",
                "unit": "kilogram",
                "location": "GLO",
                "stage_owner": "virgin_resin_production",
                "foreground_service": True,
                "citation": "bounded_operation_only",
                "construction": {
                    "kind": "operation_only",
                    "base": {
                        "source_role": "native_ecoinvent_provider",
                        "selector": _selector(
                            "polymer operation",
                            "polymer",
                            "kilogram",
                            "RER",
                        ),
                    },
                    "replace_technosphere": [
                        {
                            "replacement_id": "incorporated_precursor",
                            "selector": precursor,
                        }
                    ],
                    "replacements": [
                        {
                            "replacement_id": "incorporated_precursor",
                            "service_id": "alternative_precursor",
                        }
                    ],
                    "replace_biosphere": [
                        {
                            "selector": {
                                "name": "Carbon dioxide, fossil",
                                "unit": "kilogram",
                                "categories": ["air"],
                            },
                            "replacement_selector": {
                                "name": "Carbon dioxide, non-fossil",
                                "unit": "kilogram",
                                "categories": ["air"],
                            },
                            "reason": (
                                "the source operation identifies this direct carbon "
                                "dioxide as converted incorporated-feedstock carbon"
                            ),
                        }
                    ],
                },
            },
            {
                "provider_service_id": "alternative_precursor",
                "frontier_service": True,
                "name": "alternative precursor",
                "reference_product": "alternative precursor",
                "unit": "kilogram",
                "location": "GLO",
                "stage_owner": "virgin_resin_production",
                "foreground_service": False,
                "citation": "bounded_premise_activity",
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": _selector(
                        "alternative precursor production",
                        "alternative precursor",
                        "kilogram",
                        "EUR",
                    ),
                },
            },
            {
                "provider_service_id": "incineration_treatment",
                "name": "incineration treatment",
                "reference_product": "waste plastic treatment",
                "unit": "kilogram",
                "location": "GLO",
                "stage_owner": "incineration_treatment",
                "foreground_service": True,
                "citation": "bounded_treatment",
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "amount": -1.0,
                    "selector": _selector(
                        "treatment of waste plastic",
                        "waste plastic",
                        "kilogram",
                        "RER",
                    ),
                },
            },
            {
                "provider_service_id": "nested_treatment_recipe",
                "name": "nested treatment recipe",
                "reference_product": "nested treatment recipe",
                "unit": "kilogram",
                "location": "GLO",
                "stage_owner": "virgin_resin_production",
                "foreground_service": True,
                "citation": "bounded_nested_treatment",
                "construction": {
                    "kind": "project_recipe",
                    "components": [
                        {
                            "component_id": "process_waste_treatment",
                            "source_role": "native_ecoinvent_provider",
                            "selector": _selector(
                                "treatment of waste plastic",
                                "waste plastic",
                                "kilogram",
                                "RER",
                            ),
                            "amount": -1.0,
                        }
                    ],
                },
            },
            {
                "provider_service_id": "unit_recipe",
                "name": "unit recipe",
                "reference_product": "unit product",
                "unit": "kilogram",
                "location": "GLO",
                "stage_owner": "virgin_resin_production",
                "foreground_service": True,
                "citation": "bounded_project_recipe",
                "construction": {
                    "kind": "project_recipe",
                    "components": [
                        {
                            "component_id": "electricity",
                            "source_role": "native_ecoinvent_provider",
                            "selector": _selector(
                                "electricity supply",
                                "electricity",
                                "kilowatt hour",
                                "RER",
                            ),
                            "amount": 2.0,
                        }
                    ],
                    "biosphere_terms": [
                        {
                            "selector": {
                                "name": "Carbon dioxide, fossil",
                                "unit": "kilogram",
                                "categories": ["air"],
                            },
                            "amount": 0.1,
                        },
                    ],
                },
            },
        ],
        "routes": [
            {
                "route_id": "alternative_operation",
                "route_component_id": "alternative_operation",
                "provider_demand_type": "virgin_resin_production",
                "region": "all",
                "material": "PVC",
                "constituent": "pvc",
                "product": "polymer",
                "good": "all",
                "mapping_basis": "kg_virgin_resin_output",
                "valid_from_year": 2020,
                "valid_to_year": 2100,
                "component_role": "polymer_production",
                "component_material": "PVC",
                "virgin_feedstock_route": "atmospheric_dac",
                "ultimate_carbon_origin": "all",
                "share": 1.0,
                "throughput_multiplier": 1.0,
                "provider_service_id": "alternative_operation",
            }
        ],
    }


def _native_process(
    database: str,
    code: str,
    name: str,
    product: str,
    unit: str,
    location: str,
    exchanges: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "reference product": product,
        "unit": unit,
        "location": location,
        "type": "process",
        "exchanges": [
            {
                "input": (database, code),
                "amount": 1.0,
                "type": "production",
            },
            *(exchanges or []),
        ],
    }


def _write_bounded_provider_services(
    bd: Any,
    catalogue: dict[str, object],
) -> tuple[
    str,
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    """Build bounded service datasets through the provider frontier owner."""

    provider = provider_catalogue.PreparedProviderAuthority(
        identity="provider::bounded",
        interface_database=provider_catalogue.PROVIDER_INTERFACE_DATABASE,
        catalogue=catalogue,
        routes=provider_catalogue._route_table(catalogue),
        method_authorities=provider_catalogue._method_authority_rows(catalogue),
    )
    frontier = provider_frontier.prepare_provider_frontier(
        bd,
        background_scenario="remind_ssp2_npi",
        year=2020,
        ordinary_database="ordinary",
        technology_database="technology",
        material_recipe=physical_material_recipe(),
        provider=provider,
        resolver_cache=native_resolution._NativeResolverCache("ordinary", "biosphere"),
    )
    return frontier.database_name, frontier.semantic_rows, frontier.technology_rows


@pytest.mark.parametrize("original_properties", (True, False), ids=("original", "premise"))
def test_source_neutral_recipe_replaces_only_the_declared_native_input(
    tmp_path: Path,
    original_properties: bool,
) -> None:
    """Preserve amounts and chemical identity while replacing producer and gas metadata."""
    catalogue = _catalogue()
    operation_service, precursor_service = catalogue["services"][:2]
    operation_service["construction"]["replace_technosphere"][0]["selector"][
        "reference_product"
    ] = "benzene"
    precursor_service["reference_product"] = "benzene"
    precursor_service["construction"]["selector"]["reference_product"] = "benzene"
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="provider_authority",
        biosphere_database="biosphere",
        source_database="ordinary",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "co2"): {
                    "name": "Carbon dioxide, fossil",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
                ("biosphere", "co2_non_fossil"): {
                    "name": "Carbon dioxide, non-fossil",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
            },
            searchable=False,
        )
        bd.Database("ordinary").write(
            {
                ("ordinary", "alternative_precursor"): _native_process(
                    "ordinary",
                    "alternative_precursor",
                    "alternative precursor production",
                    "benzene",
                    "kilogram",
                    "EUR",
                ),
                ("ordinary", "precursor"): _native_process(
                    "ordinary",
                    "precursor",
                    "conventional precursor production",
                    "benzene",
                    "kilogram",
                    "RER",
                ),
                ("ordinary", "electricity"): _native_process(
                    "ordinary",
                    "electricity",
                    "electricity supply",
                    "electricity",
                    "kilowatt hour",
                    "RER",
                ),
                ("ordinary", "operation"): _native_process(
                    "ordinary",
                    "operation",
                    "polymer operation",
                    "polymer",
                    "kilogram",
                    "RER",
                    [
                        {
                            "input": ("ordinary", "precursor"),
                            "amount": 2.0,
                            "type": "technosphere",
                            "name": "benzene",
                            "reference product": "benzene",
                            "unit": "kilogram",
                            "formula": "2 * precursor_requirement",
                            **(
                                {
                                    "activity": "invented source producer",
                                    "flow": "invented benzene flow",
                                    "CAS number": "71-43-2",
                                    "chemical formula": "C6H6",
                                    "properties": {
                                        "carbon content": {"amount": 0.923},
                                        "carbon content, fossil": {"amount": 0.923},
                                        "carbon content, non-fossil": {"amount": 0.0},
                                        "dry mass": {"amount": 1.0},
                                    },
                                }
                                if original_properties
                                else {}
                            ),
                        },
                        {
                            "input": ("ordinary", "electricity"),
                            "amount": 3.0,
                            "type": "technosphere",
                        },
                        *[
                            {
                                "input": ("biosphere", "co2"),
                                "amount": amount,
                                "type": "biosphere",
                                "uncertainty type": 3,
                                "loc": amount,
                                "scale": spread,
                                "name": "Carbon dioxide, fossil",
                                "unit": "kilogram",
                                "categories": ("air",),
                                "chemical formula": "CO2",
                                **(
                                    {
                                        "flow": "co2",
                                        "properties": {
                                            "carbon content, fossil": {
                                                "amount": 0.27,
                                                "unit": "dimensionless",
                                            },
                                            "carbon content, non-fossil": {
                                                "amount": 0.0,
                                                "unit": "dimensionless",
                                            },
                                            "dry mass": {"amount": 1.0, "unit": "kilogram"},
                                            "carbon allocation": {
                                                "amount": 0.27,
                                                "unit": "kilogram",
                                            },
                                        },
                                    }
                                    if original_properties
                                    else {}
                                ),
                            }
                            for amount, spread in ((0.3, 0.1), (0.2, 0.02), (0.0, 0.001))
                        ],
                    ],
                ),
                ("ordinary", "treatment"): {
                    "name": "treatment of waste plastic",
                    "reference product": "waste plastic",
                    "unit": "kilogram",
                    "location": "RER",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": ("ordinary", "treatment"),
                            "amount": -1.0,
                            "type": "production",
                        },
                        {
                            "input": ("biosphere", "co2"),
                            "amount": 2.0,
                            "type": "biosphere",
                        },
                    ],
                },
            },
            searchable=False,
        )
        bd.Database("technology").write({}, searchable=False)

        native_operation = bd.get_node(database="ordinary", code="operation")
        native_records = [dict(exchange) for exchange in native_operation.exchanges()]
        native_direct = [exchange for exchange in native_records if exchange["type"] == "biosphere"]
        frontier_database, semantic_rows, technology_rows = _write_bounded_provider_services(
            bd,
            catalogue,
        )

        operation = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("alternative_operation"),
        )
        source_feed = next(
            row for row in native_records if row["input"] == ("ordinary", "precursor")
        )
        generated_feed = next(
            row for row in operation.technosphere() if row.input["reference product"] == "benzene"
        )
        expected_feed = deepcopy(source_feed)
        expected_feed.update(input=generated_feed.input.key, output=operation.key)
        expected_feed.pop("activity", None)
        if original_properties:
            expected_feed["properties"].pop("carbon content, fossil")
            expected_feed["properties"].pop("carbon content, non-fossil")
        assert dict(generated_feed) == expected_feed
        exchanges = {}
        for exchange in operation.exchanges():
            if exchange["type"] == "production":
                continue
            key = (
                exchange["type"],
                exchange.input.get("name", ""),
                exchange.input.get("reference product", ""),
                exchange.input.key[0],
            )
            exchanges[key] = exchanges.get(key, 0.0) + float(exchange["amount"])
        mapped_direct = [
            {key: value for key, value in dict(exchange).items() if key != "output"}
            for exchange in operation.biosphere()
        ]
        expected_direct = deepcopy(native_direct)
        for exchange in expected_direct:
            exchange.pop("output")
            exchange.update(
                input=("biosphere", "co2_non_fossil"),
                name="Carbon dioxide, non-fossil",
                unit="kilogram",
            )
            if original_properties:
                exchange["flow"] = "co2_non_fossil"
                exchange["properties"]["carbon content, fossil"]["amount"] = 0.0
                exchange["properties"]["carbon content, non-fossil"]["amount"] = 0.27
        assert sorted(mapped_direct, key=repr) == sorted(expected_direct, key=repr)
        assert exchanges == {
            (
                "biosphere",
                "Carbon dioxide, non-fossil",
                "",
                "biosphere",
            ): 0.5,
            (
                "technosphere",
                "electricity supply",
                "electricity",
                "ordinary",
            ): 3.0,
            (
                "technosphere",
                "alternative precursor",
                "benzene",
                frontier_database,
            ): 2.0,
        }
        origin_replacements = [
            row
            for row in semantic_rows
            if row.get("ledger_entry_type") == "native_biosphere_replacement"
        ]
        assert len(origin_replacements) == 3
        assert {row["source_flow_name"] for row in origin_replacements} == {
            "Carbon dioxide, fossil"
        }
        assert {row["replacement_flow_name"] for row in origin_replacements} == {
            "Carbon dioxide, non-fossil"
        }
        assert sorted(row["source_amount"] for row in origin_replacements) == [0.0, 0.2, 0.3]
        assert [dict(exchange) for exchange in native_operation.exchanges()] == native_records

        missing_feed_catalogue = json.loads(json.dumps(catalogue))
        missing_feed_catalogue["services"][0]["construction"]["replace_technosphere"][0][
            "selector"
        ]["name"] = "absent precursor"
        with pytest.raises(ValueError, match="lacks declared replacement inputs"):
            _write_bounded_provider_services(bd, missing_feed_catalogue)

        mismatched_catalogue = json.loads(json.dumps(catalogue))
        mismatched_catalogue["services"][0]["construction"]["replace_biosphere"][0]["selector"][
            "name"
        ] = "Carbon dioxide, missing"
        with pytest.raises(ValueError, match="must match at least one"):
            _write_bounded_provider_services(bd, mismatched_catalogue)

        written = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("unit_recipe"),
        )
        written_exchanges = {
            (exchange["type"], exchange.input.key): float(exchange["amount"])
            for exchange in written.exchanges()
            if exchange["type"] != "production"
        }
        assert written_exchanges == {
            ("biosphere", ("biosphere", "co2")): 0.1,
            ("technosphere", ("ordinary", "electricity")): 2.0,
        }
        treatment = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("incineration_treatment"),
        )
        treatment_exchanges = {
            (exchange["type"], exchange.input.key): float(exchange["amount"])
            for exchange in treatment.exchanges()
            if exchange["type"] != "production"
        }
        assert treatment_exchanges == {
            ("technosphere", ("ordinary", "treatment")): -1.0,
        }
        direct_treatment = bd.get_node(
            database="ordinary",
            code="treatment",
        )
        project_lca = LCA({treatment: 1.0})
        project_lca.lci()
        direct_lca = LCA({direct_treatment: -1.0})
        direct_lca.lci()
        project_nonzero_biosphere = [
            float(value) for value in project_lca.inventory.sum(axis=1).flat if value != 0.0
        ]
        direct_nonzero_biosphere = [
            float(value) for value in direct_lca.inventory.sum(axis=1).flat if value != 0.0
        ]
        assert project_nonzero_biosphere == direct_nonzero_biosphere
        assert float(project_lca.inventory.sum()) == pytest.approx(2.0)
        nested_treatment = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("nested_treatment_recipe"),
        )
        nested_lca = LCA({nested_treatment: 1.0})
        nested_lca.lci()
        nested_nonzero_biosphere = [
            float(value) for value in nested_lca.inventory.sum(axis=1).flat if value != 0.0
        ]
        assert nested_nonzero_biosphere == direct_nonzero_biosphere
        removed = next(row for row in semantic_rows if row.get("removed_amount") == 2.0)
        assert removed["semantic_class"] == "operation_only_extraction"
        assert provider_catalogue._route_table(catalogue).iloc[0]["provider_service_code"] == (
            provider_catalogue.provider_service_code("alternative_operation")
        )
        route_row = next(
            row for row in semantic_rows if row.get("ledger_entry_type") == "route_mapping"
        )
        assert route_row["semantic_class"] == "explicit_cross_material_analogue"
        semantic_classes = {
            row["semantic_class"]
            for row in (*semantic_rows, *technology_rows)
            if "semantic_class" in row
        }
        assert semantic_classes <= {
            "unchanged_same_product",
            "premise_transformed_same_product",
            "operation_only_extraction",
            "source_backed_portfolio",
            "service_normalization",
            "explicit_cross_material_analogue",
        }
