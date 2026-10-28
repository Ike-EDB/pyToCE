"""Provider recipe compilation and unit-normalization contracts."""

import json
from copy import deepcopy
from typing import cast

import pandas as pd
import pytest
from chemical_recipe_fixtures import _chemical_reject_gases, _chemical_reject_services

from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    recipe_compilation,
)
from pytoce.prepare_backgrounds.preparation.providers.chemical_recycling import (
    _chemical_preparation_details,
)
from pytoce.run_lca.calculation.carbon import (
    carbon_consistency_table,
    direct_treatment_carbon_flows,
)


def test_chemical_input_operations_keep_constituent_eligibility_and_exact_owners() -> None:
    """Separate preparation/hydrothermal inputs from recovered monomer and polymerization."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, object]], catalogue["services"])
    }

    rows = [
        row
        for row in recipe_compilation._route_semantic_rows(
            catalogue,
            services,
            routes=[
                route
                for route in cast(list[dict[str, object]], catalogue["routes"])
                if route["provider_demand_type"]
                in {"chemical_recycling_preparation", "chemical_recycling_conversion"}
            ],
            background_scenario="remind_ssp2_npi",
            year=2050,
        )
        if row["provider_demand_type"]
        in {
            "chemical_recycling_preparation",
            "chemical_recycling_conversion",
        }
    ]
    expected = {
        (material, constituent, origin, stage)
        for material, constituent in (
            ("HDPE", "hdpe"),
            ("LDPE", "ldpe"),
            ("PP", "pp"),
            ("BioPlastics", "bio_pe"),
            ("Fibres", "polypropylene"),
        )
        for origin in ("fossil", "biogenic", "atmospheric_dac")
        for stage in ("chemical_recycling_preparation", "chemical_recycling_conversion")
    }
    assert len(rows) == len(expected)
    assert {
        (
            row["material"],
            row["constituent"],
            row["ultimate_carbon_origin"],
            row["provider_demand_type"],
        )
        for row in rows
    } == expected
    assert {row["modeled_stage"] for row in rows} == {"recycling_operations"}
    assert {row["product"] for row in rows} == {"polymer"}
    assert {row["virgin_feedstock_route"] for row in rows} == {"all"}
    assert {row["chemical_recycling_exact_once_owner"] for row in rows} == {
        "chemical_recycling_input_operations"
    }
    assert {row["chemical_recycling_mapping_basis"] for row in rows} == {
        "kg_tracked_plastic_chemical_recycling_input"
    }
    assert all(
        row["chemical_recycling_resolution_status"]
        == "physical_constituent_feed_and_monomer_yields"
        for row in rows
    )
    assert all(row["project_unit"] == "kilogram" for row in rows)
    rendered = json.dumps(rows, sort_keys=True)
    assert "route_status" not in rendered
    assert "yield_alignment_status" not in rendered
    assert "recovery_profile" not in rendered


def test_chemical_preparation_removes_only_reference_and_processed_feed(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
) -> None:
    context, services = chemical_compilation_case
    dataset = recipe_compilation._service_dataset(
        context,
        services["chemical_recycling_preparation_fossil"],
        services,
        "provider_frontier",
    )

    assert dataset["exchanges"][0]["amount"] == pytest.approx(0.95)
    assert dataset["native_reference_production_amount"] == pytest.approx(1.0)
    assert dataset["tracked_plastic_denominator"] == pytest.approx(0.95)
    assert dataset["native_processed_plastic_output_amount"] == pytest.approx(0.8)
    assert dataset["native_plastic_reject_output_amount"] == pytest.approx(0.15)
    assert dataset["native_processed_feed_coefficient"] == pytest.approx(0.8 / 0.95)
    assert dataset["native_plastic_reject_coefficient"] == pytest.approx(0.15 / 0.95)
    assert dataset["chemical_preparation_normalization_status"] == (
        "tracked_plastic_functional_unit"
    )
    retained = dataset["exchanges"][1:]
    assert {(row["input"], row["amount"]) for row in retained} == {
        (("technology", "reject_market"), -0.15),
        (("technology", "metal"), -0.05),
        (("technology", "utility"), 0.2),
        (("biosphere", "water"), 0.001),
    }
    reject = next(row for row in retained if row["input"] == ("technology", "reject_market"))
    assert reject["uncertainty type"] == 3
    assert reject["loc"] == -2.0
    assert reject["scale"] == 0.2
    assert reject["amount"] / dataset["exchanges"][0]["amount"] == pytest.approx(-0.15 / 0.95)
    owners = json.loads(str(dataset["direct_treatment_carbon_owners_json"]))
    # Signed landfill child duties are .2*.4, then *.3 or *.6;
    # markets have no direct gas rows of their own.
    expected_scales = {
        "open_burning": -0.015 / 0.95,
        "landfill": -0.03 / 0.95,
        "incineration": -0.105 / 0.95,
        "landfill_mixed_plastic_wastewater_treatment": -0.03 / 0.95 * 0.08,
        "landfill_mixed_plastic_sludge_incineration": -0.03 / 0.95 * 0.024,
        "landfill_mixed_plastic_sludge_landfarming": -0.03 / 0.95 * 0.048,
    }
    assert {
        row["owner_code"]: row["owner_scale_per_provider_unit"] for row in owners
    } == pytest.approx(expected_scales)
    assert all(row["owner_database"] == "technology" for row in owners)
    assert all(row["owner_scope"] == "direct_biosphere_only" for row in owners)
    assert all(row["origin_mapping_status"] == "native_identity" for row in owners)
    assert [row["owner_code"] for row in owners] == sorted(expected_scales)
    alignment = next(
        row
        for row in context.semantic_rows
        if row.get("ledger_entry_type") == "chemical_recycling_preparation_alignment"
    )
    assert alignment["native_reference_production_amount"] == pytest.approx(1.0)
    assert alignment["tracked_plastic_denominator"] == pytest.approx(0.95)
    assert alignment["processed_plastic_output_amount"] == pytest.approx(0.8)
    assert alignment["plastic_reject_output_amount"] == pytest.approx(0.15)
    assert alignment["native_processed_feed_coefficient"] == pytest.approx(0.8 / 0.95)
    assert alignment["native_plastic_reject_coefficient"] == pytest.approx(0.15 / 0.95)
    assert alignment["normalization_status"] == "tracked_plastic_functional_unit"
    assert json.loads(str(alignment["excluded_denominator_classes_json"])) == [
        "water",
        "ferrous_metal",
        "non_ferrous_metal",
        "inert_material",
        "other_source_classified_nonplastic_output",
    ]


def test_chemical_preparation_requires_one_output_in_each_tracked_plastic_class(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
) -> None:
    context, services = chemical_compilation_case
    construction = services["chemical_recycling_preparation_fossil"]["construction"]
    missing_reject = {
        **construction,
        "plastic_reject_output_selector": {
            **construction["plastic_reject_output_selector"],
            "reference_product": "missing plastic reject",
        },
    }
    with pytest.raises(ValueError, match="one source-defined plastic reject"):
        _chemical_preparation_details(context, missing_reject)

    sorting = context.technology.activity(construction["selector"], year=context.year)
    sorting.new_exchange(
        input=("technology", "reject_market"),
        amount=-0.01,
        type="technosphere",
    ).save()
    context.native_exchanges.clear()
    with pytest.raises(ValueError, match="one source-defined plastic reject"):
        _chemical_preparation_details(context, construction)


def test_non_fossil_preparation_links_exact_internal_reject_chain(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
) -> None:
    context, services = chemical_compilation_case
    dataset = recipe_compilation._service_dataset(
        context,
        services["chemical_recycling_preparation_non_fossil"],
        services,
        "provider_frontier",
    )
    reject = next(
        row
        for row in dataset["exchanges"]
        if row.get("uncertainty type") == 3 and row.get("loc") == -2.0
    )
    assert reject["input"] == (
        "provider_frontier",
        provider_catalogue.provider_service_code("plastic_waste_market_gb_non_fossil"),
    )
    assert reject["amount"] == -0.15
    assert reject["scale"] == 0.2
    owners = json.loads(str(dataset["direct_treatment_carbon_owners_json"]))
    assert len(owners) == 6
    assert all(row["owner_database"] == "provider_frontier" for row in owners)
    assert all(row["source_role"] == "member_local_frontier_service" for row in owners)
    assert [(row["owner_database"], row["owner_code"]) for row in owners] == sorted(
        (row["owner_database"], row["owner_code"]) for row in owners
    )

    market = recipe_compilation._service_dataset(
        context,
        services["plastic_waste_market_gb_non_fossil"],
        services,
        "provider_frontier",
    )
    assert market["exchanges"][0]["amount"] == -1.0
    child_links = [
        row for row in market["exchanges"] if row["type"] == "technosphere" and row["amount"] < 0
    ]
    assert len(child_links) == 3
    assert all(row["uncertainty type"] == 3 for row in child_links)
    assert all(row["loc"] == -3.0 and row["scale"] == 0.1 for row in child_links)

    treatment = recipe_compilation._service_dataset(
        context,
        services["plastic_waste_incineration_gb_non_fossil"],
        services,
        "provider_frontier",
    )
    assert treatment["exchanges"][0]["amount"] == -1.0
    direct_names = {
        context.biosphere.node(cast(tuple[str, str], row["input"]))["name"]
        for row in treatment["exchanges"][1:]
        if row["type"] == "biosphere"
    }
    assert direct_names == {
        "Carbon dioxide, non-fossil",
        "Methane, non-fossil",
        "Carbon monoxide, non-fossil",
    }


def test_chemical_conversion_scales_native_inventory_and_retains_common_operation(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
) -> None:
    context, services = chemical_compilation_case
    dataset = recipe_compilation._service_dataset(
        context,
        services["chemical_recycling_conversion_fossil"],
        services,
        "provider_frontier",
    )

    assert dataset["exchanges"][0]["amount"] == pytest.approx(1.1875)
    assert dataset["tracked_plastic_denominator"] == pytest.approx(0.95)
    assert dataset["native_processed_feed_coefficient"] == pytest.approx(0.8 / 0.95)
    assert dataset["native_plastic_reject_coefficient"] == pytest.approx(0.15 / 0.95)
    retained = dataset["exchanges"][1:]
    assert {(row["input"], row["amount"]) for row in retained} == {
        (("technology", "utility"), 2.0),
        (("biosphere", "co_generic_f"), 0.1),
    }
    utility = next(row for row in retained if row["input"] == ("technology", "utility"))
    assert utility["uncertainty type"] == 3
    assert utility["loc"] == 0.5
    assert utility["scale"] == 0.1
    owners = json.loads(str(dataset["direct_treatment_carbon_owners_json"]))
    assert owners == [
        {
            "owner_scale_per_provider_unit": 1.0,
            "owner_code": provider_catalogue.provider_service_code(
                "chemical_recycling_conversion_fossil"
            ),
            "owner_database": "provider_frontier",
            "owner_scope": "direct_biosphere_only",
            "origin_mapping_status": "native_identity",
            "source_role": "member_local_frontier_service",
        }
    ]


@pytest.mark.parametrize("origin", ("fossil", "biogenic", "atmospheric_dac"))
def test_chemical_direct_carbon_normalizes_rejects_and_conversion_once(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
    origin: str,
) -> None:
    """Keep nested reject emissions and hydrothermal emissions with their separate stages."""
    import bw2data as bd

    context, services = chemical_compilation_case
    carbon_class = "fossil" if origin == "fossil" else "non_fossil"
    stages = ("chemical_recycling_preparation", "chemical_recycling_conversion")
    service_ids = [f"{stage}_{carbon_class}" for stage in stages]
    if carbon_class == "non_fossil":
        service_ids = list(_chemical_reject_services(context.catalogue, tuple(service_ids)))
    datasets = {
        (
            "provider_frontier",
            provider_catalogue.provider_service_code(service_id),
        ): recipe_compilation._service_dataset(
            context, services[service_id], services, "provider_frontier"
        )
        for service_id in service_ids
    }
    bd.Database("provider_frontier").write(datasets, searchable=False)
    activations = pd.DataFrame.from_records(
        [
            {
                "year": 2030,
                "region": "EUR",
                "life_cycle_stage": stage,
                "material": "HDPE",
                "constituent": "hdpe",
                "product": "polymer",
                "virgin_feedstock_route": "all",
                "good": "all",
                "material_cycle_source": "chemically_recycled_feedstock",
                "ultimate_carbon_origin": origin,
                "lca_fate": "chemical_recycling",
                "fate_origin": "remind_mfa",
                "provider_database": "provider_frontier",
                "provider_code": provider_catalogue.provider_service_code(
                    f"{stage}_{carbon_class}"
                ),
                "provider_anchor_year": 2030,
                "amount": 0.95,
                "mfa_carbon_amount_kg": (0.95 if stage == stages[0] else 0.8) * 6.0 / 7.0,
            }
            for stage in stages
        ]
    )
    result = direct_treatment_carbon_flows(bd, activations)
    preparation = result.loc[result["life_cycle_stage"] == stages[0]]
    conversion = result.loc[result["life_cycle_stage"] == stages[1]]
    assert len(preparation) == 17
    principal, _ = _chemical_reject_gases((0.1, 0.2, 0.7), year=2030)
    assert preparation.groupby("carbon_species")["flow_amount"].sum().to_dict() == pytest.approx(
        {
            name: 0.15 * sum(principal[(species, area)] for area in ("high", "urban"))
            for species, name in (
                ("co2", "carbon dioxide"),
                ("ch4", "methane"),
                ("co", "carbon monoxide"),
            )
        }
    )
    assert set(preparation["flow_compartment_json"]) == {
        '["air","non-urban air or from high stacks"]',
        '["air","urban air close to ground"]',
    }
    assert len(conversion) == 1
    assert conversion["flow_amount"].tolist() == pytest.approx([0.08])
    assert conversion["flow_compartment_json"].tolist() == ['["air"]']
    assert set(result["mfa_carbon_origin"]) == {origin}
    assert set(result["material_cycle_source"]) == {"chemically_recycled_feedstock"}
    assert set(result["flow_carbon_origin"]) == {carbon_class}
    assert set(result["origin_mapping_status"]) == {
        "native_identity" if origin == "fossil" else "exact_non_fossil_counterpart"
    }
    consistency = carbon_consistency_table(bd, activations).set_index("life_cycle_stage")
    assert consistency["mfa_treatment_input_carbon_kg"].to_dict() == pytest.approx(
        {stages[0]: 0.95 * 6 / 7, stages[1]: 0.8 * 6 / 7}
    )
    assert consistency["mfa_carbon_boundary"].to_dict() == {
        stages[0]: "treatment_gate_constituent_carbon",
        stages[1]: "processed_constituent_carbon_after_preparation",
    }


@pytest.mark.parametrize("original_properties", (True, False), ids=("original", "premise"))
def test_non_fossil_conversion_changes_only_direct_carbon_identity(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
    original_properties: bool,
) -> None:
    """Map generated CO metadata while preserving native records and distributions."""
    context, services = chemical_compilation_case
    native_records = {}
    for index in range(4):
        native = context.ordinary.node(("technology", f"htt_{index}"))
        (carbon,) = native.biosphere()
        fields = {
            "name": "Carbon monoxide, fossil",
            "unit": "kilogram",
            "categories": ("air",),
            "uncertainty type": 3,
            "loc": float(carbon["amount"]),
            "scale": 0.01,
            "comment": "Invented recycled process-gas emission",
        }
        if original_properties:
            fields.update(
                flow="co_generic_f",
                properties={
                    "carbon content, fossil": {"amount": 0.43, "unit": "dimensionless"},
                    "carbon content, non-fossil": {"amount": 0.0, "unit": "dimensionless"},
                    "carbon allocation": {"amount": 0.43, "unit": "kilogram"},
                    "dry mass": {"amount": 1.0, "unit": "kilogram"},
                },
            )
        for field, value in fields.items():
            carbon[field] = value
        carbon.save()
        native_records[native.key] = [dict(ex) for ex in native.exchanges()]
    fossil = recipe_compilation._service_dataset(
        context,
        services["chemical_recycling_conversion_fossil"],
        services,
        "provider_frontier",
    )
    non_fossil = recipe_compilation._service_dataset(
        context,
        services["chemical_recycling_conversion_non_fossil"],
        services,
        "provider_frontier",
    )
    assert fossil["exchanges"][0]["amount"] == non_fossil["exchanges"][0]["amount"]
    assert fossil["exchanges"][0]["type"] == non_fossil["exchanges"][0]["type"]
    fossil_utility = next(row for row in fossil["exchanges"] if row["type"] == "technosphere")
    non_fossil_utility = next(
        row for row in non_fossil["exchanges"] if row["type"] == "technosphere"
    )
    assert fossil_utility == non_fossil_utility
    fossil_carbon = next(row for row in fossil["exchanges"] if row["type"] == "biosphere")
    non_fossil_carbon = next(row for row in non_fossil["exchanges"] if row["type"] == "biosphere")
    assert fossil_carbon["amount"] == non_fossil_carbon["amount"]
    assert fossil_carbon["input"] == ("biosphere", "co_generic_f")
    assert non_fossil_carbon["input"] == ("biosphere", "co_generic_nf")
    expected = deepcopy(fossil_carbon)
    expected.update(input=("biosphere", "co_generic_nf"), name="Carbon monoxide, non-fossil")
    if original_properties:
        expected["flow"] = "co_generic_nf"
        expected["properties"]["carbon content, fossil"]["amount"] = 0.0
        expected["properties"]["carbon content, non-fossil"]["amount"] = 0.43
    else:
        assert "properties" not in non_fossil_carbon
        assert "flow" not in non_fossil_carbon
    assert non_fossil_carbon == expected
    assert {
        key: [dict(ex) for ex in context.ordinary.node(key).exchanges()] for key in native_records
    } == native_records
