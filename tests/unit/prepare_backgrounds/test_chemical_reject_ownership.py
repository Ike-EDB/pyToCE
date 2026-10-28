"""Chemical reject diagnostics follow the inventory selected at each anchor."""

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from chemical_recipe_fixtures import _chemical_reject_gases, _chemical_reject_services
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.run_lca.calculation.carbon import (
    carbon_consistency_table,
    direct_treatment_carbon_flows,
)


def _ordinary_chemical_source(
    bd: Any,
    context: _RecipeContext,
) -> None:
    """Write ordinary chemical operations with distinct utilities and reject shares."""
    data = {}
    treatments = (
        ("open_burning", "high", 1.0),
        ("incineration", "urban", 3.0),
    )
    for code, compartment, factor in treatments:
        native = context.technology.node(("technology", code))
        data[("ordinary", code)] = _process(
            "ordinary",
            code,
            native["name"],
            native["reference product"],
            location=native["location"],
            production_amount=-1.0,
            exchanges=[
                {
                    "input": ("biosphere", f"{species}_{compartment}_f"),
                    "amount": factor * amount,
                    "type": "biosphere",
                }
                for species, amount in (("co2", 0.5), ("ch4", 0.02), ("co", 0.01))
            ],
        )
    for code in (
        "utility",
        "metal",
        "processed_output",
        "processed_market",
        "sorting",
        "htt_0",
        "htt_1",
        "htt_2",
        "htt_3",
        *(
            node["code"]
            for node in bd.Database("technology")
            if node["code"].startswith("landfill")
        ),
    ):
        native = context.technology.node(("technology", code))
        exchanges = []
        for exchange in native.exchanges():
            if exchange["type"] == "production":
                continue
            copied = exchange.as_dict()
            copied.pop("output")
            if copied["input"][0] == "technology":
                copied["input"] = ("ordinary", copied["input"][1])
            if copied["input"] == ("ordinary", "utility"):
                copied["amount"] *= 1.5
            if code == "landfill" and copied["type"] == "biosphere":
                copied["amount"] *= 2.0
            exchanges.append(copied)
        data[("ordinary", code)] = {
            **_process(
                "ordinary",
                code,
                native["name"],
                native["reference product"],
                location=native["location"],
                production_amount=next(iter(native.production()))["amount"],
                exchanges=exchanges,
            ),
            "unit": native["unit"],
        }
    data[("ordinary", "reject_market")] = _process(
        "ordinary",
        "reject_market",
        "market for waste plastic, mixture",
        "waste plastic, mixture",
        location="GB",
        production_amount=-2.0,
        exchanges=[
            {"input": ("ordinary", code), "amount": amount, "type": "technosphere"}
            for code, amount in (("open_burning", -0.6), ("landfill", -0.4), ("incineration", -1.0))
        ],
    )
    bd.Database("ordinary").write(data, searchable=False)


def _preparation_activation(year: int, origin: str, key: tuple[str, str]) -> pd.DataFrame:
    """Demand 0.95 kg tracked plastic with its chemical-preparation coordinates."""
    return pd.DataFrame(
        [
            {
                "year": year,
                "region": "EUR",
                "life_cycle_stage": "chemical_recycling_preparation",
                "material": "HDPE",
                "constituent": "hdpe",
                "product": "polymer",
                "virgin_feedstock_route": "all",
                "good": "all",
                "material_cycle_source": "chemically_recycled_feedstock",
                "ultimate_carbon_origin": origin,
                "lca_fate": "chemical_recycling",
                "fate_origin": "remind_mfa",
                "provider_database": key[0],
                "provider_code": key[1],
                "provider_anchor_year": year,
                "amount": 0.95,
                "mfa_carbon_amount_kg": 0.95 * 6 / 7,
            }
        ]
    )


@pytest.mark.parametrize("year", (2020, 2030))
@pytest.mark.parametrize("origin", ("fossil", "biogenic", "atmospheric_dac"))
def test_preparation_reject_diagnostics_follow_the_linked_inventory(
    chemical_compilation_case: tuple[_RecipeContext, dict[str, dict[str, object]]],
    year: int,
    origin: str,
) -> None:
    """Agree with actual LCI through market and treatment production denominators."""
    import bw2data as bd

    context, services = chemical_compilation_case
    _ordinary_chemical_source(bd, context)
    selected = _chemical_reject_services(
        context.catalogue, tuple(key for key in services if key.startswith("chemical_recycling_"))
    )
    routes = [
        route for route in context.catalogue["routes"] if route["provider_service_id"] in selected
    ]
    catalogue = {**context.catalogue, "services": list(selected.values()), "routes": routes}
    provider = PreparedProviderAuthority(
        identity="chemical_reject_fixture",
        interface_database="provider_interface",
        catalogue=catalogue,
        routes=pd.DataFrame(routes),
        method_authorities=_method_authority_rows(catalogue),
    )
    ordinary = "ordinary" if year == 2020 else "technology"
    frontier = prepare_provider_frontier(
        bd,
        background_scenario=context.background_scenario,
        year=year,
        ordinary_database=ordinary,
        technology_database="technology",
        material_recipe=physical_material_recipe(),
        provider=provider,
        resolver_cache=_NativeResolverCache(ordinary, "biosphere"),
    )
    mappings = [
        row
        for row in frontier.semantic_rows
        if row.get("ledger_entry_type") == "treatment_carbon_mapping"
        and row["provider_service_id"] == "chemical_recycling_conversion_non_fossil"
    ]
    assert {row["ultimate_carbon_origin"] for row in mappings} == {"biogenic", "atmospheric_dac"}
    assert all(
        {item["ultimate_carbon_origin"] for item in json.loads(row["applicability_json"])}
        == {row["ultimate_carbon_origin"]}
        for row in mappings
    )
    market_trace = next(
        row
        for row in frontier.semantic_rows
        if row.get("ledger_entry_type") == "service_construction"
        and row["provider_service_id"] == "plastic_waste_market_gb_non_fossil"
    )
    assert market_trace["semantic_class"] == "operation_only_extraction"
    carbon_class = "fossil" if origin == "fossil" else "non_fossil"
    service_id = f"chemical_recycling_preparation_{carbon_class}"
    activity = bd.get_activity((frontier.database_name, provider_service_code(service_id)))
    reject = next(exchange for exchange in activity.technosphere() if exchange["amount"] == -0.15)
    expected_market = (
        (ordinary, "reject_market")
        if origin == "fossil"
        else (
            frontier.database_name,
            provider_service_code("plastic_waste_market_gb_non_fossil"),
        )
    )
    assert reject.input.key == expected_market
    source_row = next(
        row
        for row in frontier.technology_rows
        if row["provider_service_id"] == service_id
        and row["component_id"] == "source_defined_native_operation"
    )
    assert activity["provider_source_database"] == ordinary
    assert source_row["native_database_name"] == ordinary
    assert source_row["source_class"] == "native_ecoinvent_provider"
    assert json.loads(source_row["project_terms_json"]) == [
        {
            "native_input_database": ordinary,
            "native_input_code": "reject_market",
            "project_input_database": expected_market[0],
            "project_input_code": expected_market[1],
            "amount": -0.15,
            "exchange_type": "technosphere",
            "metadata_preserved": True,
        }
    ]
    assert reject["uncertainty type"] == 3
    assert reject["loc"] == -2.0
    assert reject["scale"] == 0.2
    conversion_id = f"chemical_recycling_conversion_{carbon_class}"
    conversion = bd.get_activity((frontier.database_name, provider_service_code(conversion_id)))
    conversion_source = next(
        row
        for row in frontier.technology_rows
        if row["provider_service_id"] == conversion_id
        and row["component_id"] == "source_defined_native_operation"
    )
    assert conversion_source["native_database_name"] == ordinary
    assert conversion_source["source_class"] == "native_ecoinvent_provider"
    conversion_utility = next(iter(conversion.technosphere()))
    assert conversion_utility.input.key == (ordinary, "utility")
    assert conversion_utility["amount"] == (3.0 if year == 2020 else 2.0)
    assert not any(
        row.get("ledger_entry_type") == "early_premise_provider_technology_clone"
        for row in frontier.technology_rows
    )
    owners = json.loads(activity["direct_treatment_carbon_owners_json"])
    assert {row["owner_database"] for row in owners} == {
        ordinary if origin == "fossil" else frontier.database_name
    }
    expected_role = (
        "member_local_frontier_service" if origin != "fossil" else "native_ecoinvent_provider"
    )
    assert {row["source_role"] for row in owners} == {expected_role}
    activations = _preparation_activation(year, origin, activity.key)
    detailed = direct_treatment_carbon_flows(bd, activations)
    # The ordinary market allocates -0.6/-0.4/-1 against -2 production;
    # the future market uses -.1/-.2/-.7 against -1. Landfill includes its
    # leachate and sludge children, excluding auxiliary carbon from diagnostics.
    weights = (0.3, 0.2, 0.5) if year == 2020 else (0.1, 0.2, 0.7)
    principal, auxiliary = _chemical_reject_gases(weights, year=year)
    expected = {
        name: 0.15 * sum(principal[(species, area)] for area in ("high", "urban"))
        for species, name in (
            ("co2", "carbon dioxide"),
            ("ch4", "methane"),
            ("co", "carbon monoxide"),
        )
    }
    assert detailed.groupby("carbon_species")["flow_amount"].sum().to_dict() == pytest.approx(
        expected
    )
    consistency = carbon_consistency_table(bd, activations).iloc[0]
    for column, species, fraction in (
        ("native_direct_co2_carbon_kg", "carbon dioxide", 12 / 44),
        ("native_direct_ch4_carbon_kg", "methane", 12 / 16),
        ("native_direct_co_carbon_kg", "carbon monoxide", 12 / 28),
    ):
        assert consistency[column] == pytest.approx(expected[species] * fraction)
    calculation = LCA({activity: 0.95})
    calculation.lci()
    inventory = calculation.inventory.sum(axis=1).A1
    # At most ten rounded factors along these signed production-normalized
    # paths: eight eps32 covers 10u/(1-10u) and the binary64 arithmetic.
    for (species, area), principal_amount in principal.items():
        expected_by_origin = {
            "f": auxiliary[(species, area)] + (principal_amount if origin == "fossil" else 0),
            "nf": 0 if origin == "fossil" else principal_amount,
        }
        for suffix, amount in expected_by_origin.items():
            flow = bd.get_node(database="biosphere", code=f"{species}_{area}_{suffix}")
            index = calculation.dicts.biosphere.get(flow.id)
            observed = 0.0 if index is None else inventory[index]
            assert observed == pytest.approx(0.15 * amount, rel=8 * np.finfo(np.float32).eps, abs=0)
    toc = bd.get_node(database="biosphere", code="toc")
    assert inventory[calculation.dicts.biosphere[toc.id]] == pytest.approx(
        0.15 * weights[1] * 0.08 * 0.001, rel=8 * np.finfo(np.float32).eps
    )


@pytest.mark.parametrize("year", (2020, 2030))
@pytest.mark.parametrize("origin", ("fossil", "biogenic", "atmospheric_dac"))
def test_conversion_waste_preserves_the_linked_inventory_and_carbon_origin(
    chemical_compilation_case: tuple[_RecipeContext, dict[str, dict[str, object]]],
    year: int,
    origin: str,
) -> None:
    """Keep dated burdens through two signed markets and converging treatment paths."""
    import bw2data as bd

    context, _ = chemical_compilation_case
    _ordinary_chemical_source(bd, context)
    ordinary = "ordinary" if year == 2020 else "technology"
    selected_ids = {
        "conversion_injection",
        "conversion_injection__non_fossil",
        "plastic_waste_market_rer_non_fossil",
        "plastic_waste_market_ch_non_fossil",
        "plastic_waste_market_gb_non_fossil",
        "plastic_waste_open_burning_glo_non_fossil",
        "plastic_waste_landfill_row_non_fossil",
        "plastic_waste_incineration_gb_non_fossil",
    }
    services = deepcopy(_chemical_reject_services(context.catalogue, tuple(selected_ids)))
    gb = services["plastic_waste_market_gb_non_fossil"]["construction"]
    ch = services["plastic_waste_market_ch_non_fossil"]["construction"]
    root = services["plastic_waste_market_rer_non_fossil"]["construction"]
    # The synthetic graph supplies two paths to the GB market and a third to its open-burning child.
    gb_link = {
        "selector": {
            **gb["selector"],
            "location": "GB",
            "exchange_type": "technosphere",
            "sign": "negative",
        },
        "service_id": "plastic_waste_market_gb_non_fossil",
    }
    ch["treatment_replacements"] = [gb_link, gb["treatment_replacements"][0]]
    root["treatment_replacements"] = [root["treatment_replacements"][0], gb_link]
    sources = {
        (ordinary, "conversion"): _process(
            ordinary,
            "conversion",
            "injection moulding",
            "injection moulding",
            production_amount=2.0,
            exchanges=[
                {
                    "input": (ordinary, "root_waste"),
                    "amount": -0.2,
                    "type": "technosphere",
                    "uncertainty type": 3,
                    "loc": -0.2,
                    "scale": 0.01,
                },
                {"input": (ordinary, "utility"), "amount": 3.0, "type": "technosphere"},
                {"input": ("biosphere", "co2_generic_f"), "amount": 0.4, "type": "biosphere"},
            ],
        ),
        (ordinary, "root_waste"): _process(
            ordinary,
            "root_waste",
            "market group for waste plastic, mixture",
            "waste plastic, mixture",
            production_amount=-2.0,
            exchanges=[
                {"input": (ordinary, "swiss_waste"), "amount": -0.5, "type": "technosphere"},
                {"input": (ordinary, "reject_market"), "amount": -1.5, "type": "technosphere"},
                {"input": (ordinary, "utility"), "amount": 0.02, "type": "technosphere"},
                {"input": ("biosphere", "water"), "amount": 0.004, "type": "biosphere"},
            ],
        ),
        (ordinary, "swiss_waste"): _process(
            ordinary,
            "swiss_waste",
            "market for waste plastic, mixture",
            "waste plastic, mixture",
            location="CH",
            production_amount=-4.0,
            exchanges=[
                {"input": (ordinary, "reject_market"), "amount": -2.0, "type": "technosphere"},
                {"input": (ordinary, "open_burning"), "amount": -2.0, "type": "technosphere"},
            ],
        ),
    }
    for (database, code), data in sources.items():
        node = bd.Database(database).new_node(
            code=code, **{k: v for k, v in data.items() if k != "exchanges"}
        )
        node.save()
        for exchange in data["exchanges"]:
            node.new_exchange(**exchange).save()
    bd.Database(ordinary).process()
    before = {
        code: [dict(row) for row in bd.get_node(database=ordinary, code=code).exchanges()]
        for code in (
            "conversion",
            "root_waste",
            "swiss_waste",
            "reject_market",
            "open_burning",
            "incineration",
            *(
                node["code"]
                for node in bd.Database(ordinary)
                if node["code"].startswith("landfill")
            ),
        )
    }
    routes = [
        route
        for route in context.catalogue["routes"]
        if route["provider_service_id"] in selected_ids
    ]
    catalogue = {**context.catalogue, "services": list(services.values()), "routes": routes}
    provider = PreparedProviderAuthority(
        identity="conversion_waste_fixture",
        interface_database="provider_interface",
        catalogue=catalogue,
        routes=pd.DataFrame(routes),
        method_authorities=_method_authority_rows(catalogue),
    )
    frontier = prepare_provider_frontier(
        bd,
        background_scenario=context.background_scenario,
        year=year,
        ordinary_database=ordinary,
        technology_database="technology",
        material_recipe=physical_material_recipe(),
        provider=provider,
        resolver_cache=_NativeResolverCache(ordinary, "biosphere"),
    )
    service_id = "conversion_injection" + ("" if origin == "fossil" else "__non_fossil")
    activity = bd.get_node(database=frontier.database_name, code=provider_service_code(service_id))
    assert next(iter(activity.production()))["amount"] == 1.6
    waste = next(row for row in activity.technosphere() if row["amount"] < 0)
    expected_waste = (
        (ordinary, "root_waste")
        if origin == "fossil"
        else (frontier.database_name, provider_service_code("plastic_waste_market_rer_non_fossil"))
    )
    assert waste.input.key == expected_waste
    assert waste["amount"] == -0.2
    assert waste["uncertainty type"] == 3
    assert waste["loc"] == -0.2
    assert waste["scale"] == 0.01
    owners = json.loads(activity["direct_treatment_carbon_owners_json"])
    # RER -> GB is 1.5/2 + (0.5/2)*(2/4) = 0.875; the separate
    # RER -> CH -> open burning path contributes (0.5/2)*(2/4) = 0.125.
    shares = (0.3, 0.2, 0.5) if year == 2020 else (0.1, 0.2, 0.7)
    weights = (0.875 * shares[0] + 0.125, 0.875 * shares[1], 0.875 * shares[2])
    for (code, suffix), weight in zip(
        (
            ("open_burning", "open_burning_glo"),
            ("landfill", "landfill_row"),
            ("incineration", "incineration_gb"),
        ),
        weights,
        strict=True,
    ):
        key = (
            (ordinary, code)
            if origin == "fossil"
            else (
                frontier.database_name,
                provider_service_code(f"plastic_waste_{suffix}_non_fossil"),
            )
        )
        owner = next(row for row in owners if (row["owner_database"], row["owner_code"]) == key)
        assert owner["owner_scale_per_provider_unit"] == pytest.approx(-0.2 / 1.6 * weight)
    for role, weight in (
        ("wastewater_treatment", 0.08),
        ("sludge_incineration", 0.024),
        ("sludge_landfarming", 0.048),
    ):
        key = (
            (ordinary, f"landfill_mixed_plastic_{role}")
            if origin == "fossil"
            else (
                frontier.database_name,
                provider_service_code(f"landfill_mixed_plastic__{role}__non_fossil"),
            )
        )
        owner = next(row for row in owners if (row["owner_database"], row["owner_code"]) == key)
        assert owner["owner_scale_per_provider_unit"] == pytest.approx(
            -0.2 / 1.6 * weights[1] * weight
        )
    assert len(owners) == 6
    calculation = LCA({activity: 1.0})
    calculation.lci()
    inventory = calculation.inventory.sum(axis=1).A1
    principal, auxiliary = _chemical_reject_gases(weights, year=year)
    # The extra converging market adds two factors to the preparation path;
    # eight eps32 still covers 12u/(1-12u) plus binary64 arithmetic.
    for (species, area), principal_amount in principal.items():
        expected_by_origin = {
            "f": auxiliary[(species, area)] + (principal_amount if origin == "fossil" else 0),
            "nf": 0 if origin == "fossil" else principal_amount,
        }
        for suffix, amount in expected_by_origin.items():
            flow = bd.get_node(database="biosphere", code=f"{species}_{area}_{suffix}")
            index = calculation.dicts.biosphere.get(flow.id)
            observed = 0.0 if index is None else inventory[index]
            assert observed == pytest.approx(
                0.2 / 1.6 * amount, rel=8 * np.finfo(np.float32).eps, abs=0
            )
    toc = bd.get_node(database="biosphere", code="toc")
    assert inventory[calculation.dicts.biosphere[toc.id]] == pytest.approx(
        0.2 / 1.6 * weights[1] * 0.08 * 0.001, rel=8 * np.finfo(np.float32).eps
    )
    direct = bd.get_node(database="biosphere", code="co2_generic_f")
    assert inventory[calculation.dicts.biosphere[direct.id]] == pytest.approx(0.4 / 1.6)
    fossil_key = (frontier.database_name, provider_service_code("conversion_injection"))
    fossil = LCA({bd.get_activity(fossil_key): 1.0})
    fossil.lci()
    utility = bd.get_node(database=ordinary, code="utility")
    assert calculation.supply_array[calculation.dicts.activity[utility.id]] == pytest.approx(
        fossil.supply_array[fossil.dicts.activity[utility.id]]
    )
    for code, records in before.items():
        assert [
            dict(row) for row in bd.get_node(database=ordinary, code=code).exchanges()
        ] == records
