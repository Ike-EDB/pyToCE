"""Finished PET waste attribution and preservation of native water exchanges."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)

_EFFLUENT_FLOWS = (
    ("bod", "BOD5, Biological Oxygen Demand", ["water"]),
    ("effluent", "COD, Chemical Oxygen Demand", ["water"]),
    ("doc", "DOC, Dissolved Organic Carbon", ["water", "surface water"]),
    ("nitrogen", "Nitrogen", ["water"]),
    ("phosphorus", "Phosphorus", ["water"]),
    ("toc", "TOC, Total Organic Carbon", ["water", "surface water"]),
)


@pytest.fixture
def fibre_frontier(tmp_path: Path) -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    """Compile actual PET declarations against small complete native fixtures."""
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    market_id = "plastic_waste_market_row_non_fossil"
    branches = services[market_id]["construction"]["treatment_replacements"]
    operation_ids = {
        "fibres_pet_conversion",
        "fibres_pet_conversion__non_fossil",
    }
    selected = operation_ids | {market_id} | {row["service_id"] for row in branches}
    bounded = {
        **catalogue,
        "services": [
            row for row in catalogue["services"] if row["provider_service_id"] in selected
        ],
        "routes": [],
    }
    provider = PreparedProviderAuthority(
        identity="invented_fibre_operations",
        interface_database="fibre_interface",
        catalogue=bounded,
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(bounded),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="fibre_operations",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        flows = {}
        flow_keys = {}
        for branch in branches:
            for rule in services[branch["service_id"]]["construction"]["replace_biosphere"]:
                for field in ("selector", "replacement_selector"):
                    spec = rule[field]
                    identity = (spec["name"], tuple(spec["categories"]))
                    if identity not in flow_keys:
                        key = ("biosphere", f"gas_{len(flow_keys)}")
                        flow_keys[identity] = key
                        flows[key] = {**spec, "type": "emission"}
        for code, categories, name, unit in (
            ("air_water", ["air"], "Water", "cubic meter"),
            ("liquid_water", ["water"], "Water", "cubic meter"),
            (
                "withdrawal",
                ["natural resource", "in water"],
                "Water, cooling, unspecified natural origin",
                "cubic meter",
            ),
            ("sludge_release", ["air"], "Invented sludge release", "kilogram"),
            *[(code, categories, name, "kilogram") for code, name, categories in _EFFLUENT_FLOWS],
        ):
            flows[("biosphere", code)] = {
                "name": name,
                "unit": unit,
                "categories": categories,
                "type": "natural resource" if code == "withdrawal" else "emission",
            }
        bd.Database("biosphere").write(flows, searchable=False)
        data = {}
        term_info = []
        market_inputs = []
        for i, branch in enumerate(branches, 1):
            sid = branch["service_id"]
            spec = services[sid]["construction"]["base"]["selector"]
            code = f"treatment_{i}"
            production = -1.0
            emissions = []
            for j, rule in enumerate(services[sid]["construction"]["replace_biosphere"], 1):
                key = flow_keys[(rule["selector"]["name"], tuple(rule["selector"]["categories"]))]
                target = flow_keys[
                    (
                        rule["replacement_selector"]["name"],
                        tuple(rule["replacement_selector"]["categories"]),
                    )
                ]
                amount = i * j / 100
                emissions.append({"input": key, "type": "biosphere", "amount": amount})
                term_info.append((sid, code, i / 78, production, key, target, amount))
            data[("native", code)] = _process(
                "native",
                code,
                spec["name"],
                spec["reference_product"],
                location=spec["exact_location"],
                production_amount=production,
                exchanges=emissions,
            )
            market_inputs.append(
                {"input": ("native", code), "type": "technosphere", "amount": -i / 39}
            )
        data[("native", "waste")] = _process(
            "native",
            "waste",
            "market for waste plastic, mixture",
            "waste plastic, mixture",
            location="RoW",
            production_amount=-2.0,
            exchanges=market_inputs,
        )
        for code, name, product, unit, location in (
            (
                "resin",
                "market for polyethylene terephthalate, granulate, amorphous",
                "polyethylene terephthalate, granulate, amorphous",
                "kilogram",
                "GLO",
            ),
            (
                "power",
                "market group for electricity, medium voltage",
                "electricity, medium voltage",
                "kilowatt hour",
                "GLO",
            ),
            (
                "heat",
                "heat production, natural gas, at industrial furnace >100kW",
                "heat, district or industrial, natural gas",
                "megajoule",
                "RoW",
            ),
            (
                "water_heat",
                "heat production, natural gas, at industrial furnace >100kW",
                "heat, district or industrial, natural gas",
                "megajoule",
                "Europe without Switzerland",
            ),
            ("water_plant", "wastewater plant", "wastewater plant", "unit", "GLO"),
        ):
            data[("native", code)] = _process(
                "native",
                code,
                name,
                product,
                location=location,
            )
            data[("native", code)]["unit"] = unit
        data[("native", "sludge")] = _process(
            "native",
            "sludge",
            "treatment of sludge",
            "sludge",
            production_amount=-1,
            exchanges=[
                {"input": ("biosphere", "sludge_release"), "type": "biosphere", "amount": 1}
            ],
        )
        for code, factor in (("water_treatment", 0.1), ("water_untreated", 0.4)):
            data[("native", code)] = _process(
                "native",
                code,
                code,
                "wastewater",
                production_amount=-1,
                exchanges=[
                    {"input": ("biosphere", flow), "type": "biosphere", "amount": i * factor}
                    for i, (flow, _name, _categories) in enumerate(_EFFLUENT_FLOWS, 1)
                ],
            )
            data[("native", code)]["unit"] = "cubic meter"
        data[("native", "water_treatment")]["exchanges"].extend(
            [
                {"input": ("native", "water_heat"), "type": "technosphere", "amount": 0.5},
                {"input": ("native", "power"), "type": "technosphere", "amount": 0.25},
                {"input": ("native", "sludge"), "type": "technosphere", "amount": -0.2},
                {"input": ("native", "water_plant"), "type": "technosphere", "amount": 1e-4},
            ]
        )
        wastewater_inputs = []
        for i, location in enumerate(("CA-QC", "Europe without Switzerland", "RoW", "CH"), 1):
            code = f"wastewater_{i}"
            data[("native", code)] = _process(
                "native",
                code,
                "market for wastewater, average",
                "wastewater, average",
                location=location,
                production_amount=-1,
                exchanges=[
                    {
                        "input": ("native", "water_treatment"),
                        "type": "technosphere",
                        "amount": -0.8,
                    },
                    {
                        "input": ("native", "water_untreated"),
                        "type": "technosphere",
                        "amount": -0.2,
                    },
                ],
            )
            data[("native", code)]["unit"] = "cubic meter"
            wastewater_inputs.append(
                {
                    "input": ("native", code),
                    "type": "technosphere",
                    "amount": -0.002 * i,
                    "uncertainty type": 2,
                    "loc": float(np.log(0.002 * i)),
                    "scale": 0.1,
                    "negative": True,
                    "comment": "Invented dated wastewater demand",
                }
            )
        native = _process(
            "native",
            "fibre",
            "polyester fibre production, finished",
            "fibre, polyester",
            location="RoW",
            production_amount=2.0,
            exchanges=[
                {"input": ("native", "resin"), "type": "technosphere", "amount": 2.4},
                {
                    "input": ("native", "waste"),
                    "type": "technosphere",
                    "amount": -0.4,
                    "activity": "invented native waste producer",
                    "flow": "invented waste plastic product",
                    "name": "waste plastic, mixture",
                    "properties": {
                        "carbon content": {"amount": 0.7},
                        "carbon content, fossil": {"amount": 0.7},
                        "carbon content, non-fossil": {"amount": 0.0},
                        "dry mass": {"amount": 1.0},
                    },
                    "uncertainty type": 3,
                    "loc": -0.4,
                    "scale": 0.01,
                },
                {"input": ("native", "power"), "type": "technosphere", "amount": 3.0},
                {"input": ("native", "heat"), "type": "technosphere", "amount": 5.0},
                {"input": ("native", "water_heat"), "type": "technosphere", "amount": 0.7},
                *wastewater_inputs,
                {
                    "input": ("biosphere", "air_water"),
                    "type": "biosphere",
                    "flow": "air_water",
                    "name": "Water",
                    "unit": "cubic meter",
                    "categories": ("air",),
                    "properties": {"water content": {"amount": 1.0, "unit": "dimensionless"}},
                    "amount": 0.6,
                    "formula": "WC*(1-CW_to_air)",
                    "uncertainty type": 3,
                    "loc": 0.6,
                    "scale": 0.02,
                },
                {
                    "input": ("biosphere", "liquid_water"),
                    "type": "biosphere",
                    "flow": "liquid_water",
                    "name": "Water",
                    "unit": "cubic meter",
                    "categories": ("water",),
                    "properties": {"water content": {"amount": 1.0, "unit": "dimensionless"}},
                    "amount": 0.05,
                    "formula": "CW_to_air*WC",
                    "uncertainty type": 3,
                    "loc": 0.05,
                    "scale": 0.01,
                },
                {"input": ("biosphere", "withdrawal"), "type": "biosphere", "amount": 0.65},
                *[
                    {"input": ("biosphere", code), "type": "biosphere", "amount": 0.03 * i}
                    for i, (code, _name, _categories) in enumerate(_EFFLUENT_FLOWS, 1)
                ],
            ],
        )
        native["exchanges"][0].update(
            {
                "uncertainty type": 2,
                "loc": 0.6,
                "scale": 0.1,
                "activity": "invented native fibre producer",
                "properties": {
                    "carbon content": {"amount": 0.625},
                    "carbon content, fossil": {"amount": 0.625},
                    "carbon content, non-fossil": {"amount": 0.0},
                },
            }
        )
        data[("native", "fibre")] = native
        bd.Database("native").write(data, searchable=False)
        original = {
            node.key: [dict(row) for row in node.exchanges()] for node in bd.Database("native")
        }
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2030,
            ordinary_database="native",
            technology_database="native",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
        yield bd, frontier, {"original": original, "term_info": term_info}


@pytest.mark.parametrize("origin", ("fossil", "biogenic", "atmospheric_dac"))
def test_pet_finished_operation_preserves_native_water_and_attributes_waste(
    fibre_frontier: tuple[Any, Any, dict[str, Any]],
    origin: str,
) -> None:
    """Keep finished production and complete retained exchanges through actual LCI."""
    bd, frontier, evidence = fibre_frontier
    sid = "fibres_pet_conversion" + ("" if origin == "fossil" else "__non_fossil")
    operation = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
    original = evidence["original"][("native", "fibre")]
    expected = []
    for source in original:
        row = {k: v for k, v in source.items() if k != "output"}
        if row["input"] == ("native", "resin"):
            continue
        if row["type"] == "production":
            row["input"] = operation.key
            row.pop("activity")
            row.pop("properties")
        elif row["input"] == ("native", "waste") and origin != "fossil":
            row["input"] = (
                frontier.database_name,
                provider_service_code("plastic_waste_market_row_non_fossil"),
            )
            row.pop("activity")
            row["properties"] = {
                "carbon content": {"amount": 0.7},
                "dry mass": {"amount": 1.0},
            }
        expected.append(row)
    actual = [
        {k: v for k, v in dict(row).items() if k != "output"} for row in operation.exchanges()
    ]
    assert sorted(actual, key=repr) == sorted(expected, key=repr)
    owners = json.loads(operation["direct_treatment_carbon_owners_json"])
    assert len(owners) == 12
    inventory_expected = {}
    for leaf_id, code, weight, production, fossil, non_fossil, amount in evidence["term_info"]:
        owner_key = (
            ("native", code)
            if origin == "fossil"
            else (frontier.database_name, provider_service_code(leaf_id))
        )
        owner = next(
            row for row in owners if (row["owner_database"], row["owner_code"]) == owner_key
        )
        assert owner["owner_scale_per_provider_unit"] == pytest.approx(-0.2 * weight)
        key = fossil if origin == "fossil" else non_fossil
        inventory_expected[key] = (
            inventory_expected.get(key, 0.0) + (-0.2 * weight / production) * amount
        )
    calculation = LCA({operation: 1.0})
    calculation.lci()
    inventory = calculation.inventory.sum(axis=1).A1
    for key, amount in inventory_expected.items():
        assert inventory[calculation.dicts.biosphere[bd.get_activity(key).id]] == pytest.approx(
            amount, rel=5 * np.finfo(np.float32).eps
        )
    for code, amount in (
        ("liquid_water", 0.025),
        ("air_water", 0.3),
        ("withdrawal", 0.325),
        ("sludge_release", 0.0016),
        *[
            (code, (0.015 + 0.0016) * i)
            for i, (code, _name, _categories) in enumerate(_EFFLUENT_FLOWS, 1)
        ],
    ):
        assert inventory[
            calculation.dicts.biosphere[bd.get_node(database="biosphere", code=code).id]
        ] == pytest.approx(amount)
    # Per kilogram of finished fiber, 0.01 m3 is disposed: 80% treated and
    # 20% untreated. Both native direct indicators and linked treatment remain;
    # the separate source treatment heat is 0.7/2 MJ per kg finished output.
    for code, amount in (
        ("resin", 0),
        ("heat", 2.5),
        ("water_heat", 0.35 + 0.004),
        ("power", 1.502),
        ("water_treatment", 0.008),
        ("water_untreated", 0.002),
        ("sludge", 0.0016),
        ("water_plant", 8e-7),
    ):
        node = bd.get_node(database="native", code=code)
        assert calculation.supply_array[calculation.dicts.activity[node.id]] == pytest.approx(
            amount
        )
    rows = [
        row
        for row in frontier.technology_rows
        if row["provider_service_id"] == sid and row.get("component_id") == "native_operation"
    ]
    terms = json.loads(rows[0]["project_terms_json"])
    # The only remaining substitution is the native plastic-waste input.
    assert len(terms) == 1
    assert not any("source_flow_categories_json" in row for row in terms)
    assert {
        node.key: [dict(row) for row in node.exchanges()] for node in bd.Database("native")
    } == evidence["original"]
