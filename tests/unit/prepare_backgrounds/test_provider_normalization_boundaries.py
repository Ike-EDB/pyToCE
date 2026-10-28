"""Native normalization metadata, source admission and fuel-operation boundaries."""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from premise.brightway25 import write_brightway_database

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


@pytest.fixture
def native_foaming(tmp_path: Path) -> Iterator[tuple[Any, PreparedProviderAuthority]]:
    """Supply the actual foaming service and invented native operation records."""
    catalogue = load_provider_catalogue()
    service = next(
        service
        for service in catalogue["services"]
        if service["provider_service_id"] == "conversion_foaming"
    )
    provider = PreparedProviderAuthority(
        identity="bounded_foaming_authority",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": [service], "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="native_foaming",
        biosphere_database="biosphere",
        source_database="native",
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
            },
            searchable=False,
        )
        bd.Database("native").write(
            {
                ("native", "auxiliary"): {
                    "name": "invented formulation auxiliary",
                    "reference product": "auxiliary",
                    "unit": "kilogram",
                    "location": "RER",
                    "type": "process",
                    "exchanges": [
                        {"input": ("native", "auxiliary"), "type": "production", "amount": 1.0},
                        {"input": ("biosphere", "co2"), "type": "biosphere", "amount": 0.3},
                    ],
                },
                ("native", "foaming"): {
                    "name": "polymer foaming",
                    "reference product": "polymer foaming",
                    "unit": "kilogram",
                    "location": "RER",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": ("native", "foaming"),
                            "type": "production",
                            "amount": 2.0,
                            "uncertainty type": 2,
                            "loc": math.log(2),
                            "scale": 0.1,
                            "allocation": "native gross output",
                        },
                        *[
                            {
                                "input": ("native", "auxiliary"),
                                "type": "technosphere",
                                "amount": amount,
                                "uncertainty type": 2,
                                "loc": math.log(amount),
                                "scale": spread,
                                "allocation": "native auxiliary",
                            }
                            for amount, spread in ((0.002, 0.1), (0.001234567, 0.2))
                        ],
                        {
                            "input": ("biosphere", "co2"),
                            "type": "biosphere",
                            "amount": 0.07,
                            "uncertainty type": 2,
                            "loc": math.log(0.07),
                            "scale": 0.3,
                        },
                    ],
                },
            },
            searchable=False,
        )
        yield bd, provider


def test_foaming_preserves_native_production_and_duplicate_exchange_distributions(
    native_foaming: tuple[Any, PreparedProviderAuthority],
) -> None:
    """Retain native records and independently verify the processed central inventory."""
    bd, provider = native_foaming
    native = bd.get_node(database="native", code="foaming")
    original = [dict(exchange) for exchange in native.exchanges()]
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
    generated = bd.get_node(
        database=frontier.database_name,
        code=provider_service_code("conversion_foaming"),
    )
    expected = []
    for exchange in original:
        record = {key: value for key, value in exchange.items() if key != "output"}
        if record["type"] == "production":
            record["input"] = generated.key
        expected.append(record)
    actual = [
        {key: value for key, value in dict(exchange).items() if key != "output"}
        for exchange in generated.exchanges()
    ]
    assert sorted(actual, key=repr) == sorted(expected, key=repr)
    assert [dict(exchange) for exchange in native.exchanges()] == original
    calculation = LCA({generated: 1.0})
    calculation.lci()
    # Duplicate records are serialized separately and then summed at float32
    # precision. Their distributions remain distinct stored records above.
    auxiliary = float(np.float32(np.float32(0.002) + np.float32(0.001234567))) / 2.0
    auxiliary_node = bd.get_node(database="native", code="auxiliary")
    assert calculation.supply_array[calculation.dicts.activity[auxiliary_node.id]] == pytest.approx(
        auxiliary, rel=1e-14, abs=0
    )
    assert float(calculation.inventory.sum()) == pytest.approx(
        float(np.float32(0.07)) / 2.0 + auxiliary * float(np.float32(0.3)),
        rel=1e-14,
        abs=0,
    )


@pytest.mark.parametrize(
    ("exchange_type", "amount"),
    [("production", math.inf), ("technosphere", math.inf), ("biosphere", -math.inf)],
)
def test_fresh_premise_nonfinite_amounts_fail_before_provider_publication(
    native_foaming: tuple[Any, PreparedProviderAuthority],
    exchange_type: str,
    amount: float,
) -> None:
    """Reject infinite fresh fast-writer output before publishing a provider frontier."""
    bd, provider = native_foaming
    transformed = [
        {
            **{key: value for key, value in dict(node).items() if key != "id"},
            "exchanges": [
                {key: value for key, value in dict(row).items() if key != "output"}
                for row in node.exchanges()
            ],
        }
        for node in bd.Database("native")
    ]
    foaming = next(node for node in transformed if node["code"] == "foaming")
    exchange = next(
        exchange for exchange in foaming["exchanges"] if exchange["type"] == exchange_type
    )
    # Supply the nonfinite amount through the selected premise fast writer.
    # This is a fresh dependency payload; no completed store is edited.
    exchange["amount"] = amount
    write_brightway_database(transformed, "premise", fast=True)
    with pytest.raises(ValueError, match="Premise activity .*foaming.*nonfinite amount"):
        prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2030,
            ordinary_database="premise",
            technology_database="premise",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
    assert not any(
        name.startswith(f"{provider.interface_database}__services__") for name in bd.databases
    )


@pytest.mark.parametrize(("native_production", "native_gas"), [(1.0, 0.04), (2.0, 0.06)])
def test_tyre_conversion_uses_source_fuel_and_separately_owned_electricity(
    tmp_path: Path,
    native_production: float,
    native_gas: float,
) -> None:
    """Solve the actual tire/furnace recipe on two independent native fuel bases."""
    catalogue = load_provider_catalogue()
    furnace_id = "tyre_natural_gas_furnace_operation"
    services = [
        service
        for service in catalogue["services"]
        if service["provider_service_id"] in {"conversion_tyre", furnace_id}
    ]
    routes = [
        route
        for route in catalogue["routes"]
        if route["material"] == "Elastomers (tyres)"
        and route["provider_demand_type"] == "processing_conversion"
    ]
    provider = PreparedProviderAuthority(
        identity="bounded_tyre_fuel_authority",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": services, "routes": routes},
        routes=pd.DataFrame(routes),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="tyre_fuel_operation",
        biosphere_database="biosphere",
        source_database="native",
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
            },
            searchable=False,
        )
        suppliers = (
            (
                "gas",
                "market group for natural gas, high pressure",
                "natural gas, high pressure",
                "cubic meter",
                "Europe without Switzerland",
                0.3,
            ),
            (
                "lv",
                "market group for electricity, low voltage",
                "electricity, low voltage",
                "kilowatt hour",
                "Europe without Switzerland",
                0.1,
            ),
            (
                "mv",
                "market group for electricity, medium voltage",
                "electricity, medium voltage",
                "kilowatt hour",
                "RER",
                0.2,
            ),
            (
                "infrastructure",
                "invented industrial furnace",
                "industrial furnace",
                "unit",
                "RER",
                0.4,
            ),
            ("maintenance", "invented furnace lubricant", "lubricant", "kilogram", "RER", 0.5),
        )
        native = {
            ("native", code): {
                "name": name,
                "reference product": product,
                "unit": unit,
                "location": location,
                "type": "process",
                "exchanges": [
                    {"input": ("native", code), "type": "production", "amount": 1.0},
                    {"input": ("biosphere", "co2"), "type": "biosphere", "amount": emission},
                ],
            }
            for code, name, product, unit, location, emission in suppliers
        }
        native[("native", "furnace")] = {
            "name": "heat production, natural gas, at industrial furnace >100kW",
            "reference product": "heat, district or industrial, natural gas",
            "unit": "megajoule",
            "location": "Europe without Switzerland",
            "type": "process",
            "exchanges": [
                {"input": ("native", "furnace"), "type": "production", "amount": native_production},
                {"input": ("native", "gas"), "type": "technosphere", "amount": native_gas},
                {"input": ("native", "lv"), "type": "technosphere", "amount": 0.004},
                {"input": ("native", "infrastructure"), "type": "technosphere", "amount": 0.0001},
                {"input": ("native", "maintenance"), "type": "technosphere", "amount": 0.002},
                {
                    "input": ("biosphere", "co2"),
                    "type": "biosphere",
                    "amount": 0.08,
                    "uncertainty type": 2,
                    "loc": math.log(0.08),
                    "scale": 0.1,
                    "allocation": "native furnace combustion",
                },
            ],
        }
        bd.Database("native").write(native, searchable=False)
        original = {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for node in bd.Database("native")
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
        furnace = bd.get_node(
            database=frontier.database_name, code=provider_service_code(furnace_id)
        )
        parent = bd.get_node(
            database=frontier.database_name, code=provider_service_code("conversion_tyre")
        )
        assert {exchange.input.key: exchange["amount"] for exchange in parent.technosphere()} == {
            ("native", "mv"): 1.764,
            furnace.key: 0.3963,
        }
        assert next(iter(furnace.production()))["amount"] == native_gas
        retained = [
            {key: value for key, value in exchange.items() if key != "output"}
            for exchange in original[("native", "furnace")]
            if exchange["type"] != "production" and exchange["input"] != ("native", "lv")
        ]
        assert sorted(
            [
                {key: value for key, value in dict(exchange).items() if key != "output"}
                for exchange in furnace.exchanges()
                if exchange["type"] != "production"
            ],
            key=repr,
        ) == sorted(retained, key=repr)
        calculation = LCA({parent: 1.0})
        calculation.lci()
        # Apply per-exchange storage precision to source-demand equations.
        scale = float(np.float32(0.3963)) / float(np.float32(native_gas))
        expected = {
            "gas": float(np.float32(0.3963)),
            "lv": 0.0,
            "mv": float(np.float32(1.764)),
            "infrastructure": scale * float(np.float32(0.0001)),
            "maintenance": scale * float(np.float32(0.002)),
        }
        assert {
            code: float(
                calculation.supply_array[
                    calculation.dicts.activity[bd.get_node(database="native", code=code).id]
                ]
            )
            for code in expected
        } == pytest.approx(expected, rel=1e-13, abs=0)
        expected_co2 = scale * float(np.float32(0.08)) + sum(
            expected[code] * float(np.float32(emission))
            for code, _name, _product, _unit, _location, emission in suppliers
        )
        assert float(calculation.inventory.sum()) == pytest.approx(expected_co2, rel=1e-13, abs=0)
        assert original == {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for node in bd.Database("native")
        }
        route_row = next(
            row for row in frontier.semantic_rows if row["ledger_entry_type"] == "route_mapping"
        )
        assert route_row["provider_service_id"] == "conversion_tyre"
