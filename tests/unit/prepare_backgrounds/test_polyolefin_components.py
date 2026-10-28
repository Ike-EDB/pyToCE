"""Allocated polymer and recovered-monomer boundaries with small native databases."""

from collections import Counter
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PROVIDER_INTERFACE_DATABASE,
    PreparedProviderAuthority,
    _method_authority_rows,
    _route_table,
    _service_source_closures,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.polyolefins import polyolefin_datasets
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.fixture
def polyolefin_case(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[tuple[Any, dict[str, Any], _RecipeContext]]:
    """Use packaged service definitions against invented allocated native inventories."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    original_properties = getattr(request, "param", True)
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    native = {}
    for code, product in (
        ("e1", "ethylene"),
        ("e2", "ethylene"),
        ("p", "propylene"),
        ("utility", "utility"),
        ("waste", "waste"),
        ("naphtha", "naphtha"),
    ):
        native[("native", code)] = _process("native", code, f"supplier {code}", product)
    flows = {}
    for species in ("Carbon dioxide", "Carbon monoxide", "Methane"):
        for origin in ("fossil", "non-fossil"):
            code = f"{species}_{origin}"
            flows[("biosphere", code)] = {
                "name": f"{species}, {origin}",
                "unit": "kilogram",
                "type": "emission",
                "categories": ("air", "non-urban air or from high stacks"),
            }
    flows[("biosphere", "unrelated")] = {
        "name": "Carbon dioxide, in air",
        "unit": "kilogram",
        "type": "natural resource",
        "categories": ("natural resource", "in air"),
    }
    for polymer in ("hdpe", "ldpe", "pp"):
        selector = services[f"{polymer}_polymerization_operation"]["construction"]["base"][
            "selector"
        ]
        native[("native", polymer)] = _process(
            "native",
            polymer,
            selector["name"],
            selector["reference_product"],
            production_amount=2,
            exchanges=[
                {
                    "input": ("native", "e1"),
                    "amount": 1.5,
                    "type": "technosphere",
                    "uncertainty type": 3,
                    "loc": 1.5,
                    "scale": 0.1,
                },
                {
                    "input": ("native", "e2"),
                    "amount": 0.5,
                    "type": "technosphere",
                    "uncertainty type": 3,
                    "loc": 0.5,
                    "scale": 0.025,
                },
                {"input": ("native", "p"), "amount": 0.5, "type": "technosphere"},
                {
                    "input": ("native", "utility"),
                    "amount": 4,
                    "type": "technosphere",
                    "uncertainty type": 3,
                    "loc": 4,
                    "scale": 0.25,
                },
                {"input": ("native", "waste"), "amount": -0.25, "type": "technosphere"},
                {"input": ("biosphere", "unrelated"), "amount": 0.125, "type": "biosphere"},
            ],
        )
    for monomer, q, naphtha, utility in (("ethylene", 4, 8, 2), ("propylene", 2, 3, 3)):
        native[("native", f"cracker_{monomer}")] = _process(
            "native",
            f"cracker_{monomer}",
            "unsaturated hydrocarbons production, steam cracking operation, average",
            monomer,
            production_amount=q,
            location="KR",
            exchanges=[
                {"input": ("native", "naphtha"), "amount": naphtha, "type": "technosphere"},
                {
                    "input": ("native", "utility"),
                    "amount": utility,
                    "type": "technosphere",
                    "uncertainty type": 3,
                    "loc": utility,
                    "scale": 0.1,
                },
                {"input": ("native", "waste"), "amount": -0.2, "type": "technosphere"},
                {"input": ("biosphere", "unrelated"), "amount": 0.3, "type": "biosphere"},
                *[
                    {
                        "input": ("biosphere", f"{species}_fossil"),
                        "amount": 0.1,
                        "type": "biosphere",
                        "uncertainty type": 3,
                        "loc": 0.1,
                        "scale": 0.01,
                        "name": f"{species}, fossil",
                        "unit": "kilogram",
                        "categories": ("air", "non-urban air or from high stacks"),
                        **(
                            {
                                "flow": f"{species}_fossil",
                                "properties": {
                                    "carbon content, fossil": {
                                        "amount": carbon,
                                        "unit": "dimensionless",
                                    },
                                    "carbon content, non-fossil": {
                                        "amount": 0.0,
                                        "unit": "dimensionless",
                                    },
                                    "carbon allocation": {"amount": carbon, "unit": "kilogram"},
                                    "wet mass": {"amount": 1.0, "unit": "kilogram"},
                                },
                            }
                            if original_properties
                            else {}
                        ),
                    }
                    for species, carbon in (
                        ("Carbon dioxide", 0.27),
                        ("Carbon monoxide", 0.43),
                        ("Methane", 0.75),
                    )
                ],
            ],
        )
    names = BrightwayNames(tmp_path / "brightway", "polyolefin", "biosphere", "native")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(flows, searchable=False)
        bd.Database("native").write(native, searchable=False)
        # Empty early technology database proves these operations resolve ordinary data.
        bd.Database("technology").write({}, searchable=False)
        before = deepcopy(bd.Database("native").load())
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2020,
            provider_database="providers",
            source=_ActivityResolver(bd, "native"),
            ordinary=_ActivityResolver(bd, "native"),
            technology=_ActivityResolver(bd, "technology"),
            biosphere=_FlowResolver(bd, "biosphere"),
            material_recipe=physical_material_recipe(),
            catalogue=catalogue,
            method_authorities={
                cast(str, row["method_authority_id"]): row
                for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        yield bd, services, context
        assert bd.Database("native").load() == before


def test_native_polymer_reconstruction_preserves_records_and_allocated_coefficients(
    polyolefin_case: tuple[Any, dict[str, Any], _RecipeContext],
) -> None:
    """Recover the native inventory and separate ethylene and propylene supplies."""
    bd, services, context = polyolefin_case
    data = cast(
        dict[tuple[str, str], dict[str, Any]], polyolefin_datasets(context, services, "providers")
    )
    for polymer in ("hdpe", "ldpe", "pp"):
        components = [
            data[("providers", provider_service_code(sid))]
            for sid in (
                f"{polymer}_polymerization_operation",
                f"{polymer}_ethylene_virgin_fossil",
                f"{polymer}_propylene_virgin_fossil",
            )
        ]
        native_records = [
            dict(e)
            for e in bd.get_activity(("native", polymer)).exchanges()
            if e["type"] != "production"
        ]
        for exchange in native_records:
            exchange.pop("output")
        actual = [e for component in components for e in component["exchanges"][1:]]
        assert sorted(actual, key=lambda e: e["input"]) == sorted(
            native_records, key=lambda e: e["input"]
        )
        assert [component["exchanges"][0]["amount"] for component in components] == [2, 2, 2]
        vector = Counter()
        for component in components:
            for exchange in component["exchanges"][1:]:
                vector[exchange["input"]] += exchange["amount"] / 2
        assert vector == {
            ("native", "e1"): 0.75,
            ("native", "e2"): 0.25,
            ("native", "p"): 0.25,
            ("native", "utility"): 2,
            ("native", "waste"): -0.125,
            ("biosphere", "unrelated"): 0.0625,
        }
        for monomer, native_keys in (
            ("ethylene", {("native", "e1"), ("native", "e2")}),
            ("propylene", {("native", "p")}),
        ):
            for route, supplier in (
                ("virgin_biogenic", f"{monomer}_biogenic"),
                ("virgin_atmospheric_dac", f"{monomer}_atmospheric_dac"),
                ("recovered_fossil", f"cracker_{monomer}_fossil"),
                ("recovered_non_fossil", f"cracker_{monomer}_non_fossil"),
            ):
                expected = deepcopy([e for e in native_records if e["input"] in native_keys])
                for exchange in expected:
                    exchange["input"] = ("providers", provider_service_code(supplier))
                supply = data[("providers", provider_service_code(f"{polymer}_{monomer}_{route}"))]
                assert supply["exchanges"][0]["amount"] == 2
                assert supply["exchanges"][1:] == expected
    virgin_e = data[("providers", provider_service_code("hdpe_ethylene_virgin_fossil"))]
    virgin_p = data[("providers", provider_service_code("hdpe_propylene_virgin_fossil"))]
    assert [e["amount"] / 2 * 0.75 for e in virgin_e["exchanges"][1:]] == [0.5625, 0.1875]
    assert [e["amount"] / 2 * 0.25 for e in virgin_p["exchanges"][1:]] == [0.0625]
    assert {row["native_database_name"] for row in context.technology_rows} == {"native"}


@pytest.mark.parametrize(
    "polyolefin_case", (True, False), indirect=True, ids=("original", "premise")
)
def test_recovered_monomers_keep_allocated_cracker_basis_and_exact_carbon_mapping(
    polyolefin_case: tuple[Any, dict[str, Any], _RecipeContext],
) -> None:
    """Suppress naphtha only, retain native signs/uncertainty and keep both q bases."""
    bd, services, context = polyolefin_case
    data = cast(
        dict[tuple[str, str], dict[str, Any]], polyolefin_datasets(context, services, "providers")
    )
    utility = 0.0
    for monomer, fraction, q in (("ethylene", 0.25, 4), ("propylene", 0.75, 2)):
        native_records = [
            dict(e)
            for e in bd.get_activity(("native", f"cracker_{monomer}")).exchanges()
            if e["type"] != "production" and e["input"] != ("native", "naphtha")
        ]
        for exchange in native_records:
            exchange.pop("output")
        for carbon in ("fossil", "non_fossil"):
            cracker = data[("providers", provider_service_code(f"cracker_{monomer}_{carbon}"))]
            assert cracker["exchanges"][0]["amount"] == q
            expected = deepcopy(native_records)
            if carbon == "non_fossil":
                for exchange in expected:
                    if exchange["type"] == "biosphere" and exchange["input"][1].endswith("_fossil"):
                        species = exchange["input"][1].removesuffix("_fossil")
                        exchange["input"] = ("biosphere", f"{species}_non-fossil")
                        exchange["name"] = f"{species}, non-fossil"
                        if "properties" in exchange:
                            exchange["flow"] = f"{species}_non-fossil"
                            properties = exchange["properties"]
                            properties["carbon content, non-fossil"]["amount"] = properties[
                                "carbon content, fossil"
                            ]["amount"]
                            properties["carbon content, fossil"]["amount"] = 0.0
            assert cracker["exchanges"][1:] == expected
        supply = data[("providers", provider_service_code(f"hdpe_{monomer}_recovered_fossil"))]
        amount = sum(e["amount"] for e in supply["exchanges"][1:]) / 2 * fraction
        assert amount == {"ethylene": 0.25, "propylene": 0.1875}[monomer]
        cracker = data[("providers", provider_service_code(f"cracker_{monomer}_fossil"))]
        utility += (
            amount
            * next(
                e["amount"] for e in cracker["exchanges"][1:] if e["input"] == ("native", "utility")
            )
            / q
        )
    assert utility == 0.40625
    assert {
        row["mapping_status"]
        for row in context.semantic_rows
        if row["ledger_entry_type"] == "cracker_direct_carbon_mapping"
    } == {"native_fossil_identity", "exact_non_fossil_counterpart"}


def test_dated_polyolefin_frontier_preserves_native_inventory_and_mixed_supply(
    polyolefin_case: tuple[Any, dict[str, Any], _RecipeContext],
) -> None:
    """Resolve, write and solve actual component services on their different native q bases."""
    bd, services, context = polyolefin_case
    selected = {
        "hdpe_polymerization_operation",
        "hdpe_ethylene_virgin_fossil",
        "hdpe_propylene_virgin_fossil",
        "hdpe_ethylene_recovered_fossil",
        "hdpe_propylene_recovered_fossil",
        "cracker_ethylene_fossil",
        "cracker_propylene_fossil",
    }
    catalogue = {
        **context.catalogue,
        "services": [services[name] for name in sorted(selected)],
        "routes": [
            route
            for route in cast(list[dict[str, Any]], context.catalogue["routes"])
            if route["provider_service_id"] in selected
        ],
    }
    authority = PreparedProviderAuthority(
        identity="provider::polyolefin_fixture",
        interface_database=PROVIDER_INTERFACE_DATABASE,
        catalogue=catalogue,
        routes=_route_table(catalogue),
        method_authorities=_method_authority_rows(catalogue),
    )
    frontier = prepare_provider_frontier(
        bd,
        background_scenario="remind_ssp2_npi",
        year=2020,
        ordinary_database="native",
        technology_database="technology",
        material_recipe=physical_material_recipe(),
        provider=authority,
        resolver_cache=_NativeResolverCache("native", "biosphere"),
    )
    database = frontier.database_name
    method = ("polyolefin_fixture", "unit factors")
    bd.Method(method).register(unit="fixture impact")
    bd.Method(method).write(
        [
            (node.key, 1.0)
            for node in bd.Database("biosphere")
            if node.key[1] == "unrelated" or node.key[1].endswith("_fossil")
        ]
    )
    terminal_keys = {("native", name) for name in ("e1", "e2", "p", "utility", "waste", "naphtha")}

    def calculate(
        demand: dict[Any, float],
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], float]:
        """Read flow and supplier totals by identity, independently of matrix order."""
        lca = LCA(demand, method=method)
        lca.lci()
        supply_array = cast(NDArray[np.float64] | None, lca.supply_array)
        assert supply_array is not None
        lca.lcia()
        totals = np.asarray(lca.inventory.sum(axis=1)).ravel()
        inventory = {
            bd.get_node(id=identifier).key: float(totals[index])
            for identifier, index in lca.dicts.biosphere.items()
        }
        suppliers = {
            bd.get_node(id=identifier).key: float(supply_array[index])
            for identifier, index in lca.dicts.activity.items()
            if bd.get_node(id=identifier).key in terminal_keys
        }
        return inventory, suppliers, float(lca.score)

    native = calculate({bd.get_activity(("native", "hdpe")): 1.0})
    for ethylene_recovered, propylene_recovered in ((0.0, 0.0), (0.25, 0.75)):
        amounts = {"hdpe_polymerization_operation": 1.0}
        for monomer, recovered in (
            ("ethylene", ethylene_recovered),
            ("propylene", propylene_recovered),
        ):
            amounts[f"hdpe_{monomer}_virgin_fossil"] = 1 - recovered
            amounts[f"hdpe_{monomer}_recovered_fossil"] = recovered
        inventory, suppliers, score = calculate(
            {
                bd.get_activity((database, provider_service_code(name))): amount
                for name, amount in amounts.items()
            }
        )
        if ethylene_recovered == 0:
            assert inventory == native[0]
            assert suppliers == native[1]
            assert score == native[2] == 0.0625
        else:
            # bw_processing serializes iterable amounts as float32 before
            # constructing the matrix. Compare that documented numerical basis
            # exactly; the record-level tests above retain the native floats.
            waste_coefficient = float(np.float32(0.2))
            unrelated_coefficient = float(np.float32(0.3))
            carbon_coefficient = float(np.float32(0.1))
            cracker_scale = 0.25 / 4 + 0.1875 / 2
            assert suppliers == {
                ("native", "e1"): 0.5625,
                ("native", "e2"): 0.1875,
                ("native", "p"): 0.0625,
                ("native", "utility"): 2.40625,
                ("native", "waste"): -0.125 - waste_coefficient * cracker_scale,
                ("native", "naphtha"): 0.0,
            }
            # Only biosphere nodes linked by exchanges receive matrix rows.
            assert inventory == {
                **{
                    ("biosphere", f"{species}_fossil"): carbon_coefficient * cracker_scale
                    for species in ("Carbon dioxide", "Carbon monoxide", "Methane")
                },
                ("biosphere", "unrelated"): 0.0625 + unrelated_coefficient * cracker_scale,
            }
            assert (
                score == 0.0625 + (unrelated_coefficient + 3 * carbon_coefficient) * cracker_scale
            )
