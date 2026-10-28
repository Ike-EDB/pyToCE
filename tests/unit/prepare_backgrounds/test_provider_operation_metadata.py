"""Native operation production, record preservation and glucose-feed uncertainty."""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from stats_arrays import LognormalUncertainty, UncertaintyBase

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
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
from pytoce.prepare_backgrounds.preparation.providers.operation_compilation import (
    _operation_only_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.fixture
def glucose_operation(tmp_path: Path) -> Iterator[tuple[Any, PreparedProviderAuthority]]:
    """Compile the retained glucose recipe with a synthetic starch supply inventory."""
    catalogue = load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    feed = services["starch_delivered_fossil"]
    provider = PreparedProviderAuthority(
        identity="bounded_glucose_composition",
        interface_database="bounded_interface",
        catalogue={
            **catalogue,
            "services": [
                services["glucose_fossil"],
                {
                    **feed,
                    # This fixture isolates glucose conversion; starch manufacture
                    # and delivery inventories are checked in separate tests.
                    "construction": {
                        "kind": "native_activity",
                        "source_role": "native_ecoinvent_provider",
                        "selector": {
                            "name": "bounded synthetic starch supply",
                            "reference_product": feed["reference_product"],
                            "unit": "kilogram",
                            "exact_location": "GLO",
                        },
                    },
                },
            ],
            "routes": [],
        },
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="glucose_records",
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
                ("biosphere", "co2_non_fossil"): {
                    "name": "Carbon dioxide, non-fossil",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
            },
            searchable=False,
        )
        bd.Database("native").write(
            {
                ("native", "synthetic_starch"): {
                    "name": "bounded synthetic starch supply",
                    "reference product": feed["reference_product"],
                    "unit": "kilogram",
                    "location": "GLO",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": ("native", "synthetic_starch"),
                            "type": "production",
                            "amount": 1.0,
                        }
                    ],
                },
                ("native", "starch"): {
                    "name": "market for maize starch",
                    "reference product": "maize starch",
                    "unit": "kilogram",
                    "location": "GLO",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": ("native", "starch"),
                            "type": "production",
                            "amount": 1.0,
                        }
                    ],
                },
                ("native", "glucose"): {
                    "name": "glucose production",
                    "reference product": "glucose",
                    "unit": "kilogram",
                    "location": "RER",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": ("native", "glucose"),
                            "type": "production",
                            "amount": 2.0,
                            "uncertainty type": 2,
                            "loc": 0.6,
                            "scale": 0.1,
                            "allocation": "native reference production",
                            "activity": "invented native glucose producer",
                            "properties": {
                                "carbon content": {"amount": 0.4},
                                "carbon content, fossil": {"amount": 0.0},
                                "carbon content, non-fossil": {"amount": 0.4},
                                "dry mass": {"amount": 1.0},
                            },
                        },
                        {
                            "input": ("native", "starch"),
                            "type": "technosphere",
                            "amount": 0.9,
                            "uncertainty type": 2,
                            "loc": -0.2,
                            "scale": 0.11,
                            "minimum": 0.4,
                            "maximum": 1.5,
                            "negative": False,
                            "allocation": "native starch input",
                            "activity": "invented native starch producer",
                            "flow": "invented maize starch product",
                            "name": "maize starch",
                            "unit": "kilogram",
                            "chemical formula": "(C6H10O5)n",
                            "properties": {
                                "carbon content": {"amount": 0.44},
                                "carbon content, fossil": {"amount": 0.0},
                                "carbon content, non-fossil": {"amount": 0.44},
                                "dry mass": {"amount": 0.86},
                            },
                        },
                        {
                            "input": ("native", "starch"),
                            "type": "technosphere",
                            "amount": 0.0,
                            "uncertainty type": 0,
                            "loc": 0.0,
                            "allocation": "explicit zero",
                        },
                        *[
                            {
                                "input": ("biosphere", "co2"),
                                "type": "biosphere",
                                "amount": amount,
                                "uncertainty type": 3,
                                "loc": amount,
                                "scale": 0.01,
                            }
                            for amount in (0.07, -0.03, 0.0)
                        ],
                    ],
                },
            },
            searchable=False,
        )
        yield bd, provider


@pytest.mark.parametrize(
    ("amount", "fields"),
    [
        (0.0, {}),
        (0.8999999761581421, {}),
        (0.8999999761581421, {"uncertainty type": 0, "loc": 0.8999999761581421}),
        (0.9, {"uncertainty type": 1, "loc": 0.9}),
        (
            0.9,
            {
                "uncertainty type": 2,
                "loc": -0.2,
                "scale": 0.11,
                "minimum": 0.4,
                "maximum": 1.5,
                "negative": False,
            },
        ),
        (
            0.9,
            {
                "uncertainty type": 3,
                "uncertainty_type": 2,
                "loc": -0.2,
                "scale": 0.11,
                "negative": False,
            },
        ),
    ],
    ids=(
        "zero_feed",
        "unspecified",
        "premise_deterministic",
        "constant",
        "lognormal",
        "native_alias",
    ),
)
def test_synthetic_glucose_preserves_native_records_and_scales_changed_feed(
    glucose_operation: tuple[Any, PreparedProviderAuthority],
    amount: float,
    fields: dict[str, object],
) -> None:
    """Preserve nonunit production and duplicate records across compilation and LCI."""
    bd, provider = glucose_operation
    native = bd.get_node(database="native", code="glucose")
    feed = next(exchange for exchange in native.technosphere() if exchange["amount"] != 0)
    for field in ("uncertainty type", "loc", "scale", "minimum", "maximum", "negative"):
        if field in feed:
            del feed[field]
    feed["amount"] = amount
    for field, value in fields.items():
        feed[field] = value
    feed.save()
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
        code=provider_service_code("glucose_fossil"),
    )
    # The prospective feed's dry glucan fraction excludes retained protein and
    # reagents; biochemical route tests independently check that composition.
    multiplier = 1.169666547771298
    expected = []
    for source in original:
        record = {key: value for key, value in source.items() if key != "output"}
        if record["type"] == "production":
            record["input"] = generated.key
            record.pop("activity")
            record["properties"] = dict(record["properties"])
            record["properties"].pop("carbon content, fossil")
            record["properties"].pop("carbon content, non-fossil")
        elif record["type"] == "technosphere":
            record["input"] = (
                frontier.database_name,
                provider_service_code("starch_delivered_fossil"),
            )
            record.pop("activity", None)
            for field in ("flow", "chemical formula", "properties"):
                record.pop(field, None)
            if "name" in record:
                record["name"] = "synthetic amylose feed, 14% water"
            record["amount"] *= multiplier
            if "loc" in record:
                distribution = record.get("uncertainty_type", record.get("uncertainty type", 0))
                record["loc"] = (
                    record["loc"] + math.log(multiplier)
                    if distribution == 2
                    else record["loc"] * multiplier
                )
            for field in ("minimum", "maximum"):
                if field in record:
                    record[field] *= multiplier
        expected.append(record)
    actual = [
        {key: value for key, value in dict(exchange).items() if key != "output"}
        for exchange in generated.exchanges()
    ]
    assert sorted(actual, key=repr) == sorted(expected, key=repr)
    assert [dict(exchange) for exchange in native.exchanges()] == original
    if fields.get("uncertainty_type", fields.get("uncertainty type")) == 2:
        source_parameters = UncertaintyBase.from_dicts(fields)
        scaled = next(row for row in actual if row["type"] == "technosphere" and row["amount"] != 0)
        generated_parameters = UncertaintyBase.from_dicts(scaled)
        quantiles = np.array([[0.1, 0.5, 0.9]])
        np.testing.assert_allclose(
            LognormalUncertainty.ppf(generated_parameters, quantiles),
            multiplier * LognormalUncertainty.ppf(source_parameters, quantiles),
            rtol=2e-15,
            atol=0,
        )
    calculation = LCA({generated: 1.0})
    calculation.lci()
    starch = bd.get_node(database="native", code="synthetic_starch")
    assert cast(NDArray[np.float64], calculation.supply_array)[
        calculation.dicts.activity[starch.id]
    ] == pytest.approx(
        float(np.float32(amount * multiplier)) / 2.0,
        rel=1e-14,
        abs=0,
    )
    emissions = float(np.float32(np.float32(0.07) + np.float32(-0.03))) / 2.0
    assert float(calculation.inventory.sum()) == pytest.approx(emissions, rel=1e-14, abs=0)


def test_biogenic_glucose_retains_complete_native_manufacture_and_delivery(
    glucose_operation: tuple[Any, PreparedProviderAuthority],
) -> None:
    """Keep native manufacture, the GLO supplier mix and freight unchanged in LCI."""
    bd, _provider = glucose_operation
    for code, name, product, unit, location, exchanges in (
        (
            "glucose_row",
            "glucose production",
            "glucose",
            "kilogram",
            "RoW",
            [("biosphere", "co2_non_fossil", "biosphere", 0.05)],
        ),
        (
            "freight",
            "invented glucose freight",
            "freight",
            "ton kilometer",
            "GLO",
            [("biosphere", "co2", "biosphere", 0.2)],
        ),
        (
            "market",
            "market for glucose",
            "glucose",
            "kilogram",
            "GLO",
            [
                ("native", "glucose", "technosphere", 0.75),
                ("native", "glucose_row", "technosphere", 2.25),
                ("native", "freight", "technosphere", 0.6),
            ],
        ),
    ):
        node = bd.Database("native").new_node(
            code=code, name=name, unit=unit, location=location, type="process"
        )
        node["reference product"] = product
        node.save()
        node.new_exchange(
            input=node.key, type="production", amount=3.0 if code == "market" else 1.0
        ).save()
        for database, input_code, kind, amount in exchanges:
            node.new_exchange(input=(database, input_code), type=kind, amount=amount).save()
    before = {node.key: [dict(ex) for ex in node.exchanges()] for node in bd.Database("native")}
    catalogue = load_provider_catalogue()
    selected = {"glucose_biogenic": "glucose", "glucose_delivered_biogenic": "market"}
    provider = PreparedProviderAuthority(
        identity="bounded_native_glucose",
        interface_database="bounded_interface",
        catalogue={
            **catalogue,
            "services": [
                s
                for s in cast(list[dict[str, Any]], catalogue["services"])
                if s["provider_service_id"] in selected
            ],
            "routes": [],
        },
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
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
    for service_id, code in selected.items():
        native = bd.get_node(database="native", code=code)
        generated = bd.get_node(
            database=frontier.database_name, code=provider_service_code(service_id)
        )
        assert generated.rp_exchange()["amount"] == 1.0
        assert [(ex.input.key, ex["amount"]) for ex in generated.technosphere()] == [
            (native.key, 1.0)
        ]
        native_lca, generated_lca = LCA({native: 1.0}), LCA({generated: 1.0})
        native_lca.lci()
        generated_lca.lci()
        for node in bd.Database("native"):
            assert (
                cast(NDArray[np.float64], generated_lca.supply_array)[
                    generated_lca.dicts.activity[node.id]
                ]
                == (
                    cast(NDArray[np.float64], native_lca.supply_array)[
                        native_lca.dicts.activity[node.id]
                    ]
                )
            )
        assert float(generated_lca.inventory.sum()) == float(native_lca.inventory.sum())
    assert {node.key: [dict(ex) for ex in node.exchanges()] for node in bd.Database("native")} == (
        before
    )


@pytest.mark.parametrize("product", ("styrene", "acrylonitrile"))
@pytest.mark.parametrize("original_properties", (True, False), ids=("original", "premise"))
def test_fossil_delivered_products_preserve_native_carbon_properties(
    glucose_operation: tuple[Any, PreparedProviderAuthority],
    product: str,
    original_properties: bool,
) -> None:
    """Preserve fossil product metadata when compiling synthetic delivery inventories."""
    bd, _provider = glucose_operation
    catalogue = load_provider_catalogue()
    service_id = f"{product}_delivered_fossil"
    target_id = f"{product}_fossil"
    selected = {service_id, target_id}
    services = [
        s
        for s in cast(list[dict[str, Any]], catalogue["services"])
        if s["provider_service_id"] in selected
    ]
    service = next(s for s in services if s["provider_service_id"] == service_id)
    construction = service["construction"]
    metadata = (
        {
            "activity": "invented native producer",
            "flow": "invented native product",
            "name": product,
            "reference product": product,
            "unit": "kilogram",
            "chemical formula": "C8H8" if product == "styrene" else "C3H3N",
            "properties": {
                "carbon content": {"amount": 0.9},
                "carbon content, fossil": {"amount": 0.9},
                "carbon content, non-fossil": {"amount": 0.0},
                "dry mass": {"amount": 1.0},
            },
        }
        if original_properties
        else {}
    )
    nodes = {}
    source_rows = []
    for index, rule in enumerate(construction["replace_technosphere"]):
        selector = rule["selector"]
        key = ("delivered_native", f"producer_{index}")
        nodes[key] = {
            "name": selector["name"],
            "reference product": product,
            "unit": "kilogram",
            "location": selector["location"],
            "type": "process",
            "exchanges": [
                {"input": key, "type": "production", "amount": 2.0},
                {"input": ("biosphere", "co2"), "type": "biosphere", "amount": 0.4},
            ],
        }
        source_rows.append(
            {
                **metadata,
                "input": key,
                "type": "technosphere",
                "amount": (1.2, 0.8)[index],
                "uncertainty type": 2,
                "loc": (-0.2, -0.4)[index],
                "scale": 0.11,
                "formula": f"native_supplier_share_{index}",
            }
        )
    base = construction["base"]["selector"]
    market_key = ("delivered_native", "market")
    nodes[market_key] = {
        "name": base["name"],
        "reference product": product,
        "unit": "kilogram",
        "location": base["exact_location"],
        "type": "process",
        "exchanges": [
            {**metadata, "input": market_key, "type": "production", "amount": 2.0},
            *source_rows,
        ],
    }
    bd.Database("delivered_native").write(nodes, searchable=False)
    original = [dict(ex) for ex in bd.get_node(key=market_key).exchanges()]
    provider = PreparedProviderAuthority(
        identity="bounded_fossil_delivery_metadata",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": services, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    frontier = prepare_provider_frontier(
        bd,
        background_scenario="remind_ssp2_npi",
        year=2030,
        ordinary_database="delivered_native",
        technology_database="delivered_native",
        material_recipe=physical_material_recipe(),
        provider=provider,
        resolver_cache=_NativeResolverCache("delivered_native", "biosphere"),
    )
    generated = bd.get_node(database=frontier.database_name, code=provider_service_code(service_id))
    expected = []
    for source in original:
        row = dict(source)
        row["output"] = generated.key
        row["input"] = (
            generated.key
            if row["type"] == "production"
            else (frontier.database_name, provider_service_code(target_id))
        )
        row.pop("activity", None)
        expected.append(row)
    actual = [dict(ex) for ex in generated.exchanges()]
    assert sorted(actual, key=repr) == sorted(expected, key=repr)
    assert [dict(ex) for ex in bd.get_node(key=market_key).exchanges()] == original
    calculation = LCA({generated: 1.0})
    calculation.lci()
    # Both market rows now share a supplier. The stored float32 coefficients
    # aggregate in that dtype before entering Brightway's float64 matrix.
    expected_inventory = (
        float(np.float32(np.float32(1.2) + np.float32(0.8))) / 2.0 * float(np.float32(0.4)) / 2.0
    )
    assert float(calculation.inventory.sum()) == pytest.approx(expected_inventory, rel=1e-14)


@pytest.mark.parametrize(
    ("service_id", "target_id", "target_product", "multiplier"),
    [
        (
            "glucose_fossil",
            "starch_delivered_fossil",
            "synthetic amylose feed, 14% water",
            1.169666547771298,
        ),
        (
            "glucose_atmospheric_dac",
            "starch_delivered_atmospheric_dac",
            "synthetic amylose feed, 14% water",
            1.169666547771298,
        ),
        (
            "acetylene_biogenic",
            "biomethane_high_pressure_biogenic",
            "biomethane, high pressure",
            1.0,
        ),
        (
            "acetylene_atmospheric_dac",
            "methane_delivered_atmospheric_dac_standard_volume",
            "methane at stated standard conditions",
            1.0,
        ),
    ],
)
def test_changed_feed_grades_do_not_inherit_donor_chemistry(
    glucose_operation: tuple[Any, PreparedProviderAuthority],
    service_id: str,
    target_id: str,
    target_product: str,
    multiplier: float,
) -> None:
    """Compile the glucose and gas declarations with their specified supplier links."""
    bd, _provider = glucose_operation
    catalogue = load_provider_catalogue()
    services = {
        s["provider_service_id"]: s for s in cast(list[dict[str, Any]], catalogue["services"])
    }
    service = services[service_id]
    construction = service["construction"]
    base = construction["base"]["selector"]
    feed = construction["replace_technosphere"][0]["selector"]
    source_feed = {
        "input": ("grade_native", "feed"),
        "type": "technosphere",
        "amount": 0.9,
        "uncertainty type": 2,
        "loc": -0.2,
        "scale": 0.11,
        "minimum": 0.4,
        "maximum": 1.5,
        "activity": "invented donor producer",
        "flow": "invented donor product",
        "name": feed["reference_product"],
        "reference product": feed["reference_product"],
        "unit": feed["unit"],
        "CAS number": "invented donor CAS",
        "chemical formula": "invented donor composition",
        "properties": {
            "carbon content": {"amount": 0.6},
            "carbon content, fossil": {"amount": 0.6},
            "carbon content, non-fossil": {"amount": 0.0},
            "dry mass": {"amount": 1.0},
        },
    }
    if multiplier == 1.0:
        source_feed["formula"] = "source_gas_requirement"
    nodes = {}
    for code, selector in (("operation", base), ("feed", feed)):
        nodes["grade_native", code] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector.get("exact_location", "RER"),
            "type": "process",
            "exchanges": [
                {"input": ("grade_native", code), "type": "production", "amount": 2.0},
                *([source_feed] if code == "operation" else []),
            ],
        }
    if multiplier == 1.0:
        group = construction["replace_technosphere"][1]["selector"]
        nodes["grade_native", "group"] = {
            "name": group["name"],
            "reference product": group["reference_product"],
            "unit": group["unit"],
            "location": "RER",
            "type": "process",
            "exchanges": [
                {"input": ("grade_native", "group"), "type": "production", "amount": 1.0}
            ],
        }
        nodes["grade_native", "operation"]["exchanges"].extend(
            [
                {**source_feed, "input": ("grade_native", "group")},
                {"input": ("biosphere", "co2"), "type": "biosphere", "amount": 0.1},
            ]
        )
    bd.Database("grade_native").write(nodes, searchable=False)
    native = bd.get_node(database="grade_native", code="operation")
    original = [dict(ex) for ex in native.exchanges()]
    resolver = _ActivityResolver(bd, "grade_native")
    context = _RecipeContext(
        background_scenario="remind_ssp2_npi",
        year=2030,
        provider_database="providers",
        source=resolver,
        ordinary=resolver,
        technology=resolver,
        biosphere=_FlowResolver(bd, "biosphere"),
        catalogue=catalogue,
        material_recipe=physical_material_recipe(),
        method_authorities={
            cast(str, r["method_authority_id"]): r for r in _method_authority_rows(catalogue)
        },
        complete_source_ids=_service_source_closures(catalogue),
        semantic_rows=[],
        technology_rows=[],
        traced_services=set(),
    )
    generated = _operation_only_dataset(context, service, services, "providers")
    rows = [
        ex
        for ex in cast(list[dict[str, Any]], generated["exchanges"])
        if ex["type"] == "technosphere"
    ]
    assert len(rows) == (2 if multiplier == 1.0 else 1)
    expected = {
        key: value
        for key, value in source_feed.items()
        if key not in {"activity", "flow", "CAS number", "chemical formula", "properties", "output"}
    }
    expected.update(
        {
            "input": ("providers", provider_service_code(target_id)),
            "name": target_product,
            "reference product": target_product,
            "amount": pytest.approx(0.9 * multiplier),
            "loc": pytest.approx(-0.2 + math.log(multiplier)),
            "minimum": pytest.approx(0.4 * multiplier),
            "maximum": pytest.approx(1.5 * multiplier),
        }
    )
    assert all(row == expected for row in rows)
    assert cast(list[dict[str, Any]], generated["exchanges"])[0]["amount"] == 2.0
    assert [dict(ex) for ex in native.exchanges()] == original


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"uncertainty type": 3}, "Composition scaling requires"),
        ({"formula": "unowned_feed_parameter"}, "without a live formula"),
        (
            {"amount": float.fromhex("0x1.fffffffffffffp+1023")},
            "exceeds the finite numerical range",
        ),
        ({"loc": math.nan}, "finite location"),
        ({"scale": 0.0}, "positive logarithmic spread"),
    ],
)
def test_glucose_admits_scaled_external_uncertainty_at_its_owner(
    glucose_operation: tuple[Any, PreparedProviderAuthority],
    changes: dict[str, object],
    message: str,
) -> None:
    """Reject unsupported or corrupt source metadata before publishing a frontier."""
    bd, provider = glucose_operation
    native = bd.get_node(database="native", code="glucose")
    feed = next(exchange for exchange in native.technosphere() if exchange["amount"] != 0)
    for field, value in changes.items():
        feed[field] = value
    feed.save()
    before = set(bd.databases)
    with pytest.raises(ValueError, match=message):
        prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2030,
            ordinary_database="native",
            technology_database="native",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
    assert set(bd.databases) == before


@pytest.mark.parametrize("production", [0.0, -1.0])
def test_operation_requires_positive_native_production(
    glucose_operation: tuple[Any, PreparedProviderAuthority],
    production: float,
) -> None:
    """Reject an edited source whose reference output cannot define the service."""
    bd, provider = glucose_operation
    native = bd.get_node(database="native", code="glucose")
    output = next(iter(native.production()))
    output["amount"] = production
    output.save()
    with pytest.raises(ValueError, match="positive self-production amount is required"):
        prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2030,
            ordinary_database="native",
            technology_database="native",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
