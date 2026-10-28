"""Delivered PBR retains market support around the actual manufacturer recipe."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe

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


def _native_pbr_inventory(database: str, dated_factor: float) -> dict[tuple[str, str], Any]:
    """Return invented manufacture, three freight modes and bounded upstream supplies.

    All quantities are invented. The upstream butadiene endpoints bound this test
    at admitted feed supply; the actual PBR manufacturer and market recipes are
    compiled unchanged. Solvent treatment belongs to manufacture, outbound freight
    to the market. The substituted butadiene endpoint supplies the feed only.
    """
    data = {
        (database, code): {
            "name": name,
            "reference product": product,
            "unit": unit,
            "location": "RER",
            "type": "process",
            "exchanges": [
                {"input": (database, code), "type": "production", "amount": production},
                {"input": ("biosphere", flow), "type": "biosphere", "amount": emission},
            ],
        }
        for code, name, product, unit, production, flow, emission in (
            ("butadiene", "market for butadiene", "butadiene", "kilogram", 1, "fossil", 8),
            (
                "feed_biogenic",
                "admitted biogenic butadiene feed",
                "butadiene",
                "kilogram",
                2,
                "nonfossil",
                dated_factor,
            ),
            (
                "feed_atmospheric_dac",
                "admitted DAC butadiene feed",
                "butadiene",
                "kilogram",
                2,
                "nonfossil",
                1.5 * dated_factor,
            ),
            ("utility", "process electricity", "electricity", "kilowatt hour", 1, "fossil", 3),
            (
                "solvent_treatment",
                "solvent waste treatment",
                "solvent waste",
                "kilogram",
                1,
                "fossil",
                4,
            ),
            ("barge", "freight, barge", "freight", "ton kilometer", 1, "fossil", 0.5),
            ("lorry", "freight, lorry", "freight", "ton kilometer", 1, "fossil", 1),
            ("rail", "freight, rail", "freight", "ton kilometer", 1, "fossil", 2),
        )
    }
    data[database, "manufacturer"] = {
        "name": "polybutadiene production, solution polymerization",
        "reference product": "polybutadiene",
        "unit": "kilogram",
        "location": "RER",
        "type": "process",
        "exchanges": [
            {
                "input": (database, "manufacturer"),
                "type": "production",
                "amount": 2.0,
                "uncertainty type": 3,
                "loc": 2.0,
                "scale": 0.125,
                "allocation": "invented native manufacturer reference output",
            },
            {
                "input": (database, "butadiene"),
                "type": "technosphere",
                "amount": 2.5,
                "uncertainty type": 2,
                "loc": 0.75,
                "scale": 0.125,
                "comment": "invented allocated feed",
            },
            {"input": (database, "butadiene"), "type": "technosphere", "amount": 0.0},
            {
                "input": (database, "utility"),
                "type": "technosphere",
                "amount": 0.5 * dated_factor,
            },
            {
                "input": (database, "solvent_treatment"),
                "type": "technosphere",
                "amount": 0.25,
                "comment": "native fossil solvent treatment remains within manufacture",
            },
            {
                "input": ("biosphere", "fossil"),
                "type": "biosphere",
                "amount": 0.125,
                "uncertainty type": 3,
                "loc": 0.125,
                "scale": 0.015625,
                "comment": "invented direct butadiene-loss carbon",
            },
        ],
    }
    data[database, "market"] = {
        "name": "market for polybutadiene",
        "reference product": "polybutadiene",
        "unit": "kilogram",
        "location": "RER",
        "type": "process",
        "exchanges": [
            {
                "input": (database, "market"),
                "type": "production",
                "amount": 4.0,
                "uncertainty type": 3,
                "loc": 4.0,
                "scale": 0.25,
                "allocation": "invented native market reference output",
            },
            *[
                {
                    "input": (database, "manufacturer"),
                    "type": "technosphere",
                    "amount": amount,
                    "uncertainty type": 3,
                    "loc": amount,
                    "scale": 0.125,
                    "comment": label,
                }
                for amount, label in ((1.0, "first supply row"), (3.0, "second supply row"))
            ],
            {
                "input": (database, "manufacturer"),
                "type": "technosphere",
                "amount": 0.0,
                "uncertainty type": 0,
                "loc": 0.0,
                "comment": "explicit zero supply",
            },
            *[
                {
                    "input": (database, mode),
                    "type": "technosphere",
                    "amount": amount * dated_factor,
                    "unit": "ton kilometer",
                    "uncertainty type": 3,
                    "loc": amount * dated_factor,
                    "scale": 0.015625,
                    "comment": "invented native outbound freight",
                }
                for mode, amount in (("barge", 0.125), ("lorry", 0.25), ("rail", 0.5))
            ],
        ],
    }
    return data


@pytest.mark.parametrize("origin", ["biogenic", "atmospheric_dac"])
@pytest.mark.parametrize(
    ("year", "ordinary", "dated_factor"), [(2025, "native", 1), (2050, "dated", 2)]
)
def test_delivered_pbr_preserves_native_market_and_manufacturer_boundaries(
    tmp_path: Path, origin: str, year: int, ordinary: str, dated_factor: float
) -> None:
    """Normalize manufacture and delivery once, preserving native support and origins."""
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    producer_id = f"polybutadiene_{origin}"
    delivery_id = f"polybutadiene_delivered_{origin}"
    feed_id = f"butadiene_{origin}"
    feed_name = (
        "admitted biogenic butadiene feed"
        if origin == "biogenic"
        else "admitted DAC butadiene feed"
    )
    bounded_feed = {
        **services[feed_id],
        "location": "RER",
        "construction": {
            "kind": "native_activity",
            "source_role": "native_ecoinvent_provider",
            "selector": {
                "name": feed_name,
                "reference_product": "butadiene",
                "unit": "kilogram",
                "exact_location": "RER",
            },
            "amount": 1.0,
        },
    }
    provider = PreparedProviderAuthority(
        identity="invented_delivered_pbr",
        interface_database="bounded_interface",
        catalogue={
            **catalogue,
            "services": [services[producer_id], services[delivery_id], bounded_feed],
            "routes": [],
        },
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="pbr_delivery",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": name,
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
                for code, name in (
                    ("fossil", "Carbon dioxide, fossil"),
                    ("nonfossil", "Carbon dioxide, non-fossil"),
                )
            },
            searchable=False,
        )
        for database, factor in (("native", 1.0), ("dated", 2.0)):
            bd.Database(database).write(_native_pbr_inventory(database, factor), searchable=False)
        original = {
            activity.key: [dict(exchange) for exchange in activity.exchanges()]
            for database in ("native", "dated")
            for activity in bd.Database(database)
        }
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=ordinary,
            technology_database="dated",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
        producer = bd.get_node(
            database=frontier.database_name, code=provider_service_code(producer_id)
        )
        delivery = bd.get_node(
            database=frontier.database_name, code=provider_service_code(delivery_id)
        )
        feed_key = (frontier.database_name, provider_service_code(feed_id))
        for native_code, generated, replacements in (
            (
                "manufacturer",
                producer,
                {
                    (ordinary, "butadiene"): feed_key,
                    ("biosphere", "fossil"): ("biosphere", "nonfossil"),
                },
            ),
            ("market", delivery, {(ordinary, "manufacturer"): producer.key}),
        ):
            replacements[ordinary, native_code] = generated.key
            expected = [
                {
                    **{key: value for key, value in row.items() if key != "output"},
                    "input": replacements.get(row["input"], row["input"]),
                }
                for row in original[ordinary, native_code]
            ]
            # The direct carbon loss changes elementary-flow identity. Its name
            # and unit must describe that counterpart; quantity, uncertainty and
            # all unrelated native exchange fields remain equal to the source.
            for row in expected:
                if row["type"] == "biosphere":
                    row.update(name="Carbon dioxide, non-fossil", unit="kilogram")
            actual = [
                {key: value for key, value in dict(exchange).items() if key != "output"}
                for exchange in generated.exchanges()
            ]
            assert sorted(actual, key=repr) == sorted(expected, key=repr)
        for key, exchanges in original.items():
            assert [
                dict(exchange) for exchange in bd.get_node(database=key[0], code=key[1]).exchanges()
            ] == exchanges

        # Four kg per market batch, two kg per manufacturer batch; the linked
        # butadiene endpoint independently produces two kg per feed batch.
        for root, market_batches in ((producer, 0.0), (delivery, 1.0 / 4.0)):
            calculation = LCA({root: 1.0})
            calculation.lci()
            assert calculation.supply_array[
                calculation.dicts.activity[producer.id]
            ] == pytest.approx(0.5)
            native_manufacturer = bd.get_node(database=ordinary, code="manufacturer")
            assert calculation.supply_array[calculation.dicts.activity[native_manufacturer.id]] == 0
            feed = bd.get_node(database=ordinary, code=f"feed_{origin}")
            assert calculation.supply_array[calculation.dicts.activity[feed.id]] == pytest.approx(
                2.5 / 2 / 2
            )
            for mode, amount in (("barge", 0.125), ("lorry", 0.25), ("rail", 0.5)):
                freight = bd.get_node(database=ordinary, code=mode)
                assert calculation.supply_array[
                    calculation.dicts.activity[freight.id]
                ] == pytest.approx(amount * dated_factor * market_batches)
            fossil = bd.get_node(database="biosphere", code="fossil")
            nonfossil = bd.get_node(database="biosphere", code="nonfossil")
            expected_fossil = (0.5 * dated_factor * 3 + 0.25 * 4) / 2
            expected_fossil += (0.125 * 0.5 + 0.25 * 1 + 0.5 * 2) * dated_factor * market_batches
            feed_emission = (1.0 if origin == "biogenic" else 1.5) * dated_factor
            expected_nonfossil = (2.5 * feed_emission / 2 + 0.125) / 2
            assert float(
                calculation.inventory[calculation.dicts.biosphere[fossil.id], :].sum()
            ) == pytest.approx(expected_fossil)
            assert float(
                calculation.inventory[calculation.dicts.biosphere[nonfossil.id], :].sum()
            ) == pytest.approx(expected_nonfossil)


def test_hips_and_fossil_abs_select_complete_pbr_supply() -> None:
    """Bind HIPS and fossil ABS to complete delivery while keeping fossil HIPS native."""
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    for origin in ("biogenic", "atmospheric_dac"):
        delivery_id = f"polybutadiene_delivered_{origin}"
        pbr = next(
            row
            for row in services[f"hips_{origin}"]["construction"]["replacements"]
            if row["replacement_id"] == "polybutadiene"
        )
        assert pbr["service_id"] == delivery_id
    fossil_pbr = next(
        row
        for row in services["abs_fossil"]["construction"]["components"]
        if row.get("component_id") == "polybutadiene"
    )
    assert fossil_pbr == {
        "component_id": "polybutadiene",
        "source_role": "native_ecoinvent_provider",
        "selector": {
            "name": "market for polybutadiene",
            "reference_product": "polybutadiene",
            "unit": "kilogram",
            "exact_location": "RER",
        },
        "amount": 0.2,
    }
    assert services["hips_fossil"]["construction"] == {
        "kind": "native_activity",
        "source_role": "native_ecoinvent_provider",
        "selector": {
            "name": "polystyrene production, high impact",
            "reference_product": "polystyrene, high impact",
            "unit": "kilogram",
            "exact_location": "RER",
        },
        "amount": 1.0,
    }
