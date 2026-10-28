"""PET recipients retain complete glycol and terephthalate delivery functions."""

from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class _FeedMarket:
    """Admitted product, producer geography and complete freight-row identities."""

    product: str
    market_location: str
    delivery_stem: str
    manufacturer_stem: str
    producer_name: str
    producer_locations: tuple[str, ...]
    freight: tuple[tuple[str, str], ...]


_ROW_FREIGHT = (
    ("barge", "RoW"),
    ("lorry", "ZA"),
    ("lorry", "BR"),
    ("lorry", "RoW"),
    ("container ship", "GLO"),
    ("rail", "ZA"),
    ("rail", "RoW"),
    ("rail", "CN"),
    ("rail", "US"),
    ("rail", "IN"),
)
_GLYCOL_PRODUCER = "ethylene glycols production, thermal hydrolysis of ethylene oxide"
_MARKETS = {
    "eg_row": _FeedMarket(
        "ethylene glycol",
        "RoW",
        "ethylene_glycol_delivered_row",
        "ethylene_glycol",
        _GLYCOL_PRODUCER,
        ("RoW",),
        _ROW_FREIGHT,
    ),
    "eg_rer": _FeedMarket(
        "ethylene glycol",
        "RER",
        "ethylene_glycol_delivered_rer",
        "ethylene_glycol",
        _GLYCOL_PRODUCER,
        ("RER",),
        (("barge", "RER"), ("lorry", "RER"), ("rail", "RER")),
    ),
    "deg_row": _FeedMarket(
        "diethylene glycol",
        "RoW",
        "diethylene_glycol_delivered_row",
        "diethylene_glycol",
        _GLYCOL_PRODUCER,
        ("RoW",),
        _ROW_FREIGHT,
    ),
    "pta_glo": _FeedMarket(
        "purified terephthalic acid",
        "GLO",
        "purified_terephthalic_acid_delivered",
        "purified_terephthalic_acid",
        "purified terephthalic acid production",
        ("RER", "CA-QC", "RoW"),
        (("lorry", "GLO"), ("container ship", "GLO"), ("rail", "GLO")),
    ),
}
# Invented recipient quantities distinguish the two real incoming-market links.
_PET_FEEDS = {
    "pet": (("eg_row", 2.5), ("deg_row", 0.125), ("pta_glo", 6.0)),
    "pet_amorphous": (("eg_rer", 2.0), ("pta_glo", 5.0)),
}


@pytest.mark.parametrize("origin", ["biogenic", "atmospheric_dac"])
@pytest.mark.parametrize(("year", "ordinary", "factor"), [(2025, "native", 1), (2050, "dated", 2)])
def test_pet_feed_markets_preserve_delivery_and_recipient_vectors(
    tmp_path: Path, origin: str, year: int, ordinary: str, factor: float
) -> None:
    """Compile both PET grades with complete markets and independently solved freight.

    Quantities are invented. Actual recipient and delivery declarations run
    unchanged; the three upstream chemical manufacturers stop at valid native
    fixture supplies. All 26 admitted freight-row positions and the tiny
    PTA supplier survive. Nonunit outputs and producer totals unequal to market
    output expose flattening, normalization or lost source uncertainty.
    """
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    selected = [services[f"{stem}_{origin}"] for stem in _PET_FEEDS]
    selected.extend(services[f"{market.delivery_stem}_{origin}"] for market in _MARKETS.values())
    manufacturers = {market.manufacturer_stem: market.product for market in _MARKETS.values()}
    for stem, product in manufacturers.items():
        service_id = f"{stem}_{origin}"
        selected.append(
            {
                **services[service_id],
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": {
                        "name": f"admitted {service_id}",
                        "reference_product": product,
                        "unit": "kilogram",
                        "exact_location": "RER",
                    },
                    "amount": 1.0,
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity="invented_pet_feed_delivery",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )

    def inventory(database: str, multiplier: float) -> dict[tuple[str, str], dict[str, Any]]:
        """Build native-shaped recipient, market and independently emitting supplies."""

        def record(code: str, amount: float, *, kind: str = "technosphere") -> dict[str, Any]:
            """Keep a distinct native amount, distribution and provenance per row."""
            return {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "type": kind,
                "amount": amount,
                "uncertainty type": 0 if amount < 1e-10 else 2,
                "loc": amount if amount < 1e-10 else log(amount),
                **({} if amount < 1e-10 else {"scale": 0.125}),
                "comment": f"invented source {code}, quantity {amount}",
            }

        data = {}
        for index, (stem, product) in enumerate(manufacturers.items(), start=1):
            code = f"{stem}_{origin}"
            data[database, code] = _process(
                database,
                code,
                f"admitted {code}",
                product,
                production_amount=2 * multiplier,
                exchanges=[record(f"chemical_{index}", index * multiplier, kind="biosphere")],
            )
        for market_code, market in _MARKETS.items():
            for index, location in enumerate(market.producer_locations):
                code = f"{market_code}_producer_{index}"
                data[database, code] = _process(
                    database,
                    code,
                    market.producer_name,
                    market.product,
                    location=location,
                    exchanges=[record("unreplaced_fossil", 16, kind="biosphere")],
                )
            producer_amounts = (
                (1.0, 1e-12, 3.5 if multiplier == 1 else 3.75)
                if market_code == "pta_glo"
                else (5.0 if multiplier == 1 else 5.5,)
            )
            freight_rows = []
            for index, (mode, location) in enumerate(market.freight, start=1):
                code = f"{market_code}_freight_{index}"
                data[database, code] = _process(
                    database,
                    code,
                    f"transport, freight, {mode}",
                    f"transport, freight, {mode}",
                    location=location,
                    exchanges=[record("freight", index / 4, kind="biosphere")],
                )
                data[database, code]["unit"] = "ton kilometer"
                freight_rows.append(record(code, index * multiplier / 64))
            data[database, market_code] = _process(
                database,
                market_code,
                f"market for {market.product}",
                market.product,
                location=market.market_location,
                exchanges=[
                    *[
                        record(f"{market_code}_producer_{index}", amount * multiplier)
                        for index, amount in enumerate(producer_amounts)
                    ],
                    *freight_rows,
                ],
            )
            data[database, market_code]["exchanges"][0] = record(
                market_code, 4 * multiplier, kind="production"
            )
        for stem, feeds in _PET_FEEDS.items():
            grade = "bottle grade" if stem == "pet" else "amorphous"
            data[database, stem] = _process(
                database,
                stem,
                f"polyethylene terephthalate production, granulate, {grade}",
                f"polyethylene terephthalate, granulate, {grade}",
                location="RoW" if stem == "pet" else "RER",
                exchanges=[
                    *[record(code, amount * multiplier) for code, amount in feeds],
                    record("pet_operation", multiplier / 2, kind="biosphere"),
                ],
            )
            data[database, stem]["exchanges"][0] = record(
                stem, (8 if stem == "pet" else 4) * multiplier, kind="production"
            )
        return data

    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="pet_feed_delivery",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": f"invented {code} inventory witness",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
                for code in (
                    "chemical_1",
                    "chemical_2",
                    "chemical_3",
                    "freight",
                    "pet_operation",
                    "unreplaced_fossil",
                )
            },
            searchable=False,
        )
        for database, multiplier in (("native", 1.0), ("dated", 2.0)):
            bd.Database(database).write(inventory(database, multiplier), searchable=False)
        original = {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for database in ("native", "dated")
            for node in bd.Database(database)
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
        deliveries = {
            code: bd.get_node(
                database=frontier.database_name,
                code=provider_service_code(f"{market.delivery_stem}_{origin}"),
            )
            for code, market in _MARKETS.items()
        }
        for code, market in _MARKETS.items():
            delivery = deliveries[code]
            manufacturer_key = (
                frontier.database_name,
                provider_service_code(f"{market.manufacturer_stem}_{origin}"),
            )
            producer_keys = {
                (ordinary, f"{code}_producer_{index}")
                for index in range(len(market.producer_locations))
            }
            expected = [
                {
                    **row,
                    "output": delivery.key,
                    "input": (
                        delivery.key
                        if row["type"] == "production"
                        else manufacturer_key
                        if row["input"] in producer_keys
                        else row["input"]
                    ),
                }
                for row in original[ordinary, code]
            ]
            assert sorted((dict(row) for row in delivery.exchanges()), key=repr) == sorted(
                expected, key=repr
            )
        for stem, feeds in _PET_FEEDS.items():
            recipient = bd.get_node(
                database=frontier.database_name, code=provider_service_code(f"{stem}_{origin}")
            )
            replacements = {(ordinary, code): deliveries[code].key for code, _amount in feeds}
            expected = [
                {
                    **row,
                    "output": recipient.key,
                    "input": (
                        recipient.key
                        if row["type"] == "production"
                        else replacements.get(row["input"], row["input"])
                    ),
                }
                for row in original[ordinary, stem]
            ]
            assert sorted((dict(row) for row in recipient.exchanges()), key=repr) == sorted(
                expected, key=repr
            )
            calculation = LCA({recipient: 1.0})
            calculation.lci()
            product_output = 8 if stem == "pet" else 4
            expected_emissions = {
                "chemical_1": 0.0,
                "chemical_2": 0.0,
                "chemical_3": 0.0,
                "freight": 0.0,
                "pet_operation": 0.5 / product_output,
                "unreplaced_fossil": 0.0,
            }
            for code, amount in feeds:
                market = _MARKETS[code]
                delivery_demand = amount / product_output
                # The market supplies more producer mass than its own output;
                # preserve that source coefficient instead of normalizing it.
                product_ratio = (
                    (1.0 + 1e-12 + (3.5 if factor == 1 else 3.75)) / 4
                    if code == "pta_glo"
                    else (5.0 if factor == 1 else 5.5) / 4
                )
                index = list(manufacturers).index(market.manufacturer_stem) + 1
                expected_emissions[f"chemical_{index}"] += (
                    delivery_demand * product_ratio * index / 2
                )
                for freight_index in range(1, len(market.freight) + 1):
                    freight = bd.get_node(database=ordinary, code=f"{code}_freight_{freight_index}")
                    freight_demand = delivery_demand * freight_index / 256
                    assert calculation.supply_array[
                        calculation.dicts.activity[freight.id]
                    ] == pytest.approx(freight_demand)
                    expected_emissions["freight"] += freight_demand * freight_index / 4
            for code, amount in expected_emissions.items():
                flow = bd.get_node(database="biosphere", code=code)
                observed = float(
                    calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum()
                )
                assert observed == pytest.approx(amount)
        assert {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for database in ("native", "dated")
            for node in bd.Database(database)
        } == original
