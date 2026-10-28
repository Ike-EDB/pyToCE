"""Selected monomer manufacture with complete native market delivery."""

from dataclasses import dataclass
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
class _Monomer:
    """Native producer identities and distinct distribution geography for one monomer."""

    product: str
    selected_manufacturer: str
    other_manufacturer: str
    other_location: str
    market_location: str
    feedstock: str
    direct_compartment: tuple[str, ...]
    gases: tuple[tuple[str, str, float], ...]
    freight: tuple[tuple[str, float, float], ...]


_STYRENE = _Monomer(
    product="styrene",
    selected_manufacturer="styrene production, from ethyl benzene dehydrogenation",
    other_manufacturer="styrene production, from ethyl benzene, PO/SM process",
    other_location="RER",
    market_location="RER",
    feedstock="ethyl benzene",
    direct_compartment=("air",),
    gases=(
        ("co2", "Carbon dioxide", 0.125),
        ("co", "Carbon monoxide", 0.015625),
        ("methane", "Methane", 0.0078125),
    ),
    freight=(("barge", 0.125, 0.5), ("lorry", 0.25, 1.0), ("rail", 0.5, 2.0)),
)
_ACRYLONITRILE = _Monomer(
    product="acrylonitrile",
    selected_manufacturer="acrylonitrile production, Sohio process",
    other_manufacturer="acrylonitrile production, Sohio process",
    other_location="RoW",
    market_location="GLO",
    feedstock="propylene",
    direct_compartment=("air", "urban air close to ground"),
    gases=(("co2", "Carbon dioxide", 0.125),),
    freight=(*_STYRENE.freight, ("sea", 0.75, 4.0)),
)


def _native_monomer_inventory(
    monomer: _Monomer, database: str, factor: float
) -> dict[tuple[str, str], dict[str, Any]]:
    """Invent a two-producer market with independently emitting supply boundaries.

    Quantities and distributions are synthetic. Ethyl-benzene/propylene feeds
    bound the upstream graph; the selected monomer manufacturer and delivery
    recipes use the package implementation. The other manufacturer's emissions
    expose any unconverted product share. Future supplier coefficients sum to
    9 kg per 8 kg output to check source production normalization.
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
            (
                "feed",
                f"market for {monomer.feedstock}",
                monomer.feedstock,
                "kilogram",
                1.0,
                "co2_fossil",
                4.0 * factor,
            ),
            (
                "feed_biogenic",
                f"admitted biogenic {monomer.feedstock}",
                monomer.feedstock,
                "kilogram",
                2.0,
                "co2_nonfossil",
                factor,
            ),
            (
                "feed_atmospheric_dac",
                f"admitted atmospheric_dac {monomer.feedstock}",
                monomer.feedstock,
                "kilogram",
                2.0,
                "co2_nonfossil",
                1.5 * factor,
            ),
            (
                "utility",
                "manufacturing electricity",
                "electricity",
                "kilowatt hour",
                1.0,
                "auxiliary_co2",
                2.0 * factor,
            ),
            *(
                (mode, f"freight, {mode}", "freight", "ton kilometer", 1.0, "auxiliary_co2", rate)
                for mode, _amount, rate in monomer.freight
            ),
        )
    }
    data[database, "other_manufacturer"] = {
        "name": monomer.other_manufacturer,
        "reference product": monomer.product,
        "unit": "kilogram",
        "location": monomer.other_location,
        "type": "process",
        "exchanges": [
            {"input": (database, "other_manufacturer"), "type": "production", "amount": 1.0},
            {"input": ("biosphere", "co2_fossil"), "type": "biosphere", "amount": 16.0 * factor},
        ],
    }
    data[database, "selected_manufacturer"] = {
        "name": monomer.selected_manufacturer,
        "reference product": monomer.product,
        "unit": "kilogram",
        "location": "RER",
        "type": "process",
        "exchanges": [
            {
                "input": (database, "selected_manufacturer"),
                "type": "production",
                "amount": 2.0 * factor,
                "uncertainty type": 3,
                "loc": 2.0 * factor,
                "scale": 0.125,
            },
            {"input": (database, "feed"), "type": "technosphere", "amount": 2.5 * factor},
            {"input": (database, "utility"), "type": "technosphere", "amount": 0.5 * factor},
            *[
                {
                    "input": ("biosphere", f"{code}_fossil"),
                    "type": "biosphere",
                    "amount": amount * factor,
                }
                for code, _name, amount in monomer.gases
            ],
        ],
    }
    data[database, "market"] = {
        "name": f"market for {monomer.product}",
        "reference product": monomer.product,
        "unit": "kilogram",
        "location": monomer.market_location,
        "type": "process",
        "exchanges": [
            {
                "input": (database, "market"),
                "type": "production",
                "amount": 4.0 * factor,
                "uncertainty type": 3,
                "loc": 4.0 * factor,
                "scale": 0.25,
                "allocation": "invented market reference output",
            },
            *[
                {
                    "input": (database, supplier),
                    "type": "technosphere",
                    "amount": amount * factor,
                    "uncertainty type": 3,
                    "loc": amount * factor,
                    "scale": spread,
                    "comment": f"invented source row for {supplier}",
                }
                for supplier, amount, spread in (
                    ("selected_manufacturer", 1.0, 0.125),
                    ("other_manufacturer", 3.0 if factor == 1 else 3.5, 0.25),
                )
            ],
            *[
                {
                    "input": (database, mode),
                    "type": "technosphere",
                    "amount": amount * factor**2,
                    "unit": "ton kilometer",
                }
                for mode, amount, _rate in monomer.freight
            ],
        ],
    }
    return data


@pytest.mark.parametrize("monomer", [_STYRENE, _ACRYLONITRILE], ids=["styrene", "acrylonitrile"])
@pytest.mark.parametrize("origin", ["fossil", "biogenic", "atmospheric_dac"])
@pytest.mark.parametrize(("year", "ordinary", "factor"), [(2025, "native", 1), (2050, "dated", 2)])
def test_complete_market_delivery_rebinds_both_native_monomer_producers(
    tmp_path: Path, monomer: _Monomer, origin: str, year: int, ordinary: str, factor: float
) -> None:
    """Retain selected manufacture, complete delivery, and the specified carbon origin."""
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    manufacturer_id = f"{monomer.product}_{origin}"
    delivered_id = f"{monomer.product}_delivered_{origin}"
    selected_services = [services[manufacturer_id], services[delivered_id]]
    if origin != "fossil":
        feed_id = (
            f"ethyl_benzene_delivered_{origin}"
            if monomer.product == "styrene"
            else f"propylene_{origin}"
        )
        selected_services.append(
            {
                **services[feed_id],
                "location": "RER",
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": {
                        "name": f"admitted {origin} {monomer.feedstock}",
                        "reference_product": monomer.feedstock,
                        "unit": "kilogram",
                        "exact_location": "RER",
                    },
                    "amount": 1.0,
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity="invented_selected_monomer_delivery",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected_services, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="monomer_delivery",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        biosphere = {
            ("biosphere", f"{code}_{suffix}"): {
                "name": f"{name}, {label}",
                "unit": "kilogram",
                "categories": monomer.direct_compartment,
                "type": "emission",
            }
            for code, name, _amount in monomer.gases
            for suffix, label in (("fossil", "fossil"), ("nonfossil", "non-fossil"))
        }
        biosphere["biosphere", "auxiliary_co2"] = {
            "name": "Carbon dioxide, fossil",
            "unit": "kilogram",
            "categories": ("air", "non-urban air or from high stacks"),
            "type": "emission",
        }
        bd.Database("biosphere").write(biosphere, searchable=False)
        for database, source_factor in (("native", 1.0), ("dated", 2.0)):
            bd.Database(database).write(
                _native_monomer_inventory(monomer, database, source_factor), searchable=False
            )
        native_market = bd.get_node(database=ordinary, code="market")
        source_rows = [dict(exchange) for exchange in native_market.exchanges()]
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
        manufacturer = bd.get_node(
            database=frontier.database_name, code=provider_service_code(manufacturer_id)
        )
        delivered = bd.get_node(
            database=frontier.database_name, code=provider_service_code(delivered_id)
        )
        source_product_rows = [
            row
            for row in source_rows
            if row["input"]
            in {(ordinary, "selected_manufacturer"), (ordinary, "other_manufacturer")}
        ]
        expected_product_rows = [
            {**row, "input": manufacturer.key, "output": delivered.key}
            for row in source_product_rows
        ]
        actual_product_rows = [
            dict(exchange)
            for exchange in delivered.technosphere()
            if exchange["input"] == manufacturer.key
        ]
        # Retain both uncertainty records when they select the same manufacturer.
        assert sorted(actual_product_rows, key=repr) == sorted(expected_product_rows, key=repr)
        source_production = next(row for row in source_rows if row["type"] == "production")
        assert dict(next(iter(delivered.production()))) == {
            **source_production,
            "input": delivered.key,
            "output": delivered.key,
        }
        assert [dict(exchange) for exchange in native_market.exchanges()] == source_rows

        calculation = LCA({delivered: 1.0})
        calculation.lci()
        product_demand = 1.0 if year == 2025 else 1.125
        other = bd.get_node(database=ordinary, code="other_manufacturer")
        assert calculation.supply_array[calculation.dicts.activity[other.id]] == 0
        native_selected = bd.get_node(database=ordinary, code="selected_manufacturer")
        assert calculation.supply_array[
            calculation.dicts.activity[native_selected.id]
        ] == pytest.approx(product_demand / (2 * factor) if origin == "fossil" else 0.0)
        assert calculation.supply_array[
            calculation.dicts.activity[manufacturer.id]
        ] == pytest.approx(product_demand if origin == "fossil" else product_demand / (2 * factor))
        for mode, amount, _rate in monomer.freight:
            freight = bd.get_node(database=ordinary, code=mode)
            assert calculation.supply_array[
                calculation.dicts.activity[freight.id]
            ] == pytest.approx(amount * factor / 4)
        auxiliary = bd.get_node(database="biosphere", code="auxiliary_co2")
        expected_auxiliary = factor * (
            0.5 * product_demand + sum(amount * rate for _, amount, rate in monomer.freight) / 4
        )
        assert float(
            calculation.inventory[calculation.dicts.biosphere[auxiliary.id], :].sum()
        ) == pytest.approx(expected_auxiliary)
        for code, _name, direct_amount in monomer.gases:
            suffix = "fossil" if origin == "fossil" else "nonfossil"
            gas = bd.get_node(database="biosphere", code=f"{code}_{suffix}")
            expected = direct_amount / 2
            if code == "co2":
                expected += (
                    1.25 * 4 * factor
                    if origin == "fossil"
                    else 1.25 / 2 * (1.0 if origin == "biogenic" else 1.5) * factor
                )
            assert float(
                calculation.inventory[calculation.dicts.biosphere[gas.id], :].sum()
            ) == pytest.approx(expected * product_demand)


@pytest.mark.parametrize(
    ("delivery_stem", "product", "market_location", "producers"),
    [
        (
            "benzene_delivered_rer",
            "benzene",
            "RER",
            (
                (
                    "p-diisopropylbenzene purification, distillation of crude p-diisopropylbenzene",
                    "RER",
                ),
                ("BTX production, from reformate, average", "RER"),
                ("coke production", "DE"),
                ("BTX production, from pyrolysis gas, average", "RER"),
            ),
        ),
        (
            "benzene_delivered_row",
            "benzene",
            "RoW",
            (
                ("BTX production, from pyrolysis gas, average", "RoW"),
                (
                    "p-diisopropylbenzene purification, distillation of crude p-diisopropylbenzene",
                    "RoW",
                ),
                ("coke production, dry quenching", "CN"),
                ("coke production, wet quenching", "CN"),
                ("coke production", "US"),
                ("coke production", "RoW"),
                ("BTX production, from reformate, average", "RoW"),
            ),
        ),
        (
            "ethyl_benzene_delivered",
            "ethyl benzene",
            "RER",
            tuple(
                ("ethyl benzene production, from benzene alkylation", location)
                for location in ("FR", "DE", "RoE", "CZ", "BE", "IT", "ES", "NL", "GB")
            ),
        ),
    ],
    ids=["benzene_rer", "benzene_row", "ethyl_benzene"],
)
@pytest.mark.parametrize("origin", ["biogenic", "atmospheric_dac"])
@pytest.mark.parametrize(("year", "ordinary", "factor"), [(2025, "native", 1), (2050, "dated", 2)])
def test_upstream_monomer_delivery_preserves_market_records_and_ethylene_feeds(
    tmp_path: Path,
    delivery_stem: str,
    product: str,
    market_location: str,
    producers: tuple[tuple[str, str], ...],
    origin: str,
    year: int,
    ordinary: str,
    factor: float,
) -> None:
    """Rebind receiving markets and retain each source row and coefficient.

    Invented quantities exercise the qualified four/seven/nine producer identities.
    Future producer totals differ from reference production; duplicate freight
    records have distinct uncertainty. Benzene manufacture and ethylene supply
    bound the graph, while the actual ethyl-benzene operation retains all eighteen
    native ethylene input records and its selected benzene delivery boundary.
    """
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    delivered_id = f"{delivery_stem}_{origin}"
    manufacturer_id = f"{product.replace(' ', '_')}_{origin}"
    selected_services = [services[delivered_id]]
    is_ethyl_benzene = product == "ethyl benzene"
    bounded_feeds = (
        ((f"benzene_delivered_row_{origin}", "benzene"), (f"ethylene_{origin}", "ethylene"))
        if is_ethyl_benzene
        else ((manufacturer_id, "benzene"),)
    )
    if is_ethyl_benzene:
        selected_services.append(services[manufacturer_id])
    for feed_id, feed_product in bounded_feeds:
        selected_services.append(
            {
                **services[feed_id],
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": {
                        "name": f"admitted {feed_id}",
                        "reference_product": feed_product,
                        "unit": "kilogram",
                        "exact_location": "RER",
                    },
                    "amount": 1.0,
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity="invented_upstream_monomer_delivery",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected_services, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    ethylene_locations = (
        "AE",
        "AR",
        "AU",
        "CA",
        "CN",
        "IL",
        "IN",
        "KR",
        "MY",
        "QA",
        "RoW",
        "SA",
        "SG",
        "TH",
        "TR",
        "TW",
        "VE",
        "ZA",
    )

    def inventory(database: str, multiplier: float) -> dict[tuple[str, str], dict[str, Any]]:
        """Supply the exact native identities with distinguishable invented records."""

        def record(code: str, amount: float, *, kind: str = "technosphere") -> dict[str, Any]:
            """Retain source metadata independently of the exchange amount."""
            return {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "type": kind,
                "amount": amount,
                "uncertainty type": 3,
                "loc": amount + 0.03125,
                "scale": 0.125,
                "comment": f"invented {code} source amount {amount}",
            }

        data = {
            (database, f"producer_{index}"): _process(
                database,
                f"producer_{index}",
                name,
                product,
                location=location,
                exchanges=[record("co2_fossil", 16 * multiplier, kind="biosphere")],
            )
            for index, (name, location) in enumerate(producers)
        }
        for feed_id, feed_product in bounded_feeds:
            data[database, feed_id] = _process(
                database,
                feed_id,
                f"admitted {feed_id}",
                feed_product,
                production_amount=2 * multiplier,
                exchanges=[record("co2_nonfossil", multiplier, kind="biosphere")],
            )
        for mode in ("road", "rail"):
            data[database, mode] = _process(
                database,
                mode,
                f"invented freight, {mode}",
                "freight",
                location="World",
                exchanges=[record("co2_fossil", 0.25, kind="biosphere")],
            )
            data[database, mode]["unit"] = "ton kilometer"
        product_rows = [
            record(f"producer_{index}", (index + 1) * multiplier) for index in range(len(producers))
        ]
        if multiplier != 1:
            product_rows[-1] = record(
                f"producer_{len(producers) - 1}", (len(producers) + 0.25) * multiplier
            )
        data[database, "market"] = _process(
            database,
            "market",
            f"market for {product}",
            product,
            location=market_location,
            exchanges=[
                *product_rows,
                record("road", 0.125 * multiplier),
                record("rail", 0.25 * multiplier),
                *([record("road", 0.375), record("rail", 0.5)] if multiplier != 1 else []),
            ],
        )
        data[database, "market"]["exchanges"][0] = record(
            "market", sum(range(1, len(producers) + 1)) * multiplier, kind="production"
        )
        if is_ethyl_benzene:
            data[database, "utility"] = _process(
                database,
                "utility",
                "invented alkylation electricity",
                "electricity",
                exchanges=[record("co2_fossil", 0.5, kind="biosphere")],
            )
            data[database, "utility"]["unit"] = "kilowatt hour"
            data[database, "benzene_feed"] = _process(
                database, "benzene_feed", "market for benzene", "benzene", location="RoW"
            )
            for location in ethylene_locations:
                code = f"ethylene_{location}"
                data[database, code] = _process(
                    database, code, "market for ethylene", "ethylene", location=location
                )
            data[database, "manufacturer"] = _process(
                database,
                "manufacturer",
                "ethyl benzene production, from benzene alkylation",
                "ethyl benzene",
                location="RoW",
                exchanges=[
                    record("benzene_feed", 2.5 * multiplier),
                    *[
                        record(f"ethylene_{location}", (index + 1) * multiplier / 64)
                        for index, location in enumerate(ethylene_locations)
                    ],
                    record("utility", 0.25 * multiplier),
                    record("co2_fossil", 0.125 * multiplier, kind="biosphere"),
                ],
            )
            data[database, "manufacturer"]["exchanges"][0] = record(
                "manufacturer", 2 * multiplier, kind="production"
            )
        return data

    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="upstream_monomer_delivery",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": f"Carbon dioxide, {label}",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
                for code, label in (("co2_fossil", "fossil"), ("co2_nonfossil", "non-fossil"))
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
        delivered = bd.get_node(
            database=frontier.database_name, code=provider_service_code(delivered_id)
        )
        manufacturer = bd.get_node(
            database=frontier.database_name, code=provider_service_code(manufacturer_id)
        )
        producer_keys = {(ordinary, f"producer_{index}") for index in range(len(producers))}
        market_rows = original[ordinary, "market"]
        expected = [
            {
                **row,
                "output": delivered.key,
                "input": (
                    delivered.key
                    if row["type"] == "production"
                    else manufacturer.key
                    if row["input"] in producer_keys
                    else row["input"]
                ),
            }
            for row in market_rows
        ]
        assert sorted((dict(exchange) for exchange in delivered.exchanges()), key=repr) == sorted(
            expected, key=repr
        )
        if is_ethyl_benzene:
            replacement_keys = {
                (ordinary, "manufacturer"): manufacturer.key,
                (ordinary, "benzene_feed"): (
                    frontier.database_name,
                    provider_service_code(f"benzene_delivered_row_{origin}"),
                ),
                ("biosphere", "co2_fossil"): ("biosphere", "co2_nonfossil"),
                **{
                    (ordinary, f"ethylene_{location}"): (
                        frontier.database_name,
                        provider_service_code(f"ethylene_{origin}"),
                    )
                    for location in ethylene_locations
                },
            }
            expected_manufacturer = [
                {
                    **row,
                    "input": replacement_keys.get(row["input"], row["input"]),
                    "output": manufacturer.key,
                }
                for row in original[ordinary, "manufacturer"]
            ]
            # Feed-origin substitution changes the direct gas identity and its
            # descriptive metadata, preserving every quantity and uncertainty.
            for row in expected_manufacturer:
                if row["type"] == "biosphere":
                    row.update(name="Carbon dioxide, non-fossil", unit="kilogram")
            assert sorted(
                (dict(exchange) for exchange in manufacturer.exchanges()), key=repr
            ) == sorted(expected_manufacturer, key=repr)
        assert {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for database in ("native", "dated")
            for node in bd.Database(database)
        } == original
        calculation = LCA({delivered: 1.0})
        calculation.lci()
        output = sum(range(1, len(producers) + 1)) * factor
        product_demand = 1 + (0.25 * factor / output if year == 2050 else 0)
        assert calculation.supply_array[
            calculation.dicts.activity[manufacturer.id]
        ] == pytest.approx(product_demand / (2 * factor) if is_ethyl_benzene else product_demand)
        for key in producer_keys:
            node = bd.get_node(database=key[0], code=key[1])
            assert calculation.supply_array[calculation.dicts.activity[node.id]] == 0
        for mode, initial, extra in (("road", 0.125, 0.375), ("rail", 0.25, 0.5)):
            node = bd.get_node(database=ordinary, code=mode)
            assert calculation.supply_array[calculation.dicts.activity[node.id]] == pytest.approx(
                (initial * factor + (extra if year == 2050 else 0)) / output
            )
        if is_ethyl_benzene:
            ethylene = bd.get_node(
                database=frontier.database_name, code=provider_service_code(f"ethylene_{origin}")
            )
            assert calculation.supply_array[
                calculation.dicts.activity[ethylene.id]
            ] == pytest.approx(product_demand * sum(range(1, 19)) / 128)
