"""Reconstructed polymer plants buy complete feeds at their input doses."""

import json
from collections import defaultdict
from copy import deepcopy
from math import fsum, log
from pathlib import Path
from typing import Any, cast

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
from pytoce.prepare_backgrounds.preparation.providers.polyester_material import (
    _pbat_material,
    _ptt_material,
)

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
# Use net external doses. PBS is per kg reactor product before finishing yield. LA/PLA quantities
# use active dry substrate/acid; .865 determines local water.
_DOSES = {
    "pbat": {"pta": 0.36633657423006194, "glycerine": 0.00096, "teos": 0.000672},
    "pbs": {"teos": 0.001369801869076384},
    "ptt": {"pta": 0.8, "teos": 0.0004688888888888889},
    "phb": {"glucose": 3.2},
    "lactic_acid": {"glucose": 1.13694549},
    "pla": {"la": 1.28478429},
}
# Synthetic coefficients retain native name/product/geography positions. Unequal sums, a small
# coproduct, and nonunit outputs check procurement normalization.
_MARKETS = {
    "pta": (
        "purified terephthalic acid",
        "GLO",
        (
            ("purified terephthalic acid production", "RER"),
            ("purified terephthalic acid production", "CA-QC"),
            ("purified terephthalic acid production", "RoW"),
        ),
        (1.25, 1e-12, 3.125),
        ("lorry", "sea", "rail"),
    ),
    "glycerine": (
        "glycerine",
        "RER",
        (
            ("treatment of used vegetable cooking oil, purified, esterification", "FR"),
            ("esterification of rape oil", "CH"),
            ("glycerine production, from epichlorohydrin", "RER"),
            ("esterification of rape oil", "Europe without Switzerland"),
        ),
        (0.15, 0.025, 1.3, 2.6),
        ("barge", "lorry", "rail"),
    ),
    "glucose": (
        "glucose",
        "GLO",
        (("glucose production", "RER"), ("glucose production", "RoW")),
        (0.7, 3.4),
        ("van", "lorry", "sea", "rail"),
    ),
    "teos": (
        "tetraethyl orthosilicate",
        "GLO",
        (("tetraethyl orthosilicate production", "GLO"),),
        (4.125,),
        ("barge", "lorry", "sea", "rail"),
    ),
    "la": (
        "lactic acid",
        "GLO",
        (("lactic acid production", "RER"), ("lactic acid production", "RoW")),
        (1.33333333333333, 2.66666666666667),
        ("barge", "lorry", "sea", "rail"),
    ),
    "hdi": (
        "hexamethylene-1,6-diisoncyanate",
        "RER",
        (("hexamethylene-1,6-diisocyanate production", "RER"),),
        (4.0,),
        ("barge", "lorry", "rail"),
    ),
}
_DELIVERY_STEMS = {
    "pta": "purified_terephthalic_acid",
    "glycerine": "glycerine",
    "glucose": "glucose",
}
_ACTUAL_FACTORIES = {
    "pta_fossil": "pta_producer_0",
    "lactic_acid_fossil": "la_producer_0",
    "titanium_butoxide": "teos_producer_0",
}


@pytest.mark.parametrize(("year", "ordinary", "factor"), [(2025, "native", 1), (2050, "dated", 2)])
def test_reconstructed_feeds_keep_physical_doses_and_complete_procurement(
    tmp_path: Path, year: int, ordinary: str, factor: int
) -> None:
    """Compile fourteen resin recipes with their LA and PBS reactor inventories.

    Synthetic activities represent upstream synthesis, local processing,
    receiving and finished resin delivery. TBOT, fossil PTA and fossil LA
    retain their native supplier selections; HDI retains its complete market.
    The DAC LA and PBS reactor recipes remain within the resin inventories,
    exposing duplicate acid or water and checking total finished resin inputs.
    """
    catalogue = load_provider_catalogue()
    services = {
        row["provider_service_id"]: row for row in cast(list[dict[str, Any]], catalogue["services"])
    }
    parents = {
        f"{stem}_{origin}": (stem, origin)
        for stem in ("pbat", "pbs", "ptt", "phb")
        for origin in _ORIGINS
    } | {
        "lactic_acid_atmospheric_dac": ("lactic_acid", "atmospheric_dac"),
        "pla_fossil": ("pla", "fossil"),
        "pla_atmospheric_dac": ("pla", "atmospheric_dac"),
    }
    delivered = {
        f"{stem}_delivered_{origin}": (market, origin)
        for market, stem in _DELIVERY_STEMS.items()
        for origin in _ORIGINS
    } | {
        "titanium_butoxide_delivered": ("teos", "fossil"),
        "lactic_acid_delivered_fossil": ("la", "fossil"),
    }
    manufacturers = {
        delivery: services[delivery]["construction"]["replacements"][0]["service_id"]
        for delivery in delivered
        if delivery != "glucose_delivered_biogenic"
    }
    # Direct recipe lookups check composition for the delivered feeds.
    # These checks cover total material mass and selected versus fixed fossil carbon.
    for material, selected, fixed in (
        (_pbat_material(services), 0.6207700759259209, 0.004858363658085175),
        (_ptt_material(services), 0.6404762459546925, 0.0002647843137254902),
    ):
        assert fsum(material.elements_kg.values()) == pytest.approx(1.0)
        assert (material.selected_carbon_kg, material.fixed_fossil_carbon_kg) == pytest.approx(
            (selected, fixed), rel=2e-12
        )

    recipes = parents | {f"pbs_reactor_product_{origin}": ("pbs", origin) for origin in _ORIGINS}
    selected_services = [services[key] for key in (*recipes, *delivered)]
    leaves = set(manufacturers.values())
    local_sources: dict[str, dict[str, Any]] = {}
    component_keys: dict[str, list[tuple[str, str, float]]] = {}
    for parent_id, (stem, origin) in recipes.items():
        components = services[parent_id]["construction"]["components"]
        # Original PBS feed coefficients belong to the reactor product; keep its
        # actual child in the finished root rather than replacing it by a leaf.
        feed_components = (
            services[f"pbs_reactor_product_{origin}"]["construction"]["components"]
            if parent_id == f"pbs_{origin}"
            else components
        )
        direct = {c["service_id"]: c["amount"] for c in feed_components if "service_id" in c}
        for market, dose in _DOSES[stem].items():
            target = (
                "lactic_acid_atmospheric_dac"
                if market == "la" and origin == "atmospheric_dac"
                else "lactic_acid_delivered_fossil"
                if market == "la"
                else "titanium_butoxide_delivered"
                if market == "teos"
                else f"{_DELIVERY_STEMS[market]}_delivered_{origin}"
            )
            assert direct[target] == dose
        if stem == "pbat":
            hdi = next(
                c
                for c in components
                if c.get("component_id") == "hexamethylene_diisocyanate_chain_extender"
            )
            assert hdi["amount"] == 0.00784
            assert hdi["selector"] == {
                "name": "market for hexamethylene-1,6-diisoncyanate",
                "reference_product": "hexamethylene-1,6-diisoncyanate",
                "unit": "kilogram",
                "exact_location": "RER",
            }
        if stem == "pla":
            water = [
                c["amount"] for c in components if c.get("component_id") == "solution_carrier_water"
            ]
            assert water == ([0.200515467225434] if origin == "fossil" else [])
            assert {c["component_id"]: c["amount"] for c in components if "component_id" in c} == {
                **({"solution_carrier_water": 0.200515467225434} if origin == "fossil" else {}),
                "natural_gas_heat": 3.23675995,
                "average_wastewater_treatment": -0.00047505,
                "hazardous_waste_treatment": -0.0058415,
            }
        component_keys[parent_id] = []
        for component in components:
            service_id = component.get("service_id")
            amount = component["amount"]
            if service_id in delivered or service_id in recipes:
                component_keys[parent_id].append(("service", service_id, amount))
            elif service_id:
                leaves.add(service_id)
                component_keys[parent_id].append(
                    ("native", _ACTUAL_FACTORIES.get(service_id, f"leaf_{service_id}"), amount)
                )
            elif component.get("component_id") == "hexamethylene_diisocyanate_chain_extender":
                component_keys[parent_id].append(("native", "hdi", amount))
            else:
                selector = component["selector"]
                matches = [
                    key for key, spec in local_sources.items() if spec["selector"] == selector
                ]
                key = matches[0] if matches else f"local_{len(local_sources)}"
                local_sources[key] = {"selector": selector, "negative": amount < 0}
                component_keys[parent_id].append(("native", key, amount))
    for service_id in sorted(leaves):
        source = services[service_id]
        if service_id in _ACTUAL_FACTORIES:
            selected_services.append(source)
        else:
            selected_services.append(
                {
                    **source,
                    "construction": {
                        "kind": "native_activity",
                        "source_role": "native_ecoinvent_provider",
                        "selector": {
                            "name": f"bounded {service_id}",
                            "reference_product": source["reference_product"],
                            "unit": source["unit"],
                            "exact_location": "RER",
                        },
                    },
                }
            )
    provider = PreparedProviderAuthority(
        identity="invented_reconstructed_feed_procurement",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected_services, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    factory_markers = set(manufacturers.values()) | {"hdi_factory"}

    def quantity(value: float, multiplier: int) -> float:
        """Retain source float32 precision and dated supplier shares."""
        return float(np.float32(value)) if multiplier == 2 else value

    def inventory(database: str, multiplier: int) -> dict[tuple[str, str], dict[str, Any]]:
        """Supply full native market vectors, actual factory selectors and signed local duties."""

        def row(code: str, amount: float, kind: str = "technosphere") -> dict[str, Any]:
            """Represent original lognormal and absent-property deterministic dated exchanges."""
            deterministic = multiplier == 2 or amount <= 1e-10
            return {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "type": kind,
                "amount": amount,
                "uncertainty type": 0 if deterministic else 2,
                "loc": amount if deterministic else log(amount),
                **({} if deterministic else {"scale": 0.19, "pedigree": {"reliability": 3}}),
                "comment": f"invented {database} source {code}",
            }

        data: dict[tuple[str, str], dict[str, Any]] = {}
        for service_id in sorted(leaves):
            source = services[service_id]
            code = _ACTUAL_FACTORIES.get(service_id, f"leaf_{service_id}")
            selector = (
                source["construction"]["selector"]
                if service_id in _ACTUAL_FACTORIES
                else {
                    "name": f"bounded {service_id}",
                    "reference_product": source["reference_product"],
                    "unit": source["unit"],
                    "exact_location": "RER",
                }
            )
            negative = service_id in {
                "hydraulic_water_collection_europe",
                "phb_residue_fossil_incineration_operation",
            }
            data[database, code] = _process(
                database,
                code,
                selector["name"],
                selector["reference_product"],
                location=selector["exact_location"],
                production_amount=(-2 if negative else 2) * multiplier,
                exchanges=[
                    row(
                        service_id if service_id in factory_markers else "local",
                        multiplier,
                        "biosphere",
                    )
                ],
            )
            data[database, code]["unit"] = selector["unit"]
        for code, spec in local_sources.items():
            selector = spec["selector"]
            data[database, code] = _process(
                database,
                code,
                selector["name"],
                selector["reference_product"],
                location=selector["exact_location"],
                production_amount=(-2 if spec["negative"] else 2) * multiplier,
                exchanges=[row("local", multiplier, "biosphere")],
            )
            data[database, code]["unit"] = selector["unit"]
        for market, (product, geography, producers, shares, modes) in _MARKETS.items():
            inputs = []
            for index, ((name, location), share) in enumerate(zip(producers, shares, strict=True)):
                code = f"{market}_producer_{index}"
                if (database, code) not in data:
                    data[database, code] = _process(
                        database,
                        code,
                        name,
                        product,
                        location=location,
                        production_amount=2 * multiplier,
                        exchanges=[
                            row(
                                "hdi_factory" if market == "hdi" else "unreplaced",
                                multiplier,
                                "biosphere",
                            )
                        ],
                    )
                inputs.append(row(code, quantity(share, multiplier) * multiplier))
            for index, mode in enumerate(modes, start=1):
                code = f"{market}_{mode}"
                location = (
                    geography
                    if multiplier == 1 or mode == "barge"
                    else "EUR"
                    if geography == "RER"
                    else "World"
                )
                data[database, code] = _process(
                    database,
                    code,
                    f"native {mode} freight",
                    f"freight, {mode}",
                    location=location,
                    production_amount=2 * multiplier,
                    exchanges=[row("freight", multiplier, "biosphere")],
                )
                data[database, code]["unit"] = "ton kilometer"
                inputs.append(row(code, quantity(index * 0.0713, multiplier) * multiplier))
            production = row(market, 4 * multiplier, "production")
            for index, record in enumerate([production, *inputs[: len(producers)]]):
                record.update(
                    name=product,
                    unit="kilogram",
                    activity="invented native producer UUID",
                    formula="original amount expression",
                )
                if multiplier == 2:
                    record.update(
                        product=product,
                        location=geography if index == 0 else producers[index - 1][1],
                    )
                else:
                    record.update(
                        {
                            "reference product": product,
                            "flow": f"source {market} flow",
                            "CAS number": f"source {market} CAS",
                            "chemical formula": "C8H20O4Si"
                            if market == "teos"
                            else f"source {market} formula",
                            "classifications": {"CPC": f"source {market} grade"},
                            "properties": {
                                "carbon content": {"amount": 0.4, "unit": "dimensionless"},
                                "carbon content, fossil": {
                                    "amount": 0.12
                                    if market == "glycerine"
                                    else 0.0
                                    if market == "glucose"
                                    else 0.4,
                                    "unit": "dimensionless",
                                },
                                "carbon content, non-fossil": {
                                    "amount": 0.28
                                    if market == "glycerine"
                                    else 0.4
                                    if market == "glucose"
                                    else 0.0,
                                    "unit": "dimensionless",
                                },
                                "dry mass": {"amount": 1.0, "unit": "kg"},
                                "water content": {"amount": 0.0, "unit": "kg"},
                            },
                        }
                    )
            data[database, market] = _process(
                database, market, f"market for {product}", product, location=geography
            )
            data[database, market]["exchanges"] = [production, *inputs]
        return data

    names = BrightwayNames(
        tmp_path / "brightway", "reconstructed_feed_procurement", "biosphere", "native"
    )
    with selected_brightway_project(names) as bd:
        biosphere = {
            code: (f"invented {code}", "kilogram", ("air",))
            for code in (*factory_markers, "local", "freight", "unreplaced")
        } | {
            "co2_fossil": ("Carbon dioxide, fossil", "kilogram", ("air",)),
            "co2_non_fossil": ("Carbon dioxide, non-fossil", "kilogram", ("air",)),
            "cod": ("COD, Chemical Oxygen Demand", "kilogram", ("water",)),
            "water": ("Water", "cubic meter", ("water",)),
        }
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": name,
                    "unit": unit,
                    "categories": categories,
                    "type": "emission",
                }
                for code, (name, unit, categories) in biosphere.items()
            },
            searchable=False,
        )
        for database, multiplier in (("native", 1), ("dated", 2)):
            bd.Database(database).write(inventory(database, multiplier), searchable=False)
        original = {
            node.key: deepcopy(node.as_dict())
            | {"exchanges": [dict(ex) for ex in node.exchanges()]}
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
        for delivery, (market, origin) in delivered.items():
            node = bd.get_node(
                database=frontier.database_name, code=provider_service_code(delivery)
            )
            if delivery == "glucose_delivered_biogenic":
                # The complete native market owns both manufacturers and freight.
                assert node.rp_exchange()["amount"] == 1.0
                assert [(ex.input.key, ex["amount"]) for ex in node.technosphere()] == [
                    ((ordinary, market), 1.0)
                ]
                continue
            factory = manufacturers[delivery]
            supplier = (frontier.database_name, provider_service_code(factory))
            _, _, producers, _, modes = _MARKETS[market]
            producer_keys = {(ordinary, f"{market}_producer_{i}") for i in range(len(producers))}
            expected = []
            for source in original[ordinary, market]["exchanges"]:
                record = deepcopy(source)
                record["output"] = node.key
                if source["type"] == "production" or source["input"] in producer_keys:
                    product_owner = delivery if source["type"] == "production" else factory
                    record["input"] = node.key if source["type"] == "production" else supplier
                    record.pop("activity")
                    record["name"] = services[product_owner]["reference_product"]
                    if factor == 2:
                        record["product"] = services[product_owner]["reference_product"]
                        record["location"] = services[product_owner]["location"]
                    else:
                        record["reference product"] = services[product_owner]["reference_product"]
                        if market == "teos":
                            for field in (
                                "flow",
                                "CAS number",
                                "chemical formula",
                                "properties",
                                "classifications",
                            ):
                                record.pop(field)
                        elif not (market in {"pta", "la"} and origin == "fossil"):
                            record["properties"].pop("carbon content, fossil")
                            record["properties"].pop("carbon content, non-fossil")
                expected.append(record)
            assert sorted((dict(ex) for ex in node.exchanges()), key=repr) == sorted(
                expected, key=repr
            )
            assert len(list(node.technosphere())) == len(producers) + len(modes)

        # The actual LA and PBS reactor children are flattened by the real
        # compiler. Expand only these admitted relationships in the input oracle.
        for parent_id, (stem, origin) in parents.items():
            if parent_id == "lactic_acid_atmospheric_dac":
                # The nested operation's complete purchase vector is solved through DAC PLA.
                continue
            inputs = list(component_keys[parent_id])
            if stem == "pbs":
                reactor_id = f"pbs_reactor_product_{origin}"
                expanded = []
                for entry in inputs:
                    if entry[1] == reactor_id:
                        expanded.extend(
                            (kind, code, (1 / 0.98) * amount)
                            for kind, code, amount in component_keys[reactor_id]
                        )
                    else:
                        expanded.append(entry)
                inputs = expanded
            if parent_id == "pla_atmospheric_dac":
                inputs = [entry for entry in inputs if entry[1] != "lactic_acid_atmospheric_dac"]
                inputs.extend(
                    (kind, code, 1.28478429 * amount)
                    for kind, code, amount in component_keys["lactic_acid_atmospheric_dac"]
                )
            expected_inputs: dict[tuple[str, str], float] = defaultdict(float)
            for kind, code, amount in inputs:
                key = (
                    (ordinary, "glucose")
                    if code == "glucose_delivered_biogenic"
                    else (frontier.database_name, provider_service_code(code))
                    if kind == "service"
                    else (ordinary, code)
                )
                expected_inputs[key] += amount
            parent = bd.get_node(
                database=frontier.database_name, code=provider_service_code(parent_id)
            )
            assert {ex.input.key: ex["amount"] for ex in parent.technosphere()} == pytest.approx(
                dict(expected_inputs)
            )
            assert len(list(parent.technosphere())) == len(expected_inputs)
            calculation = LCA({parent: 1.0})
            calculation.lci()
            expected_emissions: dict[str, float] = defaultdict(float)
            for kind, code, amount in inputs:
                market = (
                    delivered[code][0] if kind == "service" else "hdi" if code == "hdi" else None
                )
                if market is not None:
                    _, _, _, shares, modes = _MARKETS[market]
                    marker = (
                        "unreplaced"
                        if code == "glucose_delivered_biogenic"
                        else manufacturers[code]
                        if kind == "service"
                        else "hdi_factory"
                    )
                    expected_emissions[marker] += (
                        amount * fsum(quantity(a, factor) for a in shares) / 8
                    )
                    for index, mode in enumerate(modes, start=1):
                        expected_emissions["freight"] += (
                            amount * quantity(index * 0.0713, factor) / 8
                        )
                else:
                    expected_emissions["local"] += abs(amount) / 2
            if stem == "phb":
                expected_emissions["co2_fossil"] = 1.8356368563685637 if origin == "fossil" else 0.0
                expected_emissions["co2_non_fossil"] = (
                    0.06436314363143632 if origin == "fossil" else 1.9
                )
                expected_emissions.update(cod=0.9, water=0.0331)
            for code in biosphere:
                flow = bd.get_node(database="biosphere", code=code)
                observed = float(
                    calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum()
                )
                assert observed == pytest.approx(expected_emissions[code]), (parent_id, code)

        # Exact source records, including TEOS chemistry and source uncertainty,
        # remain available after compilation; no generated metadata rewrites leak.
        assert {
            node.key: deepcopy(node.as_dict())
            | {"exchanges": [dict(ex) for ex in node.exchanges()]}
            for database in ("native", "dated")
            for node in bd.Database(database)
        } == original
        for service_id in ("pbat_biogenic", "ptt_fossil"):
            traces = [
                row
                for row in frontier.semantic_rows
                if row.get("provider_service_id") == service_id
                and row.get("ledger_entry_type") == "service_construction"
            ]
            assert len(traces) == 1
            assert traces[0]["foreground_stage_owner"] == "virgin_resin_production"
            assert (
                json.loads(cast(str, traces[0]["scaling_algebra"]))
                == services[service_id]["construction"]
            )
            assert (
                json.loads(cast(str, traces[0]["method_details_json"]))
                == services[service_id]["representative_method"]
            )
