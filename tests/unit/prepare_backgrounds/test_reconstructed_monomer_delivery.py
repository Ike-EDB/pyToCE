"""Actual reconstructed resin recipes retain complete net-monomer procurement."""

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
from numpy.typing import NDArray
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

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
# Use net external kg per kg reaction product (PBS before finishing). BDO circulates within the
# factory. Finished PBS scales reactor inputs once.
_FEEDS = {
    "polyamide_6_resin": {"caprolactam": 1.0},
    "polyamide_66_resin": {
        "adipic_acid": 0.6457317073170732,
        "hexamethylenediamine": 0.5134676564156946,
    },
    "pbat": {"butanediol": 0.4392278262574763, "adipic_acid": 0.36633657423006194},
    "pbs": {"butanediol": 0.5227003759301838, "succinic_acid": 0.6849009345381919},
}
# Independently retained local recipe duties. These explicit values catch a
# procurement correction accidentally changing water, polymerization or waste.
_LOCAL_AMOUNTS = {
    "polyamide_6_resin": {
        "electricity": 0.9719129859188831,
        "heat_transfer_duty": 0.6377121193886148,
        "steam": 2.613401131303406,
        "nitrogen_supply_and_vent": 0.022,
        "process_and_scrubber_water": 2.0,
        "wastewater_treatment": -0.002,
        "hazardous_waste_treatment": -0.00035,
        "heat_transfer_medium_makeup": 0.0000325,
    },
    "polyamide_66_resin": {
        "process_and_scrubber_water": 2.090800636267232,
        "electricity": 0.5417766051011434,
        "heat_transfer_duty": 3.872232189973615,
        "steam": 0.7773720316622691,
        "wastewater_treatment": -0.00225,
        "hazardous_waste_treatment": -0.00035,
    },
    "pbat": {
        "purified_terephthalic_acid_delivered": 0.36633657423006194,
        "glycerine_delivered": 0.00096,
        "titanium_butoxide_delivered": 0.000672,
        "phosphoric_acid": 0.000192,
        "hexamethylene_diisocyanate_chain_extender": 0.00784,
        "steam": 0.15834934,
        "electricity": 0.04600414,
        "cooling_water": 7.09830384,
        "reaction_water_treatment": -0.00017212497471760034,
        "tetrahydrofuran_purge_treatment": -0.00944,
        "finished_resin_delivery": 1.0,
    },
    "pbs": {
        "titanium_butoxide_delivered": 0.001369801869076384,
        "process_water": 0.49,
        "cooling_water": 0.26,
        "electricity": 0.07,
        "low_pressure_steam": 0.09,
        "reaction_water_treatment": -0.0002089711123374521,
    },
}
# Eight source-position succinic rows include succinic coproducts of other factories.
_MARKETS = {
    "butanediol": (
        "1,4-butanediol",
        "GLO",
        (("1,4-butanediol production", "RER"), ("1,4-butanediol production", "RoW")),
        (1.25, 3.0),
    ),
    "adipic_acid": (
        "adipic acid",
        "GLO",
        (("adipic acid production", "RoW"), ("adipic acid production", "RER")),
        (3.5, 0.75),
    ),
    "succinic_acid": (
        "succinic acid",
        "GLO",
        (
            ("acetic acid production, from naphtha oxidation", "IN"),
            ("acetic acid production, from naphtha oxidation", "UN-EASIA"),
            ("acetic acid production, from naphtha oxidation", "RoW"),
            ("adipic acid production", "RoW"),
            ("adipic acid production", "RER"),
            ("succinic acid production", "GLO"),
            ("acetic acid production, from naphtha oxidation", "RER"),
            ("acetic acid production, from naphtha oxidation", "US"),
        ),
        (1e-12, 0.1, 0.2, 0.3, 0.4, 3.0, 0.05, 0.025),
    ),
    "hexamethylenediamine": (
        "hexamethylenediamine",
        "RER",
        (("hexamethylenediamine production", "RER"),),
        (4.125,),
    ),
}


@pytest.mark.parametrize(("year", "ordinary", "factor"), [(2025, "native", 1), (2050, "dated", 2)])
def test_reconstructed_resins_keep_net_procurement_and_local_duties(
    tmp_path: Path, year: int, ordinary: str, factor: int
) -> None:
    """Solve all ten actual parents with complete markets at the real frontier.

    Only the external upstream manufacture/independent local supply boundaries
    use small emitting native fixtures. The ten parent recipes, their inline PBS
    reactors and thirteen procurement declarations are unchanged package
    resources. Invented nonunit market/producer outputs and non-normalized shares
    expose lost normalization, duplicate purchases or missing freight. No
    licensed inventory is read.
    """
    catalogue = load_provider_catalogue()
    services = {
        row["provider_service_id"]: row for row in cast(list[dict[str, Any]], catalogue["services"])
    }
    parents = {
        f"{stem}_{origin}": (stem, origin)
        for stem in _FEEDS
        for origin in (_ORIGINS[1:] if stem.startswith("polyamide") else _ORIGINS)
    }
    delivered = {
        f"{feed}_delivered_{origin}": (feed, origin)
        for stem, origin in parents.values()
        for feed in _FEEDS[stem]
    }
    manufacturers = {f"{feed}_{origin}" for feed, origin in delivered.values()}
    selected = [services[key] for key in (*parents, *delivered)]
    selected.extend(services[f"pbs_reactor_product_{origin}"] for origin in _ORIGINS)
    # Local inputs retain the declared supplier selections. Receiving and
    # finished resin delivery use synthetic activities with signed reference
    # outputs here; their full inventories are outside this fixture.
    local_sources: dict[str, dict[str, Any]] = {}
    local_keys: dict[str, dict[str, str]] = {}
    local_amounts: dict[str, dict[str, float]] = {}
    leaf_services = set(manufacturers)
    for parent_id, (stem, origin) in parents.items():
        local_keys[parent_id] = {}
        components = services[parent_id]["construction"]["components"]
        local_amounts[parent_id] = _LOCAL_AMOUNTS[stem]
        if stem == "pbs":
            # Finish the reactor product with its nested feed purchases, signed receiving supply,
            # and final-kg delivery.
            reactor_id = f"pbs_reactor_product_{origin}"
            components = [
                *services[reactor_id]["construction"]["components"],
                *(row for row in components if row.get("service_id") != reactor_id),
            ]
            local_amounts[parent_id] = {
                label: (1 / 0.98) * amount for label, amount in _LOCAL_AMOUNTS[stem].items()
            } | {
                "finishing_electricity": 1.1 / (3.6 * 0.98),
                "finishing_reject_treatment": (1 - 0.98) / 0.98,
                "finished_resin_delivery": 1.0,
            }
        for component in components:
            service_id = component.get("service_id")
            if service_id in delivered:
                continue
            label = component.get("component_id")
            if service_id:
                leaf_services.add(service_id)
                label = label or service_id.removesuffix(f"_{origin}")
                code = f"leaf_{service_id}"
            else:
                selector = component["selector"]
                matching = [
                    key for key, value in local_sources.items() if value["selector"] == selector
                ]
                code = matching[0] if matching else f"local_{len(local_sources)}"
                local_sources[code] = {"selector": selector, "negative": component["amount"] < 0}
            local_keys[parent_id][label] = code
        assert set(local_keys[parent_id]) == set(local_amounts[parent_id])
    for service_id in sorted(leaf_services):
        source = services[service_id]
        selected.append(
            {
                **source,
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": {
                        "name": f"admitted {service_id}",
                        "reference_product": source["reference_product"],
                        "unit": source["unit"],
                        "exact_location": "RER",
                    },
                    "amount": 1.0,
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity="invented_reconstructed_monomer_delivery",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )

    def inventory(database: str, multiplier: int) -> dict[tuple[str, str], dict[str, Any]]:
        """Build synthetic markets with native record structure and signed supplies."""

        def row(code: str, amount: float, kind: str = "technosphere") -> dict[str, Any]:
            """Distinguish original lognormal records from dated deterministic rows."""
            deterministic = multiplier == 2 or amount < 1e-10
            return {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "type": kind,
                "amount": amount,
                "uncertainty type": 0 if deterministic else 2,
                "loc": amount if deterministic else log(amount),
                **({} if deterministic else {"scale": 0.15, "pedigree": {"reliability": 3}}),
                "comment": f"invented source row {code}",
            }

        data = {}
        for service_id in sorted(leaf_services):
            source = services[service_id]
            negative = service_id == "hydraulic_water_collection_europe"
            code = f"leaf_{service_id}"
            data[database, code] = _process(
                database,
                code,
                f"admitted {service_id}",
                source["reference_product"],
                production_amount=(-2 if negative else 2) * multiplier,
                exchanges=[
                    row(
                        service_id if service_id in manufacturers else "local",
                        multiplier,
                        "biosphere",
                    )
                ],
            )
            data[database, code]["unit"] = source["unit"]
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
        for stem, (product, location, producers, amounts) in _MARKETS.items():
            records = []
            for index, ((name, geography), amount) in enumerate(
                zip(producers, amounts, strict=True)
            ):
                code = f"{stem}_producer_{index}"
                data[database, code] = _process(
                    database,
                    code,
                    name,
                    product,
                    location=geography,
                    exchanges=[row("unreplaced", 20, "biosphere")],
                )
                # A dated source changes its final producer coefficient as well
                # as output scale, so freezing original shares cannot pass.
                quantity = amount + (0.0625 if multiplier == 2 and index == len(amounts) - 1 else 0)
                records.append(row(code, quantity * multiplier))
            production = row(stem, 4 * multiplier, "production")
            for index, record in enumerate([production, *records]):
                record.update(
                    {"name": product, "unit": "kilogram", "activity": "source-producer-uuid"}
                )
                if multiplier == 2:
                    record["product"] = product
                    record["location"] = location if index == 0 else producers[index - 1][1]
                else:
                    record.update(
                        {
                            "reference product": product,
                            "flow": f"source-{stem}-flow",
                            "CAS number": f"source-{stem}-CAS",
                            "chemical formula": "C6H16N2"
                            if stem == "hexamethylenediamine"
                            else "source formula",
                            "classifications": {"CPC": f"source {product}"},
                            "properties": {
                                "carbon content": {"amount": 0.4, "unit": "dimensionless"},
                                "carbon content, fossil": {"amount": 0.4, "unit": "dimensionless"},
                                "carbon content, non-fossil": {
                                    "amount": 0.0,
                                    "unit": "dimensionless",
                                },
                                "dry mass": {"amount": 1.0, "unit": "kg"},
                            },
                        }
                    )
                    if stem == "succinic_acid" and index in (4, 5):
                        record["properties"] = {
                            "carbon content": {"amount": 0.0, "unit": "dimensionless"}
                        }
                    elif stem == "succinic_acid" and index == 0:
                        record["properties"]["dry mass"]["amount"] = 0.96
            modes = (
                ("barge", "lorry", "rail")
                if location == "RER"
                else ("barge", "lorry", "ship", "rail")
            )
            for index, mode in enumerate(modes, start=1):
                code = f"{stem}_{mode}"
                geography = (
                    location
                    if multiplier == 1 or mode == "barge"
                    else "EUR"
                    if location == "RER"
                    else "World"
                )
                data[database, code] = _process(
                    database,
                    code,
                    f"native {mode} freight",
                    f"freight, {mode}",
                    location=geography,
                    production_amount=2 * multiplier,
                    exchanges=[row("freight", multiplier, "biosphere")],
                )
                data[database, code]["unit"] = "ton kilometer"
                records.append(row(code, index * multiplier / 16))
            data[database, stem] = _process(
                database, stem, f"market for {product}", product, location=location
            )
            data[database, stem]["exchanges"] = [production, *records]
        return data

    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="reconstructed_monomer_delivery",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": "NMVOC, non-methane volatile organic compounds"
                    if code == "nmvoc"
                    else f"invented {code}",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
                for code in (*sorted(manufacturers), "local", "freight", "unreplaced", "nmvoc")
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
        for delivery_id, (feed, origin) in delivered.items():
            stem = "hexamethylenediamine" if feed == "caprolactam" else feed
            product, _, producers, _ = _MARKETS[stem]
            target = services[delivery_id]["reference_product"]
            node = bd.get_node(
                database=frontier.database_name, code=provider_service_code(delivery_id)
            )
            supplier = (frontier.database_name, provider_service_code(f"{feed}_{origin}"))
            producer_keys = {
                (ordinary, f"{stem}_producer_{index}") for index in range(len(producers))
            }
            expected = []
            for source in original[ordinary, stem]["exchanges"]:
                record = deepcopy(source)
                record["output"] = node.key
                if source["type"] == "production" or source["input"] in producer_keys:
                    record["input"] = node.key if source["type"] == "production" else supplier
                    record.pop("activity")
                    record["name"] = target
                    if factor == 2:
                        record["product"] = target
                        record["location"] = services[
                            delivery_id if source["type"] == "production" else f"{feed}_{origin}"
                        ]["location"]
                    else:
                        record["reference product"] = target
                        if feed == "caprolactam":
                            for field in (
                                "flow",
                                "CAS number",
                                "chemical formula",
                                "properties",
                                "classifications",
                            ):
                                record.pop(field)
                        elif origin != "fossil":
                            record["properties"].pop("carbon content, fossil", None)
                            record["properties"].pop("carbon content, non-fossil", None)
                expected.append(record)
            assert sorted((dict(row) for row in node.exchanges()), key=repr) == sorted(
                expected, key=repr
            )
            # The delivery activity supplies the target monomer with its selected manufacturer.
            assert len([ex for ex in node.technosphere() if ex.input.key == supplier]) == len(
                producers
            )

        for parent_id, (stem, origin) in parents.items():
            parent = bd.get_node(
                database=frontier.database_name, code=provider_service_code(parent_id)
            )
            # Net monomers and their complete market freight share the same
            # reactor-to-finished basis. Final-resin freight is a separate leaf.
            reactor_scale = 1 / 0.98 if stem == "pbs" else 1.0
            expected_inputs = {
                (
                    frontier.database_name,
                    provider_service_code(f"{feed}_delivered_{origin}"),
                ): reactor_scale * quantity
                for feed, quantity in _FEEDS[stem].items()
            }
            # Reactor and finishing electricity select the same source. Sum the
            # actual ordered duties before inspecting its one compiled purchase.
            expected_local: dict[tuple[str, str], float] = defaultdict(float)
            for label, amount in local_amounts[parent_id].items():
                expected_local[ordinary, local_keys[parent_id][label]] += amount
            expected_inputs.update(expected_local)
            assert {ex.input.key: ex["amount"] for ex in parent.technosphere()} == expected_inputs
            assert len(list(parent.technosphere())) == len(expected_inputs)
            assert next(iter(parent.production()))["amount"] == 1.0
            calculation = LCA({parent: 1.0})
            calculation.lci()
            expected_emissions = {
                key: 0.0 for key in (*manufacturers, "local", "freight", "unreplaced", "nmvoc")
            }
            for feed, quantity in _FEEDS[stem].items():
                quantity = reactor_scale * quantity
                market = "hexamethylenediamine" if feed == "caprolactam" else feed
                _, location, _, amounts = _MARKETS[market]
                ratio = (fsum(amounts) + (0.0625 if factor == 2 else 0)) / 4
                expected_emissions[f"{feed}_{origin}"] = quantity * ratio / 2
                modes = (
                    ("barge", "lorry", "rail")
                    if location == "RER"
                    else ("barge", "lorry", "ship", "rail")
                )
                for index, mode in enumerate(modes, start=1):
                    freight = bd.get_node(database=ordinary, code=f"{market}_{mode}")
                    demand = quantity * index / 64 / (2 * factor)
                    assert cast(NDArray[np.float64], calculation.supply_array)[
                        calculation.dicts.activity[freight.id]
                    ] == pytest.approx(demand)
                    expected_emissions["freight"] += demand * factor
            for (database, code), amount in expected_local.items():
                source = bd.get_node(database=database, code=code)
                demand = abs(amount) / (2 * factor)
                assert cast(NDArray[np.float64], calculation.supply_array)[
                    calculation.dicts.activity[source.id]
                ] == pytest.approx(demand)
                expected_emissions["local"] += abs(amount) / 2
            expected_emissions["nmvoc"] = {
                "polyamide_6_resin": 0.0000405,
                "polyamide_66_resin": 0.00002,
            }.get(stem, 0.0)
            for code, amount in expected_emissions.items():
                flow = bd.get_node(database="biosphere", code=code)
                observed = float(
                    calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum()
                )
                assert observed == pytest.approx(amount)
        assert {
            node.key: deepcopy(node.as_dict())
            | {"exchanges": [dict(ex) for ex in node.exchanges()]}
            for database in ("native", "dated")
            for node in bd.Database(database)
        } == original
