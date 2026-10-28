"""Actual dry-feed preparation, pellet finishing and separate freight in Brightway."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from recovery_delivery_fixtures import recovery_biosphere, recovery_native_inventory

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


@pytest.mark.parametrize(
    ("material", "dry_feed", "pellets"),
    [
        ("hdpe", 3.2, (31199.69 / 33540) * 0.99 * 0.98 * 0.97),
        ("ldpe", 3.0, (31199.69 / 33540) * 0.99 * 0.99 * 0.98 * 0.97),
        ("pp", 2.4, (19377.32 / 21930) * 0.99 * 0.99 * 0.97),
        ("ps", 2.88, (3847.08 / 5160) * 0.99 * 0.83 * 0.97),
        ("pvc", 3.6, 0.9 * 0.9),
        ("pet", 2.7, (19205.93 / 21930) * 0.99 * 0.99 / 1.2),
        ("biopolyester", 3.2, 0.55 * 0.99 * 0.97),
        ("pa", 3.2, 0.98),
        ("engineering", 3.2, 0.999 * 0.96 * 0.97),
    ],
)
def test_recovery_operations_use_actual_material_and_preserve_native_records(
    tmp_path: Path, material: str, dry_feed: float, pellets: float
) -> None:
    """Use the real compiler and LCI solver with nonunit reference productions.

    Synthetic archive records supply wet feed, moisture and production for the
    material recipe. Native fixture operations then retain their own reference
    amounts and emission records. Discarded principal purchases/rejects have
    nonzero sentinel burdens, so accidentally retaining them changes the LCI.
    """
    catalogue = load_provider_catalogue()
    separate_finishing = material in {"hdpe", "ldpe", "pet"}
    integrated_finishing = material in {"biopolyester", "pa", "engineering"}
    pellet_material = "pet" if material == "pet" else "polyethylene"
    if integrated_finishing:
        service_ids = (
            "recycling_weee_preparation",
            "recycling_pet_finishing",
            "recycling_pet_pellet_delivery",
        )
        productions = (2.0, 2.0, 5.0)
        emissions = (13.0, 17.0, 23.0)
    elif separate_finishing:
        service_ids = (
            f"recycling_{material}_preparation",
            f"recycling_{material}_finishing",
            f"recycling_{material}_flake_delivery",
            f"recycling_{pellet_material}_pellet_delivery",
        )
        productions = (2.0, 2.0, 4.0, 5.0)
        emissions = (13.0, 17.0, 19.0, 23.0)
    else:
        service_ids = (f"recycling_{material}_preparation", f"recycling_{material}_pellet_delivery")
        productions = (2.0, 5.0)
        emissions = (13.0, 23.0)
    parent_id = f"mechanical_recycling_{material}" + ("_fibre" if material == "pet" else "")
    selected = set(service_ids) | {parent_id}
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["provider_service_id"] in selected
    }
    provider = PreparedProviderAuthority(
        identity=f"invented_{material}_recovery",
        interface_database="interface",
        catalogue={**catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="recovery",
        biosphere_database="biosphere",
        source_database="original",
    )
    with selected_brightway_project(names) as bd:
        flow = ("biosphere", "co2")
        bd.Database("biosphere").write(
            {
                **recovery_biosphere(),
                ("biosphere", "solvent"): {
                    "name": "VOC, volatile organic compounds",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
            },
            searchable=False,
        )
        bd.Database("original").write(recovery_native_inventory("original"), searchable=False)
        native_data = recovery_native_inventory("native")
        source_keys = {}
        for service_id in service_ids:
            selector = services[service_id]["construction"]["base"]["selector"]
            identity = (
                selector["name"],
                selector["reference_product"],
                selector["unit"],
                selector["exact_location"],
            )
            source_keys[identity] = ("native", service_id)
        for index, service_id in enumerate(service_ids):
            construction = services[service_id]["construction"]
            selector = construction["base"]["selector"]
            key = ("native", service_id)
            exchanges = [
                {
                    "input": key,
                    "type": "production",
                    "amount": productions[index],
                    "uncertainty type": 0,
                    "loc": productions[index],
                },
                *[
                    {
                        "input": flow,
                        "type": "biosphere",
                        "amount": amount,
                        "uncertainty type": 3,
                        "loc": amount,
                        "scale": 0.01,
                    }
                    for amount in (emissions[index], -2.0, 0.0)
                ],
                {
                    "input": ("native", "auxiliary"),
                    "type": "technosphere",
                    "amount": 0.123,
                    "uncertainty type": 3,
                    "loc": 0.13,
                    "scale": 0.04,
                },
            ]
            if construction.get("omit_biosphere"):
                exchanges.append(
                    {"input": ("biosphere", "solvent"), "type": "biosphere", "amount": 100.0}
                )
            for rule in construction["omit_technosphere"]:
                omission = rule["selector"]
                identity = (
                    omission["name"],
                    omission["reference_product"],
                    omission["unit"],
                    omission["location"],
                )
                omitted_key = source_keys.get(
                    identity, ("native", omission["name"] + omission["location"])
                )
                if identity not in source_keys:
                    native_data[omitted_key] = {
                        "name": omission["name"],
                        "reference product": omission["reference_product"],
                        "unit": omission["unit"],
                        "location": omission["location"],
                        "type": "process",
                        "exchanges": [
                            {"input": omitted_key, "type": "production", "amount": 1.0},
                            {"input": flow, "type": "biosphere", "amount": 100.0},
                        ],
                    }
                exchanges.append({"input": omitted_key, "type": "technosphere", "amount": 0.5})
            native_data[key] = {
                "name": selector["name"],
                "reference product": selector["reference_product"],
                "unit": selector["unit"],
                "location": selector["exact_location"],
                "type": "process",
                "exchanges": exchanges,
            }
        bd.Database("native").write(native_data, searchable=False)
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2030,
            ordinary_database="native",
            technology_database="native",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        generated = bd.get_node(
            database=frontier.database_name, code=provider_service_code(parent_id)
        )
        calculation = LCA({generated: 1.0})
        calculation.lci()
        # Dry kg per preparation activity comes independently from the invented
        # archive's wet feed and moisture. PE has a separate 2 kg finisher and
        # flake delivery; PET has a separately admitted finishing feed ratio.
        # PP, PS and PVC already make pellets; the integrated WEEE transfers
        # require pelletizing but have no off-site flake movement.
        flakes = (
            pellets * 1.2 if material == "pet" else pellets / 0.97 if separate_finishing else 0.0
        )
        demands = (
            (2 / dry_feed, pellets, pellets)
            if integrated_finishing
            else (
                (2 / dry_feed, pellets, flakes, pellets)
                if separate_finishing
                else (2 / dry_feed, pellets)
            )
        )
        assert {
            exchange.input.key: exchange["amount"] for exchange in generated.technosphere()
        } == (
            pytest.approx(
                {
                    **{
                        (frontier.database_name, provider_service_code(service_id)): demands[index]
                        for index, service_id in enumerate(service_ids)
                    },
                },
                rel=2e-14,
                abs=0,
            )
        )
        # bw_processing.utils.resolve_dict_iterator stores matrix amounts as float32;
        # verify unrounded stored demands above and that exact matrix conversion here.
        activities = tuple(
            float(np.float32(demand)) / productions[index] for index, demand in enumerate(demands)
        )
        for index, service_id in enumerate(service_ids):
            operation = bd.get_node(
                database=frontier.database_name, code=provider_service_code(service_id)
            )
            assert calculation.supply_array[
                calculation.dicts.activity[operation.id]
            ] == pytest.approx(activities[index], rel=2e-14)
            original = bd.get_node(database="native", code=service_id)
            expected = []
            for exchange in original.exchanges():
                record = {key: value for key, value in dict(exchange).items() if key != "output"}
                if record["type"] == "technosphere" and record["input"] != ("native", "auxiliary"):
                    continue
                if record["input"] == ("biosphere", "solvent"):
                    continue
                if record["type"] == "production":
                    record["input"] = operation.key
                expected.append(record)
            actual = [
                {key: value for key, value in dict(exchange).items() if key != "output"}
                for exchange in operation.exchanges()
            ]
            assert sorted(actual, key=repr) == sorted(expected, key=repr)
        # Retained native operation emissions and their independent auxiliary
        # supply are the whole fixture LCI; written amounts use float32.
        assert float(calculation.inventory.sum()) == pytest.approx(
            sum((emissions[index] - 2) * activity for index, activity in enumerate(activities))
            + sum(activities) * float(np.float32(0.123)) * float(np.float32(0.1)),
            rel=2e-14,
        )
