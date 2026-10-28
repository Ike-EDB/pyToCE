"""Calendering preserves signed PVC waste and its actual carbon-origin chain."""

import json
from copy import deepcopy
from pathlib import Path

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


@pytest.mark.parametrize("year", [2025, 2040])
@pytest.mark.parametrize("non_fossil", [False, True])
def test_calendering_preserves_native_pvc_waste_signs_and_origin(
    tmp_path: Path, year: int, non_fossil: bool
) -> None:
    """Use invented dated amounts through real preparation and an ordinary tiny LCI."""
    catalogue = load_provider_catalogue()
    selected = {
        "conversion_calendering",
        "conversion_calendering__non_fossil",
        "pvc_waste_market_rer_non_fossil",
        "pvc_waste_market_ch_non_fossil",
        "pvc_waste_od500_non_fossil",
        "pvc_waste_incineration_ch_non_fossil",
    }
    services = {
        row["provider_service_id"]: deepcopy(row)
        for row in catalogue["services"]
        if row["provider_service_id"] in selected
    }
    # The synthetic graph uses the declared activity and flow selectors. The resource defines the
    # complete regional source graph.
    for service in services.values():
        construction = service["construction"]
        if construction["kind"] == "waste_market_origin_operation":
            construction["treatment_replacements"] = [
                row
                for row in construction["treatment_replacements"]
                if row["service_id"] in selected
            ]
    routes = [row for row in catalogue["routes"] if row["provider_service_id"] in selected]
    catalogue = {**catalogue, "services": list(services.values()), "routes": routes}
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="calendering",
        biosphere_database="biosphere",
        source_database="ordinary",
    )
    with selected_brightway_project(names) as bd:
        biosphere = {
            ("biosphere", "process"): {
                "name": "Carbon dioxide, fossil",
                "unit": "kilogram",
                "categories": ("air",),
                "type": "emission",
            }
        }
        for area, compartment in (
            ("urban", "urban air close to ground"),
            ("high", "non-urban air or from high stacks"),
        ):
            for gas, name in (
                ("co2", "Carbon dioxide"),
                ("ch4", "Methane"),
                ("co", "Carbon monoxide"),
            ):
                for origin, label in (("f", "fossil"), ("nf", "non-fossil")):
                    biosphere[("biosphere", f"{gas}_{area}_{origin}")] = {
                        "name": f"{name}, {label}",
                        "unit": "kilogram",
                        "categories": ("air", compartment),
                        "type": "emission",
                    }
        bd.Database("biosphere").write(biosphere, searchable=False)
        for database, factor in (("ordinary", 1.0), ("future", 2.0)):
            data = {
                (database, "utility"): _process(database, "utility", "utility", "utility"),
                (database, "conversion"): _process(
                    database,
                    "conversion",
                    "calendering, rigid sheets",
                    "calendering, rigid sheets",
                    production_amount=2.0,
                    exchanges=[
                        {
                            "input": (database, "root"),
                            "amount": -0.2,
                            "type": "technosphere",
                            "uncertainty type": 3,
                            "loc": -0.2,
                            "scale": 0.01,
                        },
                        {
                            "input": (database, "utility"),
                            "amount": 3.0 * factor,
                            "type": "technosphere",
                        },
                        {
                            "input": ("biosphere", "process"),
                            "amount": 0.4 * factor,
                            "type": "biosphere",
                        },
                    ],
                ),
                (database, "root"): _process(
                    database,
                    "root",
                    "market group for waste polyvinylchloride",
                    "waste polyvinylchloride",
                    production_amount=-2.0,
                    exchanges=[
                        {"input": (database, "ch"), "amount": -2.0, "type": "technosphere"},
                        {"input": (database, "utility"), "amount": 0.02, "type": "technosphere"},
                    ],
                ),
                (database, "ch"): _process(
                    database,
                    "ch",
                    "market for waste polyvinylchloride",
                    "waste polyvinylchloride",
                    location="CH",
                    production_amount=-4.0,
                    exchanges=[
                        {"input": (database, "dump"), "amount": -1.0, "type": "technosphere"},
                        {
                            "input": (database, "incineration"),
                            "amount": -3.0,
                            "type": "technosphere",
                        },
                    ],
                ),
            }
            for code, suffix, area, gas_amounts, utility in (
                ("dump", "od500", "high", (("co2", 0.4), ("ch4", 0.02)), 0.3),
                (
                    "incineration",
                    "incineration_ch",
                    "urban",
                    (("co2", 0.8), ("ch4", 0.01), ("co", 0.03)),
                    0.4,
                ),
            ):
                selector = services[f"pvc_waste_{suffix}_non_fossil"]["construction"]["base"][
                    "selector"
                ]
                data[(database, code)] = _process(
                    database,
                    code,
                    selector["name"],
                    selector["reference_product"],
                    location=selector["exact_location"],
                    production_amount=-1.0,
                    exchanges=[
                        {
                            "input": (database, "utility"),
                            "amount": utility * factor,
                            "type": "technosphere",
                        },
                        *[
                            {
                                "input": ("biosphere", f"{gas}_{area}_f"),
                                "amount": amount * factor,
                                "type": "biosphere",
                                "uncertainty type": 3,
                                "loc": amount * factor,
                                "scale": 0.001,
                            }
                            for gas, amount in gas_amounts
                        ],
                    ],
                )
            bd.Database(database).write(data, searchable=False)
        ordinary = "ordinary" if year == 2025 else "future"
        factor = 1.0 if year == 2025 else 2.0
        before = {
            node.key: [dict(row) for row in node.exchanges()] for node in bd.Database(ordinary)
        }
        provider = PreparedProviderAuthority(
            identity="invented_calendering",
            interface_database="provider_interface",
            catalogue=catalogue,
            routes=pd.DataFrame(routes),
            method_authorities=_method_authority_rows(catalogue),
        )
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=ordinary,
            technology_database="future",
            provider=provider,
            material_recipe=physical_material_recipe(),
            resolver_cache=_NativeResolverCache("ordinary", "biosphere"),
        )
        service_id = "conversion_calendering" + ("__non_fossil" if non_fossil else "")
        activity = bd.get_node(
            database=frontier.database_name, code=provider_service_code(service_id)
        )
        # Invented finished fraction .5: 2 kg service produces 1 kg finished.
        assert next(iter(activity.production()))["amount"] == 1.0
        waste = next(row for row in activity.technosphere() if row["amount"] < 0)
        assert waste.input.key == (
            (frontier.database_name, provider_service_code("pvc_waste_market_rer_non_fossil"))
            if non_fossil
            else (ordinary, "root")
        )
        assert (waste["amount"], waste["loc"], waste["scale"]) == (-0.2, -0.2, 0.01)
        owners = json.loads(activity["direct_treatment_carbon_owners_json"])
        expected_owners = {
            (
                frontier.database_name if non_fossil else ordinary,
                provider_service_code(f"pvc_waste_{suffix}_non_fossil") if non_fossil else code,
            ): scale
            for code, suffix, scale in (
                ("dump", "od500", -0.05),
                ("incineration", "incineration_ch", -0.15),
            )
        }
        assert {(row["owner_database"], row["owner_code"]) for row in owners} == set(
            expected_owners
        )
        for row in owners:
            assert row["owner_scale_per_provider_unit"] == pytest.approx(
                expected_owners[(row["owner_database"], row["owner_code"])]
            )
        calculation = LCA({activity: 1.0})
        calculation.lci()
        inventory = calculation.inventory.sum(axis=1).A1
        for gas, area, amount in (
            ("co2", "high", 0.05 * 0.4),
            ("ch4", "high", 0.05 * 0.02),
            ("co2", "urban", 0.15 * 0.8),
            ("ch4", "urban", 0.15 * 0.01),
            ("co", "urban", 0.15 * 0.03),
        ):
            for origin in ("f", "nf"):
                flow = bd.get_node(database="biosphere", code=f"{gas}_{area}_{origin}")
                index = calculation.dicts.biosphere.get(flow.id)
                observed = 0.0 if index is None else inventory[index]
                expected = amount * factor if (origin == "nf") == non_fossil else 0.0
                assert observed == pytest.approx(expected, rel=1e-6, abs=0)
        direct = bd.get_node(database="biosphere", code="process")
        assert inventory[calculation.dicts.biosphere[direct.id]] == pytest.approx(0.4 * factor)
        utility = bd.get_node(database=ordinary, code="utility")
        assert calculation.supply_array[calculation.dicts.activity[utility.id]] == pytest.approx(
            (3.0 + 0.05 * 0.3 + 0.15 * 0.4) * factor + 0.1 * 0.02
        )
        for node in bd.Database(ordinary):
            assert [dict(row) for row in node.exchanges()] == before[node.key]
