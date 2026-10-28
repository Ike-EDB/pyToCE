"""TPS formulation, protected delivery and native normalization through Brightway."""

import copy
from math import fsum
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray

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


@pytest.mark.parametrize("year", [2025, 2030])
def test_tps_manufacture_and_recovery_keep_separate_complete_sack_operations(tmp_path, year):
    """Test invented nonunit sources, fixed original yield and all linked duties.

    The original 2 kg reference and signed waste exchanges of -0.1 and -0.3 kg
    give a retained fraction of 0.8. Different dated quantities expose errors in
    that fixed yield, missing auxiliaries or applying normalization twice.
    Formulations are small invented authorities, not licensed recipe values.
    Virgin facility and delivery must not leak into recovery's source-only use.
    """
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    selected = {
        "mechanical_recycling_tps_native",
        "mechanical_recycling_tps_synthetic",
        "tps_pellet_sack_delivery",
        "recycling_pet_pellet_delivery",
        "plastic_residue_haul",
    }
    selected_services = [copy.deepcopy(services[key]) for key in sorted(selected)]
    formulations = (
        ("biogenic", 1.1, 0.1),
        ("fossil", 1.2, 0.15),
        ("atmospheric_dac", 1.3, 0.2),
    )
    for origin, k, water in formulations:
        definition = services[f"tps_{origin}"]
        components = {
            x.get("component_id", x.get("service_id")): x
            for x in definition["construction"]["components"]
        }
        electricity = components["extrusion_electricity"]
        facility = components["manufacturing_facility"]
        original_g = definition["representative_method"]["gross_technical_output_kg"]
        selected_services.append(
            {
                **definition,
                "foreground_service": True,
                "representative_method": {
                    "gross_technical_output_kg": k,
                    "co_delivered_water_kg": water,
                },
                "construction": {
                    "kind": "project_recipe",
                    "components": [
                        {**electricity, "amount": 3.54 * k},
                        {**facility, "amount": facility["amount"] * k / original_g},
                        {**components["tps_pellet_sack_delivery"], "amount": k},
                    ],
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity="invented_tps",
        interface_database="interface",
        routes=pd.DataFrame(),
        catalogue={**catalogue, "services": selected_services, "routes": []},
        method_authorities=_method_authority_rows(catalogue),
    )
    sack = services["tps_pellet_sack_delivery"]["construction"]
    native_recovery = services["mechanical_recycling_tps_native"]["construction"]
    freight = services["recycling_pet_pellet_delivery"]["construction"]
    residue_haul = services["plastic_residue_haul"]["construction"]
    recipe = physical_material_recipe()
    film_yield = next(
        x.finished_kg_per_native_service_kg
        for x in recipe.conversion
        if x.operation == "conversion_film"
    )
    recovered = next(
        x.recovered_kg_per_input_kg for x in recipe.mechanical_recovery if x.name == "tps_compound"
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="tps",
        biosphere_database="biosphere",
        source_database="original",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "water"): {
                    "name": "Water",
                    "unit": "cubic meter",
                    "categories": ("air",),
                    "type": "emission",
                },
                ("biosphere", "co2"): {
                    "name": "Carbon dioxide, fossil",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
            },
            searchable=False,
        )

        def inventory(database):
            """Build complete small selected operations with invented source coefficients."""
            data = {}

            def ex(code, amount, *, biosphere=False):
                """Retain a distinct distribution location as an exchange-metadata witness."""
                return {
                    "input": ("biosphere" if biosphere else database, code),
                    "type": "biosphere" if biosphere else "technosphere",
                    "amount": amount,
                    "uncertainty type": 3,
                    "loc": amount + 0.013,
                    "scale": 0.002,
                }

            def add(code, selector, production, exchanges):
                """Supply one selected native dataset and complete vector."""
                data[(database, code)] = {
                    "name": selector["name"],
                    "reference product": selector["reference_product"],
                    "unit": selector["unit"],
                    "location": selector["exact_location"],
                    "type": "process",
                    "exchanges": [
                        {"input": (database, code), "type": "production", "amount": production},
                        *exchanges,
                    ],
                }

            for role, code, production in (
                ("ldpe", "ldpe", 1),
                ("bagging_electricity", "electricity", 1),
                ("packing_cutoff", "cutoff", 3),
            ):
                add(
                    code,
                    sack[role]["selector"],
                    production,
                    [] if code == "cutoff" else [ex("co2", 0.07, biosphere=True)],
                )
            for code, component in (
                ("facility", native_recovery["facility"]),
                ("water", native_recovery["fresh_water"]),
            ):
                add(code, component["selector"], 1, [ex("co2", 0.11, biosphere=True)])
            for code, name, product, unit, location, production in (
                (
                    "factory",
                    "market for plastic processing factory",
                    "plastic processing factory",
                    "unit",
                    "GLO",
                    1,
                ),
                (
                    "waste",
                    "market for waste plastic, mixture",
                    "waste plastic, mixture",
                    "kilogram",
                    "FR",
                    -5,
                ),
                (
                    "bag_electricity",
                    "market for electricity, medium voltage",
                    "electricity, medium voltage",
                    "kilowatt hour",
                    "FR",
                    1,
                ),
                (
                    "auxiliary",
                    "invented film auxiliary",
                    "invented film auxiliary",
                    "kilogram",
                    "RER",
                    1,
                ),
                ("lorry", "invented dated lorry", "invented freight", "ton kilometer", "EUR", 1),
                ("rail", "invented dated rail", "invented rail freight", "ton kilometer", "EUR", 1),
            ):
                add(
                    code,
                    {
                        "name": name,
                        "reference_product": product,
                        "unit": unit,
                        "exact_location": location,
                    },
                    production,
                    [ex("co2", 0.19, biosphere=True)],
                )
            original = database == "original"
            bag_production = 2 if original or year == 2025 else 4
            add(
                "bag",
                sack["bag_making"]["selector"],
                bag_production,
                [
                    ex("bag_electricity", 0.12),
                    ex("factory", 0.008),
                    ex("waste", -0.1 if original else -0.2),
                    ex("cutoff", -0.3 if original else -0.4),
                ],
            )
            add(
                "film",
                sack["film"]["selector"],
                5,
                [
                    ex("auxiliary", 0.3),
                    ex("factory", 0.01),
                    ex("waste", -0.02),
                    ex("electricity", 0.9),
                    ex("co2", 0.05, biosphere=True),
                ],
            )
            omitted = freight["omit_technosphere"][0]["selector"]
            add(
                "donor_pellets",
                {**omitted, "exact_location": omitted["location"]},
                1,
                [ex("co2", 100, biosphere=True)],
            )
            add(
                "freight",
                freight["base"]["selector"],
                2,
                [
                    ex("donor_pellets", 2),
                    ex("lorry", 0.4 if year == 2025 else 0.7),
                    ex("rail", 0.2),
                ],
            )
            for index, selector in enumerate(residue_haul["omit_disposal"]):
                add(
                    f"excluded_residue_disposal_{index}",
                    {**selector, "exact_location": selector["location"]},
                    -1,
                    [ex("co2", 100, biosphere=True)],
                )
            # Invented -4 kg waste reference and dated 0.8/1.2 tkm duties expose
            # both the negative-production normalization and the actual wet mass.
            add(
                "residue_market",
                residue_haul["base"]["selector"],
                -4,
                [
                    ex("lorry", 0.8 if year == 2025 else 1.2),
                    *(ex(f"excluded_residue_disposal_{index}", -1) for index in range(3)),
                ],
            )
            return data

        bd.Database("original").write(inventory("original"), searchable=False)
        native = inventory("native")
        bd.Database("native").write(native, searchable=False)
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database="native",
            technology_database="native",
            provider=provider,
            material_recipe=recipe,
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        delivery = bd.get_node(
            database=frontier.database_name,
            code=provider_service_code("recycling_pet_pellet_delivery"),
        )
        expected_freight = {
            x["input"]: x
            for x in native[("native", "freight")]["exchanges"]
            if x["type"] == "technosphere" and x["input"] != ("native", "donor_pellets")
        }
        for exchange in delivery.technosphere():
            expected = expected_freight[exchange.input.key]
            for field in ("input", "amount", "uncertainty type", "loc", "scale"):
                assert exchange[field] == expected[field]
        (delivery_production,) = tuple(delivery.production())
        assert delivery_production["amount"] == 2
        p_bag = 2 if year == 2025 else 4

        for formulation, k, water in (("native", 1.1, 0.1), ("synthetic", 1.2, 0.15)):
            parent = bd.get_node(
                database=frontier.database_name,
                code=provider_service_code(f"mechanical_recycling_tps_{formulation}"),
            )
            amounts = {ex.input.key: ex["amount"] for ex in parent.technosphere()}
            delivered = recovered * k
            # The TPS fixture has no air/water principal exits. The managed
            # remainder travels with its original technical moisture/auxiliaries.
            residue = (1 - recovered) * k
            assert amounts[
                (frontier.database_name, provider_service_code("plastic_residue_haul"))
            ] == pytest.approx(residue)
            packing = delivered * (0.1 / 24.9)
            gross_bag = packing / 0.8
            gross_film = gross_bag / film_yield
            bagging_electricity = delivered * 0.00365
            cargo = delivered + 2 * packing
            assert amounts[("native", "bag")] == pytest.approx(gross_bag)
            assert amounts[("native", "film")] == pytest.approx(gross_film)
            assert amounts[("native", "ldpe")] == pytest.approx(gross_film)
            assert amounts[("native", "electricity")] == pytest.approx(
                3.54 * delivered + bagging_electricity
            )
            assert amounts[("native", "water")] == water
            assert amounts[("native", "facility")] == pytest.approx(2e-9 * k)
            assert amounts[("native", "cutoff")] == pytest.approx(-packing)
            assert fsum(ex["amount"] for ex in parent.biosphere()) == pytest.approx(water / 1000)
            calculation = LCA({parent: 1.0})
            calculation.lci()
            supply_array = cast(NDArray[np.float64] | None, calculation.supply_array)
            assert supply_array is not None

            def supply(code, database="native"):
                """Read the solved activity level of one linked fixture process."""
                node = bd.get_node(database=database, code=code)
                return cast(NDArray[np.float64], supply_array)[calculation.dicts.activity[node.id]]

            assert supply("bag") == pytest.approx(gross_bag / p_bag, rel=3e-7)
            assert supply("factory") == pytest.approx(
                gross_bag / p_bag * 0.008 + gross_film / 5 * 0.01, rel=3e-7
            )
            assert supply("auxiliary") == pytest.approx(gross_film / 5 * 0.3, rel=3e-7)
            assert supply("bag_electricity") == pytest.approx(gross_bag / p_bag * 0.12, rel=3e-7)
            assert supply("waste") == pytest.approx(
                (gross_bag / p_bag * -0.2 + gross_film / 5 * -0.02) / -5, rel=3e-7
            )
            assert supply("cutoff") == pytest.approx(
                (-packing + gross_bag / p_bag * -0.4) / 3, rel=3e-7
            )
            assert supply("lorry") == pytest.approx(
                cargo / 2 * (0.4 if year == 2025 else 0.7)
                + residue / 4 * (0.8 if year == 2025 else 1.2),
                rel=3e-7,
            )
            assert supply("rail") == pytest.approx(cargo / 2 * 0.2, rel=3e-7)
            assert supply("donor_pellets") == 0
            assert all(supply(f"excluded_residue_disposal_{index}") == 0 for index in range(3))

        for origin, k, _ in formulations:
            virgin = bd.get_node(
                database=frontier.database_name, code=provider_service_code(f"tps_{origin}")
            )
            assert virgin["provider_stage_owner"] == "virgin_resin_production"
            virgin_amounts = {ex.input.key: ex["amount"] for ex in virgin.technosphere()}
            assert virgin_amounts[("native", "facility")] == pytest.approx(
                1.23625 * k / (10000 * 1000 * 50), rel=1e-12, abs=0
            )
            virgin_packing = k * 0.1 / 24.9
            assert virgin_amounts[("native", "bag")] == pytest.approx(virgin_packing / 0.8)
            assert virgin_amounts[("native", "film")] == pytest.approx(
                virgin_packing / 0.8 / film_yield
            )
            assert virgin_amounts[("native", "ldpe")] == pytest.approx(
                virgin_packing / 0.8 / film_yield
            )
            assert virgin_amounts[("native", "cutoff")] == pytest.approx(-virgin_packing)
            assert virgin_amounts[("native", "electricity")] == pytest.approx(
                3.54 * k + k * 0.00365
            )
            assert virgin_amounts[
                (frontier.database_name, provider_service_code("recycling_pet_pellet_delivery"))
            ] == pytest.approx(k + 2 * virgin_packing)
