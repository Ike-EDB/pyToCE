"""Complete cooling utilities and separated aromatic-product allocation."""

from pathlib import Path
from typing import Any, cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import (
    _assert_native_inventory,
    _prepare_selected_native_frontier,
    _process,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.native_recipe import (
    _native_exchange,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import (
    _RecipeContext,
)


def test_xylene_inventories_follow_complete_source_slate_and_product_purity() -> None:
    """Allocate every shared input and process-carbon term to each separated product."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        s["provider_service_id"]: s for s in cast(list[dict[str, Any]], catalogue["services"])
    }
    # Source product kt/year, 2018 EUR/t and nominal carbon mass fractions:
    # ACS Table S14; thesis Chapter 5 Tables S7, S9 and S12. Heavy aromatics
    # use the declared C10H8 surrogate, and C4 uses the reported mass blend.
    slate = (
        (21, 1075, 24 / 28),
        (195, 605, 24 / 30),
        (68, 1016, 36 / 42),
        (641, 605, 36 / 44),
        (921, 960, (2.38 * 48 / 56 + 20.83 * 48 / 58) / (2.38 + 20.83)),
        (240, 1147, 108 / 120),
        (32, 540, 120 / 128),
        (5, 1098, 96 / 106),
        (972, 1105, 96 / 106),
        (535, 1032, 72 / 78),
    )
    revenue = sum(mass * price for mass, price, _ in slate)
    product_mass = sum(mass for mass, _, _ in slate)
    residual_carbon = 8800 * 12 / 32 - sum(mass * carbon for mass, _, carbon in slate)
    for origin in ("biogenic", "atmospheric_dac"):
        for product, mass, price, purity in (
            ("p_xylene", 972, 1105, 1),
            ("o_xylene", 5, 1098, 0.999),
        ):
            service = services[f"{product}_{origin}"]
            allocation = mass * price / revenue
            scale = allocation / mass / purity
            expected = {
                f"methanol_{origin}": 8800 * scale,
                "green_hydrogen": 17 * scale,
                "zeolite_catalyst": 4 * scale,
                "medium_pressure_steam_energy": 4493 * scale,
                "ambient_cooling_operation": 13941 * scale,
                "r134a_refrigeration_energy": 854 * scale,
                "ethylene_refrigeration_energy": 183 * scale,
                "methane_refrigeration_energy": 1163 * scale,
                "electricity": 198 * scale / 3.6,
                "average_wastewater_treatment": -4914 * scale / 1000,
                "spent_zeolite_disposal": -4 * scale,
                # Native generic capital per kg chemical output applies to the
                # whole ten-product slate before revenue allocation and purity.
                "chemical_factory": 4e-10 * product_mass * scale,
            }
            components = service["construction"]["components"]
            assert {
                c.get("component_id", c.get("service_id")): c["amount"] for c in components
            } == pytest.approx(expected, rel=1e-13, abs=0)
            assert service["representative_method"]["economic_allocation_factor"] == pytest.approx(
                allocation, rel=1e-14
            )
            direct = service["construction"]["biosphere_terms"]
            assert len(direct) == 1
            assert direct[0]["selector"] == {
                "name": "Carbon dioxide, non-fossil",
                "unit": "kilogram",
                "categories": ["air"],
            }
            assert direct[0]["amount"] == pytest.approx(
                residual_carbon * 44 / 12 * scale, rel=1e-13
            )
            for c in components:
                if "refrigeration_energy" in c.get("component_id", ""):
                    temperature = {
                        "r134a_refrigeration_energy": -25,
                        "ethylene_refrigeration_energy": -100,
                        "methane_refrigeration_energy": -160,
                    }[c["component_id"]]
                    assert c["selector"] == {
                        "name": f"market for cooling energy, at {temperature} °C",
                        "reference_product": f"cooling energy, at {temperature} °C",
                        "unit": "megajoule",
                        "exact_location": "GLO",
                    }
        ox = services[f"o_xylene_{origin}"]["representative_method"]
        assert ox["source_product_mass_fraction"] == 0.999
        assert ox["source_capacity_kt_per_year"] == 5
    binding = cast(dict[str, Any], catalogue["method_authority_registry"])["exact_source_bindings"]
    assert (
        "ioannou_2023_integrated_co2_refinery"
        not in binding["krestnikova_2026_atmospheric_glucose_lactic_acid_pla"]
    )
    closures = provider_catalogue._service_source_closures(catalogue)
    for service in ("lactic_acid_atmospheric_dac", "p_xylene_biogenic", "o_xylene_atmospheric_dac"):
        assert "medrano_garcia_2023_vcm" in closures[service]


@pytest.mark.parametrize("year", (2025, 2030))
def test_cooling_and_aromatic_products_compile_complete_native_exchanges(
    tmp_path: Path, year: int
) -> None:
    """Preserve MJ cooling, VCM totals, both xylene origins and separate LA suppliers."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        s["provider_service_id"]: s for s in cast(list[dict[str, Any]], catalogue["services"])
    }
    utility = services["ambient_cooling_operation"]
    assert utility["unit"] == "megajoule"
    px = services["p_xylene_biogenic"]["construction"]
    vcm = services["vinyl_chloride_biogenic"]["construction"]
    nodes, native_keys, selector_keys = {}, {}, {}

    def add_supplier(code: str, selector: dict, production: float = 1.0) -> tuple[str, str]:
        """Admit one invented process for each exact selected native product."""
        identity = (
            selector["name"],
            selector["reference_product"],
            selector["unit"],
            (
                selector["location_by_period"]["baseline" if year == 2025 else "future"]
                if "location_by_period" in selector
                else selector["exact_location"]
                if "exact_location" in selector
                else selector["preferred_locations"][0]
            ),
        )
        if identity in selector_keys:
            return selector_keys[identity]
        key = ("native", code)
        node = _process(
            *key, identity[0], identity[1], location=identity[3], production_amount=production
        )
        node["unit"] = identity[2]
        nodes[key] = node
        selector_keys[identity] = key
        return key

    for construction in (utility["construction"], px, vcm):
        for c in construction["components"]:
            if "selector" in c:
                native_keys[c["component_id"]] = add_supplier(
                    c["component_id"], c["selector"], -1 if c["amount"] < 0 else 1
                )
    for origin in ("biogenic", "atmospheric_dac"):
        name = f"methanol_{origin}"
        native_keys[name] = add_supplier(name, services[name]["construction"]["selector"])
    process_power = next(
        c
        for c in services["lactic_acid_atmospheric_dac"]["construction"]["components"]
        if c.get("component_id") == "process_electricity"
    )
    assert process_power["selector"]["exact_location"] == "RER"
    rer_key = add_supplier("rer_power", process_power["selector"])
    power_selector = utility["construction"]["components"][1]["selector"]
    add_supplier(
        "other_period_power",
        {
            "name": power_selector["name"],
            "reference_product": power_selector["reference_product"],
            "unit": power_selector["unit"],
            "exact_location": "EUR" if year == 2025 else "RER",
        },
    )
    assert utility["construction"]["components"][1]["selector"]["location_by_period"] == {
        "baseline": "RER",
        "future": "EUR",
    }
    assert utility["construction"]["components"][0]["selector"] == {
        "name": "market group for tap water",
        "reference_product": "tap water",
        "unit": "kilogram",
        "exact_location": "RER",
    }
    olefins = services["ethylene_biogenic"]["construction"]
    donor = add_supplier("olefins", olefins["base"]["selector"])
    fossil_methanol = [
        add_supplier(f"fossil_methanol_{i}", rule["selector"])
        for i, rule in enumerate(olefins["replace_technosphere"])
    ]
    # VCM's ethylene precursor buys one methanol share through its complete CN
    # market. The separate coal-manufacturer input is an internal handoff.
    # This invented market retains both producers and 0.2 tonne-km/kg freight.
    delivery = services["methanol_delivered_cn_biogenic"]["construction"]
    for index, rule in enumerate(delivery["replace_technosphere"]):
        selector = dict(rule["selector"])
        selector["exact_location"] = selector.pop("location")
        supplier = add_supplier(f"market_methanol_{index}", selector)
        nodes[fossil_methanol[0]]["exchanges"].append(
            {"input": supplier, "amount": (0.4, 0.6)[index], "type": "technosphere"}
        )
    freight = add_supplier(
        "methanol_freight",
        {
            "name": "invented methanol freight",
            "reference_product": "invented methanol freight",
            "unit": "ton kilometer",
            "exact_location": "CN",
        },
    )
    nodes[fossil_methanol[0]]["exchanges"].append(
        {"input": freight, "amount": 0.2, "type": "technosphere"}
    )
    # Invented coefficients exercise the retained MTO replacement through each
    # actual origin-specific methanol service; no licensed exchanges are copied.
    nodes[donor]["exchanges"].extend(
        [
            {"input": fossil_methanol[0], "amount": 2.0, "type": "technosphere"},
            {"input": fossil_methanol[1], "amount": 0.5, "type": "technosphere"},
            {"input": ("biosphere", "co2_f"), "amount": 0.3, "type": "biosphere"},
            {"input": ("biosphere", "co_f"), "amount": 0.04, "type": "biosphere"},
        ]
    )
    flows = {}
    for code, name, category in (
        ("water_air", "Water", "air"),
        ("water_water", "Water", "water"),
        ("co2_f", "Carbon dioxide, fossil", "air"),
        ("co2_nf", "Carbon dioxide, non-fossil", "air"),
        ("co_f", "Carbon monoxide, fossil", "air"),
        ("co_nf", "Carbon monoxide, non-fossil", "air"),
    ):
        flows[("biosphere", code)] = {
            "name": name,
            "unit": "cubic meter" if name == "Water" else "kilogram",
            "categories": (category,),
            "type": "emission",
        }
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="shared_cooling",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(flows, searchable=False)
        bd.Database("native").write(nodes, searchable=False)
        original = {n.key: [dict(e) for e in n.exchanges()] for n in bd.Database("native")}
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=year,
            provider_database="provider_frontier",
            source=_ActivityResolver(bd, "native"),
            ordinary=_ActivityResolver(bd, "native"),
            technology=_ActivityResolver(bd, "native"),
            biosphere=_FlowResolver(bd, "biosphere"),
            material_recipe=physical_material_recipe(),
            catalogue=catalogue,
            method_authorities={
                cast(str, r["method_authority_id"]): r
                for r in provider_catalogue._method_authority_rows(catalogue)
            },
            complete_source_ids=provider_catalogue._service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        # Printed water returns total 0.612 kg/MJ versus 0.611 kg supplied;
        # preserve the source rounding instead of enforcing an exact balance.
        cooling_vector = {
            (native_keys["tap_water"], "technosphere"): 0.611,
            (native_keys["high_voltage_electricity"], "technosphere"): 0.00166,
            (("biosphere", "water_air"), "biosphere"): 0.000408,
            (("biosphere", "water_water"), "biosphere"): 0.000204,
        }
        selected = tuple(
            f"{product}_{origin}"
            for product in ("p_xylene", "o_xylene", "vinyl_chloride")
            for origin in ("biogenic", "atmospheric_dac")
        )
        frontier = _prepare_selected_native_frontier(bd, catalogue, selected, year=year)
        compiled = {}
        for name in selected:
            key = (frontier.database_name, provider_catalogue.provider_service_code(name))
            product = bd.get_node(database=key[0], code=key[1])
            expected = {}
            if name.startswith("vinyl_chloride"):
                origin = name.removeprefix("vinyl_chloride_")
                # Independent pre-extraction source totals, including all cooling.
                expected.update(
                    {
                        (native_keys[f"methanol_{origin}"], "technosphere"): 0.453 * 2.5,
                        # The two kg market share carries incoming freight. The 0.5 kg coal share
                        # uses the manufacturer directly.
                        (freight, "technosphere"): 0.453 * 2.0 * 0.2,
                        (native_keys["chlorine"], "technosphere"): 0.571,
                        (native_keys["oxygen"], "technosphere"): 0.135,
                        (native_keys["tap_water"], "technosphere"): 5.2991895,
                        (native_keys["natural_gas_heat"], "technosphere"): 2.38,
                        (native_keys["high_voltage_electricity"], "technosphere"): 0.53225387,
                        (("biosphere", "water_air"), "biosphere"): 0.003282156,
                        (("biosphere", "water_water"), "biosphere"): 0.006051078,
                        (("biosphere", "co2_nf"), "biosphere"): 0.0178 + 0.453 * 0.3,
                        (("biosphere", "co_nf"), "biosphere"): 0.453 * 0.04,
                    }
                )
            else:
                origin = name.split("_", 2)[2]
                multiplier = (1098 / 1105) / 0.999 if name.startswith("o_") else 1
                for c in px["components"]:
                    if c.get("service_id") == "ambient_cooling_operation":
                        for exchange, amount in cooling_vector.items():
                            expected[exchange] = c["amount"] * multiplier * amount
                    else:
                        supplier = (
                            native_keys[f"methanol_{origin}"]
                            if "service_id" in c
                            else native_keys[c["component_id"]]
                        )
                        expected[(supplier, "technosphere")] = c["amount"] * multiplier
                expected[(("biosphere", "co2_nf"), "biosphere")] = 0.182708833924356 * multiplier
            assert product["unit"] == "kilogram"
            compiled[key] = expected
        # The lactic-acid fixture uses the RER process-electricity supplier.
        la_power = {}
        _native_exchange(
            context,
            process_power,
            service_id="lactic_acid_atmospheric_dac",
            scale=1,
            exchanges=la_power,
        )
        assert la_power == {(rer_key, "technosphere"): 0.00025685}
        assert (rer_key == native_keys["high_voltage_electricity"]) == (year == 2025)
        for key, expected in compiled.items():
            node = bd.get_node(database=key[0], code=key[1])
            _assert_native_inventory(bd, node, expected)
        assert {n.key: [dict(e) for e in n.exchanges()] for n in bd.Database("native")} == original
