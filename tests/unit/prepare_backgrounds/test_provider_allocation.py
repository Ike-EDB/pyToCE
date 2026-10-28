"""Source allocation survives recursive compilation into product inventories."""

from pathlib import Path
from typing import Any, cast

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


def test_mta_and_b3_allocation_reaches_complete_product_exchanges(tmp_path: Path) -> None:
    """Allocate inventory exchanges once, retaining feed origins and waste signs."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        s["provider_service_id"]: s for s in cast(list[dict[str, Any]], catalogue["services"])
    }
    b3 = services["ethanol_to_butadiene_b3_operation"]["construction"]
    mta = services["benzene_biogenic"]["construction"]
    nodes = {}
    native_keys = {}
    selector_keys = {}

    def add_supplier(code: str, selector: dict, production: float = 1.0) -> tuple[str, str]:
        """Create an invented native boundary once for each exact selected product."""
        identity = (
            selector["name"],
            selector["reference_product"],
            selector["unit"],
            (
                selector["exact_location"]
                if "exact_location" in selector
                else selector["preferred_locations"][0]
            ),
        )
        if identity in selector_keys:
            return selector_keys[identity]
        key = ("native", code)
        node = _process(
            *key,
            identity[0],
            identity[1],
            location=identity[3],
            production_amount=production,
        )
        node["unit"] = identity[2]
        nodes[key] = node
        selector_keys[identity] = key
        return key

    for construction in (b3, mta):
        for component in construction["components"]:
            if "selector" in component:
                native_keys[component["component_id"]] = add_supplier(
                    component["component_id"],
                    component["selector"],
                    -1.0 if component["amount"] < 0 else 1.0,
                )
    for name in ("ethanol_biogenic", "methanol_biogenic", "methanol_atmospheric_dac"):
        native_keys[name] = add_supplier(name, services[name]["construction"]["selector"])
    hydration = services["ethanol_atmospheric_dac"]["construction"]
    olefins = services["ethylene_atmospheric_dac"]["construction"]
    hydration_key = add_supplier("hydration", hydration["base"]["selector"])
    olefins_key = add_supplier("olefins", olefins["base"]["selector"])
    ethylene_key = add_supplier("ethylene", hydration["replace_technosphere"][0]["selector"])
    methanol_key = add_supplier("methanol", olefins["replace_technosphere"][0]["selector"])
    coal_methanol_key = add_supplier(
        "coal_methanol", olefins["replace_technosphere"][1]["selector"]
    )

    # This allocation fixture supplies a small complete methanol market. Its
    # invented 40/60 producer mix sums to one; delivery is 0.2 tonne-km/kg.
    # The separate coal purchase above remains outside this market function.
    delivery = services["methanol_delivered_cn_atmospheric_dac"]["construction"]
    for index, rule in enumerate(delivery["replace_technosphere"]):
        selector = dict(rule["selector"])
        selector["exact_location"] = selector.pop("location")
        supplier = add_supplier(f"market_methanol_{index}", selector)
        nodes[methanol_key]["exchanges"].append(
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
    nodes[methanol_key]["exchanges"].append(
        {"input": freight, "amount": 0.2, "type": "technosphere"}
    )

    flows = {}
    for code, name, categories in (
        ("co2_f", "Carbon dioxide, fossil", ("air",)),
        ("co2_nf", "Carbon dioxide, non-fossil", ("air",)),
        ("co_f", "Carbon monoxide, fossil", ("air",)),
        ("co_nf", "Carbon monoxide, non-fossil", ("air",)),
        ("co2_urban_f", "Carbon dioxide, fossil", ("air", "urban air close to ground")),
        ("co2_urban_nf", "Carbon dioxide, non-fossil", ("air", "urban air close to ground")),
        ("water", "Water", ("air",)),
        ("oxygen", "Oxygen", ("air",)),
        ("nitrogen", "Nitrogen", ("air",)),
    ):
        flows[("biosphere", code)] = {
            "name": name,
            "unit": "cubic meter" if code == "water" else "kilogram",
            "categories": categories,
            "type": "emission",
        }
    # Synthetic hydration/MTO coefficients exercise two operation replacements and DAC feed
    # recursion.
    nodes[hydration_key]["exchanges"].extend(
        [
            {"input": ethylene_key, "amount": 1.25, "type": "technosphere"},
            {"input": ("biosphere", "co2_urban_f"), "amount": 0.2, "type": "biosphere"},
        ]
    )
    nodes[olefins_key]["exchanges"].extend(
        [
            {"input": methanol_key, "amount": 2.0, "type": "technosphere"},
            {"input": coal_methanol_key, "amount": 0.5, "type": "technosphere"},
            {"input": ("biosphere", "co2_f"), "amount": 0.3, "type": "biosphere"},
            {"input": ("biosphere", "co_f"), "amount": 0.04, "type": "biosphere"},
        ]
    )
    allocation = 1208 / (1208 + 0.0477 * 599 + 0.035 * 852 + 0.0147 * 1024 + 0.0403 * 1960)
    # The catalog test independently checks every gross source coefficient.
    # Here the complete expected vector tests their composition at the product owner.
    operation_tech = {
        (native_keys[c["component_id"]], "technosphere"): c["amount"] for c in b3["components"]
    }
    octane_co2 = 0.0357 * (8 * 44.009) / (8 * 12.011 + 18 * 1.008)
    expected_bio = {
        (("biosphere", "co2_f"), "biosphere"): (0.6787 + octane_co2) * allocation,
        (("biosphere", "co2_nf"), "biosphere"): (0.8452 - octane_co2) * allocation,
        (("biosphere", "water"), "biosphere"): 0.0009901 * allocation,
        (("biosphere", "oxygen"), "biosphere"): 2.2293 * allocation,
        (("biosphere", "nitrogen"), "biosphere"): 13.5795 * allocation,
    }
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="product_allocation",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(flows, searchable=False)
        bd.Database("native").write(nodes, searchable=False)
        original = {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for node in bd.Database("native")
        }
        frontier = _prepare_selected_native_frontier(
            bd, catalogue, ("butadiene_biogenic", "butadiene_atmospheric_dac", "benzene_biogenic")
        )
        products = {}
        for product in ("butadiene_biogenic", "butadiene_atmospheric_dac", "benzene_biogenic"):
            key = (frontier.database_name, provider_catalogue.provider_service_code(product))
            expected = {}
            if product.startswith("butadiene"):
                expected.update(operation_tech)
                expected.update(expected_bio)
                water_key = (native_keys["softened_water"], "technosphere")
                expected[water_key] += 2.4245 * 0.07 * allocation
                ethanol = 2.4245 * 0.93 * allocation
                if product == "butadiene_biogenic":
                    expected[(native_keys["ethanol_biogenic"], "technosphere")] = ethanol
                else:
                    expected[(native_keys["methanol_atmospheric_dac"], "technosphere")] = (
                        ethanol * 1.25 * 2.5
                    )
                    # Two kg of market methanol receive freight; the extra
                    # 0.5 kg direct coal-methanol handoff receives none.
                    expected[(freight, "technosphere")] = ethanol * 1.25 * 2.0 * 0.2
                    expected[(("biosphere", "co2_urban_nf"), "biosphere")] = ethanol * 0.2
                    expected[(("biosphere", "co2_nf"), "biosphere")] += ethanol * 1.25 * 0.3
                    expected[(("biosphere", "co_nf"), "biosphere")] = ethanol * 1.25 * 0.04
            else:
                expected.update(
                    {
                        (native_keys["methanol_biogenic"], "technosphere"): 2.351603174033248,
                        (native_keys["mta_electricity"], "technosphere"): (
                            24_091_588 / 49_284 / 3600 + 79_506_982 / 17_902 / 3600
                        ),
                        (native_keys["plant"], "technosphere"): 0.000015 / 49_284,
                        (native_keys["zeolite"], "technosphere"): 33.645 / 49_284,
                        (
                            native_keys["btx_process_water_best_available_comparator"],
                            "technosphere",
                        ): 14_482 / 17_902,
                        (("biosphere", "co2_nf"), "biosphere"): 1500 / 49_284,
                    }
                )
            products[key] = expected
        for key, expected in products.items():
            stored = bd.get_node(database=key[0], code=key[1])
            assert stored["unit"] == "kilogram"
            _assert_native_inventory(bd, stored, expected)
        assert {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for node in bd.Database("native")
        } == original


def test_oxime_and_pa6_preserve_allocation_compartments_and_material_proxies(
    tmp_path: Path,
) -> None:
    """Compile precursor chains and preserve allocated energy and vents."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        s["provider_service_id"]: s for s in cast(list[dict[str, Any]], catalogue["services"])
    }
    names = ("polyamide_6_resin", "caprolactam", "cyclohexanone_oxime", "benzene")
    nodes, native_keys, selector_keys = {}, {}, {}

    def add_supplier(code: str, selector: dict, production: float = 1.0) -> tuple[str, str]:
        """Provide an invented native leaf for each selected product and location."""
        identity = (
            selector["name"],
            selector["reference_product"],
            selector["unit"],
            (
                selector["exact_location"]
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

    for name in names:
        for component in services[f"{name}_biogenic"]["construction"]["components"]:
            if "selector" in component:
                native_keys[component["component_id"]] = add_supplier(
                    component["component_id"],
                    component["selector"],
                    -1 if component["amount"] < 0 else 1,
                )
    for origin in ("biogenic", "atmospheric_dac"):
        name = f"methanol_{origin}"
        native_keys[name] = add_supplier(name, services[name]["construction"]["selector"])
    for name, feed_amount, power, carbon_dioxide in (
        ("cyclohexane", 1.2, 0.01, 0.02),
        ("cyclohexanone", 1.3, 0.03, 0.04),
    ):
        construction = services[f"{name}_biogenic"]["construction"]
        donor = add_supplier(name, construction["base"]["selector"])
        # Benzene uses its RER market; cyclohexanone receives the selected
        # complete GLO cyclohexane market, independently of producer geography.
        feed = add_supplier(
            f"{name}_fossil_feed",
            {
                **construction["replace_technosphere"][0]["selector"],
                "exact_location": "RER" if name == "cyclohexane" else "GLO",
            },
        )
        # Distinct synthetic donor quantities exercise replacement and retained burdens.
        nodes[donor]["exchanges"].extend(
            [
                {"input": feed, "type": "technosphere", "amount": feed_amount},
                {"input": native_keys["electricity"], "type": "technosphere", "amount": power},
                {
                    "input": ("biosphere", "co2_f_urban"),
                    "type": "biosphere",
                    "amount": carbon_dioxide,
                },
            ]
        )
        if name == "cyclohexane":
            # The receiving benzene market has a nonunit reference and unequal
            # producer total; its complete delivery must reach the PA6 owner.
            nodes[feed]["exchanges"][0]["amount"] = 2.0
            delivery = services["benzene_delivered_rer_biogenic"]["construction"]
            for index, (rule, amount) in enumerate(
                zip(delivery["replace_technosphere"], (0.25, 0.5, 0.75, 0.625), strict=True)
            ):
                selector = rule["selector"]
                producer = add_supplier(
                    f"benzene_producer_{index}",
                    {**selector, "exact_location": selector["location"]},
                )
                nodes[feed]["exchanges"].append(
                    {"input": producer, "type": "technosphere", "amount": amount}
                )
            native_keys["benzene_delivery_freight"] = add_supplier(
                "benzene_delivery_freight",
                {
                    "name": "invented benzene delivery freight",
                    "reference_product": "freight",
                    "unit": "ton kilometer",
                    "exact_location": "RER",
                },
            )
            nodes[feed]["exchanges"].append(
                {
                    "input": native_keys["benzene_delivery_freight"],
                    "type": "technosphere",
                    "amount": 0.125,
                }
            )
        else:
            # The two native producer rows remain separate and normalize to
            # one kg; an invented freight marker exposes lost procurement.
            nodes[feed]["exchanges"][0]["amount"] = 3.0
            delivery = services["cyclohexane_delivered_biogenic"]["construction"]
            for index, (rule, amount) in enumerate(
                zip(delivery["replace_technosphere"], (1.25, 1.75), strict=True)
            ):
                selector = rule["selector"]
                producer = add_supplier(
                    f"cyclohexane_producer_{index}",
                    {**selector, "exact_location": selector["location"]},
                )
                nodes[feed]["exchanges"].append(
                    {"input": producer, "type": "technosphere", "amount": amount}
                )
            native_keys["cyclohexane_delivery_freight"] = add_supplier(
                "cyclohexane_delivery_freight",
                {
                    "name": "invented cyclohexane delivery freight",
                    "reference_product": "freight",
                    "unit": "ton kilometer",
                    "exact_location": "GLO",
                },
            )
            nodes[feed]["exchanges"].append(
                {
                    "input": native_keys["cyclohexane_delivery_freight"],
                    "type": "technosphere",
                    "amount": 0.09,
                }
            )
    # PA6 purchases finished caprolactam. Procurement must remain outside
    # the caprolactam factory's coproduct-allocation factor.
    delivery = services["caprolactam_delivered_biogenic"]["construction"]
    market = add_supplier("caprolactam_procurement", delivery["base"]["selector"], 2.0)
    selector = delivery["replace_technosphere"][0]["selector"]
    manufacturer = add_supplier(
        "hmda_procurement_source", {**selector, "exact_location": selector["location"]}
    )
    native_keys["caprolactam_delivery_freight"] = add_supplier(
        "caprolactam_delivery_freight",
        {
            "name": "invented caprolactam delivery freight",
            "reference_product": "freight",
            "unit": "ton kilometer",
            "exact_location": "RER",
        },
    )
    nodes[market]["exchanges"].extend(
        [
            {"input": manufacturer, "type": "technosphere", "amount": 2.0},
            {
                "input": native_keys["caprolactam_delivery_freight"],
                "type": "technosphere",
                "amount": 0.16,
            },
        ]
    )
    descriptors = (
        (
            "nmvoc_stack",
            "NMVOC, non-methane volatile organic compounds",
            ("air", "non-urban air or from high stacks"),
        ),
        ("nmvoc_air", "NMVOC, non-methane volatile organic compounds", ("air",)),
        ("benzene_stack", "Benzene", ("air", "non-urban air or from high stacks")),
        ("co2_f_urban", "Carbon dioxide, fossil", ("air", "urban air close to ground")),
        ("co2_nf_urban", "Carbon dioxide, non-fossil", ("air", "urban air close to ground")),
        ("co2_nf_air", "Carbon dioxide, non-fossil", ("air",)),
    )
    flows = {
        ("biosphere", code): {
            "name": name,
            "categories": categories,
            "unit": "kilogram",
            "type": "emission",
        }
        for code, name, categories in descriptors
    }
    project = BrightwayNames(
        store=tmp_path / "brightway",
        project="polyamide_allocation",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(project) as bd:
        bd.Database("biosphere").write(flows, searchable=False)
        bd.Database("native").write(nodes, searchable=False)
        original = {n.key: [dict(e) for e in n.exchanges()] for n in bd.Database("native")}
        frontier = _prepare_selected_native_frontier(
            bd,
            catalogue,
            tuple(f"polyamide_6_resin_{origin}" for origin in ("biogenic", "atmospheric_dac")),
        )
        products = {}

        def add_native(vector: dict, component: str, amount: float) -> None:
            """Accumulate an independently specified material or utility requirement."""
            key = (native_keys[component], "technosphere")
            vector[key] = vector.get(key, 0.0) + amount

        for origin in ("biogenic", "atmospheric_dac"):
            # MTA input quantities are checked independently in the catalog tests.
            # Here the expected downstream inventory applies explicit source factors.
            benzene = {}
            for c in services[f"benzene_{origin}"]["construction"]["components"]:
                add_native(benzene, c.get("component_id", c.get("service_id")), c["amount"])
            benzene[(("biosphere", "co2_nf_air"), "biosphere")] = 0.030435841246652058
            benzene = {key: value * 2.125 / 2 for key, value in benzene.items()}
            add_native(benzene, "benzene_delivery_freight", 0.125 / 2)
            cyclohexane = {k: 1.2 * v for k, v in benzene.items()}
            add_native(cyclohexane, "electricity", 0.01)
            cyclohexane[(("biosphere", "co2_nf_urban"), "biosphere")] = 0.02
            ketone = {k: 1.3 * v for k, v in cyclohexane.items()}
            add_native(ketone, "cyclohexane_delivery_freight", 1.3 * 0.09 / 3)
            add_native(ketone, "electricity", 0.03)
            ketone[(("biosphere", "co2_nf_urban"), "biosphere")] += 0.04
            oxime = {k: 0.5377332979851538 * v for k, v in ketone.items()}
            for component, amount in (
                ("hydroxylamine", 0.18097030752916227),
                ("sulfuric_acid", 0.26866666666666666),
                ("ammonia", 0.0933123011664899),
                ("electricity", 151 / 1.02 / 3600),
                ("steam", 626 / 1.02 / 1000),
            ):
                add_native(oxime, component, amount)
            oxime[(("biosphere", "nmvoc_stack"), "biosphere")] = 0.62 * 0.000024
            allocation = 2500 / (2500 + 1.612 * 180)
            pa6 = {k: allocation * v for k, v in oxime.items()}
            add_native(pa6, "caprolactam_delivery_freight", 0.16 / 2)
            for component, amount in (
                ("sulfuric_acid", 1.0720564757622846),
                ("ammonia", 0.3723051379086162),
                ("electricity", 223 / 1.01 / 3600),
                ("steam", 4899 / 1.01 / 1000),
                ("process_water", 61.67081048091456),
                ("mixed_wastewater_treatment", -0.06167081048091456),
                ("electricity", 0.9719129859188831),
                ("heat_transfer_duty", 0.6377121193886148),
                ("steam", 2.613401131303406),
                ("nitrogen_supply_and_vent", 0.022),
                ("process_and_scrubber_water", 2.0),
                ("wastewater_treatment", -0.002),
                ("hazardous_waste_treatment", -0.00035),
                ("heat_transfer_medium_makeup", 0.0000325),
            ):
                add_native(pa6, component, amount)
            pa6[(("biosphere", "benzene_stack"), "biosphere")] = allocation * 1.3e-6 + 21e-6
            pa6[(("biosphere", "nmvoc_stack"), "biosphere")] += (
                allocation * (0.11e-6 + 10.7e-6) + 179e-6 + 130e-6
            )
            pa6[(("biosphere", "nmvoc_air"), "biosphere")] = 8e-6 + 32.5e-6
            key = (
                frontier.database_name,
                provider_catalogue.provider_service_code(f"polyamide_6_resin_{origin}"),
            )
            products[key] = pa6
            proxy_rows = [
                r
                for r in frontier.technology_rows
                if r["provider_service_id"] == f"polyamide_6_resin_{origin}"
                and r["component_id"] == "heat_transfer_medium_makeup"
            ]
            assert len(proxy_rows) == 1
            assert proxy_rows[0]["semantic_class"] == "explicit_cross_material_analogue"
            assert (
                proxy_rows[0]["activity_name"],
                proxy_rows[0]["reference_product"],
                proxy_rows[0]["native_unit"],
                proxy_rows[0]["location"],
            ) == (
                "market for chemical, organic, aromatics",
                "chemical, organic, aromatics",
                "kilogram",
                "GLO",
            )
        for key, expected in products.items():
            node = bd.get_node(database=key[0], code=key[1])
            assert node["unit"] == "kilogram"
            _assert_native_inventory(bd, node, expected)
        assert {n.key: [dict(e) for e in n.exchanges()] for n in bd.Database("native")} == original
