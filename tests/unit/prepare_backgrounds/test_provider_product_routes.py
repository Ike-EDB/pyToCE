"""Focused provider catalog authority contracts."""

import json
from typing import Any, cast

import pytest

from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)


def test_pur_routes_preserve_polymer_carbon_custody_and_application_weights() -> None:
    """Apply PUR application weights to complete polyol and isocyanate supply chains."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    expected_components = {
        "biogenic": [
            {"service_id": "pur_flexible_biogenic", "amount": 0.44},
            {"service_id": "pur_rigid_biogenic", "amount": 0.32},
            {"service_id": "pur_adhesive_biogenic", "amount": 0.24},
        ],
        "atmospheric_dac": [
            {"service_id": "pur_flexible_atmospheric_dac", "amount": 0.44},
            {"service_id": "pur_rigid_atmospheric_dac", "amount": 0.32},
            {"service_id": "pur_adhesive_atmospheric_dac", "amount": 0.24},
        ],
    }

    for origin, expected in expected_components.items():
        service = services[f"pur_{origin}"]
        assert service["representative_method"]["composition_authority"] == (
            "accepted_pur_composition_authority"
        )
        assert service["construction"]["components"] == expected
        assert sum(component["amount"] for component in expected) == 1.0
        adhesive = services[f"pur_adhesive_{origin}"]["construction"]
        assert adhesive["kind"] == "operation_only"
        assert adhesive["base"]["selector"]["name"] == ("polyurethane adhesive production")
        # Exact delivered feed choices and native supplier quantities are
        # exercised once by test_pur_feed_delivery through real Brightway.
        assert all("amount" not in item for item in adhesive["replacements"])

        flexible = services[f"pur_flexible_{origin}"]["construction"]
        rigid = services[f"pur_rigid_{origin}"]["construction"]
        assert flexible["kind"] == "operation_only"
        assert rigid["kind"] == "operation_only"
        assert flexible["base"]["selector"]["name"] == (
            "polyurethane production, flexible foam, TDI-based, high density"
        )
        assert rigid["base"]["selector"]["name"] == ("polyurethane production, rigid foam")

    assert "polyether_polyol_atmospheric_dac" not in services
    assert "rapeseed_polyol_biogenic" not in services
    for grade in ("long_chain", "short_chain"):
        for origin in ("biogenic", "atmospheric_dac"):
            polyol = services[f"polyether_polyols_{grade}_{origin}"]
            construction = polyol["construction"]
            assert construction["kind"] == "operation_only"
            assert construction["base"]["selector"]["name"] == (
                f"polyether polyols production, {grade.replace('_', ' ')}"
            )
            assert "replace_biosphere" not in construction
            assert "mixed-origin" in polyol["representative_method"]["carbon_claim"]

    rigid_dac = services["pur_rigid_atmospheric_dac"]
    assert (
        "conventional blowing agent" in rigid_dac["representative_method"]["carbon_origin_boundary"]
    )

    pur_conversion = [
        route
        for route in cast(list[dict[str, Any]], catalogue["routes"])
        if route["material"] == "PUR" and route["provider_demand_type"] == "processing_conversion"
    ]
    assert [(r["constituent"], r["provider_service_id"], r["share"]) for r in pur_conversion] == [
        ("pur", "pur_conversion_included_in_finished_products", 1.0)
    ]
    assert services["pur_conversion_included_in_finished_products"]["construction"] == {
        "kind": "project_recipe",
        "components": [],
    }


def test_others_routes_preserve_the_ten_class_plastics_only_portfolio() -> None:
    """Retain all Others classes on their specified plastic-mass basis."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    selected = {
        "adhesives": (
            "pur_adhesive",
            "pur_adhesive",
            "pur_conversion_included_in_finished_products",
        ),
        "general_coatings": ("epoxy_resin", "epoxy_resin", "others_epoxy_coating_conversion"),
        "marine_coatings": ("epoxy_resin", "epoxy_resin", "others_epoxy_coating_conversion"),
        "non_tyre_elastomers": (
            "emulsion_sbr",
            "emulsion_sbr",
            "others_esbr_compounding_conversion",
        ),
        "other_thermoplastics": ("pmma", "pmma", "conversion_injection"),
        "pa": ("nylon_6", "polyamide_6_resin", "conversion_injection"),
        "pc": ("polycarbonate", "polycarbonate", "conversion_injection"),
        "road_marking_coatings": ("pmma", "pmma", "others_road_marking_conversion"),
        "sealants": (
            "pur_adhesive",
            "pur_adhesive",
            "pur_conversion_included_in_finished_products",
        ),
        "thermosets": (
            "unsaturated_polyester",
            "unsaturated_polyester_resin",
            "others_upr_hand_lay_up_conversion",
        ),
    }
    resin_routes = {
        (r["constituent"], r["virgin_feedstock_route"]): r
        for r in cast(list[dict[str, Any]], catalogue["routes"])
        if r["material"] == "Others" and r["provider_demand_type"] == "virgin_resin_production"
    }
    assert set(resin_routes) == {
        (constituent, origin)
        for constituent in selected
        for origin in ("fossil", "biogenic", "atmospheric_dac")
    }
    for (constituent, origin), route in resin_routes.items():
        fossil, adapted, _ = selected[constituent]
        service_id = f"{fossil if origin == 'fossil' else adapted}_{origin}"
        assert route["provider_service_id"] == service_id
        assert route["share"] == 1.0
        assert services[service_id]["foreground_service"] is True
        method = provider_catalogue._method_authority(catalogue, route["method_authority_id"])
        sources = json.loads(cast(str, method["source_authority_ids_json"]))
        assert "accepted_others_composition_authority" in sources
        assert ("us_epa_epoxy_coating_binder" in sources) == (
            constituent in {"general_coatings", "marine_coatings"}
        )
        assert ("txdot_2004_mma_road_marking" in sources) == (
            constituent == "road_marking_coatings"
        )
    conversion_routes = [
        r
        for r in cast(list[dict[str, Any]], catalogue["routes"])
        if r["material"] == "Others" and r["provider_demand_type"] == "processing_conversion"
    ]
    for origin in ("fossil", "biogenic", "atmospheric_dac"):
        # The declared base SBR operation is fossil. Its two constructed
        # counterparts keep the same auxiliaries but route principal carbon
        # through the matching filled-scrap landfill chain. Unlike injection,
        # biogenic and DAC identities therefore remain separate services.
        nonfossil = {
            "conversion_injection": "conversion_injection__non_fossil",
            "others_esbr_compounding_conversion": f"others_esbr_compounding_conversion__{origin}",
            "others_epoxy_coating_conversion": f"others_epoxy_coating_conversion__{origin}",
            "others_road_marking_conversion": f"others_road_marking_conversion__{origin}",
            "others_upr_hand_lay_up_conversion": f"others_upr_hand_lay_up_conversion_{origin}",
        }
        assert [
            (r["constituent"], r["provider_service_id"], r["share"])
            for r in conversion_routes
            if r["ultimate_carbon_origin"] in {"all", origin}
        ] == [
            (
                constituent,
                operation if origin == "fossil" else nonfossil.get(operation, operation),
                1.0,
            )
            for constituent, (_, _, operation) in selected.items()
        ]

    esbr_components = {
        component["component_id"]: component["amount_kg_per_gross_principal_kg"]
        for component in services["others_esbr_compounding_conversion"]["construction"][
            "ingredients"
        ]
    }
    # TRC (1995), vol.II PDF p.21 / EPA AP-42 Table 4.12-2: kg additive
    # per kg gross principal SBR, before source releases and final trimming.
    assert esbr_components == {
        "carbon_black_n330": 0.585,
        "zinc_oxide": 0.1,
        "stearic_acid": 0.02,
        "tmq": 0.02,
        "six_ppd": 0.01,
        "protective_wax": 0.015,
        "processing_oil": 0.07,
        "rubber_makers_sulfur": 0.02,
        "tbbs": 0.018,
    }
    assert services["others_esbr_compounding_conversion"]["construction"]["kind"] == (
        "sbr_compounding"
    )

    # The shared material owner supplies actual application quantities; each
    # principal origin has its own container treatment. Avoid a second copy of
    # those numbers here: real frontier and material tests own their equations.
    for origin in ("fossil", "biogenic", "atmospheric_dac"):
        sid = "others_road_marking_conversion"
        if origin != "fossil":
            sid += f"__{origin}"
        road = services[sid]["construction"]
        assert road["kind"] == "road_coating"
        assert road["principal_carbon_origin"] == origin
        assert road["mixing"]["service_id"] == "others_road_marking_mixing_operation"
        assert road["supplies"]["others_recycled_glass_bead_manufacturing"] == {
            "service_id": "others_recycled_glass_bead_manufacturing"
        }
    sources = provider_catalogue._service_source_closures(catalogue)[
        "others_road_marking_conversion"
    ]
    assert {
        "dbc_2020_mma_mortar_application",
        "jrc_2013_flat_glass_composition",
        "pq_n_silicate_weight_modulus",
        "silikal_2008_powder_hardener_sds",
        "doka_2023_waste_output_guidance",
    } <= set(sources)

    bead_components = {
        component.get("component_id", component.get("service_id")): component["amount"]
        for component in services["others_recycled_glass_bead_manufacturing"]["construction"][
            "components"
        ]
    }
    assert bead_components == {
        "sorted_recycled_glass_cullet": 1.009515432592356,
        "others_glass_bead_furnace_operation": 0.5812210139133216,
        "uncoated_bead_production_electricity": 0.044071298213832875,
        "others_road_marking_mixing_operation": 1.0,
        "glass_dust_landfill": -0.009995202302895,
        "methyl_silicone_oil_pdms": 0.000119942427635,
        "sodium_silicate_coating_active": 0.000071965456581,
        "coating_dilution_water": 0.000287861826323,
    }
    wet_coating = 0.000479769710539
    assert bead_components["methyl_silicone_oil_pdms"] == pytest.approx(wet_coating * 0.25)
    assert bead_components["sodium_silicate_coating_active"] == pytest.approx(
        wet_coating * 0.75 * 0.20
    )
    assert bead_components["coating_dilution_water"] == pytest.approx(wet_coating * 0.75 * 0.80)
    assert (
        bead_components["methyl_silicone_oil_pdms"]
        + bead_components["sodium_silicate_coating_active"]
        + bead_components["coating_dilution_water"]
    ) == pytest.approx(wet_coating)
    assert (
        bead_components["sorted_recycled_glass_cullet"]
        + wet_coating
        + bead_components["glass_dust_landfill"]
    ) == pytest.approx(1.0)

    furnace = services["others_glass_bead_furnace_operation"]["construction"]
    assert furnace["kind"] == "input_normalized_operation"
    assert furnace["base"]["selector"]["name"] == (
        "heat production, natural gas, at industrial furnace >100kW"
    )
    assert furnace["denominator_selector"]["unit"] == "cubic meter"
    assert furnace["omit_technosphere"] == [
        {
            "name": "market group for electricity, low voltage",
            "reference_product": "electricity, low voltage",
            "unit": "kilowatt hour",
        }
    ]

    assert not {
        "others_general_coating_resin_blend_biogenic",
        "others_general_coating_resin_blend_atmospheric_dac",
        "others_general_coating_resin_blend_fossil",
    } & set(services)

    upr_selector = {
        "name": "orthophthalic acid based unsaturated polyester resin production",
        "reference_product": "orthophthalic acid based unsaturated polyester resin",
        "unit": "kilogram",
        "exact_location": "RER",
    }
    for origin in ("biogenic", "atmospheric_dac"):
        upr = services[f"unsaturated_polyester_resin_{origin}"]
        assert upr["reference_product"] == ("orthophthalic acid based unsaturated polyester resin")
        assert upr["citation"] == ("ecoinvent_3_12_complete_orthophthalic_upr_selected_origin")
        assert upr["construction"]["base"]["selector"] == upr_selector
        assert [item["replacement_id"] for item in upr["construction"]["replace_technosphere"]] == [
            "maleic_anhydride",
            "phthalic_anhydride",
            "propylene_glycol",
            "styrene",
        ]
        # Delivered-feed owner tests solve all four selected supplies and their
        # preserved native freight; this test owns only the UPR route boundary.
        assert "replace_biosphere" not in upr["construction"]

    assert services["unsaturated_polyester_fossil"]["construction"] == {
        "kind": "native_activity",
        "source_role": "native_ecoinvent_provider",
        "amount": 1.0,
        "selector": upr_selector,
    }


def test_selected_origin_operations_preserve_direct_carbon_origin() -> None:
    """Move only feedstock-derived direct carbon to native non-fossil flows."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    expected = {
        "acetylene": [
            ("Carbon dioxide, fossil", ("air",)),
        ],
        "formaldehyde": [
            ("Carbon dioxide, fossil", ("air", "urban air close to ground")),
            ("Carbon monoxide, fossil", ("air", "urban air close to ground")),
        ],
        "maleic_anhydride": [
            ("Carbon dioxide, fossil", ("air", "urban air close to ground")),
            ("Carbon monoxide, fossil", ("air", "urban air close to ground")),
        ],
        "succinic_acid": [
            ("Carbon dioxide, fossil", ("air",)),
        ],
        "propylene_oxide": [("Carbon dioxide, fossil", ("air", "urban air close to ground"))],
        "phenol": [("Carbon dioxide, fossil", ("air",))],
        "acetone": [("Carbon dioxide, fossil", ("air",))],
        "bisphenol_a": [("Carbon dioxide, fossil", ("air", "urban air close to ground"))],
        "methyl_methacrylate": [("Carbon dioxide, fossil", ("air",))],
        "pmma": [("Carbon dioxide, fossil", ("air",))],
        "phosgene": [
            ("Carbon dioxide, fossil", ("air",)),
            ("Carbon monoxide, fossil", ("air",)),
        ],
        "polycarbonate": [("Carbon dioxide, fossil", ("air",))],
        "propylene_glycol": [("Carbon dioxide, fossil", ("air", "urban air close to ground"))],
        "phthalic_anhydride": [
            ("Carbon dioxide, fossil", ("air",)),
            ("Carbon monoxide, fossil", ("air", "urban air close to ground")),
        ],
    }
    for base_service_id, expected_sources in expected.items():
        for origin in ("biogenic", "atmospheric_dac"):
            construction = services[f"{base_service_id}_{origin}"]["construction"]
            actual_sources = [
                (
                    rule["selector"]["name"],
                    tuple(rule["selector"]["categories"]),
                )
                for rule in construction["replace_biosphere"]
            ]
            assert actual_sources == expected_sources
            for rule in construction["replace_biosphere"]:
                assert rule["replacement_selector"] == {
                    "name": rule["selector"]["name"].replace("fossil", "non-fossil"),
                    "unit": "kilogram",
                    "categories": rule["selector"]["categories"],
                }

    for origin in ("biogenic", "atmospheric_dac"):
        assert "replace_biosphere" not in services[f"cumene_{origin}"]["construction"]


@pytest.mark.parametrize(
    ("constituent", "material"),
    (
        ("general_coatings", "epoxy"),
        ("marine_coatings", "epoxy"),
        ("road_marking_coatings", "road"),
    ),
)
def test_coating_terminal_routes_keep_principal_basis_and_all_dump_classes(
    constituent: str, material: str
) -> None:
    """Route actual coatings and three origins through every complete fate."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    expected = {
        "incineration_treatment": f"{material}_incineration",
        "landfill_treatment": f"{material}_landfill",
        "mismanaged_open_burning_treatment": f"{material}_open_burning",
        "mismanaged_open_dump_treatment": f"{material}_open_dump",
    }
    for origin in ("fossil", "biogenic", "atmospheric_dac"):
        routes = [
            route
            for route in cast(list[dict[str, Any]], catalogue["routes"])
            if route["constituent"] == constituent
            and route["ultimate_carbon_origin"] == origin
            and route["provider_demand_type"] in expected
        ]
        assert len(routes) == 8  # Three single fates plus the five native dump climates.
        assert len({route["route_component_id"] for route in routes}) == 8
        for route in routes:
            service = services[route["provider_service_id"]]
            assert service["construction"]["kind"] == expected[route["provider_demand_type"]]
            assert service["construction"]["principal_carbon_origin"] == origin
            assert route["throughput_multiplier"] == 1.0
            assert route["method_authority_id"] == service["citation"]
            if route["provider_demand_type"] == "mismanaged_open_dump_treatment":
                assert route["share"] is None
                assert route["material_component_share"] == 1.0
                assert route["infiltration_source_id"] == "ecoinvent_3_12_row_open_dump_portfolio"
            else:
                assert route["share"] == 1.0


def test_tyre_sbr_provider_mix_is_source_consistent_across_carbon_origins() -> None:
    """Bind the REMIND-MFA functional basis and JRC tire-weighted SBR analog."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    expected = {
        "fossil": [
            {"service_id": "emulsion_sbr_fossil", "amount": 0.682451253481894},
            {"service_id": "solution_sbr_fossil", "amount": 0.317548746518106},
        ],
        "biogenic": [
            {"service_id": "emulsion_sbr_biogenic", "amount": 0.682451253481894},
            {"service_id": "solution_sbr_biogenic", "amount": 0.317548746518106},
        ],
        "atmospheric_dac": [
            {
                "service_id": "emulsion_sbr_atmospheric_dac",
                "amount": 0.682451253481894,
            },
            {
                "service_id": "solution_sbr_atmospheric_dac",
                "amount": 0.317548746518106,
            },
        ],
    }
    for origin, components in expected.items():
        service = services[f"tyre_sbr_{origin}"]
        assert service["construction"]["components"] == components
        assert service["representative_method"]["provider_technology_weights"] == {
            "emulsion_sbr": 0.682451253481894,
            "solution_sbr": 0.317548746518106,
        }
        method = service["representative_method"]
        assert method["upstream_material"] == "Elastomers (tyres)"
        assert method["upstream_carbon_mass_fraction"] == pytest.approx(35 / 39)
        assert (
            "SBR-equivalent dry synthetic-rubber polymer"
            in method["upstream_functional_interpretation"]
        )
        assert "Fixed global tire-use technology mix from 2002" in method["transfer_limitation"]
        assert sum(component["amount"] for component in components) == 1.0


def test_fibres_route_separates_resin_from_fibre_conversion() -> None:
    """Bind the fiber portfolio to complete resin and conversion owners."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }

    expected_suppliers = {
        "polyethylene_terephthalate": "pet_amorphous",
        "polyamide_6_resin": "polyamide_6_resin",
        "polyamide_66_resin": "polyamide_66_resin",
    }
    for origin in ("fossil", "biogenic", "atmospheric_dac"):
        selected = {
            route["constituent"]: route["provider_service_id"]
            for route in cast(list[dict[str, Any]], catalogue["routes"])
            if route["material"] == "Fibres"
            and route["provider_demand_type"] == "virgin_resin_production"
            and route["virgin_feedstock_route"] == origin
        }
        expected = {key: f"{supplier}_{origin}" for key, supplier in expected_suppliers.items()}
        if origin == "fossil":
            expected.update(
                polyamide_6_resin="nylon_6_fossil", polyamide_66_resin="nylon_66_fossil"
            )
        assert selected == expected
    assert services["nylon_66_fossil"]["construction"]["selector"] == {
        "name": "nylon 6-6 production",
        "reference_product": "nylon 6-6",
        "unit": "kilogram",
        "exact_location": "RER",
    }

    pa6 = services["polyamide_6_resin_biogenic"]["construction"]
    pa6_components = {
        component.get("component_id", component.get("service_id")): component
        for component in pa6["components"]
    }
    assert pa6_components["caprolactam_delivered_biogenic"]["amount"] == 1.0
    assert pa6_components["electricity"]["amount"] == pytest.approx(0.9719129859188831)
    assert pa6_components["heat_transfer_duty"]["amount"] == pytest.approx(0.6377121193886148)
    assert pa6_components["steam"]["amount"] == pytest.approx(2.613401131303406)
    assert pa6_components["process_and_scrubber_water"]["amount"] == 2.0
    assert pa6_components["wastewater_treatment"]["amount"] == -0.002
    assert pa6_components["nitrogen_supply_and_vent"]["amount"] == 0.022
    assert pa6_components["heat_transfer_medium_makeup"]["amount"] == (0.0000325)
    assert pa6_components["heat_transfer_medium_makeup"]["selector"] == {
        "name": "market for chemical, organic, aromatics",
        "reference_product": "chemical, organic, aromatics",
        "unit": "kilogram",
        "exact_location": "GLO",
    }
    assert pa6["biosphere_terms"] == [
        {
            "selector": {
                "name": "NMVOC, non-methane volatile organic compounds",
                "unit": "kilogram",
                "categories": ["air"],
            },
            "amount": (8 + 32.5) / 1_000_000,
        }
    ]
    assert (
        pa6_components["hazardous_waste_treatment"]["selector"]["reference_product"]
        == "hazardous waste, for incineration"
    )

    pa66 = services["polyamide_66_resin_biogenic"]["construction"]
    assert pa66["components"][:2] == [
        {"service_id": "adipic_acid_delivered_biogenic", "amount": 0.6457317073170732},
        {
            "service_id": "hexamethylenediamine_delivered_biogenic",
            "amount": 0.5134676564156946,
        },
    ]
    pa66_components = {
        component.get("component_id", component.get("service_id")): component
        for component in pa66["components"]
    }
    assert pa66_components["process_and_scrubber_water"]["amount"] == (2.0908006362672322)
    assert pa66_components["wastewater_treatment"]["amount"] == -0.00225
    assert pa66["biosphere_terms"][0]["amount"] == 0.00002
    pa66_method = services["polyamide_66_resin_biogenic"]["representative_method"]
    assert "Deionized water supplies" in pa66_method["water_source_limitation"]

    oxime = services["cyclohexanone_oxime_biogenic"]["construction"]
    assert oxime["components"][:4] == [
        {"service_id": "cyclohexanone_biogenic", "amount": 0.5377332979851538},
        {
            "component_id": "hydroxylamine",
            "source_role": "native_ecoinvent_provider",
            "selector": {
                "name": "market for hydroxylamine",
                "reference_product": "hydroxylamine",
                "unit": "kilogram",
                "exact_location": "RER",
            },
            "amount": 0.18097030752916227,
        },
        {
            "component_id": "sulfuric_acid",
            "source_role": "native_ecoinvent_provider",
            "selector": {
                "name": "market for sulfuric acid",
                "reference_product": "sulfuric acid",
                "unit": "kilogram",
                "exact_location": "RER",
            },
            "amount": 0.26866666666666666,
        },
        {
            "component_id": "ammonia",
            "source_role": "native_ecoinvent_provider",
            "selector": {
                "name": "market for ammonia, anhydrous, liquid",
                "reference_product": "ammonia, anhydrous, liquid",
                "unit": "kilogram",
                "exact_location": "RER",
            },
            "amount": 0.0933123011664899,
        },
    ]
    assert oxime["biosphere_terms"] == [
        {
            "selector": {
                "name": "NMVOC, non-methane volatile organic compounds",
                "unit": "kilogram",
                "categories": ["air", "non-urban air or from high stacks"],
            },
            "amount": 0.62 * 0.000024,
        }
    ]
    caprolactam = services["caprolactam_biogenic"]["construction"]
    caprolactam_components = {
        component.get("component_id", component.get("service_id")): component
        for component in caprolactam["components"]
    }
    assert caprolactam_components["cyclohexanone_oxime_biogenic"]["amount"] == (0.8960059638156952)
    assert caprolactam_components["sulfuric_acid"]["amount"] == 1.0720564757622846
    assert caprolactam_components["ammonia"]["amount"] == 0.3723051379086162
    assert caprolactam_components["process_water"]["amount"] == 61.67081048091456
    assert caprolactam_components["mixed_wastewater_treatment"]["amount"] == (-0.06167081048091456)
    assert {
        term["selector"]["name"]: term["amount"] for term in caprolactam["biosphere_terms"]
    } == {
        "Benzene": 0.0000221648077529604,
        "NMVOC, non-methane volatile organic compounds": 0.00031868582446884766,
    }

    caprolactam_method = services["caprolactam_biogenic"]["representative_method"]
    assert caprolactam_method["economic_allocation_factor"] == 0.8960059638156952
    assert caprolactam_method["economic_allocation_equation"] == ("2500/(2500 + 1.612*180)")
    assert (
        "already quasi-microscopically allocated energy"
        in caprolactam_method["allocation_boundary"]
    )
    assert (
        caprolactam_method["mixed_wastewater_carbon_kg_per_kg_caprolactam_diagnostic"]
        == 0.029601989030838993
    )
    assert caprolactam_method["process_water_kg_per_kg_caprolactam"] == (61.67081048091456)

    for origin in ("biogenic", "atmospheric_dac"):
        expected_direct_carbon = {
            f"cyclohexane_{origin}": ["air", "urban air close to ground"],
            f"cyclohexanone_{origin}": ["air", "urban air close to ground"],
            f"hydrogen_cyanide_{origin}": [
                "air",
                "urban air close to ground",
            ],
            f"adiponitrile_{origin}": ["air"],
            f"hexamethylenediamine_{origin}": ["air"],
        }
        for service_id, categories in expected_direct_carbon.items():
            method = services[service_id]["representative_method"]
            assert "_carbon =" in method["carbon_balance_equation"]
            rules = services[service_id]["construction"]["replace_biosphere"]
            assert rules == [
                {
                    "selector": {
                        "name": "Carbon dioxide, fossil",
                        "unit": "kilogram",
                        "categories": categories,
                    },
                    "replacement_selector": {
                        "name": "Carbon dioxide, non-fossil",
                        "unit": "kilogram",
                        "categories": categories,
                    },
                    "reason": rules[0]["reason"],
                }
            ]

    amorphous_pet = services["pet_amorphous_biogenic"]["construction"]
    assert {replacement["service_id"] for replacement in amorphous_pet["replacements"]} == {
        "ethylene_glycol_delivered_rer_biogenic",
        "purified_terephthalic_acid_delivered_biogenic",
    }

    pp_conversion = services["fibres_pp_conversion"]["construction"]
    assert pp_conversion["omit_technosphere"][0]["selector"]["reference_product"] == (
        "polypropylene, granulate"
    )
    assert pp_conversion["omission_balance_groups"][0]["omission_ids"] == ["principal_polymer"]
    pet_conversion = services["fibres_pet_conversion"]["construction"]
    assert pet_conversion["waste_replacements"][0]["service_id"] == (
        "plastic_waste_market_row_non_fossil"
    )
    for grade in ("pa6", "pa66"):
        components = services[f"fibres_{grade}_conversion"]["construction"]["components"]
        assert components[:2] == [
            {"service_id": f"{grade}_finish_concentrate", "amount": 1.0},
            {"service_id": f"{grade}_spinning_support", "amount": 1.0},
        ]
        assert [row["component_id"] for row in components[2:]] == [
            "electricity",
            "heat_transfer_duty",
            "steam",
        ]
    fibre_conversion = [
        r
        for r in cast(list[dict[str, Any]], catalogue["routes"])
        if r["material"] == "Fibres" and r["provider_demand_type"] == "processing_conversion"
    ]
    assert [(r["constituent"], r["provider_service_id"], r["share"]) for r in fibre_conversion] == [
        ("polypropylene", "fibres_pp_conversion", 1.0),
        ("polyethylene_terephthalate", "fibres_pet_conversion", 1.0),
        ("polyethylene_terephthalate", "fibres_pet_conversion__non_fossil", 1.0),
        ("polyethylene_terephthalate", "fibres_pet_conversion__non_fossil", 1.0),
        ("polyamide_6_resin", "fibres_pa6_conversion", 1.0),
        ("polyamide_66_resin", "fibres_pa66_conversion", 1.0),
    ]
    assert [
        row["ultimate_carbon_origin"]
        for row in fibre_conversion
        if row["constituent"] == "polyethylene_terephthalate"
    ] == ["fossil", "biogenic", "atmospheric_dac"]
    assert all(services[r["provider_service_id"]]["foreground_service"] for r in fibre_conversion)
    assert "project_flow_id" not in str(catalogue)
    assert "project_direct_flows" not in catalogue

    assert all(
        component.get("component_id") != "heat_recovery"
        for component in pa6["components"] + pa66["components"]
    )


def test_rosental_benzene_and_toluene_include_the_complete_btx_boundary() -> None:
    """Keep MTA operating inputs and BTX separation on their distinct reference masses."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }

    expected = {
        "mta_electricity": 24_091_588 / 49_284 / 3_600,
        "plant": 0.000015 / 49_284,
        "zeolite": 33.645 / 49_284,
        "btx_separation_electricity": 79_506_982 / 17_902 / 3_600,
        "btx_process_water_best_available_comparator": 14_482 / 17_902,
    }
    for origin in ("biogenic", "atmospheric_dac"):
        for product in ("benzene", "toluene"):
            service = services[f"{product}_{origin}"]
            construction = service["construction"]
            components = {
                component["component_id"]: component
                for component in construction["components"]
                if "component_id" in component
            }
            assert {
                component_id: components[component_id]["amount"] for component_id in expected
            } == expected
            assert construction["components"][0] == {
                "service_id": f"methanol_{origin}",
                "amount": 2.351603174033248,
            }
            assert construction["biosphere_terms"] == [
                {
                    "selector": {
                        "name": "Carbon dioxide, non-fossil",
                        "unit": "kilogram",
                        "categories": ["air"],
                    },
                    "amount": 1500 / 49_284,
                }
            ]
            assert components["btx_process_water_best_available_comparator"]["selector"] == {
                "name": "market for water, deionised",
                "reference_product": "water, deionised",
                "unit": "kilogram",
                "exact_location": "Europe without Switzerland",
            }
            assert all("wastewater" not in component_id for component_id in components)
            assert all(
                "wastewater"
                not in " ".join(
                    str(selector.get(field, "")) for field in ("name", "reference_product")
                ).casefold()
                for component in construction["components"]
                for selector in (component.get("selector", {}),)
            )
            assert service["representative_method"]["process_water"] == (
                "Native deionized water supplies the RER BTX operation."
            )
            assert service["representative_method"]["wastewater"] == (
                "No additional wastewater discharge is included."
            )
