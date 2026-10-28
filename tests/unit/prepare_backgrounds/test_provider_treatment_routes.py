"""Focused provider catalog authority contracts."""

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest
from infiltration_fixtures import resolved_infiltration_provider
from provider_recipe_fixtures import (
    _assert_native_inventory,
    _prepare_selected_native_frontier,
    _process,
)
from ptt_steam_fixtures import ptt_steam_native_inventory

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
    _sbr_treatment_authority,
)


def test_sbr_treatment_routes_preserve_each_accepted_polymer_composition(tmp_path: Path) -> None:
    """Keep native dry-tire proxies distinct from actual filled non-tire SBR."""

    provider = resolved_infiltration_provider(tmp_path)
    catalogue = provider.catalogue
    resolved_shares = provider.routes.set_index("route_component_id")["share"]
    authority = _sbr_treatment_authority()
    tyre_basis = authority["functional_bases"]["Elastomers (tyres)"]
    others_basis = authority["functional_bases"]["Others"]
    assert Fraction(1, 4) * Fraction(12, 13) + Fraction(3, 4) * Fraction(8, 9) == Fraction(35, 39)
    assert tyre_basis["styrene_derived_mass_share"] == 0.25
    assert tyre_basis["butadiene_derived_mass_share"] == 0.75
    assert others_basis["styrene_derived_mass_share"] == 0.235
    assert others_basis["butadiene_derived_mass_share"] == 0.765
    assert authority["distinct_composition_decision"]["decision"].startswith("retain 0.25/0.75")
    routes = [
        route
        for route in catalogue["routes"]
        if route["material"] in {"Elastomers (tyres)", "Others"}
        and route.get("sbr_proxy_component_role")
    ]
    rendered = json.dumps(routes, sort_keys=True).lower()
    assert "pneumatic tyre" not in rendered
    assert "natural rubber" not in rendered

    sbr_routes = routes
    assert sbr_routes
    for route in sbr_routes:
        assert route["material"] == "Elastomers (tyres)"
        assert route["material_functional_unit"] == (
            "1 kg dry SBR-equivalent synthetic-rubber polymer"
        )
        assert route["styrene_component_share"] == 0.25
        assert route["butadiene_analogue_component_share"] == 0.75
        assert route["sbr_share_in_material"] == 1.0
        assert route["whole_tyre_constituents_excluded"] is True
        assert route["no_treatment_sensitivity_executed"] is True
        if route["sbr_proxy_component_role"] in {
            "styrene_derived_fraction",
            "butadiene_derived_fraction",
        }:
            assert "polymer treatment " in route["provider_class"]
            assert route["analogue_status"] == ("sbr_styrene_pe_hydrocarbon_treatment_proxy")
        assert route["method_authority_id"] == "sbr_polymer_treatment_analogue_v1"

    fossil_routes = [
        route
        for route in sbr_routes
        if route["ultimate_carbon_origin"] == "fossil"
        and route["sbr_proxy_component_role"]
        in {"styrene_derived_fraction", "butadiene_derived_fraction"}
    ]
    # Each dry-tire native proxy portfolio sums once. Discarded non-tire
    # compound22 uses its actual-material treatment owner.
    for material, demand_type in {
        (route["material"], route["provider_demand_type"]) for route in fossil_routes
    }:
        components = [
            route
            for route in fossil_routes
            if route["provider_demand_type"] == demand_type and route["material"] == material
        ]
        assert sum(
            resolved_shares[route["route_component_id"]] for route in components
        ) == pytest.approx(1.0)


def test_phb_residue_retains_its_source_basis_and_disclosed_fate() -> None:
    """Keep the reported residue and product basis bound to their source authorities."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    method_sources = set(
        catalogue["method_authority_registry"]["exact_source_bindings"][
            "harding_2008_phb1_source_operation"
        ]
    )

    assert {
        "harding_2008_phb1",
        "righi_2017_phb_residual_biomass",
        "righi_2018_phb_residual_biomass",
        "eu_jrc_ilcd_handbook_2010",
        "ecoinvent_3_12_cutoff",
        "premise_2_4_6",
    } == method_sources

    for service_id in ("phb_biogenic", "phb_atmospheric_dac", "phb_fossil"):
        service = services[service_id]
        construction = service["construction"]
        residue = [
            row
            for row in construction["source_balance_outputs"]
            if row["source_term"] == "solid waste (residual biomass)"
        ]

        assert service["citation"] == "harding_2008_phb1_source_operation"
        assert [
            {key: row[key] for key in ("source_term", "amount", "unit")} for row in residue
        ] == [
            {
                "source_term": "solid waste (residual biomass)",
                "amount": 0.42,
                "unit": "kilogram",
            }
        ]
        assert service["representative_method"]["functional_unit"] == (
            "1 kg modelled dry P(3HB) homopolymer powder immediately after spray drying"
        )
        native_residue = next(
            component
            for component in construction["components"]
            if component.get("component_id")
            == "residual_bacterial_biomass_waste_to_energy_analogue"
        )
        assert native_residue["selector"] == {
            "name": "treatment of waste wood, untreated, municipal incineration",
            "reference_product": "waste wood, untreated",
            "unit": "kilogram",
            "exact_location": "GLO",
        }


def test_phb_retains_the_reported_cod_under_its_declared_water_mapping() -> None:
    """Preserve reported aggregate COD under the explicit study discharge convention."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}

    for service_id in ("phb_biogenic", "phb_atmospheric_dac", "phb_fossil"):
        service = services[service_id]
        construction = service["construction"]
        assert service["reference_product"] == "modelled dry P(3HB) homopolymer powder"
        assert service["name"].startswith("modelled dry P(3HB) homopolymer powder")
        assert (
            "aqueous discharge uses the reported COD"
            in service["representative_method"]["recovery_chemical_source_cutoff"]
        )
        assert all(
            component.get("component_id") != "average_wastewater_treatment"
            for component in construction["components"]
        )
        direct_water = {
            (term["selector"]["name"], tuple(term["selector"]["categories"])): term["amount"]
            for term in construction["biosphere_terms"]
            if term["selector"]["categories"] == ["water"]
        }
        assert direct_water == {
            ("COD, Chemical Oxygen Demand", ("water",)): 0.9,
            ("Water", ("water",)): 0.0331,
        }
        recovery_outputs = [
            row
            for row in construction["source_balance_outputs"]
            if "recovery" in row["source_term"]
            or row["source_term"].startswith(("hydrogen peroxide", "potassium hydroxide", "enzyme"))
        ]
        assert recovery_outputs
        assert all(
            "aggregate process outputs" in row["disposition"]
            and "aggregate COD" in row["disposition"]
            for row in recovery_outputs
        )

    atmospheric = services["phb_atmospheric_dac"]
    assert "retained biogenic primer" in atmospheric["name"]
    assert "biogenic-primer" in atmospheric["representative_method"]["feedstock_carbon_origin"]


def test_pla_uses_the_principal_source_operations_for_its_acid_parents() -> None:
    """Keep active acid and its solution carrier distinct for each PLA route."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}

    lactic_acid = services["lactic_acid_atmospheric_dac"]
    assert lactic_acid["citation"] == ("krestnikova_2026_atmospheric_glucose_lactic_acid_pla")
    assert lactic_acid["reference_product"] == ("active L-lactic acid in 86.5% aqueous solution")
    method = lactic_acid["representative_method"]
    assert method["principal_solution_mass_fraction"] == 0.865
    assert method["optical_purity"] == 0.995
    assert method["cooling_duty_mj"] == 22.5340328

    lactic_components = {
        component.get("component_id", component.get("service_id")): component
        for component in lactic_acid["construction"]["components"]
    }
    assert lactic_components["glucose_delivered_atmospheric_dac"]["amount"] == 1.13694549
    assert lactic_components["natural_gas_heat"]["amount"] == 10.5210395
    assert lactic_components["ambient_cooling_operation"]["amount"] == 22.5340328
    assert lactic_components["process_electricity"]["amount"] == 0.00025685
    assert lactic_components["enzymes"]["amount"] == 0.00371197
    assert lactic_components["hydrated_lime"]["amount"] == 0.43076934
    assert lactic_components["sulfuric_acid"]["amount"] == 0.67183143
    assert lactic_components["wood_fermentation_ethanol"]["amount"] == 0.13888103
    assert lactic_components["deionised_water"]["amount"] == 0.23449011
    assert lactic_components["wood_fermentation_ethanol"]["selector"] == {
        "name": "ethanol production from wood",
        "reference_product": ("ethanol, without water, in 95% solution state, from fermentation"),
        "unit": "kilogram",
        "exact_location": "SE",
    }
    # The MJ utility includes complete cooling-water provision and returns at the source's rounded
    # quantities.
    assert "biosphere_terms" not in lactic_acid["construction"]

    pla = services["pla_atmospheric_dac"]
    pla_components = {
        component.get("component_id", component.get("service_id")): component
        for component in pla["construction"]["components"]
    }
    assert pla["name"] == "polylactic acid from atmospheric-carbon glucose"
    assert pla_components["lactic_acid_atmospheric_dac"]["amount"] == 1.28478429
    assert pla_components["natural_gas_heat"]["amount"] == 3.23675995
    assert pla_components["average_wastewater_treatment"]["amount"] == -0.00047505
    assert pla_components["hazardous_waste_treatment"]["amount"] == -0.0058415
    assert "plastic_waste_treatment" not in pla_components
    assert "chemical_factory" not in pla_components
    assert "biosphere_terms" not in pla["construction"]

    fossil = services["pla_fossil"]
    assert fossil["citation"] == "krestnikova_2026_pla_polymerization_proxy"
    assert catalogue["method_authority_registry"]["exact_source_bindings"][fossil["citation"]] == [
        "krestnikova_guillen_gosalbez_2026_pla",
        "ecoinvent_3_12_cutoff",
        "premise_2_4_6",
    ]
    assert fossil["representative_method"]["carried_solution_water_kg"] == pytest.approx(
        1.28478429 * (1 / 0.865 - 1), rel=3e-15
    )
    assert services["lactic_acid_fossil"]["construction"]["selector"] == {
        "name": "lactic acid production",
        "reference_product": "lactic acid",
        "unit": "kilogram",
        "exact_location": "RER",
    }


def test_bioplastics_constituents_select_their_own_manufacturing_services() -> None:
    """Keep material classes distinct and activate their assigned proxies."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    suppliers = {
        "bio_pa": "polyamide_6_resin",
        "bio_pet": "pet",
        "other_bio_based": "ptt",
        "other_biodegradable": "tps",
        "pbat": "pbat",
        "pbs": "pbs",
        "pha": "phb",
        "pla": "pla",
        "starch_blends": "tps",
    }
    for origin in ("fossil", "biogenic", "atmospheric_dac"):
        routes = [
            route
            for route in catalogue["routes"]
            if route["material"] == "BioPlastics"
            and route["provider_demand_type"] == "virgin_resin_production"
            and route["virgin_feedstock_route"] == origin
        ]
        expected = {key: f"{supplier}_{origin}" for key, supplier in suppliers.items()}
        if origin == "fossil":
            expected["bio_pa"] = "nylon_6_fossil"
        assert {route["constituent"]: route["provider_service_id"] for route in routes} == expected
        assert all(route["share"] == 1.0 for route in routes)

    biogenic_acetylene = services["acetylene_biogenic"]["construction"]
    assert {replacement["service_id"] for replacement in biogenic_acetylene["replacements"]} == {
        "biomethane_high_pressure_biogenic"
    }
    assert "multiplier" not in str(biogenic_acetylene)

    dac_standard_volume = services["methane_atmospheric_dac_standard_volume"]
    assert dac_standard_volume["representative_method"] == {
        "reference_temperature_k": 288.15,
        "reference_pressure_mpa": 0.101325,
        "methane_density_kg_per_cubic_metre": 0.67983,
        "standard_volume_authority": "ecoinvent_3_12_cutoff",
    }
    assert dac_standard_volume["construction"]["components"] == [
        {"service_id": "methane_atmospheric_dac", "amount": 0.67983}
    ]
    assert catalogue["method_authority_registry"]["exact_source_bindings"][
        "ecoinvent_3_12_operation_only_acetylene"
    ] == ["ecoinvent_3_12_cutoff"]
    assert catalogue["method_authority_registry"]["exact_source_bindings"][
        "nist_srd69_methane_density_15c"
    ] == ["nist_srd69_methane_density_15c", "premise_2_4_6"]

    assert services["polyamide_6_resin_biogenic"]["construction"]["components"][0] == {
        "service_id": "caprolactam_delivered_biogenic",
        "amount": 1.0,
    }
    assert services["pbat_atmospheric_dac"]["construction"]["components"][:3] == [
        {
            "service_id": "butanediol_delivered_atmospheric_dac",
            "amount": 0.4392278262574763,
        },
        {
            "service_id": "purified_terephthalic_acid_delivered_atmospheric_dac",
            "amount": 0.36633657423006194,
        },
        {
            "service_id": "adipic_acid_delivered_atmospheric_dac",
            "amount": 0.36633657423006194,
        },
    ]
    assert services["pbs_reactor_product_biogenic"]["construction"]["components"][:2] == [
        {
            "service_id": "succinic_acid_delivered_biogenic",
            "amount": 0.6849009345381919,
        },
        {
            "service_id": "butanediol_delivered_biogenic",
            "amount": 0.5227003759301838,
        },
    ]


def test_ptt_retains_complete_operating_proxy_and_monomer_basis() -> None:
    """Keep complete operating burdens distinct from the adopted PTT material basis."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    operation = services["ptt_source_input_operation"]
    construction = operation["construction"]

    assert operation["citation"] == "van_roijen_2024_ptt_source_input_operation"
    source_binding = catalogue["method_authority_registry"]["exact_source_bindings"][
        operation["citation"]
    ]
    assert set(source_binding) == {
        "van_roijen_miller_2024",
        "shell_zimmer_2001_ptt_tbot_loading",
        "dupont_2007_ptt_continuous_recycle",
        "shell_2003_ptt_pdo_recycle_purification",
        "asahi_2007_ptt_physical_foam_control",
        "ecoinvent_3_12_cutoff",
        "premise_2_4_6",
    }
    method = operation["representative_method"]
    assert method["source_net_external_pdo_kg"] == 0.37
    assert method["source_pta_kg"] == 0.8
    assert method["monomer_mass_remainder_kg"] == 0.17
    assert method["product_to_monomer_input_mass_ratio"] == 0.8547008547008547
    assert method["titanium_butoxide_kg"] == 0.0004688888888888889
    assert method["catalyst_equation"] == "0.800 * (0.0198 + 0.0857) / 180"
    assert construction["kind"] == "operation_only"
    assert construction["base"]["selector"]["exact_location"] == "CA-QC"
    assert {rule["replacement_id"] for rule in construction["replace_technosphere"]} == {
        "electricity",
        "direct_heat",
        "steam",
    }
    assert {
        rule["selector"]["reference_product"] for rule in construction["omit_technosphere"]
    } == {
        "ethylene glycol",
        "purified terephthalic acid",
    }
    assert not construction.get("omit_biosphere")

    expected_monomers = {
        "biogenic": ("propanediol_biogenic", "purified_terephthalic_acid_delivered_biogenic"),
        "atmospheric_dac": (
            "propanediol_atmospheric_dac",
            "purified_terephthalic_acid_delivered_atmospheric_dac",
        ),
        "fossil": ("propanediol_fossil", "purified_terephthalic_acid_delivered_fossil"),
    }
    for origin, (propanediol, terephthalic_acid) in expected_monomers.items():
        root = services[f"ptt_{origin}"]
        assert root["citation"] == "van_roijen_2024_ptt_source_input_operation"
        assert root["stage_owner"] == "virgin_resin_production"
        assert root["construction"] == {
            "kind": "project_recipe",
            "source_balance_outputs": [
                {
                    "source_term": "approximate monomer mass remainder",
                    "amount": 0.17,
                    "unit": "kilogram",
                    "disposition": (
                        "Input-output mass difference between the monomer inputs and 1 kg PTT. "
                        "Native operational water is retained as a separate inventory quantity."
                    ),
                }
            ],
            "components": [
                {"service_id": "ptt_source_input_operation", "amount": 1.0},
                {"service_id": propanediol, "amount": 0.37},
                {"service_id": terephthalic_acid, "amount": 0.8},
                {"service_id": "titanium_butoxide_delivered", "amount": 0.0004688888888888889},
                {"service_id": "recycling_pet_pellet_delivery", "amount": 1.0},
            ],
        }
        assert "biosphere_terms" not in root["construction"]
        component_ids = {
            component.get("component_id", component.get("service_id", ""))
            for component in root["construction"]["components"]
        }
        assert all(
            all(forbidden not in component_id for component_id in component_ids)
            for forbidden in ("purge", "treatment", "toner", "stabilizer", "antifoam")
        )

    # Virgin delivery belongs to supply before conversion. Recovered PTT uses its separate pellet
    # delivery.
    delivery = services["recycling_pet_pellet_delivery"]
    assert not delivery["foreground_service"]
    assert delivery["frontier_service"]
    recovery = services["mechanical_recycling_biopolyester"]["construction"]
    assert recovery["pellet_delivery"]["service_id"] == "recycling_pet_pellet_delivery"


def test_pbat_closes_the_principal_functional_group_operation() -> None:
    """Bind PBAT to one functional-group closure and native purge treatments."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    origin_services = {
        "biogenic": (
            "butanediol_delivered_biogenic",
            "purified_terephthalic_acid_delivered_biogenic",
            "adipic_acid_delivered_biogenic",
            "glycerine_delivered_biogenic",
        ),
        "atmospheric_dac": (
            "butanediol_delivered_atmospheric_dac",
            "purified_terephthalic_acid_delivered_atmospheric_dac",
            "adipic_acid_delivered_atmospheric_dac",
            "glycerine_delivered_atmospheric_dac",
        ),
        "fossil": (
            "butanediol_delivered_fossil",
            "purified_terephthalic_acid_delivered_fossil",
            "adipic_acid_delivered_fossil",
            "glycerine_delivered_fossil",
        ),
    }
    expected_source_outputs = {
        "polycondensation reaction water": 0.17212497471760035,
        "tetrahydrofuran purge": 0.00944,
    }

    for origin, source_services in origin_services.items():
        service = services[f"pbat_{origin}"]
        method = service["representative_method"]
        construction = service["construction"]
        components = construction["components"]
        component_amounts = {
            component.get("service_id", component.get("component_id")): component["amount"]
            for component in components
        }
        outputs = {
            output["source_term"]: output["amount"]
            for output in construction["source_balance_outputs"]
        }

        assert service["citation"] == "wang_2024_pbat_stoichiometric_closure"
        assert method["fixed_source_inputs_kg"] == {
            "glycerol": 0.00096,
            "titanium_butoxide": 0.000672,
            "phosphoric_acid": 0.000192,
            "hdi": 0.00784,
        }
        assert method["molar_masses_g_per_mol"]["hdi"] == 168.196
        assert method["hydroxyl_equivalent_equation"] == (
            "2 * n_bdo + 3 * n_glycerol = 2 * n_adipic_acid + 2 * n_pta + 2 * n_hdi + 2 * n_thf"
        )
        assert "Both HDI isocyanate groups react" in method["functional_group_assumption"]
        assert method["mass_balance_kg"] == {
            "inputs": 1.1815649747176002,
            "outputs": 1.1815649747176002,
        }
        assert "unresolved_source_balance_kg_per_kg" not in method
        assert outputs == expected_source_outputs
        assert [component_amounts[key] for key in source_services] == [
            0.4392278262574763,
            0.36633657423006194,
            0.36633657423006194,
            0.00096,
        ]
        assert component_amounts["titanium_butoxide_delivered"] == 0.000672
        assert component_amounts["phosphoric_acid"] == 0.000192
        assert component_amounts["hexamethylene_diisocyanate_chain_extender"] == 0.00784
        assert component_amounts["steam"] == 0.15834934
        assert component_amounts["electricity"] == 0.04600414
        assert component_amounts["cooling_water"] == 7.09830384
        assert component_amounts["hydraulic_water_collection_europe"] == (-0.00017212497471760035)
        assert component_amounts["tetrahydrofuran_purge_treatment"] == -0.00944
        assert next(
            row for row in components if row.get("component_id") == "reaction_water_treatment"
        ) == {
            "component_id": "reaction_water_treatment",
            "service_id": "hydraulic_water_collection_europe",
            "amount": -0.00017212497471760035,
        }
        assert next(
            row
            for row in components
            if row.get("component_id") == "tetrahydrofuran_purge_treatment"
        )["selector"] == {
            "name": "market for spent solvent mixture",
            "reference_product": "spent solvent mixture",
            "unit": "kilogram",
            "exact_location": "Europe without Switzerland",
        }
        fresh_material_inputs = [
            *source_services,
            "titanium_butoxide_delivered",
            "phosphoric_acid",
            "hexamethylene_diisocyanate_chain_extender",
        ]
        assert (
            math.fsum(component_amounts[key] for key in fresh_material_inputs)
            - 1.0
            - outputs["polycondensation reaction water"]
            - outputs["tetrahydrofuran purge"]
        ) == pytest.approx(0.0, abs=5e-16)
        assert construction["product_associated_constituents"] == [
            {
                "source_term": "TBOT and phosphoric-acid process residue",
                "amount": 0.000864,
                "unit": "kilogram",
                "disposition": (
                    "Assumed product-associated catalyst and deactivator residue included "
                    "within the 1 kg reference product."
                ),
            }
        ]
        assert method["product_associated_catalyst_deactivator_residue_kg"] == (
            component_amounts["titanium_butoxide_delivered"] + component_amounts["phosphoric_acid"]
        )
        molar = method["molar_masses_g_per_mol"]
        n_bdo = component_amounts[source_services[0]] * 1000 / molar["bdo"]
        n_pta = component_amounts[source_services[1]] * 1000 / molar["pta"]
        n_adipic = component_amounts[source_services[2]] * 1000 / molar["adipic_acid"]
        n_glycerol = component_amounts[source_services[3]] * 1000 / molar["glycerol"]
        n_hdi = component_amounts["hexamethylene_diisocyanate_chain_extender"] * (
            1000 / molar["hdi"]
        )
        n_thf = outputs["tetrahydrofuran purge"] * 1000 / molar["tetrahydrofuran"]
        assert 2 * n_bdo + 3 * n_glycerol == pytest.approx(
            2 * n_adipic + 2 * n_pta + 2 * n_hdi + 2 * n_thf
        )
        assert "internal BDO recycle" not in outputs


def test_ptt_compilation_retains_complete_operating_proxy(tmp_path: Path) -> None:
    """Persist the complete donor operation with signed receiving and utility substitution."""
    catalogue = provider_catalogue.load_provider_catalogue()
    # Keep synthetic source quantities and identities fixed while varying omission declarations.
    water = {
        "cooling": (
            "Water, cooling, unspecified natural origin",
            ("natural resource", "in water"),
            0.008,
        ),
        "process": ("Water, unspecified natural origin", ("natural resource", "in water"), 0.002),
        "air": ("Water", ("air",), 0.003),
        "return": ("Water", ("water",), 0.007),
    }
    air = ("air", "urban air close to ground")
    aqueous = ("water", "surface water")
    chemical_releases = {
        "nmvoc": ("NMVOC, non-methane volatile organic compounds", air, 0.011),
        "pm_fine": ("Particulate Matter, < 2.5 um", air, 0.012),
        "pm_coarse": ("Particulate Matter, > 10 um", air, 0.013),
        "pm_middle": ("Particulate Matter, > 2.5 um and < 10um", air, 0.014),
        "bod": ("BOD5, Biological Oxygen Demand", aqueous, 0.015),
        "cod": ("COD, Chemical Oxygen Demand", aqueous, 0.016),
        "doc": ("DOC, Dissolved Organic Carbon", aqueous, 0.017),
        "hydrocarbons": ("Hydrocarbons, unspecified", aqueous, 0.018),
        "solids": ("Suspended solids, unspecified", aqueous, 0.019),
        "toc": ("TOC, Total Organic Carbon", aqueous, 0.020),
    }
    biosphere_specs = {
        **{code: (*spec, "cubic meter") for code, spec in water.items()},
        **{code: (*spec, "kilogram") for code, spec in chemical_releases.items()},
    }
    waste = {
        "ash": ("average incineration residue", -0.01),
        "hazardous": ("hazardous waste, for underground deposit", -0.02),
        "municipal": ("municipal solid waste", -0.03),
        "plastic": ("waste plastic, mixture", -0.04),
    }
    monomers = {"glycol": ("ethylene glycol", 0.4), "pta": ("purified terephthalic acid", 0.64)}
    # The steam purchase must traverse its actual project construction;
    # no native leaf may stand in for the package's fixed steam reference.
    supplier_specs = (
        (
            "electricity",
            "market for electricity, medium voltage",
            "electricity, medium voltage",
            "kilowatt hour",
            2.0,
        ),
        (
            "heat",
            "market for heat, district or industrial, natural gas",
            "heat, district or industrial, natural gas",
            "megajoule",
            3.0,
        ),
        (
            "other_heat",
            "market for heat, district or industrial, other than natural gas",
            "heat, district or industrial, other than natural gas",
            "megajoule",
            5.0,
        ),
        (
            "steam",
            "market for steam, in chemical industry",
            "steam, in chemical industry",
            "kilogram",
            4.0,
        ),
        (
            "wind",
            "electricity production, wind, 1-3MW turbine, onshore",
            "electricity, high voltage",
            "kilowatt hour",
            0.0,
        ),
        (
            "factory",
            "market for chemical factory, organics",
            "chemical factory, organics",
            "unit",
            0.00001,
        ),
        ("nitrogen", "market for nitrogen, liquid", "nitrogen, liquid", "kilogram", 0.6),
    )
    utility_services = {
        "electricity": "ptt_wind_electricity",
        "heat": "ptt_biogas_heat",
        "other_heat": "ptt_biogas_heat",
        "steam": "ptt_renewable_steam",
    }
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="ptt_complete_operation",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": name,
                    "categories": categories,
                    "unit": unit,
                    "type": "natural resource"
                    if categories[0] == "natural resource"
                    else "emission",
                }
                for code, (name, categories, _amount, unit) in biosphere_specs.items()
            },
            searchable=False,
        )
        donor_exchanges = [
            {"input": ("biosphere", code), "amount": amount, "type": "biosphere", "unit": unit}
            for code, (_name, _categories, amount, unit) in biosphere_specs.items()
        ]
        nodes = {}
        for code, (product, amount) in {**monomers, **waste}.items():
            nodes[("native", code)] = _process(
                "native",
                code,
                f"market for {product}",
                product,
                production_amount=-1.0 if code in waste else 1.0,
            )
            donor_exchanges.append(
                {
                    "input": ("native", code),
                    "amount": amount,
                    "type": "technosphere",
                    "unit": "kilogram",
                }
            )
        for code, name, product, unit, amount in supplier_specs:
            nodes[("native", code)] = {
                **_process("native", code, name, product, location="RoW"),
                "unit": unit,
            }
            if amount:
                donor_exchanges.append(
                    {
                        "input": ("native", code),
                        "amount": amount,
                        "type": "technosphere",
                        "unit": unit,
                    }
                )
        for row in donor_exchanges:
            if row["type"] == "technosphere":
                product = nodes[row["input"]]["reference product"]
                row.update(
                    {
                        "activity": "invented producer " + row["input"][1],
                        "flow": "invented product " + product,
                        "name": product,
                        "reference product": product,
                        "properties": {
                            "price": {"amount": 2.0, "unit": "EUR2005"},
                            "carbon content, fossil": {"amount": 0.0},
                            "carbon content, non-fossil": {"amount": 0.0},
                        },
                        "formula": "source_amount * support_scale",
                    }
                )
            row.update(
                {
                    "uncertainty type": 2,
                    "loc": math.log(abs(row["amount"])) + 0.17,
                    "scale": 0.3,
                    "scale without pedigree": 0.2,
                    "pedigree": {"reliability": 2},
                    "comment": f"Independent synthetic {row['input'][1]}",
                }
            )
        nodes[("native", "pet")] = _process(
            "native",
            "pet",
            "polyethylene terephthalate production, granulate, amorphous",
            "polyethylene terephthalate, granulate, amorphous",
            location="CA-QC",
            exchanges=donor_exchanges,
        )
        nodes[("native", "pet")]["exchanges"][0].update(
            {
                "activity": "invented PET producer",
                "flow": "invented PET product",
                "name": "polyethylene terephthalate, granulate, amorphous",
                "reference product": "polyethylene terephthalate, granulate, amorphous",
                "unit": "kilogram",
                "CAS number": "25038-59-9",
                "chemical formula": "(C10H8O4)n",
                "properties": {"carbon content": {"amount": 0.625}},
                "formula": "RP",
            }
        )
        nodes.update(ptt_steam_native_inventory("native"))
        # The reusable converter buys 8 MJ per 2 kg. This complete invented
        # producer has a 3 MJ output, two heat inputs, nonheat electricity and
        # an independently closed water boundary. The steam owner changes
        # only heat supply; the operation's 4 kg steam purchase remains once.
        steam_water = {"process": 0.003, "air": 0.001, "return": 0.002}
        for code, name, output, inputs in (
            (
                "steam_row",
                "steam production, as energy carrier, in chemical industry",
                3.0,
                [
                    {"input": ("native", "heat"), "type": "technosphere", "amount": 2.4},
                    {"input": ("native", "other_heat"), "type": "technosphere", "amount": 1.5},
                    {"input": ("native", "electricity"), "type": "technosphere", "amount": 0.6},
                    *[
                        {"input": ("biosphere", key), "type": "biosphere", "amount": amount}
                        for key, amount in steam_water.items()
                    ],
                ],
            ),
            (
                "steam_row_market",
                "market for heat, from steam, in chemical industry",
                1.0,
                [{"input": ("native", "steam_row"), "type": "technosphere", "amount": 1.0}],
            ),
        ):
            nodes["native", code] = {
                **_process(
                    "native",
                    code,
                    name,
                    "heat, from steam, in chemical industry",
                    location="RoW",
                    production_amount=output,
                    exchanges=inputs,
                ),
                "unit": "megajoule",
            }
        bd.Database("native").write(nodes, searchable=False)
        original = {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for node in bd.Database("native")
        }
        frontier = _prepare_selected_native_frontier(bd, catalogue, ("ptt_source_input_operation",))
        generated = bd.get_node(
            database=frontier.database_name,
            code=provider_catalogue.provider_service_code("ptt_source_input_operation"),
        )
        exchanges = [dict(exchange) for exchange in generated.exchanges()]
        production = [row for row in exchanges if row["type"] == "production"]
        assert [(row["input"], row["amount"]) for row in production] == [(generated.key, 1.0)]
        assert production[0] == {
            "input": generated.key,
            "output": generated.key,
            "type": "production",
            "amount": 1.0,
            "name": "PTT polymerisation operating subsystem",
            "reference product": "PTT polymerisation operating subsystem",
            "unit": "kilogram",
            "formula": "RP",
        }
        retained = [row for row in donor_exchanges if row["input"][1] not in monomers]
        assert len(exchanges) == 1 + len(retained)
        actual = {row["comment"]: row for row in exchanges if row["type"] != "production"}
        assert set(actual) == {row["comment"] for row in retained}
        for source in retained:
            code = source["input"][1]
            # Brightway adds the donor output key while writing its fixture.
            # A copied exchange belongs to the generated operation instead.
            expected = {
                **source,
                "output": generated.key,
                "input": (
                    frontier.database_name,
                    provider_catalogue.provider_service_code(utility_services[code]),
                )
                if code in utility_services
                else source["input"],
            }
            if code in utility_services:
                expected.pop("activity")
                product = {
                    "electricity": "electricity, high voltage",
                    "heat": "heat, central or small-scale, other than natural gas",
                    "other_heat": "heat, central or small-scale, other than natural gas",
                    "steam": "steam, in chemical industry",
                }[code]
                expected.update(name=product, **{"reference product": product})
                if code != "steam":
                    expected.pop("flow")
                    expected.pop("properties")
            assert actual[source["comment"]] == expected
        assert water["cooling"][2] + water["process"][2] == water["air"][2] + water["return"][2]
        assert steam_water["process"] == steam_water["air"] + steam_water["return"]
        # Four kg steam use 16 MJ on the original 8/2 MJ/kg reference.
        # Scale complete source duties per its 3 MJ output; retain the
        # operation's independent 3+5 MJ heat purchases exactly once.
        steam_scale = 4.0 * (8.0 / 2.0) / 3.0
        _assert_native_inventory(
            bd,
            generated,
            {
                (("native", "wind"), "technosphere"): 2.0,
                (("native", "ptt_biogas"), "technosphere"): 8.0 + (2.4 + 1.5) * steam_scale,
                (("native", "electricity"), "technosphere"): 0.6 * steam_scale,
                (("native", "factory"), "technosphere"): 0.00001,
                (("native", "nitrogen"), "technosphere"): 0.6,
                **{
                    (("native", code), "technosphere"): amount
                    for code, (_product, amount) in waste.items()
                },
                **{
                    (("biosphere", code), "biosphere"): amount
                    + steam_scale * steam_water.get(code, 0.0)
                    for code, (_name, _categories, amount, _unit) in biosphere_specs.items()
                },
            },
        )
        assert {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for node in bd.Database("native")
        } == original
