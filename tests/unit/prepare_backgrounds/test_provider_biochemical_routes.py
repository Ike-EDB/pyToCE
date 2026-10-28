"""Biochemical production recipes preserve mass, allocation and carbon origins."""

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import (
    _RecipeContext,
)


def _retained_reaction_solutes() -> float:
    """Calculate reagent-equivalent solutes after the specified membrane sequence."""

    titre = 1.640 * 162.0 / 180.0 / 1000.0
    # Source charge masses per liter, including actual powder rather than the
    # equal-phosphorus manufacturing proxy. Methanol consumption follows the
    # product-stoichiometric convention; electrolyte speciation is unmodeled.
    soluble_charge = math.fsum(
        (
            0.003204186,
            0.0238301,
            0.005844277,
            0.000476055,
            0.0000016144,
            0.000212657,
            0.000427201,
            0.0000292243,
            0.0124,
        )
    )
    methanol_mw = 12.011 + 4.0 * 1.008 + 15.999
    glucan_mw = 6.0 * 12.011 + 10.0 * 1.008 + 5.0 * 15.999
    used_methanol = (titre - 0.000050) * 6.0 * methanol_mw / glucan_mw
    return (soluble_charge - used_methanol) * 0.6 * math.exp(-10.0 / 1.2) * 8.6


def test_methanol_glucan_reaction_and_recovery_follow_declared_construction() -> None:
    """Close the prospective charge, recovery, phosphorus and drying balances."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    dry_glucan_titre_g_per_l = 1.640 * 162.0 / 180.0
    reaction_volume_l = 1000.0 / dry_glucan_titre_g_per_l
    selected_share = (dry_glucan_titre_g_per_l - 0.050) / dry_glucan_titre_g_per_l
    primer_share = 1.0 - selected_share

    for origin in ("atmospheric_dac", "fossil"):
        reaction = services[f"methanol_glucan_reaction_{origin}"]
        reaction_method = reaction["representative_method"]
        components = {
            component.get("component_id", component.get("service_id")): component
            for component in reaction["construction"]["components"]
        }
        assert reaction_method["reaction_volume_l_per_kg_gross_amylose"] == pytest.approx(
            reaction_volume_l
        )
        assert reaction_method["aqueous_process_density_kg_per_l"] == 1.0
        assert reaction_method["selected_origin_share_of_dry_amylose"] == pytest.approx(
            selected_share
        )
        assert reaction_method["biogenic_primer_share_of_dry_amylose"] == pytest.approx(
            primer_share
        )
        assert components[f"methanol_{origin}"]["amount"] == pytest.approx(
            0.1 * 32.04186 * reaction_volume_l / 1000.0
        )
        assert components["dextrin_primer_best_available_maize_starch_analogue"][
            "amount"
        ] == pytest.approx(primer_share / 0.86)
        assert components["asap_enzymes"]["amount"] == pytest.approx(
            sum(reaction_method["enzyme_loads_g_per_l"].values()) * reaction_volume_l / 1000.0
        )
        phosphate = components["polyphosphate_as_sodium_tripolyphosphate"]
        # Preserve the specified powder mass. The stated 120 mmol P/L is rounded;
        # manufacture and separation must use P in that same powder.
        powder_molar_mass = 47 * 22.990 + 45 * 30.974 + 136 * 15.999
        powder_p_per_litre = 0.0124 * 45 * 30.974 / powder_molar_mass
        proxy_p_fraction = 3 * 30.974 / (5 * 22.990 + 3 * 30.974 + 10 * 15.999)
        assert phosphate["amount"] * proxy_p_fraction == pytest.approx(
            powder_p_per_litre * reaction_volume_l, rel=1e-12
        )
        assert reaction_method["polyphosphate_phosphorus_mol_per_l"] == pytest.approx(
            powder_p_per_litre * 1000 / 30.974, rel=1e-12
        )
        assert reaction_method["source_nominal_polyphosphate_phosphorus_mol_per_l"] == 0.120
        actual_powder = 0.0124 * reaction_volume_l
        assert phosphate["physical_quantity_kg"] == pytest.approx(actual_powder)
        elements = reaction_method["polyphosphate_physical_elements_kg_per_l"]
        assert math.fsum(elements.values()) == pytest.approx(0.0124, rel=1e-12)
        assert elements["phosphorus"] == pytest.approx(powder_p_per_litre, rel=1e-12)
        mass_components = [
            component["amount"]
            for key, component in components.items()
            if key
            not in {
                "reaction_mixing_electricity",
                "reaction_sensible_heat",
                "polyphosphate_as_sodium_tripolyphosphate",
            }
        ]
        assert math.fsum(mass_components) + actual_powder == pytest.approx(reaction_volume_l)
        glucan_mw = 6.0 * 12.011 + 10.0 * 1.008 + 5.0 * 15.999
        oxygen = selected_share * 3.0 * (2.0 * 15.999) / glucan_mw
        assert reaction_method["ambient_oxygen_boundary"][
            "amount_kg_per_kg_gross_amylose"
        ] == pytest.approx(oxygen)
        assert components["reaction_mixing_electricity"]["amount"] == pytest.approx(
            reaction_volume_l / 1000.0 * 0.250 * 4.0
        )
        assert components["reaction_sensible_heat"]["amount"] == pytest.approx(
            reaction_volume_l * 4.184 * 10.0 / 1000.0
        )

        recovery_service = services[f"starch_{origin}"]
        recovery = recovery_service["representative_method"]
        retained_solutes = _retained_reaction_solutes()
        residual_protein = 0.005
        dry_amylose = 0.86 - residual_protein - retained_solutes
        gross_amylose = dry_amylose / 0.85
        reaction_volume = gross_amylose * reaction_volume_l
        charged_solution_mass = reaction_volume * 1.0
        protease = reaction_volume * (4.0 / 3.2) / 1_000_000.0
        assert recovery["retained_low_molecular_solutes_kg"] == pytest.approx(retained_solutes)
        assert recovery["residual_protein_kg"] == pytest.approx(residual_protein)
        assert recovery["dry_amylose_kg"] == pytest.approx(dry_amylose)
        assert recovery["reaction_gross_amylose_kg"] == pytest.approx(gross_amylose)
        assert recovery["aqueous_process_density_kg_per_l"] == 1.0
        assert recovery["reaction_volume_l"] == pytest.approx(reaction_volume)
        assert recovery["charged_solution_mass_kg"] == pytest.approx(charged_solution_mass)
        assert recovery["protease_kg"] == pytest.approx(protease)
        assert recovery["initial_retentate_kg"] == pytest.approx(0.6 * charged_solution_mass)
        assert recovery["diafiltration_water_kg"] == pytest.approx(5.0 * charged_solution_mass)
        assert recovery["redilution_water_kg"] == pytest.approx(0.4 * charged_solution_mass)
        assert recovery["membrane_permeate_volume_m3"] == pytest.approx(
            recovery["membrane_permeate_mass_kg"]
            / recovery["aqueous_process_density_kg_per_l"]
            / 1000.0
        )
        assert recovery["ultrafiltration_electricity_kwh"] == pytest.approx(
            0.49 * recovery["membrane_permeate_volume_m3"]
        )
        permeate_mass = 6.4 * reaction_volume + oxygen * gross_amylose + protease - 8.6
        assert recovery["membrane_permeate_mass_kg"] == pytest.approx(permeate_mass)
        recovery_components = {
            item.get("component_id", item.get("service_id")): item["amount"]
            for item in recovery_service["construction"]["components"]
        }
        assert recovery_components["aqueous_residue_treatment"] == pytest.approx(
            -permeate_mass / 1000.0
        )
        heat = (8.6 - 1.0) * 4.87
        electricity = heat / 27.0 / 3.6
        assert recovery["spray_dryer_natural_gas_heat_mj"] == pytest.approx(heat)
        assert recovery_components["spray_dryer_heat"] == pytest.approx(heat)
        assert recovery["spray_dryer_electricity_kwh"] == pytest.approx(electricity)
        assert recovery_components["spray_dryer_electricity"] == pytest.approx(electricity)

        # The actual rounded reagent charges remain fixed. Elemental fractions
        # come from the selected ADP and thiamine-pyrophosphate free formulas.
        adp_molar_mass = 10 * 12.011 + 15 * 1.008 + 5 * 14.007 + 10 * 15.999 + 2 * 30.974
        tpp_molar_mass = 12 * 12.011 + 19 * 1.008 + 4 * 14.007 + 7 * 15.999 + 2 * 30.974 + 32.06
        phosphorus_per_litre = (
            powder_p_per_litre
            + 0.000427201 * 2 * 30.974 / adp_molar_mass
            + 0.000212657 * 2 * 30.974 / tpp_molar_mass
        )
        assert reaction_method["phosphorus_input_kg_per_l"] == pytest.approx(
            phosphorus_per_litre, rel=1e-12
        )
        charged_p = phosphorus_per_litre * reaction_volume
        retained_p = phosphorus_per_litre * 0.6 * math.exp(-10.0 / 1.2) * 8.6
        native_p = 0.105 * permeate_mass / 1000.0
        excess_p = charged_p - retained_p - native_p
        assert recovery["phosphorus_charged_kg"] == pytest.approx(charged_p)
        assert recovery["phosphorus_retained_in_product_kg"] == pytest.approx(retained_p)
        assert recovery["phosphorus_in_effluent_kg"] == pytest.approx(charged_p - retained_p)
        assert recovery["phosphorus_accounted_by_native_service_kg"] == pytest.approx(native_p)
        assert recovery["excess_phosphorus_to_surface_water_kg"] == pytest.approx(excess_p)
        terms = recovery_service["construction"]["biosphere_terms"]
        assert len(terms) == 2
        assert terms[0] == {
            "selector": {"name": "Water", "unit": "cubic meter", "categories": ["air"]},
            "amount": 0.0076,
        }
        assert terms[1]["selector"] == {
            "name": "Phosphorus",
            "unit": "kilogram",
            "categories": ["water", "surface water"],
        }
        assert terms[1]["amount"] == pytest.approx(excess_p)


def test_synthetic_glucose_supplies_the_dry_glucan_required_for_hydrolysis() -> None:
    """Specify ideal synthetic-glucan hydrolysis with the retained operating proxy."""

    services = {
        service["provider_service_id"]: service
        for service in cast(
            list[dict[str, Any]], provider_catalogue.load_provider_catalogue()["services"]
        )
    }
    for origin in ("atmospheric_dac", "fossil"):
        beta = 0.855 - _retained_reaction_solutes()
        construction = services[f"glucose_{origin}"]["construction"]
        assert construction["kind"] == "operation_only"
        assert construction["base"]["selector"]["name"] == "glucose production"
        assert len(construction["replacements"]) == 1
        replacement = construction["replacements"][0]
        assert replacement["service_id"] == f"starch_delivered_{origin}"
        assert replacement["multiplier"] == pytest.approx(1.0 / beta)
        assert 0.9 * replacement["multiplier"] * beta * 180.0 / 162.0 == pytest.approx(1.0)


def test_pet_uses_the_selected_bottle_grade_operation_inputs() -> None:
    """Follow the exact RoW operation instead of the unrelated CA-QC graph."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }

    for origin in ("biogenic", "atmospheric_dac"):
        pet = services[f"pet_{origin}"]["construction"]
        assert pet["base"]["selector"]["exact_location"] == "RoW"
        replacement_services = {
            replacement["replacement_id"]: replacement["service_id"]
            for replacement in pet["replacements"]
        }
        assert replacement_services == {
            "ethylene_glycol": f"ethylene_glycol_delivered_row_{origin}",
            "diethylene_glycol": f"diethylene_glycol_delivered_row_{origin}",
            "purified_terephthalic_acid": f"purified_terephthalic_acid_delivered_{origin}",
        }
        assert all("amorphous" not in str(value).casefold() for value in pet.values())

        diethylene_glycol = services[f"diethylene_glycol_{origin}"]["construction"]
        assert diethylene_glycol["base"]["selector"] == {
            "name": "ethylene glycols production, thermal hydrolysis of ethylene oxide",
            "reference_product": "diethylene glycol",
            "unit": "kilogram",
            "exact_location": "RER",
        }
        assert diethylene_glycol["replacements"] == [
            {
                "replacement_id": "ethylene_oxide",
                "service_id": f"ethylene_oxide_{origin}",
            }
        ]

        ethylene_oxide = services[f"ethylene_oxide_{origin}"]
        assert (
            "sole material carbon input"
            in ethylene_oxide["representative_method"]["carbon_origin_boundary"]
        )
        origin_rules = ethylene_oxide["construction"]["replace_biosphere"]
        assert {
            rule["selector"]["name"]: rule["replacement_selector"]["name"] for rule in origin_rules
        } == {
            "Carbon dioxide, fossil": "Carbon dioxide, non-fossil",
            "Carbon monoxide, fossil": "Carbon monoxide, non-fossil",
            "Methane, fossil": "Methane, non-fossil",
        }
        assert all(
            rule["selector"]["categories"] == ["air", "urban air close to ground"]
            and rule["replacement_selector"]["categories"] == ["air", "urban air close to ground"]
            for rule in origin_rules
        )


def test_tps_uses_one_source_basis_for_every_carbon_origin() -> None:
    """Normalize Surendren's gross output to one kilogram of plastics mass."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    source_feed = 0.77625 / 0.86
    source_separate_water = 0.1725 - source_feed * 0.14

    for origin in ("biogenic", "atmospheric_dac", "fossil"):
        service = services[f"tps_{origin}"]
        method = service["representative_method"]
        components = {
            component.get("component_id", component.get("service_id")): component
            for component in service["construction"]["components"]
        }
        if origin == "biogenic":
            source_plastics = 0.77625 + 0.2875
            product_glucan = method["product_starch_kg"]
            co_delivered = {"co-delivered water": method["co_delivered_water_kg"]}
        else:
            feed = services[f"starch_{origin}"]["representative_method"]
            product_glucan = method["product_amylose_kg"]
            source_amylose = source_feed * feed["dry_amylose_kg"]
            source_plastics = source_amylose + 0.2875
            assert method["source_residual_protein_kg"] == pytest.approx(
                source_feed * feed["residual_protein_kg"]
            )
            assert method["source_retained_low_molecular_solutes_kg"] == pytest.approx(
                source_feed * feed["retained_low_molecular_solutes_kg"]
            )
            co_delivered = {
                "retained non-plastic protein impurity": method["co_delivered_protein_kg"],
                "retained low-molecular solutes": method["co_delivered_low_molecular_solutes_kg"],
                "co-delivered water": method["co_delivered_water_kg"],
            }
        plastics_fraction = source_plastics / 1.15
        scale = 1.0 / plastics_fraction
        assert method["plastics_mass_fraction_of_gross_output"] == pytest.approx(plastics_fraction)
        assert method["normalization_factor"] == pytest.approx(scale)
        assert product_glucan + method["product_glycerine_kg"] == pytest.approx(1.0)
        assert components[f"starch_{origin}"]["amount"] == pytest.approx(source_feed * scale)
        assert components[f"glycerine_{origin}"]["amount"] == pytest.approx(0.2875 * scale)
        assert components["deionised_water"]["amount"] == pytest.approx(
            source_separate_water * scale
        )
        assert components["lorry_transport"]["amount"] == pytest.approx(0.077 * scale)
        assert components["extrusion_electricity"]["amount"] == pytest.approx(3.54 * scale)
        # The 0.15 kg plastics loss retains the product formulation; its treatment
        # service includes the associated water and other retained constituents.
        assert (
            components["process_loss_landfill"]["service_id"] == f"tps_residue_treatment_{origin}"
        )
        assert components["process_loss_landfill"]["amount"] == pytest.approx(0.15)
        assert components["process_residue_haul"]["service_id"] == "plastic_residue_haul"
        assert components["process_residue_haul"]["amount"] == pytest.approx(0.15 * scale)
        incoming_wet_mass = (0.8625 + 0.2875 + 0.08625) * scale
        assert components["manufacturing_facility"]["amount"] == pytest.approx(
            incoming_wet_mass / (10000 * 1000 * 50), rel=1e-12, abs=0
        )
        assert components["manufacturing_facility"]["selector"] == {
            "name": "market for waste preparation facility",
            "reference_product": "waste preparation facility",
            "unit": "unit",
            "exact_location": "GLO",
        }
        assert components["tps_pellet_sack_delivery"]["amount"] == pytest.approx(scale)
        assert service["construction"]["biosphere_terms"][0]["amount"] == pytest.approx(
            0.08625 * scale / 1000.0
        )
        outputs = {
            output["source_term"]: output["amount"]
            for output in service["construction"]["source_balance_outputs"]
        }
        assert outputs["thermoplastic-starch plastics-mass content"] == 1.0
        assert {key: outputs[key] for key in co_delivered} == pytest.approx(co_delivered)
        assert sum(outputs.values()) == pytest.approx(method["gross_technical_output_kg"])


def test_principal_glycerine_routes_keep_complete_origin_specific_operations() -> None:
    """Use native rape glycerine and the complete DAC epichlorohydrin chain."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }

    assert services["glycerine_biogenic"]["construction"] == {
        "kind": "native_activity",
        "source_role": "native_ecoinvent_provider",
        "amount": 1.0,
        "selector": {
            "name": "esterification of rape oil",
            "reference_product": "glycerine",
            "unit": "kilogram",
            "exact_location": "Europe without Switzerland",
        },
    }
    atmospheric = services["glycerine_atmospheric_dac"]["construction"]
    assert atmospheric["base"]["selector"]["name"] == ("glycerine production, from epichlorohydrin")
    assert atmospheric["replacements"] == [
        {
            "replacement_id": "epichlorohydrin",
            "service_id": "epichlorohydrin_delivered_atmospheric_dac",
        }
    ]
    assert services["epichlorohydrin_atmospheric_dac"]["construction"]["replacements"] == [
        {
            "replacement_id": "allyl_chloride",
            "service_id": "allyl_chloride_delivered_atmospheric_dac",
        }
    ]
    assert services["allyl_chloride_atmospheric_dac"]["construction"]["replacements"] == [
        {
            "replacement_id": "propylene",
            "service_id": "propylene_atmospheric_dac",
        }
    ]


def test_pbs_uses_gross_internal_and_net_monomer_basis() -> None:
    """Keep recovered BDO internal and demand only the external net quantity."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    expected_native_amounts = {
        "process_water": 0.49,
        "cooling_water": 0.26,
        "electricity": 0.07,
        "low_pressure_steam": 0.09,
    }
    expected_native_selectors = {
        "process_water": {
            "name": "market for water, deionised",
            "reference_product": "water, deionised",
            "unit": "kilogram",
            "exact_location": "Europe without Switzerland",
        },
        "cooling_water": {
            "name": "market for water, decarbonised",
            "reference_product": "water, decarbonised",
            "unit": "kilogram",
            "exact_location": "RoW",
        },
        "electricity": {
            "name": "market group for electricity, medium voltage",
            "reference_product": "electricity, medium voltage",
            "unit": "kilowatt hour",
            "exact_location": "RER",
        },
        "low_pressure_steam": {
            "name": "market for steam, in chemical industry",
            "reference_product": "steam, in chemical industry",
            "unit": "kilogram",
            "exact_location": "RER",
        },
    }

    for origin in ("biogenic", "atmospheric_dac", "fossil"):
        # Finishing scales the complete PBS reactor inventory to finished resin;
        # gross BDO charge, internal recovery and net external demand stay distinct.
        reactor = services[f"pbs_reactor_product_{origin}"]
        construction = reactor["construction"]
        native_amounts = {
            component["component_id"]: component["amount"]
            for component in construction["components"]
            if component.get("source_role") == "native_ecoinvent_provider"
        }
        assert native_amounts == expected_native_amounts
        native_selectors = {
            component["component_id"]: component["selector"]
            for component in construction["components"]
            if component.get("source_role") == "native_ecoinvent_provider"
        }
        assert native_selectors == expected_native_selectors
        method = reactor["representative_method"]
        assert method["source_gross_bdo_kg"] == 0.68
        assert method["backbone_normalization_factor"] == pytest.approx(
            1.0 / (1.0 + 0.002 * 0.6858404)
        )
        assert method["scaled_gross_bdo_kg"] == 0.6790685347290281
        assert method["internal_recovered_bdo_kg"] == 0.1563681587988443
        assert method["external_net_bdo_kg"] == 0.5227003759301838
        assert (
            method["succinic_acid_kg"]
            + method["external_net_bdo_kg"]
            + method["product_associated_catalyst_derived_residue_kg"]
        ) == pytest.approx(1.0 + method["reaction_water_kg"], abs=5e-16)
        assert method["catalyst_concentration_ppm_relative_to_succinic_acid"] == 2000.0
        assert method["titanium_butoxide_kg"] == 0.001369801869076384
        assert "biosphere_terms" not in construction
        assert [
            {field: row[field] for field in ("source_term", "amount", "unit")}
            for row in construction["source_balance_outputs"]
        ] == [
            {
                "source_term": "internal recovered BDO",
                "amount": 0.1563681587988443,
                "unit": "kilogram",
            },
            {
                "source_term": "polycondensation reaction water",
                "amount": 0.2089711123374521,
                "unit": "kilogram",
            },
        ]
        assert construction["components"][:3] == [
            {
                "service_id": f"succinic_acid_delivered_{origin}",
                "amount": 0.6849009345381919,
            },
            {
                "service_id": f"butanediol_delivered_{origin}",
                "amount": 0.5227003759301838,
            },
            {"service_id": "titanium_butoxide_delivered", "amount": 0.001369801869076384},
        ]
        assert [
            {field: row[field] for field in ("source_term", "amount", "unit")}
            for row in construction["product_associated_constituents"]
        ] == [
            {
                "source_term": "titanium-alkoxide catalyst-derived residue",
                "amount": 0.001369801869076384,
                "unit": "kilogram",
            }
        ]


def test_tbot_manufacturing_proxy_preserves_the_complete_native_inventory(
    tmp_path: Path,
) -> None:
    """Link one kg of native TEOS with its original chemistry and exchanges."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, object]], catalogue["services"])
    }
    sources = provider_catalogue._service_source_closures(catalogue)
    assert sources["titanium_butoxide"] == ("ecoinvent_3_12_cutoff", "premise_2_4_6")
    for polymer in ("ptt", "pbat", "pbs"):
        for origin in ("fossil", "biogenic", "atmospheric_dac"):
            assert set(sources["titanium_butoxide"]) <= set(sources[f"{polymer}_{origin}"])

    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="tbot_proxy",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "silicon"): {
                    "name": "Silicon",
                    "unit": "kilogram",
                    "categories": ("natural resource", "in ground"),
                    "type": "natural resource",
                },
                ("biosphere", "ethanol"): {
                    "name": "Ethanol",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
                ("biosphere", "water"): {
                    "name": "Water",
                    "unit": "cubic meter",
                    "categories": ("water",),
                    "type": "emission",
                },
            },
            searchable=False,
        )
        nodes = {
            ("native", "feed"): _process(
                "native",
                "feed",
                "feed supply",
                "feed",
                exchanges=[
                    {"input": ("biosphere", "silicon"), "amount": 0.125, "type": "biosphere"},
                ],
            ),
            ("native", "waste"): _process(
                "native",
                "waste",
                "waste treatment",
                "waste",
                production_amount=-1.0,
                exchanges=[{"input": ("biosphere", "water"), "amount": 0.5, "type": "biosphere"}],
            ),
        }
        # Invented coefficients exercise the complete native boundary, including
        # negative waste demand and uncertainty; no licensed inventory is copied.
        teos_exchanges = [
            {
                "input": ("native", "feed"),
                "amount": 1.25,
                "type": "technosphere",
                "uncertainty type": 3,
                "loc": 1.25,
                "scale": 0.1,
            },
            {"input": ("native", "waste"), "amount": -0.125, "type": "technosphere"},
            {
                "input": ("biosphere", "ethanol"),
                "amount": 0.0625,
                "type": "biosphere",
                "uncertainty type": 3,
                "loc": 0.0625,
                "scale": 0.001,
            },
        ]
        for code, name, location in (
            ("teos", "tetraethyl orthosilicate production", "GLO"),
            ("regional_teos", "tetraethyl orthosilicate production", "RER"),
            ("teos_market", "market for tetraethyl orthosilicate", "GLO"),
        ):
            nodes[("native", code)] = _process(
                "native",
                code,
                name,
                "tetraethyl orthosilicate",
                location=location,
                exchanges=teos_exchanges if code == "teos" else [],
            )
        bd.Database("native").write(nodes, searchable=False)
        original = {
            node.key: (dict(node), [dict(exchange) for exchange in node.exchanges()])
            for node in bd.Database("native")
        }
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2040,
            provider_database="provider_frontier",
            source=_ActivityResolver(bd, "native"),
            ordinary=_ActivityResolver(bd, "native"),
            technology=_ActivityResolver(bd, "native"),
            biosphere=_FlowResolver(bd, "biosphere"),
            material_recipe=physical_material_recipe(),
            catalogue=catalogue,
            method_authorities={
                str(row["method_authority_id"]): row
                for row in provider_catalogue._method_authority_rows(catalogue)
            },
            complete_source_ids=sources,
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        key = ("provider_frontier", provider_catalogue.provider_service_code("titanium_butoxide"))
        compiled = _service_dataset(context, services["titanium_butoxide"], services, key[0])
        assert compiled["reference product"] == "titanium butoxide"
        assert compiled["location"] == "GLO"
        assert compiled["exchanges"] == [
            {"input": key, "amount": 1.0, "type": "production"},
            {"input": ("native", "teos"), "amount": 1.0, "type": "technosphere"},
        ]
        assert len(context.semantic_rows) == len(context.technology_rows) == 1
        service_row, native_row = context.semantic_rows[0], context.technology_rows[0]
        assert service_row["semantic_class"] == "explicit_cross_material_analogue"
        assert native_row["semantic_class"] == "explicit_cross_material_analogue"
        assert native_row["activity_code"] == "teos"
        assert native_row["reference_product"] == "tetraethyl orthosilicate"
        assert service_row["functional_equivalence_claim"] == (
            "Manufacture of 1 kg TBOT is represented by 1 kg native TEOS production."
        )
        bd.Database(key[0]).write({key: compiled}, searchable=False)
        direct = LCA({bd.get_node(database="native", code="teos").id: 1.0})
        direct.lci()
        proxy = LCA({bd.get_node(database=key[0], code=key[1]).id: 1.0})
        proxy.lci()
        assert direct.dicts.biosphere == proxy.dicts.biosphere
        np.testing.assert_array_equal(
            direct.inventory.sum(axis=1),
            proxy.inventory.sum(axis=1),
        )
        assert {
            node.key: (dict(node), [dict(exchange) for exchange in node.exchanges()])
            for node in bd.Database("native")
        } == original


def test_biobutadiene_uses_native_fermentation_ethanol_and_complete_b3_inventory() -> None:
    """Preserve the allocated B3 process inventory and distinct ethanol origins."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {
        service["provider_service_id"]: service
        for service in cast(list[dict[str, Any]], catalogue["services"])
    }
    sources = {
        source["source_id"]: source
        for source in cast(list[dict[str, Any]], catalogue["source_authorities"])
    }

    biogenic_ethanol = services["ethanol_biogenic"]
    assert biogenic_ethanol["construction"] == {
        "kind": "native_activity",
        "source_role": "native_ecoinvent_provider",
        "amount": 1.0,
        "selector": {
            "name": ("market for ethanol, without water, in 95% solution state, from fermentation"),
            "reference_product": (
                "ethanol, without water, in 95% solution state, from fermentation"
            ),
            "unit": "kilogram",
            "exact_location": "BR",
        },
    }

    operation = services["ethanol_to_butadiene_b3_operation"]
    method = operation["representative_method"]
    expected_allocation = 1208 / (1208 + 0.0477 * 599 + 0.035 * 852 + 0.0147 * 1024 + 0.0403 * 1960)
    assert method["allocation_factor_to_butadiene"] == pytest.approx(
        expected_allocation,
        abs=1e-15,
    )
    components = {
        component["component_id"]: component["amount"]
        for component in operation["construction"]["components"]
    }
    # Gross B3 LCI inputs: thesis Table A.2.2, except gas from Table 9's
    # 11.598 GJ/t and Meili's reference-gas NCV of 36 MJ/Sm3 (15 C, 1013 mbar),
    # consistent with the selected native BR fuel.
    # Allocate inputs once and retain the native signs of waste demands.
    gross_components = {
        "tantalum": 2.38e-6,
        "copper": 0.000119,
        "n_n_dimethylformamide": 0.0009,
        "natural_gas": 11.598 / 36,
        "zeolite": 0.00443,
        "petrol": 0.0357,
        "hydrogen_peroxide": 0.0927,
        "copper_oxide": 0.00006,
        "silica_sand": 0.00137,
        "chlorine": 2.33e-6,
        "electricity": 0.78983,
        "steam": 5.2209,
        "softened_water": 19.72,
        "waste_zeolite_treatment": -0.00442,
        "hazardous_waste_underground_treatment": -0.0015,
        "vegetable_oil_refinery_wastewater_treatment": -0.0010867,
        "average_wastewater_treatment": -0.0067144,
    }
    assert components == pytest.approx(
        {name: amount * expected_allocation for name, amount in gross_components.items()},
        rel=1e-14,
        abs=0,
    )
    direct = {
        term["selector"]["name"]: term["amount"]
        for term in operation["construction"]["biosphere_terms"]
    }
    air_boundary = operation["construction"]["operation_boundary_terms"]
    assert len(air_boundary) == 1
    assert air_boundary[0]["source_term"] == "ambient air"
    assert air_boundary[0]["gross_amount_kg_per_kg_butadiene"] == 12.3113
    assert air_boundary[0]["source_condition"] == "25 degrees Celsius and 1 bar"
    assert "compressed_air" not in components
    octane_carbon_dioxide = 0.0357 * (8 * 44.009) / (8 * 12.011 + 18 * 1.008)
    gross_direct = {
        "Carbon dioxide, fossil": 0.6787 + octane_carbon_dioxide,
        "Carbon dioxide, non-fossil": 0.8452 - octane_carbon_dioxide,
        "Water": 0.9901 / 1000,
        "Oxygen": 2.2293,
        "Nitrogen": 13.5795,
    }
    assert direct == pytest.approx(
        {name: amount * expected_allocation for name, amount in gross_direct.items()},
        rel=1e-14,
        abs=0,
    )
    assert direct["Carbon dioxide, fossil"] + direct["Carbon dioxide, non-fossil"] == pytest.approx(
        (0.6787 + 0.8452) * expected_allocation
    )

    hydration = services["ethanol_atmospheric_dac"]["construction"]
    assert hydration["replace_biosphere"] == [
        {
            "selector": {
                "name": "Carbon dioxide, fossil",
                "unit": "kilogram",
                "categories": ["air", "urban air close to ground"],
            },
            "replacement_selector": {
                "name": "Carbon dioxide, non-fossil",
                "unit": "kilogram",
                "categories": ["air", "urban air close to ground"],
            },
            "reason": (
                "Direct CO2 from combustion of ethylene-derived hydration coproducts follows "
                "the ethylene carbon origin; its native amount is retained."
            ),
        }
    ]

    dry_ethanol = 2.4245 * 0.93 * expected_allocation
    for origin in ("biogenic", "atmospheric_dac"):
        assert services[f"butadiene_{origin}"]["location"] == "BR"
        route = services[f"butadiene_{origin}"]["construction"]["components"]
        assert route[0]["amount"] == pytest.approx(dry_ethanol)
        assert route[2] == {
            "service_id": "ethanol_to_butadiene_b3_operation",
            "amount": 1.0,
        }
    expected_feed_water = 2.4245 * 0.07 * expected_allocation
    assert services["butadiene_biogenic"]["construction"]["components"][1][
        "amount"
    ] == pytest.approx(expected_feed_water)
    assert services["butadiene_atmospheric_dac"]["construction"]["components"][1][
        "amount"
    ] == pytest.approx(expected_feed_water)

    method_id = "cabrera_camacho_2020_b3_ethanol_to_butadiene"
    assert cast(dict[str, Any], catalogue["method_authority_registry"])["exact_source_bindings"][
        method_id
    ] == [
        "cabrera_camacho_2020_butadiene_thesis",
        "cabrera_camacho_2020_one_step_prices",
        "cabrera_camacho_2022_butadiene",
        "meili_2021_reference_natural_gas",
    ]
    for name in (
        "ethanol_to_butadiene_b3_operation",
        "butadiene_biogenic",
        "butadiene_atmospheric_dac",
    ):
        assert services[name]["citation"] == method_id
    assert sources["cabrera_camacho_2020_butadiene_thesis"]["retained_pdf_sha256"] == (
        "532b5ea37c30b5d8813f920d595d66d37b3a7212fed4904c4c9e9e02277039b3"
    )
    assert sources["cabrera_camacho_2020_one_step_prices"]["supporting_information_sha256"] == (
        "530476333d4736c070e5d52fbb7aadf66a75d3ca64c79f6a55c9c416ee1578f7"
    )
