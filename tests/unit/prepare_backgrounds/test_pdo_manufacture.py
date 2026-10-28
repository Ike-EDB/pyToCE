"""Keep the coupled PDO process, actual solution and residual supply boundaries."""

import json
from collections import defaultdict
from math import fsum, isfinite, log
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from native_chemical_fixtures import native_chemical_biosphere, native_chemical_inventory
from pa_water_receiving_fixtures import _dated_water_inventory, water_receiving_boundary
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.chemical_organic_wastewater import (
    chemical_organic_wastewater_datasets,
)
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
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import _CarbonOrigin
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.pdo_manufacture import _pdo_reference
from pytoce.shared.runtime.brightway import write_brightway_database

# Independent mass contributions per molecular reference from the admitted
# process formulas. Main process compounds use nominal masses; cyanocobalamin
# uses its separately admitted CIAAW convention. No production atom helper is
# used to prove the owner's own balance.
_MOLECULAR_MASSES = {
    "glucose": {"C": 72.0, "H": 12.0, "O": 96.0},
    "pdo": {"C": 36.0, "H": 8.0, "O": 32.0},
    "glycerol": {"C": 36.0, "H": 8.0, "O": 48.0},
    "cells": {"C": 12.0, "H": 1.78, "O": 9.6, "N": 2.66},
    "biopolymer": {"C": 12.0, "H": 1.78, "O": 9.6, "N": 2.66},
    "csl_protein": {"C": 12.0, "H": 1.57, "O": 4.96, "N": 4.06, "S": 0.224},
    "csl_lactate": {"C": 36.0, "H": 6.0, "O": 48.0},
    "corn_oil": {"C": 216.0, "H": 34.0, "O": 32.0},
    "tween80": {"C": 768.0, "H": 124.0, "O": 416.0},
    "vitamin_b12": {
        "C": 756.693,
        "H": 88.704,
        "N": 196.098,
        "O": 223.986,
        "P": 30.974,
        "Co": 58.933,
    },
    "ammonium_sulfate": {"N": 28.0, "H": 8.0, "S": 32.0, "O": 64.0},
    "potassium_dihydrogen_phosphate": {"K": 39.0, "H": 2.0, "P": 31.0, "O": 64.0},
    "magnesium_sulfate": {"Mg": 24.0, "S": 32.0, "O": 64.0},
    "calcium_chloride": {"Ca": 40.0, "Cl": 71.0},
    "ammonia": {"N": 14.0, "H": 3.0},
    "sodium_hydroxide": {"Na": 23.0, "O": 16.0, "H": 1.0},
    "hydrochloric_acid": {"H": 1.0, "Cl": 35.5},
    "sulfuric_acid": {"H": 2.0, "S": 32.0, "O": 64.0},
    "sodium_sulfate": {"Na": 46.0, "S": 32.0, "O": 64.0},
    "sodium_chloride": {"Na": 23.0, "Cl": 35.5},
    "water": {"H": 2.0, "O": 16.0},
    "carbon_dioxide": {"C": 12.0, "O": 32.0},
    "oxygen": {"O": 32.0},
    "nitrogen": {"N": 28.0},
}


def _element_totals(compounds: Mapping[str, float]) -> dict[str, float]:
    """Evaluate a mass ledger against independent source molecular mass ratios."""
    totals: dict[str, float] = defaultdict(float)
    for name, mass in compounds.items():
        molecule = _MOLECULAR_MASSES[name]
        molecular_mass = fsum(molecule.values())
        for element, contribution in molecule.items():
            totals[element] += mass * contribution / molecular_mass
    return dict(totals)


@pytest.mark.parametrize("origin", ("biogenic", "fossil", "atmospheric_dac"))
def test_pdo_source_recipe_conserves_actual_atoms_and_distinct_carbon_origins(
    origin: _CarbonOrigin,
) -> None:
    """Close the modeled factory material balance.

    Emptage's broth performance and NREL's once-only seed consumption fix fresh
    glucose. Jankovic's solution defines active output; B12 and dry salt doses
    keep their own references. Residual solutes retain their composition, with no enzyme-reference
    mass or native donor sewage.
    """
    reference = _pdo_reference(origin)
    ledger = reference.ledger
    inputs, outputs = ledger["physical_inputs"], ledger["physical_outputs"]
    residual = ledger["residual_solutes"]
    source_active_output = 2869 * 0.9992
    production_glucose = (23800 * 0.129 / 0.34 + 23800 * 0.0001) / source_active_output
    fresh_glucose = production_glucose / (1 - 109 / 2418)
    medium_scale = fresh_glucose / 2418
    b12 = production_glucose * 0.000025
    assert inputs["glucose"] == pytest.approx(fresh_glucose, rel=2e-14)
    assert inputs["vitamin_b12"] == pytest.approx(b12, rel=2e-14)
    assert reference.service_amounts["vitamin_b12"] == pytest.approx(b12, rel=2e-14)
    assert inputs["potassium_dihydrogen_phosphate"] == pytest.approx(67 * medium_scale * 2 / 4.3)
    assert inputs["magnesium_sulfate"] == pytest.approx(67 * medium_scale * 0.3 / 4.3 * 120 / 246)
    assert inputs["calcium_chloride"] == pytest.approx(67 * medium_scale * 0.4 / 4.3 * 111 / 147)
    assert residual["vitamin_b12"] == inputs["vitamin_b12"]
    assert residual["pdo"] == pytest.approx(23800 * 0.129 / source_active_output - 1)
    assert residual["glycerol"] == pytest.approx(23800 * 0.0148 / source_active_output)
    assert residual["glucose"] == pytest.approx(23800 * 0.0001 / source_active_output)
    assert outputs["pdo"] - residual["pdo"] == pytest.approx(1.0)
    assert reference.product_water_kg == pytest.approx(0.0008 / 0.9992)

    incoming_atoms, outgoing_atoms = _element_totals(inputs), _element_totals(outputs)
    assert set(incoming_atoms) == {"C", "H", "O", "N", "S", "P", "K", "Mg", "Ca", "Cl", "Na", "Co"}
    assert outgoing_atoms == pytest.approx(incoming_atoms, rel=3e-14, abs=2e-14)
    assert fsum(outputs.values()) == pytest.approx(fsum(inputs.values()), rel=3e-14)
    residual_atoms = _element_totals(residual)
    assert reference.load.elements_kg == pytest.approx(residual_atoms, rel=2e-14)
    assert residual_atoms["Co"] == pytest.approx(b12 * 58.933 / 1355.388)
    assert reference.load.chloride_kg == residual_atoms["Cl"]
    assert reference.water_kg + fsum(residual.values()) == pytest.approx(
        1000 * reference.solution_volume_m3
    )

    # Track origins independently of the manufacturer's environmental vector.
    # No actual auxiliary atom becomes additional principal plastic.
    actual_carbon: dict[str, float] = defaultdict(float)
    for pool in (*reference.load.carbon, *reference.carbon_dioxide):
        actual_carbon[pool.origin] += pool.carbon_kg
        assert not pool.principal
    primer = 1.0 if origin == "biogenic" else 25 / 738
    actual_carbon["biogenic"] += (36 / 76) * primer
    if origin != "biogenic":
        actual_carbon[origin] += (36 / 76) * (1 - primer)
    glucose_carbon = fresh_glucose * 72 / 180
    protein_carbon = inputs["csl_protein"] * 12 / 22.814
    lactate_carbon = inputs["csl_lactate"] * 36 / 90
    oil_carbon = inputs["corn_oil"] * 216 / 282
    tween_carbon = inputs["tween80"] * 768 / 1308
    cofactor_carbon = b12 * 756.693 / 1355.388
    expected_carbon: dict[str, float] = defaultdict(float)
    expected_carbon["biogenic"] = (
        glucose_carbon * primer
        + protein_carbon
        + lactate_carbon
        + oil_carbon
        + tween_carbon * 24 / 64
        + cofactor_carbon * 62 / 63
    )
    expected_carbon["fossil"] = tween_carbon * 40 / 64 + cofactor_carbon / 63
    if origin != "biogenic":
        expected_carbon[origin] += glucose_carbon * (1 - primer)
    assert dict(actual_carbon) == pytest.approx(dict(expected_carbon), rel=3e-14)
    assert fsum(pool.carbon_kg for pool in reference.carbon_dioxide) == pytest.approx(
        outputs["carbon_dioxide"] * 12 / 44
    )
    assert fsum(pool.carbon_kg for pool in reference.load.carbon) == pytest.approx(
        residual_atoms["C"]
    )


def test_pdo_water_reservations_recycle_and_separate_neutralization_are_paid_once() -> None:
    """Resolve source water reservations and membrane-adjusted IX regeneration."""
    reference = _pdo_reference("biogenic")
    water = reference.ledger["water"]
    inputs = reference.ledger["physical_inputs"]
    broth = reference.ledger["broth_solutes"]
    active = 2869 * 0.9992
    assert water["broth_water"] == pytest.approx(23800 / active - fsum(broth.values()))
    # Use NREL's 427/2418 stock ratio.
    assert water["glucose_stock_reservation"] == pytest.approx(inputs["glucose"] * 427 / 2418)
    assert water["salt_preparation_reservation"] == pytest.approx(
        inputs["magnesium_sulfate"] * 126 / 120 + inputs["calcium_chloride"] * 36 / 111
    )
    df_water = 15508 / active
    filter_water = (10611 * 0.8663 + 456 * 0.8397) / active
    diverted = 0.02 * 28241 / active
    recycle = water["broth_water"] + df_water + diverted - filter_water - 0.0008 / 0.9992
    assert water["diafiltration"] == pytest.approx(df_water)
    assert water["filter_retentate"] == pytest.approx(filter_water)
    assert water["ix_rinse_diverted"] == pytest.approx(diverted)
    assert water["recycle"] == pytest.approx(recycle)
    assert water["recycled_medium"] == pytest.approx(recycle - df_water)
    assert water["fresh_process"] == pytest.approx(water["prepared_medium"] - recycle + df_water)
    assert water["prepared_medium"] == pytest.approx(
        water["glucose_stock_reservation"]
        + water["salt_preparation_reservation"]
        + water["free_preparation"]
    )
    assert water["recycled_df"] == water["diafiltration"]
    assert df_water < recycle < df_water + water["prepared_medium"]
    assert water["prepared_medium"] == pytest.approx(
        water["broth_water"]
        - water["reaction_formed"]
        + water["air_out"]
        - water["air_in"]
        - water["csl_water"]
    )
    # Source entrainment sizes IX beds. B12 retains Co/P as molecular residual atoms.
    passage = 1 - (10611 * 0.0120 + 456 * 0.0975) / (23800 * 0.129)
    equivalents = (
        1000
        * passage
        * (
            broth["sodium_sulfate"] / 71
            + broth["potassium_dihydrogen_phosphate"] / 136
            + broth["magnesium_sulfate"] / 60
            + broth["calcium_chloride"] / 55.5
        )
    )
    cation, anion = equivalents * 0.99 * 24 / 1800, equivalents * 0.99 * 24 / 1300
    assert inputs["hydrochloric_acid"] == pytest.approx(cation * 2 * 60 / 24)
    assert water["ix_carrier"] == pytest.approx((cation * 940 + anion * 965) * 2 / 24)
    assert water["ix_rinse"] == pytest.approx((cation * 3.5 + anion * 3) * 1000 / 24)
    assert water["ix_backwash"] == pytest.approx(
        (cation * 20 / (13 * 0.3048) + anion * 6 / (10 * 0.3048)) * 1000 / (3 * 24)
    )
    cycles = 0.00088006234803261255 * inputs["glucose"]
    assert water["cip_cleaning"] == pytest.approx(2990 * cycles)
    assert water["sip_contact"] == pytest.approx(500 * cycles)
    assert water["cip_cleaning"] == pytest.approx(
        water["cip_stock_carrier_reservation"] + water["cip_free_cleaning"]
    )
    assert water["cip_neutralization_formed"] == pytest.approx(inputs["sulfuric_acid"] * 36 / 98)
    assert water["ix_neutralization_formed"] == pytest.approx(
        inputs["hydrochloric_acid"] * 18 / 36.5
    )
    regeneration = water["ix_carrier"] + water["ix_rinse"] + water["ix_backwash"]
    assert reference.supply_amounts["deionised_water"] == pytest.approx(
        water["fresh_process"] + regeneration + water["cip_cleaning"] + water["sip_contact"]
    )
    assert reference.water_kg == pytest.approx(
        water["filter_retentate"]
        + regeneration
        - water["ix_rinse_diverted"]
        + water["cip_cleaning"]
        + water["sip_contact"]
        + water["cip_neutralization_formed"]
        + water["ix_neutralization_formed"]
    )


def test_pdo_membrane_partition_sizes_actual_kill_and_supplied_operating_duties() -> None:
    """Keep source entrainment, cell deactivation and whole supplied meters distinct."""
    reference = _pdo_reference("biogenic")
    ledger = reference.ledger
    broth, water = ledger["broth_solutes"], ledger["water"]
    active = 2869 * 0.9992
    df_fraction = 10611 * 0.0120 / (23800 * 0.129)
    uf_fraction = 456 * 0.0975 / (23800 * 0.129)
    polymer_df_fraction = 10611 * 0.0001 / (23800 * 0.0010)
    phases = {name: ledger[f"{name}_solutes"] for name in ("df", "uf", "ix", "bottoms")}
    for name, amount in broth.items():
        total = fsum(phase.get(name, 0) for phase in phases.values())
        assert total == pytest.approx(amount - (1 if name == "pdo" else 0))
        if name == "cells":
            assert phases["df"][name] == amount
        elif name == "biopolymer":
            assert phases["df"][name] == pytest.approx(amount * polymer_df_fraction)
            assert phases["uf"][name] == pytest.approx(amount * (1 - polymer_df_fraction))
        elif name in {"corn_oil", "tween80"}:
            assert phases["uf"][name] == amount
        else:
            assert phases["df"][name] == pytest.approx(amount * df_fraction)
            assert phases["uf"][name] == pytest.approx(amount * uf_fraction)
    for name in (
        "sodium_sulfate",
        "potassium_dihydrogen_phosphate",
        "magnesium_sulfate",
        "calcium_chloride",
    ):
        assert ledger["ix_feed_ions"][name] == pytest.approx(
            broth[name] * (1 - df_fraction - uf_fraction)
        )
        assert phases["ix"][name] == pytest.approx(0.99 * ledger["ix_feed_ions"][name])
    assert phases["ix"]["vitamin_b12"] == pytest.approx(
        broth["vitamin_b12"] * (1 - df_fraction - uf_fraction) * 0.10
    )

    heat = ledger["heat_mj"]
    inputs = ledger["physical_inputs"]
    dissolved = fsum(
        amount
        for name, amount in broth.items()
        if name not in {"cells", "biopolymer", "corn_oil", "tween80"}
    )
    killed = (
        10611 * 0.8663 / active
        + broth["cells"]
        + polymer_df_fraction * broth["biopolymer"]
        + df_fraction * dissolved
    )
    assert heat["retentate_deactivation"] == pytest.approx(killed * 0.00418 * 84)
    fermentation_naoh = (
        (inputs["csl_protein"] * 0.224 / 22.814 + inputs["ammonium_sulfate"] * 32 / 132) * 80 / 32
    )
    charged = (
        water["prepared_medium"]
        + fermentation_naoh
        + water["csl_water"]
        + fsum(
            mass
            for name, mass in inputs.items()
            if name
            not in {
                "water",
                "oxygen",
                "nitrogen",
                "sodium_hydroxide",
                "hydrochloric_acid",
                "sulfuric_acid",
            }
        )
    )
    assert heat["feed_sterilization"] == pytest.approx(charged * 0.00418 * 100)
    assert heat["feed_cooling"] == pytest.approx(charged * 0.00418 * 86)
    cycles = 0.00088006234803261255 * inputs["glucose"]
    assert reference.supply_amounts["heat"] == pytest.approx(
        (charged * 100 + killed * 84) * 0.00418 + 1375 * cycles + 0.403 * 3.6 / 0.9992
    )
    assert reference.supply_amounts["electricity"] == pytest.approx(
        5340 * inputs["glucose"] / 2418 + 0.480 / 0.9992 + 17.4 * cycles / 3.6
    )
    # NREL includes chiller/compressor rejection. Recovery cooling is separate;
    # the source-backed cell kill adds no active return-cooling purchase.
    assert reference.service_amounts["ambient_cooling"] == pytest.approx(
        charged * 0.00418 * 86
        + (8.104 + 0.468 + 1.172) * 4184 * inputs["glucose"] / 2418
        + 21000 / 1.27 / 8000 * 3600 / active
    )
    assert reference.supply_amounts["chilled_cooling"] == pytest.approx(
        25000 / 15.95 / 8000 * 3600 / active
    )


@pytest.mark.parametrize("future", (False, True))
def test_actual_pdo_residual_is_assembled_once_into_complete_dated_ch_receiving(
    tmp_path: Path, future: bool
) -> None:
    """Compile the real PDO load into collection, treatment and all sludge terminals.

    Existing CH fixtures own source calibration and nonunit native products.
    This case protects the project-frontier handoff of the assembled
    process/CIP/IX residue, rather than repeating Doka's internal equations.
    """
    reference = _pdo_reference("fossil")
    with water_receiving_boundary(tmp_path, future=future) as (bd, original_context):
        database = original_context.ordinary.database_name
        context = original_context
        # Preserve native NaN parameters in the comparison: unlike JSON text,
        # IEEE NaN values are unequal to themselves after separate reads.
        native_before = {
            node.key: json.dumps([dict(row) for row in node.exchanges()], sort_keys=True)
            for node in bd.Database(database)
        }
        service_id = "pdo_fossil_water_collection_ch"
        records = chemical_organic_wastewater_datasets(
            context,
            service_id=service_id,
            load=reference.load,
            solution_volume_m3=reference.solution_volume_m3,
            water_kg=reference.water_kg,
            complete_carbon_oxidation=True,
            phosphorus_equilibrium=True,
        )
        collection_key = (context.provider_database, provider_service_code(service_id))
        plant_key = (
            context.provider_database,
            provider_service_code("pdo_fossil_water_treatment_ch"),
        )
        assert len(records) == 10
        collection_terms = json.loads(
            next(
                row["project_terms_json"]
                for row in context.technology_rows
                if row["provider_service_id"] == service_id
                and row["component_id"] == "leachate_collection"
            )
        )
        plant_terms = json.loads(
            next(
                row["project_terms_json"]
                for row in context.technology_rows
                if row["provider_service_id"] == "pdo_fossil_water_treatment_ch"
                and row["component_id"] == "actual_wastewater_treatment"
            )
        )
        raw = collection_terms["raw_load_per_caller_reference"]
        water = plant_terms["load_per_caller_reference"]
        assert raw["elements_kg"] == pytest.approx(reference.load.elements_kg)
        assert raw["chloride_kg"] == reference.load.chloride_kg
        assert water["volume_m3"] == reference.solution_volume_m3
        assert water["water"]["incoming_kg"] == reference.water_kg
        assert fsum(raw["elements_kg"].values()) + reference.water_kg == pytest.approx(
            1000 * reference.solution_volume_m3
        )
        assert {pool["origin"] for pool in raw["carbon"]} == {"biogenic", "fossil"}
        assert all(not pool["principal"] for pool in raw["carbon"])
        assert collection_terms["treated_fraction"] == 0.8
        for element in ("Co", "Ca", "K", "Mg"):
            assert raw["elements_kg"][element] > 0
            assert water["final_sludge"]["elements_kg"][element] > 0

        write_brightway_database(bd, context.provider_database, records, {})
        collection = bd.get_node(database=collection_key[0], code=collection_key[1])
        plant = bd.get_node(database=plant_key[0], code=plant_key[1])
        assert next(iter(collection.production()))["amount"] == (-10 if future else -5)
        assert next(iter(plant.production()))["amount"] == (-14 if future else -7)
        result = LCA({collection: -reference.solution_volume_m3})
        result.lci()
        assert all(isfinite(value) for value in result.supply_array)
        assert result.supply_array[result.dicts.activity[plant.id]] == pytest.approx(
            0.8 * reference.solution_volume_m3 / (14 if future else 7), rel=3e-7
        )
        for key in records:
            receiving = bd.get_node(database=key[0], code=key[1])
            assert result.supply_array[result.dicts.activity[receiving.id]] > 0
        # Residual Co, Ca, K and Mg must reach elementary flows in the complete
        # inventory, beyond their intermediate ash quantities.
        by_name: dict[str, float] = defaultdict(float)
        for identifier, index in result.dicts.biosphere.items():
            flow = bd.get_node(id=identifier)
            by_name[str(flow["name"])] += float(result.inventory[index, :].sum())
        for name in ("Cobalt II", "Calcium II", "Potassium I", "Magnesium"):
            assert by_name[name] > 0
        # Actual cyanocobalamin-derived cobalt requires conventional scrubber
        # precipitation; both original/dated source suppliers retain that duty.
        for suffix in ("a", "b"):
            ferric = bd.get_node(database=database, code=f"organic_ch_FeCl3_{suffix}")
            assert result.supply_array[result.dicts.activity[ferric.id]] > 0
        assert {
            key: json.dumps(
                [dict(row) for row in bd.get_node(database=key[0], code=key[1]).exchanges()],
                sort_keys=True,
            )
            for key in native_before
        } == native_before


def _pdo_procurement_sources(
    services: dict[str, Any],
    database: str,
    existing: Mapping[tuple[str, str], Mapping[str, Any]],
    emission: tuple[str, str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, tuple[str, str]]]:
    """Supply complete small native functions for the actual PDO procurement graph.

    Native outputs are deliberately nonunit. Two manufacturer positions and
    four transport modes are retained for each new market; the dated salt
    producer has several actual acid/alkali suppliers. The complete CH fixture
    supplies its independently tested receiving internals unchanged.
    """
    factor = 2.0 if database == "future" else 1.0
    records: dict[tuple[str, str], dict[str, Any]] = {}
    keys: dict[str, tuple[str, str]] = {}
    descriptors = {
        (node["name"], node["reference product"], node["unit"], node["location"]): key
        for key, node in existing.items()
    }

    def row(key: tuple[str, str], amount: float) -> dict[str, Any]:
        """Keep observable amount, uncertainty and native-allocation metadata."""
        return {
            "input": key,
            "amount": amount,
            "type": "biosphere" if key == emission else "technosphere",
            "uncertainty type": 2 if amount > 0 else 0,
            "loc": log(amount) + 0.07 if amount > 0 else amount,
            "scale": 0.12,
            "allocation": "invented native source allocation",
        }

    def source(
        tag: str,
        selector: dict[str, Any],
        *,
        production: float = 2.0,
        exchanges: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        """Add a synthetic native inventory with distinct CH supplier identities."""
        descriptor = (
            selector["name"],
            selector["reference_product"],
            selector["unit"],
            selector["exact_location"],
        )
        if descriptor in descriptors:
            keys[tag] = descriptors[descriptor]
            return keys[tag]
        key = database, f"pdo_fixture_{tag}"
        record = _process(
            database,
            key[1],
            descriptor[0],
            descriptor[1],
            location=descriptor[3],
            production_amount=production * factor,
            exchanges=exchanges if exchanges is not None else [row(emission, 0.2 * factor)],
        )
        record["unit"] = descriptor[2]
        records[key] = record
        descriptors[descriptor] = key
        keys[tag] = key
        return key

    def simple(tag: str, *, product: str | None = None, production: float = 2.0) -> tuple[str, str]:
        """Make an independently emitting material, operation or receiving source."""
        return source(
            tag,
            {
                "name": f"invented {tag}",
                "reference_product": product or tag,
                "unit": "kilogram",
                "exact_location": "RER",
            },
            production=production,
        )

    for label, selector in services["propanediol_manufacture_biogenic"]["construction"][
        "supplies"
    ].items():
        source(label, selector, production=-2 if label.startswith("spent_") else 2)
    independent = (
        "glucose_biogenic",
        "glucose_fossil",
        "glucose_atmospheric_dac",
        "maize_starch_market",
        "pa_oleth10",
        "ambient_cooling_operation",
    )
    for service_id in independent:
        service = services[service_id]
        source(
            service_id,
            {
                "name": f"admitted {service_id}",
                "reference_product": service["reference_product"],
                "unit": service["unit"],
                "exact_location": "RER",
            },
        )
    source("koh", services["pdo_potassium_hydroxide"]["construction"]["selector"])
    salt = services["pdo_potassium_dihydrogen_phosphate_manufacture"]["construction"]
    salt_rows = []
    for index, amount in enumerate((0.2, 0.3, 0.6 if database == "future" else 0.5)):
        alkali = simple(
            f"alkali_{index}",
            product=salt["replace_technosphere"][0]["selector"]["reference_product"],
        )
        salt_rows.append(row(alkali, amount * factor))
    for index, amount in enumerate((0.7, 1.1 if database == "future" else 0.9)):
        acid = simple(
            f"acid_{index}", product=salt["scale_technosphere"][0]["selector"]["reference_product"]
        )
        salt_rows.append(row(acid, amount * factor))
    salt_waste = simple("salt_waste", production=-2)
    salt_rows.extend((row(salt_waste, -0.04 * factor), row(emission, 0.07 * factor)))
    salt_key = source("salt_production", salt["base"]["selector"], exchanges=salt_rows)
    enzyme = services["pdo_vitamin_b12_manufacture"]["construction"]["components"][0]["selector"]
    enzyme_feed = simple("enzyme_feed")
    source(
        "enzyme_production",
        enzyme,
        exchanges=[row(enzyme_feed, 3 * factor), row(emission, 0.25 * factor)],
    )
    markets = (
        ("b12", "pdo_vitamin_b12_delivered"),
        ("tween", "pdo_tween80_delivered"),
        ("salt", "pdo_potassium_dihydrogen_phosphate_delivered"),
        ("pdo", "propanediol_delivered_biogenic"),
    )
    for tag, service_id in markets:
        construction = services[service_id]["construction"]
        rows = []
        for index, replacement in enumerate(construction["replace_technosphere"]):
            if tag == "salt":
                producer = salt_key
            else:
                producer = source(f"{tag}_donor_{index}", replacement["selector"])
            # Source allocation can change separately from reference output.
            amount = (
                4.0
                if len(construction["replace_technosphere"]) == 1
                else (1.0, 3.5 if database == "future" else 3.0)[index]
            )
            producer_row = row(producer, amount * factor)
            producer_row.update(
                {
                    "activity": f"invented-{tag}-producer-uuid-{index}",
                    "flow": f"invented-{tag}-donor-product-uuid",
                    "name": replacement["selector"]["reference_product"],
                    "reference product": replacement["selector"]["reference_product"],
                    "properties": {"water content": {"amount": 0.2, "unit": "dimensionless"}},
                }
            )
            rows.append(producer_row)
        for index, mode in enumerate(("road", "rail", "sea", "barge"), start=1):
            freight = source(
                f"{tag}_{mode}",
                {
                    "name": f"{tag} {mode} transport",
                    "reference_product": f"{tag} {mode} transport",
                    "unit": "ton kilometer",
                    "exact_location": "GLO",
                },
            )
            rows.append(row(freight, index * factor / 20))
        source(f"{tag}_market", construction["base"]["selector"], production=4, exchanges=rows)
    return records, keys


@pytest.mark.parametrize("future", (False, True))
def test_pdo_active_root_preserves_dated_complete_procurement_and_actual_solution(
    tmp_path: Path, future: bool
) -> None:
    """Solve all origins through actual manufacturing, delivery and CH declarations.

    The independently admitted glucose, CSL, Oleth10 and ambient operation
    terminate at complete native fixture functions. Cofactor, salt and delivery
    transformations use the production frontier compiler and matrices.
    """
    catalogue = load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    origins: tuple[_CarbonOrigin, ...] = ("biogenic", "fossil", "atmospheric_dac")
    selected_ids = {
        "pdo_vitamin_b12_manufacture",
        "pdo_vitamin_b12_delivered",
        "pdo_tween80_delivered",
        "pdo_potassium_dihydrogen_phosphate_manufacture",
        "pdo_potassium_dihydrogen_phosphate_delivered",
        "pdo_potassium_hydroxide",
        *(
            f"propanediol{suffix}_{origin}"
            for origin in origins
            for suffix in ("", "_manufacture", "_delivered")
        ),
    }
    selected = [services[service_id] for service_id in sorted(selected_ids)]
    independent = (
        "glucose_biogenic",
        "glucose_fossil",
        "glucose_atmospheric_dac",
        "maize_starch_market",
        "pa_oleth10",
        "ambient_cooling_operation",
    )
    for service_id in independent:
        service = services[service_id]
        selected.append(
            {
                **service,
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": {
                        "name": f"admitted {service_id}",
                        "reference_product": service["reference_product"],
                        "unit": service["unit"],
                        "exact_location": "RER",
                    },
                },
            }
        )
    selected.extend(
        service
        for service in services.values()
        if any(service["provider_service_id"].startswith(f"pdo_{origin}_") for origin in origins)
        and service["provider_service_id"].endswith("_ch")
    )
    provider = PreparedProviderAuthority(
        identity="invented_complete_pdo",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(tmp_path / "brightway", "pdo_procurement", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        biosphere = native_chemical_biosphere()
        emission = next(
            key
            for key, flow in biosphere.items()
            if flow["name"] == "Carbon dioxide, fossil" and tuple(flow["categories"]) == ("air",)
        )
        bd.Database("biosphere").write(biosphere, searchable=False)
        by_database = {}
        databases = ("original", "future") if future else ("original",)
        for database in databases:
            native = (
                _dated_water_inventory(catalogue, database)
                if database == "future"
                else native_chemical_inventory(catalogue, database)
            )
            additions, keys = _pdo_procurement_sources(services, database, native, emission)
            native.update(additions)
            bd.Database(database).write(native, searchable=False)
            by_database[database] = keys
        database = "future" if future else "original"
        keys = by_database[database]
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            ordinary_database=database,
            technology_database=database,
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )

        def prepared(service_id: str) -> Any:
            """Read a persisted package activity from the actual compiled frontier."""
            return bd.get_node(
                database=frontier.database_name, code=provider_service_code(service_id)
            )

        salt = prepared("pdo_potassium_dihydrogen_phosphate_manufacture")
        source_salt = bd.get_node(database=database, code=keys["salt_production"][1])
        assert (
            next(iter(salt.production()))["amount"]
            == next(iter(source_salt.production()))["amount"]
        )
        salt_rows = list(salt.technosphere())
        for source_row in source_salt.technosphere():
            input_key = source_row.input.key
            is_alkali = input_key in {keys[f"alkali_{i}"] for i in range(3)}
            target = prepared("pdo_potassium_hydroxide").key if is_alkali else input_key
            multiplier = (
                120 / 136 * 56 / 40
                if is_alkali
                else 120 / 136
                if input_key in {keys["acid_0"], keys["acid_1"]}
                else 1
            )
            expected_amount = source_row["amount"] * multiplier
            matching = [
                row
                for row in salt_rows
                if row.input.key == target and abs(row["amount"] - expected_amount) < 1e-12
            ]
            assert len(matching) == 1
            assert matching[0]["loc"] == pytest.approx(
                source_row["loc"] + log(multiplier)
                if source_row["amount"] > 0
                else source_row["loc"]
            )
            assert matching[0]["scale"] == source_row["scale"]
            assert matching[0]["allocation"] == source_row["allocation"]
        assert len(salt_rows) == 6
        assert [row["amount"] for row in salt.biosphere()] == [
            row["amount"] for row in source_salt.biosphere()
        ]
        # Each distribution operation retains its complete native production
        # reference and every transport row, including source uncertainty.
        for tag, service_id in (
            ("b12", "pdo_vitamin_b12_delivered"),
            ("tween", "pdo_tween80_delivered"),
            ("salt", "pdo_potassium_dihydrogen_phosphate_delivered"),
            ("pdo", "propanediol_delivered_biogenic"),
        ):
            market = bd.get_node(database=database, code=keys[f"{tag}_market"][1])
            delivered = prepared(service_id)
            assert (
                next(iter(delivered.production()))["amount"]
                == next(iter(market.production()))["amount"]
            )
            replaced_id = services[service_id]["construction"]["replacements"][0]["service_id"]
            replacement = prepared(replaced_id)
            replacement_rows = [
                row for row in delivered.technosphere() if row.input.key == replacement.key
            ]
            source_producers = [row for row in market.technosphere() if "activity" in row]
            assert len(replacement_rows) == len(source_producers)
            for row in replacement_rows:
                assert "activity" not in row
                assert "flow" not in row
                assert "properties" not in row
                assert row["reference product"] == services[replaced_id]["reference_product"]
            for mode in ("road", "rail", "sea", "barge"):
                key = keys[f"{tag}_{mode}"]
                source_row = next(row for row in market.technosphere() if row.input.key == key)
                delivered_row = next(
                    row for row in delivered.technosphere() if row.input.key == key
                )
                for field in ("amount", "uncertainty type", "loc", "scale", "allocation"):
                    assert delivered_row[field] == source_row[field]
        q_enzyme = (1.004 / 0.0027) / ((4.16666666666667 * 0.8) * 180.156 / 162.141)
        for origin in origins:
            reference = _pdo_reference(origin)
            root, manufacturer = (
                prepared(f"propanediol_{origin}"),
                prepared(f"propanediol_manufacture_{origin}"),
            )
            assert next(iter(root.technosphere()))["amount"] == pytest.approx(1 / 0.9992)
            production = next(iter(manufacturer.production()))
            assert production["amount"] == 1
            assert production["properties"]["water in wet mass"]["amount"] == pytest.approx(0.0008)
            assert production["properties"]["dry mass"]["amount"] == 0.9992
            properties = production["properties"]
            pdo_elements = _MOLECULAR_MASSES["pdo"]
            dry_carbon = pdo_elements["C"] / fsum(pdo_elements.values())
            # The admitted synthetic-glucose route retains 25/738 biogenic primer.
            fossil_share = 1 - 25 / 738 if origin == "fossil" else 0.0
            fossil_content = properties["carbon content, fossil"]["amount"]
            nonfossil_content = properties["carbon content, non-fossil"]["amount"]
            assert fossil_content == pytest.approx(dry_carbon * fossil_share)
            assert nonfossil_content == pytest.approx(dry_carbon * (1 - fossil_share))
            assert properties["carbon allocation"]["amount"] == pytest.approx(
                properties["dry mass"]["amount"] * (fossil_content + nonfossil_content)
            )
            expected_emissions: dict[str, float] = defaultdict(float)
            for pool in reference.carbon_dioxide:
                name = (
                    "Carbon dioxide, fossil"
                    if pool.origin == "fossil"
                    else "Carbon dioxide, non-fossil"
                )
                expected_emissions[name] += pool.carbon_kg * 44 / 12 * 0.9992
            expected_emissions["Water"] = (
                (reference.ledger["water"]["air_out"] - reference.ledger["water"]["air_in"])
                * 0.9992
                / 1000
            )
            assert {
                row.input["name"]: row["amount"] for row in manufacturer.biosphere()
            } == pytest.approx(dict(expected_emissions))
            actual_rows: dict[tuple[str, str], float] = defaultdict(float)
            for exchange in manufacturer.technosphere():
                actual_rows[exchange.input.key] += exchange["amount"]
            expected_rows: dict[tuple[str, str], float] = defaultdict(float)
            for label, amount in reference.supply_amounts.items():
                expected_rows[keys[label]] += amount * 0.9992
            dependencies = services[f"propanediol_manufacture_{origin}"]["construction"]["services"]
            for label, amount in reference.service_amounts.items():
                expected_rows[prepared(dependencies[label]["service_id"]).key] += amount * 0.9992
            collection = prepared(f"pdo_{origin}_water_collection_ch")
            expected_rows[collection.key] = -reference.solution_volume_m3 * 0.9992
            assert dict(actual_rows) == pytest.approx(dict(expected_rows))
            calculation = LCA({root: 1.0})
            calculation.lci()
            assert calculation.supply_array[
                calculation.dicts.activity[manufacturer.id]
            ] == pytest.approx(1 / 0.9992, rel=3e-7)
            assert calculation.supply_array[
                calculation.dicts.activity[collection.id]
            ] == pytest.approx(reference.solution_volume_m3 / (10 if future else 5), rel=3e-7)
            cargos = {
                "b12": reference.service_amounts["vitamin_b12"],
                "tween": reference.service_amounts["tween80"],
                "salt": reference.service_amounts["potassium_dihydrogen_phosphate"],
                "pdo": 1 / 0.9992,
            }
            for tag, cargo in cargos.items():
                market = bd.get_node(database=database, code=keys[f"{tag}_market"][1])
                denominator = next(iter(market.production()))["amount"]
                for mode in ("road", "rail", "sea", "barge"):
                    freight = bd.get_node(database=database, code=keys[f"{tag}_{mode}"][1])
                    amount = next(
                        row["amount"]
                        for row in market.technosphere()
                        if row.input.key == freight.key
                    )
                    freight_output = next(iter(freight.production()))["amount"]
                    assert calculation.supply_array[
                        calculation.dicts.activity[freight.id]
                    ] == pytest.approx(cargo * amount / denominator / freight_output, rel=4e-7)
            enzyme = bd.get_node(database=database, code=keys["enzyme_production"][1])
            enzyme_output = next(iter(enzyme.production()))["amount"]
            producer_share = 4.5 / 4 if future else 1.0
            oleth = prepared("pa_oleth10")
            assert calculation.supply_array[calculation.dicts.activity[oleth.id]] == pytest.approx(
                cargos["tween"] * producer_share, rel=4e-7
            )
            for tag in ("tween_donor_0", "tween_donor_1", "b12_donor_0", "pdo_donor_0"):
                donor = bd.get_node(database=database, code=keys[tag][1])
                assert calculation.supply_array[calculation.dicts.activity[donor.id]] == 0
            expected_enzyme = cargos["b12"] * producer_share * q_enzyme / enzyme_output
            assert calculation.supply_array[calculation.dicts.activity[enzyme.id]] == pytest.approx(
                expected_enzyme, rel=4e-7
            )
            enzyme_feed = bd.get_node(database=database, code=keys["enzyme_feed"][1])
            feed_amount = next(iter(enzyme.technosphere()))["amount"]
            assert calculation.supply_array[
                calculation.dicts.activity[enzyme_feed.id]
            ] == pytest.approx(
                expected_enzyme * feed_amount / next(iter(enzyme_feed.production()))["amount"],
                rel=4e-7,
            )


def test_pdo_salt_rejects_a_dated_source_without_its_declared_acid_feed(tmp_path: Path) -> None:
    """Reject the synthetic dated salt source when its phosphoric acid feeds are absent."""
    catalogue = load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    identifiers = ("pdo_potassium_hydroxide", "pdo_potassium_dihydrogen_phosphate_manufacture")
    provider = PreparedProviderAuthority(
        identity="invented_missing_phosphate_feed",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": [services[name] for name in identifiers], "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(tmp_path / "brightway", "pdo_missing_acid", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        flow = ("biosphere", "carbon_dioxide")
        bd.Database("biosphere").write(
            {
                flow: {
                    "name": "Carbon dioxide, fossil",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
            },
            searchable=False,
        )
        for database in ("original", "future"):
            records, keys = _pdo_procurement_sources(services, database, {}, flow)
            if database == "future":
                salt = records[keys["salt_production"]]
                salt["exchanges"] = [
                    row
                    for row in salt["exchanges"]
                    if row["input"] not in {keys["acid_0"], keys["acid_1"]}
                ]
            bd.Database(database).write(records, searchable=False)
        with pytest.raises(ValueError, match="feed quantity scaling.*absent selectors") as failure:
            prepare_provider_frontier(
                bd,
                background_scenario="remind_ssp2_npi",
                year=2050,
                ordinary_database="future",
                technology_database="future",
                material_recipe=physical_material_recipe(),
                provider=provider,
                resolver_cache=_NativeResolverCache("original", "biosphere"),
            )
        assert "phosphoric acid, industrial grade, without water, in 85% solution state" in str(
            failure.value
        )
        assert "pdo_potassium_dihydrogen_phosphate_manufacture" in str(failure.value)
