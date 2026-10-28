"""Manufacture the declared PDO solution and receive its actual residual medium.

Emptage's glucose performance, Jankovic's recovery and the transferred NREL
industrial operation define one stationary engineering reference. Native suppliers follow the
selected background; the target recipe, atoms, and water quantities remain fixed.
Complete supplied functions retain their internal inventories. Molecular
residues, empirical wastewater indicators and environmental supply proxies
have distinct meanings, including the trace cyanocobalamin supply proxy.
"""

import json
from dataclasses import asdict, dataclass
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .leachate_load import _CarbonOrigin, _LeachateCarbon, _LeachateLoad
from .native_recipe import _service_trace
from .recipe_context import _RecipeContext

# The admitted main reference uses nominal kg/kmol, including half-integer Cl. B12 alone retains its
# separate CIAAW molecular convention below; its atoms pass into residuals unchanged using that
# separate convention.
_ATOMS = {
    "C": 12.0,
    "H": 1.0,
    "O": 16.0,
    "N": 14.0,
    "S": 32.0,
    "P": 31.0,
    "K": 39.0,
    "Mg": 24.0,
    "Ca": 40.0,
    "Cl": 35.5,
    "Na": 23.0,
}
# Anex/Ogletree's organic-cell formula represents cells and lysed polymer.
# NREL Appendix D supplies CSL protein/lactate; nominal oleic acid represents
# its oil. The medium includes Tween80 C64H124O26 and the specified salts.
_FORMULAS = {
    "glucose": {"C": 6.0, "H": 12.0, "O": 6.0},
    "pdo": {"C": 3.0, "H": 8.0, "O": 2.0},
    "glycerol": {"C": 3.0, "H": 8.0, "O": 3.0},
    "cells": {"C": 1.0, "H": 1.78, "O": 0.60, "N": 0.19},
    "biopolymer": {"C": 1.0, "H": 1.78, "O": 0.60, "N": 0.19},
    "csl_protein": {"C": 1.0, "H": 1.57, "O": 0.31, "N": 0.29, "S": 0.007},
    "csl_lactate": {"C": 3.0, "H": 6.0, "O": 3.0},
    "corn_oil": {"C": 18.0, "H": 34.0, "O": 2.0},
    "tween80": {"C": 64.0, "H": 124.0, "O": 26.0},
    "ammonium_sulfate": {"N": 2.0, "H": 8.0, "S": 1.0, "O": 4.0},
    "potassium_dihydrogen_phosphate": {"K": 1.0, "H": 2.0, "P": 1.0, "O": 4.0},
    "magnesium_sulfate": {"Mg": 1.0, "S": 1.0, "O": 4.0},
    "calcium_chloride": {"Ca": 1.0, "Cl": 2.0},
    "ammonia": {"N": 1.0, "H": 3.0},
    "sodium_hydroxide": {"Na": 1.0, "O": 1.0, "H": 1.0},
    "hydrochloric_acid": {"H": 1.0, "Cl": 1.0},
    "sulfuric_acid": {"H": 2.0, "S": 1.0, "O": 4.0},
    "sodium_sulfate": {"Na": 2.0, "S": 1.0, "O": 4.0},
    "sodium_chloride": {"Na": 1.0, "Cl": 1.0},
    "water": {"H": 2.0, "O": 1.0},
    "carbon_dioxide": {"C": 1.0, "O": 2.0},
    "oxygen": {"O": 2.0},
    "nitrogen": {"N": 2.0},
}
_B12_ATOMS = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "P": 30.974, "Co": 58.933}
# NIST cyanocobalamin formula; the admitted CIAAW rounding above yields
# 1355.388 kg/kmol, distinct from NIST's differently rounded molecular total.
_B12_FORMULA = {"C": 63, "H": 88, "N": 14, "O": 14, "P": 1, "Co": 1}
# Jankovic Case 1a: 2869 kg/h liquid at 99.92 wt% PDO, from 23800 kg/h broth.
_PDO_PURITY = 0.9992
_ACTIVE_OUTPUT = 2869 * _PDO_PURITY
# Existing synthetic amylose/glucose primer, preserved by the PTT carbon owner.
_GLUCOSE_PRIMER = 25 / 738


@dataclass(frozen=True, slots=True)
class _PdoReference:
    """One active PDO kg with actual collected water and supplied-function demand.

    ``supply_amounts`` uses kg, except electricity (kWh), heat/chilled_cooling
    (MJ), valve_air (normal m3), and membrane_module/plant (unit). Spent-resin
    kg demands are negative. ``service_amounts`` uses kg except ambient_cooling
    (MJ); corn_steep_liquor is the admitted dry-starch *supply proxy* quantity.
    The CSL dose remains in ``ledger``. ``load`` contains solute element
    masses and separate Doka organic descriptors; water is not added to those solutes.
    All ledger masses, carbon pools and volumes share this active-kg reference.
    """

    load: _LeachateLoad
    solution_volume_m3: float
    water_kg: float
    product_water_kg: float
    supply_amounts: dict[str, float]
    service_amounts: dict[str, float]
    carbon_dioxide: tuple[_LeachateCarbon, ...]
    ledger: dict[str, dict[str, float]]


def _pdo_elements(compound: str, mass: float) -> dict[str, float]:
    """Calculate kg elements in one declared PDO compound."""
    atoms = _B12_ATOMS if compound == "vitamin_b12" else _ATOMS
    formula = _B12_FORMULA if compound == "vitamin_b12" else _FORMULAS[compound]
    molecular = fsum(atoms[element] * count for element, count in formula.items())
    return {
        element: mass * count * atoms[element] / molecular for element, count in formula.items()
    }


def _pdo_element_sum(compounds: dict[str, float]) -> dict[str, float]:
    """Sum compound element masses, keeping water separately selectable."""
    pools = [_pdo_elements(compound, mass) for compound, mass in compounds.items()]
    return {
        element: fsum(pool.get(element, 0.0) for pool in pools)
        for element in sorted({element for pool in pools for element in pool})
    }


def _pdo_glucose_carbon(
    component: str, carbon_kg: float, origin: _CarbonOrigin
) -> tuple[_LeachateCarbon, ...]:
    """Attribute glucose-derived auxiliary carbon with the existing primer."""
    biogenic = 1.0 if origin == "biogenic" else _GLUCOSE_PRIMER
    result = [_LeachateCarbon(component, False, "biogenic", biogenic * carbon_kg)]
    if origin != "biogenic":
        result.append(_LeachateCarbon(component, False, origin, (1 - biogenic) * carbon_kg))
    return tuple(result)


def _pdo_residual_load(
    compounds: dict[str, float], *, protein_carbon_kg: float, origin: _CarbonOrigin
) -> _LeachateLoad:
    """Receive all real organics/ions with molecular atoms and separate indicators.

    CSL-protein C supplies biomass first; remaining biomass C follows glucose.
    B12 is wholly residual, with 62 biological carbons and one fossil cyanide.
    Tween's sorbitan/fatty-acid 24 C and ethoxylate 40 C remain distinct from
    its manufacturing proxy. These pools are chemical auxiliaries. ThOD ends organic N at NH3.
    """
    elements = _pdo_element_sum(compounds)
    biomass_c = fsum(_pdo_elements(name, compounds[name])["C"] for name in ("cells", "biopolymer"))
    carbon: list[_LeachateCarbon] = [
        _LeachateCarbon("csl_protein_in_biomass", False, "biogenic", protein_carbon_kg),
        *_pdo_glucose_carbon("glucose_in_biomass", biomass_c - protein_carbon_kg, origin),
    ]
    cod = 0.0
    for name, mass in compounds.items():
        if name == "vitamin_b12":
            c = _pdo_elements(name, mass)["C"]
            carbon.extend(
                (
                    _LeachateCarbon(name, False, "biogenic", c * 62 / 63),
                    _LeachateCarbon(name, False, "fossil", c / 63),
                )
            )
            # Doka2021 pp24-25: default COD/TOC for this unmeasured trace cofactor.
            cod += 3.22 * c
            continue
        formula = _FORMULAS[name]
        if "C" not in formula:
            continue
        molecular = fsum(_ATOMS[e] * count for e, count in formula.items())
        # C + H/4 - O/2 - 3N/4 mol O2 per formula, with N ending at NH3.
        cod += (
            mass
            / molecular
            * 32
            * (formula["C"] + formula["H"] / 4 - formula["O"] / 2 - 3 * formula.get("N", 0.0) / 4)
        )
        c = _pdo_elements(name, mass)["C"]
        if name in {"pdo", "glycerol", "glucose"}:
            carbon.extend(_pdo_glucose_carbon(name, c, origin))
        elif name in {"csl_lactate", "corn_oil"}:
            carbon.append(_LeachateCarbon(name, False, "biogenic", c))
        elif name == "tween80":
            carbon.extend(
                (
                    _LeachateCarbon(name, False, "biogenic", c * 24 / 64),
                    _LeachateCarbon(name, False, "fossil", c * 40 / 64),
                )
            )
    total_c = elements["C"]
    # WWcalc B103/B104 and Doka section19: empirical organic descriptors.
    return _LeachateLoad(
        elements_kg=elements,
        carbon=tuple(carbon),
        organic_hydrogen_kg=0.1225 * total_c,
        organic_oxygen_kg=0.64 * total_c,
        toc_kg=total_c,
        doc_kg=0.68 * total_c,
        cod_kg_o2=cod,
        bod5_kg_o2=1.7586552173908316 * total_c,
        chloride_kg=elements["Cl"],
    )


def _pdo_separation(compounds: dict[str, float]) -> dict[str, dict[str, float]]:
    """Partition broth before ion exchange and residual treatment.

    Jankovic SI Table S1 gives DF3/UF flows of 10611/456 kg/h, with PDO fractions
    .0120/.0975 from 23800*.1290 kg/h broth PDO. Apply these entrainment fractions
    to dissolved glucose, glycerol, lactate, salts and nominal unbound B12.
    All cells leave in DF3. The source's nonzero DF3 polymer fraction precedes
    UF removal of the remainder; oil and Tween are assigned to UF removal.
    """
    df_fraction = 10611 * 0.0120 / (23800 * 0.1290)
    uf_fraction = 456 * 0.0975 / (23800 * 0.1290)
    # TableS1 DF3 polymer0.0001, initial broth polymer0.0010. Preserve the
    # explicit nonzero DF fraction; UF receives the complementary polymer.
    polymer_df_fraction = 10611 * 0.0001 / (23800 * 0.0010)
    ions = {
        "sodium_sulfate",
        "potassium_dihydrogen_phosphate",
        "magnesium_sulfate",
        "calcium_chloride",
    }
    phases: dict[str, dict[str, float]] = {"df": {}, "uf": {}, "ix": {}, "bottoms": {}}
    for name, mass in compounds.items():
        if name == "cells":
            phases["df"][name] = mass
            continue
        if name == "biopolymer":
            phases["df"][name] = mass * polymer_df_fraction
            phases["uf"][name] = mass * (1 - polymer_df_fraction)
            continue
        if name in {"corn_oil", "tween80"}:
            phases["uf"][name] = mass
            continue
        phases["df"][name] = mass * df_fraction
        phases["uf"][name] = mass * uf_fraction
        entering_ix = mass * (1 - df_fraction - uf_fraction)
        # Jankovic's IX removes99% ions,1%PDO and10%other organics; all
        # residual routes meet again downstream. The final product is1kg PDO.
        removed = 0.99 if name in ions else 0.01 if name == "pdo" else 0.10
        phases["ix"][name] = entering_ix * removed
        phases["bottoms"][name] = entering_ix * (1 - removed) - (1.0 if name == "pdo" else 0.0)
    return phases


def _pdo_reference(origin: _CarbonOrigin) -> _PdoReference:
    """Calculate the coupled stationary factory per active kg PDO.

    Emptage Example 9/Table 7 supplies the consumed-glucose yield. Jankovic
    Table 1 and Case 1a supply the broth mass fractions, recovery and separation
    reference. NREL's
    returning seed consumes 109 of 2418 kg/h external glucose. Its industrial
    meter and medium are transferred together. Environmental proxy mass and
    upstream water remain outside this mass-balance ledger.
    """
    # Jankovic Table1 fractions and Case1a broth/output, combined with
    # Emptage's 0.34 consumed-glucose yield. Select table glucose0.01 wt%,
    # rather than the conflicting 0.10 wt% prose.
    broth = 23800 / _ACTIVE_OUTPUT
    pdo = 0.1290 * broth
    glycerol = 0.0148 * broth
    residual_glucose = 0.0001 * broth
    cells, biopolymer = 0.0535 * broth, 0.0010 * broth
    main_glucose = pdo / 0.34 + residual_glucose
    glucose = main_glucose / (1 - 109 / 2418)
    scale = glucose / 2418  # Source kg/h multiplied by h per active product kg.
    # NREL Table30 gives external CSL/host rates; Appendix E stream446 gives
    # oil, all in kg/h. Appendix D CSL is 50% water,25% lactate,25% protein;
    # Table20 host mass ratios sum to4.3.
    csl, host, oil = 164 * scale, 67 * scale, 13 * scale
    protein, lactate = 0.25 * csl, 0.25 * csl
    ammonium_sulfate = host * 1.4 / 4.3
    phosphate = host * 2 / 4.3
    mg_hydrate, ca_hydrate = host * 0.3 / 4.3, host * 0.4 / 4.3
    magnesium_sulfate = mg_hydrate * 120 / 246
    calcium_chloride = ca_hydrate * 111 / 147
    salt_preparation_water = mg_hydrate * 126 / 246 + ca_hydrate * 36 / 147
    tween = host * 0.2 / 4.3
    # US7745184B2 Example6/Table4: 25 mg active B12 per kg production glucose,
    # distinct from total G's industrial seed allowance and its supply proxy.
    b12 = 0.000025 * main_glucose

    biomass = _pdo_element_sum({"cells": cells, "biopolymer": biopolymer})
    protein_elements = _pdo_elements("csl_protein", protein)
    sulfate_elements = _pdo_elements("ammonium_sulfate", ammonium_sulfate)
    ammonia = (biomass["N"] - protein_elements["N"] - sulfate_elements["N"]) * 17 / 14
    sulfur = protein_elements["S"] + sulfate_elements["S"]
    # The admitted pH/receiving convention collects assimilated protein/salt S as Na2SO4. This NaOH
    # is already reacted.
    fermentation_naoh = sulfur / 32 * 2 * 40
    fermentation_sulfate = sulfur / 32 * 142
    reaction_inputs = {
        "glucose": glucose - residual_glucose,
        "csl_protein": protein,
        "ammonium_sulfate": ammonium_sulfate,
        "ammonia": ammonia,
        "sodium_hydroxide": fermentation_naoh,
    }
    reaction_outputs = {
        "pdo": pdo,
        "glycerol": glycerol,
        "cells": cells,
        "biopolymer": biopolymer,
        "sodium_sulfate": fermentation_sulfate,
    }
    left, right = _pdo_element_sum(reaction_inputs), _pdo_element_sum(reaction_outputs)
    co2 = (left["C"] - right["C"]) * 44 / 12
    reaction_water = (left["H"] - right["H"]) * 18 / 2
    oxygen = right["O"] + co2 * 32 / 44 + reaction_water * 16 / 18 - left["O"]
    reaction_inputs["oxygen"] = oxygen
    reaction_outputs.update(carbon_dioxide=co2, water=reaction_water)

    broth_solutes = {
        "pdo": pdo,
        "glycerol": glycerol,
        "glucose": residual_glucose,
        "cells": cells,
        "biopolymer": biopolymer,
        "csl_lactate": lactate,
        "tween80": tween,
        "corn_oil": oil,
        "vitamin_b12": b12,
        "potassium_dihydrogen_phosphate": phosphate,
        "magnesium_sulfate": magnesium_sulfate,
        "calcium_chloride": calcium_chloride,
        "sodium_sulfate": fermentation_sulfate,
    }
    # B12 and every dry salt displace water within the fixed broth mass.
    broth_water = broth - fsum(broth_solutes.values())
    product_water = (1 - _PDO_PURITY) / _PDO_PURITY
    csl_water = csl / 2
    air_oxygen, air_nitrogen = 7440 * scale, 24504 * scale
    air_water_in, air_water_out = 638 * scale, (737 + 37) * scale
    medium_water = broth_water - reaction_water + air_water_out - air_water_in - csl_water
    stock_water = 427 * scale
    # Stock and dry-salt water are reservations *inside* prepared medium water.
    free_preparation_water = medium_water - stock_water - salt_preparation_water

    separation = _pdo_separation(broth_solutes)
    # Molel/Jankovic: only actual ions passing DF/UF size two alternating beds of each type. Organic
    # B12 Co/P retains its separate molecular form.
    ix_ions = {
        name: broth_solutes[name] - separation["df"][name] - separation["uf"][name]
        for name in (
            "sodium_sulfate",
            "potassium_dihydrogen_phosphate",
            "magnesium_sulfate",
            "calcium_chloride",
        )
    }
    equivalents = 1000 * (
        2 * ix_ions["sodium_sulfate"] / 142
        + ix_ions["potassium_dihydrogen_phosphate"] / 136
        + 2 * ix_ions["magnesium_sulfate"] / 120
        + 2 * ix_ions["calcium_chloride"] / 111
    )
    cation_bed = 0.99 * equivalents * 24 / 1800
    anion_bed = 0.99 * equivalents * 24 / 1300
    # Two BV/day, 6 wt% HCl and 3.5 wt% NaOH; aqueous density 1000 kg/m3.
    hcl = 2 * cation_bed * 1000 * 0.06 / 24
    ix_naoh = 2 * anion_bed * 1000 * 0.035 / 24
    regeneration_carrier = 2 * (cation_bed * 0.94 + anion_bed * 0.965) * 1000 / 24
    ix_rinse = (3.5 * cation_bed + 3 * anion_bed) * 1000 / 24
    # Twenty minutes per day at 20/6 m/h; 13/10 ft bed heights, .3048 m/ft.
    backwash = (cation_bed / (13 * 0.3048) * 20 + anion_bed / (10 * 0.3048) * 6) / 3 * 1000 / 24
    regeneration_water = regeneration_carrier + ix_rinse + backwash
    # Complete HCl + NaOH -> NaCl + H2O. Only IX's unreacted base participates;
    # buy the positive final trim rather than silently discarding residual acid.
    ix_final_naoh = hcl / 36.5 * 40 - ix_naoh
    ix_salt = hcl / 36.5 * 58.5
    ix_formed_water = hcl / 36.5 * 18

    # Oraby ten-m3 CIP/SIP schedule per total fresh glucose. H2SO4 needs 2 NaOH; tau trims the
    # source 20 kg charge to neutralize all 25 kg acid. The standalone CIP equation leaves molar
    # masses symbolic. Deliberately use this coupled main-factory convention:
    # H2SO4=98,NaOH=40,Na2SO4=142, H2O=18 kg/kmol.
    cycles = 0.00088006234803261255 * glucose
    cip_acid = 25 * cycles
    cip_base = (20 + (2 * 25 / 98 - 20 / 40) * 40) * cycles
    cip_sulfate = cip_acid / 98 * 142
    cip_formed_water = cip_acid / 98 * 36
    cip_cleaning_water, sip_contact_water = 2990 * cycles, 500 * cycles
    # The 50%-NaOH stock contributes cip_base kg of carrier within the 2990 kg water total.
    cip_free_water = cip_cleaning_water - cip_base

    # Fixed separation-water functions: DF water once, two biological aqueous
    # retentates, and process dilution diverted from already counted IX rinse.
    diafiltration_water = 15508 / _ACTIVE_OUTPUT
    filter_water = (10611 * 0.8663 + 456 * 0.8397) / _ACTIVE_OUTPUT
    diverted_rinse = 0.02 * 28241 / _ACTIVE_OUTPUT
    recycle = broth_water + diafiltration_water + diverted_rinse - filter_water - product_water
    # In this fixed reference, recycle supplies all DF and then part of medium
    # preparation. There is no surplus purge or alternative operating regime.
    recycled_df = diafiltration_water
    recycled_medium = recycle - diafiltration_water
    fresh_process_water = medium_water - recycled_medium
    terminal_water = (
        filter_water
        + regeneration_water
        - diverted_rinse
        + cip_cleaning_water
        + sip_contact_water
        + ix_formed_water
        + cip_formed_water
    )
    final_solutes = dict(broth_solutes)
    final_solutes["pdo"] -= 1.0
    final_solutes["sodium_sulfate"] += cip_sulfate
    final_solutes["sodium_chloride"] = ix_salt
    load = _pdo_residual_load(final_solutes, protein_carbon_kg=protein_elements["C"], origin=origin)

    physical_inputs = {
        "glucose": glucose,
        "csl_protein": protein,
        "csl_lactate": lactate,
        "ammonium_sulfate": ammonium_sulfate,
        "potassium_dihydrogen_phosphate": phosphate,
        "magnesium_sulfate": magnesium_sulfate,
        "calcium_chloride": calcium_chloride,
        "tween80": tween,
        "corn_oil": oil,
        "vitamin_b12": b12,
        "ammonia": ammonia,
        "sodium_hydroxide": fermentation_naoh + ix_naoh + ix_final_naoh + cip_base,
        "hydrochloric_acid": hcl,
        "sulfuric_acid": cip_acid,
        "water": (
            fresh_process_water
            + regeneration_water
            + cip_cleaning_water
            + sip_contact_water
            + csl_water
            + air_water_in
        ),
        "oxygen": air_oxygen,
        "nitrogen": air_nitrogen,
    }
    physical_outputs = dict(final_solutes)
    physical_outputs["pdo"] += 1.0
    physical_outputs.update(
        water=terminal_water + product_water + air_water_out,
        carbon_dioxide=co2,
        oxygen=air_oxygen - oxygen,
        nitrogen=air_nitrogen,
    )
    water_ledger = {
        "broth_water": broth_water,
        "prepared_medium": medium_water,
        "glucose_stock_reservation": stock_water,
        "salt_preparation_reservation": salt_preparation_water,
        "free_preparation": free_preparation_water,
        "csl_water": csl_water,
        "air_in": air_water_in,
        "air_out": air_water_out,
        "reaction_formed": reaction_water,
        "diafiltration": diafiltration_water,
        "filter_retentate": filter_water,
        "ix_carrier": regeneration_carrier,
        "ix_rinse": ix_rinse,
        "ix_backwash": backwash,
        "ix_rinse_diverted": diverted_rinse,
        "recycle": recycle,
        "recycled_df": recycled_df,
        "recycled_medium": recycled_medium,
        "fresh_process": fresh_process_water,
        "cip_cleaning": cip_cleaning_water,
        "cip_stock_carrier_reservation": cip_base,
        "cip_free_cleaning": cip_free_water,
        "sip_contact": sip_contact_water,
        "ix_neutralization_formed": ix_formed_water,
        "cip_neutralization_formed": cip_formed_water,
        "terminal": terminal_water,
        "product": product_water,
    }
    # Molel's kill step is for the cell-bearing MF/DF material. Heat actual DF3 water and solids.
    retentate_mass = 10611 * 0.8663 / _ACTIVE_OUTPUT + fsum(separation["df"].values())
    charged_medium = fsum(
        (
            glucose,
            csl,
            ammonium_sulfate,
            phosphate,
            magnesium_sulfate,
            calcium_chloride,
            tween,
            oil,
            b12,
            ammonia,
            fermentation_naoh,
            medium_water,
        )
    )
    # Indirect aqueous reference, Cp=4.18 kJ/(kg K). Actual charged medium
    # goes 21->121->35 C; biological retentate enters deactivation at 37 C.
    heat = {
        "feed_sterilization": charged_medium * 4.18e-3 * (121 - 21),
        "feed_cooling": charged_medium * 4.18e-3 * (121 - 35),
        "retentate_deactivation": retentate_mass * 4.18e-3 * (121 - 37),
        "cip_sip": 1375 * cycles,
        # Jankovic Table4 reports kWh thermal per kg solution; convert to MJ
        # and active output. The full recovery meter includes MF/DF/UF/IX.
        "recovery": 0.403 * 3.6 / _PDO_PURITY,
        # NREL total includes its chiller and compressor cooling; Gcal->MJ.
        "fermentation_cooling": (8.104 + 0.468 + 1.172) * 4184 * scale,
        # SI S8 annual cooling/chilled costs divided by $/MWh and 8000 h/y,
        # then 3600 MJ/MWh and the active kg/h reference. Distinct from NREL.
        "recovery_cooling": 21000 / 1.27 / 8000 * 3600 / _ACTIVE_OUTPUT,
        "recovery_chilled": 25000 / 15.95 / 8000 * 3600 / _ACTIVE_OUTPUT,
    }
    # Two alternating beds, five years at 8000 h/y. Native shipping masses
    # are 800/640 kg/m3; spent resin uses its own complete native treatment.
    cation_resin = 2 * cation_bed * 800 / (5 * 8000)
    anion_resin = 2 * anion_bed * 640 / (5 * 8000)
    # Jankovic MF, three DF units and UF membrane areas; 1.5-year life and
    # 50 m2 per native supplied module. No inferred fabrication or disposal.
    membranes = (13189 / 119 + 3 * 15508 / 28 + 28241 / 28) / (1.5 * 8000 * 50 * _ACTIVE_OUTPUT)
    supply_amounts = {
        "ammonium_sulfate": ammonium_sulfate,
        "magnesium_sulfate": magnesium_sulfate,
        "calcium_chloride": calcium_chloride,
        "corn_oil": oil,
        "ammonia": ammonia,
        "sodium_hydroxide": physical_inputs["sodium_hydroxide"],
        "hydrochloric_acid": hcl,
        "sulfuric_acid": cip_acid,
        "deionised_water": fresh_process_water
        + regeneration_water
        + cip_cleaning_water
        + sip_contact_water,
        # Full 5340-kW NREL operating reference, recovery and separate CIP
        # circulation17.4 MJ/cycle. No central NREL CIP or extra compressor.
        "electricity": 5340 * scale + 0.480 / _PDO_PURITY + 17.4 / 3.6 * cycles,
        "heat": fsum(
            heat[name]
            for name in ("feed_sterilization", "retentate_deactivation", "cip_sip", "recovery")
        ),
        "chilled_cooling": heat["recovery_chilled"],
        "valve_air": 1.82 * cycles,
        "membrane_module": membranes,
        "cation_resin": cation_resin,
        "anion_resin": anion_resin,
        "spent_cation_resin": -cation_resin,
        "spent_anion_resin": -anion_resin,
        "plant": 4e-10,  # Complete native RER chemical-factory allocation per active kg.
    }
    service_amounts = {
        "glucose": glucose,
        # The selected wet-milling inventory supplies the manufacturing proxy; CSL composition
        # follows the specified medium.
        "corn_steep_liquor": csl * 0.5 / 0.86,
        "potassium_dihydrogen_phosphate": phosphate,
        "tween80": tween,
        "vitamin_b12": b12,
        "ambient_cooling": fsum(
            heat[name]
            for name in (
                "feed_cooling",
                "fermentation_cooling",
                "recovery_cooling",
            )
        ),
    }
    ledger = {
        "physical_inputs": physical_inputs,
        "physical_outputs": physical_outputs,
        "elements_in": _pdo_element_sum(physical_inputs),
        "elements_out": _pdo_element_sum(physical_outputs),
        "reaction_inputs": reaction_inputs,
        "reaction_outputs": reaction_outputs,
        "broth_solutes": broth_solutes,
        "residual_solutes": final_solutes,
        "water": water_ledger,
        "heat_mj": heat,
        "df_solutes": separation["df"],
        "uf_solutes": separation["uf"],
        "ix_solutes": separation["ix"],
        "bottoms_solutes": separation["bottoms"],
        "ix_feed_ions": ix_ions,
        "operating": {
            "main_glucose_kg": main_glucose,
            "total_glucose_kg": glucose,
            "seed_glucose_consumed_kg": 109 * scale,
            "csl_as_received_kg": csl,
            "magnesium_sulfate_hydrate_equivalent_kg": mg_hydrate,
            "calcium_chloride_hydrate_equivalent_kg": ca_hydrate,
            "fermentation_naoh_kg": fermentation_naoh,
            "ix_naoh_kg": ix_naoh,
            "ix_final_naoh_kg": ix_final_naoh,
            "cip_naoh_kg": cip_base,
            "cip_cycles": cycles,
            "ix_equivalents": equivalents,
            "cation_bed_m3": cation_bed,
            "anion_bed_m3": anion_bed,
            "charged_medium_kg": charged_medium,
            "retentate_kg": retentate_mass,
            "oxygen_consumed_kg": oxygen,
        },
    }
    return _PdoReference(
        load=load,
        solution_volume_m3=(terminal_water + fsum(final_solutes.values())) / 1000,
        water_kg=terminal_water,
        product_water_kg=product_water,
        supply_amounts=supply_amounts,
        service_amounts=service_amounts,
        carbon_dioxide=_pdo_glucose_carbon("fermentation", co2 * 12 / 44, origin),
        ledger=ledger,
    )


def _pdo_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Bind a kg of 99.92% PDO solution to complete dated supplies and receiving.

    The pure reference is per active kg; every environmental and term
    is scaled by .9992 once. The delivered-liquid market and active-PDO root
    separately own actual cargo and the inverse purity factor. Native supplier
    inventories and their own production/uncertainty remain inside those links.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    reference = _pdo_reference(cast(_CarbonOrigin, construction["origin"]))
    key = database_name, provider_service_code(service_id)
    # Nominal C3H8O2 contains 36 kg C per 76 kg active dry PDO. Content is per dry kg; allocation is
    # per kg supplied PDO solution.
    dry_carbon_fraction = 36 / 76
    origin = cast(_CarbonOrigin, construction["origin"])
    fossil_share = 1 - _GLUCOSE_PRIMER if origin == "fossil" else 0.0
    properties = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": _PDO_PURITY, "unit": "kg"},
        "water in wet mass": {"amount": 1 - _PDO_PURITY, "unit": "kg"},
        "carbon allocation": {"amount": _PDO_PURITY * dry_carbon_fraction, "unit": "kg"},
        "carbon content, fossil": {
            "amount": dry_carbon_fraction * fossil_share,
            "unit": "dimensionless",
        },
        "carbon content, non-fossil": {
            "amount": dry_carbon_fraction * (1 - fossil_share),
            "unit": "dimensionless",
        },
    }
    rows: list[dict[str, object]] = [
        {
            "input": key,
            "type": "production",
            "amount": 1.0,
            "unit": "kilogram",
            "properties": properties,
        }
    ]
    selectors = cast(dict[str, dict[str, object]], construction["supplies"])
    for label, amount in reference.supply_amounts.items():
        node, trace = context.activity(
            selectors[label],
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id=label,
            semantic_class="source_backed_portfolio",
        )
        rows.append(
            {
                "input": cast(tuple[str, str], node.key),
                "type": "technosphere",
                "amount": _PDO_PURITY * amount,
                "unit": str(node["unit"]),
            }
        )
        trace["project_terms_json"] = json.dumps(
            {
                "amount_per_active_pdo_kg": amount,
                "unit": node["unit"],
                "active_pdo_per_solution_kg": _PDO_PURITY,
            },
            sort_keys=True,
        )
    dependencies = cast(dict[str, dict[str, object]], construction["services"])
    for label, amount in reference.service_amounts.items():
        target = str(dependencies[label]["service_id"])
        rows.append(
            {
                "input": (database_name, provider_service_code(target)),
                "type": "technosphere",
                "amount": _PDO_PURITY * amount,
                "unit": "megajoule" if label == "ambient_cooling" else "kilogram",
            }
        )

    # The dated frontier owns the complete actual-load CH chain. Its negative
    # collection demand is solution volume per delivered kg; native
    # collection production and the treated fraction are applied inside it once.
    collection_id = str(cast(dict[str, object], construction["water_collection"])["service_id"])
    rows.append(
        {
            "input": (database_name, provider_service_code(collection_id)),
            "type": "technosphere",
            "amount": -_PDO_PURITY * reference.solution_volume_m3,
            "unit": "cubic meter",
        }
    )
    emissions: dict[tuple[str, str], float] = {}
    for pool in reference.carbon_dioxide:
        name = "Carbon dioxide, fossil" if pool.origin == "fossil" else "Carbon dioxide, non-fossil"
        identity = name, "kilogram"
        emissions[identity] = emissions.get(identity, 0.0) + pool.carbon_kg * 44 / 12
    # Humid air enters and leaves the site. Only the net site evaporation
    # is an elementary water release; 1000 kg/m3 is the freshwater convention.
    water = reference.ledger["water"]
    emissions[("Water", "cubic meter")] = (water["air_out"] - water["air_in"]) / 1000
    for (name, unit), amount in emissions.items():
        flow = context.biosphere.flow({"name": name, "unit": unit, "categories": ["air"]})
        rows.append(
            {
                "input": cast(tuple[str, str], flow.key),
                "type": "biosphere",
                "amount": _PDO_PURITY * amount,
                "unit": unit,
            }
        )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
        "pdo_reference_schema": "pytoce_pdo_manufacturing_reference_schema_v1",
        "pdo_reference_per_active_kg": asdict(reference),
        "comment": (
            "One kg of 99.92% PDO solution from the glucose and growth-medium engineering "
            "reference. Reaction, separation, water and acid-base quantities share one "
            "active-kg basis. Dated material, utility, membrane, resin and factory "
            "inventories provide the supplies. Collected compounds enter Swiss municipal "
            "co-treatment, retaining their carbon origins and the municipal organic "
            "indicators. Product water follows liquid delivery and the PTT operating input."
        ),
    }
