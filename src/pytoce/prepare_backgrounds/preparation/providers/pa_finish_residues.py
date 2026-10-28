"""Bind PA fiber-finish manufacturing residues to their complete thermal treatment.

Castor refining, hydrogenation and ester/amide manufacture retain their own
quantity owners. This module converts their named residues to the
shared hazardous-incineration reference.
Mineral residues use the selected mixed-waste proxy.
"""

from math import fsum
from typing import cast

from ..chemical_residue_incineration import chemical_residue_datasets
from ..chemical_residue_load import ChemicalResidueLoad
from ..hazardous_incineration import reagent_element_masses
from .catalogue import provider_service_code
from .native_recipe import _service_trace
from .pa_castor_hco import _hco_reference
from .pa_ester_ingredients import _pa_ester_reference
from .pa_finish_concentrate import _pa_blending_residue_load
from .pa_finish_material import _pa_finish_load, _pa_finish_reference, _PAFinishReference
from .pa_spinning_support import _pa_spinning_support_reference
from .recipe_context import _RecipeContext

# CIAAW weights used by the admitted PA precursor/catalyst calculations,
# g/mol. Keep them separate from the shared receiving model's nominal factors.
_ATOMS = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "Na": 22.98976928,
    "S": 32.06,
    "Ni": 58.6934,
    "Si": 28.085,
    "P": 30.973761998,
    "K": 39.0983,
    "Li": 6.94,
}
# Nominal formulas define ester/amide and phosphate-solvent residue inputs.
_RESIDUE_FORMULAS = {
    "C3H8O3": {"C": 3, "H": 8, "O": 3},
    "C6H14O5": {"C": 6, "H": 14, "O": 5},
    "C9H20O7": {"C": 9, "H": 20, "O": 7},
    "C18H34O2": {"C": 18, "H": 34, "O": 2},
    "C6H14": {"C": 6, "H": 14},
    "C6H12O": {"C": 6, "H": 12, "O": 1},
    "K2SO4": {"K": 2, "S": 1, "O": 4},
    "LiH2PO4": {"Li": 1, "H": 2, "P": 1, "O": 4},
    "H3PO4": {"H": 3, "P": 1, "O": 4},
}
# q kJ/mol input and newly formed mol H2O/mol. NBS TN270-8 (1981) Table 98 and TN270-3 (1968) Table
# 19, with NIST H2O(g)/P4O10. Both already oxidized phosphates release water by dehydration.
# Products are water VAPOR. Dry H3PO4 uses a crystalline reference at 25 C.
_INORGANIC_REACTIONS = {
    "LiH2PO4": (-76.5764, 1.0),  # ->LiPO3(cr)+H2O(g).
    "H3PO4": (-163.8248, 1.5),  # ->1/4 P4O10(cr)+1.5 H2O(g).
}


def _pa_molecular_residue_load(
    compounds: dict[str, float],
    *,
    aggregate_organic: dict[str, float],
    biogenic_carbon: float,
    mineral: float,
) -> ChemicalResidueLoad:
    """Combine actual PA ester and phosphorus compounds with their heat and fate references.

    Molecular amounts use kg and g/mol, so kg/(g/mol) numerically gives kmol.
    kmol times kJ/mol gives MJ; kmol times g/mol gives kg. Organic Michel (via Muñoz et al. (2007),
    Eq. 9) heat,
    inorganic Hess heat, intact salts and unknown mineral remain separate.
    """
    pools = reagent_element_masses(
        compounds, {name: _RESIDUE_FORMULAS[name] for name in compounds}, _ATOMS
    )
    elements = {element: 0.0 for element in ("C", "H", "O", "N", "S", "Na", "P", "Li")}
    organic = {element: aggregate_organic.get(element, 0.0) for element in ("C", "H", "O")}
    salts: dict[str, dict[str, float]] = {}
    water_mass = 2 * _ATOMS["H"] + _ATOMS["O"]
    energy, formed = 0.0, 0.0
    for name, atoms in pools.items():
        if name == "K2SO4":
            salts["Potassium sulfate"] = atoms
            continue
        for element, amount in atoms.items():
            elements[element] += amount
        if name in _INORGANIC_REACTIONS:
            molar_mass = fsum(_ATOMS[e] * count for e, count in _RESIDUE_FORMULAS[name].items())
            kmol = compounds[name] / molar_mass
            heat, reaction_water = _INORGANIC_REACTIONS[name]
            energy += kmol * heat
            formed += kmol * reaction_water * water_mass
        else:
            for element, amount in atoms.items():
                organic[element] += amount
    for element, amount in aggregate_organic.items():
        elements[element] += amount
    # The nominal heat reference uses 9 kg water/kg organic H and 2.4417
    # MJ/kg water (Harvey 1998, NISTIR 5078 Table 1, 2441.7 kJ/kg at 25 C).
    # Actual molecular water below uses the same CIAAW masses as the parent.
    energy += fsum(
        organic[e] * factor for e, factor in {"C": 34.016, "H": 124.265, "O": -9.8324}.items()
    )
    energy -= 2.4417 * (9 * organic["H"])
    formed += organic["H"] * water_mass / (2 * _ATOMS["H"])
    return ChemicalResidueLoad(
        wet_mass_kg=fsum(compounds.values()) + fsum(aggregate_organic.values()) + mineral,
        carrier_water_kg=0.0,
        active_elements_kg=elements,
        net_energy_mj=energy,
        nonfuel_salt_elements_kg=salts,
        waste_formed_water_kg=formed,
        biogenic_carbon_kg=biogenic_carbon,
        unknown_mineral_kg=mineral,
    )


def _pa_maintenance_residue_load(
    finish: _PAFinishReference,
    *,
    principal_by_origin_kg: tuple[float, float, float],
    embedded_dose_kg: float,
) -> ChemicalResidueLoad:
    """Receive destroyed PA equipment deposits with their inherited finish.

    The material/custody owners supply actual destroyed principal kg in fossil,
    biogenic, atmospheric-DAC order and its destroyed embedded dose. Use the maintenance residue
    quantities supplied by the material calculation.
    Principal PA6/PA66 use nominal C6H11NO/113; fixed auxiliary compounds retain
    their own CIAAW atoms and fossil/biological moieties. The load contains the dry deposit
    quantities.
    """
    inherited = _pa_finish_load(
        {name: embedded_dose_kg * mass for name, mass in finish.retained.compounds_kg.items()}
    )
    principal = fsum(principal_by_origin_kg)
    principal_carbon = cast(
        tuple[float, float, float], tuple(mass * 72 / 113 for mass in principal_by_origin_kg)
    )
    elements = dict(inherited.elements_kg)
    for element, numerator in (("C", 72), ("H", 11), ("N", 14), ("O", 16)):
        elements[element] += principal * numerator / 113
    # Sum independently tracked C pools on their actual mass basis. This keeps
    # the numerator and denominator identical for purely non-fossil principal,
    # rather than obtaining a tiny false fossil remainder from regrouped sums.
    elements["C"] = (
        fsum(principal_carbon) + inherited.fossil_carbon_kg + inherited.biogenic_carbon_kg
    )
    # Michel (via Doka) estimates HHV in MJ/kg organic element. Phosphorus and
    # sodium have no added fuel coefficient.
    hhv = fsum(
        elements[element] * factor
        for element, factor in {
            "C": 34.016,
            "H": 124.265,
            "O": -9.8324,
            "N": 6.276,
            "S": 19.079,
        }.items()
    )
    # The principal retains nominal H1/O16 water; auxiliary H uses CIAAW
    # H1.008/O15.999. Harvey (1998), NISTIR 5078 Table 1, gives the 25 C
    # pure-water latent heat: 2441.7 kJ/kg / 1000 = 2.4417 MJ/kg.
    formed = principal * 11 / 113 * 9 + inherited.elements_kg["H"] * 18.015 / (2 * 1.008)
    return ChemicalResidueLoad(
        wet_mass_kg=principal + fsum(inherited.compounds_kg.values()),
        carrier_water_kg=0.0,
        active_elements_kg=elements,
        net_energy_mj=hhv - 2.4417 * formed,
        nonfuel_salt_elements_kg={},
        waste_formed_water_kg=formed,
        biogenic_carbon_kg=inherited.biogenic_carbon_kg + principal_carbon[1],
        principal_carbon_kg=principal_carbon,
    )


def _pa_finish_residue_load(
    construction: dict[str, object],
    *,
    residue_grade: str,
) -> ChemicalResidueLoad:
    """Derive actual wet residue and heat on the parent service's reference basis.

    Michel's (via Muñoz et al. (2007), Eq. 9) coefficients give MJ/kg of the known organic elements;
    the
    admitted metallic-Ni oxidation reference adds its separate heat. The
    2.4417 MJ/kg latent-water deduction (Harvey 1998, NISTIR 5078 Table 1,
    25 degrees Celsius) applies only to incoming water and hydrogen in the
    known organic material. Minerals and intact salts
    are nonfuel. No supporting-fuel or upstream carbon joins this residue.
    """
    grade = residue_grade
    if grade == "finish_blending":
        return _pa_blending_residue_load(_pa_finish_reference(str(construction["pa_grade"])))
    if grade == "spinning_htm":
        # Retired PA heating fluid leaves the recovery circuit; condensed fluid returns to use.
        return _pa_spinning_support_reference(construction).residue
    if grade == "phosphate_solvents":
        # PA monooleyl-phosphate workup separately condenses the one-pass
        # hexane and MIBK purchases. Their actual catalog quantities are
        # fossil solvent waste; the mixed aqueous stream has a separate owner.
        receiving = cast(dict[str, dict[str, float]], construction["receiving_loads"])
        return _pa_molecular_residue_load(
            receiving["solvents"], aggregate_organic={}, biogenic_carbon=0.0, mineral=0.0
        )
    if grade in {
        "diglycerol_bottoms",
        "dioleate_settling",
        "dimer_monomer",
        "dimer_mineral",
    }:
        parent_grade, residue_key = {
            "diglycerol_bottoms": ("diglycerol_dioleate", "diglycerol_residue"),
            "dioleate_settling": ("diglycerol_dioleate", "settling_residue"),
            "dimer_monomer": ("hmda_dimeramide", "monomer_residue"),
            "dimer_mineral": ("hmda_dimeramide", "mineral_residue"),
        }[grade]
        residue = _pa_ester_reference(parent_grade).residues[residue_key]
        return _pa_molecular_residue_load(
            residue.known_compounds_kg,
            aggregate_organic=residue.aggregate_organic_elements_kg,
            biogenic_carbon=residue.biogenic_kg_c,
            mineral=residue.prepared_mineral_kg,
        )
    hco = _hco_reference()
    components = hco.cake_components_kg
    nickel, silica = components["Ni"], components["SiO2"]
    # Nominal hydrogenated castor oil C57H110O9 defines the retained HCO
    # component of the cake load.
    organic_components = reagent_element_masses(
        {"oil": components["oil"], "citric_acid": components["citric_acid"]},
        {
            "oil": {"C": 57, "H": 110, "O": 9},
            "citric_acid": {"C": 6, "H": 8, "O": 7},
        },
        _ATOMS,
    )
    # Collected citric-acid-equivalent matter has its own nominal atoms and
    # biogenic carbon, separate from retained HCO and catalyst preparation.
    organic = {
        element: fsum(atoms[element] for atoms in organic_components.values())
        for element in ("C", "H", "O")
    }
    support = reagent_element_masses(
        {"silica": silica},
        {"silica": {"Si": 1, "O": 2}},
        _ATOMS,
    )["silica"]
    elements = {
        e: organic.get(e, 0.0) + support.get(e, 0.0) for e in ("C", "H", "O", "N", "S", "Na", "Si")
    }
    elements["Ni"] = nickel
    biogenic, mineral = organic["C"], components["unknown_mineral"]
    energy = fsum(
        organic[e] * factor for e, factor in {"C": 34.016, "H": 124.265, "O": -9.8324}.items()
    )
    # OECD NEA NiO formation is 239.7 kJ/mol at 58.6934 g Ni/mol;
    # the numeric ratio kJ/g gives MJ/kg for the supplied metal.
    energy += nickel * 239.7 / _ATOMS["Ni"]
    energy -= 2.4417 * (9 * organic["H"])
    # Actual HCO/citric-acid hydrogen forms CIAAW H2O. Michel's (via Muñoz et al. (2007), Eq. 9) NCV
    # above
    # independently retains its published 9 kg water/kg hydrogen coefficient.
    formed = organic["H"] * (2 * _ATOMS["H"] + _ATOMS["O"]) / (2 * _ATOMS["H"])
    return ChemicalResidueLoad(
        hco.cake_kg,
        0.0,
        elements,
        energy,
        {},
        biogenic,
        mineral,
        waste_formed_water_kg=formed,
    )


def _pa_finish_residue_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Compile the actual negative-kg PA residue or its single complete ash child.

    The formulation reference identifies the material-balance calculation. Shared source-native
    functions retain full fuel,
    reagents, fixed operations, collection, cement and hydrated disposal.
    """
    _service_trace(context, service)
    construction = cast(dict[str, object], service["construction"])
    kind = str(construction["kind"])
    thermal = (
        service
        if kind == "pa_finish_residue_incineration"
        else services[str(construction["incineration_service_id"])]
    )
    thermal_construction = cast(dict[str, object], thermal["construction"])
    parent = services[str(thermal_construction["formulation_service_id"])]
    parent_construction = cast(dict[str, object], parent["construction"])
    load = _pa_finish_residue_load(
        parent_construction,
        residue_grade=str(thermal_construction["residue_grade"]),
    )
    hwi_key = database_name, provider_service_code(str(thermal["provider_service_id"]))
    ash_key = (
        database_name,
        provider_service_code(
            str(cast(dict[str, object], thermal_construction["ash"])["service_id"])
        ),
    )
    datasets = chemical_residue_datasets(
        context,
        load,
        incineration_key=hwi_key,
        ash_key=ash_key,
        residue_name=str(thermal["reference_product"]),
    )
    return datasets[hwi_key if kind == "pa_finish_residue_incineration" else ash_key]
