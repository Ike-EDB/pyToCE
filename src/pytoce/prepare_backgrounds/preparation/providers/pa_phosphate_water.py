"""Receive the actual mixed wash from PA triethanolamine monooleylphosphate.

The manufacturing calculation supplies four collected compound masses. Doka's
complete Swiss municipal co-treatment reference acts on their actual elements,
including suspended or emulsified monoester, using its hydraulic and organic
allocation indicators.
"""

from math import fsum

from ..chemical_organic_wastewater import chemical_organic_wastewater_datasets
from ..hazardous_incineration import reagent_element_masses
from .leachate_load import _LeachateCarbon, _LeachateLoad
from .recipe_context import _RecipeContext

# CIAAW masses match the actual PA ingredient reference, g/mol. Total P includes
# both the collected organic monoester and unreacted inorganic phosphoric acid.
_ATOMS = {"C": 12.011, "H": 1.008, "O": 15.999, "P": 30.973761998, "Na": 22.98976928, "Cl": 35.45}
_FORMULAS = {
    "monooleylphosphate": {"C": 18, "H": 37, "O": 4, "P": 1},
    "phosphoric_acid": {"H": 3, "P": 1, "O": 4},
    "sodium_chloride": {"Na": 1, "Cl": 1},
}


def pa_phosphate_water_datasets(
    context: _RecipeContext,
    *,
    water_kg: float,
    sodium_chloride_kg: float,
    monooleylphosphate_kg: float,
    phosphoric_acid_kg: float,
) -> dict[tuple[str, str], dict[str, object]]:
    """Build ten Swiss treatment inventories for the mixed wash.

    Inputs are kg per parent reference, before the collection fraction.
    The parent demands minus total wet kg/1000 m3 from
    ``pa_phosphate_water_collection_ch``, using Doka's assumed one litre
    per wet kilogram. Carrier water remains separate from solutes.

    TOC is biogenic monoester C. DOC uses Doka's fixed 68% Swiss residential
    reference; BOD5 uses the Wastewater2021 BOD/TOC ratio. Theoretical COD
    follows C18H37O4P + 26.5 O2 -> 18 CO2 + 17 H2O + H3PO4. These overlapping
    indicators describe the inlet; the treatment equations determine removal.
    Doka organic H/O descriptors are separate from molecular H/O in the
    incoming elemental ledger.
    """
    compounds = {
        "monooleylphosphate": monooleylphosphate_kg,
        "phosphoric_acid": phosphoric_acid_kg,
        "sodium_chloride": sodium_chloride_kg,
    }
    pools = reagent_element_masses(compounds, _FORMULAS, _ATOMS)
    elements = {
        element: fsum(pool.get(element, 0.0) for pool in pools.values()) for element in _ATOMS
    }
    carbon = elements["C"]
    monoester_kmol = monooleylphosphate_kg / fsum(
        _ATOMS[element] * count for element, count in _FORMULAS["monooleylphosphate"].items()
    )
    load = _LeachateLoad(
        elements_kg=elements,
        carbon=(_LeachateCarbon("pa_monooleylphosphate_workup", False, "biogenic", carbon),),
        # Doka Wastewater2021 WWcalc B103/B104: selected organic-composition
        # descriptors, separate from the actual molecular H/O in elements.
        organic_hydrogen_kg=0.1225 * carbon,
        organic_oxygen_kg=0.64 * carbon,
        toc_kg=carbon,
        doc_kg=0.68 * carbon,
        cod_kg_o2=monoester_kmol * 26.5 * 2 * _ATOMS["O"],
        bod5_kg_o2=1.7586552173908316 * carbon,
        chloride_kg=elements["Cl"],
    )
    return chemical_organic_wastewater_datasets(
        context,
        service_id="pa_phosphate_water_collection_ch",
        load=load,
        solution_volume_m3=fsum((water_kg, *compounds.values())) / 1000,
        water_kg=water_kg,
        complete_carbon_oxidation=False,
        phosphorus_equilibrium=False,
    )
