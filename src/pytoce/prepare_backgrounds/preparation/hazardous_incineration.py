"""Calculate reagent allocations for constructed hazardous waste incineration (HWI).

Treatment calculations supply their admitted element pools, phase fractions
and atomic weight conventions. Inventory selection and the carbon, ash, heat, and water fates belong
to the receiving calculations.
"""

from collections.abc import Mapping
from math import fsum

# Doka (2003), report 13 part II, section 4.1.3 adopts the reagent-consumption
# functions in Tables 3.2/3.3 for hazardous waste incineration. Acid-capture
# fractions and weights in kg reagent per kg captured element
# allocate original NaOH/CaO doses. The acid-capture fractions differ from
# HWI non-air phase fractions, particularly for N.
ACID_CAPTURE_FRACTIONS = {"S": 0.9994, "N": 0.0793, "P": 0.9993, "Cl": 0.9997, "F": 0.9995}
NAOH_ACID_WEIGHTS = {"S": 2.51, "N": 2.69, "P": 2.43, "Cl": 1.06, "F": 1.98}
LIME_ACID_WEIGHTS = {"N": 0.48, "P": 0.43, "Cl": 0.19, "F": 0.35}


def scrubber_capture_mass(
    element_mass_kg: Mapping[str, float], nonair_fractions: Mapping[str, float]
) -> float:
    """Sum kg of selected elements allocated to the HWI scrubber.

    Each treatment calculation selects the reported or actual waste elements
    and their non-air fractions. The returned allocation mass
    sizes HCl and flocculant demand.
    """
    return fsum(mass * nonair_fractions[element] for element, mass in element_mass_kg.items())


def acid_neutralization_mass(
    element_mass_kg: Mapping[str, float], reagent_weights: Mapping[str, float]
) -> float:
    """Return kg reagent equivalent for the selected acid elements.

    Sum element kg times acid-capture fraction times the source's NaOH or CaO
    consumption weight. The treatment calculation scales its original reagent dose
    by the actual/source ratio, preserving the original source normalization.
    Iteration follows the supplied weights, including a treatment's deliberate
    restriction to its actual acid-forming elements.
    """
    return fsum(
        element_mass_kg[element] * ACID_CAPTURE_FRACTIONS[element] * weight
        for element, weight in reagent_weights.items()
    )


def reagent_element_masses(
    reagent_kg: Mapping[str, float],
    molecular_formulas: Mapping[str, Mapping[str, float]],
    atomic_weights: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Convert the treatment's active reagents into kg of each constituent element.

    Formula entries are atom counts and atomic weights are g/mol. Each caller
    supplies its admitted atomic-weight convention and reagent set.
    For each reagent, element kg = reagent kg * atomic weight * count /
    molar mass. Each treatment calculation supplies and assigns solution water.
    """
    result: dict[str, dict[str, float]] = {}
    for reagent, formula in molecular_formulas.items():
        molar_mass = fsum(atomic_weights[element] * count for element, count in formula.items())
        result[reagent] = {
            element: reagent_kg[reagent] * atomic_weights[element] * count / molar_mass
            for element, count in formula.items()
        }
    return result
