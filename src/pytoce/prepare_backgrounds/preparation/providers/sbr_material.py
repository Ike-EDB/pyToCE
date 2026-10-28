"""Actual compound22 material before separate treatment phase calculations.

Compound22 contains auxiliaries retained after its manufacturing withdrawals.
Its fixed composition follows either retained material or homogeneous trim;
landfill and burning transfers belong to their respective receiving owners.
"""

from dataclasses import dataclass
from math import fsum

from .leachate_load import _CarbonOrigin

# Nominal compound22 auxiliary atoms in the finishing scrap associated with
# one retained principal kg, after mixer/organic/extrusion withdrawals and
# Carli's 5% final trim. Retained auxiliaries are 19 times these same pools.
_AUXILIARY_FOSSIL_CARBON = 0.03633106199969588
_AUXILIARY_BIOGENIC_CARBON = 0.0007595306139770401
_AUXILIARY_SCRAP_KG = {
    "C": _AUXILIARY_FOSSIL_CARBON + _AUXILIARY_BIOGENIC_CARBON,
    "H": 0.0007065002213789833,
    "O": 0.0011458766746025022,
    "N": 0.00023885398274193468,
    "S": 0.0011679102550677155,
    "Zn": 0.004223906787873709,
}
# Actual retained principal C after selective styrene, butadiene and particle
# exits, in kg C/kg principal.
_PRINCIPAL_CARBON_FRACTION = 0.8969229843817300

# Finishing-scrap heat, MJ per kg retained principal. Michel (via Muñoz et al. (2007), Eq. 9)
# coefficients
# 34.016C + 124.265H - 9.8324O + 6.276N + 19.079S use kg organic elements
# after withdrawals. The 5% trim takes 1/20 of pre-trim auxiliaries and
# 1/19 kg of principal polymer.
_CARBON_SCRAP_HEAT_MJ = {
    "tallow_stearic_acid": 0.04048490708458465,
    "tmq_antioxidant": 0.03950089537793921,
    "6ppd_antioxidant": 0.01955289072119727,
    "paraffin_wax": 0.035933869489311786,
    "processing_oil": 0.1447630905215719,
    "tbbs_accelerator": 0.028031953825988554,
    # NIST graphite oxidation 393.52 kJ/mol / nominal 12 g/mol carbon.
    "carbon_black_graphite_reference": 1.0086222040423072,
    "retained_principal_sbr": 2.2799261885543816,
}
# SO2 formation 296.84 kJ/mol / nominal 32 g/mol elemental sulfur, applied
# only to the original elemental charge. Organic ingredient S is above.
# Dispersed ZnO contributes no combustion heat; it remains in the atom input.
_SULFUR_SCRAP_HEAT_MJ = 0.008581509836581791


@dataclass(frozen=True, slots=True)
class _SbrMaterial:
    """Dry inlet kg and gross MJ before any gas, leachate or residue transfer.

    Carbon dictionaries distinguish tracked principal from fixed compounding
    auxiliaries. ``carbon_heat_mj`` contains the carbon-bearing ingredients'
    complete heat, including their H/N/O/S; ``sulfur_heat_mj`` contains only
    separately charged elemental sulfur. Their sum is the gross reference.
    Keeping them separate preserves the landfill's C- and S-phase allocation.
    """

    mass_kg: float
    elements_kg: dict[str, float]
    principal_carbon_kg: dict[_CarbonOrigin, float]
    auxiliary_carbon_kg: dict[_CarbonOrigin, float]
    carbon_heat_mj: float
    sulfur_heat_mj: float


def _sbr_compound_material(origin: _CarbonOrigin, *, principal_kg: float) -> _SbrMaterial:
    """Return fixed compound22 material accompanying the actual principal kg.

    Retained end-of-life material uses one kg principal. Manufacturing trim
    uses 1/19 kg, from 5% discarded /95% retained. Multiplying the canonical
    scrap pools by ``19*principal_kg`` preserves both homogeneous quantities. Source water is zero;
    ZnO remains dispersed in the compound rather than a separate bulk phase.
    """
    scale = 19 * principal_kg
    principal_carbon = principal_kg * _PRINCIPAL_CARBON_FRACTION
    elements = {element: amount * scale for element, amount in _AUXILIARY_SCRAP_KG.items()}
    elements["C"] += principal_carbon
    elements["H"] += principal_kg * (1 - _PRINCIPAL_CARBON_FRACTION)
    return _SbrMaterial(
        mass_kg=fsum(elements.values()),
        elements_kg=elements,
        principal_carbon_kg={origin: principal_carbon},
        auxiliary_carbon_kg={
            "fossil": _AUXILIARY_FOSSIL_CARBON * scale,
            "biogenic": _AUXILIARY_BIOGENIC_CARBON * scale,
        },
        carbon_heat_mj=fsum(_CARBON_SCRAP_HEAT_MJ.values()) * scale,
        sulfur_heat_mj=_SULFUR_SCRAP_HEAT_MJ * scale,
    )
