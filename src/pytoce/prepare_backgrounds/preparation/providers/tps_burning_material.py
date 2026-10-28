"""Construct uncontrolled burning of each historical thermoplastic starch mixture.

The three fixed TPS formulations retain their glucan, glycerol, reaction
reagents, protein and water through material custody. Their complete inputs
replace the mixed-plastic composition in Doka's uncontrolled-burning function.
The original source supplies its method identity.
"""

from math import fsum

from .leachate_load import _CarbonOrigin
from .open_burning import _OpenBurningMaterial, _organic_heat_potential
from .tps_residue_load import _TPSMaterialLoad

_ORGANIC_COMPONENTS = {"glucan", "glycerol", "protein", "methanol", "hepes", "edta"}
# The existing inlet formulas are ADP C10H15N5O10P2 and thiamine pyrophosphate
# C12H19N4O7P2S. Their qualified heat references use the corresponding organic
# parents C10H13N5O4 and C12H17N4OS, on the full cofactor mass denominator.
# These H/O ratios remove H2P2O6 only from the empirical heat predictor; there
# is no modeled hydrolysis, extra water or change to the incoming atoms.
_COFACTOR_HEAT_FRACTIONS = {
    "adp": {"H": 13 / 15, "O": 4 / 10},
    "thiamine_pyrophosphate": {"H": 17 / 19, "O": 1 / 7},
}
_ORIGINS: tuple[_CarbonOrigin, ...] = ("fossil", "biogenic", "atmospheric_dac")


def _tps_burning_material(inlet: _TPSMaterialLoad) -> _OpenBurningMaterial:
    """Keep the actual TPS atoms and carbon while admitting its organic gross heat.

    All amounts describe the material accompanying one kg principal TPS.
    Michel (via Doka) estimates gross heat from the declared organic component atoms.
    Carried water and sodium, magnesium, zinc and polyphosphate salts supply no
    oxidation-fuel heat. The complete reagent atoms enter the burnable matrix, including components
    with zero fuel heat.
    """
    heats: list[float] = []
    for component in inlet.components:
        if component.component in _ORGANIC_COMPONENTS:
            heats.append(_organic_heat_potential(component.elements_kg))
        elif component.component in _COFACTOR_HEAT_FRACTIONS:
            fractions = _COFACTOR_HEAT_FRACTIONS[component.component]
            parent_elements = {
                element: amount * fractions.get(element, 1.0)
                for element, amount in component.elements_kg.items()
                if element != "P"
            }
            heats.append(_organic_heat_potential(parent_elements))
    principal: dict[_CarbonOrigin, float] = {
        origin: fsum(
            pool.input_carbon_kg
            for pool in inlet.carbon
            if pool.principal and pool.origin == origin
        )
        for origin in _ORIGINS
    }
    auxiliary: dict[_CarbonOrigin, float] = {
        origin: fsum(
            pool.input_carbon_kg
            for pool in inlet.carbon
            if not pool.principal and pool.origin == origin
        )
        for origin in _ORIGINS
    }
    return _OpenBurningMaterial(
        wet_mass_kg=inlet.wet_mass_kg,
        water_kg=inlet.water_kg,
        burnable_elements_kg=inlet.elements_kg,
        inert_elements_kg={},
        principal_carbon_kg=principal,
        auxiliary_carbon_kg=auxiliary,
        heat_potential_mj=fsum(heats),
    )
