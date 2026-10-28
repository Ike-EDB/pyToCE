"""Define TPS residue and landfill-phase loads per kg principal glucan/glycerol.

The SP/IVL (2000), Chapter 7 biochemical-class reference sends 99% of
principal/protein carbon to gas and 1% to leachate. Retained low-molecular
reagents instead enter leachate completely under the declared mobility
reference. These integrated allocations are not site-specific kinetics.
Wastewater removal, digestion, sludge and gas treatment belong to subsequent
owners. No inventory supplier or elementary flow is created here.
"""

from dataclasses import dataclass
from math import fsum
from typing import Literal, cast

from .leachate_load import _LeachateCarbon, _LeachateLoad
from .material_carbon import _TPS_PRINCIPAL_ELEMENT_FRACTIONS
from .open_burning import _organic_heat_potential
from .waste_gas_load import _GasPool, _LandfillGasLoad

_TPSOrigin = Literal["fossil", "biogenic", "atmospheric_dac"]
_FORMULATION_ORIGINS: dict[str, _TPSOrigin] = {
    "tps_fossil": "fossil",
    "tps_biogenic": "biogenic",
    "tps_atmospheric_dac": "atmospheric_dac",
}
# Adopted atomic molar masses (g/mol) convert the retained TPS reagent and
# protein formulas to mass fractions. Glucan/glycerol use the same C/H/O masses
# through the shared _TPS_PRINCIPAL_ELEMENT_FRACTIONS from material accounting.
_ATOMIC_MASSES = {
    "C": 12.011,
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "Na": 22.990,
    "Mg": 24.305,
    "Zn": 65.38,
    "Cl": 35.45,
}
# Charged free-species and anhydrous references. S4379 retains its
# powder; Na5P3O10 supplies only the manufacturing proxy.
_REAGENTS: tuple[tuple[str, str, dict[str, float]], ...] = (
    ("methanol", "", {"C": 1, "H": 4, "O": 1}),
    (
        "hepes",
        "hepes_best_available_organic_chemical_analogue",
        {"C": 8, "H": 18, "N": 2, "O": 4, "S": 1},
    ),
    ("sodium_chloride", "sodium_chloride", {"Na": 1, "Cl": 1}),
    (
        "magnesium_chloride",
        "magnesium_chloride_best_available_inorganic_chemical_analogue",
        {"Mg": 1, "Cl": 2},
    ),
    (
        "zinc_sulfate",
        "zinc_sulfate_best_available_inorganic_chemical_analogue",
        {"Zn": 1, "S": 1, "O": 4},
    ),
    (
        "thiamine_pyrophosphate",
        "thiamine_pyrophosphate_best_available_organic_chemical_analogue",
        {"C": 12, "H": 19, "N": 4, "O": 7, "P": 2, "S": 1},
    ),
    (
        "adp",
        "adp_best_available_organic_chemical_analogue",
        {"C": 10, "H": 15, "N": 5, "O": 10, "P": 2},
    ),
    ("edta", "edta", {"C": 10, "H": 16, "N": 2, "O": 8}),
    (
        "s4379",
        "polyphosphate_as_sodium_tripolyphosphate",
        {"Na": 47, "P": 45, "O": 136},
    ),
)
# Senko et al. (1995) averagine, from UCSF Protein Prospector, supplies
# average amino-acid atom counts for the TPS enzyme reference.
_PROTEIN_ATOMS = {"C": 4.9384, "H": 7.7583, "N": 1.3577, "O": 1.4773, "S": 0.0417}
# Carbon fractions in CH4 follow these formula-derived gas ratios: glucan C6H10O5 + H2O -> 3 CH4 + 3
# CO2; glycerol C3H8O3 -> 1.75 CH4 + 1.25 CO2 + 0.5 H2O (1.75/3 = 7/12); averagine + 3.2993 H2O ->
# 2.5501 CH4 + 2.3883 CO2 + 1.3577 NH3 + 0.0417 H2S. SP/IVL (2000), Chapter 7 supplies the separate
# 99% gas/1% leachate extent. These ratios assign TPS gas carbon.
_CLASS_METHANE_CARBON_SHARE = {
    "glucan": 0.5,
    "glycerol": 7 / 12,
    "protein": 2.5501 / 4.9384,
}


@dataclass(frozen=True, slots=True)
class _TPSComponentLoad:
    """One incoming dry component; element quantities are kg per principal kg."""

    component: str
    principal: bool
    mass_kg: float
    elements_kg: dict[str, float]


@dataclass(frozen=True, slots=True)
class _TPSMaterialCarbon:
    """One component's incoming kg C by origin, before any treatment allocation."""

    component: str
    principal: bool
    origin: _TPSOrigin
    input_carbon_kg: float


@dataclass(frozen=True, slots=True)
class _TPSMaterialLoad:
    """Actual formulation per kg principal TPS, independent of its receiving process.

    Dry components retain their original elemental and carbon-origin accounts.
    Wet mass includes the separately carried water once, on the incoming material basis.
    """

    formulation_service_id: str
    components: tuple[_TPSComponentLoad, ...]
    elements_kg: dict[str, float]
    water_kg: float
    wet_mass_kg: float
    carbon: tuple[_TPSMaterialCarbon, ...]


@dataclass(frozen=True, slots=True)
class _TPSCarbonLoad:
    """One component/origin's kg C before gas or leachate treatment.

    Methane and carbon dioxide fields contain carbon mass.
    The three phase quantities exhaust incoming carbon under the selected
    complete-class or dissolved-reagent reference.
    """

    component: str
    principal: bool
    origin: _TPSOrigin
    input_carbon_kg: float
    methane_carbon_kg: float
    carbon_dioxide_carbon_kg: float
    leachate_carbon_kg: float


@dataclass(frozen=True, slots=True)
class _TPSProteinElementPhases:
    """Protein N or S, in kg element, under the Doka mixed-landfill transfer."""

    gas_kg: float
    short_leachate_kg: float
    later_leachate_kg: float


@dataclass(frozen=True, slots=True)
class _TPSResidueLoad:
    """Material inputs and integrated landfill phases for 1 kg principal TPS.

    Components and their element totals exclude the separately carried water.
    Protein and reagents are auxiliary material, including when their carbon
    shares a principal origin. H/O speciation, gas treatment, wastewater treatment,
    and terminal inventories are calculated by their respective process functions.
    """

    formulation_service_id: str
    components: tuple[_TPSComponentLoad, ...]
    elements_kg: dict[str, float]
    water_kg: float
    wet_mass_kg: float
    carbon: tuple[_TPSCarbonLoad, ...]
    protein_nitrogen: _TPSProteinElementPhases
    protein_sulfur: _TPSProteinElementPhases
    leachate: _LeachateLoad


def _auxiliary_elements(mass: float, atoms: dict[str, float]) -> dict[str, float]:
    """Normalize an admitted reagent/protein formula to its actual charged mass."""
    molecular_mass = fsum(number * _ATOMIC_MASSES[element] for element, number in atoms.items())
    return {
        element: mass * number * _ATOMIC_MASSES[element] / molecular_mass
        for element, number in atoms.items()
    }


def _material_carbon(
    component: _TPSComponentLoad, origin: _TPSOrigin, fraction: float
) -> _TPSMaterialCarbon:
    """Assign the component's original carbon to its declared feedstock origin."""
    return _TPSMaterialCarbon(
        component.component, component.principal, origin, component.elements_kg["C"] * fraction
    )


def _landfill_carbon(pool: _TPSMaterialCarbon) -> _TPSCarbonLoad:
    """Apply the SP class or dissolved-reagent landfill rule to incoming carbon."""
    carbon = pool.input_carbon_kg
    if pool.component in _CLASS_METHANE_CARBON_SHARE:
        # SP/IVL2000 Chapter7: 99% of class carbon forms gas, 1% enters leachate.
        gas = 0.99 * carbon
        methane = gas * _CLASS_METHANE_CARBON_SHARE[pool.component]
        return _TPSCarbonLoad(
            pool.component,
            pool.principal,
            pool.origin,
            carbon,
            methane,
            gas - methane,
            0.01 * carbon,
        )
    return _TPSCarbonLoad(pool.component, False, pool.origin, carbon, 0.0, 0.0, carbon)


def _retained_reagents(
    services: dict[str, dict[str, object]], origin: _TPSOrigin, amylose_kg: float
) -> tuple[_TPSComponentLoad, ...]:
    """Carry the selected reaction charges through its existing membrane coefficient.

    Reaction components use kg per kg gross glucan. The concentration coefficient
    already cancels reaction-volume scaling; applying gross recovery feed again
    would double count it. Consumed methanol follows the source's product
    stoichiometry, separately from the glucose-equivalent titer convention.
    """
    recovery = cast(dict[str, object], services[f"starch_{origin}"]["representative_method"])
    reaction = services[f"methanol_glucan_reaction_{origin}"]
    method = cast(dict[str, object], reaction["representative_method"])
    construction = cast(dict[str, object], reaction["construction"])
    inputs = cast(list[dict[str, object]], construction["components"])
    named = {str(row["component_id"]): row for row in inputs if "component_id" in row}
    methanol = next(row for row in inputs if row.get("service_id") == f"methanol_{origin}")
    # CH4O and the C6H10O5 glucan repeat unit give g/mol. Six methanol carbon
    # atoms supply each fully methanol-derived glucan unit under this reaction.
    methanol_mass = _ATOMIC_MASSES["C"] + 4 * _ATOMIC_MASSES["H"] + _ATOMIC_MASSES["O"]
    glucan_mass = 6 * _ATOMIC_MASSES["C"] + 10 * _ATOMIC_MASSES["H"] + 5 * _ATOMIC_MASSES["O"]
    consumed = (
        float(cast(float, method["selected_origin_share_of_dry_amylose"]))
        * 6
        * methanol_mass
        / glucan_mass
    )
    factor = (
        float(cast(float, recovery["fully_passing_solute_coefficient"]))
        * amylose_kg
        / float(cast(float, recovery["dry_amylose_kg"]))
    )
    result: list[_TPSComponentLoad] = []
    for name, component_id, atoms in _REAGENTS:
        if name == "methanol":
            charge = float(cast(float, methanol["amount"])) - consumed
        elif name == "s4379":
            charge = float(cast(float, named[component_id]["physical_quantity_kg"]))
        else:
            charge = float(cast(float, named[component_id]["amount"]))
        mass = charge * factor
        result.append(_TPSComponentLoad(name, False, mass, _auxiliary_elements(mass, atoms)))
    return tuple(result)


def _incoming_leachate(
    components: tuple[_TPSComponentLoad, ...],
    carbon: tuple[_TPSCarbonLoad, ...],
    nitrogen: _TPSProteinElementPhases,
    sulfur: _TPSProteinElementPhases,
) -> _LeachateLoad:
    """Combine fatty-acid-equivalent class C with fully dissolved reagent loads.

    The acetate reference retains its admitted C12.01/H1.008/O16 convention.
    Doka's reagent H/C=.1225 and O/C=.64 are empirical process descriptors;
    sulfate/polyphosphate oxygen must not enter those organic ratios.
    """
    reagents = tuple(c for c in components if c.component not in _CLASS_METHANE_CARBON_SHARE)
    reagent_elements = {
        element: fsum(component.elements_kg.get(element, 0.0) for component in reagents)
        for element in _ATOMIC_MASSES
    }
    acetate_carbon = fsum(
        pool.leachate_carbon_kg for pool in carbon if pool.component in _CLASS_METHANE_CARBON_SHARE
    )
    acetate = {
        "C": acetate_carbon,
        "H": acetate_carbon * 4 * 1.008 / (2 * 12.01),
        "O": acetate_carbon * 2 * 16 / (2 * 12.01),
    }
    elements = dict(reagent_elements)
    for element, amount in acetate.items():
        elements[element] += amount
    elements["N"] += nitrogen.short_leachate_kg
    elements["S"] += sulfur.short_leachate_kg
    reagent_carbon = reagent_elements["C"]
    toc = acetate_carbon + reagent_carbon
    return _LeachateLoad(
        elements_kg=elements,
        carbon=tuple(
            _LeachateCarbon(pool.component, pool.principal, pool.origin, pool.leachate_carbon_kg)
            for pool in carbon
        ),
        organic_hydrogen_kg=acetate["H"] + 0.1225 * reagent_carbon,
        organic_oxygen_kg=acetate["O"] + 0.64 * reagent_carbon,
        toc_kg=toc,
        doc_kg=toc,
        # C2H4O2 requires 2 O2: 4*16/(2*12.01) kg O2 per kg acetate C.
        # Doka's average untreated COD/TOC supplies the reagent mixture's
        # empirical kg O2/kg C indicator.
        cod_kg_o2=acetate_carbon * 4 * 16 / (2 * 12.01) + 3.223281269732706 * reagent_carbon,
        # Doka 2021 avgWWT!B73 / WWcalc!B84/B91: empirical BOD5/TOC,
        # in kg O2/kg C, applied to the full TPS leachate organic-carbon load.
        bod5_kg_o2=1.7586552173908316 * toc,
        # Retained NaCl and MgCl2 from synthetic TPS manufacture dissolve as chloride. Carry their
        # kg Cl as the known ionic subset of total Cl so untreated leachate uses the existing
        # chloride flow.
        chloride_kg=fsum(
            component.elements_kg["Cl"]
            for component in reagents
            if component.component in {"sodium_chloride", "magnesium_chloride"}
        ),
    )


def _tps_material_load(
    services: dict[str, dict[str, object]], formulation_service_id: str
) -> _TPSMaterialLoad:
    """Construct one historical TPS formulation before assigning a treatment fate.

    Args:
        services: Admitted provider services, including TPS, technical amylose
            recovery and the selected methanol/glucan reaction.
        formulation_service_id: Actual ``tps_fossil``, ``tps_biogenic`` or
            ``tps_atmospheric_dac`` formulation, independently of aggregate C origin.

    Returns:
        Components, water, wet mass and incoming carbon per exactly one kg
        principal TPS. The caller scales this reference to its existing principal
        mass. This function applies no recovery rejection, family or treatment
        share.

    Notes:
        Averagine describes protein composition. Its biogenic carbon origin
        follows the selected potato-starch-based enzyme production model.
    """
    origin = _FORMULATION_ORIGINS[formulation_service_id]
    method = cast(dict[str, object], services[formulation_service_id]["representative_method"])
    glucan_key = "product_starch_kg" if origin == "biogenic" else "product_amylose_kg"
    glucan = float(cast(float, method[glucan_key]))
    glycerol = float(cast(float, method["product_glycerine_kg"]))
    water = float(cast(float, method["co_delivered_water_kg"]))
    components = [
        _TPSComponentLoad(
            name,
            True,
            mass,
            {
                element: mass * fraction
                for element, fraction in _TPS_PRINCIPAL_ELEMENT_FRACTIONS[name].items()
            },
        )
        for name, mass in (("glucan", glucan), ("glycerol", glycerol))
    ]
    carbon = [_material_carbon(components[1], origin, 1.0)]
    if origin == "biogenic":
        carbon.append(_material_carbon(components[0], "biogenic", 1.0))
    else:
        primer_method = cast(dict[str, object], services["starch_fossil"]["representative_method"])
        primer = float(cast(float, primer_method["biogenic_primer_share_of_dry_amylose"]))
        carbon.extend(
            (
                _material_carbon(components[0], origin, 1 - primer),
                _material_carbon(components[0], "biogenic", primer),
            )
        )
        protein_mass = float(cast(float, method["co_delivered_protein_kg"]))
        protein = _TPSComponentLoad(
            "protein", False, protein_mass, _auxiliary_elements(protein_mass, _PROTEIN_ATOMS)
        )
        components.append(protein)
        carbon.append(_material_carbon(protein, "biogenic", 1.0))
        reagents = _retained_reagents(services, origin, glucan)
        components.extend(reagents)
        carbon.extend(
            _material_carbon(
                component, origin if component.component == "methanol" else "fossil", 1.0
            )
            for component in reagents
            if "C" in component.elements_kg
        )
    wet_mass = fsum(component.mass_kg for component in components) + water
    return _TPSMaterialLoad(
        formulation_service_id=formulation_service_id,
        components=tuple(components),
        elements_kg={
            element: fsum(component.elements_kg.get(element, 0.0) for component in components)
            for element in _ATOMIC_MASSES
        },
        water_kg=water,
        wet_mass_kg=wet_mass,
        carbon=tuple(carbon),
    )


def _tps_residue_load(
    services: dict[str, dict[str, object]], formulation_service_id: str
) -> _TPSResidueLoad:
    """Allocate the actual TPS formulation to its admitted landfill phases.

    SP/IVL Chapter7 defines principal/protein carbon conversion. The selected
    Doka mixed-landfill functions define protein N/S phases; retained reagents
    follow the fully dissolved reference. All quantities remain per principal
    kg.
    """
    material = _tps_material_load(services, formulation_service_id)
    carbon = tuple(_landfill_carbon(pool) for pool in material.carbon)
    if formulation_service_id == "tps_biogenic":
        # The native starch/glycerol formulation has no retained enzyme protein.
        nitrogen = _TPSProteinElementPhases(0.0, 0.0, 0.0)
        sulfur = _TPSProteinElementPhases(0.0, 0.0, 0.0)
    else:
        protein = next(item for item in material.components if item.component == "protein")
        protein_n = protein.elements_kg["N"]
        # Doka avgTK!G25: gas share of fully mobilized protein N. The remaining
        # N enters short-term leachate; no artificial kinetic tail is introduced.
        nitrogen_gas = 0.06438442211055276 * protein_n
        nitrogen = _TPSProteinElementPhases(nitrogen_gas, protein_n - nitrogen_gas, 0.0)
        protein_s = protein.elements_kg["S"]
        # avgTK!I24 and G24: mobile S/input S, then gas S/mobile S. The residual
        # mobile pool enters short-term leachate and immobilized S enters later.
        mobile_s = 0.437649447721238 * protein_s
        sulfur_gas = 0.1490935840646033 * mobile_s
        sulfur = _TPSProteinElementPhases(sulfur_gas, mobile_s - sulfur_gas, protein_s - mobile_s)
    return _TPSResidueLoad(
        formulation_service_id=formulation_service_id,
        components=material.components,
        elements_kg=material.elements_kg,
        water_kg=material.water_kg,
        wet_mass_kg=material.wet_mass_kg,
        carbon=carbon,
        protein_nitrogen=nitrogen,
        protein_sulfur=sulfur,
        leachate=_incoming_leachate(material.components, carbon, nitrogen, sulfur),
    )


def _landfill_decomposition(
    load: _TPSResidueLoad, methane: float, sulfur: float
) -> tuple[float, dict[str, float]]:
    """Allocate organic heat to soil after gas and actual acetate/N/S leachate.

    Passed-through low-MW reagents retain their energy at the water owner and
    cancel here. Michel's organic formula (via Muñoz et al. (2007), Eq. 9) uses the organic H/O
    quantities.
    No remaining
    carbon reservoir permits the donor's long-term water heat or a clipped tail.
    """
    organic = [
        item for item in load.components if item.component in {"glucan", "glycerol", "protein"}
    ]
    # Michel (via Doka 2021 WWcalc) coefficients estimate MJ from kg of
    # eligible organic elements.
    input_energy = fsum(_organic_heat_potential(item.elements_kg) for item in organic)
    protein_n = fsum(item.elements_kg.get("N", 0.0) for item in organic)
    protein_s = fsum(item.elements_kg.get("S", 0.0) for item in organic)
    acetate_c = fsum(
        pool.leachate_carbon_kg
        for pool in load.carbon
        if pool.component in _CLASS_METHANE_CARBON_SHARE
    )
    # The same C2H4O2 reference used by _incoming_leachate, in kg elements.
    leach_energy = _organic_heat_potential(
        {
            "C": acetate_c,
            "H": acetate_c * 4 * 1.008 / (2 * 12.01),
            "O": acetate_c * 2 * 16 / (2 * 12.01),
        }
    )
    leach_energy += 6.276 * (protein_n - load.protein_nitrogen.gas_kg)
    leach_energy += 19.079 * (protein_s - sulfur)
    # Chase (1998), via NIST Chemistry WebBook liquid-water HHV, using Doka's nominal kg/kmol
    # masses.
    methane_hhv = (-74.87 + 393.52 + 2 * 285.83) / 16
    sulfide_hhv = (-20.50 + 296.84 + 285.83) / 34
    gas_energy = methane_hhv * methane + sulfide_hhv * sulfur * 34 / 32
    heat = input_energy - leach_energy - gas_energy
    return heat, {
        "input_organic_hhv_mj": input_energy,
        "leachate_energy_mj": leach_energy,
        "raw_gas_hhv_mj": gas_energy,
        "heat_soil_mj": heat,
    }


def _tps_landfill_gas_load(load: _TPSResidueLoad) -> _LandfillGasLoad:
    """Project TPS class gas and its remaining decomposition heat into treatment."""
    pools = tuple(
        _GasPool(
            x.component, x.principal, x.origin, x.methane_carbon_kg, x.carbon_dioxide_carbon_kg
        )
        for x in load.carbon
    )
    # Gas pools carry kg C; nominal molar masses convert that to kg CH4.
    methane = fsum(x.methane_carbon_kg for x in pools) * 16 / 12
    sulfur = load.protein_sulfur.gas_kg
    heat, trace = _landfill_decomposition(load, methane, sulfur)
    return _LandfillGasLoad(pools, load.protein_nitrogen.gas_kg, sulfur, heat, trace)
