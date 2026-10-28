"""Reconstruct uncontrolled burning of admitted actual material references.

Doka Open Burning2020 supplies the complete uncontrolled function: emissions
to air and residues left in industrial soil.
Its generic partitions and wet-mass emission factors are transferred to the
actual material. Molecular balances determine SO2, water, oxygen and residual
CO2. Collection to the disposal location is external. PCB and TCDD-equivalent
quantities use source-reference atom accounts.
"""

from dataclasses import dataclass, field
from math import fsum

from .leachate_load import _CarbonOrigin
from .pa_finish_material import _pa_finish_load, _PAFinishReference

_OpenBurningFlow = tuple[str, str, tuple[str, ...]]
_AIR = ("air", "non-urban air or from high stacks")
_SOIL = ("soil", "industrial")
_INDICATOR = ("inventory indicator", "waste")
_PAH = "PAH, polycyclic aromatic hydrocarbons"
_PCB = "Polychlorinated biphenyls"
_DIOXIN = "Dioxins, measured as 2,3,7,8-tetrachlorodibenzo-p-dioxin"
# The compiler binds these descriptors to native endpoints. Ionic flow quantities are kg element.
_ELEMENT_NAMES = {
    "C": "Carbon",
    "N": "Nitrogen",
    "S": "Sulfur",
    "P": "Phosphorus",
    "B": "Boron",
    "Cl": "Chloride",
    "Co": "Cobalt II",
    "Zn": "Zinc II",
    "Ti": "Titanium ion",
    "Si": "Silicon",
    "Ca": "Calcium II",
    "Al": "Aluminium III",
    "K": "Potassium I",
    "Mg": "Magnesium",
    "Na": "Sodium I",
}
# Source-equivalent C12H10-xClx.
_PCB_CARBON = 0.441533266
_PCB_MASS = 144 / _PCB_CARBON
_PCB_CHLORINATION = (_PCB_MASS - 154) / (35.453 - 1)
_PCB_HYDROGEN = (10 - _PCB_CHLORINATION) / _PCB_MASS
_PCB_CHLORINE = 35.453 * _PCB_CHLORINATION / _PCB_MASS
# C12H4Cl4O2 supplies the reference-equivalent accounting for the TEQ flow.
_TCDD_MASS = 144 + 4 + 4 * 35.453 + 32
_TCDD_ATOMS = {
    "C": 144 / _TCDD_MASS,
    "H": 4 / _TCDD_MASS,
    "Cl": 4 * 35.453 / _TCDD_MASS,
    "O": 32 / _TCDD_MASS,
}

# Both PA grades retain the principal MFA's nominal C6H11NO/113 convention.
# The finish owner supplies its separately admitted CIAAW elemental masses.
_PRINCIPAL = {"C": 72 / 113, "H": 11 / 113, "N": 14 / 113, "O": 16 / 113}
# Admitted burnable elements from Doka2020 TK J23:J61: g residue element
# per kg input element. Bulk-only minerals use their unchanged soil atoms.
_RESIDUE = {
    "C": 5.0150854170955386 / 1000,
    "S": 856.4343815936636 / 1000,
    "N": 10.012170059468751 / 1000,
    "P": 993.12931599358296 / 1000,
    "Cl": 332.88933288933293 / 1000,
    "Co": 959.66711945301552 / 1000,
    "Zn": 992.5600513870927 / 1000,
    "Ti": 989.39666878576372 / 1000,
    "Si": 999.99532744142095 / 1000,
    "Mg": 997.00410581231313 / 1000,
    "Na": 975.36945812807858 / 1000,
}
# Ash oxides F23:F61: oxide mass minus element mass, kg O/kg element.
# Retain PA's established nominal expressions, including half-carbonate C.
_RESIDUE_OXYGEN = {
    "C": 2.0,
    "S": 1.5,
    "N": 48 / 14,
    "P": 40 / 30.97,
    "Cl": 1.4513017234084562 - 1,
    "Co": 1.2715085694892245 - 1,
    "Zn": 1.2447605935444392 - 1,
    "Ti": 1.6683375104427733 - 1,
    "Si": 2.1393576871038951 - 1,
    "Mg": 1.6581112207963145 - 1,
    "Na": 8 / 23,
}
# Source organic factors (kg/kg wet waste) and nominal molecular atom counts. These seven outputs
# supplement the separate EPA16 PAH species listed below.
_ORGANICS = {
    "Benzene": (0.00097975, 6, 6, 0),
    "Phenol": (0.000014799342788831124, 6, 6, 1),
    "Styrene": (0.0005275, 8, 8, 0),
    "Toluene": (0.000372, 7, 8, 0),
    "Formaldehyde": (0.00044365, 1, 2, 1),
    "Acetaldehyde": (0.0004284, 2, 4, 1),
    "Ethyl benzene": (0.00032746373845053444, 8, 10, 0),
}
# EPA600/R-97-134a nonrecycler runs 4 and 5 supply the aggregate C/H
# composition proxy in mg/kg consumed. Doka supplies the wet-waste factor.
_PAH_PROFILE = (
    (12, 10, 0.9578, 0.5780),  # Acenaphthene
    (12, 8, 13.6424, 8.9577),  # Acenaphthylene
    (14, 10, 2.3724, 1.2910),  # Anthracene
    (18, 12, 3.1364, 1.3425),  # Benzo(a)anthracene
    (20, 12, 3.1275, 1.1167),  # Benzo(a)pyrene
    (20, 12, 3.7585, 1.5016),  # Benzo(b)fluoranthene
    (22, 12, 2.8148, 1.1340),  # Benzo(ghi)perylene
    (20, 12, 1.6424, 0.4085),  # Benzo(k)fluoranthene
    (18, 12, 3.5588, 1.5136),  # Chrysene
    (22, 14, 0.4861, 0.1595),  # Dibenzo(ah)anthracene
    (16, 10, 5.1917, 2.9436),  # Fluoranthene
    (13, 10, 4.7756, 2.5360),  # Fluorene
    (22, 12, 2.7997, 0.9547),  # Indeno(123cd)pyrene
    (10, 8, 18.9598, 16.1032),  # Naphthalene
    (14, 10, 8.9946, 5.6546),  # Phenanthrene
    (16, 10, 6.1419, 3.5157),  # Pyrene
)
# Paired sums equal normalized paired means: their common factor 1/2 cancels.
_PAH_CARBON = fsum(
    (run4 + run5) * 12 * carbon / (12 * carbon + hydrogen)
    for carbon, hydrogen, run4, run5 in _PAH_PROFILE
) / fsum(run4 + run5 for _, _, run4, run5 in _PAH_PROFILE)


@dataclass(frozen=True, slots=True)
class _OpenBurningMaterial:
    """Material reference with mass, phase quantities and tracked carbon.

    Masses are kg and gross heat is MJ. Water is included in wet mass. Omitted
    elements are model zeros; inert atoms retain their original H/O. Principal
    and auxiliary C weights refer to the burnable phase. Geological carbonate
    C is included in wet mass and recorded separately from organic fate.
    Tracked C can differ from a proxy's elemental C. The caller supplies the
    material composition and selected transfer factors.
    """

    wet_mass_kg: float
    water_kg: float
    burnable_elements_kg: dict[str, float]
    inert_elements_kg: dict[str, float]
    principal_carbon_kg: dict[_CarbonOrigin, float]
    auxiliary_carbon_kg: dict[_CarbonOrigin, float]
    heat_potential_mj: float
    geological_carbon_kg: float = field(default=0.0, kw_only=True)


@dataclass(frozen=True, slots=True)
class _OpenBurningLoad:
    """One actual receiving load, mass-balance ledger and existing native outputs.

    Unallocated quantities are in kg, except the named heat indicator.
    Dry input elements include the burnable and inert phases, excluding incoming
    water and separately recorded geological carbon. Principal and auxiliary
    carbon totals refer to the burnable phase. Air-compound balances include water and N2;
    registered biosphere flows supply the corresponding emission identities. Atmospheric nitrogen is
    the input trace for thermal NOx and the selected process-N2O
    reference. Soil elements include inert original atoms and reference-organic
    atoms once. Soil compounds describe overlapping speciation of those atoms. Air oxides, soil
    atoms, native PM, and inventory indicators retain their respective reporting roles. Source
    rounding preserves wet mass minus water and admitted dry atoms. PCB/TCDD accounts use
    reference-equivalent quantities.
    Native keys retain name, unit and compartment.

    Principal fractions are attached per carbon-bearing native output. Fossil
    and non-fossil gases use their own group's carbon denominator; unsplit
    organics use tracked burnable C, and soil uses post-transfer tracked
    burnable C. Auxiliary carbon contributes to the total-carbon denominators; principal-carbon
    numerators retain principal carbon only.
    """

    wet_mass_kg: float
    incoming_water_kg: float
    elements_kg: dict[str, float]
    principal_carbon_kg: dict[_CarbonOrigin, float]
    auxiliary_carbon_kg: dict[_CarbonOrigin, float]
    air_compounds_kg: dict[str, float]
    air_oxide_elements_kg: dict[str, float]
    residue_elements_kg: dict[str, float]
    formed_water_kg: float
    atmospheric_oxygen_kg: float
    atmospheric_nitrogen_kg: float
    residue_mass_kg: float
    heat_potential_mj: float
    biosphere_amounts: dict[_OpenBurningFlow, float]
    principal_origin_fractions: dict[_OpenBurningFlow, dict[_CarbonOrigin, float]]
    burnable_elements_kg: dict[str, float]
    inert_elements_kg: dict[str, float]
    soil_compounds_kg: dict[str, float]
    source_rounding_kg: float
    geological_carbon_kg: float


def _organic_heat_potential(elements_kg: dict[str, float]) -> float:
    """Return Michel's (via Muñoz et al. (2007), Eq. 9) gross-heat reference for admitted organic
    atoms, in MJ.

    The caller owns eligible organic or parent-species selection and kg atom
    inputs. The CHONS estimate excludes inert oxides, salts and other elements.
    """
    return fsum(
        elements_kg.get(element, 0.0) * coefficient
        for element, coefficient in {
            "C": 34.016,
            "H": 124.265,
            "O": -9.8324,
            "N": 6.276,
            "S": 19.079,
        }.items()
    )


def _pa_open_burning_load(
    finish: _PAFinishReference,
    *,
    principal_by_origin_kg: dict[_CarbonOrigin, float],
    finish_dose_kg: float,
    water_kg: float,
) -> _OpenBurningLoad:
    """Burn an admitted actual principal/finish mixture at one uncontrolled site.

    Principal quantities are polymer kg by origin. The already allocated
    scalar dose multiplies the material owner's retained kg/kg-principal
    vector. The input mixture carries its existing coating and embedded finish. The caller supplies
    moisture separately. The actual family has no chlorine,
    bromine, fluorine
    or recovered metals; mixed-waste halogen emissions cannot transfer to it.
    Preparation owns input admission and the eventual native-flow bindings.
    """
    principal = fsum(principal_by_origin_kg.values())
    auxiliary = _pa_finish_load(
        {name: finish_dose_kg * mass for name, mass in finish.retained.compounds_kg.items()}
    )
    elements = {
        element: mass + principal * _PRINCIPAL.get(element, 0.0)
        for element, mass in auxiliary.elements_kg.items()
    }
    principal_carbon: dict[_CarbonOrigin, float] = {
        origin: mass * 72 / 113 for origin, mass in principal_by_origin_kg.items()
    }
    auxiliary_carbon: dict[_CarbonOrigin, float] = {
        "fossil": auxiliary.fossil_carbon_kg,
        "biogenic": auxiliary.biogenic_carbon_kg,
    }
    # Michel's (via Muñoz et al. (2007), Eq. 9) admitted nominal CHONS estimate, MJ/kg element. This
    # keeps the source gross-potential heat indicator.
    heat = _organic_heat_potential(elements)
    return _open_burning_load(
        _OpenBurningMaterial(
            wet_mass_kg=fsum(elements.values()) + water_kg,
            water_kg=water_kg,
            burnable_elements_kg=elements,
            inert_elements_kg={},
            principal_carbon_kg=principal_carbon,
            auxiliary_carbon_kg=auxiliary_carbon,
            heat_potential_mj=heat,
        )
    )


def _open_burning_load(material: _OpenBurningMaterial) -> _OpenBurningLoad:
    """Apply the source function to one material reference.

    Process factors use wet mass. Element transfers apply to the burnable
    matrix; bulk-inert atoms remain on site. Check mathematical complements against their specified
    bounds. The caller admits material and emission endpoints.
    Fixed N2O uses atmospheric-N accounting. PCB/TCDD retain their source
    quantities and reference-equivalent elemental account.
    """
    burnable = material.burnable_elements_kg
    inert = material.inert_elements_kg
    water_kg = material.water_kg
    wet = material.wet_mass_kg
    elements = {
        element: burnable.get(element, 0.0) + inert.get(element, 0.0)
        for element in dict.fromkeys((*burnable, *inert))
    }
    principal_carbon = material.principal_carbon_kg
    auxiliary_carbon = material.auxiliary_carbon_kg
    burnable_carbon = fsum((*principal_carbon.values(), *auxiliary_carbon.values()))
    residue = {
        element: burnable[element] * fraction
        for element, fraction in _RESIDUE.items()
        if element in burnable
    }
    airborne = {element: burnable[element] - mass for element, mass in residue.items()}
    air_oxides = {
        element: mass for element, mass in airborne.items() if element not in {"C", "N", "S", "Cl"}
    }
    air_oxides["O"] = fsum(mass * _RESIDUE_OXYGEN[element] for element, mass in air_oxides.items())

    air = {name: wet * factor for name, (factor, _, _, _) in _ORGANICS.items()}
    # EF D31: mg PAH/kg whole wet waste, divided by 1e6.
    air[_PAH] = wet * 0.00034430210526037831
    organic_carbon = (
        fsum(air[name] * 12 * c / (12 * c + h + 16 * o) for name, (_, c, h, o) in _ORGANICS.items())
        + air[_PAH] * _PAH_CARBON
    )
    organic_hydrogen = fsum(
        air[name] * h / (12 * c + h + 16 * o) for name, (_, c, h, o) in _ORGANICS.items()
    ) + air[_PAH] * (1 - _PAH_CARBON)
    organic_oxygen = fsum(
        air[name] * 16 * o / (12 * c + h + 16 * o) for name, (_, c, h, o) in _ORGANICS.items()
    )
    organic_chlorine = 0.0
    soil_compounds: dict[str, float] = {}
    soil_organic_atoms: dict[str, float] = {}
    # Chlorinated contamination is zero for the admitted zero-Cl case. Positive Cl uses the
    # unchanged wet PCB factor and burnable-Cl dioxin relation.
    if burnable.get("Cl", 0.0):
        air[_PCB] = wet * 0.00000011189198274953115
        air[_DIOXIN] = wet * 0.001683 * (1e6 * burnable["Cl"] / wet) ** 1.296881 / 1e12
        soil_compounds[_DIOXIN] = air[_DIOXIN] / 3.6
        soil_organic_atoms = {
            element: soil_compounds[_DIOXIN] * fraction for element, fraction in _TCDD_ATOMS.items()
        }
        organic_carbon += air[_PCB] * _PCB_CARBON + air[_DIOXIN] * _TCDD_ATOMS["C"]
        organic_hydrogen += air[_PCB] * _PCB_HYDROGEN + air[_DIOXIN] * _TCDD_ATOMS["H"]
        organic_oxygen += air[_DIOXIN] * _TCDD_ATOMS["O"]
        organic_chlorine = air[_PCB] * _PCB_CHLORINE + air[_DIOXIN] * _TCDD_ATOMS["Cl"]
        residue["C"] -= soil_organic_atoms["C"]
        residue["Cl"] -= soil_organic_atoms["Cl"]
    air_chlorine = airborne.get("Cl", 0.0) - organic_chlorine
    if "Cl" in airborne:
        # HCl/Cl molecular masses convert the remaining chlorine to acid gas.
        air["Hydrochloric acid"] = air_chlorine * (36.453 / 35.453)

    # EF D21/D22: 38600 mg CO and 5900 mg CH4/kg wet waste; /1e6 gives kg.
    # These remain municipal-waste process-factor transfers.
    air["Carbon monoxide"] = wet * 0.0386
    air["Methane"] = wet * 0.0059
    remaining_carbon = (
        airborne["C"] - air["Carbon monoxide"] * 12 / 28 - air["Methane"] * 12 / 16 - organic_carbon
    )
    air["Carbon dioxide"] = remaining_carbon * 44 / 12
    # Source fuel-N conversion plus the selected native control's 810 mg
    # thermal NOx/kg wet waste (/1e6 to kg). Thermal N comes from the atmosphere.
    fuel_conversion = 0.37383241657031102
    fuel_nox = airborne.get("N", 0.0) * 46 / 14 * fuel_conversion
    thermal_nox = wet * 0.000810
    air["Nitrogen oxides"] = fuel_nox + thermal_nox
    # EF D23 assigns 116.25 mg N2O/kg wet waste, converted by 1e6.
    # Assign this process factor to atmospheric N, separately from waste-N
    # conversion, for all polymers.
    air["Dinitrogen monoxide"] = wet * 0.00011625
    air["Nitrogen"] = airborne.get("N", 0.0) * (1 - fuel_conversion)
    air["Sulfur dioxide"] = airborne.get("S", 0.0) * 64 / 32
    remaining_hydrogen = (
        burnable["H"]
        - air["Methane"] * 4 / 16
        - organic_hydrogen
        - soil_organic_atoms.get("H", 0.0)
        - air.get("Hydrochloric acid", 0.0) / 36.453
    )
    formed_water = 9 * remaining_hydrogen
    air["Water"] = water_kg + formed_water

    # Oxidize only the complementary burnable residue. Inert original atoms
    # and the separately represented soil-organic atoms enter the ledger once.
    residue["O"] = fsum(mass * _RESIDUE_OXYGEN[element] for element, mass in residue.items())
    soil_elements = {
        element: residue.get(element, 0.0)
        + inert.get(element, 0.0)
        + soil_organic_atoms.get(element, 0.0)
        for element in dict.fromkeys((*residue, *inert, *soil_organic_atoms))
    }
    # The PM chlorine oxide equivalent overlaps HCl.
    oxygen = fsum(
        (
            air["Carbon dioxide"] * 32 / 44,
            air["Carbon monoxide"] * 16 / 28,
            air["Sulfur dioxide"] * 32 / 64,
            air["Nitrogen oxides"] * 32 / 46,
            air["Dinitrogen monoxide"] * 16 / 44,
            organic_oxygen,
            formed_water * 8 / 9,
            air_oxides["O"],
            soil_elements["O"],
            -elements.get("O", 0.0),
        )
    )
    complements = {
        "air carbon": remaining_carbon,
        "non-water hydrogen": remaining_hydrogen,
        "air chlorine": air_chlorine,
        "burnable soil chlorine": residue.get("Cl", 0.0),
        "burnable soil carbon": residue["C"],
        "atmospheric oxygen": oxygen,
    }
    if any(mass < 0 for mass in complements.values()):
        raise ValueError(
            f"Open-burning reference has inadmissible source complements: {complements}"
        )
    amounts: dict[_OpenBurningFlow, float] = {
        (name, "kilogram", _AIR): air[name]
        for name in (*_ORGANICS, _PAH, "Nitrogen oxides", "Dinitrogen monoxide", "Sulfur dioxide")
    }
    global_fractions: dict[_CarbonOrigin, float] = {
        origin: carbon / burnable_carbon for origin, carbon in principal_carbon.items()
    }
    fractions: dict[_OpenBurningFlow, dict[_CarbonOrigin, float]] = {
        (name, "kilogram", _AIR): global_fractions for name in (*_ORGANICS, _PAH)
    }
    for name in (_PCB, _DIOXIN, "Hydrochloric acid"):
        if name in air:
            amounts[name, "kilogram", _AIR] = air[name]
            if name in {_PCB, _DIOXIN}:
                fractions[name, "kilogram", _AIR] = global_fractions
    # The UPR nominal/retained reference distinguishes total C and C traced by origin.
    # Use tracked burnable carbon for origin shares; leave source gas quantities unchanged.
    origin_groups: tuple[tuple[str, tuple[_CarbonOrigin, ...]], ...] = (
        ("fossil", ("fossil",)),
        ("non-fossil", ("biogenic", "atmospheric_dac")),
    )
    for label, origins in origin_groups:
        group_carbon = fsum(
            principal_carbon.get(origin, 0.0) + auxiliary_carbon.get(origin, 0.0)
            for origin in origins
        )
        group_fractions: dict[_CarbonOrigin, float] = (
            {origin: principal_carbon.get(origin, 0.0) / group_carbon for origin in origins}
            if group_carbon
            else {}
        )
        for name in ("Carbon dioxide", "Carbon monoxide", "Methane"):
            key = (f"{name}, {label}", "kilogram", _AIR)
            amounts[key] = air[name] * group_carbon / burnable_carbon
            fractions[key] = group_fractions

    soil_principal: dict[_CarbonOrigin, float] = {
        origin: carbon * _RESIDUE["C"] for origin, carbon in principal_carbon.items()
    }
    soil_auxiliary: dict[_CarbonOrigin, float] = {
        origin: carbon * _RESIDUE["C"] for origin, carbon in auxiliary_carbon.items()
    }
    soil_carbon = fsum((*soil_principal.values(), *soil_auxiliary.values()))
    soil_fractions: dict[_CarbonOrigin, float] = {
        origin: carbon / soil_carbon for origin, carbon in soil_principal.items()
    }
    # Both soil reference organics and the elemental complement share this post-transfer pool.
    # Proportional subtraction leaves the same fractions; the total-C indicator includes both.
    for element, name in _ELEMENT_NAMES.items():
        if element in soil_elements:
            amounts[name, "kilogram", _SOIL] = residue.get(element, 0.0) + inert.get(element, 0.0)
    fractions["Carbon", "kilogram", _SOIL] = soil_fractions
    for name, mass in soil_compounds.items():
        amounts[name, "kilogram", _SOIL] = mass
        fractions[name, "kilogram", _SOIL] = soil_fractions
    for element, mass in air_oxides.items():
        if element != "O":
            amounts[_ELEMENT_NAMES[element], "kilogram", _AIR] = mass
    pm = fsum(air_oxides.values()) + airborne.get("Cl", 0.0) * (1 + _RESIDUE_OXYGEN["Cl"])
    # Downard (2015), via Doka (2017), section 4: coarse .20, intermediate .28, fine .52.
    for name, share in (
        ("Particulate Matter, > 10 um", 0.20),
        ("Particulate Matter, > 2.5 um and < 10um", 0.28),
        ("Particulate Matter, < 2.5 um", 0.52),
    ):
        amounts[name, "kilogram", _AIR] = share * pm
    residue_mass = fsum(soil_elements.values())
    amounts["Water", "cubic meter", _AIR] = air["Water"] / 1000
    amounts["Oxygen", "kilogram", ("natural resource", "in air")] = oxygen
    amounts["Heat, waste", "megajoule", _AIR] = material.heat_potential_mj
    amounts["Non-hazardous waste disposed", "kilogram", _INDICATOR] = wet
    amounts["Waste mass, total, placed in landfill", "kilogram", _INDICATOR] = residue_mass
    amounts["Organic carbon, placed in landfill", "kilogram", _INDICATOR] = soil_elements["C"]
    fractions["Organic carbon, placed in landfill", "kilogram", _INDICATOR] = soil_fractions
    return _OpenBurningLoad(
        wet_mass_kg=wet,
        incoming_water_kg=water_kg,
        elements_kg=elements,
        principal_carbon_kg=principal_carbon,
        auxiliary_carbon_kg=auxiliary_carbon,
        air_compounds_kg=air,
        air_oxide_elements_kg=air_oxides,
        residue_elements_kg=soil_elements,
        formed_water_kg=formed_water,
        atmospheric_oxygen_kg=oxygen,
        atmospheric_nitrogen_kg=thermal_nox * 14 / 46 + air["Dinitrogen monoxide"] * 28 / 44,
        residue_mass_kg=residue_mass,
        heat_potential_mj=material.heat_potential_mj,
        biosphere_amounts=amounts,
        principal_origin_fractions=fractions,
        burnable_elements_kg=burnable,
        inert_elements_kg=inert,
        soil_compounds_kg=soil_compounds,
        source_rounding_kg=(
            wet
            - water_kg
            - fsum(burnable.values())
            - fsum(inert.values())
            - material.geological_carbon_kg
        ),
        geological_carbon_kg=material.geological_carbon_kg,
    )
