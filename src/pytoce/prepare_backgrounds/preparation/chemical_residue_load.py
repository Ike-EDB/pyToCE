"""Calculate complete chemical-residue fates from admitted native source values.

The actual wet load, element pools, chemicals, water, heat and ash share these
equations in preparation and in composition-dependent PA receiving. Brightway
lookup, source identity and exchange binding belong to chemical_residue_incineration.
"""

import dataclasses
from dataclasses import asdict, dataclass
from math import fsum
from typing import TypedDict, cast

from .hazardous_incineration import (
    LIME_ACID_WEIGHTS,
    NAOH_ACID_WEIGHTS,
    acid_neutralization_mass,
    reagent_element_masses,
    scrubber_capture_mass,
)

_SURFACE = ("water", "surface water")
_LONG = ("water", "ground-, long-term")
_URBAN = ("air", "urban air close to ground")
_DOC = "DOC, Dissolved Organic Carbon"
_CARBON_INDICATORS = {
    _DOC,
    "TOC, Total Organic Carbon",
    "COD, Chemical Oxygen Demand",
    "BOD5, Biological Oxygen Demand",
}
_HCL = "hydrochloric acid, without water, in 30% solution state"
_NAOH = "sodium hydroxide, without water, in 50% solution state"
_PEROXIDE = "hydrogen peroxide, without water, in 50% solution state"
_OIL_HEAT = "heat, central or small-scale, other than natural gas"
_GAS_HEAT = "heat, district or industrial, natural gas"
_AMMONIA = "ammonia, anhydrous, liquid"
_FERRIC = "iron(III) chloride, without water, in 40% solution state"
_TMT = "chemical, organic, basic precursors and materials"
# Doka's CH waste-treatment reference uses nominal C/H/O/N/S/P molar masses (g/mol). The admitted
# technical reagents retain the listed metal/halogen masses; caller-specific salt and PA auxiliary
# atoms retain their supplied values.
_ATOMS = {
    "C": 12.0,
    "H": 1.0,
    "O": 16.0,
    "N": 14.0,
    "S": 32.0,
    "P": 31.0,
    "Na": 22.99,
    "Cl": 35.45,
    "Ca": 40.078,
    "Ti": 47.867,
    "V": 50.942,
    "Fe": 55.845,
}
# Road-container glass, rutile and alumina introduce these known elements to the selected HWI. Doka
# (2003) II Table 4.1 reports independent, rounded air and solid fractions; direct-water fractions
# are zero. Section 4.1.5 applies oxide-equivalent terminal masses. Entries are native air name, air
# fraction, solid fraction and kg oxide O/kg element (TiO2, Al2O3, K2O, MgO). The conventional
# masses match road_material.
_ADDED_MINERAL_PHASES = {
    "Ti": ("Titanium ion", 0.001, 0.999, 32 / 47.867),
    "Al": ("Aluminium III", 0.00156, 0.998, 48 / (2 * 26.982)),
    "K": ("Potassium I", 0.00301, 0.997, 16 / (2 * 39.0983)),
    "Mg": ("Magnesium", 0.00138, 0.999, 16 / 24.305),
}
# Retained non-air fractions, kg non-air element/kg incoming element.
# Carbon and nitrogen fractions are assumed from Doka (2003). The captured
# element amounts scale HCl and polyacrylamide demand for hazardous waste.
_NONAIR = {
    "O": 1 - 0.917,
    "H": 0.0,
    "C": 1 - 0.9889898,
    "S": 1 - 0.0006,
    "N": 1 - 0.989,
    "P": 1 - 0.0007,
    "B": 1 - 0.12,
    "Cl": 1 - 0.0003,
    "F": 1 - 0.0005,
    "Cd": 1 - 0.00899,
    "Co": 1 - 0.0007,
    "Cr": 1 - 0.0000000739,
    "Cu": 1 - 0.0007,
    "Hg": 1 - 0.0402,
    "Ni": 1 - 0.0007,
    "Pb": 1 - 0.00331,
    "Zn": 1 - 0.0007,
    "Si": 1 - 0.00233,
    "Na": 1 - 0.00941,
    "Ca": 1 - 0.00167,
    **{element: 1 - air for element, (_, air, _, _) in _ADDED_MINERAL_PHASES.items()},
}
_FORMULAS = {
    "HCl": {"H": 1.0, "Cl": 1.0},
    "PAM": {"C": 3.0, "H": 5.0, "N": 1.0, "O": 1.0},
    "NaOH": {"Na": 1.0, "O": 1.0, "H": 1.0},
    "CaO": {"Ca": 1.0, "O": 1.0},
    "H2O2": {"H": 2.0, "O": 2.0},
    "TiO2": {"Ti": 1.0, "O": 2.0},
    "V2O5": {"V": 2.0, "O": 5.0},
    "FeCl3": {"Fe": 1.0, "Cl": 3.0},
    "CaCl2": {"Ca": 1.0, "Cl": 2.0},
    "TMT": {"Na": 3.0, "C": 3.0, "N": 3.0, "S": 3.0},
}
# Cation, existing anion flow, whole-salt air transfer, and short-term release. For PA diglycerol
# manufacture, Doka II Table4.1 supplies K entrainment .00301; Part III TableA8 supplies K
# short-term .2819, cumulative1. The sulfate follows that same salt.
_SALT_FATES = {
    "Potassium sulfate": ("K", "Sulfate", 0.00301, 0.2819),
}


@dataclass(frozen=True, slots=True)
class ChemicalResidueLoad:
    """Receiving reference per caller unit, before wet normalization.

    ``active_elements_kg`` contains known C/H/O/N/S/Na and, for PA castor-oil
    processing residues, Ni/Si; known phosphorus and calcium use the same pool.
    Road-container residues supply known Ti, Al, K, and Mg from their actual ingredients.
    It excludes the separately tracked nonfuel potassium sulfate
    and incoming carrier water. ``net_energy_mj`` includes the caller's admitted
    organic/carbon/metal oxidation heat less hydrogen/water vaporization;
    it also includes named inorganic reaction heat when present, but excludes
    intact nonfuel salts and unknown mineral. ``carrier_water_kg`` records incoming waste moisture.
    ``nonfuel_salt_elements_kg`` maps the admitted potassium sulfate reference to constituent
    element kg. This
    preserves each manufacturing owner's atomic-weight convention. Whole salt
    mass follows its declared air/terminal transfer, with no fuel assignment.
    ``waste_formed_water_kg`` is newly formed water from the waste only;
    incoming free or hydration water belongs to ``carrier_water_kg`` and its
    atoms are excluded from the active pool. All quantities use the same
    caller unit. ``biogenic_carbon_kg`` includes all biological waste C,
    both principal and auxiliary. ``principal_carbon_kg`` records separately
    tracked polymer carbon, in the package's fossil, biogenic, atmospheric-DAC
    order. DAC carbon uses the existing non-fossil gas flows alongside biomass;
    fixed fossil/biological auxiliaries keep their actual origins. Principal carbon records organic
    principal material. ``geological_carbon_kg`` separately records the actual road limestone carbon
    excluded from Doka's organic-carbon parameter. It stays inside wet input mass but
    supplies no organic gas, DOC or heat. Other limestone atoms enter the
    known element pool.
    """

    wet_mass_kg: float
    carrier_water_kg: float
    active_elements_kg: dict[str, float]
    net_energy_mj: float
    nonfuel_salt_elements_kg: dict[str, dict[str, float]]
    waste_formed_water_kg: float = dataclasses.field(kw_only=True)
    biogenic_carbon_kg: float = 0.0
    unknown_mineral_kg: float = 0.0
    principal_carbon_kg: tuple[float, float, float] = dataclasses.field(
        default=(0.0, 0.0, 0.0), kw_only=True
    )
    geological_carbon_kg: float = dataclasses.field(default=0.0, kw_only=True)


@dataclass(frozen=True, slots=True)
class ChemicalResidueSource:
    """Native operating values admitted once for an actual residue calculation.

    Waste and fixed supplies/releases are normalized per positive kg of native
    waste throughput and retain native supplier/flow units. Donor elements are
    kg/kg wet waste. Boiler inputs are kg oil and kWh electricity per delivered MJ. Integrated PA
    cleaning records zero for both boiler duties.
    Native identities and uncertainty remain with the preparation binding owner.
    """

    donor_elements_kg: dict[str, float]
    waste_supplies: dict[str, float]
    waste_releases: dict[tuple[str, tuple[str, ...]], float]
    fixed_supplies: dict[str, float]
    fixed_releases: dict[tuple[str, tuple[str, ...]], float]
    oil_kg_per_mj: float
    boiler_electricity_kwh_per_mj: float


@dataclass(frozen=True, slots=True)
class ChemicalResidueAshSource:
    """Admitted ash composition and native releases for the receiving equations.

    Dry fraction is kg dry/kg received ash; elemental contents are kg element
    per dry kg. Releases retain native flow units per positive kg of received
    ash. Names and compartments identify the existing source rows; Brightway
    identities, row ordering and uncertainty remain with the binding owner.
    """

    dry_fraction: float
    element_contents: dict[str, float]
    releases: dict[tuple[str, tuple[str, ...]], float]


class _ChemicalResidueReference(TypedDict):
    """Actual residue outputs before the binding owner's kg normalization.

    Element and reagent mappings use kg; native gas names use kg compound.
    Quantities with explicit unit suffixes retain those units. ``load`` is the
    declared input serialized for provenance.
    """

    load: dict[str, object]
    donor: dict[str, float]
    supplies: dict[str, float]
    fresh: dict[str, dict[str, float]]
    gases: dict[str, float]
    ash: dict[str, float]
    ash_mass_kg: float
    nonfuel_salt_elements_kg: dict[str, dict[str, float]]
    residual_doc_C_kg: dict[str, float]
    nitrogen_ratio: float
    site_water_kg: float
    formed_water_kg: float
    water_air_m3: float
    water_surface_m3: float
    oil_heat_mj: float
    oil_kg: float
    fuel_ncv_deficit_mj: float
    net_electricity_mj: float
    waste_heat_mj: float
    pooled_N2_N_kg: float
    principal_carbon_outputs_kg: dict[str, tuple[float, float, float]]
    principal_gas_origin_fractions: dict[str, tuple[float, float, float]]


def _residue_ash_release(
    name: str,
    categories: tuple[str, ...],
    native_amount: float,
    source: ChemicalResidueAshSource,
    reference: _ChemicalResidueReference,
) -> float | None:
    """Calculate one native terminal release per kg actual chemical-residue ash.

    ``native_amount`` uses the existing flow unit per positive kg source ash.
    Fresh reagent atoms, actual residual organic carbon and known waste metals
    replace source composition. ``None`` denotes a source element with no duty
    at this terminal; the furnace owns the original unknown-mineral function.
    Additional salt and titanium endpoints have their own calculation below.
    """
    b = reference["ash_mass_kg"]
    ash, fresh = reference["ash"], reference["fresh"]
    dry = source.dry_fraction
    if name == "Organic carbon, placed in landfill":
        return ash["C"] / b
    if categories[0] == "inventory indicator":
        return native_amount
    if name == "Nitrate":
        # Fresh PAM/TMT nitrogen enters this ash once; original waste nitrogen already has its
        # complete HWI release. Native kg nitrate/kg ash divided by kg N/kg ash retains each
        # source-period coefficient and its molecular conversion at the source's rounded release
        # fractions.
        return (
            native_amount
            * (fresh["PAM"]["N"] + fresh["TMT"]["N"])
            / (dry * source.element_contents["nitrogen"])
            / b
        )
    if name == "Sulfate":
        # Only fresh TMT ligand S is new here. Original waste S is already
        # in HWI; intact salts follow their separate co-release. The native
        # sulfate coefficient already includes conversion from elemental S.
        return native_amount * fresh["TMT"]["S"] / (dry * source.element_contents["sulfur"]) / b
    if name == "Phosphate":
        # Native kg phosphate/kg ash divided by kg P/kg ash retains the source's molecular
        # conversion.
        return native_amount * ash["P"] / (dry * source.element_contents["phosphorus"]) / b
    element = "carbon" if name in _CARBON_INDICATORS else name.split()[0].lower()
    symbol = {
        "carbon": "C",
        "sodium": "Na",
        "calcium": "Ca",
        "vanadium": "V",
        "iron": "Fe",
        "nickel": "Ni",
        "silicon": "Si",
        "aluminium": "Al",
        "potassium": "K",
        "magnesium": "Mg",
    }.get(element)
    if symbol is None:
        return None
    target = fresh["PAM"]["C"] + fresh["TMT"]["C"] if symbol == "C" else ash[symbol]
    amount = native_amount * target / (dry * source.element_contents[element]) / b
    if symbol == "C":
        # .0000101*C is direct HWI water, removed from this residual by _residue_reference.
        # DOC/TOC/COD/BOD describe overlapping quantities.
        carbon = reference["residual_doc_C_kg"]["surface" if categories == _SURFACE else "long"]
        amount += carbon * (32 / 12 if name.startswith(("BOD5", "COD,")) else 1) / b
    return amount


def _residue_ash_additions(
    reference: _ChemicalResidueReference,
) -> tuple[tuple[str, tuple[str, ...], float], ...]:
    """Calculate admitted titanium and intact-salt releases in kg/kg actual ash."""
    b, ash = reference["ash_mass_kg"], reference["ash"]
    rows: list[tuple[str, tuple[str, ...], float]] = []
    # Doka III Table A8 Ti fractions: .0004956 short-term and .2973 cumulative.
    # Use only the additional long-term part in that second native compartment.
    for categories, titanium in ((_SURFACE, 0.0004956), (_LONG, 0.2973 - 0.0004956)):
        rows.append(("Titanium ion", categories, ash["Ti"] * titanium / b))
        for salt, atoms in reference["nonfuel_salt_elements_kg"].items():
            cation, anion, _, short = _SALT_FATES[salt]
            fraction = short if categories == _SURFACE else 1 - short
            rows.extend(
                (
                    (
                        "Potassium I",
                        categories,
                        atoms[cation] * fraction / b,
                    ),
                    (
                        anion,
                        categories,
                        fsum(mass for element, mass in atoms.items() if element != cation)
                        * fraction
                        / b,
                    ),
                )
            )
    return tuple(rows)


def _residue_waste_release(
    name: str,
    categories: tuple[str, ...],
    native_amount: float,
    load: ChemicalResidueLoad,
    source: ChemicalResidueAshSource,
    reference: _ChemicalResidueReference,
) -> float | None:
    """Calculate one HWI release in native units per actual received batch.

    The source amount is per positive kg native waste. Actual C/N/S/P/Ni
    replaces donor chemistry; unidentified mineral retains its source function.
    Water is emitted separately from actual incoming and formed quantities.
    """
    if name in reference["gases"] and categories == _URBAN:
        return reference["gases"][name]
    if name in _CARBON_INDICATORS:
        # Assume direct-water release of .0000101 of incoming organic C, based
        # on Doka (2003); the ash treatment receives the remaining leachable C.
        amount = 0.0000101 * load.active_elements_kg["C"] if categories == _SURFACE else 0.0
        return amount * (32 / 12) if name.startswith(("BOD5", "COD,")) else amount
    if name == "Nitrate":
        return native_amount * reference["nitrogen_ratio"]
    if name in {"Phosphorus", "Phosphate"}:
        # Native combined direct/ash releases retain the unknown-mineral
        # function. Known waste P uses only Doka's .0007 direct P and .0153
        # aqueous phosphate here; .984 solid P belongs to actual ash.
        amount = native_amount * load.unknown_mineral_kg
        phosphorus = load.active_elements_kg.get("P", 0.0)
        if name == "Phosphorus":
            amount += 0.0007 * phosphorus
        elif categories == _SURFACE:
            amount += 0.0153 * phosphorus * 95 / 31
        return amount
    if name == "Sulfate":
        # Doka II Table 4.1: .0428 S enters ash and .957 direct scrubber
        # sulfate; nominal SO4/S=96/32=3 applies only to the direct term.
        donor_s = source.dry_fraction * source.element_contents["sulfur"]
        amount = (
            source.releases[name, categories] * 0.0428 * load.active_elements_kg["S"] / donor_s
            + native_amount * load.unknown_mineral_kg
        )
        if categories == _SURFACE:
            amount += 3 * 0.957 * load.active_elements_kg["S"]
        return amount
    if name == "Hazardous waste disposed":
        return load.wet_mass_kg
    if name == "Water":
        return None
    amount = native_amount * load.unknown_mineral_kg
    if name == "Nickel II" and categories == _SURFACE:
        # Doka II Table 4.1: .205 Ni reaches direct water; its independently
        # rounded .795 solid fraction is received by the actual ash child.
        amount += 0.205 * load.active_elements_kg.get("Ni", 0.0)
    return amount


def _residue_added_releases(
    load: ChemicalResidueLoad,
    ref: _ChemicalResidueReference,
) -> tuple[tuple[str, str, tuple[str, ...], float], ...]:
    """Calculate added native-flow quantities per actual chemical-residue batch.

    Names, units and compartments describe existing ecoinvent endpoints.
    Salt entrainment, actual water, known elements and waste heat complete
    the native-row quantities.
    """
    rows: list[tuple[str, str, tuple[str, ...], float]] = []
    # Only the admitted PA potassium-sulfate reference entrains intact salt here. Its two
    # existing elemental flows share one fraction.
    for salt, atoms in load.nonfuel_salt_elements_kg.items():
        cation, anion, air, _ = _SALT_FATES[salt]
        for name, amount in (
            ("Potassium I", atoms[cation]),
            (anion, fsum(value for element, value in atoms.items() if element != cation)),
        ):
            rows.append((name, "kilogram", _URBAN, amount * air))
    # Apply Doka's average HWI air fractions to the road elements.
    # Fresh air-pollution-control and catalyst atoms enter ash separately.
    rows.extend(
        (name, "kilogram", _URBAN, air * load.active_elements_kg[element])
        for element, (name, air, _, _) in _ADDED_MINERAL_PHASES.items()
        if load.active_elements_kg.get(element, 0.0) != 0.0
    )
    # Doka part II Section 4.3 (p. 59), referring to Section 3.2.10 (p. 38): emit 83% of waste-heat
    # MJ to air and 17% to water.
    for name, unit, cats, amount in (
        ("Water", "cubic meter", ("air",), ref["water_air_m3"]),
        ("Water", "cubic meter", ("water",), ref["water_surface_m3"]),
        ("Sodium I", "kilogram", _URBAN, ref["gases"]["Sodium I"]),
        # Doka's calcium air transfer uses the existing ecoinvent
        # Calcium II/urban-air flow.
        ("Calcium II", "kilogram", _URBAN, 0.00167 * load.active_elements_kg.get("Ca", 0.0)),
        # PA dimerization lithium follows the complete-dissolution assumption
        # at wet cleaning. Phosphorus uses the Doka capture and ash functions.
        ("Lithium I", "kilogram", _SURFACE, load.active_elements_kg.get("Li", 0.0)),
        ("Sodium I", "kilogram", _SURFACE, ref["fresh"]["NaOH"]["Na"] + ref["fresh"]["TMT"]["Na"]),
        (
            "Chloride",
            "kilogram",
            _SURFACE,
            fsum(ref["fresh"][k]["Cl"] for k in ("HCl", "FeCl3", "CaCl2")),
        ),
        # Ecoinvent 3.12 supplies N2 to unspecified air. This is the exact selected elemental-N
        # endpoint.
        ("Nitrogen", "kilogram", ("air",), ref["pooled_N2_N_kg"]),
        ("Heat, waste", "megajoule", _URBAN, 0.83 * ref["waste_heat_mj"]),
        ("Heat, waste", "megajoule", _SURFACE, 0.17 * ref["waste_heat_mj"]),
    ):
        if amount != 0:
            rows.append((name, unit, cats, amount))
    return tuple(rows)


def _residue_reference(
    load: ChemicalResidueLoad,
    source: ChemicalResidueSource,
    *,
    integrated_pa_site: bool,
) -> _ChemicalResidueReference:
    """Apply Doka's receiving functions to known waste and fresh reagent pools.

    Use the admitted source values. For off-site treatment, the linked boiler
    converts the fuel-NCV deficit into delivered MJ. Recovered electricity
    includes supporting fuel. The complete boiler owns fuel supply and combustion
    once. Integrated PA cleaning uses the spinning site's existing energy and
    water functions.
    """
    donor = source.donor_elements_kg
    elements = load.active_elements_kg
    # Li follows the aqueous reference. Total P retains its reagent
    # allocation and terminal treatment.
    capture = scrubber_capture_mass(
        {el: value for el, value in elements.items() if el != "Li"}, _NONAIR
    ) / scrubber_capture_mass(
        {el: donor[el] for el in _NONAIR if el not in {"Na", "Ca", *_ADDED_MINERAL_PHASES}},
        _NONAIR,
    )
    # Known target Na/Ca/Ti/Al/K/Mg enter the captured-mass calculation.
    # The native denominator provides the reagent-allocation proxy.
    acid_elements = {el: elements.get(el, 0.0) for el in NAOH_ACID_WEIGHTS}
    # PA filter aid uses a kg-equivalent treatment proxy. Its reagent and
    # trace-emission functions remain separate from oil, C, Ni and silica.
    unknown = load.unknown_mineral_kg
    nr = elements["N"] / donor["N"] + unknown
    metal_ratio = (
        elements.get("Ni", 0.0)
        * _NONAIR["Ni"]
        / fsum(donor[el] * _NONAIR[el] for el in ("Cd", "Co", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn"))
    )
    m = load.wet_mass_kg
    q = {
        "HCl": source.waste_supplies[_HCL] * (capture + unknown),
        "PAM": source.waste_supplies["chemical, inorganic"] * (capture + unknown),
        "NaOH": source.waste_supplies[_NAOH]
        * (
            acid_neutralization_mass(acid_elements, NAOH_ACID_WEIGHTS)
            / acid_neutralization_mass(donor, NAOH_ACID_WEIGHTS)
            + unknown
        ),
        "CaO": source.waste_supplies["quicklime, milled, packed"]
        * (
            acid_neutralization_mass(acid_elements, LIME_ACID_WEIGHTS)
            / acid_neutralization_mass(donor, LIME_ACID_WEIGHTS)
            + unknown
        ),
        "H2O2": source.waste_supplies[_PEROXIDE] * (elements["S"] / donor["S"] + unknown),
        "FeCl3": source.waste_supplies[_FERRIC] * (metal_ratio + unknown),
        # Doka II section3.2.8/Table3.3: calcium chloride precipitates captured
        # phosphorus (3.47 kg active/kg captured P). Scaling the selected
        # native dose preserves its own P reference and supply uncertainty.
        "CaCl2": source.waste_supplies["calcium chloride"]
        * (elements.get("P", 0.0) / donor["P"] + unknown),
        "TMT": source.waste_supplies[_TMT] * unknown,
        "NH3_waste": source.waste_supplies[_AMMONIA] * nr,
        "NH3_fixed": source.fixed_supplies[_AMMONIA] * m,
        "NG_waste": source.waste_supplies[_GAS_HEAT] * nr,
        "NG_fixed": source.fixed_supplies[_GAS_HEAT] * m,
        "TiO2": source.fixed_supplies["titanium dioxide"] * m
        + source.waste_supplies["titanium dioxide"] * nr,
        "V2O5": source.fixed_supplies["chromium"] * m + source.waste_supplies["chromium"] * nr,
    }
    fresh = reagent_element_masses(q, _FORMULAS, _ATOMS)
    # Doka part II Table 4.1 separates air, water and solid elemental fates.
    # Native CO scales with wet kg; nitrogen species scale with incoming N.
    # Nominal molecular ratios convert element kg to emitted compound kg;
    # CO/CN carbon is subtracted from the air-carbon pool before forming CO2.
    carbon = elements["C"]
    # Carbon-free phosphorus/mineral feeds cannot emit target CO or cyanide.
    # Their complete supporting boiler still owns its fuel combustion. For
    # carbon-bearing residues retain the admitted native operating functions.
    co = source.waste_releases["Carbon monoxide, fossil", _URBAN] * m if carbon else 0.0
    cyanide = source.waste_releases["Cyanide", _URBAN] * nr if carbon else 0.0
    gases = {
        name: source.waste_releases[name, _URBAN] * nr
        for name in ("Ammonia", "Dinitrogen monoxide", "Nitrogen oxides")
    }
    # Actual waste origins share each organic gas homogeneously. DAC and biological waste C use the
    # existing non-fossil elementary flows.
    nonfossil_share = (
        (load.biogenic_carbon_kg + load.principal_carbon_kg[2]) / carbon if carbon else 0.0
    )
    co2 = (0.9889898 * carbon - co * 12 / 28 - cyanide * 12 / 26) * 44 / 12
    gases.update(
        {
            "Carbon monoxide, fossil": co * (1 - nonfossil_share),
            "Carbon monoxide, non-fossil": co * nonfossil_share,
            "Cyanide": cyanide,
            "Carbon dioxide, fossil": co2 * (1 - nonfossil_share),
            "Carbon dioxide, non-fossil": co2 * nonfossil_share,
            "Sulfur dioxide": 0.0006 * elements["S"] * 2
            + source.waste_releases["Sulfur dioxide", _URBAN] * unknown,
            "Sodium I": 0.00941 * elements["Na"],
            "Nickel II": 0.0007 * elements.get("Ni", 0.0)
            + source.waste_releases["Nickel II", _URBAN] * unknown,
            "Silicon": 0.00233 * elements.get("Si", 0.0)
            + source.waste_releases["Silicon", _URBAN] * unknown,
        }
    )
    ammonia = q["NH3_fixed"] + q["NH3_waste"]
    ammonia_air = gases["Ammonia"] + source.fixed_releases["Ammonia", _URBAN] * m
    # The waste owner separates molecular reaction water and incoming moisture.
    # Fresh reacted NH3 retains the source's nominal 9 kg H2O/kg H.
    # Native flue-gas-cleaning water retains its existing operating meter.
    formed = load.waste_formed_water_kg + 9 * (ammonia - ammonia_air) * 3 / 17
    ash = {
        "C": 0.0110001 * carbon + fresh["PAM"]["C"] + fresh["TMT"]["C"],
        "N": 0.01 * elements["N"] + fresh["PAM"]["N"] + fresh["TMT"]["N"],
        "S": 0.0428 * elements["S"] + fresh["TMT"]["S"],
        # Table4.1 independently rounds Na solid transfer to .991; the independent air coefficient
        # is .00941.
        "Na": 0.991 * elements["Na"],
        # Doka II Table4.1: P .0007 air/.0153 water/.984 solid; Ca .00167 air/zero direct water/.998
        # solid. Retain the printed solid fraction.
        "P": 0.984 * elements.get("P", 0.0),
        "Ca": 0.998 * elements.get("Ca", 0.0) + fresh["CaO"]["Ca"] + fresh["CaCl2"]["Ca"],
        "Fe": fresh["FeCl3"]["Fe"],
        # Doka II Table4.1 supplies rounded Ni/Si solid fractions .795/.998;
        # preserve them alongside the independent air/direct-water coefficients.
        "Ni": 0.795 * elements.get("Ni", 0.0),
        "Si": 0.998 * elements.get("Si", 0.0),
        "Ti": fresh["TiO2"]["Ti"],
        "V": fresh["V2O5"]["V"],
        "H": fresh["PAM"]["H"],
    }
    for element, (_, _, solid, _) in _ADDED_MINERAL_PHASES.items():
        # Fresh TiO2 was already added above. Actual waste Ti and the other
        # known mineral atoms use their independent printed solid fractions.
        ash[element] = ash.get(element, 0.0) + solid * elements.get(element, 0.0)
    # Doka (2003) separates waste oxygen from oxide-equivalent terminal mass.
    # Record O in NO3/SO3/Na2O/P2O5/CaO/metal-oxide equivalents and PAM once.
    ash["O"] = (
        fresh["PAM"]["O"]
        + 0.01 * elements["N"] * 48 / 14
        + 0.0428 * elements["S"] * 48 / 32
        + ash["Na"] * 16 / (2 * _ATOMS["Na"])
        + ash["P"] * 80 / (2 * _ATOMS["P"])
        + ash["Ca"] * 16 / _ATOMS["Ca"]
        + ash["Ni"] * 15.999 / 58.6934
        + ash["Si"] * 2 * 15.999 / 28.085
        + ash["Fe"] * 24 / _ATOMS["Fe"]
        + fresh["TiO2"]["O"]
        + fresh["V2O5"]["O"]
        + fsum(
            solid * elements.get(element, 0.0) * oxygen
            for element, (_, _, solid, oxygen) in _ADDED_MINERAL_PHASES.items()
        )
    )
    # Doka (2003) oxidation gives nitrate-, SO3- and Na2O-equivalent terminal
    # service kg, with fresh solids and intact K2SO4. ash_mass_kg uses this basis.
    b = (
        0.0110001 * carbon
        + 0.01 * elements["N"] * 62 / 14
        + 0.0428 * elements["S"] * 80 / 32
        + ash["Na"] * (1 + 16 / (2 * 22.99))
        + ash["P"] * (1 + 80 / (2 * _ATOMS["P"]))
        + 0.998 * elements.get("Ca", 0.0) * (1 + 16 / _ATOMS["Ca"])
        + fsum(q[k] for k in ("PAM", "CaO", "TiO2", "V2O5"))
        + fsum(
            mass * (1 - _SALT_FATES[name][2])
            for name, salt in load.nonfuel_salt_elements_kg.items()
            for mass in salt.values()
        )
        + unknown
        + ash["Ni"] * (1 + 15.999 / 58.6934)
        + ash["Si"] * (1 + 2 * 15.999 / 28.085)
        + ash["Fe"] * (1 + 24 / _ATOMS["Fe"])
        + fresh["CaCl2"]["Ca"] * (1 + 16 / _ATOMS["Ca"])
        + fsum(fresh["TMT"][el] for el in ("C", "N", "S"))
        + fsum(
            solid * elements.get(element, 0.0) * (1 + oxygen)
            for element, (_, _, solid, oxygen) in _ADDED_MINERAL_PHASES.items()
        )
    )
    # Unknown filter aid was already dry mineral: preserve its entire kg in
    # the terminal. Applying the mixed-waste donor's .18901 ash yield would
    # discard most of it. NiO/SiO2 use the catalyst's CIAAW atom convention;
    # Fe2O3 and CaO follow the original receiving function's nominal factors.
    site_water = 0.0
    if integrated_pa_site:
        # PA6/PA66 spin-pack cleaning is inside the BREF spinning site. Its purchased energy and
        # factory already cover this operating duty. Only the actual deposited material's admitted
        # NCV becomes receiving waste heat; the site's purchased energy retains its own boundary.
        q["NG_fixed"] = q["NG_waste"] = 0.0
        oil = heat = deficit = net_electricity = 0.0
        waste_heat = load.net_energy_mj
        # Native fixed control-water m3/kg is allocated from the PA site's water meter (4.3 kg/kg in
        # the original source). Dated coefficients define the actual demand.
        site_water = (
            1000
            * m
            * source.fixed_releases["Water, well, in ground", ("natural resource", "in water")]
        )
    else:
        oil = source.oil_kg_per_mj
        electricity = source.boiler_electricity_kwh_per_mj
        # Doka part II Section 4.1.2, Eq. 4.2: 23 MJ/kg wet minimum combustion input and 42.6 MJ/kg
        # light-oil NCV. The actual boiler's kg oil/MJ converts the fuel-NCV deficit to purchased
        # heat.
        deficit = max(23 * m - load.net_energy_mj, 0.0)
        heat = deficit / (oil * 42.6)
        # Section 4.3, pp. 58-59, Table 4.9: gross electricity recovers 10% of
        # waste plus support-fuel MJ; plant own use is 0.287 kWh/kg wet. No export
        # credit. Delivered reheat and boiler auxiliary electricity also become heat.
        total_energy = load.net_energy_mj + deficit
        net_electricity = 0.1 * total_energy - 0.287 * 3.6 * m
        waste_heat = (
            total_energy
            - net_electricity
            + q["NG_fixed"]
            + q["NG_waste"]
            + 3.6 * electricity * heat
        )
    residual_doc = {
        "surface": source.waste_releases[_DOC, _SURFACE] * carbon / donor["C"] - 0.0000101 * carbon,
        "long": source.waste_releases[_DOC, _LONG] * carbon / donor["C"],
    }
    principal_outputs: dict[str, tuple[float, float, float]] = {}
    gas_fractions: dict[str, tuple[float, float, float]] = {}
    if any(load.principal_carbon_kg):
        # Follow waste-derived C through the same Doka air/direct-water/ash transfers. The ash's
        # later DOC releases draw from its captured C; DOC/TOC/COD/BOD describe overlapping
        # quantities from this pool. Fresh PAM/TMT carbon remains outside this principal ledger
        # while its full environmental burdens stay in LCI.
        for destination, amount in {
            "carbon_dioxide": co2 * 12 / 44,
            "carbon_monoxide": co * 12 / 28,
            "cyanide": cyanide * 12 / 26,
            "direct_surface_water": 0.0000101 * carbon,
            "ash_surface_water": residual_doc["surface"],
            "ash_groundwater": residual_doc["long"],
            "retained_in_ash": 0.0110001 * carbon - fsum(residual_doc.values()),
        }.items():
            principal_outputs[destination] = cast(
                tuple[float, float, float],
                tuple(amount * part / carbon for part in load.principal_carbon_kg),
            )
        for species, organic_amount in (("Carbon dioxide", co2), ("Carbon monoxide", co)):
            for suffix, origin_indices in (("fossil", (0,)), ("non-fossil", (1, 2))):
                name = f"{species}, {suffix}"
                if gases[name] != 0:
                    # Fractions refer to this emitted origin group.
                    gas_fractions[name] = cast(
                        tuple[float, float, float],
                        tuple(
                            organic_amount * part / carbon / gases[name]
                            if index in origin_indices
                            else 0.0
                            for index, part in enumerate(load.principal_carbon_kg)
                        ),
                    )
    return {
        "load": asdict(load),
        "donor": donor,
        "supplies": q,
        "fresh": fresh,
        "gases": gases,
        "ash": ash,
        "ash_mass_kg": b,
        "nonfuel_salt_elements_kg": {
            name: {element: mass * (1 - _SALT_FATES[name][2]) for element, mass in atoms.items()}
            for name, atoms in load.nonfuel_salt_elements_kg.items()
        },
        "residual_doc_C_kg": residual_doc,
        "nitrogen_ratio": nr,
        "site_water_kg": site_water,
        "formed_water_kg": formed,
        "water_air_m3": load.carrier_water_kg / 1000
        + source.fixed_releases["Water", ("air",)] * m
        + formed / 1000,
        "water_surface_m3": source.fixed_releases["Water", ("water",)] * m,
        "oil_heat_mj": heat,
        "oil_kg": oil * heat,
        "fuel_ncv_deficit_mj": deficit,
        "net_electricity_mj": net_electricity,
        "waste_heat_mj": waste_heat,
        "pooled_N2_N_kg": 0.989 * (elements["N"] + unknown * donor["N"])
        - gases["Dinitrogen monoxide"] * 28 / 44
        - gases["Nitrogen oxides"] * 14 / 46
        - cyanide * 14 / 26
        + (ammonia - ammonia_air) * 14 / 17,
        "principal_carbon_outputs_kg": principal_outputs,
        "principal_gas_origin_fractions": gas_fractions,
    }
