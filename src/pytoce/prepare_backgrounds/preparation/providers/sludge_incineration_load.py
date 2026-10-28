"""Calculate complete no-FLUWA landfill-leachate sludge incineration.

TPS residues and filled SBR scrap use their original municipal/plastic source
factors. Actual PA sanitary sludge selects the published P phases
when its plastic-mixture reference has no P rows. Swiss FLUWA is calculated
separately. Doka (2021), WWcalc/sludgeIncin, and Doka (2003), report 13-II, supply
the operating, elemental, nominal thermal and ash equations. This module
accesses no native inventory; source admission and exchange compilation are separate.
"""

from dataclasses import dataclass
from math import fsum

from .ash_treatment import _AshLoad, _TreatmentCarbon
from .leachate_load import _WastewaterLoad

_URBAN = ("air", "urban air close to ground")
_SURFACE = ("water", "surface water")
_AIR_OXYGEN = ("natural resource", "in air")
_Emission = tuple[str, str, tuple[str, ...]]

# Doka2021 sludgeIncin rows21:61: kg element/kg incoming element in slag,
# boiler ash, ESP ash, scrubber sludge, treated water, air; final entry is
# kg oxide-equivalent allocand/kg element. No FLUWA acid-wash redistribution.
# These transfers also define donor denominators. Municipal sludge omits
# Cl/Br/F/I and
# Ag/Be/Sc/Sr/Ti/Tl/W properties; the plastic donor includes Be/Sr/Ti/Tl.
# Only actual supplied properties contribute to each donor denominator.
# fmt: off
_TRANSFER: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "O": (  # sludgeIncin row21
        0.024545454545454502, 0.0022996632996633, 0.0114983164983165,
        0.00105050505050505, 0, 0.960606060606061,
        1,
    ),
    "H": (  # sludgeIncin row22
        0, 0, 0,
        0, 0, 1,
        1,
    ),
    "C": (  # sludgeIncin row23
        0.004011064312571921, 0, 0.00100402110452362,
        0.000502010552261809, 0.000502010552261809, 0.993980893478381,
        1,
    ),
    "S": (  # sludgeIncin row24
        0.5480341642763491, 0, 0.308400217317315,
        0.0584896963877666, 0.079758676892409, 0.0053172451261606,
        2.5,
    ),
    "N": (  # sludgeIncin row25
        0.009951818307848415, 0, 6.03517516203356e-05,
        0, 0.000987829940531229, 0.989,
        4.428571428571429,
    ),
    "P": (  # sludgeIncin row26
        0.8536803006664344, 0.0292960116233505, 0.110153003703798,
        0.00574398019829103, 0.000110461157659443, 0.00101624265046688,
        2.2915724895059735,
    ),
    "B": (  # sludgeIncin row27
        0.8787121287871212, 0.0307342147141218, 0.0901536964947573,
        0.0003999600039996, 0, 0,
        3.222222222222222,
    ),
    "Cl": (  # sludgeIncin row28
        0.01098901098901106, 0, 0.321900321900322,
        0.000555000555000555, 0.666000666000666, 0.000555000555000555,
        1,
    ),
    "As": (  # sludgeIncin row33
        0.2999070716715154, 0.0504918513880302, 0.64107820645669,
        0.000689614998619893, 3.52210539301733e-11, 0.00783325544992373,
        1.3203416978109983,
    ),
    "Ba": (  # sludgeIncin row34
        0.4995951674548841, 0.0901977703509225, 0.368990878708319,
        0.0372399522169023, 5.11691825893681e-05, 0.00392506208638238,
        1.1164991990680064,
    ),
    "Cd": (  # sludgeIncin row35
        0.0666725617480568, 0, 0.927669672665904,
        0.00509708611354892, 5.09708611354892e-05, 0.000509708611354892,
        1.1423487544483986,
    ),
    "Co": (  # sludgeIncin row36
        0.7768726953957842, 0.014061109542864, 0.168733314514368,
        0.0385042693786036, 9.93302655350391e-06, 0.00181867814182699,
        1.2715085694892245,
    ),
    "Cr": (  # sludgeIncin row37
        0.5343728679901094, 0.0309423964212966, 0.433193549898152,
        0.000540328062714814, 7.76472767878532e-05, 0.000873210350939887,
        1.4615384615384615,
    ),
    "Cu": (  # sludgeIncin row38
        0.8889923182044733, 0, 0.109908595837155,
        0.000915904965309626, 9.15904965309626e-05, 9.15904965309626e-05,
        1.2518098835379288,
    ),
    "Hg": (  # sludgeIncin row39
        0.019265335054808763, 0.00212882988002605, 0.372545229004559,
        0.577200577200577, 0.00050631629578998, 0.0283537125642389,
        1.079764694152251,
    ),
    "Mn": (  # sludgeIncin row40
        0.6137593449508394, 0.0290560438514833, 0.348672526217799,
        0.00820218232395907, 0.000118925172179846, 0.000190977483738632,
        1.2912373948815028,
    ),
    "Mo": (  # sludgeIncin row41
        0.7598732305834627, 0.0376933865353016, 0.197441548518246,
        0.00362341402452563, 5.26315514793685e-05, 0.00131578878698421,
        1.5003126954346466,
    ),
    "Ni": (  # sludgeIncin row42
        0.9387880845776044, 0.0214613748129773, 0.0383238529354955,
        0.000125947508442985, 4.45167365402162e-05, 0.00125622342893965,
        1.2725259751320048,
    ),
    "Pb": (  # sludgeIncin row43
        0.48080856453377674, 0, 0.510788146602759,
        0.00823851849359288, 8.23851849359288e-05, 8.23851849359288e-05,
        1.0772238042376563,
    ),
    "Sb": (  # sludgeIncin row44
        0.022707626224371613, 0.0212039481580293, 0.95608602244554,
        8.01057352953746e-07, 8.01057352953746e-07, 8.01057352953746e-07,
        1.1971252566735113,
    ),
    "Se": (  # sludgeIncin row45
        0.03157894736842093, 0.134730931762649, 0.747021775291268,
        0.0747368421052632, 1.59488853909816e-06, 0.0119299085838604,
        1.2026342451874368,
    ),
    "Sn": (  # sludgeIncin row46
        0.3729206491002469, 0.0264962278006861, 0.600581163482219,
        6.53205615919693e-07, 6.53205615919693e-07, 6.53205615919693e-07,
        1.269609908164125,
    ),
    "V": (  # sludgeIncin row47
        0.4264836544756224, 0.0496307474012121, 0.446676726610909,
        0.0559479167347435, 5.20485261509757e-05, 0.0212089062513626,
        1.7852375343541422,
    ),
    "Zn": (  # sludgeIncin row48
        0.29548378495256766, 0, 0.697076266434525,
        0.00670265640802428, 6.70265640802428e-05, 0.000670265640802428,
        1.2447605935444392,
    ),
    "Be": (  # sludgeIncin row49; plastic-leachate sludge donor property.
        0.8659418414801368, 0.0336020628785258, 0.0985660511103424,
        0.00189004453099505, 0, 0, 2.7777777777777777,
    ),
    "Sr": (  # sludgeIncin row51; plastic-leachate sludge donor property.
        0.999, 0, 0, 0.0009, 0, 0.0001, 1.1826067107966218,
    ),
    "Ti": (  # sludgeIncin row52; plastic-leachate sludge donor property.
        0.7850194831678489, 0.0505448953678714, 0.153832290250043,
        0.00951062150895247, 0.000263059743864643, 0.000829649961419258,
        1.6683375104427733,
    ),
    "Tl": (  # sludgeIncin row53; plastic-leachate sludge donor property.
        0.34808399999999995, 0, 0.649916, 0.001, 0, 0.001, 1.0391421987151572,
    ),
    "Si": (  # sludgeIncin row55
        0.9065441558582809, 0.0302342025710159, 0.0632169690121242,
        1.55751952638567e-06, 1.55751952638567e-06, 1.55751952638567e-06,
        2.139357687103895,
    ),
    "Fe": (  # sludgeIncin row56
        0.9695135624996827, 0, 0.0286745635180226,
        0.00179216021987641, 1.79216021987641e-05, 1.79216021987641e-06,
        1.4297378599054578,
    ),
    "Ca": (  # sludgeIncin row57
        0.7245544621289884, 0.0675657347251541, 0.184270185614057,
        0.00786987251060033, 0.00786987251060033, 0.00786987251060033,
        1.3992015968063873,
    ),
    "Al": (  # sludgeIncin row58
        0.8198062202379207, 0.044360197786882, 0.13500929761225,
        0.000815315888177871, 8.15315888177871e-06, 8.15315888177871e-07,
        1.8894983599873987,
    ),
    "K": (  # sludgeIncin row59
        0.412080464231229, 0.0823718888035191, 0.490726146063518,
        0.00494050030057791, 0.00494050030057791, 0.00494050030057791,
        1.2045931154416654,
    ),
    "Mg": (  # sludgeIncin row60
        0.874977538316047, 0.0310237036007456, 0.0910028638955204,
        0.000622646795142004, 0.00131847077363618, 0.00105477661890894,
        1.6581112207963145,
    ),
    "Na": (  # sludgeIncin row61
        0.41707717569786507, 0.139573070607553, 0.41871921182266,
        0.00821018062397373, 0.00821018062397373, 0.00821018062397373,
        1.3478260869565217,
    ),
}
# fmt: on

# sludgeIncin Q/S/W:BN source weights are kg reagent per kg aqueous element. Transfer their ratios
# between donor and actual sludge.
_NAOH_WEIGHTS = {"S": 2.51, "N": 2.69, "P": 2.43, "Cl": 1.06, "Br": 0.47, "F": 1.98, "I": 0.30}
_LIME_WEIGHTS = {"N": 0.48, "P": 0.43, "Cl": 0.19, "Br": 0.08, "F": 0.35, "I": 0.053}
_HEAVY_METALS = frozenset(
    (
        "Ag",
        "As",
        "Ba",
        "Cd",
        "Co",
        "Cr",
        "Cu",
        "Hg",
        "Mn",
        "Mo",
        "Ni",
        "Pb",
        "Sb",
        "Se",
        "Sn",
        "V",
        "Zn",
        "Be",
        "Sc",
        "Sr",
        "Ti",
        "Tl",
        "W",
    )
)

# Exact ecoinvent3.12 products at the selected RoW WWT-SLF incinerator.
# Grouping by product retains every dated supplier share; all doses
# are calculated from the original source, including the 8/3/9 supplier groups.
# Native triazine-compound row857029 is omitted: admitted leachate sludge has no
# Hg/Cd and therefore no TMT duty. Donor Hg/Cd remain in the other source drivers.
_SUPPLY = {
    "NH3": ("ammonia, anhydrous, liquid", "nitrogen"),
    "reheat": ("heat, district or industrial, natural gas", "nitrogen"),
    "TiO2": ("titanium dioxide", "nitrogen"),
    "V2O5": ("chromium trioxide, flakes", "nitrogen"),
    "PAM": ("chemical, inorganic", "aqueous"),
    "HCl": ("hydrochloric acid, without water, in 30% solution state", "aqueous"),
    "FeCl3": ("iron(III) chloride, without water, in 40% solution state", "metals"),
    "CaO": ("quicklime, milled, packed", "lime"),
    "NaOH": ("sodium hydroxide, without water, in 50% solution state", "alkali"),
    "water": ("water, decarbonised", "scrubber"),
    "facility": ("municipal waste incineration facility", "wet"),
}

# Source-supported fixed APC references, CIAAW2024 conventional atomic masses
# (g/mol). Neutral PAM C3H5NO represents the polyelectrolyte.
_ATOMS = {
    "C": 12.011,
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "Na": 22.990,
    "Cl": 35.45,
    "Ca": 40.078,
    "Fe": 55.845,
    "Ti": 47.867,
    "V": 50.942,
}
_APC_FORMULAS = {
    "NaOH": {"Na": 1, "O": 1, "H": 1},
    "CaO": {"Ca": 1, "O": 1},
    "FeCl3": {"Fe": 1, "Cl": 3},
    "HCl": {"H": 1, "Cl": 1},
    "PAM": {"C": 3, "H": 5, "N": 1, "O": 1},
}

# Native flow names are retained. Trace-organic emissions use Doka's
# nominal C12/H1/O16/Cl35.45 stoichiometry and mixed-waste allocation.
# Chlorinated VOCs use reference-waste Cl, separately from target sludge
# and post-furnace APC chlorine; C/H attribution follows that reference.
_TRACE_ORGANICS = {
    "Benzene": (6, 6, 0, 0),
    "Benzo(a)pyrene": (20, 12, 0, 0),
    "Hexachlorobenzene": (6, 0, 0, 6),
    "Pentachlorobenzene": (6, 1, 0, 5),
    "Pentachlorophenol": (6, 1, 1, 5),
    "Toluene": (7, 8, 0, 0),
}
_NMVOC = "NMVOC, non-methane volatile organic compounds"
_PROCESS_FLOWS = frozenset(_TRACE_ORGANICS) | {
    _NMVOC,
    "Particulate Matter, < 2.5 um",
    "Particulate Matter, > 2.5 um and < 10um",
    "Dioxins, measured as 2,3,7,8-tetrachlorodibenzo-p-dioxin",
    "Non-hazardous waste disposed",
}
_CARBON_INDICATORS = {
    "DOC, Dissolved Organic Carbon": 1.0,
    "TOC, Total Organic Carbon": 1.0,
    # Elemental-carbon oxidation uses 32 kg O2 per 12 kg C. These
    # overlapping kg-O2 indicators describe the same organic material.
    "COD, Chemical Oxygen Demand": 32 / 12,
    "BOD5, Biological Oxygen Demand": 32 / 12,
}
_FLOW_ELEMENTS = {
    "Aluminium III": "Al",
    "Antimony ion": "Sb",
    "Arsenic ion": "As",
    "Barium II": "Ba",
    "Cadmium II": "Cd",
    "Calcium II": "Ca",
    "Chromium III": "Cr",
    "Cobalt II": "Co",
    "Copper ion": "Cu",
    "Iron ion": "Fe",
    "Lead II": "Pb",
    "Magnesium": "Mg",
    "Manganese II": "Mn",
    "Mercury II": "Hg",
    "Molybdenum VI": "Mo",
    "Nickel II": "Ni",
    "Phosphorus": "P",
    "Potassium I": "K",
    "Selenium IV": "Se",
    "Silicon": "Si",
    "Sodium I": "Na",
    "Sulfur dioxide": "S",
    "Tin ion": "Sn",
    "Vanadium V": "V",
    "Zinc II": "Zn",
    "Nitrate": "N",
    "Phosphate": "P",
    "Sulfate": "S",
    "Strontium II": "Sr",
    "Thallium I": "Tl",
    "Titanium ion": "Ti",
}


@dataclass(frozen=True, slots=True)
class SludgeIncinerationSource:
    """Original no-FLUWA composition and normalized operating rows.

    Elements are kg per wet reference kg, inputs retain native product units
    per wet kg, and emissions retain name/unit/compartment keys. Production
    signs are normalized during source admission. Absent elements remain absent.
    """

    elements_kg: dict[str, float]
    inputs: dict[str, float]
    emissions: dict[_Emission, float]


@dataclass(frozen=True, slots=True)
class _IncinerationLoad:
    """Conditional furnace quantities on the caller's reference basis.

    Reagents are dry-active kg, except reheat MJ, water kg and facility units.
    Emissions retain native units and compartments. Thermal terms use Doka's
    energy allocations. Process-organic emissions use the mixed-waste reference;
    their Cl is outside the sludge and air-pollution-control elemental pools.
    """

    formulation_service_id: str
    wet_mass_kg: float
    dry_mass_kg: float
    water_kg: float
    elements_kg: dict[str, float]
    allocated_hydrogen_kg: float
    allocated_oxygen_kg: float
    carbon: tuple[_TreatmentCarbon, ...]
    duties: dict[str, float]
    fresh_apc_elements_kg: dict[str, dict[str, float]]
    aqueous_apc_elements_kg: dict[str, float]
    emissions: dict[_Emission, float]
    thermal: dict[str, float]
    bottom_ash: _AshLoad
    residual_ash: _AshLoad


def _sludge_drivers(elements: dict[str, float], wet_mass: float) -> dict[str, float]:
    """Evaluate Doka's operating allocation quantities in kg per reference.

    Aqueous throughput comprises scrubber sludge and treated water; total
    scrubber throughput also includes air. N duty is wet mass +100*N, based
    on equal thermal and fuel-N contributions at 0.01 kg-N/kg-wet. Donor heavy
    metals remain in the source denominators.
    """
    aqueous = {el: mass * (_TRANSFER[el][3] + _TRANSFER[el][4]) for el, mass in elements.items()}
    return {
        "wet": wet_mass,
        "nitrogen": wet_mass + 100 * elements["N"],
        "aqueous": fsum(aqueous.values()),
        "scrubber": fsum(mass * fsum(_TRANSFER[el][3:6]) for el, mass in elements.items()),
        "alkali": fsum(weight * aqueous.get(el, 0.0) for el, weight in _NAOH_WEIGHTS.items()),
        "lime": fsum(weight * aqueous.get(el, 0.0) for el, weight in _LIME_WEIGHTS.items()),
        "metals": fsum(mass for el, mass in aqueous.items() if el in _HEAVY_METALS),
    }


def _sludge_apc(
    duties: dict[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float], float]:
    """Allocate fresh air-pollution-control materials after combustion.

    Doka's phase assumptions allocate 100/101 of Fe and half of Ca to solids,
    and all Na/Cl to water. Neutral PAM remains solid. Element amounts are
    retained separately; oxide factors size disposal services.
    """
    fresh = {}
    for reagent, formula in _APC_FORMULAS.items():
        molecular = fsum(_ATOMS[el] * count for el, count in formula.items())
        fresh[reagent] = {
            el: duties[reagent] * _ATOMS[el] * count / molecular for el, count in formula.items()
        }
    solid: dict[str, float] = {}
    water: dict[str, float] = {}
    for reagent, atoms in fresh.items():
        for el, mass in atoms.items():
            if reagent == "PAM":
                fraction = 1.0
            elif el == "Fe":
                fraction = 100 / 101
            elif reagent == "CaO":
                fraction = 0.5
            else:
                fraction = 0.0
            solid[el] = solid.get(el, 0.0) + mass * fraction
            water[el] = water.get(el, 0.0) + mass * (1 - fraction)
    # Mineral sizing follows the accepted sludgeIncin oxide factors, while
    # the actual H/O balance calculated above is retained.
    disposal = solid["Fe"] * _TRANSFER["Fe"][6] + solid["Ca"] * _TRANSFER["Ca"][6] + duties["PAM"]
    return fresh, solid, water, disposal


def _sludge_ash(
    elements: dict[str, float],
    carbon: tuple[_TreatmentCarbon, ...],
    fresh: dict[str, dict[str, float]],
    solids: dict[str, float],
    disposal: float,
    duties: dict[str, float],
    *,
    bottom_ash_water_to_dry_ratio: float,
) -> tuple[_AshLoad, _AshLoad]:
    """Construct bottom/residual payloads before their separate complete treatments.

    Doka no-FLUWA reference: boiler+ESP+scrubber solids form raw residue.
    Installed catalyst retains the original TiO2/V2O5 dose and native suppliers.
    Chromium trioxide is the manufacturing-supply proxy for the declared installed oxide.
    """
    bottom = {el: mass * _TRANSFER[el][0] for el, mass in elements.items()}
    residual = {el: mass * fsum(_TRANSFER[el][1:4]) for el, mass in elements.items()}
    # Doka 2003 II Section 2.6.2 sizes leachate sludge ash from complete oxide
    # equivalents. Original H/O remain separate material descriptors below;
    # adding their masses would count oxygen already supplied by oxide factors.
    bottom_dry = fsum(
        mass * _TRANSFER[el][6] for el, mass in bottom.items() if el not in {"H", "O"}
    )
    # The selected ash receiver owns its quench-water/dry-solid ratio.
    # This water stays in the ash payload and the furnace water account.
    quench = bottom_dry * bottom_ash_water_to_dry_ratio
    titanium = duties["TiO2"] * _ATOMS["Ti"] / (_ATOMS["Ti"] + 2 * _ATOMS["O"])
    vanadium = duties["V2O5"] * 2 * _ATOMS["V"] / (2 * _ATOMS["V"] + 5 * _ATOMS["O"])
    auxiliary = dict(solids)
    auxiliary["Ti"] = titanium
    auxiliary["V"] = vanadium
    auxiliary["O"] += duties["TiO2"] + duties["V2O5"] - titanium - vanadium
    residual_mass = fsum(
        mass * _TRANSFER[el][6] for el, mass in residual.items() if el not in {"H", "O"}
    )
    bottom_carbon = tuple(
        _TreatmentCarbon(x.component, x.principal, x.origin, x.carbon_kg * _TRANSFER["C"][0])
        for x in carbon
    )
    residual_carbon = tuple(
        _TreatmentCarbon(
            x.component, x.principal, x.origin, x.carbon_kg * fsum(_TRANSFER["C"][1:4])
        )
        for x in carbon
    ) + (_TreatmentCarbon("APC_PAM", False, "fossil", fresh["PAM"]["C"]),)
    return (
        _AshLoad(
            bottom_dry + quench,
            quench,
            {el: mass for el, mass in bottom.items() if el not in {"H", "O"}},
            bottom["H"],
            bottom["O"],
            {},
            bottom_carbon,
        ),
        _AshLoad(
            residual_mass + disposal + duties["TiO2"] + duties["V2O5"],
            0.0,
            {el: mass for el, mass in residual.items() if el not in {"H", "O"}},
            residual["H"],
            residual["O"],
            auxiliary,
            residual_carbon,
        ),
    )


def _sludge_emissions(
    source: dict[_Emission, float],
    donor: dict[str, float],
    elements: dict[str, float],
    wet: float,
    aqueous: dict[str, float],
    carbon: tuple[_TreatmentCarbon, ...],
    source_carbon_origins: tuple[str, ...],
    *,
    phosphorus_phase_reference: bool,
) -> dict[_Emission, float]:
    """Apply actual loads and Doka's operating emission factors.

    Operating CO, CH4, trace organics and particulate matter scale with wet load.
    Chlorinated emissions use the mixed-waste reference in ``_TRACE_ORGANICS``.
    Element/species coefficients already include molar conversion. Fresh aqueous
    air-pollution-control material adds Fe/Ca/Na/Cl after Doka's included
    scrubber-water treatment.
    """
    nitrogen = (wet + 100 * elements["N"]) / (1 + 100 * donor["N"])
    result: dict[_Emission, float] = {}
    for key, coefficient in source.items():
        name, _, categories = key
        if phosphorus_phase_reference and name in {"Phosphorus", "Phosphate"}:
            continue  # The selected actual-P function owns both outputs below.
        if name in _PROCESS_FLOWS:
            amount = coefficient * wet
        elif name in {"Ammonia", "Nitrogen oxides"}:
            amount = coefficient * nitrogen
        elif name in {"Cyanide", "Dinitrogen monoxide"}:
            amount = coefficient * elements["N"] / donor["N"]
        elif name in _CARBON_INDICATORS:
            amount = elements["C"] * _TRANSFER["C"][4] * _CARBON_INDICATORS[name]
        elif name in {
            "Carbon dioxide, fossil",
            "Carbon dioxide, non-fossil",
            "Carbon monoxide, fossil",
            "Carbon monoxide, non-fossil",
            "Methane, fossil",
            "Methane, non-fossil",
            "Water",
            "Heat, waste",
            "Oxygen",
        }:
            continue  # Complete carbon, water and energy equations below.
        else:
            element = _FLOW_ELEMENTS[name]
            amount = coefficient * elements.get(element, 0.0) / donor[element]
            # These three native ions report kg of the corresponding element.
            if categories == _SURFACE and element in {"Na", "Ca", "Fe"}:
                amount += aqueous[element]
        result[key] = amount
    if phosphorus_phase_reference:
        # Doka2021 sludgeIncin G26:L26: kg P/kg incoming P; air stays kg P. Municipal2020 water
        # emissions B71=(31+4*16)/31 reports kg PO4/kg P. AE167 includes ash disposal. Keep actual
        # ash P in _sludge_ash; select this before thermal/water.
        phosphorus = elements["P"]
        result["Phosphorus", "kilogram", _URBAN] = phosphorus * _TRANSFER["P"][5]
        result["Phosphate", "kilogram", _SURFACE] = phosphorus * _TRANSFER["P"][4] * 95 / 31
    # Use the published phase transfer plus the actual fresh acid/FeCl3 chlorine.
    result["Chloride", "kilogram", _SURFACE] = elements["Cl"] * _TRANSFER["Cl"][4] + aqueous["Cl"]
    carbon_mass = elements["C"]
    # Use each actual origin pool, as in the Swiss sludge owner. The elemental
    # ledger and component sum can round differently; subtracting a rounded
    # fossil fraction from one would invent a negative absent-origin release.
    total_carbon = fsum(x.carbon_kg for x in carbon)
    origin_shares = {
        origin: fsum(x.carbon_kg for x in carbon if (x.origin == "fossil") == fossil) / total_carbon
        for origin, fossil in (("fossil", True), ("non-fossil", False))
    }
    gases = {}
    for species in ("Carbon monoxide", "Methane"):
        gases[species] = wet * fsum(
            source[f"{species}, {origin}", "kilogram", _URBAN] for origin in source_carbon_origins
        )
    # CH1.85 represents anonymous NMVOC carbon. PM and dioxin TEQ overlap other indicators.
    trace_carbon = 0.0
    for name, (c, h, o, cl) in _TRACE_ORGANICS.items():
        molecular = 12 * c + h + 16 * o + 35.45 * cl
        trace_carbon += result[name, "kilogram", _URBAN] * 12 * c / molecular
    other_carbon = (
        gases["Carbon monoxide"] * 12 / 28
        + gases["Methane"] * 12 / 16
        + result[_NMVOC, "kilogram", _URBAN] * 12 / 13.85
        + result["Cyanide", "kilogram", _URBAN] * 12 / 26
        + trace_carbon
    )
    gases["Carbon dioxide"] = (carbon_mass * _TRANSFER["C"][5] - other_carbon) * 44 / 12
    for species, amount in gases.items():
        for origin, share in origin_shares.items():
            result[f"{species}, {origin}", "kilogram", _URBAN] = amount * share
    return result


def _sludge_thermal_water(
    elements: dict[str, float],
    wet: float,
    water: float,
    duties: dict[str, float],
    quench: float,
    emissions: dict[_Emission, float],
    source: dict[_Emission, float],
    scrubber_ratio: float,
    *,
    organic_oxygen_kg: float,
) -> dict[str, float]:
    """Calculate gross heat, residual energy and nominal reaction water.

    Michel's (via Doka (2021)) coefficients use bulk N/S, nominal H and the organic O from
    Doka (2021). Mineral oxide O determines dry mass, operating duties and ash.
    The internal-energy convention uses 2.2 MJ/kg water and 9 kg-H2O/kg-H. Gross-heat release uses
    the water-vapor reference.
    Gas formation enthalpies use Chase (1998), via NIST Chemistry WebBook, in kJ/mol at 298.15 K,
    with molecular masses from the Doka gas reference.
    """
    hhv = 34.016 * elements["C"] + 124.265 * elements["H"] - 9.8324 * organic_oxygen_kg
    hhv += 6.276 * elements["N"] + 19.079 * elements["S"]
    lhv = hhv - 2.2 * water - 2.2 * 9 * elements["H"]
    # Doka2003 section3.3.8: gross efficiencies .13 electrical/.244 thermal; internal .144kWh and
    # .839MJ per kg wet sludge. Energy outputs receive zero allocation.
    net_electricity = max(0.13 * lhv - 0.144 * 3.6 * wet, 0.0)
    internal_electricity = max(0.144 * 3.6 * wet - 0.13 * lhv, 0.0)
    internal_heat = max(0.839 * wet - 0.244 * lhv, 0.0)
    co = fsum(
        emissions[f"Carbon monoxide, {x}", "kilogram", _URBAN] for x in ("fossil", "non-fossil")
    )
    methane = fsum(emissions[f"Methane, {x}", "kilogram", _URBAN] for x in ("fossil", "non-fossil"))
    nmvoc = emissions[_NMVOC, "kilogram", _URBAN]
    nh3_air = emissions["Ammonia", "kilogram", _URBAN]
    residual = (
        co * (393.52 - 110.53) / 28
        + methane * (393.52 + 2 * 285.83 - 74.87) / 16
        + nmvoc * (12 * 34.016 + 1.85 * 124.265) / 13.85
    )
    trace_h = 0.0
    for name, (c, h, o, cl) in _TRACE_ORGANICS.items():
        molecular = 12 * c + h + 16 * o + 35.45 * cl
        amount = emissions[name, "kilogram", _URBAN]
        trace_h += amount * h / molecular
        residual += amount * (34.016 * 12 * c + 124.265 * h - 9.8324 * 16 * o) / molecular
    residual += emissions["Cyanide", "kilogram", _URBAN] * (34.016 * 12 + 6.276 * 14) / 26
    oxide_energy = emissions["Nitrogen oxides", "kilogram", _URBAN] * 33.10 / 46
    oxide_energy += emissions["Dinitrogen monoxide", "kilogram", _URBAN] * 82.05 / 44
    retained_energy = hhv * fsum(_TRANSFER["C"][:5])
    ammonia_energy = (duties["NH3"] - nh3_air) * (1.5 * 285.83 - 45.90) / 17
    heat = hhv - retained_energy - residual + ammonia_energy - oxide_energy
    heat += duties["reheat"] + internal_electricity - net_electricity
    # Source air/surface Heat rows supply the partition. Delivered reheat dissipates here, with
    # upstream suppliers linked. Doka2003 section3.2.10 assigns co-incinerator internal heat at its
    # producer.
    heat_air_share = source["Heat, waste", "megajoule", _URBAN] / fsum(
        source["Heat, waste", "megajoule", compartment] for compartment in (_URBAN, _SURFACE)
    )
    emissions["Heat, waste", "megajoule", _URBAN] = heat * heat_air_share
    emissions["Heat, waste", "megajoule", _SURFACE] = heat * (1 - heat_air_share)
    h_out = nh3_air * 3 / 17 + methane * 4 / 16 + nmvoc * 1.85 / 13.85 + trace_h
    # H excludes supplied moisture and fresh post-furnace APC. The 18/(2*1)
    # conversion uses the nominal source dry-H reference.
    reaction_water = 9 * (elements["H"] + duties["NH3"] * 3 / 17 - h_out)
    # Retain the source process-water duty for this receiving function.
    # Waste moisture and reacted hydrogen add water; ash retains the quench.
    free_water = water + duties["water"] + reaction_water - quench
    # Apply the air/water shares from Doka (2021), Section 17.4.1 (92.7%/7.3%),
    # to free water after quench. The plant water meter includes
    # 3 kg dilution water/kg NH3.
    emissions["Water", "cubic meter", _URBAN] = 0.927 * free_water / 1000
    emissions["Water", "cubic meter", _SURFACE] = 0.073 * free_water / 1000
    emissions["Oxygen", "kilogram", _AIR_OXYGEN] = (
        source["Oxygen", "kilogram", _AIR_OXYGEN] * scrubber_ratio
    )
    return {
        "HHV_MJ": hhv,
        "calorific_organic_oxygen_kg": organic_oxygen_kg,
        "source_LHV_MJ": lhv,
        "nonair_carbon_energy_MJ": retained_energy,
        "carbon_species_residual_MJ": residual,
        "NH3_net_energy_MJ": ammonia_energy,
        "nitrogen_oxide_formation_MJ": oxide_energy,
        "delivered_reheat_MJ": duties["reheat"],
        "internal_electricity_duty_MJ": 0.144 * 3.6 * wet,
        "internal_heat_duty_MJ": 0.839 * wet,
        "source_internal_electricity_in_MJ": internal_electricity,
        "source_internal_heat_in_MJ": internal_heat,
        "source_net_electricity_out_MJ": net_electricity,
        "dissipated_heat_MJ": heat,
        "nominal_reaction_water_kg": reaction_water,
        "quench_water_kg": quench,
        "free_water_kg": free_water,
        "NH3_dilution_within_process_water_kg": 3 * duties["NH3"],
    }


def calculate_sludge_incineration(
    source: SludgeIncinerationSource,
    wastewater: _WastewaterLoad,
    formulation_service_id: str,
    *,
    source_carbon_origins: tuple[str, ...],
    additional_apc_water_flows: tuple[str, ...],
    phosphorus_phase_reference: bool,
    bottom_ash_water_to_dry_ratio: float,
    additional_mineral_elements: tuple[str, ...] = (),
) -> _IncinerationLoad:
    """Calculate the complete conditional furnace for the supplied sludge load.

    The caller passes calculated wastewater treatment, with no collection fraction.
    Quantities retain that input's reference basis: a material recipe's
    principal kg or a PA activity's complete wastewater load.
    Its incineration wet/water amounts already contain disposal share and
    70% moisture. Apply the share only to the remaining dry elements/carbon.
    Original fixed operating coefficients persist at every background date;
    dated supplier technologies may vary independently of these quantities.
    The required P choice preserves native municipal factors when false and
    uses report air/direct-water P phases for reconstructed PA when true.
    The selected bottom-ash water/dry-solid ratio sizes retained quench water. Waste-element
    transfers use their separately declared coefficients.
    Source origin templates and added APC water flows are admitted construction
    choices, not inferred from absent source values. Ash P remains elemental.
    ``additional_mineral_elements`` selects the road's declared mineral transfers.
    """
    donor, inputs, source_emissions = source.elements_kg, source.inputs, source.emissions
    share = wastewater.incineration_dry_share
    final = wastewater.final_sludge
    dry = share * final.dry_mass_kg
    wet = wastewater.incinerated_sludge_wet_kg
    water = wastewater.water.incinerated_sludge_water_kg
    elements = {el: share * mass for el, mass in final.elements_kg.items()}
    hydrogen, oxygen = share * final.allocated_hydrogen_kg, share * final.allocated_oxygen_kg
    nominal = {**elements, "H": hydrogen, "O": oxygen}
    carbon = tuple(
        _TreatmentCarbon(x.component, x.principal, x.origin, share * x.final_sludge_carbon_kg)
        for x in wastewater.carbon
    )
    actual_drivers, donor_drivers = _sludge_drivers(nominal, wet), _sludge_drivers(donor, 1.0)
    duties = {
        name: inputs[product] * actual_drivers[driver] / donor_drivers[driver]
        for name, (product, driver) in _SUPPLY.items()
    }
    fresh, solids, aqueous, disposal = _sludge_apc(duties)
    bottom, residual = _sludge_ash(
        nominal,
        carbon,
        fresh,
        solids,
        disposal,
        duties,
        bottom_ash_water_to_dry_ratio=bottom_ash_water_to_dry_ratio,
    )
    emissions = _sludge_emissions(
        source_emissions,
        donor,
        nominal,
        wet,
        aqueous,
        carbon,
        source_carbon_origins,
        phosphorus_phase_reference=phosphorus_phase_reference,
    )
    # The plastic-leachate sludge donor receives the declared aqueous half of newly supplied APC
    # lime. Municipal sludge retains its existing calcium flow.
    for flow in additional_apc_water_flows:
        emissions[(flow, "kilogram", _SURFACE)] = aqueous[_FLOW_ELEMENTS[flow]]
    # Actual UPR glass and road glass/TiO2/limestone introduce known minerals absent from their
    # selected plastic/paint references. Original sludgeIncin rows52/55/57--61 L/K supply elemental
    # air/water fractions; native positive calibrated rows, when present, remain intact. Sulfate
    # reports kg SO4, hence 96/32 kg sulfate per kg elemental sulfur transferred to water. Add
    # incoming Ca to the already owned fresh-APC aqueous Ca. B's corresponding phases are zero.
    for name, element in (
        ("Silicon", "Si"),
        ("Calcium II", "Ca"),
        ("Aluminium III", "Al"),
        ("Titanium ion", "Ti"),
        ("Potassium I", "K"),
        ("Magnesium", "Mg"),
        ("Sodium I", "Na"),
    ):
        if element not in ("Si", "Ca", *additional_mineral_elements):
            continue  # Other material owners retain their admitted transfer selection.
        for compartment, phase in ((_URBAN, 5), (_SURFACE, 4)):
            key = (name, "kilogram", compartment)
            if key not in source_emissions and nominal[element] != 0:
                emissions[key] = (
                    emissions.get(key, 0.0) + nominal[element] * _TRANSFER[element][phase]
                )
    sulfate = ("Sulfate", "kilogram", _SURFACE)
    if "S" in additional_mineral_elements and sulfate not in source_emissions:
        emissions[sulfate] = nominal["S"] * _TRANSFER["S"][4] * 96 / 32
    sulfur_dioxide = ("Sulfur dioxide", "kilogram", _URBAN)
    if "S" in additional_mineral_elements and sulfur_dioxide not in source_emissions:
        # The corresponding source air fraction is kg S; 64/32 gives kg SO2.
        emissions[sulfur_dioxide] = nominal["S"] * _TRANSFER["S"][5] * 64 / 32
    thermal = _sludge_thermal_water(
        nominal,
        wet,
        water,
        duties,
        bottom.water_kg,
        emissions,
        source_emissions,
        actual_drivers["scrubber"] / donor_drivers["scrubber"],
        organic_oxygen_kg=share * final.organic_oxygen_kg,
    )
    return _IncinerationLoad(
        formulation_service_id,
        wet,
        dry,
        water,
        elements,
        hydrogen,
        oxygen,
        carbon,
        duties,
        fresh,
        aqueous,
        emissions,
        thermal,
        bottom,
        residual,
    )
