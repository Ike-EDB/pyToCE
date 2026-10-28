"""Calculate Swiss municipal sludge co-incineration and FLUWA ash washing.

PA fiber manufacture/recycling, phosphate-ester and PDO wastewater supply actual
sludge through the shared wastewater calculation. Doka's municipal model allocates
mixed-plant operations using nominal H/O, gas volume, energy and oxide masses.
The caller supplies these fixed constructed inlet families or their positive
scalings, not arbitrary wastewater. Phosphate-ester sludge requires neutralizing
lime; PDO's cobalt requires conventional scrubber precipitation. PA6/PA66 finish manufacture and
washing use zero duty for both quantities under the same source equations.
"""

from dataclasses import dataclass
from math import fsum

from .ash_treatment import _AshLoad, _TreatmentCarbon
from .leachate_load import _WastewaterLoad

Emission = tuple[str, str, tuple[str, ...]]
URBAN = ("air", "urban air close to ground")
SURFACE = ("water", "surface water")
AIR_OXYGEN = ("natural resource", "in air")

# Municipal Incineration 2020, MSWI calculation rows 23:61: slag, boiler ash,
# ESP ash, scrubber solids, treated water and air (G/I/K/M/O/Q divided by 1000).
# F in ash oxides gives oxide-equivalent disposal mass. AW gives gas oxygen
# kg/kg-element; BC gives mol gas/kg-element. None is an elementary emission.
# These complete original donor descriptors also determine native duty ratios.
# Table columns: six transfers, oxide mass factor, gas oxygen, gas moles.
# fmt: off
ELEMENT_FUNCTIONS: dict[str, tuple[float, ...]] = {
    "C": (
        0.004011064312571921, 0.0, 0.001004021104523617,
        0.0005020105522618085, 0.0005020105522618085, 0.9939808934783808,
        3, 2.6666666666666665, 83.33333333333333,
    ),
    "S": (
        0.5480341642763491, 0.0, 0.30840021731731476,
        0.0584896963877666, 0.07975867689240898, 0.0053172451261606,
        2.5, 1, 31.25,
    ),
    "N": (
        0.009951818307848415, 0.0, 6.035175162033557e-05,
        0.0, 0.0009878299405312295, 0.989,
        4.428571428571429, 2.2857142857142856, 71.42857142857143,
    ),
    "P": (
        0.8536803006664344, 0.02929601162335051, 0.1101530037037979,
        0.005743980198291034, 0.00011046115765944297, 0.0010162426504668753,
        2.2915724895059735, 1.5498869874071683, 0,
    ),
    "B": (
        0.8787121287871212, 0.030734214714121817, 0.09015369649475731,
        0.0003999600039996001, 0.0, 0.0,
        3.222222222222222, 8.888888888888888, 0,
    ),
    "Cl": (
        0.01098901098901106, 0.0, 0.3219003219003219,
        0.0005550005550005551, 0.666000666000666, 0.0005550005550005551,
        1.4513017234084562, -0.2256508617042281, 28.206357713028513,
    ),
    "Br": (
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.2002252534100863, -0.10011262670504317, 12.514078338130398,
    ),
    "F": (
        0.413, 0.0, 0.48039066655374363,
        0.054799999999999995, 0.045399999999999996, 0.006409333446256383,
        1.8421052631578947, -0.42105263157894735, 52.63157894736842,
    ),
    "I": (
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.1260835303388494, -0.06304176516942474, 7.880220646178093,
    ),
    "Ag": (
        0.6365341420126576, 0.0, 0.3583337127409253,
        0.0014663272132620986, 0.0036658180331552463, 0.0,
        1.074163344766849, 0.07416334476684905, 0,
    ),
    "As": (
        0.2999070716715154, 0.05049185138803017, 0.6410782064566897,
        0.0006896149986198926, 3.5221053930173254e-11, 0.007833255449923731,
        1.3203416978109983, 0.3203416978109983, 0,
    ),
    "Ba": (
        0.4995951674548841, 0.09019777035092247, 0.3689908787083192,
        0.03723995221690232, 5.1169182589368075e-05, 0.003925062086382385,
        1.1164991990680064, 0.11649919906800643, 0,
    ),
    "Cd": (
        0.0666725617480568, 0.0, 0.9276696726659037,
        0.005097086113548922, 5.097086113548922e-05, 0.0005097086113548922,
        1.1423487544483986, 0.14234875444839856, 8.896797153024911,
    ),
    "Co": (
        0.7768726953957841, 0.014061109542863982, 0.16873331451436774,
        0.038504269378603614, 9.93302655350391e-06, 0.0018186781418269907,
        1.2715085694892245, 0.2715085694892245, 0,
    ),
    "Cr": (
        0.5343728679901094, 0.03094239642129659, 0.4331935498981521,
        0.0005403280627148138, 7.764727678785316e-05, 0.0008732103509398873,
        1.4615384615384615, 0.46153846153846145, 0,
    ),
    "Cu": (
        0.8889923182044733, 0.0, 0.10990859583715508,
        0.0009159049653096258, 9.15904965309626e-05, 9.15904965309626e-05,
        1.2518098835379288, 0.25180988353792877, 0,
    ),
    "Hg": (
        0.019265335054808763, 0.0021288298800260525, 0.37254522900455916,
        0.5772005772005772, 0.0005063162957899799, 0.028353712564238884,
        1.079764694152251, 0.07976469415225096, 4.985293384515678,
    ),
    "Mn": (
        0.6137593449508394, 0.029056043851483276, 0.3486725262177993,
        0.008202182323959074, 0.00011892517217984618, 0.00019097748373863227,
        1.2912373948815028, 0.2912373948815028, 0,
    ),
    "Mo": (
        0.7598732305834627, 0.03769338653530159, 0.19744154851824647,
        0.0036234140245256334, 5.263155147936848e-05, 0.0013157887869842122,
        1.5003126954346466, 0.5003126954346466, 0,
    ),
    "Ni": (
        0.9387880845776043, 0.02146137481297733, 0.0383238529354955,
        0.00012594750844298547, 4.451673654021624e-05, 0.001256223428939654,
        1.2725259751320048, 0.2725259751320048, 0,
    ),
    "Pb": (
        0.48080856453377674, 0.0, 0.5107881466027585,
        0.008238518493592882, 8.238518493592881e-05, 8.238518493592881e-05,
        1.0772238042376563, 0.07722380423765629, 0,
    ),
    "Sb": (
        0.022707626224371613, 0.021203948158029284, 0.9560860224455402,
        8.010573529537456e-07, 8.010573529537456e-07, 8.010573529537456e-07,
        1.1971252566735113, 0.19712525667351133, 0,
    ),
    "Se": (
        0.03157894736842093, 0.13473093176264872, 0.7470217752912678,
        0.07473684210526317, 1.594888539098157e-06, 0.011929908583860407,
        1.2026342451874368, 0.20263424518743678, 0,
    ),
    "Sn": (
        0.3729206491002469, 0.026496227800686143, 0.6005811634822192,
        6.53205615919693e-07, 6.53205615919693e-07, 6.53205615919693e-07,
        1.269609908164125, 0.269609908164125, 0,
    ),
    "V": (
        0.4264836544756224, 0.04963074740121209, 0.4466767266109088,
        0.055947916734743544, 5.204852615097572e-05, 0.02120890625136257,
        1.7852375343541422, 0.7852375343541422, 0,
    ),
    "Zn": (
        0.29548378495256766, 0.0, 0.6970762664345254,
        0.006702656408024283, 6.702656408024283e-05, 0.0006702656408024283,
        1.2447605935444392, 0.24476059354443924, 15.297537096527458,
    ),
    "Be": (
        0.8659418414801368, 0.033602062878525824, 0.0985660511103424,
        0.0018900445309950515, 0.0, 0.0,
        2.7777777777777777, 1.7777777777777777, 0,
    ),
    "Sc": (
        0.999, 0.0, 0.0,
        0.0005, 0.0, 0.0005,
        1.5333333333333334, 0.5333333333333334, 0,
    ),
    "Sr": (
        0.999, 0.0, 0.0,
        0.0009, 0.0, 0.0001,
        1.1826067107966218, 0.18260671079662183, 0,
    ),
    "Ti": (
        0.7850194831678489, 0.0505448953678714, 0.1538322902500434,
        0.009510621508952473, 0.0002630597438646428, 0.0008296499614192581,
        1.6683375104427733, 0.6683375104427733, 0,
    ),
    "Tl": (
        0.3480839999999999, 0.0, 0.6499160000000002,
        0.001, 0.0, 0.001,
        1.0391421987151572, 0.03914219871515723, 0,
    ),
    "W": (
        0.7507694117647059, 0.0, 0.24923058823529406,
        0.0, 0.0, 0.0,
        1.2610824041338047, 0.2610824041338047, 0,
    ),
    "Si": (
        0.9065441558582809, 0.030234202571015927, 0.0632169690121242,
        1.5575195263856683e-06, 1.5575195263856683e-06, 1.5575195263856683e-06,
        2.139357687103895, 1.139357687103895, 0,
    ),
    "Fe": (
        0.9695135624996827, 0.0, 0.028674563518022608,
        0.001792160219876413, 1.7921602198764128e-05, 1.792160219876413e-06,
        1.2864919066036384, 0.28649190660363844, 0,
    ),
    "Ca": (
        0.7245544621289884, 0.06756573472515406, 0.1842701856140565,
        0.007869872510600333, 0.007869872510600333, 0.007869872510600333,
        1.3992015968063873, 0.39920159680638734, 0,
    ),
    "Al": (
        0.8198062202379208, 0.044360197786881994, 0.13500929761224956,
        0.0008153158881778709, 8.153158881778708e-06, 8.15315888177871e-07,
        1.8894983599873987, 0.8894983599873987, 0,
    ),
    "K": (
        0.412080464231229, 0.08237188880351912, 0.49072614606351805,
        0.004940500300577905, 0.004940500300577905, 0.004940500300577905,
        1.2045931154416654, 0.20459311544166536, 0,
    ),
    "Mg": (
        0.874977538316047, 0.031023703600745606, 0.09100286389552044,
        0.000622646795142004, 0.0013184707736361796, 0.001054776618908944,
        1.6581112207963145, 0.6581112207963145, 0,
    ),
    "Na": (
        0.41707717569786507, 0.13957307060755336, 0.4187192118226601,
        0.008210180623973728, 0.008210180623973728, 0.008210180623973728,
        1.3478260869565217, 0.34782608695652173, 0,
    ),
}
# fmt: on

# Exact native property names, independent of element-emission coefficients.
ELEMENT_NAMES = {
    "C": "carbon",
    "S": "sulfur",
    "N": "nitrogen",
    "P": "phosphorus",
    "B": "boron",
    "Cl": "chlorine",
    "Br": "bromine",
    "F": "fluorine",
    "I": "iodine",
    "Ag": "silver",
    "As": "arsenic",
    "Ba": "barium",
    "Cd": "cadmium",
    "Co": "cobalt",
    "Cr": "chromium",
    "Cu": "copper",
    "Hg": "mercury",
    "Mn": "manganese",
    "Mo": "molybdenum",
    "Ni": "nickel",
    "Pb": "lead",
    "Sb": "antimony",
    "Se": "selenium",
    "Sn": "tin",
    "V": "vanadium",
    "Zn": "zinc",
    "Be": "beryllium",
    "Sc": "scandium",
    "Sr": "strontium",
    "Ti": "titanium",
    "Tl": "thallium",
    "W": "tungsten",
    "Si": "silicon",
    "Fe": "iron",
    "Ca": "calcium",
    "Al": "aluminium",
    "K": "potassium",
    "Mg": "magnesium",
    "Na": "sodium",
    "H": "hydrogen",
    "O": "oxygen",
}

# Doka (2015), Table 7.1 / FLUWA Z/AB/AD: washed matrix, hydroxide cake, treated effluent fractions.
# Actual PA/PDO loads and fresh WWT Fe/PAM define the incoming elements.
FLUWA_TRANSFER = {
    "C": (0.8879097253261209, 0.056045137336939554, 0.056045137336939554),
    "N": (0, 0, 1),
    "P": (0.702085457400281, 0.2922935134940639, 0.005621029105655075),
    "S": (0.6543735315858822, 0.297914542599719, 0.04771192581439881),
    "Na": (0.42005932274909696, 0.030923796242223033, 0.54901688100868),
    "Fe": (0.9920900086605335, 0.007831674593531056, 7.831674593530975e-05),
    # Original FLUWA Z/AB/AD rows36/57/59/60: kg element/kg
    # element entering filter-ash/scrubber-liquid extraction. Unlike the MSWI
    # combustion table, these fractions require no per-1000 conversion.
    "Co": (0.7587483233192104, 0.24118945653443066, 0.0000622201463589627),
    "Ca": (0.8368600023988669, 0.08156999880056649, 0.08156999880056649),
    "K": (0.557244444655638, 0.023608763992007138, 0.4191467913523548),
    "Mg": (0.9145217448313052, 0.027418618269793263, 0.05805963689890142),
}
# Original MSWI technology F37: fraction of filter ash receiving FLUWA treatment. This weights the
# actual ash phases.
FLUWA_SHARE = 0.4621512526302507

# Original auxiliaries D47:D59: kg active alkali per kg aqueous element.
NAOH_WEIGHTS = {
    "S": 2.5,
    "N": 0.38632187273616825,
    "P": 0.34893588505202294,
    "Cl": 0.15256717117930482,
    "Br": 0.06769094140558643,
    "F": 0.28465822201612395,
    "I": 0.042620222366480345,
}
LIME_WEIGHTS = {
    "N": 2.0210746663693135,
    "P": 1.825486795430348,
    "Cl": 0.7981677102727895,
    "Br": 0.35413073002716383,
    "F": 1.4892129120615993,
    "I": 0.22297120038747353,
}
# Municipal Incineration 2020, auxiliaries G61 = SUM(MSWI!AF32:AF54,AJ32:AJ54): aqueous toxic-metal
# kg before conventional (non-FLUWA) share H61. These metals determine native FeCl3 demand.
TOXIC_METALS = (
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

# Native CH products, source rows 493228:493244, retain original quantity ratios. Dated suppliers
# provide the corresponding technologies.
SUPPLIES = {
    "NH3": ("ammonia, anhydrous, liquid", "denox"),
    "reheat": ("heat, district or industrial, natural gas", "denox"),
    "TiO2": ("titanium dioxide", "denox"),
    "V2O5": ("chromium trioxide, flakes", "denox"),
    "PAM": ("chemical, inorganic", "conventional_aqueous"),
    "FeCl3": ("iron(III) chloride, without water, in 40% solution state", "toxic_metals"),
    "HCl": ("hydrochloric acid, without water, in 30% solution state", "acid"),
    "H2O2": ("hydrogen peroxide, without water, in 50% solution state", "filter_ash"),
    "NaOH": ("sodium hydroxide, without water, in 50% solution state", "alkali"),
    "water": ("water, decarbonised", "aqueous"),
    "facility": ("municipal waste incineration facility", "wet"),
}

# Doka (2013) mixed-waste trace species use nominal C/H/O/Cl counts. PM and dioxin TEQ overlap other
# indicators.
TRACE_ORGANICS = {
    "Benzene": (6, 6, 0, 0),
    "Benzo(a)pyrene": (20, 12, 0, 0),
    "Hexachlorobenzene": (6, 0, 0, 6),
    "Pentachlorobenzene": (6, 1, 0, 5),
    "Pentachlorophenol": (6, 1, 1, 5),
    "Toluene": (7, 8, 0, 0),
}
NMVOC = "NMVOC, non-methane volatile organic compounds"
PROCESS_FLOWS = frozenset(TRACE_ORGANICS) | {
    NMVOC,
    "Particulate Matter, < 2.5 um",
    "Particulate Matter, > 2.5 um and < 10um",
    "Dioxins, measured as 2,3,7,8-tetrachlorodibenzo-p-dioxin",
}
# Municipal water-emission rows B73:B76 use the original 32/14 empirical
# BOD/COD descriptor.
CARBON_INDICATORS = {
    "DOC, Dissolved Organic Carbon": 1.0,
    "TOC, Total Organic Carbon": 1.0,
    "COD, Chemical Oxygen Demand": 32 / 14,
    "BOD5, Biological Oxygen Demand": 32 / 14,
}
# Actual target elements present in the native CH emissions. All other donor
# elements have zero target dose; their source rows remain in provenance.
FLOW_ELEMENTS = {
    "Iron ion": "Fe",
    "Sodium I": "Na",
    # The original CH incinerator supplies these exact ion names separately in air and surface
    # water. Its native phase calibration includes units and molecular conventions.
    "Cobalt II": "Co",
    "Calcium II": "Ca",
    "Potassium I": "K",
    "Magnesium": "Mg",
    "Phosphorus": "P",
    "Sulfur dioxide": "S",
    "Nitrate": "N",
    "Phosphate": "P",
    "Sulfate": "S",
}


@dataclass(frozen=True, slots=True)
class SwissSludgeSource:
    """Original source descriptors and exchange quantities per native waste kg."""

    elements_kg: dict[str, float]
    inputs: dict[str, float]
    emissions: dict[Emission, float]


@dataclass(frozen=True, slots=True)
class SwissSludgeLoad:
    """Conditional Swiss furnace and complete residue quantities on the caller basis.

    The caller basis is shared with its wastewater load, before collection share.
    Elements/carbon retain their material origins; nominal H/O and oxide-equivalent
    masses are separate. Duties are active kg except reheat MJ and facility units.
    """

    wet_mass_kg: float
    dry_mass_kg: float
    water_kg: float
    elements_kg: dict[str, float]
    allocated_hydrogen_kg: float
    allocated_oxygen_kg: float
    carbon: tuple[_TreatmentCarbon, ...]
    duties: dict[str, float]
    allocands: dict[str, float]
    phases: dict[str, dict[str, float]]
    fresh_elements_kg: dict[str, dict[str, float]]
    aqueous_elements_kg: dict[str, float]
    emissions: dict[Emission, float]
    thermal: dict[str, float]
    bottom_ash: _AshLoad
    residual_ash: _AshLoad
    hydroxide_kg: float
    hydroxide_water_kg: float
    hydroxide_carbon: tuple[_TreatmentCarbon, ...]


def source_allocands(elements: dict[str, float], wet: float) -> dict[str, float]:
    """Evaluate original Swiss operating quantities from the supplied composition.

    H/O are the source's nominal dry descriptors. MSWI AW66/AY66/AZ68 defines
    net oxygen; lambda 2 and 21% oxygen define air throughput, not consumption.
    BC72 includes 70% moisture and molar gas terms at 22.413837 m3/kmol.
    """
    values = {el: q for el, q in elements.items() if el in ELEMENT_FUNCTIONS}
    ash_oxygen = fsum(
        q * fsum(ELEMENT_FUNCTIONS[e][:4]) * (ELEMENT_FUNCTIONS[e][6] - 1)
        for e, q in values.items()
    )
    # MSWI AW22: 8 = 16/2 kg oxygen per kg nominal dry hydrogen.
    oxygen = max(
        8 * elements["H"]
        + fsum(q * ELEMENT_FUNCTIONS[e][5] * ELEMENT_FUNCTIONS[e][7] for e, q in values.items())
        + ash_oxygen
        - elements["O"],
        0.0,
    )
    air = 2 * oxygen / 32 / 0.21 * 22.413837
    # BC20/BC22: 1000/18 mol per kg water; 500 mol water gas per kg
    # nominal dry H. Gas moles are converted to kmol by /1000 below.
    moles = 0.7 * wet * (1000 / 18) + elements["H"] * 500
    moles += fsum(
        q * fsum(ELEMENT_FUNCTIONS[e][3:6]) * ELEMENT_FUNCTIONS[e][8] for e, q in values.items()
    )
    flue = moles * 22.413837 / 1000 + air - oxygen / 32 * 22.413837
    aqueous = {e: q * fsum(ELEMENT_FUNCTIONS[e][3:5]) for e, q in values.items()}
    total = fsum(aqueous.values())
    fa = FLUWA_SHARE * fsum(
        q * fsum(ELEMENT_FUNCTIONS[e][1:3]) * ELEMENT_FUNCTIONS[e][6] for e, q in values.items()
    )
    # NOx C14/C39 and energy H38: independent fuel-N and average-air references.
    # Equal half-weights cancel in source/target dose ratios.
    fuel_n = elements["N"] * ELEMENT_FUNCTIONS["N"][5] / (0.00320605256990132 * 0.989)
    thermal_n = air / 7.259570458672163
    # Doka (2013), Eq. 2.1 coal/MSW references process J98/J96. The nonnegative process function
    # defines the allocation weight.
    process = max(0.62 * wet + 0.38 * (14.2855 * wet - flue) / (14.2855 - 7.895503583194274), 0.0)
    return {
        "wet": wet,
        "oxygen": oxygen,
        "air": air,
        "flue": flue,
        "denox": fuel_n + thermal_n,
        "fuel_n": fuel_n,
        "thermal_n": thermal_n,
        "process": process,
        "aqueous": total,
        "conventional_aqueous": (1 - FLUWA_SHARE) * total,
        "toxic_metals": (1 - FLUWA_SHARE)
        * fsum(aqueous.get(element, 0.0) for element in TOXIC_METALS),
        "filter_ash": fa,
        # Doka (2015), pp. 18-19: 0.0332 kg HCl, 0.0241 kg NaOH and
        # 0.0472 kg CaO per kg FLUWA ash; acid base coefficient is auxiliaries D60.
        "acid": 0.01164447831218716 * (1 - FLUWA_SHARE) * total + 0.0332 * fa,
        "alkali": fsum(q * NAOH_WEIGHTS.get(e, 0.0) for e, q in aqueous.items()) + 0.0241 * fa,
        "lime_before_purchase": fsum(q * LIME_WEIGHTS.get(e, 0.0) for e, q in aqueous.items())
        - 0.0472 * fa,
    }


def _phases(elements: dict[str, float]) -> dict[str, dict[str, float]]:
    """Partition each actual element through furnace and selected FLUWA fraction."""
    result = {name: {} for name in ("bottom", "residual", "hydroxide", "water", "air")}
    for el, q in elements.items():
        slag, boiler, esp, scrub, water, air = ELEMENT_FUNCTIONS[el][:6]
        matrix, cake, liquid = FLUWA_TRANSFER[el]
        entering = FLUWA_SHARE * q * (boiler + esp + scrub + water)
        result["bottom"][el] = q * slag + entering * matrix
        result["residual"][el] = (1 - FLUWA_SHARE) * q * (boiler + esp + scrub)
        result["hydroxide"][el] = entering * cake
        result["water"][el] = (1 - FLUWA_SHARE) * q * water + entering * liquid
        result["air"][el] = q * air
    return result


def _fresh_reagents(
    duties: dict[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Partition fresh post-combustion reagents between residue solids and water.

    Doka's admitted integer molecular masses define PAM C3H5NO and active
    NaOH/HCl/H2O2. CIAAW conventional Ti 47.867, V 50.942 and O 15.999
    define installed-oxide atoms. The
    installed catalyst atoms follow the declared Ti/V oxide composition. Forming losses stay
    upstream.
    ``fresh`` contains solid atoms only. Source Fe/Ca scrubber-to-water transfer
    ratios apply after reagent addition.
    """
    pam = duties["PAM"]
    ti = duties["TiO2"] * 47.867 / (47.867 + 2 * 15.999)
    va = duties["V2O5"] * 2 * 50.942 / (2 * 50.942 + 5 * 15.999)
    # Doka's nominal active references: FeCl3 (55.85 + 3*35.45) and CaO (40.08 + 16) g/mol. The
    # native process-water demand includes solution-strength water.
    iron = duties["FeCl3"] * 55.85 / (55.85 + 3 * 35.45)
    calcium = duties["CaO"] * 40.08 / (40.08 + 16)
    lime_oxygen = duties["CaO"] * 16 / (40.08 + 16)
    # Municipal scrubber-solid/treated-water ratios: Fe = 100/1; Ca = 1/1.
    solid_iron = iron * 100 / 101
    solid_calcium = calcium / 2
    fresh = {
        "PAM": {"C": pam * 36 / 71, "H": pam * 5 / 71, "N": pam * 14 / 71, "O": pam * 16 / 71},
        "SCR": {"Ti": ti, "V": va, "O": duties["TiO2"] + duties["V2O5"] - ti - va},
        "FeCl3": {"Fe": solid_iron},
        "CaO": {"Ca": solid_calcium, "O": lime_oxygen / 2},
    }
    aqueous = {
        "Na": duties["NaOH"] * 23 / 40,
        "Cl": duties["HCl"] * 35.45 / 36.45 + duties["FeCl3"] * 3 * 35.45 / (55.85 + 3 * 35.45),
        "Fe": iron / 101,
        "Ca": calcium / 2,
        "H": duties["HCl"] / 36.45 + duties["NaOH"] / 40 + 2 * duties["H2O2"] / 34,
        "O": 16 * duties["NaOH"] / 40 + 32 * duties["H2O2"] / 34 + lime_oxygen / 2,
    }
    return fresh, aqueous


def _ash_loads(
    elements: dict[str, float],
    nominal_o: float,
    phases: dict[str, dict[str, float]],
    carbon: tuple[_TreatmentCarbon, ...],
    fresh: dict[str, dict[str, float]],
    duties: dict[str, float],
) -> tuple[_AshLoad, _AshLoad, float, float, tuple[_TreatmentCarbon, ...]]:
    """Size Swiss residues with municipal oxide equivalents and explicit water.

    Original raw-slag oxygen contributes to quench only. Final dry ashes replace
    that nominal oxygen by municipal oxides; adding both would double its mass.
    FLUWA matrix 46.4286% and cake 75% moisture are original B81/B82 references.
    """
    raw = {e: q * ELEMENT_FUNCTIONS[e][0] for e, q in elements.items()}
    dry_slag = fsum(q * ELEMENT_FUNCTIONS[e][6] for e, q in raw.items())
    matrix = {e: phases["bottom"][e] - raw[e] for e in elements}
    dry_matrix = fsum(q * ELEMENT_FUNCTIONS[e][6] for e, q in matrix.items())
    dry_cake = fsum(q * ELEMENT_FUNCTIONS[e][6] for e, q in phases["hydroxide"].items())
    dry_residual = fsum(q * ELEMENT_FUNCTIONS[e][6] for e, q in phases["residual"].items())
    # slag descrapping A87 and MSWI G21/1000: raw-element quench, before oxides.
    q = 0.16842710315919179
    quench = (fsum(raw.values()) + nominal_o * 0.0245454545454545) * q / (1 - q)
    matrix_water = dry_matrix * (0.4642857142857143 / (1 - 0.4642857142857143))
    cake_water = 3 * dry_cake  # 75% wet cake: water/dry = 0.75/0.25.
    auxiliary: dict[str, float] = {}
    for atoms in fresh.values():
        for e, mass in atoms.items():
            auxiliary[e] = auxiliary.get(e, 0.0) + mass
    pools = {
        name: tuple(
            _TreatmentCarbon(
                x.component, x.principal, x.origin, x.carbon_kg * phases[name]["C"] / elements["C"]
            )
            for x in carbon
        )
        for name in ("bottom", "residual", "hydroxide")
    }
    bottom_dry = dry_slag + dry_matrix
    bottom = _AshLoad(
        bottom_dry + quench + matrix_water,
        quench + matrix_water,
        phases["bottom"],
        0.0,
        bottom_dry - fsum(phases["bottom"].values()),
        {},
        pools["bottom"],
    )
    # Fresh scrubber Fe/Ca enter residual disposal once after furnace/FLUWA. Oxide-equivalent sizing
    # defines service mass; actual reagent atoms define residue chemistry.
    fresh_oxide_mass = (
        fresh["FeCl3"]["Fe"] * ELEMENT_FUNCTIONS["Fe"][6]
        + fresh["CaO"]["Ca"] * ELEMENT_FUNCTIONS["Ca"][6]
    )
    fresh_oxide_oxygen = (
        fresh_oxide_mass - fsum(fresh["FeCl3"].values()) - fsum(fresh["CaO"].values())
    )
    residual = _AshLoad(
        dry_residual + duties["PAM"] + duties["TiO2"] + duties["V2O5"] + fresh_oxide_mass,
        0.0,
        phases["residual"],
        0.0,
        dry_residual - fsum(phases["residual"].values()) + fresh_oxide_oxygen,
        auxiliary,
        pools["residual"] + (_TreatmentCarbon("APC_PAM", False, "fossil", fresh["PAM"]["C"]),),
    )
    return bottom, residual, dry_cake + cake_water, cake_water, pools["hydroxide"]


def _emissions(
    source: SwissSludgeSource,
    elements: dict[str, float],
    phases: dict[str, dict[str, float]],
    actual: dict[str, float],
    donor: dict[str, float],
    aqueous: dict[str, float],
    carbon: tuple[_TreatmentCarbon, ...],
) -> dict[Emission, float]:
    """Bind complete process functions and actual five-phase chemistry to native flows."""
    result: dict[Emission, float] = {}
    process = actual["process"] / donor["process"]
    denox = actual["denox"] / donor["denox"]
    for key, coefficient in source.emissions.items():
        name, _, categories = key
        if name in PROCESS_FLOWS:
            amount = coefficient * process
        elif name in {"Ammonia", "Nitrogen oxides"}:
            amount = coefficient * denox
        elif name in {"Cyanide", "Dinitrogen monoxide"}:
            amount = coefficient * elements["N"] / source.elements_kg["N"]
        elif name in CARBON_INDICATORS:
            amount = phases["water"]["C"] * CARBON_INDICATORS[name]
        elif name == "Non-hazardous waste disposed":
            amount = actual["wet"]
        elif name in FLOW_ELEMENTS:
            e = FLOW_ELEMENTS[name]
            phase = "water" if categories == SURFACE else "air"
            # Native ion/compound coefficients already contain molecular factors.
            original_phase = _phases({e: source.elements_kg[e]})[phase][e]
            amount = coefficient * phases[phase][e] / original_phase
            if phase == "water" and e in {"Na", "Fe", "Ca"}:
                amount += aqueous[e]
        else:
            continue
        result[key] = amount
    result["Chloride", "kilogram", SURFACE] = aqueous["Cl"]
    gases = {
        species: process
        * fsum(
            source.emissions[f"{species}, {origin}", "kilogram", URBAN]
            for origin in ("fossil", "non-fossil")
        )
        for species in ("Carbon monoxide", "Methane")
    }
    trace = fsum(
        result[name, "kilogram", URBAN] * 12 * c / (12 * c + h + 16 * o + 35.45 * cl)
        for name, (c, h, o, cl) in TRACE_ORGANICS.items()
    )
    # Nominal molecular C fractions: CO 12/28, CH4 12/16, anonymous
    # CH1.85 12/13.85 and cyanide 12/26. CO2 then uses 44/12.
    other = gases["Carbon monoxide"] * 12 / 28 + gases["Methane"] * 12 / 16
    other += (
        result[NMVOC, "kilogram", URBAN] * 12 / 13.85
        + result["Cyanide", "kilogram", URBAN] * 12 / 26
        + trace
    )
    gases["Carbon dioxide"] = (phases["air"]["C"] - other) * 44 / 12
    # Derive both origin groups from their actual pools. Subtracting a rounded
    # fossil fraction from one creates spurious non-fossil gas for all-fossil
    # inlets when independent carbon totals differ by floating rounding.
    total_carbon = fsum(x.carbon_kg for x in carbon)
    origin_shares = {
        origin: fsum(x.carbon_kg for x in carbon if (x.origin == "fossil") == fossil) / total_carbon
        for origin, fossil in (("fossil", True), ("non-fossil", False))
    }
    for name, amount in gases.items():
        for origin, share in origin_shares.items():
            result[f"{name}, {origin}", "kilogram", URBAN] = amount * share
    return result


def _thermal_water(
    elements: dict[str, float],
    wet: float,
    water: float,
    duties: dict[str, float],
    allocands: dict[str, float],
    bottom: _AshLoad,
    cake_water: float,
    emissions: dict[Emission, float],
    *,
    organic_oxygen_kg: float,
) -> dict[str, float]:
    """Calculate Swiss internal energy, nominal water and air/surface heat allocation.

    HHV retains WWcalc DT18's coefficients (MJ/kg-element), bulk N/S and
    nominal H. The report organic O reference (Doka 2021, section 8.1.1,
    avgWWT L79) replaces mass-estimation oxide O only in this calorific
    function. Dry mass, operating allocation quantities and ash keep total allocated O.
    LHV alone subtracts
    2.2 MJ/kg moisture/nominal water. Gas formation enthalpies are Chase (1998), via NIST Chemistry
    WebBook,
    kJ/mol at 298.15 K using the admitted nominal molecular masses. Environmental HHV release uses
    the stated gross-heat reference.
    """
    hhv = 34.016 * elements["C"] + 124.265 * elements["H"] - 9.8324 * organic_oxygen_kg
    hhv += 6.276 * elements["N"] + 19.079 * elements["S"]
    lhv = hhv - 2.2 * water - 2.2 * 9 * elements["H"]
    # Original energy I36:I42/I46: raw gas,LHV,air,three wet-mass duties,
    # conventional aqueous load and combustion-air heat. FLUWA 0.1463 kWh/kg
    # uses incoming filter ash already weighted by the FLUWA share.
    internal_e = 0.00747901579308153 * allocands["flue"] + 0.0022407203755096163 * lhv
    internal_e += 0.0019809514121561524 * allocands["air"]
    internal_e += (0.009163023516161654 + 0.009163023516161654 + 0.0066177392061167505) * wet
    internal_e += (
        0.5147153409931297 * allocands["conventional_aqueous"] + 0.1463 * allocands["filter_ash"]
    )
    internal_h = 0.06684411052212387 * allocands["air"]
    # R21/R22 CH efficiencies; 3.6 converts kWh to MJ before net/deficit subtraction.
    net_e = max(0.1584 * lhv - 3.6 * internal_e, 0.0)
    deficit_e = max(3.6 * internal_e - 0.1584 * lhv, 0.0)
    deficit_h = max(internal_h - 0.2851 * lhv, 0.0)
    co = fsum(
        emissions[f"Carbon monoxide, {x}", "kilogram", URBAN] for x in ("fossil", "non-fossil")
    )
    methane = fsum(emissions[f"Methane, {x}", "kilogram", URBAN] for x in ("fossil", "non-fossil"))
    nmvoc = emissions[NMVOC, "kilogram", URBAN]
    ammonia = emissions["Ammonia", "kilogram", URBAN]
    residual = co * (393.52 - 110.53) / 28 + methane * (393.52 + 2 * 285.83 - 74.87) / 16
    residual += nmvoc * (12 * 34.016 + 1.85 * 124.265) / 13.85
    trace_h = 0.0
    for name, (c, h, o, cl) in TRACE_ORGANICS.items():
        molecular = 12 * c + h + 16 * o + 35.45 * cl
        amount = emissions[name, "kilogram", URBAN]
        trace_h += amount * h / molecular
        residual += amount * (34.016 * 12 * c + 124.265 * h - 9.8324 * 16 * o) / molecular
    residual += emissions["Cyanide", "kilogram", URBAN] * (34.016 * 12 + 6.276 * 14) / 26
    ammonia_energy = (duties["NH3"] - ammonia) * (1.5 * 285.83 - 45.90) / 17
    no_formation = emissions["Nitrogen oxides", "kilogram", URBAN] * 33.10 / 46
    no_formation += emissions["Dinitrogen monoxide", "kilogram", URBAN] * 82.05 / 44
    heat = hhv * ELEMENT_FUNCTIONS["C"][5] - residual + ammonia_energy - no_formation
    heat += duties["reheat"] + deficit_e - net_e
    emissions["Heat, waste", "megajoule", URBAN] = heat * 0.7178438958353931
    emissions["Heat, waste", "megajoule", SURFACE] = heat * 0.2821561041646069
    h_out = 3 * ammonia / 17 + 4 * methane / 16 + 1.85 * nmvoc / 13.85 + trace_h
    reaction_water = 9 * (elements["H"] + 3 * duties["NH3"] / 17 - h_out)
    # Retain the source water duty for flue-gas cleaning.
    # Bottom ash and hydroxide cake retain water; partition the remainder.
    remaining = duties["water"] - bottom.water_kg - cake_water
    emissions["Water", "cubic meter", URBAN] = (water + reaction_water + remaining * 23 / 35) / 1000
    emissions["Water", "cubic meter", SURFACE] = remaining * 12 / 35 / 1000
    # AZ68 is net consumed oxygen; lambda 2 belongs only to air-throughput duty.
    emissions["Oxygen", "kilogram", AIR_OXYGEN] = allocands["oxygen"]
    nox_n = emissions["Nitrogen oxides", "kilogram", URBAN] * 14 / 46
    thermal_n = nox_n * allocands["thermal_n"] / allocands["denox"]
    n2 = elements["N"] * ELEMENT_FUNCTIONS["N"][5] + 14 * duties["NH3"] / 17 + thermal_n
    n2 -= nox_n + 14 * ammonia / 17 + 28 * emissions["Dinitrogen monoxide", "kilogram", URBAN] / 44
    n2 -= 14 * emissions["Cyanide", "kilogram", URBAN] / 26
    return {
        "HHV_MJ": hhv,
        "calorific_organic_oxygen_kg": organic_oxygen_kg,
        "LHV_MJ": lhv,
        "internal_electricity_kwh": internal_e,
        "internal_heat_MJ": internal_h,
        "net_electricity_MJ": net_e,
        "electricity_deficit_MJ": deficit_e,
        "heat_deficit_MJ": deficit_h,
        "environmental_heat_MJ": heat,
        "residual_species_energy_MJ": residual,
        "nominal_reaction_water_kg": reaction_water,
        "process_water_remaining_kg": remaining,
        "NH3_dilution_within_meter_kg": 3 * duties["NH3"],
        "thermal_air_nitrogen_kg": thermal_n,
        "dinitrogen_pool_kgN": n2,
    }


def calculate_swiss_sludge(
    source: SwissSludgeSource, wastewater: _WastewaterLoad
) -> SwissSludgeLoad:
    """Calculate Swiss incineration for admitted PA finish, phosphate-ester or PDO sludge.

    Wastewater supplies its actual elements, independent H/O descriptors,
    origin pools and wet/dry disposition. Its collection fraction is applied
    only by collection; its fresh Fe/PAM must not be purchased again here.
    """
    share = wastewater.incineration_dry_share
    final = wastewater.final_sludge
    elements = {e: share * q for e, q in final.elements_kg.items() if q != 0}
    # PA/PDO inputs and fresh wastewater PAM/Fe define the element set;
    # explicit zeros indicate absent elements.
    elements = {e: elements.get(e, 0.0) for e in FLUWA_TRANSFER}
    h, o = share * final.allocated_hydrogen_kg, share * final.allocated_oxygen_kg
    nominal = dict(elements, H=h, O=o)
    wet = wastewater.incinerated_sludge_wet_kg
    water = wastewater.water.incinerated_sludge_water_kg
    carbon = tuple(
        _TreatmentCarbon(x.component, x.principal, x.origin, share * x.final_sludge_carbon_kg)
        for x in wastewater.carbon
    )
    actual = source_allocands(nominal, wet)
    donor = source_allocands(source.elements_kg, 1.0)
    duties = {d: source.inputs[p] * actual[a] / donor[a] for d, (p, a) in SUPPLIES.items()}
    # Auxiliaries B73 admits purchased active lime only after the source FLUWA
    # deduction. The original CH donor buys none, so no donor-row ratio exists.
    duties["CaO"] = max(actual["lime_before_purchase"], 0.0)
    phases = _phases(elements)
    fresh, aqueous = _fresh_reagents(duties)
    bottom, residual, cake, cake_water, cake_carbon = _ash_loads(
        elements, o, phases, carbon, fresh, duties
    )
    emissions = _emissions(source, elements, phases, actual, donor, aqueous, carbon)
    thermal = _thermal_water(
        nominal,
        wet,
        water,
        duties,
        actual,
        bottom,
        cake_water,
        emissions,
        organic_oxygen_kg=share * final.organic_oxygen_kg,
    )
    return SwissSludgeLoad(
        wet,
        share * final.dry_mass_kg,
        water,
        elements,
        h,
        o,
        carbon,
        duties,
        actual,
        phases,
        fresh,
        aqueous,
        emissions,
        thermal,
        bottom,
        residual,
        cake,
        cake_water,
        cake_carbon,
    )
