"""Calculate municipal treatment, auxiliaries and ash for material inputs.

Municipal2020/Doka describes co-treatment. Source quantities define operating
ratios; wet material and carbon pools determine treatment loads.
Source admission and dated supplier selection have separate owners.
"""

from dataclasses import dataclass, field
from math import fsum

from .ash_treatment import _AshLoad, _TreatmentCarbon
from .municipal_incineration_emissions import _Emission, municipal_emissions

# Original Municipal2020 slag descrapping B20/L21: kg quench water/kg wet slag.
# Existing materials retain this source reference. Road paint ash admits its
# separately selected original moisture at the binding owner.
_MUNICIPAL_BOTTOM_WATER_FRACTION = 168427.10315919179 / 1e6


@dataclass(frozen=True, slots=True)
class MunicipalIncinerationSource:
    """Original intensive donor composition and normalized operating coefficients.

    Elements and water are kg/kg wet donor; input product amounts and native
    emission quantities are per positive wet kg. Missing original phosphorus
    stays absent: the selected phase reference partitions actual input phosphorus.
    """

    elements_kg: dict[str, float]
    water_kg: float
    inputs: dict[str, float]
    emissions: dict[_Emission, float]


@dataclass(frozen=True, slots=True)
class MunicipalIncinerationMaterial:
    """Wet input, dry elemental/carbon masses and gross heat.

    Masses are kg and ``heat_potential_mj`` is total MJ. The material calculation
    selects heat-bearing components; mineral oxygen is excluded from the
    organic CHONS correlation. ``elements_kg`` contains the reacting matrix.
    Bulk-inert atoms retain their original oxygen and exclude separately
    recorded ``geological_carbon_kg``. Together these fields give the bulk mass
    sent to bottom ash, outside the combustion equations.
    """

    wet_mass_kg: float
    water_kg: float
    elements_kg: dict[str, float]
    carbon: tuple[_TreatmentCarbon, ...]
    heat_potential_mj: float
    bulk_inert_elements_kg: dict[str, float] = field(default_factory=dict, kw_only=True)
    geological_carbon_kg: float = field(default=0.0, kw_only=True)


@dataclass(frozen=True, slots=True)
class MunicipalIncinerationLoad:
    """Actual municipal duties, complete direct releases and two ash loads.

    Duties are active kg, water kg, reheat MJ and facility units. Emission
    keys carry native units/compartments; carbon gases remain fossil templates
    until inventory compilation assigns their actual carbon origins.
    Oxide service mass is distinct from the actual element/carbon ledger.
    """

    source_elements: dict[str, float]
    source_drivers: dict[str, float]
    drivers: dict[str, float]
    duties: dict[str, float]
    emissions: dict[_Emission, float]
    thermal: dict[str, float]
    bottom_ash: _AshLoad
    residual_ash: _AshLoad
    native_emission_factors: dict[_Emission, float]
    material: MunicipalIncinerationMaterial
    # Non-H/O element pools allocated among the six source phases.
    air_elements_kg: dict[str, float]
    water_elements_kg: dict[str, float]
    # Fresh APC ions added after the furnace, separate from waste phases.
    aqueous_auxiliary_elements_kg: dict[str, float]


@dataclass(frozen=True, slots=True)
class _MunicipalElement:
    """MSWI element transfers and nominal oxidation/gas coefficients.

    Phases are slag, boiler ash, ESP ash, scrubber sludge, treated water and
    air, in kg element/kg input element. Oxide and oxygen coefficients use
    kg/kg; gas_moles uses mol/kg element.
    """

    property_name: str
    phases: tuple[float, float, float, float, float, float]
    oxide: float
    oxygen: float
    gas_moles: float


# Original Municipal (2020) "MSWI calculation" rows 23:61: the six phase columns G/I/K/M/O/Q use
# g/kg; stored fractions divide those values by 1000. AW/BC supply oxygen/gas factors; "ash oxides"
# column F supplies oxide factors. The selected phosphorus row 26 is included. The complete original
# composition defines the operating reference ratios; the actual material supplies its own element
# quantities. C oxide 3 is the source carbonate/elemental-carbon service allocation.
_MUNICIPAL_ELEMENTS = {
    # Municipal2020 MSWI rows27/55/57, G/I/K/M/O/Q and ash-oxide F apply to reacting B/Si/Ca only.
    # Separately admitted bulk minerals bypass these phases. B has zero air/water transfer. Incoming
    # oxide O offsets ash oxidation at the driver.
    "B": _MunicipalElement(
        "boron",
        (
            0.87871212878712129,
            0.030734214714121816,
            0.090153696494757313,
            0.00039996000399960013,
            0.0,
            0.0,
        ),
        3.2222222222222219,
        2.2222222222222219,
        0,
    ),
    "Si": _MunicipalElement(
        "silicon",
        (
            0.90654415585828087,
            0.030234202571015928,
            0.063216969012124196,
            0.0000015575195263856684,
            0.0000015575195263856684,
            0.0000015575195263856684,
        ),
        2.1393576871038951,
        1.1393576871038951,
        0,
    ),
    "Ca": _MunicipalElement(
        "calcium",
        (
            0.72455446212898846,
            0.067565734725154059,
            0.18427018561405652,
            0.0078698725106003327,
            0.0078698725106003327,
            0.0078698725106003327,
        ),
        1.3992015968063873,
        0.3992015968063873,
        0,
    ),
    # MSWI calculation row 23: C; ash oxides F supplies the oxide factor.
    "C": _MunicipalElement(
        "carbon",
        (
            0.004011064312571921,
            0.0,
            0.001004021104523617,
            0.0005020105522618085,
            0.0005020105522618085,
            0.9939808934783808,
        ),
        3,
        2.6666666666666665,
        83.33333333333333,
    ),
    # MSWI calculation row 24: S; ash oxides F supplies the oxide factor.
    "S": _MunicipalElement(
        "sulfur",
        (
            0.5480341642763491,
            0.0,
            0.30840021731731476,
            0.0584896963877666,
            0.07975867689240898,
            0.0053172451261606,
        ),
        2.5,
        1,
        31.25,
    ),
    # MSWI calculation row 25: N; ash oxides F supplies the oxide factor.
    "N": _MunicipalElement(
        "nitrogen",
        (0.009951818307848415, 0.0, 6.035175162033557e-05, 0.0, 0.0009878299405312295, 0.989),
        4.428571428571429,
        2.2857142857142856,
        71.42857142857143,
    ),
    # MSWI calculation row 28: Cl; ash oxides F supplies the oxide factor.
    "Cl": _MunicipalElement(
        "chlorine",
        (
            0.01098901098901106,
            0.0,
            0.3219003219003219,
            0.0005550005550005551,
            0.666000666000666,
            0.0005550005550005551,
        ),
        1.4513017234084562,
        -0.2256508617042281,
        28.206357713028513,
    ),
    # MSWI calculation row 29: Br; ash oxides F supplies the oxide factor.
    "Br": _MunicipalElement(
        "bromine",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        1.2002252534100863,
        -0.10011262670504317,
        12.514078338130398,
    ),
    # MSWI calculation row 30: F; ash oxides F supplies the oxide factor.
    "F": _MunicipalElement(
        "fluorine",
        (
            0.413,
            0.0,
            0.48039066655374363,
            0.054799999999999995,
            0.045399999999999996,
            0.006409333446256383,
        ),
        1.8421052631578947,
        -0.42105263157894735,
        52.63157894736842,
    ),
    # MSWI calculation row 33: As; ash oxides F supplies the oxide factor.
    "As": _MunicipalElement(
        "arsenic",
        (
            0.2999070716715154,
            0.05049185138803017,
            0.6410782064566897,
            0.0006896149986198926,
            3.5221053930173254e-11,
            0.007833255449923731,
        ),
        1.3203416978109983,
        0.3203416978109983,
        0,
    ),
    # MSWI calculation row 34: Ba; ash oxides F supplies the oxide factor.
    "Ba": _MunicipalElement(
        "barium",
        (
            0.4995951674548841,
            0.09019777035092247,
            0.3689908787083192,
            0.03723995221690232,
            5.1169182589368075e-05,
            0.003925062086382385,
        ),
        1.1164991990680064,
        0.11649919906800643,
        0,
    ),
    # MSWI calculation row 35: Cd; ash oxides F supplies the oxide factor.
    "Cd": _MunicipalElement(
        "cadmium",
        (
            0.0666725617480568,
            0.0,
            0.9276696726659037,
            0.005097086113548922,
            5.097086113548922e-05,
            0.0005097086113548922,
        ),
        1.1423487544483986,
        0.14234875444839856,
        8.896797153024911,
    ),
    # MSWI calculation row 36: Co; ash oxides F supplies the oxide factor.
    "Co": _MunicipalElement(
        "cobalt",
        (
            0.7768726953957841,
            0.014061109542863982,
            0.16873331451436774,
            0.038504269378603614,
            9.93302655350391e-06,
            0.0018186781418269907,
        ),
        1.2715085694892245,
        0.2715085694892245,
        0,
    ),
    # MSWI calculation row 37: Cr; ash oxides F supplies the oxide factor.
    "Cr": _MunicipalElement(
        "chromium",
        (
            0.5343728679901094,
            0.03094239642129659,
            0.4331935498981521,
            0.0005403280627148138,
            7.764727678785316e-05,
            0.0008732103509398873,
        ),
        1.4615384615384615,
        0.46153846153846145,
        0,
    ),
    # MSWI calculation row 38: Cu; ash oxides F supplies the oxide factor.
    "Cu": _MunicipalElement(
        "copper",
        (
            0.8889923182044733,
            0.0,
            0.10990859583715508,
            0.0009159049653096258,
            9.15904965309626e-05,
            9.15904965309626e-05,
        ),
        1.2518098835379288,
        0.25180988353792877,
        0,
    ),
    # MSWI calculation row 39: Hg; ash oxides F supplies the oxide factor.
    "Hg": _MunicipalElement(
        "mercury",
        (
            0.019265335054808763,
            0.0021288298800260525,
            0.37254522900455916,
            0.5772005772005772,
            0.0005063162957899799,
            0.028353712564238884,
        ),
        1.079764694152251,
        0.07976469415225096,
        4.985293384515678,
    ),
    # MSWI calculation row 40: Mn; ash oxides F supplies the oxide factor.
    "Mn": _MunicipalElement(
        "manganese",
        (
            0.6137593449508394,
            0.029056043851483276,
            0.3486725262177993,
            0.008202182323959074,
            0.00011892517217984618,
            0.00019097748373863227,
        ),
        1.2912373948815028,
        0.2912373948815028,
        0,
    ),
    # MSWI calculation row 42: Ni; ash oxides F supplies the oxide factor.
    "Ni": _MunicipalElement(
        "nickel",
        (
            0.9387880845776043,
            0.02146137481297733,
            0.0383238529354955,
            0.00012594750844298547,
            4.451673654021624e-05,
            0.001256223428939654,
        ),
        1.2725259751320048,
        0.2725259751320048,
        0,
    ),
    # MSWI calculation row 43: Pb; ash oxides F supplies the oxide factor.
    "Pb": _MunicipalElement(
        "lead",
        (
            0.48080856453377674,
            0.0,
            0.5107881466027585,
            0.008238518493592882,
            8.238518493592881e-05,
            8.238518493592881e-05,
        ),
        1.0772238042376563,
        0.07722380423765629,
        0,
    ),
    # MSWI calculation row 44: Sb; ash oxides F supplies the oxide factor.
    "Sb": _MunicipalElement(
        "antimony",
        (
            0.022707626224371613,
            0.021203948158029284,
            0.9560860224455402,
            8.010573529537456e-07,
            8.010573529537456e-07,
            8.010573529537456e-07,
        ),
        1.1971252566735113,
        0.19712525667351133,
        0,
    ),
    # MSWI calculation row 45: Se; ash oxides F supplies the oxide factor.
    "Se": _MunicipalElement(
        "selenium",
        (
            0.03157894736842093,
            0.13473093176264872,
            0.7470217752912678,
            0.07473684210526317,
            1.594888539098157e-06,
            0.011929908583860407,
        ),
        1.2026342451874368,
        0.20263424518743678,
        0,
    ),
    # MSWI calculation row 46: Sn; ash oxides F supplies the oxide factor.
    "Sn": _MunicipalElement(
        "tin",
        (
            0.3729206491002469,
            0.026496227800686143,
            0.6005811634822192,
            6.53205615919693e-07,
            6.53205615919693e-07,
            6.53205615919693e-07,
        ),
        1.269609908164125,
        0.269609908164125,
        0,
    ),
    # MSWI calculation row 47: V; ash oxides F supplies the oxide factor.
    "V": _MunicipalElement(
        "vanadium",
        (
            0.4264836544756224,
            0.04963074740121209,
            0.4466767266109088,
            0.055947916734743544,
            5.204852615097572e-05,
            0.02120890625136257,
        ),
        1.7852375343541422,
        0.7852375343541422,
        0,
    ),
    # MSWI calculation row 48: Zn; ash oxides F supplies the oxide factor.
    "Zn": _MunicipalElement(
        "zinc",
        (
            0.29548378495256766,
            0.0,
            0.6970762664345254,
            0.006702656408024283,
            6.702656408024283e-05,
            0.0006702656408024283,
        ),
        1.2447605935444392,
        0.24476059354443924,
        15.297537096527458,
    ),
    # MSWI calculation row 49: Be; ash oxides F supplies the oxide factor.
    "Be": _MunicipalElement(
        "beryllium",
        (
            0.8659418414801368,
            0.033602062878525824,
            0.0985660511103424,
            0.0018900445309950515,
            0.0,
            0.0,
        ),
        2.7777777777777777,
        1.7777777777777777,
        0,
    ),
    # MSWI calculation row 51: Sr; ash oxides F supplies the oxide factor.
    "Sr": _MunicipalElement(
        "strontium",
        (0.999, 0.0, 0.0, 0.0009, 0.0, 0.0001),
        1.1826067107966218,
        0.18260671079662183,
        0,
    ),
    # MSWI calculation row 52: Ti; ash oxides F supplies the oxide factor.
    "Ti": _MunicipalElement(
        "titanium",
        (
            0.7850194831678489,
            0.0505448953678714,
            0.1538322902500434,
            0.009510621508952473,
            0.0002630597438646428,
            0.0008296499614192581,
        ),
        1.6683375104427733,
        0.6683375104427733,
        0,
    ),
    # MSWI calculation row 53: Tl; ash oxides F supplies the oxide factor.
    "Tl": _MunicipalElement(
        "thallium",
        (0.3480839999999999, 0.0, 0.6499160000000002, 0.001, 0.0, 0.001),
        1.0391421987151572,
        0.03914219871515723,
        0,
    ),
    # MSWI calculation row 56: Fe; ash oxides F supplies the oxide factor.
    "Fe": _MunicipalElement(
        "iron",
        (
            0.9695135624996827,
            0.0,
            0.028674563518022608,
            0.001792160219876413,
            1.7921602198764128e-05,
            1.792160219876413e-06,
        ),
        1.2864919066036384,
        0.28649190660363844,
        0,
    ),
    # MSWI calculation row 58: Al; ash oxides F supplies the oxide factor.
    "Al": _MunicipalElement(
        "aluminium",
        (
            0.8198062202379208,
            0.044360197786881994,
            0.13500929761224956,
            0.0008153158881778709,
            8.153158881778708e-06,
            8.15315888177871e-07,
        ),
        1.8894983599873987,
        0.8894983599873987,
        0,
    ),
    # MSWI calculation row 61: Na; ash oxides F supplies the oxide factor.
    "Na": _MunicipalElement(
        "sodium",
        (
            0.41707717569786507,
            0.13957307060755336,
            0.4187192118226601,
            0.008210180623973728,
            0.008210180623973728,
            0.008210180623973728,
        ),
        1.3478260869565217,
        0.34782608695652173,
        0,
    ),
    # MSWI row 26 G/I/K/M/O/Q is elemental P, separate from whole-chain ash release.
    # P2O5 equivalent uses 2.5 O/P; AW26 uses 48/30.97 kg O/kg P.
    # BC26 is empty and cached BD26 is zero in this selected source model.
    "P": _MunicipalElement(
        "phosphorus",
        (
            0.8536803006664344,
            0.02929601162335051,
            0.11015300370379789,
            0.005743980198291034,
            0.00011046115765944296,
            0.0010162426504668753,
        ),
        (30.97 + 2.5 * 16) / 30.97,
        48 / 30.97,
        0,
    ),
}


_MUNICIPAL_SUPPLIES = {
    "NH3": ("ammonia, anhydrous, liquid", "denox"),
    "PAM": ("chemical, inorganic", "aqueous"),
    "V2O5": ("chromium trioxide, flakes", "denox"),
    "reheat": ("heat, district or industrial, natural gas", "denox"),
    "HCl": ("hydrochloric acid, without water, in 30% solution state", "aqueous"),
    "facility": ("municipal waste incineration facility", "wet"),
    "TiO2": ("titanium dioxide", "denox"),
    "water": ("water, decarbonised", "aqueous"),
    "CaO": ("quicklime, milled, packed", "lime"),
    "NaOH": ("sodium hydroxide, without water, in 50% solution state", "alkali"),
}


def _municipal_drivers(
    elements: dict[str, float], moisture: float, wet_mass: float = 1.0
) -> dict[str, float]:
    """Apply original municipal incineration equations to elemental kg and wet mass.

    The incoming dry elements and independent water descriptor belong to the
    original reference or actual material load. Waste oxygen is subtracted once after
    nominal gas/ash oxidation; lambda = 2 determines combustion-air throughput.
    """
    oxidation = fsum(
        amount
        * (
            fsum(_MUNICIPAL_ELEMENTS[el].phases[:4]) * (_MUNICIPAL_ELEMENTS[el].oxide - 1)
            + _MUNICIPAL_ELEMENTS[el].phases[5] * _MUNICIPAL_ELEMENTS[el].oxygen
        )
        for el, amount in elements.items()
        if el not in {"H", "O"}
    )
    # 8 kg O/kg reacted H; source air with 21 vol% O2, O2 at 32 kg/kmol and
    # standard molar gas volume 22.413837 m3/kmol (MSWI AW/AY/AZ).
    oxygen = 8 * elements["H"] + oxidation - elements["O"]
    air = 2 * oxygen / 32 / 0.21 * 22.413837
    # BC20 = 1000/18 mol water/kg; BC22 = 500 mol/kg H. Other nominal
    # gas moles use scrubber+treated-water+air elemental transfers (BC).
    moles = (
        moisture * (1000 / 18)
        + elements["H"] * 500
        + fsum(
            amount * fsum(_MUNICIPAL_ELEMENTS[el].phases[3:6]) * _MUNICIPAL_ELEMENTS[el].gas_moles
            for el, amount in elements.items()
            if el not in {"H", "O"}
        )
    )
    flue = moles * 22.413837 / 1000 + air - oxygen / 32 * 22.413837
    # Doka (2013) equal fuel-N/air-N allocation. NOx!C14/C39 and energy!H38
    # define the reference; common one-half cancels in the target/source ratio.
    denox = (
        elements.get("N", 0) * _MUNICIPAL_ELEMENTS["N"].phases[5] / (0.00320605256990132 * 0.989)
    )
    denox += air / 7.259570458672163
    # Doka (2013), Eq. 2.1: 62% mass and 38% combustion-condition allocation;
    # process!J98/J96 are coal/MSW raw-gas reference volumes in m3/kg.
    process = 0.62 * wet_mass + 0.38 * (14.2855 * wet_mass - flue) / (14.2855 - 7.895503583194274)
    aqueous = fsum(
        amount * fsum(_MUNICIPAL_ELEMENTS[el].phases[3:5])
        for el, amount in elements.items()
        if el not in {"H", "O"}
    )
    return {
        "wet": wet_mass,
        "oxygen": oxygen,
        "air": air,
        "flue": flue,
        "denox": denox,
        "process": process,
        "aqueous": aqueous,
    }


# Municipal2020 caustic-soda/lime tables: kg active reagent/kg elemental aqueous transfer. Sludge
# uses its separately declared reagent coefficients.
_ALKALI_WEIGHTS = {
    "S": 2.5,
    "N": 0.38632187273616825,
    "P": 0.34893588505202294,
    "Cl": 0.15256717117930482,
    "Br": 0.06769094140558643,
    "F": 0.28465822201612395,
}
_LIME_WEIGHTS = {
    "N": 2.0210746663693135,
    "P": 1.825486795430348,
    "Cl": 0.7981677102727895,
    "Br": 0.35413073002716383,
    "F": 1.4892129120615993,
}


def _municipal_reagent_driver(elements: dict[str, float], weights: dict[str, float]) -> float:
    """Sum the selected reagent weights over actual scrubber/water element kg."""
    return fsum(
        amount * fsum(_MUNICIPAL_ELEMENTS[element].phases[3:5]) * weights[element]
        for element, amount in elements.items()
        if element in weights
    )


def _municipal_ash(
    material: MunicipalIncinerationMaterial,
    duties: dict[str, float],
    *,
    bottom_ash_water_fraction: float,
) -> tuple[_AshLoad, _AshLoad, dict[str, float]]:
    """Allocate actual material, fresh municipal APC and spent catalyst to full ash.

    Doka phase transfers conserve each non-H/O element. Oxide equivalents size
    disposal services; nominal raw oxygen only defines slag quench. Fresh PAM
    retains the municipal integer C3H5NO/71 reference, distinct from the sludge
    owner's CIAAW reagent reference. Source lime is half solid/half aqueous;
    active NaOH sodium and HCl chlorine reach treated scrubber water.
    """
    elements = material.elements_kg
    slag = {
        e: q * _MUNICIPAL_ELEMENTS[e].phases[0] for e, q in elements.items() if e not in {"H", "O"}
    }
    residual = {
        e: q * fsum(_MUNICIPAL_ELEMENTS[e].phases[1:4])
        for e, q in elements.items()
        if e not in {"H", "O"}
    }
    # slag descrapping A87 and MSWI G21: the reacting matrix's raw slag O allocation is 24.5454545
    # g/kg incoming O. Doka2003 II section3.2.1 puts bulk inert material in slag with its own atoms.
    bulk = material.bulk_inert_elements_kg
    bulk_mass = fsum(bulk.values()) + material.geological_carbon_kg
    raw_o = elements["O"] * 24.5454545454545 / 1000
    water_share = bottom_ash_water_fraction
    quench = (fsum(slag.values()) + raw_o + bulk_mass) * water_share / (1 - water_share)
    slag_dry = fsum(q * _MUNICIPAL_ELEMENTS[e].oxide for e, q in slag.items()) + bulk_mass
    residual_dry = fsum(q * _MUNICIPAL_ELEMENTS[e].oxide for e, q in residual.items())
    for element, mass in bulk.items():
        if element not in {"H", "O"}:
            slag[element] = slag.get(element, 0.0) + mass
    pam = {e: duties["PAM"] * a / 71 for e, a in (("C", 36), ("H", 5), ("N", 14), ("O", 16))}
    # CIAAW oxide masses: TiO2 79.865, V2O5 181.88 and CaO 56.077 g/mol. Catalyst residues contain
    # Ti/V. The retained chromium-trioxide supply proxies V2O5 manufacture.
    titanium = duties["TiO2"] * 47.867 / 79.865
    vanadium = duties["V2O5"] * (2 * 50.9415) / 181.88
    calcium = duties["CaO"] * 0.5 * 40.078 / 56.077
    auxiliary = {**pam, "Ti": titanium, "V": vanadium, "Ca": calcium}
    auxiliary["O"] += duties["TiO2"] + duties["V2O5"] - titanium - vanadium
    auxiliary["O"] += duties["CaO"] * 0.5 - calcium
    aqueous = {
        "Ca": calcium,
        "Na": duties["NaOH"] * 22.99 / (22.99 + 15.999 + 1.008),
        "Cl": duties["HCl"] * 35.45 / 36.45,
    }
    bottom_carbon = tuple(
        _TreatmentCarbon(
            x.component, x.principal, x.origin, x.carbon_kg * _MUNICIPAL_ELEMENTS["C"].phases[0]
        )
        for x in material.carbon
    )
    residual_carbon = tuple(
        _TreatmentCarbon(
            x.component,
            x.principal,
            x.origin,
            x.carbon_kg * fsum(_MUNICIPAL_ELEMENTS["C"].phases[1:4]),
        )
        for x in material.carbon
    ) + (_TreatmentCarbon("APC_PAM", False, "fossil", pam["C"]),)
    return (
        _AshLoad(
            slag_dry + quench,
            quench,
            slag,
            bulk.get("H", 0.0),
            raw_o + bulk.get("O", 0.0),
            {},
            bottom_carbon,
            geological_carbon_kg=material.geological_carbon_kg,
        ),
        _AshLoad(
            residual_dry + duties["PAM"] + duties["TiO2"] + duties["V2O5"] + 0.5 * duties["CaO"],
            0,
            residual,
            0,
            0,
            auxiliary,
            residual_carbon,
        ),
        aqueous,
    )


def calculate_municipal_incineration(
    material: MunicipalIncinerationMaterial,
    source: MunicipalIncinerationSource,
    *,
    bottom_ash_water_fraction: float = _MUNICIPAL_BOTTOM_WATER_FRACTION,
) -> MunicipalIncinerationLoad:
    """Evaluate one complete actual material before origin attribution or date shares.

    Native donor composition calibrates dose ratios once. The received material
    supplies actual wet mass, moisture and reacting atoms. Bulk minerals contribute whole slag mass
    and quench water; combustion and aqueous reagent allocation use their specified reacting inputs.
    """
    source_drivers = _municipal_drivers(source.elements_kg, source.water_kg)
    drivers = _municipal_drivers(material.elements_kg, material.water_kg, material.wet_mass_kg)
    for name, weights in (("alkali", _ALKALI_WEIGHTS), ("lime", _LIME_WEIGHTS)):
        source_drivers[name] = _municipal_reagent_driver(source.elements_kg, weights)
        drivers[name] = _municipal_reagent_driver(material.elements_kg, weights)
    ratios = {
        k: drivers[k] / source_drivers[k]
        for k in ("wet", "denox", "process", "aqueous", "alkali", "lime")
    }
    duties = {d: source.inputs[p] * ratios[r] for d, (p, r) in _MUNICIPAL_SUPPLIES.items()}
    bottom, residual, aqueous = _municipal_ash(
        material, duties, bottom_ash_water_fraction=bottom_ash_water_fraction
    )
    # Doka2013 section 2.11: the process water meter includes water retained in
    # bottom ash. Admitted bulk fillers bypass the reacting aqueous driver,
    # so the scaled native dose cannot supply their additional quench. Add
    # only that actual bulk duty at the selected bottom ash's wet water share;
    # retain the native dose and existing organic-ash water allocation intact.
    bulk_quench = (
        (fsum(material.bulk_inert_elements_kg.values()) + material.geological_carbon_kg)
        * bottom_ash_water_fraction
        / (1 - bottom_ash_water_fraction)
    )
    duties["water"] += bulk_quench
    elements = material.elements_kg
    air = {
        e: q * _MUNICIPAL_ELEMENTS[e].phases[5] for e, q in elements.items() if e not in {"H", "O"}
    }
    water = {
        e: q * _MUNICIPAL_ELEMENTS[e].phases[4] for e, q in elements.items() if e not in {"H", "O"}
    }
    emissions, thermal, native_factors = municipal_emissions(
        source.emissions,
        ratios,
        duties,
        drivers,
        bottom.water_kg,
        elements=elements,
        source_elements=source.elements_kg,
        phase_air=air,
        phase_water=water,
        moisture=material.water_kg,
        wet_mass=material.wet_mass_kg,
        aqueous_auxiliary=aqueous,
        heat_potential_mj=material.heat_potential_mj,
    )
    thermal["bulk_quench_water_kg"] = bulk_quench
    return MunicipalIncinerationLoad(
        source.elements_kg,
        source_drivers,
        drivers,
        duties,
        emissions,
        thermal,
        bottom,
        residual,
        native_factors,
        material,
        air,
        water,
        aqueous,
    )
