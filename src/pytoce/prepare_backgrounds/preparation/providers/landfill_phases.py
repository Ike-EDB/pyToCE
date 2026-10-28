"""Landfill2020 phases and untreated dump releases for actual polymer mixtures.

PA finish, cured resins and filled SBR supply component masses and amounts
assigned to degradation. The calculation retains the source climate, element
release, later kinetics and direct dump discharges. The PA receiving and inventory modules manage
event state, source records, treatment suppliers, and activity years.
"""

from dataclasses import dataclass, field
from math import exp, fsum, log

from .landfill_site import _landfill_leachate_emissions
from .leachate_load import _CarbonOrigin, _LeachateLoad
from .waste_gas_load import _gas_leakage_reference, _GasEmission, _LandfillGasLoad

# Landfill2020 MSWLF calculation H/J rows21--26,61: fraction of mobilized
# element entering gas, and element-specific release relative to degradation.
_PHASE_FACTORS = {
    "C": (0.970873786407767, 1.0),
    "H": (0.970873786407767, 1.0),
    "O": (0.970873786407767, 1.0),
    "N": (0.06438442211055276, 2.4976427240513064),
    "S": (0.1490935840646033, 0.437649447721238),
    "P": (0.0, 0.0559165999037902),
    "Na": (0.00024993751562109475, 4.13932455312919),
    # Landfill2020 MSWLF H/J rows27,36,55,57,58 supply transfer functions for actual UPR
    # glass/silica and dispersed cobalt/sizing.
    "B": (0.0, 6.7287126940448214),
    "Co": (0.00024993751562109475, 0.32150064200725031),
    "Si": (0.00024993751562109475, 0.05),
    "Ca": (0.00024993751562109475, 0.13024173278843815),
    "Al": (0.00024993751562109475, 0.05),
    # MSWLF H/J row48 describes particle transfer of dispersed ZnO in compound22 rubber.
    "Zn": (0.00022217285047767166, 0.04738342800955586),
    # Landfill2020 MSWLF H/J rows52/59/60 describe particle entrainment of road rutile/glass ions.
    "Ti": (0.00024993751562109475, 0.05),
    "K": (0.00024993751562109475, 0.73130714062628965),
    "Mg": (0.00024993751562109475, 0.61723443585572391),
}


@dataclass(frozen=True, slots=True)
class _LandfillSite:
    """Original site climate and methane correction, independent of regional custody.

    Preparation admits the selected site's temperature in degrees Celsius,
    precipitation in mm/year and methane correction. The sanitary reference
    has correction 1; each selected 7 m open dump uses 1-2.0604/7 from
    Landfill2020 control board E133.
    """

    mean_annual_temperature_c: float
    mean_annual_precipitation_mm: float
    methane_correction: float


@dataclass(frozen=True, slots=True)
class _LandfillPhase:
    """Actual kg element entering gas, short/later leachate and retained stock."""

    input_kg: float
    gas_kg: float
    short_leachate_kg: float
    later_leachate_kg: float
    stored_kg: float


@dataclass(frozen=True, slots=True)
class _LandfillCarbon:
    """One principal or auxiliary origin's separate incoming and phase kg C."""

    component: str
    principal: bool
    origin: _CarbonOrigin
    phases: _LandfillPhase


@dataclass(frozen=True, slots=True)
class _LandfillLoad:
    """Actual whole-mixture phases before site and downstream treatment.

    Quantities are kg for the caller's actual load and its declared reference.
    Water remains separate from the admitted dry-material elements. Short-phase
    leachate contains elemental quantities plus empirical organic
    descriptors; its hydraulic volume belongs to the selected site owner.
    The gas pools contain methane/CO2 carbon before gas-treatment operations.
    Geological carbon is an additional input already inside wet mass,
    outside the selected model's organic element phases and carbon emissions.
    """

    wet_mass_kg: float
    water_kg: float
    elements_kg: dict[str, float]
    phases: dict[str, _LandfillPhase]
    carbon: tuple[_LandfillCarbon, ...]
    leachate: _LeachateLoad
    gas: _LandfillGasLoad
    geological_carbon_kg: float = field(default=0.0, kw_only=True)


def _climate_alpha(site: _LandfillSite) -> float:
    """Apply Landfill2020 Clim E43/E54/E46/E59/E63 at the admitted sites."""
    # 29.372386... per meter precipitation and .3 per degree Celsius are
    # source climate coefficients. /1000 converts mm to m; +15 sets the
    # temperature reference. All admitted sites exceed its dry/cold cutoffs.
    precipitation = 60 * (1 - exp(-29.372386439548432 * site.mean_annual_precipitation_mm / 1000))
    temperature = 60 * (1 - exp(-0.3 * (site.mean_annual_temperature_c + 15)))
    # 60 and 220 kg CH4/t are mean-MSW calibration potentials.
    denominator = log(1 - 60 / 220)
    return log(1 - precipitation / 220) * log(1 - temperature / 220) / denominator**2


def _later_release(alpha: float) -> dict[str, float]:
    """Calculate source mean-MSW kinetics, independently of target composition."""
    # avgTK C/I/N rows 25,26,61 are mean-MSW degradation, element release multiplier and later
    # kinetic coefficient. The target uses its separately specified linear MSWLF degradation. The
    # source later horizon is 60000-100 years.
    later: dict[str, float] = dict.fromkeys(("C", "H", "O", "N", "S", "P", "Na"), 1.0)
    for element, mean_degradation, multiplier, kinetic in (
        ("N", 0.20199069396374611, 2.4976427240513059, 0.061771875011314883),
        ("P", 0.27120402960925999, 0.0559165999037902, 0.0072978929039903915),
        ("Na", 0.049438637076944972, 4.1393245531291898, 1.0),
    ):
        working = (1 - exp(alpha * log(1 - mean_degradation))) * multiplier
        # Each admitted site's mean-MSW working point is below one. These
        # source kinetics remain distinct from the target-mixture release.
        if element == "P":
            # avgTK L26="lin", Q26=1: additional release of the residual pool.
            # The source caps this expression at one. Even alpha=1 gives
            # less than .068 here, so that cap is inactive at all six sites.
            later[element] = working * kinetic * 59900 / (100 * (1 - working))
        else:
            # avgTK L25/L61="exp", Q25/Q61=1 cancel the carbonate subdivision.
            later[element] = 1 - exp(log(1 - working) * kinetic * 59900 / 100)
    return later


def _mineral_later_release(alpha: float, carbonate_end_year: float) -> dict[str, float]:
    """Return finite cured-material mineral release after year 100 at an admitted site.

    The six original sanitary/dump sites supply climate alpha and the source's
    rounded carbonate-end year. Landfill2020 avgTK uses mean-MSW kinetics,
    independently of the target composite's one-percent initial degradation.
    These are integrated source factors, not extra MFA years or a time series.
    """
    # avgTK row55: mean-MSW Si degradation .08561625, release multiplier .05,
    # linear rate .1 and post-carbonate rate multiplier .01. None of its caps
    # is active at the six admitted sites (total release .01006--.03939).
    working = (1 - exp(alpha * log(1 - 0.085616251444943711))) * 0.05
    duration = carbonate_end_year - 100 + 0.01 * (60000 - carbonate_end_year)
    silicon = working * 0.1 * duration / (100 * (1 - working))
    # The same original equations complete B/Co/Ca/Al release at every site. B's working-point cap
    # is active and its exponential endpoint rounds to one. Co/Ca/Al uncapped endpoints exceed one
    # (minimum 332/1943/67), so the original cumulative cap gives these exact remaining-pool
    # fractions. Road Ti/Mg avgTK rows52/60 have capped linear endpoints above one at all six
    # admitted sites. K row59 is exponential, C=.28396885, I=.73130714, N=Q=1; its remaining-pool
    # complement is below 3.3e-61 at those sites. Preserve that source equation's binary64 result.
    potassium_working = (1 - exp(alpha * log(1 - 0.28396884871158018))) * 0.73130714062628965
    potassium = 1 - exp(log(1 - potassium_working) * 59900 / 100)
    return {
        "B": 1.0,
        "Co": 1.0,
        "Si": silicon,
        "Ca": 1.0,
        "Al": 1.0,
        "Ti": 1.0,
        "K": potassium,
        "Mg": 1.0,
    }


def _element_phase(
    element: str, mass: float, degraded: float, alpha: float, later: float
) -> _LandfillPhase:
    """Apply whole-element release to the admitted plastic-material families."""
    gas_fraction, release = _PHASE_FACTORS[element]
    # MSWLF E=alpha*D/B, followed by L/N's IF(E*J>1,1-1e-9,E*J). Multiplication by B gives the mass
    # directly, including for zero-valued elements. Its cap is inactive for this fixed family:
    # alpha<1; C/H/O have d<=.27, N has .27*J<.675, P/S only occur in the fixed finish and have
    # d*J<.016/.119. Sulfonate supplies all finish Na and has d=.01, so its
    # d*J=.01*4.13932455312919<.042. Principal PA contains zero Na. Any P/T mixture retains these
    # bounds. Epoxy has only CHON at d=.01, so its largest d*J is N=.02498. Both families remain
    # below their source caps. UPR applies homogeneous .01 liberation to its filled article; its
    # largest multiplier is B=6.728713, hence .01*alpha*J<.0673 at the six source sites. Compound22
    # uses the same .01 matrix estimate; Zn's J=.04738343 keeps its working point below .000474 at
    # the five selected dump sites. Road uses .01 for matrix/minerals and .05 for limestone. Ca's
    # largest weighted release stays below .00652; Ti/K/Mg stay below .00732.
    mobilized = alpha * degraded * release
    remainder = mass - mobilized
    return _LandfillPhase(
        mass,
        mobilized * gas_fraction,
        mobilized * (1 - gas_fraction),
        remainder * later,
        remainder * (1 - later),
    )


_AIR = ("air", "non-urban air or from high stacks")
_LATER_AIR = ("air", "low population density, long-term")
_GROUND = ("water", "ground-")
_LATER_GROUND = ("water", "ground-, long-term")
_SOIL = ("soil", "industrial")
_WASTE = ("inventory indicator", "waste")


@dataclass(frozen=True, slots=True)
class _DumpDischarges:
    """Direct inventory and mass-balance ledgers for one unmanaged polymer mixture.

    Gas emissions preserve principal and auxiliary carbon origins. Evaporite
    air and soil ledgers use elemental kg; sulfate, nitrate and particle-size
    descriptors overlap those elements. Nitrogen remaining after the empirical
    NOx output is recorded as N2.
    """

    gas: tuple[_GasEmission, ...]
    emissions: dict[tuple[str, tuple[str, ...]], float]
    short_air_elements_kg: dict[str, float]
    later_air_elements_kg: dict[str, float]
    soil_elements_kg: dict[str, float]
    nitrogen_as_nox_kg: float
    dinitrogen_kg: float
    short_water_kg: float
    later_water_kg: float


def _dump_discharges(
    load: _LandfillLoad, site: _LandfillSite, *, upward_leachate: bool
) -> _DumpDischarges:
    """Apply the finite dump's direct gas and untreated phase-output convention.

    All quantities are actual kg, MJ or m3 before any conditional dump-class
    share. Climate and degradation are already applied by the actual material owner.
    No capture, combustion plant, sewer, wastewater or donor composition is added.
    """
    leakage = _gas_leakage_reference(
        load.gas.carbon, load.gas.nitrogen_kg, load.gas.sulfur_kg, 0.0, 1.0
    )
    # Landfill2020 air & energy D101/I83: 0.419 g NO2 per m3 gas divided
    # by 49 g N2 plus that NO2's nitrogen. Retain the empirical NOx output
    # and record complementary nitrogen as N2.
    nox = load.gas.nitrogen_kg * 0.419 / (49 + 0.419 * 14 / 46)
    nox_nitrogen = nox * 14 / 46
    gas = tuple(row for row in leakage.emissions if row.name != "Ammonia") + (
        _GasEmission("Nitrogen oxides", nox, "kilogram"),
    )
    emissions = {
        ("Sodium I", _AIR): load.phases["Na"].gas_kg,
        ("Organic carbon, placed in landfill", _WASTE): load.elements_kg["C"],
        ("Waste mass, total, placed in landfill", _WASTE): load.wet_mass_kg,
        ("Non-hazardous waste disposed", _WASTE): load.wet_mass_kg,
    }
    # Actual UPR minerals and SBR zinc have elemental rural-air transfers,
    # separate from methane/CO2 and the later evaporite particle allocation.
    minerals = (
        ("B", "Boron"),
        ("Si", "Silicon"),
        ("Ca", "Calcium II"),
        ("Al", "Aluminium III"),
        ("Co", "Cobalt II"),
        ("Zn", "Zinc II"),
        ("Ti", "Titanium ion"),
        ("K", "Potassium I"),
        ("Mg", "Magnesium"),
    )
    for element, name in minerals:
        if element in load.phases:
            emissions[name, _AIR] = load.phases[element].gas_kg
    # Landfill2020 water coefficients H=0, J=1, P=Q=1 partition carried water
    # between short mobilization and the later phase. PA moisture follows
    # principal-polymer decomposition at .01; cured epoxy is dry.
    short_water = 0.01 * _climate_alpha(site) * load.water_kg
    later_water = load.water_kg - short_water
    # Doka2016 section2.3: evaporation at reversed-infiltration sites versus downward transport at
    # wet sites. /1000 converts actual kg water to m3. These are returns of carried moisture.
    water_compartments = (_AIR, _LATER_AIR) if upward_leachate else (_GROUND, _LATER_GROUND)
    emissions["Water", water_compartments[0]] = short_water / 1000
    emissions["Water", water_compartments[1]] = later_water / 1000
    energy = load.gas.decomposition_trace
    # Doka2003 section6.5.1 and Landfill2020 air&energy C134:C135 assign
    # short decomposition heat to soil and later heat to groundwater.
    # Apply the component HHV partition and keep raw-gas chemical energy
    # separate from dissipated heat.
    emissions["Heat, waste", _SOIL] = (
        load.gas.decomposition_heat_mj + energy["short_leachate_hhv_mj"]
    )
    # Heat uses the native later-groundwater compartment at every site.
    # Leachate follows the separate site-water calculation. Stored HHV is
    # exactly zero for this admitted material family.
    emissions["Heat, waste", _LATER_GROUND] = energy["later_leachate_hhv_mj"]
    short = {element: phase.short_leachate_kg for element, phase in load.phases.items()}
    later = {element: phase.later_leachate_kg for element, phase in load.phases.items()}
    short_air: dict[str, float] = {}
    later_air: dict[str, float] = {}
    soil: dict[str, float] = {}
    if upward_leachate:
        # Control board E98 and MSWLF AJ/AO/AY/BC: 10% of the evaporite solids is wind-eroded; 90%
        # returns to industrial soil. This applies only to the -250 mm site.
        short_air = {element: 0.1 * mass for element, mass in short.items()}
        later_air = {element: 0.1 * mass for element, mass in later.items()}
        soil = {element: 0.9 * (short[element] + later[element]) for element in short}
        for phase, compartment in ((short_air, _AIR), (later_air, _LATER_AIR)):
            # MSWLF AJ/AY24--26: air sulfate and nitrate carry nominal compound masses, whereas P/Na
            # retain elemental kg. The O/H/C phase masses also enter PM total (AJ/AY62). Named ions
            # overlap particle mass.
            compounds = {
                "Sulfate": phase["S"] * 96 / 32,
                "Nitrate": phase["N"] * 62 / 14,
                "Phosphorus": phase["P"],
                "Sodium I": phase["Na"],
            }
            compounds.update({name: phase[e] for e, name in minerals if e in phase})
            particle_mass = fsum(compounds.values()) + fsum(phase[e] for e in ("C", "H", "O"))
            for name, amount in compounds.items():
                key = name, compartment
                emissions[key] = emissions.get(key, 0.0) + amount
            # MSWLF AL64:AL66 is the admitted 50/30/20 size distribution.
            # These three exclusive size classes partition one particle mass.
            for name, share in (
                ("Particulate Matter, > 10 um", 0.5),
                ("Particulate Matter, > 2.5 um and < 10um", 0.3),
                ("Particulate Matter, < 2.5 um", 0.2),
            ):
                emissions[name, compartment] = particle_mass * share
        for element, name in (
            ("C", "Carbon"),
            ("N", "Nitrogen"),
            ("S", "Sulfur"),
            ("P", "Phosphorus"),
            ("Na", "Sodium I"),
            *minerals,
        ):
            if element in soil:
                emissions[name, _SOIL] = soil[element]
    else:
        emissions.update(_landfill_leachate_emissions(short, compartment=_GROUND))
        emissions.update(_landfill_leachate_emissions(later, compartment=_LATER_GROUND))
    return _DumpDischarges(
        gas,
        emissions,
        short_air,
        later_air,
        soil,
        nox_nitrogen,
        load.gas.nitrogen_kg - nox_nitrogen,
        short_water,
        later_water,
    )
