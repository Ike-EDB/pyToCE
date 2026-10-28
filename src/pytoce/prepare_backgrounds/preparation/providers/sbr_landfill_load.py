"""Calculate filled SBR trim at landfill and retired compound at dumps.

Finishing loss contains principal SBR, carbon black and retained auxiliaries.
Doka (2003) supplies the homogeneous-matrix release estimate of 1% per century;
Landfill2020 supplies the selected site's climate and subsequent phase transfers.
"""

from dataclasses import dataclass
from math import exp, log

from .landfill_phases import (
    _PHASE_FACTORS,
    _climate_alpha,
    _element_phase,
    _LandfillCarbon,
    _LandfillLoad,
    _LandfillPhase,
    _LandfillSite,
    _later_release,
)
from .leachate_load import _CarbonOrigin, _LeachateCarbon, _LeachateLoad
from .sbr_material import _sbr_compound_material
from .waste_gas_load import _GasPool, _LandfillGasLoad


@dataclass(frozen=True, slots=True)
class _SBRLandfillLoad:
    """Actual filled scrap and separate carbon custody per retained principal kg."""

    mass_kg: float
    elements_kg: dict[str, float]
    carbon: tuple[_LeachateCarbon, ...]
    phases: dict[str, _LandfillPhase]
    leachate: _LeachateLoad
    gas: _LandfillGasLoad


def _sbr_landfill_phases(elements: dict[str, float]) -> dict[str, _LandfillPhase]:
    """Apply the selected site's climate and source short/long release functions.

    Landfill2020 Clim E43:E63 uses mean-MSW methane calibrations 60/220 to
    derive climate alpha; those constants are not SBR methane yields. The
    selected original site has 16.743579766537 C and 1250.84108527132 mm/year.
    Its source unit cap is attained for C/H/O/S/Zn by 60000 years. Nitrogen's
    exponential expression remains below one, so its tiny stored pool persists.
    Long-term source compartments retain their release horizons within each activity's inventory.
    """
    # Clim E40=29.372386... per meter precipitation; E27=1000 converts
    # site MAP from mm to m. E51=.3 per degree C and E54's +15C offset
    # define the source temperature response above its -15C lower threshold.
    # These climate coefficients are transferred from the stated source calibration.
    precipitation = 60 * (1 - exp(-29.372386439548432 * 1250.84108527132 / 1000))
    temperature = 60 * (1 - exp(-0.3 * (16.743579766537 + 15)))
    alpha = log(1 - precipitation / 220) / log(1 - 60 / 220)
    alpha *= log(1 - temperature / 220) / log(1 - 60 / 220)
    # MSWLF E is linear alpha times the estimated 0.01 matrix degradation.
    degradation = 0.01 * alpha
    # avgTK C25=.201990693... is mean-MSW degradation; I25=2.497642724... converts degradation to
    # its nitrogen release working point. N25=.061771875... is the later-release kinetic factor in
    # the source exponential. Q25=1 removes the carbonate-time subdivision. The 100 year short phase
    # leaves 59900 years in Doka's 60000 year source horizon. Integrated releases belong to the
    # original treatment event.
    nitrogen_average = (1 - exp(alpha * log(1 - 0.2019906939637461))) * 2.497642724051306
    nitrogen_later = 1 - exp(log(1 - nitrogen_average) / 100 * 0.06177187501131488 * 59900)
    phases = {}
    for element, mass in elements.items():
        gas_share, release = _PHASE_FACTORS[element]
        gas = degradation * release * gas_share
        short = degradation * release * (1 - gas_share)
        later = nitrogen_later if element == "N" else 1.0
        total = gas + short + (1 - gas - short) * later
        phases[element] = _LandfillPhase(
            mass, mass * gas, mass * short, mass * (total - gas - short), mass * (1 - total)
        )
    return phases


def _sbr_landfill_load(principal_origin: _CarbonOrigin) -> _SBRLandfillLoad:
    """Keep principal origin independent of fixed fossil and tallow auxiliaries.

    All loads describe the one manufacturing scrap stream per kg retained SBR.
    The same scalar operation applies to fossil, biogenic and DAC principal;
    only corresponding carbon-flow identities differ. Utilities and auxiliaries
    retain their own origins. Nominal source quantities are fixed across years
    and backgrounds; dated upstream supplier inventories vary independently.
    """
    # Five percent discarded /95 percent retained gives 1/19 principal scrap.
    material = _sbr_compound_material(principal_origin, principal_kg=1 / 19)
    elements = material.elements_kg
    phases = _sbr_landfill_phases(elements)
    carbon = (
        _LeachateCarbon(
            "principal_sbr", True, principal_origin, material.principal_carbon_kg[principal_origin]
        ),
        _LeachateCarbon(
            "fossil_compounding_auxiliaries",
            False,
            "fossil",
            material.auxiliary_carbon_kg["fossil"],
        ),
        _LeachateCarbon(
            "tallow_compounding_auxiliaries",
            False,
            "biogenic",
            material.auxiliary_carbon_kg["biogenic"],
        ),
    )
    short_fraction = phases["C"].short_leachate_kg / elements["C"]
    toc = phases["C"].short_leachate_kg
    leachate = _LeachateLoad(
        {el: p.short_leachate_kg for el, p in phases.items()},
        tuple(
            _LeachateCarbon(x.component, x.principal, x.origin, x.carbon_kg * short_fraction)
            for x in carbon
        ),
        # Doka empirical organic H/C and O/C, separate from phase H/O.
        0.1225 * toc,
        0.64 * toc,
        toc,
        toc,
        # Landfill2020 B85/B86: raw leachate COD/TOC and BOD5/TOC.
        1.09375 * toc,
        0.25892857142857145 * toc,
    )
    # Doka2003 Tables 6.2/6.3 nominal CH4/CO2 mass determines gas-C split.
    methane_fraction = (335.7 * 12 / 16) / (335.7 * 12 / 16 + 726.8 * 12 / 44)
    gas_fraction = phases["C"].gas_kg / elements["C"]
    pools = tuple(
        _GasPool(
            x.component,
            x.principal,
            x.origin,
            x.carbon_kg * gas_fraction * methane_fraction,
            x.carbon_kg * gas_fraction * (1 - methane_fraction),
        )
        for x in carbon
    )
    # The actual inlet owns ingredient heat. This receiving owner allocates
    # carbon-bearing ingredients and separate elemental sulfur to their phases.
    phase_heat = {
        phase: material.carbon_heat_mj * getattr(phases["C"], phase + "_kg") / elements["C"]
        + material.sulfur_heat_mj * getattr(phases["S"], phase + "_kg") / elements["S"]
        for phase in ("gas", "short_leachate", "later_leachate", "stored")
    }
    # Chase (1998), via NIST Chemistry WebBook liquid-water HHV; retain Doka's nominal molecular
    # masses.
    raw_hhv = phases["C"].gas_kg * methane_fraction * (393.52 + 2 * 285.83 - 74.87) / 12
    raw_hhv += phases["S"].gas_kg * (296.84 + 285.83 - 20.50) / 32
    soil = phase_heat["gas"] - raw_hhv
    gas = _LandfillGasLoad(
        pools,
        phases["N"].gas_kg,
        phases["S"].gas_kg,
        soil,
        {**phase_heat, "raw_gas_hhv_mj": raw_hhv, "heat_soil_mj": soil},
    )
    return _SBRLandfillLoad(material.mass_kg, elements, carbon, phases, leachate, gas)


def _sbr_dump_load(origin: _CarbonOrigin, site: _LandfillSite) -> _LandfillLoad:
    """Partition one principal kg and its filled compound at a selected open dump.

    Retained rubber and incorporated auxiliaries share the source's homogeneous
    matrix-release estimate. Original site climate and methane correction vary
    across the five selected dumps. Fixed fossil and tallow carbon retain their declared ingredient
    origins. No sanitary equipment or treatment enters
    this unmanaged function.
    """
    material = _sbr_compound_material(origin, principal_kg=1.0)
    # Compound 22 has zero phosphorus and sodium. These inputs retain the common source species
    # table.
    elements = {**material.elements_kg, "P": 0.0, "Na": 0.0}
    alpha = _climate_alpha(site)
    # Landfill2020 avgTK row 48 reaches its original cumulative unit cap at all
    # five admitted sites (smallest uncapped endpoint exceeds 9). Thus all Zn
    # remaining after short release enters the later phase at those sites.
    later = {**_later_release(alpha), "Zn": 1.0}
    # Doka (2003) estimates .01 matrix release over 100 years;
    # the original site climate scales that estimate.
    phases = {
        element: _element_phase(element, mass, 0.01 * mass, alpha, later[element])
        for element, mass in elements.items()
    }
    carbon_inputs: tuple[tuple[str, bool, _CarbonOrigin, float], ...] = (
        ("principal_sbr", True, origin, material.principal_carbon_kg[origin]),
        ("fossil_compounding_auxiliaries", False, "fossil", material.auxiliary_carbon_kg["fossil"]),
        (
            "tallow_compounding_auxiliaries",
            False,
            "biogenic",
            material.auxiliary_carbon_kg["biogenic"],
        ),
    )
    carbon = tuple(
        _LandfillCarbon(
            component,
            principal,
            carbon_origin,
            _element_phase("C", mass, 0.01 * mass, alpha, later["C"]),
        )
        for component, principal, carbon_origin, mass in carbon_inputs
    )
    toc = phases["C"].short_leachate_kg
    leachate = _LeachateLoad(
        {element: phase.short_leachate_kg for element, phase in phases.items()},
        tuple(
            _LeachateCarbon(
                pool.component, pool.principal, pool.origin, pool.phases.short_leachate_kg
            )
            for pool in carbon
        ),
        # Source organic H/C and O/C descriptors, separate from H/O.
        0.1225 * toc,
        0.64 * toc,
        toc,
        toc,
        # Landfill2020 B85/B86 raw COD/TOC and BOD5/TOC; no treatment is added.
        1.09375 * toc,
        0.25892857142857145 * toc,
    )
    # Source CH4/CO2 masses determine the carbon fraction; the dump's methane correction determines
    # gas formation.
    methane_share = (335.7 * 12 / 16) / (335.7 * 12 / 16 + 726.8 * 12 / 44)
    methane_share *= site.methane_correction
    gas_carbon = tuple(
        _GasPool(
            pool.component,
            pool.principal,
            pool.origin,
            pool.phases.gas_kg * methane_share,
            pool.phases.gas_kg * (1 - methane_share),
        )
        for pool in carbon
    )
    # Ingredient heat and separately charged elemental-sulfur heat follow their
    # respective carbon/sulfur phases, as in the existing sanitary calculation.
    energy = {
        "input_hhv_mj": material.carbon_heat_mj + material.sulfur_heat_mj,
        **{
            key: material.carbon_heat_mj * getattr(phases["C"], phase + "_kg") / elements["C"]
            + material.sulfur_heat_mj * getattr(phases["S"], phase + "_kg") / elements["S"]
            for key, phase in (
                ("gas_allocated_hhv_mj", "gas"),
                ("short_leachate_hhv_mj", "short_leachate"),
                ("later_leachate_hhv_mj", "later_leachate"),
                ("stored_hhv_mj", "stored"),
            )
        },
    }
    # Chase (1998), via NIST Chemistry WebBook HHV to liquid water, kJ/mol divided by nominal g C or
    # S/mol.
    # Raw-gas chemical energy is separate from dissipated soil/groundwater heat.
    raw_hhv = phases["C"].gas_kg * methane_share * (393.52 + 2 * 285.83 - 74.87) / 12
    raw_hhv += phases["S"].gas_kg * (296.84 + 285.83 - 20.50) / 32
    soil_heat = energy["gas_allocated_hhv_mj"] - raw_hhv
    gas = _LandfillGasLoad(
        gas_carbon,
        phases["N"].gas_kg,
        phases["S"].gas_kg,
        soil_heat,
        {**energy, "raw_gas_hhv_mj": raw_hhv, "heat_soil_mj": soil_heat},
    )
    return _LandfillLoad(material.mass_kg, 0.0, elements, phases, carbon, leachate, gas)
