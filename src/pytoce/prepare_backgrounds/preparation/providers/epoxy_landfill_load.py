"""Partition cured GY250/MXDA using the homogeneous paint class.

One retained principal kilogram carries its fixed fossil hardener. Doka's
estimated one-percent temperate/century degradation applies to the cured
matrix. Site climate and later releases follow the source functions.
"""

from math import fsum

from .epoxy_material import _EPOXY_HARDENER_RATIO, _epoxy_material
from .landfill_phases import (
    _climate_alpha,
    _element_phase,
    _LandfillCarbon,
    _LandfillLoad,
    _LandfillSite,
    _later_release,
)
from .leachate_load import _CarbonOrigin, _LeachateCarbon, _LeachateLoad
from .waste_gas_load import _GasPool, _LandfillGasLoad


def _epoxy_landfill_load(origin: _CarbonOrigin, site: _LandfillSite) -> _LandfillLoad:
    """Partition actual dry matrix per principal kg, retaining fixed fossil MXDA.

    Short-phase H/O and the wastewater model's empirical organic H/O
    descriptors remain distinct. No native pigment, donor moisture or upstream
    supply-proxy mass enters this reference.
    """
    material = _epoxy_material(
        (
            float(origin == "fossil"),
            float(origin == "biogenic"),
            float(origin == "atmospheric_dac"),
        ),
        _EPOXY_HARDENER_RATIO,
    )
    # The admitted network contains CHON. Explicit zero S/P/Na inputs retain the common Landfill2020
    # element/species table.
    elements = {**material.elements_kg, "S": 0.0, "P": 0.0, "Na": 0.0}
    alpha = _climate_alpha(site)
    later = _later_release(alpha)
    # Doka2003 Table6.1 and Doka2023 homogeneous-waste guidance: estimated .01
    # temperate/100-year degradation, shared by resin and incorporated hardener.
    phases = {
        element: _element_phase(element, mass, 0.01 * mass, alpha, later[element])
        for element, mass in elements.items()
    }
    carbon_inputs: tuple[tuple[str, bool, _CarbonOrigin, float], ...] = (
        ("principal_epoxy", True, origin, fsum(material.principal_carbon_kg)),
        ("cured_mxda", False, "fossil", material.fossil_auxiliary_carbon_kg),
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
        # Doka oxidation descriptors.
        0.1225 * toc,
        0.64 * toc,
        toc,
        toc,
        # Landfill2020 B85/B86 raw/bypass indicators; treated effluent retains
        # the separate 3.7/.91 ratios at the existing wastewater-load owner.
        1.09375 * toc,
        0.25892857142857145 * toc,
    )
    # Landfill2020 air&energy C74/K74, nominal CH4/CO2 gas composition. MCF
    # changes methane formation, independently of capture, flare or utilization.
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
    # Michel (via Muñoz et al. (2007), Eq. 9) HHV follows this homogeneous carbon partition.
    fraction = _element_phase("C", 1.0, 0.01, alpha, later["C"])
    energy = {
        "input_hhv_mj": material.heat_potential_mj,
        **{
            name: material.heat_potential_mj * value
            for name, value in (
                ("gas_allocated_hhv_mj", fraction.gas_kg),
                ("short_leachate_hhv_mj", fraction.short_leachate_kg),
                ("later_leachate_hhv_mj", fraction.later_leachate_kg),
                ("stored_hhv_mj", fraction.stored_kg),
            )
        },
    }
    # Chase (1998), via NIST Chemistry WebBook methane HHV to liquid water, kJ/mol / nominal g
    # C/mol.
    raw_hhv = fsum(pool.methane_carbon_kg for pool in gas_carbon) * (
        (393.52 + 2 * 285.83 - 74.87) / 12
    )
    soil_heat = energy["gas_allocated_hhv_mj"] - raw_hhv
    gas = _LandfillGasLoad(
        gas_carbon,
        phases["N"].gas_kg,
        0.0,
        soil_heat,
        {**energy, "raw_gas_hhv_mj": raw_hhv, "heat_soil_mj": soil_heat},
    )
    return _LandfillLoad(material.mass_kg, 0.0, elements, phases, carbon, leachate, gas)
