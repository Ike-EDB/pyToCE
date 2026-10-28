"""Actual cured UPR/glass phases under the homogeneous plastic-waste reference.

The material owner supplies the dry retained article per principal kilogram.
Landfill2020 supplies its climate, element release and later mineral kinetics;
the same composition applies to homogeneous manufacturing trim. Total carbon includes the declared
nominal/tracked reference difference, while the
principal diagnostic remains limited to tracked carbon.
"""

from math import fsum

from pytoce.download_data.sources.ecoinvent_materials import ConversionRequirement

from .landfill_phases import (
    _climate_alpha,
    _element_phase,
    _LandfillCarbon,
    _LandfillLoad,
    _LandfillSite,
    _later_release,
    _mineral_later_release,
)
from .leachate_load import _CarbonOrigin, _LeachateCarbon, _LeachateLoad
from .upr_material import _upr_material
from .waste_gas_load import _HHV_CH4, _GasPool, _LandfillGasLoad


def _upr_landfill_load(
    conversion: ConversionRequirement,
    origin: _CarbonOrigin,
    site: _LandfillSite,
    *,
    carbonate_end_year: float,
) -> _LandfillLoad:
    """Partition one principal kg and its auxiliaries at the selected site.

    The site supplies the rounded carbonate-end year. Doka's estimated 1%
    century degradation applies to the whole cured composite, followed by
    element-specific reprecipitation and later release.
    """
    material = _upr_material(
        conversion,
        (
            float(origin == "fossil"),
            float(origin == "biogenic"),
            float(origin == "atmospheric_dac"),
        ),
    )
    # Both matrix and embedded mineral atoms enter this source's homogeneous
    # phase definition. The fixed article contains no N/S/P/Na. Bulk oxide O
    # remains a atom pool, separate from organic heat eligibility.
    elements = {
        element: fsum(
            (
                material.burnable_elements_kg.get(element, 0.0),
                material.inert_elements_kg.get(element, 0.0),
            )
        )
        for element in ("C", "H", "O", "N", "S", "P", "Na", "B", "Si", "Ca", "Al", "Co")
    }
    alpha = _climate_alpha(site)
    later = {**_later_release(alpha), **_mineral_later_release(alpha, carbonate_end_year)}
    phases = {
        element: _element_phase(element, mass, 0.01 * mass, alpha, later[element])
        for element, mass in elements.items()
    }
    # The positive reference difference retains the principal's origin but is outside its tracked
    # diagnostic. The three pools sum to nominal total carbon.
    carbon_inputs: tuple[tuple[str, bool, _CarbonOrigin, float], ...] = (
        ("principal_upr", True, origin, fsum(material.principal_carbon_kg)),
        (
            "upr_principal_reference_difference",
            False,
            origin,
            fsum(material.principal_carbon_bridge_kg),
        ),
        ("upr_retained_auxiliaries", False, "fossil", material.fossil_auxiliary_carbon_kg),
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
        # Doka oxidation descriptors supply H/O. Raw/bypass B85/B86 indicators and WWT effluent use
        # their respective conventions.
        0.1225 * toc,
        0.64 * toc,
        toc,
        toc,
        1.09375 * toc,
        0.25892857142857145 * toc,
    )
    # Landfill2020 air&energy C74/K74: nominal gas masses in g/m3. MCF
    # changes methane formation, separately from capture and treatment shares.
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
    # The admitted component HHV excludes bulk oxides and keeps the separate
    # whole-organosiloxane proxy. Only its homogeneous carbon-phase allocation
    # is transferred here; no Michel (via Muñoz et al. (2007), Eq. 9) equation is applied to total
    # oxide O.
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
    # Shared Chase (1998), via NIST Chemistry WebBook methane HHV is MJ/kg CH4; 16/12 converts
    # carbon kg.
    raw_hhv = fsum(pool.methane_carbon_kg for pool in gas_carbon) * 16 / 12 * _HHV_CH4
    # Doka2003 III section6.5.1 transfers liberated HHV. After reserving the
    # short-leachate HHV at its receiving owner, this net site allocation is
    # slightly negative for sanitary UPR. It reports the net site allocation
    # after short-leachate heat is reserved. Preserve this signed residual.
    soil_heat = energy["gas_allocated_hhv_mj"] - raw_hhv
    gas = _LandfillGasLoad(
        gas_carbon,
        phases["N"].gas_kg,
        phases["S"].gas_kg,
        soil_heat,
        {**energy, "raw_gas_hhv_mj": raw_hhv, "heat_soil_mj": soil_heat},
    )
    return _LandfillLoad(
        material.mass_kg, material.water_kg, elements, phases, carbon, leachate, gas
    )
