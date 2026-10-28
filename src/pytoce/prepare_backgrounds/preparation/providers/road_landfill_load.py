"""Actual road-marking fractions in one sanitary or unmanaged disposal function."""

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
from .road_material import _road_material
from .waste_gas_load import _HHV_CH4, _GasPool, _LandfillGasLoad


def _road_landfill_load(
    conversion: ConversionRequirement,
    origin: _CarbonOrigin,
    site: _LandfillSite,
    *,
    carbonate_end_year: float,
) -> _LandfillLoad:
    """Receive one principal kg with its organic, mineral and limestone pools.

    Doka (2023) combines polymer, glass and limestone in one disposal inventory,
    with temperate-century accessibility of 0.01, 0.01 and 0.05, respectively.
    Glass also represents embedded bulk minerals. Climate and later kinetics
    follow the selected site. Geological C remains outside organic-carbon fate
    calculations.
    """
    material = _road_material(
        conversion,
        (
            float(origin == "fossil"),
            float(origin == "biogenic"),
            float(origin == "atmospheric_dac"),
        ),
    )
    # The selected organic-fate convention tracks limestone carbon separately, retaining its actual
    # mass in the dedicated field.
    limestone = {e: q for e, q in material.limestone_elements_kg.items() if e != "C"}
    elements = {
        element: fsum(
            (
                material.burnable_elements_kg.get(element, 0.0),
                material.mineral_elements_kg.get(element, 0.0),
                limestone.get(element, 0.0),
            )
        )
        for element in ("C", "H", "O", "N", "S", "P", "Na", "Si", "Ca", "Al", "Ti", "K", "Mg")
    }
    alpha = _climate_alpha(site)
    later = {**_later_release(alpha), **_mineral_later_release(alpha, carbonate_end_year)}
    phases = {
        element: _element_phase(
            element,
            mass,
            # Doka's three source accessibility fractions apply before the elemental H/J transfer.
            0.01
            * (
                material.burnable_elements_kg.get(element, 0.0)
                + material.mineral_elements_kg.get(element, 0.0)
            )
            + 0.05 * limestone.get(element, 0.0),
            alpha,
            later[element],
        )
        for element, mass in elements.items()
    }
    carbon_inputs: tuple[tuple[str, bool, _CarbonOrigin, float], ...] = (
        ("principal_road_binder", True, origin, fsum(material.principal_carbon_kg)),
        ("road_retained_auxiliaries", False, "fossil", material.fossil_auxiliary_carbon_kg),
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
        # Landfill2020 B85/B86 and its organic H/O descriptors are kg per actual organic TOC.
        0.1225 * toc,
        0.64 * toc,
        toc,
        toc,
        1.09375 * toc,
        0.25892857142857145 * toc,
    )
    # Landfill2020 air&energy C74/K74: source gas g/m3, converted to C.
    # Dump MCF changes methane formation, independently of any collection.
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
    # Only organic pools have gross heat, all with the same .01 accessibility.
    # No mineral oxygen enters a heat correlation or calcination-energy duty.
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
    # 16/12 converts actual methane C to CH4 kg; the gas owner retains its
    # chemical HHV. Site heat is the signed Doka allocation after that transfer.
    raw_hhv = fsum(pool.methane_carbon_kg for pool in gas_carbon) * 16 / 12 * _HHV_CH4
    soil_heat = energy["gas_allocated_hhv_mj"] - raw_hhv
    gas = _LandfillGasLoad(
        gas_carbon,
        phases["N"].gas_kg,
        phases["S"].gas_kg,
        soil_heat,
        {**energy, "raw_gas_hhv_mj": raw_hhv, "heat_soil_mj": soil_heat},
    )
    return _LandfillLoad(
        material.mass_kg,
        material.water_kg,
        elements,
        phases,
        carbon,
        leachate,
        gas,
        geological_carbon_kg=material.geological_carbon_kg,
    )
