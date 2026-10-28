"""Allocate actual PA fiber and retained finish through Doka landfill phases.

Landfill2020's element-weighted short release and mean-MSW later kinetics
apply to the actual principal, fixed finish vector and moisture.
The organic and plastic/paint class proxies use 27% and 1% carbon release,
respectively.
This owner supplies loads; site emissions and complete downstream
receiving operations remain at their own inventory owners.
"""

from math import fsum

from .landfill_phases import (
    _climate_alpha,
    _element_phase,
    _LandfillCarbon,
    _LandfillLoad,
    _LandfillSite,
    _later_release,
)
from .leachate_load import _CarbonOrigin, _LeachateCarbon, _LeachateLoad
from .pa_finish_material import _pa_finish_load, _PAFinishLoad, _PAFinishReference
from .waste_gas_load import _GasPool, _LandfillGasLoad

# Doka (2003), Table 6.1: the selected temperate organic-class reference is 27%; its plastic/paint
# reference is 1%. These functions apply to the actual salt and dimer-rich batch.
_DEGRADATION = {
    "tridecyl_stearate": 0.27,
    "diglycerol_dioleate": 0.27,
    "oleth10": 0.27,
    "peg40_hco": 0.27,
    "tea_oleyl_phosphate": 0.27,
    "sodium_dodecanesulfonate": 0.01,
    "hmda_dimeramide": 0.01,
}
# Both PA6 and PA66 have the same nominal elemental mass fractions:
# C6H11NO/113 and C12H22N2O2/226. Preserve the principal MFA carbon authority
# (72/113), while the finish owner separately uses its admitted CIAAW atoms.
_PRINCIPAL_ELEMENTS = {"C": 72 / 113, "H": 11 / 113, "N": 14 / 113, "O": 16 / 113}


def _phase_energy(
    principal_kg: float, components: dict[str, _PAFinishLoad], alpha: float, carbon_later: float
) -> dict[str, float]:
    """Allocate actual component HHV by its own carbon release and remaining pool.

    PA energy uses the same component phase boundary as carbon. Michel's (via Doka)
    fuel equation estimates heat from CHONS; P/Na salts remain outside that estimate.
    """
    actual = [
        (
            {element: principal_kg * fraction for element, fraction in _PRINCIPAL_ELEMENTS.items()},
            0.01,
        ),
        *((load.elements_kg, _DEGRADATION[name]) for name, load in components.items()),
    ]
    energy_terms: dict[str, list[float]] = {
        name: []
        for name in (
            "input_hhv_mj",
            "gas_allocated_hhv_mj",
            "short_leachate_hhv_mj",
            "later_leachate_hhv_mj",
            "stored_hhv_mj",
        )
    }
    for elements, degradation in actual:
        # Michel (via Doka2021 WWcalc DT18): MJ/kg organic C,H,O,N,S, respectively. Apply the
        # selected component's actual atoms.
        energy = fsum(
            elements.get(element, 0.0) * coefficient
            for element, coefficient in (
                ("C", 34.016),
                ("H", 124.265),
                ("O", -9.8324),
                ("N", 6.276),
                ("S", 19.079),
            )
        )
        fractions = _element_phase("C", 1.0, degradation, alpha, carbon_later)
        energy_terms["input_hhv_mj"].append(energy)
        energy_terms["gas_allocated_hhv_mj"].append(energy * fractions.gas_kg)
        energy_terms["short_leachate_hhv_mj"].append(energy * fractions.short_leachate_kg)
        energy_terms["later_leachate_hhv_mj"].append(energy * fractions.later_leachate_kg)
        energy_terms["stored_hhv_mj"].append(energy * fractions.stored_kg)
    return {name: fsum(terms) for name, terms in energy_terms.items()}


def _pa_landfill_load(
    finish: _PAFinishReference,
    *,
    principal_by_origin_kg: dict[_CarbonOrigin, float],
    finish_dose_kg: float,
    water_kg: float,
    site: _LandfillSite,
) -> _LandfillLoad:
    """Partition actual PA, fixed retained finish and water at one admitted site.

    Principal values are actual polymer kg by origin; their sum is P. The
    scalar dose T multiplies the finish owner's retained kg/kg-principal
    vector, after the caller allocates surface and inherited material. No
    coating is applied here, and no upper T/P ratio is imposed. The calculation uses the prepared
    source and material values.
    """
    principal = fsum(principal_by_origin_kg.values())
    components = {
        name: _pa_finish_load({name: finish_dose_kg * amount})
        for name, amount in finish.retained.compounds_kg.items()
    }
    # The admitted PA/finish material owns its elemental coordinates. The shared release table also
    # serves glass-filled resin and zinc-filled SBR.
    elements = {
        element: principal * _PRINCIPAL_ELEMENTS.get(element, 0.0)
        + fsum(load.elements_kg[element] for load in components.values())
        for element in finish.retained.elements_kg
    }
    # Principal PA retains the admitted one-percent plastic-class reference.
    # Sum the actual degraded allocands BEFORE element-specific release;
    # independently capping each sodium-bearing ingredient is incorrect.
    degraded = {
        element: 0.01 * principal * _PRINCIPAL_ELEMENTS.get(element, 0.0)
        + fsum(load.elements_kg[element] * _DEGRADATION[name] for name, load in components.items())
        for element in elements
    }
    alpha = _climate_alpha(site)
    later = _later_release(alpha)
    phases = {
        element: _element_phase(element, mass, degraded[element], alpha, later[element])
        for element, mass in elements.items()
    }
    carbon_inputs = [
        (_LeachateCarbon("principal_pa", True, origin, amount * 72 / 113), 0.01)
        for origin, amount in principal_by_origin_kg.items()
    ]
    for name, load in components.items():
        carbon_inputs.extend(
            (
                (_LeachateCarbon(name, False, "fossil", load.fossil_carbon_kg), _DEGRADATION[name]),
                (
                    _LeachateCarbon(name, False, "biogenic", load.biogenic_carbon_kg),
                    _DEGRADATION[name],
                ),
            )
        )
    # C has J=1 and every component d<=.27, so its cap is inactive at all six climates. Each carbon
    # origin's short phase follows its own degraded allocand; later release follows its own
    # remaining pool.
    carbon = tuple(
        _LandfillCarbon(
            pool.component,
            pool.principal,
            pool.origin,
            _element_phase("C", pool.carbon_kg, pool.carbon_kg * d, alpha, later["C"]),
        )
        for pool, d in carbon_inputs
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
        # Doka organic H/C and O/C are oxidation descriptors.
        0.1225 * toc,
        0.64 * toc,
        toc,
        toc,
        # Landfill2020 MSWLF B85/B86: source raw-leachate COD/TOC and BOD/TOC.
        1.09375 * toc,
        0.25892857142857145 * toc,
    )
    # Landfill2020 air & energy C74, K74: default 335.7 g CH4 and 726.8 g CO2 per m3 set raw gas-C
    # fractions. The source MCF transfers methane C to CO2 C and preserves total gas carbon.
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
    energy = _phase_energy(principal, components, alpha, later["C"])
    # Chase (1998), via NIST Chemistry WebBook combustion to liquid water, kJ/mol divided by nominal
    # g/mol: kg methane C uses /12; kg sulfide S uses /32. These produce MJ. Nitrogen in the
    # admitted raw landfill gas is N2.
    raw_gas_hhv = fsum(pool.methane_carbon_kg for pool in gas_carbon) * (
        (393.52 + 2 * 285.83 - 74.87) / 12
    )
    raw_gas_hhv += phases["S"].gas_kg * (296.84 + 285.83 - 20.50) / 32
    soil_heat = energy["gas_allocated_hhv_mj"] - raw_gas_hhv
    # No source-floor substitute: the accepted principal/complete-finish
    # endpoints have nonnegative decomposition heat at all six sites. Carbon
    # phases/HHV are additive on their nonnegative P/T family, so mixtures do too.
    gas = _LandfillGasLoad(
        gas_carbon,
        phases["N"].gas_kg,
        phases["S"].gas_kg,
        soil_heat,
        {**energy, "raw_gas_hhv_mj": raw_gas_hhv, "heat_soil_mj": soil_heat},
    )
    dry_finish = finish_dose_kg * fsum(finish.retained.compounds_kg.values())
    return _LandfillLoad(
        fsum((principal, dry_finish, water_kg)),
        water_kg,
        elements,
        phases,
        carbon,
        leachate,
        gas,
    )
