"""Cured epoxy phases preserve elemental mass, carbon origins and source-class limits."""

from math import exp, fsum, log

import pytest

from pytoce.prepare_backgrounds.preparation.providers.epoxy_landfill_load import (
    _epoxy_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import _CarbonOrigin


@pytest.mark.parametrize("origin", ("fossil", "biogenic", "atmospheric_dac"))
def test_cured_epoxy_uses_whole_matrix_class_and_separate_hardener(origin: _CarbonOrigin) -> None:
    """Apply published source phases to an independent GY250/MXDA molecular oracle."""
    ratio = 15.44 / 84.56
    repeat = (2 * 186 - 340) / 284
    principal_carbon = 12 * (21 + 18 * repeat) / 372
    hardener_carbon = ratio * 96 / 136
    elements = {
        "C": principal_carbon + hardener_carbon,
        "H": (24 + 20 * repeat) / 372 + ratio * 12 / 136,
        "O": 16 * (4 + 3 * repeat) / 372,
        "N": ratio * 28 / 136,
        "S": 0.0,
        "P": 0.0,
        "Na": 0.0,
    }
    sites = (
        (16.743579766537, 1250.84108527132, 1.0),
        *((8.0, p, 1 - 2.0604 / 7) for p in (350, 550, 900, 1100, 1900)),
    )
    for temperature, precipitation, correction in sites:
        actual = _epoxy_landfill_load(origin, _LandfillSite(temperature, precipitation, correction))
        assert actual.wet_mass_kg == pytest.approx(1 + ratio)
        assert actual.water_kg == 0
        assert actual.elements_kg == pytest.approx(elements)
        assert fsum(actual.elements_kg.values()) == pytest.approx(actual.wet_mass_kg)
        # Evaluate the Landfill2020 climate equation from source mean-MSW60/220 calibrations.
        p = 60 * (1 - exp(-29.372386439548432 * precipitation / 1000))
        t = 60 * (1 - exp(-0.3 * (temperature + 15)))
        alpha = log(1 - p / 220) * log(1 - t / 220) / log(1 - 60 / 220) ** 2
        for element in ("C", "H", "O"):
            phase = actual.phases[element]
            assert phase.gas_kg == pytest.approx(elements[element] * 0.01 * alpha * 100 / 103)
            assert phase.short_leachate_kg == pytest.approx(
                elements[element] * 0.01 * alpha * 3 / 103
            )
            assert phase.stored_kg == 0
        nitrogen = actual.phases["N"]
        assert nitrogen.gas_kg == pytest.approx(
            elements["N"] * 0.01 * alpha * 2.4976427240513064 * 0.06438442211055276
        )
        assert nitrogen.stored_kg >= 0
        for phase in actual.phases.values():
            assert fsum(
                (phase.gas_kg, phase.short_leachate_kg, phase.later_leachate_kg, phase.stored_kg)
            ) == pytest.approx(phase.input_kg)
        assert [(pool.component, pool.principal, pool.origin) for pool in actual.carbon] == [
            ("principal_epoxy", True, origin),
            ("cured_mxda", False, "fossil"),
        ]
        assert actual.carbon[0].phases.input_kg == pytest.approx(principal_carbon)
        assert actual.carbon[1].phases.input_kg == pytest.approx(hardener_carbon)
        for field in ("gas_kg", "short_leachate_kg", "later_leachate_kg"):
            assert getattr(actual.carbon[0].phases, field) / principal_carbon == pytest.approx(
                getattr(actual.carbon[1].phases, field) / hardener_carbon
            )
        methane_fraction = (335.7 * 12 / 16) / (335.7 * 12 / 16 + 726.8 * 12 / 44) * correction
        assert fsum(pool.methane_carbon_kg for pool in actual.gas.carbon) == pytest.approx(
            actual.phases["C"].gas_kg * methane_fraction
        )
        assert fsum(
            pool.methane_carbon_kg + pool.carbon_dioxide_carbon_kg for pool in actual.gas.carbon
        ) == pytest.approx(actual.phases["C"].gas_kg)
        energy = actual.gas.decomposition_trace
        assert energy["input_hhv_mj"] == pytest.approx(
            fsum(
                energy[key]
                for key in (
                    "gas_allocated_hhv_mj",
                    "short_leachate_hhv_mj",
                    "later_leachate_hhv_mj",
                    "stored_hhv_mj",
                )
            )
        )
        assert actual.gas.decomposition_heat_mj >= 0
        assert actual.leachate.elements_kg["H"] == actual.phases["H"].short_leachate_kg
        assert actual.leachate.organic_hydrogen_kg == pytest.approx(0.1225 * actual.leachate.toc_kg)
        assert actual.leachate.organic_oxygen_kg == pytest.approx(0.64 * actual.leachate.toc_kg)
        assert actual.leachate.cod_kg_o2 == pytest.approx(1.09375 * actual.leachate.toc_kg)
        assert actual.leachate.bod5_kg_o2 == pytest.approx(
            0.25892857142857145 * actual.leachate.toc_kg
        )
