"""Actual PA/finish landfill phases retain elements, carbon origins and energy."""

from math import exp, fsum, log

import pytest

from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_load,
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_load import (
    _pa_landfill_load,
)

# RoW municipal site and the five admitted GLO open-dump climates. Doka2020
# control board E133 gives 1-2.0604/7 for the five 7 m open dumps.
_SITES = (
    (16.743579766537, 1250.84108527132, 1.0),
    *((8.0, precipitation, 1 - 2.0604 / 7) for precipitation in (350, 550, 900, 1100, 1900)),
)
_ORGANIC_CLASS = {
    "tridecyl_stearate",
    "diglycerol_dioleate",
    "oleth10",
    "peg40_hco",
    "tea_oleyl_phosphate",
}


@pytest.mark.parametrize("grade", ("pa6", "pa66"))
@pytest.mark.parametrize("temperature,precipitation,correction", _SITES)
def test_actual_pa_phases_keep_source_kinetics_elements_and_carbon_origins(
    grade: str, temperature: float, precipitation: float, correction: float
) -> None:
    """Close mixed material at every source climate with origin-specific degradation."""
    finish = _pa_finish_reference(grade)
    # Two actual kg PA contain three principal origins; a separate five-dose
    # finish vector represents inherited material plus the current surface.
    actual = _pa_landfill_load(
        finish,
        principal_by_origin_kg={"fossil": 0.5, "biogenic": 0.75, "atmospheric_dac": 0.75},
        finish_dose_kg=5,
        water_kg=0.4,
        site=_LandfillSite(temperature, precipitation, correction),
    )
    assert actual.wet_mass_kg == pytest.approx(
        2.4 + 5 * fsum(finish.retained.compounds_kg.values())
    )
    assert actual.water_kg == 0.4
    assert fsum(actual.elements_kg.values()) + actual.water_kg == pytest.approx(actual.wet_mass_kg)
    # Evaluate the published climate equation with mean-MSW60/220 calibration.
    p = 60 * (1 - exp(-29.372386439548432 * precipitation / 1000))
    t = 60 * (1 - exp(-0.3 * (temperature + 15)))
    alpha = (log(1 - p / 220) / log(1 - 60 / 220)) * (log(1 - t / 220) / log(1 - 60 / 220))
    # The separately tested ingredient calculation supplies elemental masses only;
    # five finish ingredients use the 27% source class and two use the 1% class.
    degraded = _pa_finish_load(
        {
            name: 5 * amount * (0.27 if name in _ORGANIC_CLASS else 0.01)
            for name, amount in finish.retained.compounds_kg.items()
        }
    ).elements_kg
    nominal = {"C": 72 / 113, "H": 11 / 113, "N": 14 / 113, "O": 16 / 113}
    factors = {
        "C": (0.970873786407767, 1),
        "H": (0.970873786407767, 1),
        "O": (0.970873786407767, 1),
        "N": (0.06438442211055276, 2.4976427240513064),
        "P": (0, 0.0559165999037902),
        "S": (0.1490935840646033, 0.437649447721238),
        "Na": (0.00024993751562109475, 4.13932455312919),
    }
    for element, phase in actual.phases.items():
        incoming = 2 * nominal.get(element, 0) + 5 * finish.retained.elements_kg[element]
        gas_share, release = factors[element]
        mobilized = alpha * (degraded[element] + 0.02 * nominal.get(element, 0)) * release
        # The complete-vector Na ratio remains below its cap across doses and P/T mixtures.
        assert mobilized <= incoming
        assert phase.input_kg == pytest.approx(incoming)
        assert phase.gas_kg == pytest.approx(mobilized * gas_share)
        assert phase.short_leachate_kg == pytest.approx(mobilized * (1 - gas_share))
        assert fsum(
            (phase.gas_kg, phase.short_leachate_kg, phase.later_leachate_kg, phase.stored_kg)
        ) == (pytest.approx(incoming, rel=3e-15))
        assert phase.stored_kg >= 0
    # Mean-MSW calibration determines later N/P release. Phosphorus retains its specified
    # partial-release fraction.
    k_n = (1 - (1 - 0.20199069396374611) ** alpha) * 2.4976427240513059
    k_p = (1 - (1 - 0.27120402960925999) ** alpha) * 0.0559165999037902
    n_q = 1 - (1 - k_n) ** (0.061771875011314883 * 599)
    p_q = k_p * 0.0072978929039903915 * 599 / (1 - k_p)
    for element, q in (("N", n_q), ("P", p_q)):
        phase = actual.phases[element]
        remaining = phase.input_kg - phase.gas_kg - phase.short_leachate_kg
        assert phase.later_leachate_kg == pytest.approx(remaining * q, rel=5e-14)
        assert phase.stored_kg == pytest.approx(remaining * (1 - q), rel=5e-14)
    assert 0 < p_q < 1
    assert actual.phases["Na"].stored_kg == 0
    carbon = actual.phases["C"]
    for field in ("input_kg", "gas_kg", "short_leachate_kg", "later_leachate_kg", "stored_kg"):
        assert fsum(getattr(row.phases, field) for row in actual.carbon) == pytest.approx(
            getattr(carbon, field), rel=3e-15, abs=1e-18
        )
    for row in actual.carbon:
        d = 0.27 if row.component in _ORGANIC_CLASS else 0.01
        mobilized = row.phases.input_kg * d * alpha
        assert row.phases.gas_kg == pytest.approx(mobilized * factors["C"][0])
        assert row.phases.later_leachate_kg == pytest.approx(row.phases.input_kg - mobilized)
        assert row.principal == (row.component == "principal_pa")
    principal_c = {row.origin: row.phases.input_kg for row in actual.carbon if row.principal}
    assert principal_c == pytest.approx(
        {
            "fossil": 0.5 * 72 / 113,
            "biogenic": 0.75 * 72 / 113,
            "atmospheric_dac": 0.75 * 72 / 113,
        }
    )
    for origin, amount in (
        ("fossil", finish.retained.fossil_carbon_kg),
        ("biogenic", finish.retained.biogenic_carbon_kg),
    ):
        assert fsum(
            row.phases.input_kg
            for row in actual.carbon
            if not row.principal and row.origin == origin
        ) == (pytest.approx(5 * amount))
    assert all(row.principal or row.origin != "atmospheric_dac" for row in actual.carbon)
    methane_share = correction * (335.7 / 16) / (335.7 / 16 + 726.8 / 44)
    assert fsum(row.methane_carbon_kg for row in actual.gas.carbon) == pytest.approx(
        carbon.gas_kg * methane_share
    )
    assert fsum(
        row.methane_carbon_kg + row.carbon_dioxide_carbon_kg for row in actual.gas.carbon
    ) == (pytest.approx(carbon.gas_kg))
    water = actual.leachate
    assert water.elements_kg == {e: x.short_leachate_kg for e, x in actual.phases.items()}
    assert fsum(row.carbon_kg for row in water.carbon) == pytest.approx(water.toc_kg)
    assert water.doc_kg == water.toc_kg == carbon.short_leachate_kg
    assert water.organic_hydrogen_kg == pytest.approx(0.1225 * water.toc_kg)
    assert water.organic_oxygen_kg == pytest.approx(0.64 * water.toc_kg)
    assert water.cod_kg_o2 == pytest.approx(1.09375 * water.toc_kg)
    assert water.bod5_kg_o2 == pytest.approx(0.25892857142857145 * water.toc_kg)
    assert water.chloride_kg == 0


@pytest.mark.parametrize("grade", ("pa6", "pa66"))
@pytest.mark.parametrize("temperature,precipitation,correction", _SITES)
def test_principal_and_complete_finish_energy_endpoints_bound_every_mixture(
    grade: str, temperature: float, precipitation: float, correction: float
) -> None:
    """Keep nonnegative soil heat and complete HHV closure on the entire P/T family."""
    finish = _pa_finish_reference(grade)
    site = _LandfillSite(temperature, precipitation, correction)
    principal = _pa_landfill_load(
        finish, principal_by_origin_kg={"fossil": 1}, finish_dose_kg=0, water_kg=0, site=site
    )
    auxiliary = _pa_landfill_load(
        finish, principal_by_origin_kg={}, finish_dose_kg=1, water_kg=0, site=site
    )
    mixture = _pa_landfill_load(
        finish, principal_by_origin_kg={"biogenic": 2}, finish_dose_kg=50, water_kg=1, site=site
    )
    # Component HHV follows the admitted empirical CHONS equation.
    heat_factors = {"C": 34.016, "H": 124.265, "O": -9.8324, "N": 6.276, "S": 19.079}
    principal_hhv = (72 * 34.016 + 11 * 124.265 + 14 * 6.276 - 16 * 9.8324) / 113
    degraded_finish_hhv = 0.0
    for name, amount in finish.retained.compounds_kg.items():
        atoms = _pa_finish_load({name: amount}).elements_kg
        component_hhv = fsum(atoms[e] * factor for e, factor in heat_factors.items())
        degraded_finish_hhv += (0.27 if name in _ORGANIC_CLASS else 0.01) * component_hhv
    p = 60 * (1 - exp(-29.372386439548432 * precipitation / 1000))
    t = 60 * (1 - exp(-0.3 * (temperature + 15)))
    alpha = log(1 - p / 220) * log(1 - t / 220) / log(1 - 60 / 220) ** 2
    for actual, principal_kg, dose in ((principal, 1, 0), (auxiliary, 0, 1), (mixture, 2, 50)):
        energy = actual.gas.decomposition_trace
        expected_hhv = fsum(actual.elements_kg[e] * factor for e, factor in heat_factors.items())
        # Apply each source class to component HHV. Carbon's later-release fraction is one.
        degraded_hhv = alpha * (0.01 * principal_kg * principal_hhv + dose * degraded_finish_hhv)
        assert energy["gas_allocated_hhv_mj"] == pytest.approx(degraded_hhv * 0.970873786407767)
        assert energy["short_leachate_hhv_mj"] == pytest.approx(
            degraded_hhv * (1 - 0.970873786407767)
        )
        assert energy["later_leachate_hhv_mj"] == pytest.approx(expected_hhv - degraded_hhv)
        assert energy["stored_hhv_mj"] == 0
        assert energy["input_hhv_mj"] == pytest.approx(expected_hhv)
        assert fsum(
            energy[name]
            for name in (
                "gas_allocated_hhv_mj",
                "short_leachate_hhv_mj",
                "later_leachate_hhv_mj",
                "stored_hhv_mj",
            )
        ) == pytest.approx(expected_hhv)
        raw = (
            fsum(pool.methane_carbon_kg for pool in actual.gas.carbon)
            * (393.52 + 2 * 285.83 - 74.87)
            / 12
        )
        raw += actual.phases["S"].gas_kg * (296.84 + 285.83 - 20.50) / 32
        assert energy["raw_gas_hhv_mj"] == pytest.approx(raw)
        assert actual.gas.decomposition_heat_mj > 0
        assert actual.gas.decomposition_heat_mj + raw == pytest.approx(
            energy["gas_allocated_hhv_mj"]
        )
        assert actual.gas.nitrogen_kg == actual.phases["N"].gas_kg
        assert actual.gas.sulfur_kg == actual.phases["S"].gas_kg
    # Source climate and fixed-vector classes make initial phases/energy
    # additive in P/T, though the complete receiving LCI is nonlinear.
    for field, value in mixture.gas.decomposition_trace.items():
        assert value == pytest.approx(
            2 * principal.gas.decomposition_trace[field]
            + 50 * auxiliary.gas.decomposition_trace[field],
            rel=3e-15,
            abs=1e-18,
        )
    for element, phase in principal.phases.items():
        if element in {"S", "P", "Na"}:
            assert (
                phase.input_kg
                == phase.gas_kg
                == phase.short_leachate_kg
                == phase.later_leachate_kg
                == phase.stored_kg
                == 0
            )
    assert all(row.phases.input_kg == 0 for row in principal.carbon if not row.principal)


def test_large_inherited_dose_preserves_element_phases_and_named_pools() -> None:
    """Scale the complete finish vector at each requested dose."""
    finish = _pa_finish_reference("pa6")
    site = _LandfillSite(*_SITES[0])
    first = _pa_landfill_load(
        finish, principal_by_origin_kg={"fossil": 0.2}, finish_dose_kg=7, water_kg=0.1, site=site
    )
    scaled = _pa_landfill_load(
        finish, principal_by_origin_kg={"fossil": 200}, finish_dose_kg=7000, water_kg=100, site=site
    )
    assert scaled.wet_mass_kg == pytest.approx(1000 * first.wet_mass_kg)
    assert len(first.carbon) == len(scaled.carbon)
    for left, right in zip(first.carbon, scaled.carbon, strict=True):
        assert (left.component, left.principal, left.origin) == (
            right.component,
            right.principal,
            right.origin,
        )
        assert right.phases.input_kg == pytest.approx(1000 * left.phases.input_kg)
        assert right.phases.gas_kg == pytest.approx(1000 * left.phases.gas_kg)
        assert right.phases.later_leachate_kg == pytest.approx(1000 * left.phases.later_leachate_kg)
    for element in first.phases:
        assert scaled.phases[element].short_leachate_kg == pytest.approx(
            1000 * first.phases[element].short_leachate_kg
        )
    assert scaled.gas.decomposition_heat_mj == pytest.approx(1000 * first.gas.decomposition_heat_mj)
