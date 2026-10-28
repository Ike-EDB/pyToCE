"""Keep actual PA maintenance chemistry inside its existing site boundary."""

from math import fsum
from pathlib import Path
from typing import cast

import pytest
from chemical_residue_fixtures import pa_maintenance_context

from pytoce.prepare_backgrounds.preparation.chemical_residue_incineration import (
    CHEMICAL_RESIDUE_FIXED_SOURCE,
    CHEMICAL_RESIDUE_SOURCE,
    _chemical_residue_source,
    _native_treatment,
)
from pytoce.prepare_backgrounds.preparation.chemical_residue_load import _residue_reference
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_load,
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_residues import (
    _pa_maintenance_residue_load,
)


@pytest.mark.parametrize("original", [True, False])
@pytest.mark.parametrize(
    "grade,shares,embedded",
    [
        ("pa6", (0.4, 0.35, 0.25), 0.0000075),
        ("pa66", (0.4, 0.35, 0.25), 0.0000075),
        ("pa6", (1.0, 0.0, 0.0), 0.0),
        ("pa6", (0.0, 1.0, 0.0), 0.0),
        ("pa6", (0.0, 0.0, 1.0), 0.0),
    ],
    ids=["pa6_mixed", "pa66_mixed", "fossil", "biogenic", "dac"],
)
def test_integrated_maintenance_preserves_actual_physical_pools(
    tmp_path: Path, original: bool, grade: str, shares: tuple[float, float, float], embedded: float
) -> None:
    """Check heat, water and elemental origins against independent source coefficients.

    Separate tests cover serialization and source binding.
    """
    with pa_maintenance_context(tmp_path, original=original) as (_, context, _):
        hwi = _native_treatment(context, CHEMICAL_RESIDUE_SOURCE, "pa_maintenance_test")
        fixed = _native_treatment(context, CHEMICAL_RESIDUE_FIXED_SOURCE, "pa_maintenance_test")
        reheat = (
            "heat, district or industrial, natural gas"
            if original
            else "heat, district or industrial"
        )
        source = _chemical_residue_source(
            context, hwi, fixed, CHEMICAL_RESIDUE_SOURCE, reheat, integrated_pa_site=True
        )
    principal_mass = 0.00002
    principal = cast(tuple[float, float, float], tuple(principal_mass * x for x in shares))
    finish = _pa_finish_reference(grade)
    inherited = _pa_finish_load(
        {name: embedded * quantity for name, quantity in finish.retained.compounds_kg.items()}
    )
    load = _pa_maintenance_residue_load(
        finish, principal_by_origin_kg=principal, embedded_dose_kg=embedded
    )
    reference = _residue_reference(load, source, integrated_pa_site=True)
    mass, elements = load.wet_mass_kg, load.active_elements_kg
    assert mass == pytest.approx(principal_mass + fsum(inherited.compounds_kg.values()))
    assert load.principal_carbon_kg == pytest.approx(tuple(x * 72 / 113 for x in principal))
    assert load.biogenic_carbon_kg == pytest.approx(
        inherited.biogenic_carbon_kg + principal[1] * 72 / 113
    )
    for element, numerator in (("C", 72), ("H", 11), ("N", 14), ("O", 16)):
        assert elements[element] == pytest.approx(
            principal_mass * numerator / 113 + inherited.elements_kg[element]
        )
    expected_formed = 9 * principal_mass * 11 / 113 + inherited.elements_kg["H"] * 18.015 / (
        2 * 1.008
    )
    expected_hhv = fsum(
        elements[name] * value
        for name, value in (
            ("C", 34.016),
            ("H", 124.265),
            ("O", -9.8324),
            ("N", 6.276),
            ("S", 19.079),
        )
    )
    assert load.carrier_water_kg == 0
    assert load.waste_formed_water_kg == pytest.approx(expected_formed)
    assert load.net_energy_mj == pytest.approx(expected_hhv - 2.4417 * expected_formed)
    assert fsum(elements.values()) == pytest.approx(mass)
    for field in ("oil_heat_mj", "oil_kg", "fuel_ncv_deficit_mj", "net_electricity_mj"):
        assert reference[field] == 0
    supplies, gases = reference["supplies"], reference["gases"]
    assert supplies["NG_fixed"] == supplies["NG_waste"] == 0
    assert reference["waste_heat_mj"] == load.net_energy_mj
    assert reference["site_water_kg"] == pytest.approx(5 * mass)
    nitrogen_ratio = elements["N"] / 0.01
    assert supplies["NH3_fixed"] == pytest.approx(0.002 * mass)
    assert supplies["NH3_waste"] == pytest.approx(0.0001 * nitrogen_ratio)
    assert supplies["TiO2"] == pytest.approx(0.0002 * mass + 0.0001 * nitrogen_ratio)
    assert supplies["V2O5"] == pytest.approx(0.00005 * mass + 0.00002 * nitrogen_ratio)
    ammonia_air = gases["Ammonia"] + 0.00002 * mass
    formed = expected_formed + 27 / 17 * (
        supplies["NH3_fixed"] + supplies["NH3_waste"] - ammonia_air
    )
    assert reference["formed_water_kg"] == pytest.approx(formed)
    assert reference["water_air_m3"] * 1000 == pytest.approx(0.5 * mass + formed)
    assert reference["water_surface_m3"] * 1000 == pytest.approx(4.5 * mass)
    assert fsum(reference["ash"].values()) == pytest.approx(reference["ash_mass_kg"])
    assert reference["ash"]["N"] == pytest.approx(
        0.01 * elements["N"] + reference["fresh"]["PAM"]["N"] + reference["fresh"]["TMT"]["N"]
    )
    assert (
        reference["pooled_N2_N_kg"]
        + gases["Dinitrogen monoxide"] * 28 / 44
        + gases["Nitrogen oxides"] * 14 / 46
        + gases["Cyanide"] * 14 / 26
    ) == pytest.approx(
        0.989 * elements["N"]
        + (supplies["NH3_fixed"] + supplies["NH3_waste"] - ammonia_air) * 14 / 17
    )
    # Native coefficients per wet kg: C=.3, N=.01, CO=1e-5, cyanide=4e-6;
    # DOC is 3e-5 to surface water and .002 to later groundwater.
    co_c = 0.00001 * mass * 12 / 28
    cn_c = 0.000004 * nitrogen_ratio * 12 / 26
    expected_destinations = {
        "carbon_dioxide": 0.9889898 * elements["C"] - co_c - cn_c,
        "carbon_monoxide": co_c,
        "cyanide": cn_c,
        "direct_surface_water": 0.0000101 * elements["C"],
        "ash_surface_water": (0.00003 / 0.3 - 0.0000101) * elements["C"],
        "ash_groundwater": 0.002 / 0.3 * elements["C"],
        "retained_in_ash": (0.0110001 - (0.00003 / 0.3 - 0.0000101) - 0.002 / 0.3) * elements["C"],
    }
    outcomes = reference["principal_carbon_outputs_kg"]
    assert set(outcomes) == set(expected_destinations)
    for destination, carbon in expected_destinations.items():
        assert outcomes[destination] == pytest.approx(
            tuple(carbon * part / elements["C"] for part in load.principal_carbon_kg)
        )
        assert all(part >= 0 for part in outcomes[destination])
    assert tuple(fsum(row[i] for row in outcomes.values()) for i in range(3)) == pytest.approx(
        load.principal_carbon_kg
    )
    assert reference["fresh"]["PAM"]["C"] > 0
    assert reference["ash"]["C"] > fsum(outcomes["retained_in_ash"])
    for gas in ("Carbon dioxide", "Carbon monoxide"):
        for suffix, indices in (("fossil", (0,)), ("non-fossil", (1, 2))):
            name = gas + ", " + suffix
            group_carbon = (
                elements["C"] - load.biogenic_carbon_kg - load.principal_carbon_kg[2]
                if suffix == "fossil"
                else load.biogenic_carbon_kg + load.principal_carbon_kg[2]
            )
            if group_carbon == 0:
                assert gases[name] == 0
                assert name not in reference["principal_gas_origin_fractions"]
            else:
                assert reference["principal_gas_origin_fractions"][name] == pytest.approx(
                    tuple(
                        part / group_carbon if i in indices else 0.0
                        for i, part in enumerate(load.principal_carbon_kg)
                    )
                )
