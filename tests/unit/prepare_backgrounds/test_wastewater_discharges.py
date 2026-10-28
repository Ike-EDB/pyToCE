"""Keep wastewater releases on one basis with separate principal carbon."""

from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _ROW_WASTEWATER_PROFILE,
    _LeachateCarbon,
    _LeachateLoad,
    _wastewater_load,
    _WastewaterLoad,
    _WastewaterProcessProfile,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_treatment import (
    _wastewater_discharges,
)


def _mixed_origin_water(profile: _WastewaterProcessProfile, factor: float) -> _WastewaterLoad:
    """Treat one nominal PA monomer kg in nine water kg, with explicit origin pools."""
    # The admitted principal caprolactam convention is C6H11NO, nominal
    # 72 C + 11 H + 14 N + 16 O = 113 g/mol. These are input atoms;
    # the source's organic H/O and BOD descriptors below are independent.
    carbon = factor * 72 / 113
    load = _LeachateLoad(
        elements_kg={
            "C": carbon,
            "H": factor * 11 / 113,
            "N": factor * 14 / 113,
            "O": factor * 16 / 113,
            "P": 0.0,
            "S": 0.0,
            "Cl": 0.0,
            "Na": 0.0,
            "Mg": 0.0,
            "Zn": 0.0,
        },
        # Invented accounting mixture: 60% principal, 40% auxiliary carbon.
        # Biological removal sees the same material, while diagnostics must
        # distinguish the origins inside each shared native CO2 flow.
        carbon=(
            _LeachateCarbon("principal_fossil", True, "fossil", 0.20 * carbon),
            _LeachateCarbon("principal_biogenic", True, "biogenic", 0.30 * carbon),
            _LeachateCarbon("principal_dac", True, "atmospheric_dac", 0.10 * carbon),
            _LeachateCarbon("auxiliary_fossil", False, "fossil", 0.15 * carbon),
            _LeachateCarbon("auxiliary_biogenic", False, "biogenic", 0.25 * carbon),
        ),
        organic_hydrogen_kg=0.1225 * carbon,
        organic_oxygen_kg=0.64 * carbon,
        toc_kg=carbon,
        doc_kg=0.68 * carbon,
        # Nominal oxidation to CO2, NH3 and water consumes 7.5 O2:
        # 7.5*32/113 kg O2/kg input; the source BOD is kg O2/kg C.
        cod_kg_o2=factor * 240 / 113,
        bod5_kg_o2=1.7586552173908316 * carbon,
    )
    return _wastewater_load(
        load,
        # Ten kg actual solution at the declared 1000 kg/m3 density.
        volume_m3=factor * 0.01,
        incoming_water_kg=factor * 9,
        evaporation_fraction=0.01,
        profile=profile,
    )


@pytest.mark.parametrize(
    "profile", (_ROW_WASTEWATER_PROFILE, _CH_WASTEWATER_PROFILE), ids=("RoW", "CH")
)
def test_actual_load_discharge_basis_and_principal_origin_groups(
    profile: _WastewaterProcessProfile,
) -> None:
    """Preserve origin numerators and scaling on the supplied reference basis."""
    water = _mixed_origin_water(profile, 1.0)
    # Independently supplied absolute operating/energy allocation on this
    # ten-kg solution: kWh electricity and MJ gas/purchased heat, respectively.
    releases = _wastewater_discharges(
        water, profile, gross_electricity_kwh=0.4, gas_heat_air_mj=0.03, purchased_heat_mj=0.2
    )
    biological = [row for row in releases if row.name.startswith("Carbon dioxide,")]
    assert len(biological) == 4
    for row in biological:
        expected = (4 / 7, 0.0, 0.0) if row.name.endswith(", fossil") else (0.0, 6 / 13, 2 / 13)
        assert row.principal_carbon_origin_fractions == pytest.approx(expected, rel=2e-14)
        assert row.unit == "kilogram"
    # Both compartments together exhaust biological carbon once; uncredited
    # auxiliaries stay in CO2 even though they are absent from principal shares.
    assert fsum(row.amount * 12 / 44 for row in biological) == pytest.approx(
        fsum(pool.biological_air_carbon_kg for pool in water.carbon), rel=2e-14
    )
    for row in releases:
        if not row.name.startswith("Carbon dioxide,"):
            assert row.principal_carbon_origin_fractions is None
    assert releases[-1].name == "Non-hazardous waste disposed"
    assert releases[-1].amount == 10.0
    assert releases[-1].unit == "kilogram"
    assert releases[-2].name == "Oxygen"
    assert releases[-2].amount == water.duties.biological_oxygen_kg
    assert releases[-2].categories == ("natural resource", "in air")
    assert fsum(
        row.amount for row in releases if row.name == "Heat, waste" and row.categories[0] == "air"
    ) == pytest.approx(3.6 * 0.4 + 0.03 + 0.2, rel=2e-14)

    # Same composition, 3.25 times actual water/solute/energy. Descriptor order,
    # units and origin fractions are invariant; every absolute release scales.
    factor = 3.25
    larger = _wastewater_discharges(
        _mixed_origin_water(profile, factor),
        profile,
        gross_electricity_kwh=factor * 0.4,
        gas_heat_air_mj=factor * 0.03,
        purchased_heat_mj=factor * 0.2,
    )
    for initial, scaled in zip(releases, larger, strict=True):
        assert (initial.name, initial.unit, initial.categories) == (
            scaled.name,
            scaled.unit,
            scaled.categories,
        )
        assert scaled.amount == pytest.approx(factor * initial.amount, rel=2e-14, abs=0)
        if initial.principal_carbon_origin_fractions is None:
            assert scaled.principal_carbon_origin_fractions is None
        else:
            assert scaled.principal_carbon_origin_fractions == pytest.approx(
                initial.principal_carbon_origin_fractions, rel=2e-14
            )
