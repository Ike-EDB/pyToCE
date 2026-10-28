"""Source reference for retained compound 22 material and homogeneous trim."""

from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers.sbr_material import (
    _sbr_compound_material,
)


@pytest.mark.parametrize("origin", ("fossil", "biogenic", "atmospheric_dac"))
def test_sbr_compound_preserves_full_material_and_homogeneous_trim(origin):
    """Retain the filled material and its fixed auxiliary quantities."""
    compound = _sbr_compound_material(origin, principal_kg=1)
    # Retained compound 22 composition, in kg per retained principal kg
    # after manufacturing withdrawals.
    expected = {
        "C": 0.8969229843817300 + 0.6902901779942218 + 0.014431081665563762,
        "H": 1 - 0.8969229843817300 + 0.013423504206200683,
        "O": 0.02177165681744754,
        "N": 0.004538225672096758,
        "S": 0.022190294846286596,
        "Zn": 0.08025422896960048,
    }
    assert compound.mass_kg == pytest.approx(1.8468991701714175, rel=2e-14)
    assert compound.elements_kg == pytest.approx(expected, rel=2e-14)
    assert compound.mass_kg == fsum(compound.elements_kg.values())
    assert compound.principal_carbon_kg == {origin: 0.8969229843817300}
    assert compound.auxiliary_carbon_kg == pytest.approx(
        {"fossil": 0.6902901779942218, "biogenic": 0.014431081665563762}, rel=2e-14
    )
    assert fsum(
        (*compound.principal_carbon_kg.values(), *compound.auxiliary_carbon_kg.values())
    ) == (pytest.approx(compound.elements_kg["C"], rel=2e-14))
    # Source ingredient heat subtotal times 19, with NIST graphite and separate
    # elemental sulfur retained. ZnO remains in the element mass balance
    # but contributes no fuel heat.
    assert compound.carbon_heat_mj == pytest.approx(68.33950399272836, rel=2e-14)
    assert compound.sulfur_heat_mj == pytest.approx(0.16304868689505403, rel=2e-14)

    # Carli 5% trim /95% retained gives the inlet's 1/19 principal basis.
    trim = _sbr_compound_material(origin, principal_kg=1 / 19)
    assert trim.mass_kg == pytest.approx(compound.mass_kg / 19, rel=2e-14)
    assert trim.elements_kg == pytest.approx(
        {element: amount / 19 for element, amount in compound.elements_kg.items()}, rel=2e-14
    )
    assert trim.principal_carbon_kg == pytest.approx({origin: 0.8969229843817300 / 19}, rel=2e-14)
    assert trim.auxiliary_carbon_kg == pytest.approx(
        {key: value / 19 for key, value in compound.auxiliary_carbon_kg.items()}, rel=2e-14
    )
    assert trim.carbon_heat_mj == pytest.approx(compound.carbon_heat_mj / 19, rel=2e-14)
    assert trim.sulfur_heat_mj == pytest.approx(compound.sulfur_heat_mj / 19, rel=2e-14)
