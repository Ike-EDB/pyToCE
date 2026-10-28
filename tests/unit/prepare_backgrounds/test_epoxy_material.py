"""Actual cured epoxy, manufacturing losses and supplied application quantities."""

from math import fsum
from typing import Literal

import pytest
from material_recipe_fixtures import physical_material_recipe

from pytoce.download_data.sources.ecoinvent_materials import ConversionRequirement
from pytoce.prepare_backgrounds.preparation.providers.epoxy_material import (
    _epoxy_material,
    _epoxy_reference,
    _epoxy_residue_load,
)


@pytest.fixture(scope="module")
def epoxy_conversion() -> ConversionRequirement:
    """Read the principal conversion through the EcoSpold parser using invented XML."""
    return next(
        row
        for row in physical_material_recipe().conversion
        if row.operation == "others_epoxy_coating_conversion"
    )


@pytest.mark.parametrize(
    ("origin", "origin_index"),
    [("fossil", 0), ("biogenic", 1), ("atmospheric_dac", 2)],
)
def test_epoxy_manufacturing_exits_preserve_principal_and_actual_hwi_atoms(
    epoxy_conversion: ConversionRequirement,
    origin: Literal["fossil", "biogenic", "atmospheric_dac"],
    origin_index: int,
) -> None:
    """Carry fixed fossil MXDA and cleanup through the four principal-owned exits."""
    reference = _epoxy_reference(epoxy_conversion, origin)
    gross = epoxy_conversion.principal_feed_kg_per_finished_kg
    ratio = 15.44 / 84.56  # EP1838750B1 Table 6, comparative formulation 3.
    exits = {
        "retained": reference.retained,
        "preparation_residue": reference.preparation_residue,
        "settled_overspray": reference.settled_overspray,
        "airborne_overspray": reference.airborne_overspray,
    }
    principal = {
        "retained": 1.0,
        **{row.name: row.kg_per_finished_kg for row in epoxy_conversion.principal_exits},
    }
    assert set(exits) == set(principal)
    assert reference.gross_principal_kg == gross
    assert fsum(principal.values()) == pytest.approx(gross)
    for name, material in exits.items():
        expected_origins = tuple(
            principal[name] if index == origin_index else 0.0 for index in range(3)
        )
        assert material.principal_by_origin_kg == pytest.approx(expected_origins)
        assert material.mxda_kg == pytest.approx(ratio * principal[name])
        assert material.acetone_kg == (
            reference.supply_amounts["acetone"] if name == "preparation_residue" else 0.0
        )
        assert material.mass_kg == pytest.approx(
            principal[name] * (1 + ratio) + material.acetone_kg
        )
        assert fsum(material.elements_kg.values()) == pytest.approx(material.mass_kg)
    assert fsum(material.mxda_kg for material in exits.values()) == pytest.approx(
        reference.supply_amounts["mxda"]
    )
    assert fsum(material.mass_kg for material in exits.values()) == pytest.approx(
        gross + reference.supply_amounts["mxda"] + reference.supply_amounts["acetone"]
    )

    # Only preparation residue and settled overspray enter this external HWI.
    managed = reference.managed_residue
    managed_principal = principal["preparation_residue"] + principal["settled_overspray"]
    mxda = ratio * managed_principal
    acetone = reference.supply_amounts["acetone"]
    assert managed.mass_kg == pytest.approx(
        reference.preparation_residue.mass_kg + reference.settled_overspray.mass_kg
    )
    assert managed.principal_by_origin_kg[origin_index] == pytest.approx(managed_principal)
    # At EEW186, n=8/71 in C(21+18n)H(24+20n)O(4+3n) gives nominal element fractions. MXDA is
    # C8H12N2; cleanup acetone is C3H6O. Manufacturing uses its specified proxies.
    expected_elements = {
        "C": managed_principal * 1635 / 2201 + mxda * 96 / 136 + acetone * 36 / 58,
        "H": managed_principal * 466 / 6603 + mxda * 12 / 136 + acetone * 6 / 58,
        "O": managed_principal * 1232 / 6603 + acetone * 16 / 58,
        "N": mxda * 28 / 136,
        "S": 0.0,
        "Na": 0.0,
    }
    expected_principal_carbon = tuple(
        managed_principal * 1635 / 2201 if index == origin_index else 0.0 for index in range(3)
    )
    gross_heat = (
        34.016 * expected_elements["C"]
        + 124.265 * expected_elements["H"]
        - 9.8324 * expected_elements["O"]
        + 6.276 * expected_elements["N"]
    )
    formed_water = 9 * expected_elements["H"]
    load = _epoxy_residue_load(managed)
    assert load.wet_mass_kg == pytest.approx(managed_principal + mxda + acetone)
    assert load.active_elements_kg == pytest.approx(expected_elements)
    assert load.carrier_water_kg == 0.0
    assert load.nonfuel_salt_elements_kg == {}
    assert load.unknown_mineral_kg == 0.0
    assert load.waste_formed_water_kg == pytest.approx(formed_water)
    assert managed.heat_potential_mj == pytest.approx(gross_heat)
    assert load.net_energy_mj == pytest.approx(gross_heat - 2.4417 * formed_water)
    assert load.principal_carbon_kg == pytest.approx(expected_principal_carbon)
    assert load.biogenic_carbon_kg == pytest.approx(expected_principal_carbon[1])
    assert managed.fossil_auxiliary_carbon_kg == pytest.approx(mxda * 96 / 136 + acetone * 36 / 58)
    assert fsum(load.principal_carbon_kg) + managed.fossil_auxiliary_carbon_kg == pytest.approx(
        load.active_elements_kg["C"]
    )


def test_retired_epoxy_keeps_mixed_principal_origins_and_incorporated_hardener() -> None:
    """A mixed-origin cohort carries fixed fossil hardener and no cleanup solvent."""
    principal = (0.2, 0.6, 1.2)
    mxda = 2 * 15.44 / 84.56
    material = _epoxy_material(principal, mxda)
    expected_carbon = tuple(mass * 1635 / 2201 for mass in principal)
    expected_elements = {
        "C": 2 * 1635 / 2201 + mxda * 96 / 136,
        "H": 2 * 466 / 6603 + mxda * 12 / 136,
        "O": 2 * 1232 / 6603,
        "N": mxda * 28 / 136,
    }
    assert material.elements_kg == pytest.approx(expected_elements)
    assert material.principal_carbon_kg == pytest.approx(expected_carbon)
    assert material.fossil_auxiliary_carbon_kg == pytest.approx(mxda * 96 / 136)
    assert material.mass_kg == pytest.approx(2 + mxda)
    assert fsum(material.elements_kg.values()) == pytest.approx(material.mass_kg)
    assert material.acetone_kg == 0.0
    assert fsum(material.principal_carbon_kg) + material.fossil_auxiliary_carbon_kg == (
        pytest.approx(material.elements_kg["C"])
    )


def test_epoxy_mixing_cleanup_and_pump_use_their_distinct_material_bases(
    epoxy_conversion: ConversionRequirement,
) -> None:
    """Mix gross dry binder, clean its preparation and pump only spray-gate coating."""
    reference = _epoxy_reference(epoxy_conversion, "fossil")
    gross = epoxy_conversion.principal_feed_kg_per_finished_kg
    ratio = 15.44 / 84.56
    # The principal recipe already owns preparation losses and 67.5% spray
    # retention. Per retained kg, only 1/eta reaches the pump and atomizer.
    sprayed_principal = 1 / 0.675
    spray_litres = sprayed_principal * (1 / 1.17 + ratio / 1.032)
    gross_litres = gross * (1 / 1.17 + ratio / 1.032)
    cleanup = (562.5 / 1700) * gross_litres * 0.791
    # EPA2006 measures 2 SCFM pump air at 380 psig and .4875 L/min.
    # Transfer that complete pumping duty per hydraulic work to 2822 psig;
    # EPA's independent 3 SCFM gun air is absent from airless application.
    source_pump_air_per_litre = 2 * 0.028316846592 / 0.4875
    pressure_ratio = 2822 / 380
    normal_to_source_temperature = 273.15 / (273.15 + 190 / 9)
    expected_air = (
        spray_litres * source_pump_air_per_litre * pressure_ratio * normal_to_source_temperature
    )
    assert reference.gross_binder_kg == pytest.approx(gross * (1 + ratio))
    assert reference.spray_volume_l == pytest.approx(spray_litres)
    assert reference.supply_amounts == pytest.approx(
        {"mxda": ratio * gross, "acetone": cleanup, "compressed_air": expected_air}
    )
    assert reference.preparation_residue.acetone_kg == pytest.approx(cleanup)
    assert reference.gross_binder_kg + cleanup == pytest.approx(
        reference.retained.mass_kg
        + reference.managed_residue.mass_kg
        + reference.airborne_overspray.mass_kg
    )
