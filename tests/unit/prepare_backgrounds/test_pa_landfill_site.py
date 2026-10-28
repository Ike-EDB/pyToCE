"""Check actual PA placement and later species at the shared sanitary site."""

from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.landfill_site import (
    _sanitary_site_emissions,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_load import (
    _pa_landfill_load,
)

_AIR = ("air", "non-urban air or from high stacks")
_LONG = ("water", "ground-, long-term")
_LAND = ("natural resource", "land")
_WASTE = ("inventory indicator", "waste")


@pytest.mark.parametrize("grade", ("pa6", "pa66"))
def test_actual_pa_site_keeps_placement_and_all_later_species(grade: str) -> None:
    """Use real PA phases; wet placement and elemental receiving have separate bases."""
    finish = _pa_finish_reference(grade)
    # Use the original RoW sanitary-site climate.
    site = _LandfillSite(16.743579766537, 1250.84108527132, 1.0)
    load = _pa_landfill_load(
        finish,
        principal_by_origin_kg={"fossil": 1.25, "atmospheric_dac": 0.75},
        finish_dose_kg=3.5,
        water_kg=0.4,
        site=site,
    )
    later = {
        element: load.phases[element].later_leachate_kg for element in ("C", "N", "S", "P", "Na")
    }
    gas = {"Na": load.phases["Na"].gas_kg}
    result = _sanitary_site_emissions(
        wet_mass_kg=load.wet_mass_kg,
        carbon_kg=load.elements_kg["C"],
        later_elements_kg=later,
        gas_elements_kg=gas,
    )
    # Original-site allocation uses 1.8e9 kg capacity, 90,000 m2 landfill, and 18,000 m2 road. The
    # per-kg area/time coefficients describe initial placement; downstream sites use separate
    # allocations.
    land_per_kg = {
        "Occupation, construction site": 0.00025,
        "Occupation, dump site": 0.0015,
        "Occupation, shrub land, sclerophyllous": 0.00025,
        "Occupation, traffic area, road network": 0.00185,
        "Transformation, from dump site, sanitary landfill": 0.00005,
        "Transformation, from pasture, man made": 0.00006,
        "Transformation, from shrub land, sclerophyllous": 0.00005,
        "Transformation, to dump site, sanitary landfill": 0.00005,
        "Transformation, to forest, unspecified": 0.00005,
        "Transformation, to shrub land, sclerophyllous": 0.00005,
        "Transformation, to traffic area, road network": 0.00001,
    }
    for name, per_kg in land_per_kg.items():
        assert result[name, _LAND] == pytest.approx(per_kg * load.wet_mass_kg, rel=2e-14)
    assert result["Organic carbon, placed in landfill", _WASTE] == load.elements_kg["C"]
    assert result["Waste mass, total, placed in landfill", _WASTE] == load.wet_mass_kg
    assert result["Non-hazardous waste disposed", _WASTE] == load.wet_mass_kg

    assert result["TOC, Total Organic Carbon", _LONG] == later["C"]
    assert result["DOC, Dissolved Organic Carbon", _LONG] == later["C"]
    assert result["COD, Chemical Oxygen Demand", _LONG] == pytest.approx(1.09375 * later["C"])
    assert result["BOD5, Biological Oxygen Demand", _LONG] == pytest.approx(
        0.25892857142857145 * later["C"]
    )
    # MSWLF C94:C97 express mg elemental N/L. Convert the emitted species
    # back to kg N before checking both closure and the independent weights.
    n_species = (
        ("Ammonium", 18 / 14, 115.79732353778977),
        ("Nitrogen, organic bound", 1.0, 243.33844006164554),
        ("Nitrite", 46 / 14, 2.4662120743304703),
        ("Nitrate", 62 / 14, 3.5355339059327378),
    )
    n_total = fsum(result[name, _LONG] / compound_per_n for name, compound_per_n, _ in n_species)
    assert n_total == pytest.approx(later["N"], rel=2e-14)
    n_reference = fsum(weight for _, _, weight in n_species)
    for name, compound_per_n, weight in n_species:
        assert result[name, _LONG] / compound_per_n == pytest.approx(
            later["N"] * weight / n_reference, rel=2e-14
        )
    sulfate_s = result["Sulfate", _LONG] * 32 / 96
    sulfide_s = result["Hydrogen sulfide", _LONG] * 32 / 34
    assert sulfate_s + sulfide_s == pytest.approx(later["S"], rel=2e-14)
    assert sulfate_s / sulfide_s == pytest.approx(40.00326505167857 / 2.9443208588856526)
    # Landfill2020 G99/L99 uses nominal 95/31 phosphate/P; landfarming follows its separately
    # specified conversion.
    assert result["Phosphate", _LONG] * 31 / 95 == pytest.approx(later["P"], rel=2e-14)
    assert result["Sodium I", _LONG] == later["Na"] > 0
    assert result["Sodium I", _AIR] == gas["Na"] > 0
    assert load.phases["P"].gas_kg == 0

    expected_species = {
        "TOC, Total Organic Carbon",
        "DOC, Dissolved Organic Carbon",
        "COD, Chemical Oxygen Demand",
        "BOD5, Biological Oxygen Demand",
        "Ammonium",
        "Nitrogen, organic bound",
        "Nitrite",
        "Nitrate",
        "Sulfate",
        "Hydrogen sulfide",
        "Phosphate",
        "Sodium I",
    }
    assert set(result) == (
        {(name, _LAND) for name in land_per_kg}
        | {(name, _LONG) for name in expected_species}
        | {
            ("Organic carbon, placed in landfill", _WASTE),
            ("Waste mass, total, placed in landfill", _WASTE),
            ("Non-hazardous waste disposed", _WASTE),
            ("Sodium I", _AIR),
        }
    )
    # An extra kg carried water changes placed mass and its land allocation. P/Na/C quantities
    # retain their values.
    wetter = _sanitary_site_emissions(
        wet_mass_kg=load.wet_mass_kg + 1,
        carbon_kg=load.elements_kg["C"],
        later_elements_kg=later,
        gas_elements_kg=gas,
    )
    for key, amount in result.items():
        name, compartment = key
        if compartment == _LAND:
            assert wetter[key] - amount == pytest.approx(land_per_kg[name], rel=2e-14)
        elif name in {"Waste mass, total, placed in landfill", "Non-hazardous waste disposed"}:
            assert wetter[key] - amount == pytest.approx(1.0)
        else:
            assert wetter[key] == amount
