"""Actual UPR receives source mineral phases and distinct carbon references."""

from math import fsum

import pytest

from pytoce.download_data.sources.ecoinvent_materials import (
    ConversionRequirement,
    PrincipalExit,
)
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import _CarbonOrigin
from pytoce.prepare_backgrounds.preparation.providers.upr_landfill_load import (
    _upr_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.upr_material import _upr_material


@pytest.fixture(scope="module")
def upr_conversion() -> ConversionRequirement:
    """Use the admitted conversion at the material-to-receiver boundary."""
    # Exact source-equation fractions, shared with the independent material
    # witness. Parsing and the detailed formulation are tested at their owners.
    return ConversionRequirement(
        operation="others_upr_hand_lay_up_conversion",
        principal_feed_kg_per_finished_kg=43163838324174400 / 35680245530124351,
        finished_kg_per_native_service_kg=67075941681801 / 205053863772800,
        principal_exits=(
            PrincipalExit("styrene_air", 8208310784 / 84118311615),
            PrincipalExit("post_release_solid", 103474753 / 922564095),
        ),
    )


# Landfill2020 Clim/avgTK/MSWLF reference values use rounded carbonate-end years and per-incoming-kg
# element quantities. Si columns are gas/short/later/stored. Heat is MJ per principal kg at
# whole-material Q=30.164081970657673 and C=.731932863627673.
_SOURCE_CASES = (
    pytest.param(
        _LandfillSite(16.743579766537, 1250.84108527132, 1.0),
        2863.0,
        0.009707901638808891,
        0.00029123704916426675,
        (1.249579941011392e-7, 0.0004998319764045568, 0.01432682749971587, 0.9851732155658854),
        0.06728133141927967,
        -0.0021301723591222277,
        id="sanitary",
    ),
    pytest.param(
        _LandfillSite(8.0, 350.0, 1 - 2.0604 / 7),
        3171.0,
        0.00969682667698839,
        0.0002909048003096517,
        (1.2481543960632395e-7, 0.0004992617584252957, 0.015619863120564318, 0.9838807503055708),
        0.06720457557600637,
        0.0845926149796394,
        id="hyperarid",
    ),
    pytest.param(
        _LandfillSite(8.0, 550.0, 1 - 2.0604 / 7),
        7789.0,
        0.009697217260789561,
        0.0002909165178236868,
        (1.2482046711588663e-7, 0.0004992818684635465, 0.03524351495034454, 0.9642570783607248),
        0.06720728254597283,
        0.08459602233198367,
        id="dry",
    ),
    pytest.param(
        _LandfillSite(8.0, 900.0, 1 - 2.0604 / 7),
        2658.0,
        0.009697218361495357,
        0.0002909165508448607,
        (1.2482048128393174e-7, 0.0004992819251357269, 0.01344060177080822, 0.9860599914835748),
        0.06720729017449588,
        0.08459603193425713,
        id="moist",
    ),
    pytest.param(
        _LandfillSite(8.0, 1100.0, 1 - 2.0604 / 7),
        1632.0,
        0.009697218361533007,
        0.00029091655084599025,
        (1.2482048128441638e-7, 0.0004992819251376654, 0.009080868217795593, 0.9904197250365855),
        0.06720729017475682,
        0.08459603193458559,
        id="wet",
    ),
    pytest.param(
        _LandfillSite(8.0, 1900.0, 1 - 2.0604 / 7),
        862.0,
        0.009697218361533113,
        0.0002909165508459934,
        (1.2482048128441776e-7, 0.000499281925137671, 0.005808943426510781, 0.9936916498278703),
        0.06720729017475756,
        0.08459603193458651,
        id="very-wet",
    ),
)


@pytest.mark.parametrize("origin", ("fossil", "biogenic", "atmospheric_dac"))
@pytest.mark.parametrize(
    (
        "site",
        "carbonate_end_year",
        "carbon_gas",
        "carbon_short",
        "silicon_phases",
        "boron_short",
        "net_soil_heat",
    ),
    _SOURCE_CASES,
)
def test_upr_material_enters_complete_source_phases_with_tracked_carbon_subset(
    upr_conversion: ConversionRequirement,
    origin: _CarbonOrigin,
    site: _LandfillSite,
    carbonate_end_year: float,
    carbon_gas: float,
    carbon_short: float,
    silicon_phases: tuple[float, float, float, float],
    boron_short: float,
    net_soil_heat: float,
) -> None:
    """Receive minerals and nominal gas with the specified tracked principal carbon."""
    material = _upr_material(
        upr_conversion,
        (
            float(origin == "fossil"),
            float(origin == "biogenic"),
            float(origin == "atmospheric_dac"),
        ),
    )
    actual = _upr_landfill_load(upr_conversion, origin, site, carbonate_end_year=carbonate_end_year)
    assert actual.wet_mass_kg == pytest.approx(material.mass_kg)
    assert actual.water_kg == 0.0
    assert fsum(actual.elements_kg.values()) == pytest.approx(material.mass_kg)
    for element, phase in actual.phases.items():
        expected_input = material.burnable_elements_kg.get(
            element, 0.0
        ) + material.inert_elements_kg.get(element, 0.0)
        assert phase.input_kg == pytest.approx(expected_input)
        assert actual.elements_kg[element] == pytest.approx(expected_input)
        assert fsum(
            (phase.gas_kg, phase.short_leachate_kg, phase.later_leachate_kg, phase.stored_kg)
        ) == pytest.approx(expected_input)
        assert (
            min(phase.gas_kg, phase.short_leachate_kg, phase.later_leachate_kg, phase.stored_kg)
            >= 0.0
        )
        assert actual.leachate.elements_kg[element] == phase.short_leachate_kg

    # Positive embedded minerals must reach actual phases even when selected
    # donor-reference B/Si/Ca properties are omitted.
    for element in ("B", "Si", "Ca", "Al", "Co"):
        assert actual.phases[element].short_leachate_kg > 0.0
    silicon = actual.phases["Si"]
    assert tuple(
        value / silicon.input_kg
        for value in (
            silicon.gas_kg,
            silicon.short_leachate_kg,
            silicon.later_leachate_kg,
            silicon.stored_kg,
        )
    ) == pytest.approx(silicon_phases, rel=2e-13, abs=1e-15)
    boron = actual.phases["B"]
    assert boron.gas_kg == 0.0
    assert boron.short_leachate_kg / boron.input_kg == pytest.approx(boron_short, rel=2e-13)
    for element in ("B", "Ca", "Al", "Co"):
        assert actual.phases[element].stored_kg == 0.0
    # MSWLF rows 36/57/58 have gas fraction 1/4001 and their own release
    # multipliers. Alpha is recovered from the independent carbon witness.
    alpha = carbon_gas * 103
    for element, release in (
        ("Co", 0.32150064200725031),
        ("Ca", 0.13024173278843815),
        ("Al", 0.05),
    ):
        phase = actual.phases[element]
        liberated = phase.input_kg * 0.01 * alpha * release
        assert phase.gas_kg == pytest.approx(liberated / 4001, rel=2e-13)
        assert phase.short_leachate_kg == pytest.approx(liberated * 4000 / 4001, rel=2e-13)
    for element in ("C", "H", "O"):
        phase = actual.phases[element]
        assert phase.gas_kg / phase.input_kg == pytest.approx(carbon_gas, rel=2e-13)
        assert phase.short_leachate_kg / phase.input_kg == pytest.approx(carbon_short, rel=2e-13)
    assert actual.gas.nitrogen_kg == actual.gas.sulfur_kg == 0.0

    # Nominal carbon drives gas; tracked carbon defines its reporting subset and retains the
    # accounting bridge.
    expected_pools = (
        ("principal_upr", True, origin, 0.70405325131712),
        ("upr_principal_reference_difference", False, origin, 0.005894989156257515),
        ("upr_retained_auxiliaries", False, "fossil", material.fossil_auxiliary_carbon_kg),
    )
    for pool, liquid, gas, (component, principal, pool_origin, mass) in zip(
        actual.carbon, actual.leachate.carbon, actual.gas.carbon, expected_pools, strict=True
    ):
        assert (pool.component, pool.principal, pool.origin) == (component, principal, pool_origin)
        assert (liquid.component, liquid.principal, liquid.origin) == (
            component,
            principal,
            pool_origin,
        )
        assert (gas.component, gas.principal, gas.origin) == (component, principal, pool_origin)
        assert pool.phases.input_kg == pytest.approx(mass)
        assert liquid.carbon_kg == pytest.approx(mass * carbon_short)
        assert gas.methane_carbon_kg + gas.carbon_dioxide_carbon_kg == pytest.approx(
            mass * carbon_gas
        )
    assert actual.phases["C"].gas_kg == pytest.approx(0.731932863627673 * carbon_gas)
    assert fsum(pool.phases.gas_kg for pool in actual.carbon) == pytest.approx(
        actual.phases["C"].gas_kg
    )
    assert fsum(pool.phases.gas_kg for pool in actual.carbon if pool.principal) == pytest.approx(
        0.70405325131712 * carbon_gas
    )
    assert fsum(pool.phases.input_kg for pool in actual.carbon[:2]) == pytest.approx(
        0.7099482404733776
    )

    toc = actual.leachate.toc_kg
    assert toc == pytest.approx(0.731932863627673 * carbon_short)
    assert actual.leachate.doc_kg == toc
    assert actual.leachate.organic_hydrogen_kg == pytest.approx(0.1225 * toc)
    assert actual.leachate.organic_oxygen_kg == pytest.approx(0.64 * toc)
    assert actual.leachate.organic_oxygen_kg < actual.leachate.elements_kg["O"]
    assert actual.leachate.cod_kg_o2 == pytest.approx(1.09375 * toc)
    assert actual.leachate.bod5_kg_o2 == pytest.approx(0.25892857142857145 * toc)
    assert actual.leachate.chloride_kg == 0.0

    energy = actual.gas.decomposition_trace
    assert energy["input_hhv_mj"] == pytest.approx(30.164081970657673)
    assert energy["input_hhv_mj"] == material.heat_potential_mj
    assert energy["short_leachate_hhv_mj"] == pytest.approx(30.164081970657673 * carbon_short)
    assert energy["stored_hhv_mj"] == 0.0
    assert actual.gas.decomposition_heat_mj == pytest.approx(net_soil_heat, rel=2e-13, abs=2e-14)
    assert energy["heat_soil_mj"] == actual.gas.decomposition_heat_mj
    # The sanitary net residual is signed after separately reserving leachate
    # chemical energy. A zero floor would break this source-allocation balance.
    assert fsum(
        (
            energy["raw_gas_hhv_mj"],
            actual.gas.decomposition_heat_mj,
            energy["short_leachate_hhv_mj"],
            energy["later_leachate_hhv_mj"],
            energy["stored_hhv_mj"],
        )
    ) == pytest.approx(energy["input_hhv_mj"])
