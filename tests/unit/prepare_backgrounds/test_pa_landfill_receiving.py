"""Use complete invented sanitary sources for actual PA fiber disposal."""

import json
from copy import deepcopy
from dataclasses import replace
from math import fsum, log
from pathlib import Path

import pytest
from pa_landfill_receiving_fixtures import landfill_receiving_boundary
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources
from tps_gas_fixtures import GAS_SOURCES

from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    admit_pa_landfill_receiving_reference,
    admit_pa_landfill_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingGasReference,
    _admit_gas_sources,
    _gas_fuel_reference,
    reference_from_payload,
    reference_to_payload,
    sources_from_payload,
    sources_to_payload,
)
from pytoce.run_lca.calculation.pa_landfill_receiving import receive_pa_landfill


@pytest.mark.parametrize("grade,future", (("pa6", False), ("pa66", True)))
def test_actual_pa_landfill_closes_all_receiving_functions_after_source_detachment(
    tmp_path: Path, grade: str, future: bool
) -> None:
    """Preserve wet-site support, complete descendants and separate C/P fates.

    The real source graph has nonunit signed productions, distinct suppliers,
    original-only composition and a deliberately different future leachate
    coefficient. The actual grade has all principal carbon origins and inherited
    finish; native donor chemistry cannot represent this mixed input.
    """
    reference = pa_receiving_reference()
    sources = pa_receiving_sources(2050 if future else 2025, background_scenario="member")
    with landfill_receiving_boundary(tmp_path, future=future) as (_, context):
        landfill = admit_pa_landfill_receiving_reference(context)
        dated = admit_pa_landfill_receiving_sources(context, landfill)
        gas_reference = PaReceivingGasReference(
            _gas_fuel_reference(context, dict(GAS_SOURCES["chp"]), "biogas"),
            _gas_fuel_reference(
                context, dict(GAS_SOURCES["furnace"]), "natural gas, high pressure"
            ),
        )
        gas_selectors = deepcopy(GAS_SOURCES)
        if future:
            # The combined invented future source uses this declared product;
            # original fuel calibration keeps its separate original selector.
            gas_selectors["furnace"]["reference_product"] = "heat, district or industrial"
        gas_sources = _admit_gas_sources(context, {"gas_sources": gas_selectors})
        reference = replace(reference, landfill=landfill, gas=gas_reference)
        sources = replace(sources, landfill=dated, gas=gas_sources)
        # The consumer reads the persisted preparation records after the source project closes.
        reference_payload = json.loads(json.dumps(reference_to_payload(reference), allow_nan=False))
        sources_payload = json.loads(json.dumps(sources_to_payload(sources), allow_nan=False))

    reference = reference_from_payload(reference_payload)
    sources = sources_from_payload(sources_payload)
    finish = _pa_finish_reference(grade)
    result = receive_pa_landfill(
        constituent="polyamide_6_resin" if grade == "pa6" else "polyamide_66_resin",
        principal_by_origin_kg={"fossil": 1.1, "biogenic": 0.6, "atmospheric_dac": 0.3},
        total_finish_dose_kg=5.0,
        water_kg=0.086,
        reference=reference,
        sources=sources,
    )
    # The input contains two principal kg, five fixed-vector doses, and carried water.
    wet = 2.0 + 5.0 * fsum(finish.retained.compounds_kg.values()) + 0.086
    assert result.load.wet_mass_kg == pytest.approx(wet, rel=2e-14)
    assert result.load.water_kg == 0.086
    database = "future" if future else "original"
    site_amounts = [row for row in result.amounts if row.function == "pa_sanitary_landfill"]
    site_supplies = {
        row.endpoint.key: row
        for row in site_amounts
        if isinstance(row.endpoint, PaReceivingActivity)
    }
    # Synthetic site intensities are per wet kg; the -4/-9 kg production references have already
    # been normalized.
    intensities = {
        "site_equipment": 0.003 if future else 0.002,
        "site_admin_a": 0.04 if future else 0.02,
        "site_admin_b": 0.02 if future else 0.03,
        "site_heat": 0.07 if future else 0.04,
        "site_retirement": -0.001,
    }
    assert set(site_supplies) == {(database, code) for code in intensities}
    for code, intensity in intensities.items():
        row = site_supplies[database, code]
        assert row.amount == pytest.approx(wet * intensity, rel=2e-14)
        assert row.source.key == (database, "site")
        assert row.metadata["uncertainty type"] == 2
        assert row.metadata["loc"] == pytest.approx(log(abs(row.amount)) + 0.03, rel=2e-14)
        assert row.metadata["scale"] == 0.21

    # Every native internal child is expanded to actual receiving quantities.
    # Keeping even one whole donor child would reintroduce its waste chemistry.
    children = {
        operation.activity.key
        for operation in (
            sources.landfill.landfill,
            sources.landfill.collection,
            sources.landfill.wastewater,
            sources.landfill.sludge_incineration_market,
            sources.landfill.sludge_landfarming_market,
            sources.landfill.sludge_incineration,
            sources.landfill.sludge_landfarming,
            sources.landfill.bottom_ash_market,
            sources.landfill.residual_ash_market,
            sources.landfill.bottom_ash_treatment,
            sources.landfill.residual_ash_treatment,
        )
    }
    assert not children.intersection(
        row.endpoint.key for row in result.amounts if isinstance(row.endpoint, PaReceivingActivity)
    )
    # Residue gas supplies Doka pumping power; site support follows the retained inventory.
    assert {row.function for row in result.amounts} == {
        "pa_sanitary_landfill",
        "pa_landfill_gas_leakage",
        "pa_landfill_gas_flare",
        "pa_landfill_gas_chp",
        "pa_sanitary_collection",
        "pa_sanitary_wastewater",
        "pa_wastewater_gas_leakage",
        "pa_wastewater_gas_flare",
        "pa_wastewater_gas_chp",
        "pa_sanitary_sludge_collection",
        "pa_sanitary_sludge_incineration",
        "pa_sanitary_bottom_ash_collection",
        "pa_sanitary_bottom_ash",
        "pa_sanitary_residual_ash_collection",
        "pa_sanitary_residual_ash",
        "pa_sanitary_sludge_landfarming_market",
        "pa_sanitary_sludge_landfarming",
    }
    releases = {
        (row.endpoint.name, row.endpoint.categories): row
        for row in site_amounts
        if isinstance(row.endpoint, PaReceivingFlow)
    }
    waste = ("inventory indicator", "waste")
    assert releases["Waste mass, total, placed in landfill", waste].amount == pytest.approx(wet)
    assert (
        releases["Organic carbon, placed in landfill", waste].amount == result.load.elements_kg["C"]
    )
    # Later groundwater follows the site's own material phase.
    long = ("water", "ground-, long-term")
    assert releases["Phosphate", long].amount * 31 / 95 == pytest.approx(
        result.load.phases["P"].later_leachate_kg, rel=2e-14
    )
    assert releases["TOC, Total Organic Carbon", long].amount == (
        result.load.phases["C"].later_leachate_kg
    )
    assert releases["Heat, waste", ("soil", "industrial")].amount == (
        result.load.gas.decomposition_heat_mj
    )
    # Original fixture volume is .006 m3/kg wet waste, including at future
    # anchors whose donor row deliberately says .009. The solvent separately
    # subtracts actual short-leachate solutes at 1000 kg/m3.
    wastewater = result.water.wastewater
    assert wastewater.volume_m3 == pytest.approx(0.006 * wet, rel=2e-14)
    assert wastewater.water.incoming_kg == pytest.approx(
        1000 * 0.006 * wet - fsum(result.load.leachate.elements_kg.values()), rel=2e-14
    )
    # Close the complete incoming short-phase P through actual native outputs
    # and uncharacterized terminal stocks. Collection is already present in
    # inventory amounts, but retained ash/crop pools remain conditional.
    # Native water/soil receivers use 94.97/30.97; furnace/ash use nominal95/31.
    emitted_p = fsum(
        row.amount
        * (
            1.0
            if row.endpoint.name == "Phosphorus"
            else 30.97 / 94.97
            if row.function in {"pa_sanitary_wastewater", "pa_sanitary_sludge_landfarming"}
            else 31 / 95
        )
        for row in result.amounts
        if isinstance(row.endpoint, PaReceivingFlow)
        and row.endpoint.name in {"Phosphorus", "Phosphate"}
        and row.function != "pa_sanitary_landfill"
    )
    retained_p = result.water.treatment_fraction * (
        result.sludge.bottom_ash.retained_profile_elements_kg["P"]
        + result.sludge.residual_ash.retained_profile_elements_kg["P"]
        + result.landfarming.load.phosphorus_crop_uptake_kg
    )
    assert emitted_p + retained_p == pytest.approx(result.load.leachate.elements_kg["P"], rel=3e-13)
    assert all(
        row.amount >= 0 for row in result.amounts if isinstance(row.endpoint, PaReceivingFlow)
    )
