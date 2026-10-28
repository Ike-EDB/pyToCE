"""Use one actual PA water load in detached and native receiving calculations."""

from collections import defaultdict
from dataclasses import replace
from math import fsum
from pathlib import Path
from typing import Literal, cast

import pytest
from pa_landfill_receiving_fixtures import landfill_receiving_boundary
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources
from pa_water_receiving_fixtures import water_receiving_boundary

from pytoce.prepare_backgrounds.preparation.providers.catalogue import provider_service_code
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_treatment import (
    _leachate_market_dataset,
    _wastewater_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_water import (
    _pa_finish_water_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_load import (
    _pa_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    admit_pa_landfill_receiving_reference,
    admit_pa_landfill_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    admit_pa_water_receiving_reference,
    admit_pa_water_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.waste_gas import _GasTreatment
from pytoce.run_lca.calculation.pa_gas_receiving import receive_pa_wastewater_gas
from pytoce.run_lca.calculation.pa_receiving import PaReceivingAmount
from pytoce.run_lca.calculation.pa_water_receiving import receive_pa_water


def _gas_exchange(row: PaReceivingAmount) -> dict[str, object]:
    """Give the existing native water binder the actual gas owner's calculated row."""
    return {
        **row.metadata,
        "input": row.endpoint.key,
        "amount": row.amount,
        "type": "biosphere",
    }


@pytest.mark.parametrize(
    ("receiving", "future", "finish_dose"),
    (
        ("swiss", False, 1.0),
        ("swiss", True, 1.0),
        ("sanitary", False, 1.0),
        ("sanitary", True, 1.0),
        ("sanitary", False, 0.0),
    ),
)
def test_actual_water_receiving_matches_native_owners_and_preserves_raw_phosphorus(
    tmp_path: Path, future: bool, receiving: Literal["swiss", "sanitary"], finish_dose: float
) -> None:
    """Retain real collection/plant portfolios while applying actual PA chemistry.

    Bounded native graphs distinguish original/dated production bases and
    Swiss versus RoW compartments and hauling. Gas has its separately tested
    detached fixture; the unchanged native plant consumes that same gas result.
    This tests its integration and normalization, rather than reproducing gas
    equations or re-running the full prepared-background graph.
    """
    reference = pa_receiving_reference()
    sources = pa_receiving_sources(2050 if future else 2025, background_scenario="member")
    finish = _pa_finish_reference("pa6")
    leachate = _pa_finish_water_load(
        finish.collected,
        caprolactam_by_origin_kg={
            "fossil": finish.collected_caprolactam_kg * 0.2,
            "biogenic": finish.collected_caprolactam_kg * 0.3,
            "atmospheric_dac": finish.collected_caprolactam_kg * 0.5,
        },
    )
    volume_m3, incoming_water_kg = 0.0009, finish.collected_water_kg
    boundary = water_receiving_boundary if receiving == "swiss" else landfill_receiving_boundary
    expected: dict[tuple[str, str], float] = defaultdict(float)
    with boundary(tmp_path, future=future) as (_, context):
        if receiving == "swiss":
            water_reference = admit_pa_water_receiving_reference(context)
            water_sources = admit_pa_water_receiving_sources(context, water_reference)
            reference = replace(reference, water=water_reference)
            sources = replace(sources, water=water_sources)
            profile = _CH_WASTEWATER_PROFILE
            collection_id = "pa_phosphate_water_collection_ch"
            plant_id = "pa_phosphate_water_treatment_ch"
        else:
            landfill_reference = admit_pa_landfill_receiving_reference(context)
            landfill_sources = admit_pa_landfill_receiving_sources(context, landfill_reference)
            reference = replace(reference, landfill=landfill_reference)
            sources = replace(sources, landfill=landfill_sources)
            water_reference, water_sources = landfill_reference, landfill_sources
            profile = _ROW_WASTEWATER_PROFILE
            collection_id = "sbr_filled_fossil_leachate_collection"
            plant_id = "sbr_filled_fossil_wastewater_treatment"
            # Sanitary receiving uses landfill phase chemistry, including a neat-principal inlet
            # with zero auxiliary P/S/Na.
            landfill = _pa_landfill_load(
                finish,
                principal_by_origin_kg={"fossil": 0.2, "biogenic": 0.3, "atmospheric_dac": 0.5},
                finish_dose_kg=finish_dose,
                water_kg=0.1,
                site=_LandfillSite(10.0, 1000.0, 1.0),
            )
            leachate = landfill.leachate
            volume_m3 = landfill.wet_mass_kg * landfill_reference.leachate_volume_m3_per_wet_kg
            incoming_water_kg = 1000 * volume_m3 - fsum(leachate.elements_kg.values())
        wastewater = _wastewater_load(
            leachate,
            volume_m3=volume_m3,
            incoming_water_kg=incoming_water_kg,
            evaporation_fraction=water_reference.evaporation_fraction,
            profile=profile,
        )
        gas = receive_pa_wastewater_gas(wastewater, profile, reference.gas, sources.gas)
        gas_native = _GasTreatment(
            tuple(_gas_exchange(row) for row in gas.amounts),
            gas.heat_air_mj,
            gas.heat_soil_mj,
            gas.internal_electricity_kwh,
            gas.internal_heat_mj,
            {},
        )
        services = {
            str(service["provider_service_id"]): service
            for service in cast(list[dict[str, object]], context.catalogue["services"])
        }
        collection = _leachate_market_dataset(
            context, services[collection_id], leachate, wastewater, "calculated"
        )
        plant = _wastewater_dataset(
            context, services[plant_id], leachate, wastewater, gas_native, "calculated", profile
        )
        plant_construction = cast(dict[str, object], services[plant_id]["construction"])
        child_keys = {
            (
                "calculated",
                provider_service_code(
                    str(cast(dict[str, str], plant_construction[role])["service_id"])
                ),
            )
            for role in ("incineration", "landfarming")
        }
        child_keys.add(("calculated", provider_service_code(plant_id)))
        treated = -fsum(
            row.amount
            for row in water_sources.collection.rows
            if row.endpoint.key == water_sources.wastewater.activity.key
        )
        for record, quantity in ((collection, volume_m3), (plant, volume_m3 * treated)):
            rows = cast(list[dict[str, object]], record["exchanges"])
            production = next(row for row in rows if row["type"] == "production")
            multiplier = quantity / -float(cast(float, production["amount"]))
            for row in rows:
                key = cast(tuple[str, str], row["input"])
                if row["type"] != "production" and key not in child_keys:
                    expected[key] += float(cast(float, row["amount"])) * multiplier
        if receiving == "sanitary" and finish_dose != 0:
            # The plastic collector has no native P row. The accepted explicit
            # raw elemental-P source is the additional actual PA responsibility.
            phosphorus = next(
                flow
                for flow in water_sources.biosphere
                if flow.name == "Phosphorus" and flow.categories == ("water", "surface water")
            )
            expected[phosphorus.key] += leachate.elements_kg["P"] * (1 - treated)

    # The calculation uses admitted records and finite source equations.
    actual = receive_pa_water(
        leachate,
        volume_m3=volume_m3,
        incoming_water_kg=incoming_water_kg,
        reference=reference,
        sources=sources,
        receiving=receiving,
    )
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in actual.amounts:
        totals[row.endpoint.key] += row.amount
    assert dict(totals) == pytest.approx(dict(expected), rel=3e-13, abs=1e-20)
    assert actual.treatment_fraction == treated
    assert actual.wastewater == wastewater
    assert {row.endpoint.unit for row in actual.amounts} >= {
        "kilogram",
        "kilowatt hour",
        "cubic meter",
    }
    source_child_keys = {
        water_sources.wastewater.activity.key,
        water_sources.sludge_incineration_market.activity.key,
        water_sources.sludge_landfarming_market.activity.key,
    }
    assert not any(row.endpoint.key in source_child_keys for row in actual.amounts)
    raw_p = [
        row
        for row in actual.amounts
        if row.function == f"pa_{receiving}_collection" and row.endpoint.name == "Phosphorus"
    ]
    assert fsum(row.amount for row in raw_p) == pytest.approx(
        leachate.elements_kg["P"] * (1 - treated)
    )
    assert all(row.principal_carbon_origin_fractions is None for row in raw_p)
