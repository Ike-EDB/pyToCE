"""Trace actual PA water through its complete agricultural sludge market and soil."""

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
from pytoce.prepare_backgrounds.preparation.providers.leachate_treatment import (
    _sludge_market_dataset,
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
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    admit_pa_water_receiving_reference,
    admit_pa_water_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import (
    _scaled_native_exchange,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_landfarming import (
    _landfarming_dataset,
)
from pytoce.run_lca.calculation.pa_landfarming_receiving import (
    receive_pa_landfarmed_sludge,
)
from pytoce.run_lca.calculation.pa_water_receiving import receive_pa_water


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
def test_actual_landfarming_keeps_market_spreading_and_native_distributions(
    tmp_path: Path, receiving: Literal["swiss", "sanitary"], future: bool, finish_dose: float
) -> None:
    """Compare final native units after nonunit production and collection exactly once.

    Actual wash water and actual landfill leachate use their real admission,
    water and agricultural owners. The separately tested pure soil equations
    are reused; this check covers their complete market/operation expansion,
    original distributions and dated supplies after leaving the source store.
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
    distributions: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    with boundary(tmp_path, future=future) as (_, context):
        if receiving == "swiss":
            water_reference = admit_pa_water_receiving_reference(context)
            water_sources = admit_pa_water_receiving_sources(context, water_reference)
            reference = replace(reference, water=water_reference)
            sources = replace(sources, water=water_sources)
            market_id = "pa_phosphate_sludge_landfarming_market_ch"
            operation_id = "pa_phosphate_sludge_landfarming_ch"
        else:
            landfill_reference = admit_pa_landfill_receiving_reference(context)
            landfill_sources = admit_pa_landfill_receiving_sources(context, landfill_reference)
            reference = replace(reference, landfill=landfill_reference)
            sources = replace(sources, landfill=landfill_sources)
            water_sources = landfill_sources
            market_id = "sbr_filled_fossil_sludge_landfarming_market"
            operation_id = "sbr_filled_fossil_sludge_landfarming"
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
        water = receive_pa_water(
            leachate,
            volume_m3=volume_m3,
            incoming_water_kg=incoming_water_kg,
            reference=reference,
            sources=sources,
            receiving=receiving,
        )
        services = {
            str(service["provider_service_id"]): service
            for service in cast(list[dict[str, object]], context.catalogue["services"])
        }
        market = _sludge_market_dataset(context, services[market_id], water.wastewater, "actual")
        operation = _landfarming_dataset(
            context,
            services[operation_id],
            water.wastewater,
            "actual_pa_water",
            "actual",
            complete_carbon_oxidation=True,
            phosphorus_equilibrium=True,
        )
        terminal_key = ("actual", provider_service_code(operation_id))
        market_rows = cast(list[dict[str, object]], market["exchanges"])
        market_production = -float(
            cast(float, next(row for row in market_rows if row["type"] == "production")["amount"])
        )
        # Both selected native markets send one wet kg to the sole terminal.
        # A different market ratio cannot be silently dropped during expansion.
        assert (
            -fsum(
                float(cast(float, row["amount"]))
                for row in market_rows
                if row["type"] == "technosphere" and row["input"] == terminal_key
            )
            / market_production
            == 1.0
        )
        for record in (market, operation):
            rows = cast(list[dict[str, object]], record["exchanges"])
            production = next(row for row in rows if row["type"] == "production")
            scale = (
                water.wastewater.landfarmed_sludge_wet_kg
                * water.treatment_fraction
                / -float(cast(float, production["amount"]))
            )
            for row in rows:
                key = cast(tuple[str, str], row["input"])
                if row["type"] == "production" or key == terminal_key:
                    continue
                scaled = _scaled_native_exchange(row, scale)
                expected[key] += float(cast(float, scaled["amount"]))
                if scaled.get("uncertainty type") == 2:
                    distributions[key].append(
                        {
                            field: scaled[field]
                            for field in ("amount", "loc", "scale", "minimum", "maximum")
                            if field in scaled
                        }
                    )

    actual = receive_pa_landfarmed_sludge(water, reference, sources, receiving=receiving)
    totals: dict[tuple[str, str], float] = defaultdict(float)
    actual_distributions: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in actual.amounts:
        totals[row.endpoint.key] += row.amount
        if row.metadata.get("uncertainty type") == 2:
            actual_distributions[row.endpoint.key].append(
                {
                    "amount": row.amount,
                    **{
                        field: row.metadata[field]
                        for field in ("loc", "scale", "minimum", "maximum")
                        if field in row.metadata
                    },
                }
            )
    assert dict(totals) == pytest.approx(dict(expected), rel=3e-13, abs=1e-20)
    assert actual_distributions.keys() == distributions.keys()
    for key, rows in distributions.items():
        assert len(actual_distributions[key]) == len(rows)
        for actual_row, expected_row in zip(actual_distributions[key], rows, strict=True):
            assert actual_row == pytest.approx(expected_row, rel=3e-13, abs=1e-20)
    assert actual.load.wet_kg == water.wastewater.landfarmed_sludge_wet_kg
    assert not any(
        row.endpoint.key == water_sources.sludge_landfarming.activity.key for row in actual.amounts
    )
    assert any(isinstance(row.endpoint, PaReceivingActivity) for row in actual.amounts)
    # Crop uptake and emitted phosphate retain their separate fates.
    assert actual.load.phosphorus_crop_uptake_kg >= 0
    assert all("fertili" not in row.endpoint.name for row in actual.amounts)
