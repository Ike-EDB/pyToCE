"""Admit complete dated PA sanitary receiving sources and retain calibration."""

import json
from dataclasses import asdict
from math import log
from pathlib import Path
from typing import Any, cast

import pytest
from pa_landfill_receiving_fixtures import landfill_receiving_boundary
from sbr_landfill_fixtures import OWNERS

from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    admit_pa_landfill_receiving_reference,
    admit_pa_landfill_receiving_sources,
    landfill_reference_from_payload,
    landfill_sources_from_payload,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration import (
    _sludge_source,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration_load import (
    _sludge_drivers,
)


@pytest.mark.parametrize("future", (False, True))
def test_complete_sanitary_sources_keep_original_calibration_and_detach_from_store(
    tmp_path: Path, future: bool
) -> None:
    """Check eleven native sanitary-treatment functions before and after JSON serialization.

    Future rows change production, supplier shares and doses while omitting
    source properties. The fixture distinguishes absent donor phosphorus
    from phosphorus in the target waste.
    """
    with landfill_receiving_boundary(tmp_path, future=future) as (bd, context):
        reference = admit_pa_landfill_receiving_reference(context)
        sources = admit_pa_landfill_receiving_sources(context, reference)
        database = "future" if future else "original"
        assert reference.landfill.key == ("original", "site")
        assert reference.climate.mean_annual_temperature_c == 18
        assert reference.climate.mean_annual_precipitation_mm == 1200
        assert reference.climate.methane_correction == 1
        # Original synthetic source: 0.006 m3/kg with -4 kg production. The future row is 0.009;
        # original calibration stays fixed.
        assert reference.leachate_volume_m3_per_wet_kg == 0.006
        assert reference.wastewater.key == ("original", "plant")
        assert reference.evaporation_fraction == 0.0
        assert reference.sludge.operation.activity.key == ("original", "thermal_sludge")
        assert reference.sludge.operation.production_amount == -2
        assert reference.sludge.elements_kg["C"] == pytest.approx(0.3 * 0.4)
        assert reference.sludge.elements_kg["N"] == pytest.approx(0.3 * 0.04)
        assert {"Hg", "Cd", "Be", "Sr", "Ti", "Tl"}.issubset(reference.sludge.elements_kg)
        assert "P" not in reference.sludge.elements_kg
        assert reference.landfarming.dry_fraction == 0.03
        assert reference.landfarming.water_fraction == 0.97
        assert reference.landfarming.element_contents == {
            "C": 0.4,
            "N": 0.04,
            "S": 0.02,
            "Na": 0.22 / 25,
            "Fe": 0.22 / 25,
        }
        for original in (reference.landfarming, reference.bottom_ash, reference.residual_ash):
            assert "P" not in original.element_contents
            assert not any(
                row.endpoint.name in {"Phosphate", "Phosphorus"} for row in original.operation.rows
            )
        for ash in (reference.bottom_ash, reference.residual_ash):
            assert ash.operation.activity.key[0] == "original"
            assert ash.dry_fraction == 0.8
            assert ash.element_contents["C"] == 0.4
        assert "Ti" not in reference.bottom_ash.element_contents
        assert reference.residual_ash.element_contents["Ti"] == 0.0022
        assert reference.residual_ash.element_contents["V"] == 0.22 / 25
        expected_productions = (
            (-9, -5, -7, -8, -10, -5, -11, -6, -6, -7, -7)
            if future
            else (-4, -2, -3, -4, -6, -2, -8, -4, -4, -3, -3)
        )
        operations = (
            sources.landfill,
            sources.collection,
            sources.wastewater,
            sources.sludge_incineration_market,
            sources.sludge_landfarming_market,
            sources.sludge_incineration,
            sources.sludge_landfarming,
            sources.bottom_ash_market,
            sources.residual_ash_market,
            sources.bottom_ash_treatment,
            sources.residual_ash_treatment,
        )
        for operation, (code, name, product, location), production in zip(
            operations, OWNERS.values(), expected_productions, strict=True
        ):
            assert operation.activity.key == (database, code)
            assert (
                operation.activity.name,
                operation.activity.reference_product,
                operation.activity.location,
            ) == (name, product, location)
            assert operation.production_amount == production
            assert operation.activity.unit == (
                "cubic meter" if code in {"collection", "plant"} else "kilogram"
            )
            node = bd.get_node(database=database, code=code)
            if future:
                assert all("properties" not in row for row in node.production())
            if operation is sources.sludge_incineration:
                continue  # Its actual duties replace source ash, power and powder rows.
            admitted = [
                row for row in operation.rows if isinstance(row.endpoint, PaReceivingActivity)
            ]
            native = list(node.technosphere())
            assert [row.endpoint.key for row in admitted] == [row.input.key for row in native]
            assert [row.amount for row in admitted] == pytest.approx(
                [row["amount"] / abs(production) for row in native], rel=2e-14
            )
        treated = next(
            row for row in sources.collection.rows if row.endpoint == sources.wastewater.activity
        )
        assert treated.amount == pytest.approx(-0.6 if future else -0.8)
        assert all(isinstance(row.endpoint, PaReceivingActivity) for row in sources.landfill.rows)
        raw = {
            row.endpoint.name
            for row in sources.collection.rows
            if isinstance(row.endpoint, PaReceivingFlow)
        }
        assert {"Sodium I", "Hydrogen", "Oxygen"}.issubset(raw)
        assert "Phosphorus" not in raw
        collector = bd.get_node(database=database, code="collection")
        assert all(
            "phosphorus content" not in row.get("properties", {}) for row in collector.production()
        )
        assert not {"Chlorine", "Magnesium", "Zinc II"}.intersection(raw)
        ammonia = [
            row
            for row in sources.sludge_incineration.rows
            if isinstance(row.endpoint, PaReceivingActivity)
            and row.endpoint.reference_product == "ammonia, anhydrous, liquid"
        ]
        doses = (0.0006 * 0.7, 0.0006 * 0.3) if future else (0.0002 * 0.25, 0.0002 * 0.75)
        for row, amount in zip(ammonia, doses, strict=True):
            assert row.amount == pytest.approx(amount)
            assert row.metadata["loc"] == pytest.approx(log(amount) + 0.03)
            assert row.metadata["scale"] == 0.21
            assert row.metadata["minimum"] == pytest.approx(amount / 2)
            assert row.metadata["maximum"] == pytest.approx(amount * 2)
        supplies = sources.sludge_incineration.supplies()
        heat = (
            "heat, district or industrial"
            if future
            else "heat, district or industrial, natural gas"
        )
        assert supplies[heat] == pytest.approx(0.18 if future else 0.06)
        for product in ("titanium dioxide", "chromium trioxide, flakes"):
            assert supplies[product] == pytest.approx(
                reference.sludge.operation.supplies()[product] * (3 if future else 1)
            )
        assert reference.sludge.operation.supplies()["ammonia, anhydrous, liquid"] == pytest.approx(
            0.0002
        )
        # Retain cement, hydration, and retirement once on the terminal's production reference.
        ash_supplies = sources.residual_ash_treatment.supplies()
        assert ash_supplies["cement"] == pytest.approx(0.4)
        assert ash_supplies["tap water"] == pytest.approx(0.6)
        assert ash_supplies["hydrated cement"] == pytest.approx(-0.8)
        assert sources.pam.key == (database, "mc_support_mc_pam")
        assert sources.pam.reference_product == "polyacrylamide"
        flows = {(flow.name, flow.unit, flow.categories): flow for flow in sources.biosphere}
        assert ("Phosphorus", "kilogram", ("water", "surface water")) in flows
        for categories in (
            ("water", "surface water"),
            ("water", "ground-"),
            ("water", "ground-, long-term"),
        ):
            assert ("Phosphate", "kilogram", categories) in flows
        for flow in flows.values():
            native = bd.get_node(database=flow.key[0], code=flow.key[1])
            assert (native["name"], native["unit"], tuple(native["categories"])) == (
                flow.name,
                flow.unit,
                flow.categories,
            )
        service = next(
            row
            for row in cast(list[dict[str, Any]], context.catalogue["services"])
            if row["provider_service_id"] == "sbr_filled_fossil_sludge_incineration"
        )
        live_source = _sludge_source(context, service["construction"])
        live_drivers = _sludge_drivers(live_source.elements_kg, 1.0)
        reference_json = json.dumps(asdict(reference), allow_nan=False)
        sources_json = json.dumps(asdict(sources), allow_nan=False)
    # Decode the stored JSON and evaluate the thermal equations after source preparation.
    detached_reference = landfill_reference_from_payload(json.loads(reference_json))
    detached_sources = landfill_sources_from_payload(json.loads(sources_json))
    assert detached_reference == reference
    assert detached_sources == sources
    assert _sludge_drivers(detached_reference.sludge.elements_kg, 1.0) == live_drivers
    assert detached_sources.residual_ash_treatment.supplies() == ash_supplies
