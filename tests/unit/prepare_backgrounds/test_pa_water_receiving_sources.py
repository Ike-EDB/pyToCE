"""Preserve original Swiss source coefficients and dated polyamide wastewater records."""

import json
from dataclasses import asdict
from math import log
from pathlib import Path
from typing import Any, cast

import pytest
from pa_water_receiving_fixtures import water_receiving_boundary
from swiss_sludge_fixtures import wastewater as constructed_wastewater

from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    admit_pa_water_receiving_reference,
    admit_pa_water_receiving_sources,
    water_reference_from_payload,
    water_sources_from_payload,
)
from pytoce.prepare_backgrounds.preparation.providers.swiss_sludge_incineration import (
    swiss_sludge_load,
)
from pytoce.prepare_backgrounds.preparation.providers.swiss_sludge_load import (
    calculate_swiss_sludge,
)
from pytoce.prepare_backgrounds.preparation.providers.water_treatment import (
    _water_evaporation,
)


@pytest.mark.parametrize("future", (False, True))
def test_ch_receiving_records_keep_source_calibration_and_detached_complete_inputs(
    tmp_path: Path, future: bool
) -> None:
    """Save synthetic source records and reuse them in the Swiss sludge calculation.

    Both cases include Swiss collection, wastewater treatment, sludge markets,
    FLUWA and ash treatment. The future case omits production properties, doubles
    reference amounts and changes peroxide suppliers; original coefficients remain.
    This tests record persistence, not final polyamide receiving results.
    """
    with water_receiving_boundary(tmp_path, future=future) as (bd, context):
        reference = admit_pa_water_receiving_reference(context)
        sources = admit_pa_water_receiving_sources(context, reference)
        database = "future" if future else "original"
        assert reference.wastewater.key == ("original", "mibk_plant_ch")
        assert reference.evaporation_fraction == _water_evaporation(
            "Mean annual temperature (MAT) of 10; mean annual precipitation (MAP) of 1000; "
            "actual evapotranspiration (EAT) of 500.",
            _CH_WASTEWATER_PROFILE.hydraulic,
        )
        assert reference.sludge.operation.activity.key == ("original", "organic_ch_sludge")
        assert reference.sludge.operation.production_amount == -2
        assert reference.sludge.elements_kg["C"] == pytest.approx(0.3 * 0.4)
        assert reference.sludge.elements_kg["N"] == pytest.approx(0.3 * 0.04)
        # Donor metals contribute to source allocation weights; actual PA inputs use the finish
        # recipe.
        assert {"Hg", "Cd", "Mg", "Zn"}.issubset(reference.sludge.elements_kg)
        assert reference.landfarming.dry_fraction == 0.03
        assert reference.landfarming.water_fraction == 0.97
        assert reference.landfarming.element_contents == {
            "C": 0.4,
            "N": 0.04,
            "S": 0.02,
            "Fe": 0.22 / 25,
            "Na": 0.22 / 25,
        }
        assert not any(
            row.endpoint.name == "Phosphate" for row in reference.landfarming.operation.rows
        )
        assert "P" not in reference.landfarming.element_contents
        for ash in (reference.bottom_ash, reference.residual_ash):
            assert ash.operation.activity.key[0] == "original"
            assert ash.dry_fraction == 0.8
            assert ash.element_contents["C"] == pytest.approx(0.399)
            assert ash.element_contents["P"] == 0.02
        assert "Ti" not in reference.bottom_ash.element_contents
        assert reference.residual_ash.element_contents["Ti"] == 0.001
        assert reference.residual_ash.element_contents["V"] == 0.22 / 25
        assert sources.collection.activity.key == (database, "mibk_water_ch")
        assert sources.wastewater.activity.key == (database, "mibk_plant_ch")
        assert sources.collection.production_amount == (-10 if future else -5)
        assert sources.wastewater.production_amount == (-14 if future else -7)
        assert sources.sludge_incineration.production_amount == (-4 if future else -2)
        assert sources.sludge_landfarming.production_amount == (-16 if future else -8)
        treated = next(
            row for row in sources.collection.rows if row.endpoint == sources.wastewater.activity
        )
        assert treated.amount == pytest.approx(-0.8)
        raw = {
            row.endpoint.name: row.endpoint
            for row in sources.collection.rows
            if isinstance(row.endpoint, PaReceivingFlow)
        }
        assert raw["Phosphorus"].categories == ("water",)
        assert raw["Sodium I"].categories == ("water",)
        assert raw["Chlorine"].categories == ("water",)
        # Retain complete dated technosphere rows, source signs, hydraulics, and haulage.
        for operation in (
            sources.collection,
            sources.wastewater,
            sources.sludge_incineration_market,
            sources.sludge_landfarming_market,
            sources.sludge_landfarming,
            sources.bottom_ash_market,
            sources.residual_ash_market,
            sources.bottom_ash_treatment,
            sources.residual_ash_treatment,
        ):
            node = bd.get_node(database=operation.activity.key[0], code=operation.activity.key[1])
            admitted = [
                row for row in operation.rows if isinstance(row.endpoint, PaReceivingActivity)
            ]
            native = list(node.technosphere())
            assert [row.endpoint.key for row in admitted] == [row.input.key for row in native]
            assert [row.amount for row in admitted] == pytest.approx(
                [row["amount"] / abs(operation.production_amount) for row in native], rel=2e-14
            )
        peroxide = [
            row
            for row in sources.sludge_incineration.rows
            if isinstance(row.endpoint, PaReceivingActivity)
            and row.endpoint.reference_product
            == "hydrogen peroxide, without water, in 50% solution state"
        ]
        assert [row.endpoint.key for row in peroxide] == [
            (database, "organic_ch_peroxide_a"),
            (database, "organic_ch_peroxide_b"),
        ]
        doses = (0.0018 * 0.7, 0.0018 * 0.3) if future else (0.0006 * 0.25, 0.0006 * 0.75)
        for row, dose in zip(peroxide, doses, strict=True):
            assert row.amount == pytest.approx(dose)
            assert row.metadata["loc"] == pytest.approx(log(dose) + 0.03)
            assert row.metadata["scale"] == 0.21
            assert row.metadata["minimum"] == pytest.approx(dose / 2)
            assert row.metadata["maximum"] == pytest.approx(dose * 2)
        sludge_supplies = sources.sludge_incineration.supplies()
        for product in (
            "titanium dioxide",
            "chromium trioxide, flakes",
            "iron(III) chloride, without water, in 40% solution state",
        ):
            assert sludge_supplies[product] == pytest.approx(
                reference.sludge.operation.supplies()[product]
            )
        assert sludge_supplies["metalliferous hydroxide sludge"] == -0.003
        heat = (
            "heat, district or industrial"
            if future
            else "heat, district or industrial, natural gas"
        )
        assert sludge_supplies[heat] == 0.06
        assert reference.sludge.source().inputs[
            "hydrogen peroxide, without water, in 50% solution state"
        ] == pytest.approx(0.0006)
        if future:
            hydroxide = next(
                row
                for row in sources.sludge_incineration.rows
                if isinstance(row.endpoint, PaReceivingActivity)
                and row.endpoint.reference_product == "metalliferous hydroxide sludge"
            )
            assert hydroxide.metadata["negative"] is False
            assert hydroxide.metadata["loc"] == -0.003
            assert not {"scale", "shape", "minimum", "maximum"}.intersection(hydroxide.metadata)
            for operation in (
                sources.wastewater,
                sources.sludge_incineration,
                sources.sludge_landfarming,
                sources.bottom_ash_treatment,
                sources.residual_ash_treatment,
            ):
                node = bd.get_node(database=database, code=operation.activity.key[1])
                assert all("properties" not in row for row in node.production())
        assert sources.pam.reference_product == "polyacrylamide"
        assert sources.pam.key[0] == database
        flows = {(flow.name, flow.unit, flow.categories): flow for flow in sources.biosphere}
        for categories in (("water", "surface water"), ("water", "ground-")):
            phosphate = flows["Phosphate", "kilogram", categories]
            assert (
                bd.get_node(database=phosphate.key[0], code=phosphate.key[1])["categories"]
                == categories
            )
        assert ("Oxygen", "kilogram", ("natural resource", "in air")) in flows
        assert ("Chloride", "kilogram", ("water", "surface water")) in flows
        assert (
            "Non-hazardous waste disposed",
            "kilogram",
            ("inventory indicator", "waste"),
        ) in flows
        # Use the synthetic PA6 finish and caprolactam mixture from the Swiss
        # sludge tests. Carbon-origin labels are supplied test controls for this mixture.
        wastewater = constructed_wastewater("finish")
        service = next(
            service
            for service in cast(list[dict[str, Any]], context.catalogue["services"])
            if service["provider_service_id"] == "pa_phosphate_sludge_incineration_ch"
        )
        live_sludge = swiss_sludge_load(context, service["construction"], wastewater)
        reference_json = json.dumps(asdict(reference), allow_nan=False)
        sources_json = json.dumps(asdict(sources), allow_nan=False)

    decoded_reference = water_reference_from_payload(json.loads(reference_json))
    decoded_sources = water_sources_from_payload(json.loads(sources_json))
    assert decoded_reference == reference
    assert decoded_sources == sources
    detached_sludge = calculate_swiss_sludge(decoded_reference.sludge.source(), wastewater)
    assert detached_sludge == live_sludge
    assert detached_sludge.bottom_ash.treatment_mass_kg > 0
    assert detached_sludge.residual_ash.treatment_mass_kg > 0
