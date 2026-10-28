"""Connect actual PA recycling events to complete detached operating sources."""

from dataclasses import asdict, replace
from math import fsum, log
from pathlib import Path
from typing import Any, cast

import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from material_recipe_fixtures import physical_material_recipe
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources
from pa_recycling_receiving_fixtures import (
    RECYCLING_SERVICES,
    recycling_biosphere,
    recycling_native_inventory,
)
from pa_water_receiving_fixtures import water_receiving_boundary

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    pa_recycling_receiving_sources as recycling,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    admit_pa_water_receiving_reference,
    admit_pa_water_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.run_lca.calculation.burdens import build_burden_demands
from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_lca.calculation.pa_recycling_receiving import receive_pa_recycling


def _recycling_sources(
    root: Path, *, future: bool
) -> tuple[recycling.PaRecyclingReceivingReference, recycling.PaRecyclingReceivingSources]:
    """Compile complete invented recycling functions and detach their real rows.

    Separate fixture database names prevent accidental identity collisions with
    the independently admitted Swiss wastewater fixture. All operations and their
    delivery dependencies are real source or prepared graph nodes.
    """
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    names = BrightwayNames(
        root / "brightway", "pa_recycling", "recycling_biosphere", "recycling_original"
    )
    database = "recycling_future" if future else names.source_database
    with selected_brightway_project(names) as bd:
        biosphere = {
            (names.biosphere_database, code): row
            for (_, code), row in recycling_biosphere().items()
        }
        bd.Database(names.biosphere_database).write(biosphere, searchable=False)
        for source_database, dated in (
            (names.source_database, False),
            ("recycling_future", True),
        ):
            records = recycling_native_inventory(
                cast(dict[str, Any], catalogue), source_database, future=dated
            )
            for record in records.values():
                for row in record["exchanges"]:
                    if row["type"] == "biosphere":
                        row["input"] = names.biosphere_database, row["input"][1]
            bd.Database(source_database).write(records, searchable=False)
        native = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi" if future else None,
            year=2050 if future else 2025,
            provider_database="recycling_frontier",
            source=_ActivityResolver(bd, names.source_database),
            ordinary=native,
            technology=native,
            biosphere=_FlowResolver(bd, names.biosphere_database),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        reference = recycling.admit_pa_recycling_receiving_reference(context)
        records = {
            (context.provider_database, provider_service_code(service)): _service_dataset(
                context, services[service], services, context.provider_database
            )
            for service in RECYCLING_SERVICES
        }
        bd.Database(context.provider_database).write(records, searchable=False)
        sources = recycling.admit_pa_recycling_receiving_sources(context)
        reference_payload, sources_payload = asdict(reference), asdict(sources)
    return (
        recycling.recycling_reference_from_payload(reference_payload),
        recycling.recycling_sources_from_payload(sources_payload),
    )


@pytest.mark.parametrize("future", (False, True))
def test_actual_pa_recycling_uses_one_wash_dust_receiving_and_wet_delivery(
    tmp_path: Path, carbon_authorities, future: bool
) -> None:
    """Keep both PA grades and recycled cohorts on their actual mixed-material basis.

    Real MFA/custody produces the activities. The previously tested source
    fixtures provide complete WEEE/PET/bulk and Swiss water functions, including
    nonunit production and dated suppliers. This test protects their actual
    receiving handoff, not another copy of the individual chemistry tests.
    """
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
    # Move this four-year fixture to 2019--2022, so the second recycling
    # receives the 2020 cohort after biological/DAC feedstock becomes available.
    state = replace(state, years=(2019, 2020, 2021, 2022))
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    for time in (0, 1, 2):
        state.annual_inflow[time, region, good] = 1
        state.stock_by_cohort[time, time, region, good] = 1
        state.outflow_by_cohort[time + 1, time, region, good] = 1
    for time in (1, 2):
        inputs.mechanical_recycling_rate[time, region] = 0.5
        inputs.incineration_rate[time, region] = 0.25
        inputs.bio_production_rate[time, region] = 0.35
        inputs.daccu_production_rate[time, region] = 0.25
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    events = build_burden_demands(mfa, carbon, trace).pa_recycling_events
    reference = pa_receiving_reference()
    sources = pa_receiving_sources(2050 if future else 2025, background_scenario="member")
    recycling_reference, recycling_sources = _recycling_sources(
        tmp_path / "recycling", future=future
    )
    with water_receiving_boundary(tmp_path / "water", future=future) as (_, context):
        water_reference = admit_pa_water_receiving_reference(context)
        water_sources = admit_pa_water_receiving_sources(context, water_reference)
    reference = replace(reference, recycling=recycling_reference, water=water_reference)
    sources = replace(sources, recycling=recycling_sources, water=water_sources)
    assert {event.constituent for event in events} == {"polyamide_6_resin", "polyamide_66_resin"}
    assert any(event.embedded_finish_dose_kg == 0 for event in events)
    assert any(event.embedded_finish_dose_kg > 0 for event in events)
    assert any(all(mass > 0 for mass in event.principal_by_origin_kg) for event in events)
    database = "recycling_future" if future else "recycling_original"
    for event in events:
        actual = receive_pa_recycling(event, reference, sources)
        grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
        finish = _pa_finish_reference(grade)
        principal = fsum(event.principal_by_origin_kg)
        retained_mass = fsum(finish.retained.compounds_kg.values())
        incoming = principal + retained_mass * (principal + event.embedded_finish_dose_kg)
        # Original dry feed is 1.6 kg/kg and water 4.8 kg/kg, giving 3 kg wash water/kg incoming dry
        # mixture. The future operating-water coefficient is 6 kg/kg; original calibration remains
        # fixed.
        assert actual.load.incoming_dry_kg == pytest.approx(incoming, rel=2e-14)
        assert actual.load.fresh_wash_water_kg == pytest.approx(3 * incoming, rel=2e-14)
        prep = [row for row in actual.amounts if row.function == "pa_recycling_preparation"]
        expected = {
            "electricity": incoming / 1.6 * (0.8 if future else 0.6),
            "tap_water": 3 * incoming * 0.75,
            "second_water": 3 * incoming * 0.25,
            "capital": incoming / 1.6 * 0.004,
        }
        assert {row.endpoint.key for row in prep} == {(database, code) for code in expected}
        for row in prep:
            assert row.amount == pytest.approx(expected[row.endpoint.key[1]], rel=2e-14)
            assert row.metadata["loc"] == pytest.approx(log(row.amount) + 0.02, rel=2e-14)
            assert row.metadata["scale"] == 0.13
        pellet = actual.load.pellet_dry_kg
        finishing = [row for row in actual.amounts if row.function == "pa_recycling_finishing"]
        expected = {
            "electricity": pellet * (1.8 if future else 1.5),
            "tap_water": pellet * 0.0004,
            "cooling_water_receiving": -pellet * 0.0004 / 1000,
            "capital": pellet * 0.002,
        }
        assert {row.endpoint.key for row in finishing} == {(database, code) for code in expected}
        for row in finishing:
            assert row.amount == pytest.approx(expected[row.endpoint.key[1]], rel=2e-14)
            assert row.metadata["loc"] == pytest.approx(log(abs(row.amount)) + 0.02, rel=2e-14)
            assert row.metadata["negative"] == (row.amount < 0)
        pm = [row for row in actual.amounts if row.endpoint.key == ("recycling_biosphere", "pm")]
        assert len(pm) == 1
        assert pm[0].amount == pytest.approx(
            fsum(event.airborne_principal_by_origin_kg)
            + retained_mass * event.airborne_finish_dose_kg,
            rel=2e-14,
        )
        assert pm[0].function == "pa_recycling_dust"
        assert pm[0].metadata["uncertainty type"] == 0
        assert pm[0].metadata["loc"] == pm[0].amount
        assert "scale" not in pm[0].metadata
        assert pm[0].metadata["source_operations"] == (
            recycling_sources.preparation.activity.key,
            recycling_sources.finishing.activity.key,
        )
        # Both dry-pellet cargo and its carried 500 ppm water travel once.
        wet_pellet = pellet / 0.9995
        (delivery,) = (row for row in actual.amounts if row.function == "pa_recycling_delivery")
        assert delivery.amount == pytest.approx(wet_pellet, rel=2e-14)
        assert delivery.endpoint == delivery.source == recycling_sources.pellet_delivery
        assert delivery.metadata == {"uncertainty type": 0, "loc": delivery.amount}
        air = next(row for row in actual.amounts if row.function == "pa_recycling_drying")
        assert air.amount * 1000 == actual.load.air_water_kg
        assert actual.load.fresh_wash_water_kg + actual.load.incoming_water_kg == pytest.approx(
            actual.load.collected_water_kg + air.amount * 1000 + actual.load.pellet_water_kg,
            rel=2e-14,
        )
        removed = actual.load.removed_finish
        assert actual.water.wastewater.volume_m3 == pytest.approx(
            (actual.load.collected_water_kg + fsum(removed.compounds_kg.values())) / 1000,
            rel=2e-14,
        )
        assert actual.water.wastewater.water.incoming_kg == actual.load.collected_water_kg
        assert not any(pool.principal for pool in actual.water.wastewater.carbon)
        assert fsum(
            pool.input_carbon_kg
            for pool in actual.water.wastewater.carbon
            if pool.component != "wastewater_polyacrylamide"
        ) == pytest.approx(removed.fossil_carbon_kg + removed.biogenic_carbon_kg, rel=2e-14)
        assert {
            "pa_swiss_collection",
            "pa_swiss_wastewater",
            "pa_wastewater_gas_leakage",
            "pa_wastewater_gas_flare",
            "pa_wastewater_gas_chp",
            "pa_swiss_sludge_incineration",
            "pa_swiss_bottom_ash",
            "pa_swiss_residual_ash",
            "pa_swiss_sludge_landfarming",
        } <= {row.function for row in actual.amounts}
        native_children = {
            operation.activity.key
            for operation in (
                recycling_sources.preparation,
                recycling_sources.finishing,
                water_sources.collection,
                water_sources.wastewater,
                water_sources.sludge_incineration,
                water_sources.sludge_landfarming,
                water_sources.bottom_ash_treatment,
                water_sources.residual_ash_treatment,
            )
        }
        assert not any(row.endpoint.key in native_children for row in actual.amounts)
        assert not any(
            isinstance(row.endpoint, PaReceivingActivity)
            and row.endpoint.reference_product == "wastewater from recycling of waste PP"
            for row in actual.amounts
        )
        assert all(
            row.amount >= 0 for row in actual.amounts if isinstance(row.endpoint, PaReceivingFlow)
        )
