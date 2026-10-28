"""Admit real native identities and bind actual mismanaged PA fiber events."""

import json
from dataclasses import asdict
from math import fsum
from pathlib import Path
from typing import cast

import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from material_recipe_fixtures import physical_material_recipe
from pa_open_burning_fixtures import (
    _AIR,
    _FLOWS,
    _INDICATOR,
    open_burning_biosphere,
    open_burning_native_inventory,
)

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.open_burning import (
    _pa_open_burning_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_open_burning_sources import (
    PaOpenBurningReference,
    admit_pa_open_burning_reference,
    open_burning_reference_from_payload,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.run_lca.calculation.burdens import build_burden_demands
from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_lca.calculation.pa_open_burning_receiving import receive_pa_open_burning
from pytoce.run_lca.calculation.pa_receiving import PaDisposalEvent
from pytoce.run_lca.calculation.providers import MaterialCarbon


def _admitted_reference(root: Path, *, future: bool) -> PaOpenBurningReference:
    """Admit original output identities with a different ordinary source present.

    The two real tiny source graphs deliberately have different production and
    direct coefficients. Only their original donor identity is applicable; the
    actual PA amounts come from the scientific owner, not either donor vector.
    An additional mixed-waste PCB and wrong-compartment CO2 remain unselected.
    """
    authority = prepared_provider_authority()
    names = BrightwayNames(root / "brightway", "burning", "burning_biosphere", "original")
    with selected_brightway_project(names) as bd:
        flows = open_burning_biosphere(names.biosphere_database)
        bd.Database(names.biosphere_database).write(flows, searchable=False)
        for database in ("original", "future"):
            bd.Database(database).write(
                open_burning_native_inventory(database, original=database == "original"),
                searchable=False,
            )
        ordinary = _ActivityResolver(bd, "future" if future else "original")
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi" if future else None,
            year=2050 if future else 2025,
            provider_database="unused_frontier",
            source=_ActivityResolver(bd, "original"),
            ordinary=ordinary,
            technology=ordinary,
            biosphere=_FlowResolver(bd, names.biosphere_database),
            catalogue=authority.catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in authority.method_authorities
            },
            complete_source_ids=_service_source_closures(authority.catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        admitted = admit_pa_open_burning_reference(context)
        payload = json.dumps(asdict(admitted), allow_nan=False)
    # All subsequent receiving is detached from the selected Brightway project.
    restored = open_burning_reference_from_payload(cast(dict[str, object], json.loads(payload)))
    assert restored == admitted
    return restored


@pytest.fixture(scope="module")
def burned_pa_events(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> tuple[PaDisposalEvent, ...]:
    """Retire real first-use and twice-recycled cohorts with uncollected material."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
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
    # Uncollected material follows the Cottom source-matched fate split.
    inputs.collection_rate[1:, region] = 0.6
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    return tuple(
        event
        for event in build_burden_demands(mfa, carbon, trace).pa_disposal_events
        if event.demand_type == "mismanaged_open_burning_treatment"
    )


@pytest.mark.parametrize("future", (False, True))
def test_actual_pa_open_burning_binds_complete_outputs_once(
    tmp_path: Path,
    burned_pa_events: tuple[PaDisposalEvent, ...],
    future: bool,
) -> None:
    """Keep the actual mixed load, native units, origin fractions and on-site residue."""
    reference = _admitted_reference(tmp_path, future=future)
    assert reference.activity.key == ("original", "burning")
    assert reference.activity.location == "GLO"
    assert {(f.name, f.unit, f.categories) for f in reference.biosphere} == set(_FLOWS)
    assert {f.key for f in reference.biosphere} == {
        ("burning_biosphere", f"f{index}") for index in range(len(_FLOWS))
    }
    assert len(burned_pa_events) == 6
    assert {e.constituent for e in burned_pa_events} == {"polyamide_6_resin", "polyamide_66_resin"}
    assert any(all(p > 0 for p in e.principal_by_origin_kg) for e in burned_pa_events)
    assert any(e.total_finish_dose_kg > fsum(e.principal_by_origin_kg) for e in burned_pa_events)
    for event in burned_pa_events:
        # Both PA grades use the real seven-ingredient finish composition.
        # The event supplies its already allocated principal and finish dose.
        actual = receive_pa_open_burning(event, reference)
        grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
        finish = _pa_finish_reference(grade)
        principal = fsum(event.principal_by_origin_kg)
        expected = _pa_open_burning_load(
            finish,
            principal_by_origin_kg={
                "fossil": event.principal_by_origin_kg[0],
                "biogenic": event.principal_by_origin_kg[1],
                "atmospheric_dac": event.principal_by_origin_kg[2],
            },
            finish_dose_kg=event.total_finish_dose_kg,
            water_kg=event.water_kg,
        )
        assert actual.load == expected
        wet = (
            principal
            + event.total_finish_dose_kg * fsum(finish.retained.compounds_kg.values())
            + event.water_kg
        )
        assert actual.load.wet_mass_kg == pytest.approx(wet, rel=3e-14)
        assert event.water_kg == pytest.approx(principal * (0.044 if grade == "pa6" else 0.043))
        assert actual.load.principal_carbon_kg == pytest.approx(
            {
                origin: amount * 72 / 113
                for origin, amount in zip(
                    ("fossil", "biogenic", "atmospheric_dac"),
                    event.principal_by_origin_kg,
                    strict=True,
                )
            }
        )
        outputs = {
            (r.endpoint.name, r.endpoint.unit, cast(PaReceivingFlow, r.endpoint).categories): r
            for r in actual.amounts
        }
        assert set(outputs) == set(_FLOWS) == set(expected.biosphere_amounts)
        assert len(actual.amounts) == len(outputs)
        for key, row in outputs.items():
            assert isinstance(row.endpoint, PaReceivingFlow)
            assert row.source == reference.activity
            assert row.function == "pa_open_burning"
            assert row.amount == expected.biosphere_amounts[key]
            assert row.metadata == {"uncertainty type": 0, "loc": row.amount}
            assert row.principal_carbon_origin_fractions == (
                tuple(
                    expected.principal_origin_fractions[key].get(origin, 0.0)
                    for origin in ("fossil", "biogenic", "atmospheric_dac")
                )
                if key in expected.principal_origin_fractions
                else None
            )
        assert outputs[
            "Non-hazardous waste disposed", "kilogram", _INDICATOR
        ].amount == pytest.approx(wet)
        assert (
            outputs["Water", "cubic meter", _AIR].amount
            == (event.water_kg + actual.load.formed_water_kg) / 1000
        )
        # Indicators describe overlapping material pools. Ambient oxygen and nitrogen enter as
        # separate inputs.
        assert fsum(actual.load.air_compounds_kg.values()) + fsum(
            actual.load.air_oxide_elements_kg.values()
        ) + actual.load.residue_mass_kg == pytest.approx(
            wet + actual.load.atmospheric_oxygen_kg + actual.load.atmospheric_nitrogen_kg,
            rel=3e-14,
        )
