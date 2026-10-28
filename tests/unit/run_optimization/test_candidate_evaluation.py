from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from candidate_characterization_fixtures import _empty_pa_scoring
from candidate_worker_fixtures import _result_and_acc_metrics
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from material_recipe_fixtures import material_source_members
from pa_custody_fixtures import two_cycle_pa_mfa

from pytoce.download_data.sources.ecoinvent_materials import (
    EcoinventMaterialRecipe,
    derive_material_recipe,
)
from pytoce.prepare_backgrounds.preparation.premise import ACTIVE_BACKGROUND_SCENARIOS
from pytoce.run_lca.calculation.burdens import build_burden_demands
from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_lca.calculation.providers import MaterialCarbon
from pytoce.run_optimization.calculation.candidate_authority import (
    PhysicalEvaluation,
)
from pytoce.run_optimization.calculation.candidate_evaluation import (
    _characterize_prepared_selected_candidate,
    _selected_feedstock_requirements,
    _SelectedCharacterizationContext,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    _DatedProjection,
    _group_delta,
    _stable_service_response,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    decode_candidate_worker_result,
    encode_candidate_worker_result,
)


def test_worker_message_preserves_actual_pa_mixtures_and_binds_their_identity(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> None:
    """Keep actual PA event order and origin vectors through the worker's JSON boundary."""
    carbon, recipe = carbon_authorities
    mfa = two_cycle_pa_mfa(recipe, biogenic_share=0.35, dac_share=0.25)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    demands = build_burden_demands(mfa, carbon, trace)
    events = demands.pa_spinning_events
    recycling = demands.pa_recycling_events
    disposal = demands.pa_disposal_events
    assert len(events) == 4
    assert len(recycling) == 4
    assert disposal
    assert all(value > 0 for value in events[-1].maintenance_principal_by_origin_kg)
    assert all(value > 0 for value in disposal[-1].principal_by_origin_kg)
    result, _ = _result_and_acc_metrics(
        pa_spinning_events=events, pa_recycling_events=recycling, pa_disposal_events=disposal
    )

    encoded = encode_candidate_worker_result(result)
    decoded = decode_candidate_worker_result(encoded)

    assert decoded.physical.pa_spinning_events == events
    assert isinstance(decoded.physical.pa_spinning_events, tuple)
    assert decoded.physical.pa_recycling_events == recycling
    assert isinstance(decoded.physical.pa_recycling_events, tuple)
    assert decoded.physical.pa_disposal_events == disposal
    assert isinstance(decoded.physical.pa_disposal_events, tuple)
    assert isinstance(decoded.physical.pa_recycling_events[0].principal_by_origin_kg, tuple)
    assert isinstance(
        decoded.physical.pa_recycling_events[0].airborne_principal_by_origin_kg, tuple
    )
    assert isinstance(decoded.physical.pa_disposal_events[0].principal_by_origin_kg, tuple)
    assert decoded.physical.activation_identity == result.physical.activation_identity
    assert (
        decoded.physical.physical_candidate_identity == result.physical.physical_candidate_identity
    )
    assert decoded.identity == result.identity
    assert encode_candidate_worker_result(decoded) == encoded
    # A changed dose alone must change saved receiving, and worker
    # identities even when the linear provider-demand table is identical.
    changed_events = (
        *events[:-1],
        replace(events[-1], embedded_finish_dose_kg=events[-1].embedded_finish_dose_kg + 1),
    )
    changed_recycling = (
        *recycling[:-1],
        replace(
            recycling[-1],
            airborne_finish_dose_kg=recycling[-1].airborne_finish_dose_kg + 1,
        ),
    )
    changed_disposal = (
        *disposal[:-1],
        replace(disposal[-1], water_kg=disposal[-1].water_kg + 1),
    )
    assert result.physical.activation_delta is not None
    # Each actual-event group independently defines the saved candidate. A PM
    # finish change or wet-disposal change cannot reuse the old MFA result.
    for spinning_events, recycling_events, disposal_events in (
        (changed_events, recycling, disposal),
        (events, changed_recycling, disposal),
        (events, recycling, changed_disposal),
    ):
        changed, _ = _result_and_acc_metrics(
            pa_spinning_events=spinning_events,
            pa_recycling_events=recycling_events,
            pa_disposal_events=disposal_events,
        )
        assert changed.physical.activation_delta is not None
        pd.testing.assert_frame_equal(
            result.physical.activation_delta, changed.physical.activation_delta
        )
        assert changed.physical.activation_identity != result.physical.activation_identity
        assert (
            changed.physical.physical_candidate_identity
            != result.physical.physical_candidate_identity
        )
        assert changed.identity != result.identity


@pytest.mark.parametrize(
    ("material", "gross_polymer_per_finished"),
    (
        ("HDPE", 0.67 / 0.8 + 0.09 / 0.75 + 0.24 / 0.9),
        # The LDPE film proxy uses the fixture's 0.6 kg finished/kg input yield.
        ("LDPE", 1.0 / 0.6),
        # Finished-output weights across the ten Others constituents, using
        # the invented source yields and their retained principal mass.
        ("Others", 1.2231979524366756),
    ),
)
def test_selected_requirements_keep_physical_products_and_feedstock_routes(
    material: str, gross_polymer_per_finished: float
) -> None:
    """Report gross virgin monomer or polymer demand by feedstock route."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    state, inputs = empty_model(material)
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[2, region, good] = 1.0
    state.stock_by_cohort[2:, 2, region, good] = 1.0
    inputs.bio_production_rate[2, region] = 0.25
    inputs.daccu_production_rate[2, region] = 0.25
    evaluation = evaluate_mfa(state, inputs, recipe)

    requirements = _selected_feedstock_requirements(evaluation)

    assert requirements.year.between(2019, 2100).all()
    assert requirements.unit.eq("Mt").all()
    assert (requirements.amount >= 0).all()
    assert not requirements.loc[requirements.year.eq(2019), "amount"].any()
    positive = requirements.loc[requirements.amount.gt(0)]
    assert set(positive.material) == {material}
    assert set(positive.virgin_feedstock_route) == {"fossil", "biogenic", "atmospheric_dac"}
    if material != "Others":
        polymer = next(item for item in recipe.polymers if item.polymer == material.lower())
        expected = {
            "ethylene": polymer.ethylene_kg_per_kg,
            "propylene": polymer.propylene_kg_per_kg,
        }
        assert positive.quantity.eq("virgin_monomer_consumption").all()
    else:
        expected = {"polymer": 1.0}
        assert positive.quantity.eq("new_polymer_output").all()
    amounts = positive.groupby(["product", "virgin_feedstock_route"]).amount.sum()
    for product, amount in expected.items():
        for route, fraction in (("fossil", 0.5), ("biogenic", 0.25), ("atmospheric_dac", 0.25)):
            assert amounts.loc[(product, route)] == pytest.approx(
                gross_polymer_per_finished * amount * fraction, rel=1e-14, abs=0
            )


def test_provider_response_uses_reproducible_summation() -> None:
    amounts = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float64)
    unit_values = np.array(
        [[[[1.0e16], [1.0], [-1.0e16]]]],
        dtype=np.float64,
    )

    result = _stable_service_response(amounts, unit_values)

    assert result.shape == (1, 1, 1, 1)
    assert result[0, 0, 0, 0] == 1.0


def test_activation_grouping_uses_reproducible_summation() -> None:
    delta = pd.DataFrame(
        {
            "year": [2050, 2050, 2050],
            "region": ["EUR", "EUR", "EUR"],
            "provider_service_id": ["service-1", "service-1", "service-1"],
            "provider_service_code": ["code-1", "code-1", "code-1"],
            "parent_amount_kg": [0.0, 0.0, 1.0e16],
            "candidate_amount_kg": [1.0e16, 1.0, 0.0],
            "delta_amount_kg": [1.0e16, 1.0, -1.0e16],
        }
    )
    services = pd.DataFrame(
        {"provider_service_id": ["service-1"], "provider_service_code": ["code-1"]}
    )

    grouped, *_ = _group_delta(delta, services)

    assert np.count_nonzero(grouped) == 1
    assert grouped[grouped != 0.0].item() == 1.0


def test_prepared_selected_characterization_preserves_candidate_score_identities() -> None:
    backgrounds = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
    years = tuple(range(2019, 2101))
    methods = pd.DataFrame.from_records(
        [
            {
                "lcia_method": "ef31",
                "selection_order": 0,
                "method_id": "ef31",
                "method_version": "3.1",
                "impact_category_id": "climate_change",
                "impact_category_tuple_json": '["EF v3.1","climate change"]',
                "unit": "kg CO2-Eq",
                "factor_identity": "f" * 64,
            }
        ]
    )
    projection = _DatedProjection(
        identity="dated-projection",
        projection_identity="result-projection",
        backgrounds=backgrounds,
        response_bank_identity="response-bank",
        method_registry_identity="result-methods",
        method_compatibility_identity="result-method-compatibility",
        method_registry=methods,
        services=pd.DataFrame.from_records(
            [
                {
                    "provider_service_id": "service",
                    "provider_service_code": "service-code",
                    "provider_unit": "kilogram",
                }
            ]
        ),
        scores=np.zeros((len(backgrounds), len(years), 1, 1), dtype=np.float64),
        norms=np.ones((len(backgrounds), len(years), 1, 1), dtype=np.float64),
        scores_identity="dated-scores",
        norms_identity="dated-norms",
        services_identity="services",
    )
    baseline_scores = np.zeros(
        (len(backgrounds), len(H12_REGIONS), len(years), 1),
        dtype=np.float64,
    )
    baseline_scores[0, 0, 0, 0] = -0.0
    baseline_norms = np.ones_like(baseline_scores)
    _, pa_parent = _empty_pa_scoring(methods)
    context = _SelectedCharacterizationContext(
        physical_parent_identity="physical-parent",
        result_parent_identity="result-parent",
        projection=projection,
        baseline_scores=baseline_scores,
        baseline_norms=baseline_norms,
        pa_parent=pa_parent,
        pa_indices=np.array([0], dtype=np.int64),
    )

    def physical(identity: str) -> PhysicalEvaluation:
        """Bind equal activations to distinct candidates for reporting."""
        return PhysicalEvaluation(
            physical_request_identity=f"request::{identity}",
            physical_candidate_identity=f"candidate::{identity}",
            physical_parent_identity=context.physical_parent_identity,
            intervention_authority_identity=f"intervention::{identity}",
            status="physical_feasible",
            diagnostic=None,
            physical_state_identity=f"physical-state::{identity}",
            activation_identity=f"activation::{identity}",
            activation_delta=pd.DataFrame(),
            pa_spinning_events=(),
            pa_recycling_events=(),
            pa_disposal_events=(),
        )

    first = _characterize_prepared_selected_candidate(
        physical("first"),
        activation_delta=pd.DataFrame(),
        context=context,
        pa_scores=pa_parent,
    )
    second = _characterize_prepared_selected_candidate(
        physical("second"),
        activation_delta=pd.DataFrame(),
        context=context,
        pa_scores=pa_parent,
    )

    for observed in (first, second):
        assert np.array_equal(observed.annual_scores, baseline_scores)
        assert np.array_equal(np.signbit(observed.annual_scores), np.signbit(baseline_scores))
        assert np.array_equal(observed.annual_contribution_norms, baseline_norms)
    assert first.result_scoring_identity != second.result_scoring_identity
    assert first.score_identity != second.score_identity
    assert first.method_registry.equals(second.method_registry)
