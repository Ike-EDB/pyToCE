"""Focused candidate worker contracts."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from candidate_worker_fixtures import _infeasible_result, _pa_scores, _result, _summary, _task
from material_recipe_fixtures import material_source_members

from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.run_mfa.model.materials import load_material_composition, realize_materials
from pytoce.run_mfa.scenario.scenario_contract import ScenarioFeasibilityError
from pytoce.run_optimization.calculation.candidate_authority import (
    physical_evaluation_identity,
)
from pytoce.run_optimization.calculation.candidate_search import InterventionCandidate
from pytoce.run_optimization.calculation.request_search import _candidate_evaluation
from pytoce.run_optimization.calculation.worker_protocol import (
    _selected_candidate_result,
    candidate_worker_result_identity,
    decode_candidate_worker_result,
    decode_candidate_worker_task,
    encode_candidate_worker_result,
    encode_candidate_worker_task,
)


def test_task_message_is_canonical_and_round_trips() -> None:
    task = _task()

    encoded = encode_candidate_worker_task(task)
    decoded = decode_candidate_worker_task(encoded)

    assert decoded == task
    assert encode_candidate_worker_task(decoded) == encoded


@pytest.mark.parametrize("active_pa", (False, True))
def test_result_round_trip_preserves_arrow_and_array_coordinates(active_pa: bool) -> None:
    result = replace(_result(), pa_scores=_pa_scores(active=active_pa))
    result = replace(result, identity=candidate_worker_result_identity(result))

    encoded = encode_candidate_worker_result(result)
    decoded = decode_candidate_worker_result(encoded)

    assert decoded.identity == result.identity
    assert decoded.physical.activation_delta is not None
    assert result.physical.activation_delta is not None
    assert decoded.physical.activation_delta.equals(result.physical.activation_delta)
    assert decoded.physical.physical_state_identity == result.physical.physical_state_identity
    assert decoded.scores is not None
    assert result.scores is not None
    assert np.array_equal(decoded.scores.annual_scores, result.scores.annual_scores)
    assert np.array_equal(decoded.scores.annual_climate_scores, result.scores.annual_climate_scores)
    assert decoded.scores.backgrounds == result.scores.backgrounds
    assert decoded.scores.regions == result.scores.regions
    assert decoded.scores.years == result.scores.years
    assert decoded.scores.annual_categories == result.scores.annual_categories
    assert decoded.feedstock_requirements is not None and result.feedstock_requirements is not None
    assert decoded.feedstock_requirements.equals(result.feedstock_requirements)
    assert decoded.pa_scores is not None and result.pa_scores is not None
    np.testing.assert_array_equal(decoded.pa_scores.annual_scores, result.pa_scores.annual_scores)
    np.testing.assert_array_equal(
        decoded.pa_scores.annual_contribution_norms, result.pa_scores.annual_contribution_norms
    )
    pd.testing.assert_frame_equal(decoded.pa_scores.stage_scores, result.pa_scores.stage_scores)
    assert not decoded.pa_scores.annual_scores.flags.writeable
    assert not decoded.pa_scores.annual_contribution_norms.flags.writeable
    assert decoded.pa_scores.stage_scores.empty is (not active_pa)
    if active_pa:
        assert decoded.pa_scores.annual_scores.min() < 0
        assert np.all(
            decoded.pa_scores.annual_contribution_norms >= np.abs(decoded.pa_scores.annual_scores)
        )
    assert encode_candidate_worker_result(decoded) == encoded


@pytest.mark.parametrize("change", ("year", "shortfall"))
def test_physical_failure_round_trip_preserves_diagnostics_without_score_arrays(
    change: str,
) -> None:
    """Round-trip the canonical result produced for one candidate that violates MFA constraints."""

    result = _infeasible_result()

    decoded = decode_candidate_worker_result(encode_candidate_worker_result(result))
    assert decoded == result
    assert decoded.scores is None
    assert decoded.pa_scores is None
    assert decoded.physical.diagnostic == result.physical.diagnostic
    assert decoded.physical.physical_state_identity is None

    selected = _selected_candidate_result(
        decoded,
        _summary(decoded),
        selection_identities=("selection::2050",),
    )
    diagnostic = decoded.physical.diagnostic
    assert diagnostic is not None
    changed_diagnostic = (
        replace(diagnostic, coordinates=(("year", 2051), *diagnostic.coordinates[1:]))
        if change == "year"
        else replace(diagnostic, observed=0.5, shortfall=0.5)
    )
    changed_physical = replace(decoded.physical, diagnostic=changed_diagnostic)
    changed_physical = replace(
        changed_physical, physical_candidate_identity=physical_evaluation_identity(changed_physical)
    )
    changed = replace(decoded, physical=changed_physical)
    changed = replace(changed, identity=candidate_worker_result_identity(changed))
    changed_selected = _selected_candidate_result(
        changed,
        _summary(changed),
        selection_identities=selected.selection_identities,
    )

    assert selected.result.scores is None
    assert selected.result.physical.diagnostic == decoded.physical.diagnostic
    assert diagnostic.code == changed_diagnostic.code
    assert (
        decoded.physical.physical_candidate_identity != changed_physical.physical_candidate_identity
    )
    assert decoded.identity != changed.identity
    assert selected.summary.identity != changed_selected.summary.identity
    assert selected.identity != changed_selected.identity
    assert encode_candidate_worker_result(decoded) == encode_candidate_worker_result(result)
    candidate = InterventionCandidate(
        "demand_lifetime_frontier",
        decoded.intervention.demand_lifetime_evaluation_projection.decision,
        decoded.intervention.circularity_evaluation_projection.decision,
    )
    evaluation = _candidate_evaluation(candidate, selected.summary)
    assert evaluation.diagnostic == diagnostic
    assert evaluation.state_metrics == ()


def test_real_material_failure_survives_worker_transport() -> None:
    """Transport the receiving calculation's diagnostic."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented")
    with pytest.raises(ScenarioFeasibilityError) as error:
        realize_materials(
            year=2037,
            region="EUR",
            materials=("PS",),
            composition=load_material_composition(),
            recipe=recipe,
            collected=np.array([1.0]),
            route_totals=(Fraction(1), Fraction(), Fraction(), Fraction()),
            processing_requirement=(Fraction(),),
            primary_imports=(Fraction(),),
            primary_exports=(Fraction(),),
        )
    diagnostic = error.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "optimization_material_recovery_capacity_infeasible"
    assert diagnostic.shortfall is not None and diagnostic.shortfall > 0
    assert dict(diagnostic.coordinates)["year"] == 2037
    result = _infeasible_result()
    physical = replace(result.physical, diagnostic=diagnostic)
    physical = replace(physical, physical_candidate_identity=physical_evaluation_identity(physical))
    result = replace(result, physical=physical)
    result = replace(result, identity=candidate_worker_result_identity(result))
    decoded = decode_candidate_worker_result(encode_candidate_worker_result(result))
    assert decoded.physical.diagnostic == diagnostic
    assert _summary(decoded).physical.diagnostic == diagnostic


def test_pa_result_identity_binds_full_method_arrays_and_stage_coordinates() -> None:
    """Bind active PA values and their attribution, including zero-stream identity."""
    empty = _result()
    pa_scores = _pa_scores(active=True)
    active = replace(empty, pa_scores=pa_scores)
    assert candidate_worker_result_identity(active) != empty.identity
    # Two valid whole-calculation records have scaled scores and absolute
    # totals, with the matching stage table, for testing quantity serialization.
    scaled = replace(
        pa_scores,
        annual_scores=pa_scores.annual_scores * 2,
        annual_contribution_norms=pa_scores.annual_contribution_norms * 2,
        stage_scores=pa_scores.stage_scores.assign(
            score=pa_scores.stage_scores.score * 2,
            contribution_norm=pa_scores.stage_scores.contribution_norm * 2,
        ),
    )
    assert candidate_worker_result_identity(
        replace(active, pa_scores=scaled)
    ) != candidate_worker_result_identity(active)
    # Changing only the allocation between two legitimate stage coordinates
    # preserves annual arrays but must change the saved scientific result.
    stages = pa_scores.stage_scores.copy()
    stages["life_cycle_stage"] = "mechanical_recycling_reject_incineration_treatment"
    stages["stage_group"] = "recycling_residue_treatment"
    stages["cutoff_cycle_role"] = "recycling_residue_treatment"
    stages["physical_event_basis"] = (
        "same year as the mechanical-recycling operation that generates the reject"
    )
    reattributed = replace(pa_scores, stage_scores=stages)
    assert candidate_worker_result_identity(
        replace(active, pa_scores=reattributed)
    ) != candidate_worker_result_identity(active)
