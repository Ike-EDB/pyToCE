"""Selected timing composition from real candidate calculations and state selection."""

from pathlib import Path

import numpy as np
import pytest

from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.calculation.candidate_search import (
    InterventionCandidate,
    select_candidate_states,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestBridge,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    CandidateWorkerEvaluator,
    _selected_timing_candidates,
    build_candidate_worker_metric_scopes,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateResultStore
from pytoce.run_optimization.runtime.scheduler import CandidateProcessPool
from tests.integration.candidate_mfa_fixtures import (
    _BACKGROUND_SCENARIOS,
    _WORKER_FOREGROUND,
    _prepare_worker_workspace,
)


@pytest.mark.scientific_integration
def test_real_nonparent_selection_keeps_endpoints_and_state_roles_in_delayed_timings(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Retain a real selected treatment transfer and calculate each delayed path once."""
    workspace = tmp_path / "workspace"
    initializer, context = _prepare_worker_workspace(
        workspace,
        remind_mfa_source_root,
        include_generous_case=True,
        landfill_responses_only=True,
        include_public_baseline_results=False,
    )
    parent = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0),
        CircularityDecision.parent(),
    )
    # The parent MFA result has 30% incineration and 40% landfill. This transfer
    # leaves collection, recycling, feedstocks, demand and lifetime unchanged.
    changed = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0),
        CircularityDecision(65, 20, 10, 40, 30, 95, 5, 0),
    )
    candidates = (parent, changed)
    cache = workspace / "results" / _WORKER_FOREGROUND / "optimization_results/timing_fixture"
    with CandidateResultStore.open(
        cache,
        candidate_input_identity="request::selected-timing-fixture",
        worker_authority_identity=context.planner.worker_authority_identity,
    ) as result_store:
        bridge = CandidateWorkerRequestBridge(
            context.planner,
            build_candidate_worker_metric_scopes(("circularity_feedstock_anchor",)),
            result_store=result_store,
            allocated_cc_context=context.allocated_cc_context,
            search_scoring=context.search_scoring,
            worker_scientific_authority_identity=context.worker_scientific_authority_identity,
        )
        with CandidateProcessPool(worker_initializer_payload=initializer) as scheduler:
            evaluator = CandidateWorkerEvaluator(
                planner=context.planner,
                bridge=bridge,
                scheduler=scheduler,
                result_store=result_store,
            )
            evaluations = evaluator(candidates)
            assert all(item.diagnostic is None for item in evaluations)
            for evaluation in evaluations:
                assert {
                    (metric.background_scenario, metric.allocated_cc_case)
                    for metric in evaluation.state_metrics
                } == {
                    (background, case)
                    for background in _BACKGROUND_SCENARIOS
                    for case in ("direct_case", "generous_case")
                }
                assert all(metric.region_count == 12 for metric in evaluation.state_metrics)

            selection = select_candidate_states(evaluations)
            state_coordinates = {
                metric.state_identity: (metric.background_scenario, metric.allocated_cc_case)
                for metric in evaluations[0].state_metrics
            }
            selection_roles: dict[str, list[str]] = {}
            for state in selection.states:
                _background, case = state_coordinates[state.state_identity]
                if case == "direct_case":
                    assert state.candidate_identity == changed.identity
                    assert state.status == "selected_noncompliant"
                else:
                    assert state.candidate_identity == parent.identity
                    assert state.status == "baseline_compliant"
                selection_roles.setdefault(state.candidate_identity, []).append(
                    state.state_identity
                )
            assert len(selection.states) == 8
            assert set(selection.selected_candidate_identities) == {
                parent.identity,
                changed.identity,
            }

            selected_candidates = tuple(
                item for item in candidates if item.identity in selection_roles
            )
            assert len(evaluator.result_summary_identities) == 2
            timing_candidates, timing_roles = _selected_timing_candidates(
                selected_candidates,
                selection_identities=selection_roles,
            )
            evaluator(timing_candidates)
            scheduler.close()
            selected_2050 = evaluator.selected(
                selected_candidates,
                selection_identities=selection_roles,
            )
            selected_timing = evaluator.selected(
                timing_candidates,
                selection_identities=timing_roles,
            )
            assert len(evaluator.result_summary_identities) == 3
            assert len(evaluator.candidate_records) == 3
            assert len(tuple(result_store.directory.glob("[0-9]*"))) == 3
            assert select_candidate_states(evaluations) == selection

    selected_by_candidate = {item.candidate.identity: item for item in selected_2050}
    parent_scores = selected_by_candidate[parent.identity].selected_result.result.scores
    changed_scores = selected_by_candidate[changed.identity].selected_result.result.scores
    assert parent_scores is not None and changed_scores is not None
    assert np.all(changed_scores.annual_scores <= parent_scores.annual_scores)
    assert np.any(changed_scores.annual_scores < parent_scores.annual_scores)
    assert {item.candidate.target_year for item in selected_timing} == {2100}
    assert len(selected_timing) == 1
    expected_roles = tuple(sorted(selection_roles[changed.identity]))
    physical_identities = {
        item.selected_result.result.physical.physical_candidate_identity for item in selected_2050
    }
    for timing in selected_timing:
        assert timing.selection_identities == expected_roles
        assert timing.selected_result.selection_identities == expected_roles
        assert timing.candidate.optimization_block == changed.optimization_block
        assert timing.candidate.circularity.percentages() == changed.circularity.percentages()
        assert timing.candidate.demand_lifetime.uses_exact_parent
        assert (
            timing.candidate.demand_lifetime.demand_fraction
            == changed.demand_lifetime.demand_fraction
        )
        assert (
            timing.candidate.demand_lifetime.lifetime_fraction
            == changed.demand_lifetime.lifetime_fraction
        )
        result = timing.selected_result.result
        assert result.physical.diagnostic is None
        assert result.physical.status == "physical_feasible"
        assert not result.intervention.uses_exact_parent
        assert (
            result.intervention.circularity_evaluation_projection.decision
            == timing.candidate.circularity
        )
        delta = result.physical.activation_delta
        assert delta is not None and not delta.empty
        assert delta.loc[delta.year.eq(2019), "delta_amount_kg"].eq(0).all()
        assert delta.loc[delta.year.gt(2019), "delta_amount_kg"].ne(0).any()
        scores = result.scores
        assert scores is not None
        assert set(scores.backgrounds) == set(_BACKGROUND_SCENARIOS)
        assert set(scores.regions) == set(H12_REGIONS)
        assert scores.years == tuple(range(2019, 2101))
        assert len(scores.annual_categories) == 16
        assert scores.annual_scores.shape == (4, 12, 82, 16)
        np.testing.assert_array_equal(
            scores.annual_scores[:, :, 0, :], parent_scores.annual_scores[:, :, 0, :]
        )
        physical_identities.add(result.physical.physical_candidate_identity)
    assert len(physical_identities) == 3
