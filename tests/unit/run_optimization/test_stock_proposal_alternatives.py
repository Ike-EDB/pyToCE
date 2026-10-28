"""Preserve failed reference paths when stock proposals cannot restore feasibility."""

from pathlib import Path

import pytest

from pytoce.run_optimization.calculation.candidate_search import (
    InterventionCandidate,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestBridge,
    initialize_candidate_worker,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    CandidateWorkerEvaluator,
    build_candidate_worker_metric_scopes,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerTask,
    decode_candidate_worker_result,
    encode_candidate_worker_task,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateResultStore
from pytoce.run_optimization.runtime.scheduler import CandidateProcessPool
from tests.integration.candidate_mfa_fixtures import _prepare_worker_workspace


@pytest.mark.scientific_integration
def test_failed_zero_demand_reference_and_absent_stock_proposal_retain_original_candidates(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Admit one parent, observe two actual stock failures, then preserve both references."""
    workspace = tmp_path / "workspace"
    foreground = "stock_floor_parent"
    initializer, context = _prepare_worker_workspace(
        workspace,
        remind_mfa_source_root,
        foreground_scenario=foreground,
        stock_floor=True,
        include_public_baseline_results=False,
    )
    references = tuple(
        InterventionCandidate(
            "demand_lifetime_frontier",
            DemandLifetimeDecision(demand, 100),
            CircularityDecision.parent(),
        )
        for demand in (0, 25)
    )
    cache = workspace / "results" / foreground / "optimization_results/stock_floor_fixture"
    with CandidateResultStore.open(
        cache,
        candidate_input_identity="request::stock-floor-proposal-fixture",
        worker_authority_identity=context.planner.worker_authority_identity,
    ) as result_store:
        bridge = CandidateWorkerRequestBridge(
            context.planner,
            build_candidate_worker_metric_scopes(("demand_lifetime_frontier",)),
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
            failures = evaluator(references)
            assert len(failures) == 2
            assert len({item.physical_candidate_identity for item in failures}) == 2
            for reference, failure in zip(references, failures, strict=True):
                diagnostic = failure.diagnostic
                assert diagnostic is not None
                assert diagnostic.code == "optimization_stock_below_surviving_cohort"
                assert dict(diagnostic.coordinates) == {
                    "year": 2020,
                    "region": "EUR",
                    "good": "Buildings & Construction",
                }
                assert diagnostic.observed is not None and diagnostic.bound is not None
                assert diagnostic.observed < diagnostic.bound
                assert diagnostic.tolerance > 0.0
                assert diagnostic.bound - diagnostic.observed > diagnostic.tolerance
                assert failure.state_metrics == ()
                intervention = context.planner.plan_intervention(
                    demand_lifetime=reference.demand_lifetime,
                    circularity=reference.circularity,
                )
                saved = bridge.load_candidate(intervention.identity)
                assert saved.physical.diagnostic == diagnostic
                assert saved.scores is None
                assert saved.feedstock_requirements is None
            records_before_proposals = evaluator.candidate_records
            assert len(records_before_proposals) == 2

            assert evaluator(references) == failures
            assert evaluator.candidate_records == records_before_proposals
            observed = [
                bridge.load_candidate(
                    context.planner.plan_intervention(
                        demand_lifetime=reference.demand_lifetime, circularity=reference.circularity
                    ).identity
                ).repair
                for reference in references
            ]
            assert [item.status for item in observed] == [
                "unique_fixed_path_infeasible",
                "stock_infeasible_fixed_controls",
            ]
            assert [item.stock_lp_count for item in observed] == [0, 1]
            assert [item.mfa_evaluation_count for item in observed] == [1, 1]

    # A warm worker reuses the fixed-lifetime solve for identical endpoints.
    # Recorded work counts include only performed calculations. A stock rejection
    # requires no material or LCA evaluation.
    worker = initialize_candidate_worker(initializer)
    intervention = context.planner.plan_intervention(
        demand_lifetime=references[1].demand_lifetime, circularity=references[1].circularity
    )
    task = CandidateWorkerTask(
        intervention, build_candidate_worker_metric_scopes(("demand_lifetime_frontier",))
    )
    payload = encode_candidate_worker_task(task)
    cold = decode_candidate_worker_result(worker(payload))
    warm = decode_candidate_worker_result(worker(payload))
    assert cold.identity == warm.identity
    assert cold.repair.stock_lp_count == 1 and warm.repair.stock_lp_count == 0
    assert cold.repair.mfa_evaluation_count == warm.repair.mfa_evaluation_count == 1
    assert cold.repair.environmental_evaluation_elapsed_seconds == 0.0
    assert warm.repair.environmental_evaluation_elapsed_seconds == 0.0
    assert cold.repair.physical_repair_elapsed_seconds >= 0.0
    assert warm.repair.physical_repair_elapsed_seconds >= 0.0
